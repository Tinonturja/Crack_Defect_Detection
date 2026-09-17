"""Dataset and dataloader construction for the crack-detection task.

Data is expected in the standard ``torchvision.datasets.ImageFolder``
layout with two classes::

    resources/data/
        Negative/*.jpg
        Positive/*.jpg

This is the layout of the public "Concrete Crack Images for Classification"
dataset. The original notebook wrote three separate, half-finished custom
``Dataset`` classes to reinvent this; ``torchvision.datasets.ImageFolder``
already does it correctly, so this module uses that directly and adds a
proper stratified train/val split instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T


@dataclass
class DataBundle:
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    class_names: List[str]
    class_to_idx: dict


def build_transforms(image_size: int = 64, augment: bool = False) -> T.Compose:
    """Build the image transform pipeline.

    Args:
        image_size: images are resized to (image_size, image_size).
        augment: if True, adds light random flips/rotation for the
            training set (helps generalization; not used for eval).
    """
    ops = []
    if augment:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10)]
    ops += [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return T.Compose(ops)


def create_dataloaders(
    data_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 64,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    subset_size: int | None = None,
) -> DataBundle:
    """Create a stratified train/val split from an ImageFolder dataset.

    Args:
        data_dir: directory containing one subfolder per class.
        image_size: images are resized to (image_size, image_size).
        batch_size: dataloader batch size.
        val_fraction: fraction of data held out for validation.
        num_workers: dataloader worker processes.
        seed: RNG seed for the split (reproducibility).
        subset_size: if set, use only this many images total (stratified
            across classes), useful for fast iteration/CI. ``None`` uses
            the full dataset.

    Raises:
        FileNotFoundError: if ``data_dir`` does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_transform = build_transforms(image_size, augment=True)
    eval_transform = build_transforms(image_size, augment=False)

    # Two separate ImageFolder instances (same files, different transform)
    # so training gets augmentation and validation doesn't.
    base_dataset = torchvision.datasets.ImageFolder(data_dir)
    train_view = torchvision.datasets.ImageFolder(data_dir, transform=train_transform)
    val_view = torchvision.datasets.ImageFolder(data_dir, transform=eval_transform)

    targets = torch.tensor(base_dataset.targets)
    generator = torch.Generator().manual_seed(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_idx in range(len(base_dataset.classes)):
        class_indices = (targets == class_idx).nonzero(as_tuple=True)[0]
        class_indices = class_indices[torch.randperm(len(class_indices), generator=generator)]

        if subset_size is not None:
            per_class = subset_size // len(base_dataset.classes)
            class_indices = class_indices[:per_class]

        n_val = int(len(class_indices) * val_fraction)
        val_indices.extend(class_indices[:n_val].tolist())
        train_indices.extend(class_indices[n_val:].tolist())

    train_dataset = Subset(train_view, train_indices)
    val_dataset = Subset(val_view, val_indices)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return DataBundle(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        class_names=base_dataset.classes,
        class_to_idx=base_dataset.class_to_idx,
    )
