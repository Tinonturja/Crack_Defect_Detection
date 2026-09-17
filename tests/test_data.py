import pytest
from PIL import Image

from crack_detector.data import build_transforms, create_dataloaders


def test_build_transforms_produces_correct_tensor_shape():
    transform = build_transforms(image_size=64, augment=False)
    dummy_image = Image.new("RGB", (227, 227))
    tensor = transform(dummy_image)
    assert tensor.shape == (3, 64, 64)


def test_create_dataloaders_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_dataloaders(data_dir=tmp_path / "does_not_exist")


def test_create_dataloaders_stratified_split(tmp_path):
    """Build a tiny synthetic ImageFolder dataset (10 per class) and check
    the split is stratified and the requested sizes come out right, without
    needing the real 40k-image dataset.
    """
    for cls, n in [("Negative", 10), ("Positive", 10)]:
        class_dir = tmp_path / cls
        class_dir.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (50, 50)).save(class_dir / f"{i}.jpg")

    data = create_dataloaders(data_dir=tmp_path, image_size=32, batch_size=4, val_fraction=0.2)

    assert sorted(data.class_names) == ["Negative", "Positive"]
    assert len(data.train_dataloader.dataset) == 16  # 8 per class
    assert len(data.val_dataloader.dataset) == 4  # 2 per class

    X, y = next(iter(data.train_dataloader))
    assert X.shape[1:] == (3, 32, 32)
    assert y.shape[0] == X.shape[0]


def test_create_dataloaders_respects_subset_size(tmp_path):
    for cls, n in [("Negative", 20), ("Positive", 20)]:
        class_dir = tmp_path / cls
        class_dir.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (50, 50)).save(class_dir / f"{i}.jpg")

    data = create_dataloaders(
        data_dir=tmp_path, image_size=32, batch_size=4, val_fraction=0.2, subset_size=10
    )
    total = len(data.train_dataloader.dataset) + len(data.val_dataloader.dataset)
    assert total == 10
