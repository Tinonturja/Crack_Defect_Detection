"""Training and evaluation loops for the binary crack classifier."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def accuracy_fn(logits: torch.Tensor, y_true: torch.Tensor) -> int:
    """Number of correct predictions in a batch (threshold at 0.5 on sigmoid)."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    return int((preds == y_true).sum().item())


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.to(device)
    model.train()

    total_loss, total_correct, num_samples = 0.0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device).float()

        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += accuracy_fn(logits, y)
        num_samples += X.shape[0]

    return total_loss / len(dataloader), total_correct / num_samples


def evaluate_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.to(device)
    model.eval()

    total_loss, total_correct, num_samples = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).float()
            logits = model(X)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            total_correct += accuracy_fn(logits, y)
            num_samples += X.shape[0]

    return total_loss / len(dataloader), total_correct / num_samples


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate_step(model, val_dataloader, loss_fn, device)

        if verbose:
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}"
            )

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_acc"].append(train_acc)
        results["val_acc"].append(val_acc)

    return results


def evaluate_with_report(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
):
    """Full classification report + confusion matrix, not just accuracy."""
    from sklearn.metrics import classification_report, confusion_matrix

    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm
