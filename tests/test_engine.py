import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from crack_detector.engine import accuracy_fn, evaluate_step, evaluate_with_report, train_step


def _toy_dataloader(n=16):
    X = torch.randn(n, 3, 16, 16)
    y = torch.randint(0, 2, (n,)).float()
    return DataLoader(TensorDataset(X, y), batch_size=4)


def _toy_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 1), nn.Flatten(start_dim=0))


def test_accuracy_fn_counts_correct_predictions():
    logits = torch.tensor([5.0, -5.0, 5.0])  # sigmoid -> ~1, ~0, ~1
    labels = torch.tensor([1.0, 0.0, 0.0])  # last one is wrong
    assert accuracy_fn(logits, labels) == 2


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    model = _toy_model()
    dataloader = _toy_dataloader()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss_before, _ = evaluate_step(model, dataloader, loss_fn, torch.device("cpu"))
    for _ in range(5):
        train_step(model, dataloader, loss_fn, optimizer, torch.device("cpu"))
    loss_after, _ = evaluate_step(model, dataloader, loss_fn, torch.device("cpu"))

    assert loss_after < loss_before


def test_evaluate_with_report_shapes():
    model = _toy_model()
    dataloader = _toy_dataloader(n=12)
    class_names = ["Negative", "Positive"]

    report, cm = evaluate_with_report(model, dataloader, class_names, torch.device("cpu"))

    assert cm.shape == (2, 2)
    for cls in class_names:
        assert cls in report
