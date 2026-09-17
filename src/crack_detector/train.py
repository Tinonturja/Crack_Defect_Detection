"""CLI entry point that wires data, model, and engine together.

Example:
    python -m crack_detector.train --data-dir resources/data --epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from crack_detector.data import create_dataloaders
from crack_detector.engine import evaluate_with_report, train
from crack_detector.model import build_model
from crack_detector.utils import (
    get_device,
    plot_confusion_matrix,
    plot_curves,
    save_checkpoint,
    save_history,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the crack-detection CNN.")
    parser.add_argument("--data-dir", type=str, default="resources/data")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use only this many images total (stratified). Default: full dataset.",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    data = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        subset_size=args.subset_size,
    )
    print(f"Classes: {data.class_names}")
    print(
        f"Train samples: {len(data.train_dataloader.dataset)} | "
        f"Val samples: {len(data.val_dataloader.dataset)}"
    )

    model = build_model()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = train(
        model=model,
        train_dataloader=data.train_dataloader,
        val_dataloader=data.val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
    )

    output_dir = Path(args.output_dir)
    save_history(history, output_dir / "history.json")
    plot_curves(history, save_path=output_dir / "training_curves.png")
    save_checkpoint(model, output_dir / "model.pth", extra={"class_names": data.class_names})

    report, cm = evaluate_with_report(model, data.val_dataloader, data.class_names, device)
    plot_confusion_matrix(cm, data.class_names, save_path=output_dir / "confusion_matrix.png")

    print("Per-class results:")
    for cls in data.class_names:
        r = report[cls]
        print(
            f"  {cls:10s} precision={r['precision']:.3f} "
            f"recall={r['recall']:.3f} f1={r['f1-score']:.3f}"
        )
    print(f"Overall accuracy: {report['accuracy']:.3f}")

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
