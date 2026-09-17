"""Generate a grid of real model predictions on sample images for the README.

Run after training (needs results/model.pth):
    python scripts/make_prediction_demo.py
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from crack_detector.data import build_transforms
from crack_detector.model import build_model
from crack_detector.utils import get_device, load_checkpoint

DATA_DIR = Path("resources/data")
CHECKPOINT = Path("results/model.pth")
OUTPUT = Path("assets/prediction_examples.png")
N_PER_CLASS = 3
SEED = 7


def main() -> None:
    random.seed(SEED)
    device = get_device()

    model = build_model()
    model = load_checkpoint(model, CHECKPOINT, map_location=str(device))
    model.to(device)
    model.eval()

    transform = build_transforms(image_size=64, augment=False)

    samples = []
    for cls in ["Negative", "Positive"]:
        files = sorted((DATA_DIR / cls).glob("*.jpg"))
        chosen = random.sample(files, N_PER_CLASS)
        samples.extend((f, cls) for f in chosen)
    random.shuffle(samples)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (path, true_label) in zip(axes.flat, samples):
        image = Image.open(path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logit = model(input_tensor)
            prob_positive = torch.sigmoid(logit).item()
        pred_label = "Positive" if prob_positive > 0.5 else "Negative"
        confidence = prob_positive if pred_label == "Positive" else 1 - prob_positive
        correct = pred_label == true_label

        ax.imshow(image)
        ax.axis("off")
        color = "green" if correct else "red"
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})",
            color=color,
            fontsize=11,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)

    fig.suptitle(
        "Real predictions from results/model.pth on sample images\n"
        "(randomly sampled from the full dataset, not restricted to the validation split)",
        fontsize=11,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=100, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
