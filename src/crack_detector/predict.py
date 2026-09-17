"""Run inference with a trained model on a single image."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import nn


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    class_names: list[str],
    transform,
    device: torch.device,
) -> Tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        logit = model(input_tensor)
        prob_positive = torch.sigmoid(logit).item()

    pred_idx = int(prob_positive > 0.5)
    confidence = prob_positive if pred_idx == 1 else 1 - prob_positive
    return class_names[pred_idx], confidence


if __name__ == "__main__":
    import argparse

    from crack_detector.data import build_transforms
    from crack_detector.model import build_model
    from crack_detector.utils import get_device, load_checkpoint

    parser = argparse.ArgumentParser(description="Predict crack / no-crack for an image.")
    parser.add_argument("image", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--class-names", type=str, nargs="+", default=["Negative", "Positive"])
    args = parser.parse_args()

    device = get_device()
    model = build_model()
    model = load_checkpoint(model, args.checkpoint, map_location=str(device))
    transform = build_transforms(image_size=args.image_size, augment=False)

    pred_class, confidence = predict_image(model, args.image, args.class_names, transform, device)
    print(f"Prediction: {pred_class} (confidence: {confidence:.3f})")
