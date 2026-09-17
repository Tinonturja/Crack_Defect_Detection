"""A small convolutional network for binary crack classification.

The original notebook's model was ``Flatten -> Linear -> Linear`` — not a
CNN at all — and had a shape bug (``input_shape=3`` fed into a layer that
actually received 51,529 flattened features), so it never ran successfully.
This is a real, if deliberately small, convolutional architecture: three
conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool) followed by global
average pooling and a linear head with a single logit for binary
classification via ``BCEWithLogitsLoss``.
"""

from __future__ import annotations

from torch import nn


class SimpleCrackCNN(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # image_size / 2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # image_size / 4
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # image_size / 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, 1),  # single logit: BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


def build_model(base_channels: int = 16, dropout: float = 0.3) -> SimpleCrackCNN:
    return SimpleCrackCNN(base_channels=base_channels, dropout=dropout)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
