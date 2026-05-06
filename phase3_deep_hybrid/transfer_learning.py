#!/usr/bin/env python3
"""Transfer learning with MobileNetV2 for pressure plate images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from cnn_model import PressureDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransferLearningModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone = mobilenet.features
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x


def train_transfer_learning(folder_path: str | Path, epochs: int = 30, batch_size: int = 16, unfreeze_after: int = 10):
    """Train a transfer learning model with MobileNetV2."""

    dataset = PressureDataset(folder_path, augment=True)
    if len(dataset) < 10:
        print("not enough samples for transfer learning")
        return None

    normal_count = sum(1 for _, label in dataset.samples if label == 0)
    flatfoot_count = sum(1 for _, label in dataset.samples if label == 1)
    print(f"transfer learning dataset: {normal_count} normal, {flatfoot_count} flatfoot")

    model = TransferLearningModel(freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        if epoch == unfreeze_after:
            print("unfreezing backbone layers")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{epochs} loss={total_loss / max(1, len(loader)):.4f}")

    model_path = MODEL_DIR / "transfer_learning_model.pt"
    torch.save(model.state_dict(), model_path)
    print("saved transfer learning model to", model_path)

    return {"model": model, "dataset_size": len(dataset)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_transfer_learning(args.folder, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
