#!/usr/bin/env python3
"""Simple CNN from scratch for pressure plate images."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from phase1_rule_based.features import extract_features_from_csv
from phase1_rule_based.preprocess import load_and_split

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "saved"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PressureDataset(Dataset):
    def __init__(self, folder_path: str | Path, augment: bool = False):
        self.folder_path = Path(folder_path)
        self.augment = augment
        self.samples = []

        for csv_path in sorted(self.folder_path.glob("*.csv")):
            try:
                grid_data, _, feet = load_and_split(csv_path)
                for foot in feet:
                    grid = np.asarray(foot.get("grid", []), dtype=np.float32)
                    if grid.size > 0:
                        grid = np.minimum(grid / 100.0, 1.0)
                            # Allow resizing of smaller grids instead of skipping them
                            # (previously skipped grids smaller than 32x32)
                        features = extract_features_from_csv(csv_path)
                        if features:
                            label = int(features[0].get("label", 0))
                            self.samples.append((grid, label))
            except Exception:
                pass

    def _resize_pad(self, grid: np.ndarray, target_size: int = 64) -> np.ndarray:
        h, w = grid.shape
        if h == target_size and w == target_size:
            return grid

        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        from PIL import Image
        img = Image.fromarray((grid * 255).astype(np.uint8))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        grid_resized = np.asarray(img, dtype=np.float32) / 255.0

        padded = np.zeros((target_size, target_size), dtype=np.float32)
        offset_h = (target_size - new_h) // 2
        offset_w = (target_size - new_w) // 2
        padded[offset_h : offset_h + new_h, offset_w : offset_w + new_w] = grid_resized
        return padded

    def _augment(self, grid: np.ndarray) -> np.ndarray:
        if not self.augment:
            return grid

        if np.random.rand() < 0.3:
            angle = np.random.uniform(-5, 5)
            grid = np.rot90(grid, k=np.random.randint(0, 4))
        if np.random.rand() < 0.2:
            grid = np.fliplr(grid)
        if np.random.rand() < 0.1:
            noise = np.random.normal(0, 0.02, grid.shape)
            grid = np.clip(grid + noise, 0, 1)
        return grid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        grid, label = self.samples[idx]
        grid = self._resize_pad(grid, 64)
        grid = self._augment(grid)
        # ensure contiguous array and correct dtype to avoid PyTorch stride errors
        grid = np.ascontiguousarray(grid, dtype=np.float32)
        grid_tensor = torch.from_numpy(grid).unsqueeze(0)
        return grid_tensor, torch.tensor(label, dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_cnn(folder_path: str | Path, epochs: int = 50, batch_size: int = 16):
    """Train a simple CNN on the pressure plate images."""

    dataset = PressureDataset(folder_path, augment=True)
    if len(dataset) < 10:
        print("not enough samples to train CNN")
        return None

    normal_count = sum(1 for _, label in dataset.samples if label == 0)
    flatfoot_count = sum(1 for _, label in dataset.samples if label == 1)
    print(f"dataset: {normal_count} normal, {flatfoot_count} flatfoot")

    pos_weight = torch.tensor([normal_count / max(flatfoot_count, 1)], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0 / normal_count, 1.0 / flatfoot_count], device=DEVICE).to(DEVICE) if normal_count > 0 and flatfoot_count > 0 else None)
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
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

    model_path = MODEL_DIR / "cnn_model.pt"
    torch.save(model.state_dict(), model_path)
    print("saved cnn model to", model_path)

    return {"model": model, "dataset_size": len(dataset)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_cnn(args.folder, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
