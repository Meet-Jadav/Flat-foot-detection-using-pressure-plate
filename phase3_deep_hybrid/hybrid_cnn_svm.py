#!/usr/bin/env python3
"""Hybrid CNN feature extraction + SVM classifier."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2_ml_models.common import split_dataset
from phase2_ml_models.evaluate import evaluate_predictions
from transfer_learning import TransferLearningModel, PressureDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "saved"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_cnn_features(model: TransferLearningModel, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors from CNN."""

    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            features = model.get_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_features), np.concatenate(all_labels)


def train_hybrid_cnn_svm(folder_path: str | Path, cnn_model_path: str | Path | None = None, batch_size: int = 16):
    """Use CNN features with SVM."""

    dataset = PressureDataset(folder_path, augment=False)
    if len(dataset) < 20:
        print("not enough samples for hybrid model")
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = TransferLearningModel().to(DEVICE)

    if cnn_model_path is None:
        cnn_model_path = MODEL_DIR / "transfer_learning_model.pt"
    if Path(cnn_model_path).exists():
        model.load_state_dict(torch.load(cnn_model_path, map_location=DEVICE))
        print("loaded pretrained CNN from", cnn_model_path)

    X_features, y = extract_cnn_features(model, loader)
    print(f"extracted features shape: {X_features.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    svm_classifier = SVC(kernel="rbf", probability=True, class_weight="balanced")
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    y_prob = svm_classifier.predict_proba(X_test)[:, 1]
    report = evaluate_predictions(y_test, y_pred, y_prob, model_name="hybrid_cnn_svm", output_dir=FIGURE_DIR)

    bundle = {
        "model_name": "hybrid_cnn_svm",
        "cnn_model": model,
        "svm_classifier": svm_classifier,
        "scaler": scaler,
        "report": report,
    }
    model_path = MODEL_DIR / "hybrid_cnn_svm.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    print("saved hybrid model to", model_path)

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--cnn-model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_hybrid_cnn_svm(args.folder, cnn_model_path=args.cnn_model_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
