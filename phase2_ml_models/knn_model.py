#!/usr/bin/env python3
"""KNN model for flatfoot detection."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import build_dataset_from_folder, get_cv_strategy, split_dataset
from evaluate import evaluate_predictions

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "saved"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _load_input(folder: str | Path | None = None, dataset_csv: str | Path | None = None, threshold: float = 5.0, min_size: int = 20) -> pd.DataFrame:
    if dataset_csv is not None:
        return pd.read_csv(dataset_csv)
    if folder is None:
        return pd.DataFrame()
    return build_dataset_from_folder(folder, threshold=threshold, min_size=min_size)


def train_knn(folder: str | Path | None = None, dataset_csv: str | Path | None = None, threshold: float = 5.0, min_size: int = 20, random_state: int = 42):
    """Train KNN, try a few k values, and save the elbow plot."""

    df = _load_input(folder=folder, dataset_csv=dataset_csv, threshold=threshold, min_size=min_size)
    if df.empty:
        print("no samples to train knn")
        return None

    X_train, X_test, y_train, y_test, feature_columns, groups_train = split_dataset(df, random_state=random_state)
    if X_train is None:
        print("could not prepare training data")
        return None

    candidate_ks = list(range(3, 12))
    elbow_scores = []
    cv, cv_groups = get_cv_strategy(y_train, groups_train, n_splits=5)

    for k_value in candidate_ks:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k_value)),
            ]
        )
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, groups=cv_groups, scoring="accuracy")
        elbow_scores.append(float(np.mean(cv_scores)))
        print(f"k={k_value} cv accuracy={elbow_scores[-1]:.4f}")

    best_index = int(np.argmax(elbow_scores))
    best_k = candidate_ks[best_index]
    print("best k from sweep:", best_k)

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=best_k)),
        ]
    )
    final_pipeline.fit(X_train, y_train)
    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    report = evaluate_predictions(y_test, y_pred, y_prob, model_name="knn", output_dir=FIGURE_DIR)

    elbow_path = FIGURE_DIR / "knn_k_sweep.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(candidate_ks, elbow_scores, marker="o", color="#8d99ae")
    ax.set_title("KNN accuracy vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("cross-val accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(elbow_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    bundle = {
        "model_name": "knn",
        "model": final_pipeline,
        "best_k": best_k,
        "feature_columns": feature_columns,
        "report": report,
        "k_sweep": dict(zip(candidate_ks, elbow_scores)),
    }
    model_path = MODEL_DIR / "knn.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    print("saved knn model to", model_path)
    print("saved knn elbow plot to", elbow_path)

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--min-size", type=int, default=20)
    args = parser.parse_args()

    train_knn(folder=args.folder, dataset_csv=args.dataset_csv, threshold=args.threshold, min_size=args.min_size)


if __name__ == "__main__":
    main()
