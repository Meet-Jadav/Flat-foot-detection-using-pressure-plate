#!/usr/bin/env python3
"""SVM model for flatfoot detection."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def train_svm(folder: str | Path | None = None, dataset_csv: str | Path | None = None, threshold: float = 5.0, min_size: int = 20, random_state: int = 42):
    """Train and evaluate an RBF SVM."""

    df = _load_input(folder=folder, dataset_csv=dataset_csv, threshold=threshold, min_size=min_size)
    if df.empty:
        print("no samples to train svm")
        return None

    X_train, X_test, y_train, y_test, feature_columns, groups_train = split_dataset(df, random_state=random_state)
    if X_train is None:
        print("could not prepare training data")
        return None

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
        ]
    )

    cv, cv_groups = get_cv_strategy(y_train, groups_train, n_splits=5)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, groups=cv_groups, scoring="accuracy")
    print("5-fold cv accuracy:", float(np.mean(cv_scores)))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    report = evaluate_predictions(y_test, y_pred, y_prob, model_name="svm_rbf", output_dir=FIGURE_DIR)

    importance = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=random_state, scoring="accuracy")
    ranking = sorted(zip(feature_columns, importance.importances_mean), key=lambda item: item[1], reverse=True)
    top_features = ranking[:10]
    print("top svm features:")
    for name, value in top_features:
        print(f"  {name}: {value:.4f}")

    importance_path = FIGURE_DIR / "svm_permutation_importance.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [name for name, _ in top_features][::-1]
    values = [value for _, value in top_features][::-1]
    ax.barh(names, values, color="#355c7d")
    ax.set_title("SVM permutation importance")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(importance_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    bundle = {
        "model_name": "svm_rbf",
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "report": report,
        "permutation_importance": top_features,
    }
    model_path = MODEL_DIR / "svm_rbf.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    print("saved svm model to", model_path)
    print("saved svm importance plot to", importance_path)

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--min-size", type=int, default=20)
    args = parser.parse_args()

    train_svm(folder=args.folder, dataset_csv=args.dataset_csv, threshold=args.threshold, min_size=args.min_size)


if __name__ == "__main__":
    main()
