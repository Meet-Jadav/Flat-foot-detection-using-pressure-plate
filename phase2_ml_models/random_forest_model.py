#!/usr/bin/env python3
"""Random forest model for flatfoot detection."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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


def train_random_forest(folder: str | Path | None = None, dataset_csv: str | Path | None = None, threshold: float = 5.0, min_size: int = 20, random_state: int = 42):
    """Train and evaluate a 100-tree random forest."""

    df = _load_input(folder=folder, dataset_csv=dataset_csv, threshold=threshold, min_size=min_size)
    if df.empty:
        print("no samples to train random forest")
        return None

    X_train, X_test, y_train, y_test, feature_columns, groups_train = split_dataset(df, random_state=random_state)
    if X_train is None:
        print("could not prepare training data")
        return None

    # Harden RF to reduce overfitting: limit depth, require min samples per leaf,
    # restrict max features and enable OOB score for a sanity check.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=random_state,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
    )
    cv, cv_groups = get_cv_strategy(y_train, groups_train, n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, groups=cv_groups, scoring="accuracy")
    print("5-fold cv accuracy:", float(np.mean(cv_scores)))

    model.fit(X_train, y_train)
    # Print OOB score if available for sanity checking against test accuracy
    try:
        print("oob_score:", float(getattr(model, "oob_score_", np.nan)))
    except Exception:
        pass
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = evaluate_predictions(y_test, y_pred, y_prob, model_name="random_forest", output_dir=FIGURE_DIR)

    importances = getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
    ranking = sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)
    top_features = ranking[:15]

    importance_path = FIGURE_DIR / "random_forest_feature_importance.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [name for name, _ in top_features][::-1]
    values = [value for _, value in top_features][::-1]
    ax.barh(names, values, color="#3c6e71")
    ax.set_title("Random forest feature importance")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(importance_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    bundle = {
        "model_name": "random_forest",
        "model": model,
        "feature_columns": feature_columns,
        "report": report,
        "feature_importance": top_features,
    }
    model_path = MODEL_DIR / "random_forest.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    print("saved random forest model to", model_path)
    print("saved feature importance plot to", importance_path)

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--min-size", type=int, default=20)
    args = parser.parse_args()

    train_random_forest(folder=args.folder, dataset_csv=args.dataset_csv, threshold=args.threshold, min_size=args.min_size)


if __name__ == "__main__":
    main()
