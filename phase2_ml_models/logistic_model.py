#!/usr/bin/env python3
"""Logistic regression model for flatfoot detection."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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


def train_logistic(folder: str | Path | None = None, dataset_csv: str | Path | None = None, threshold: float = 5.0, min_size: int = 20, random_state: int = 42):
    """Train a logistic regression model and print the coefficients."""

    df = _load_input(folder=folder, dataset_csv=dataset_csv, threshold=threshold, min_size=min_size)
    if df.empty:
        print("no samples to train logistic regression")
        return None

    X_train, X_test, y_train, y_test, feature_columns, groups_train = split_dataset(df, random_state=random_state)
    if X_train is None:
        print("could not prepare training data")
        return None

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=random_state)),
        ]
    )

    cv, cv_groups = get_cv_strategy(y_train, groups_train, n_splits=5)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, groups=cv_groups, scoring="accuracy")
    print("5-fold cv accuracy:", float(cv_scores.mean()))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    report = evaluate_predictions(y_test, y_pred, y_prob, model_name="logistic_regression", output_dir=FIGURE_DIR)

    logistic_step = pipeline.named_steps["logreg"]
    coefficients = logistic_step.coef_.ravel()
    ranking = sorted(zip(feature_columns, coefficients), key=lambda item: abs(item[1]), reverse=True)
    print("logistic regression coefficients:")
    for name, coef in ranking[:15]:
        direction = "toward flatfoot" if coef > 0 else "toward normal"
        print(f"  {name}: {coef:.4f} ({direction})")

    coef_path = FIGURE_DIR / "logistic_coefficients.txt"
    coef_lines = [f"{name}: {coef:.6f}" for name, coef in ranking]
    coef_path.write_text("\n".join(coef_lines), encoding="utf-8")

    bundle = {
        "model_name": "logistic_regression",
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "report": report,
        "coefficients": ranking,
    }
    model_path = MODEL_DIR / "logistic_regression.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    print("saved logistic model to", model_path)
    print("saved logistic coefficients to", coef_path)

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--min-size", type=int, default=20)
    args = parser.parse_args()

    train_logistic(folder=args.folder, dataset_csv=args.dataset_csv, threshold=args.threshold, min_size=args.min_size)


if __name__ == "__main__":
    main()
