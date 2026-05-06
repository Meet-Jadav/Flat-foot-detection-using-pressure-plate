#!/usr/bin/env python3
"""Shared utilities for the Phase-2 classical ML scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, GroupKFold, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phase1_rule_based.features import extract_features_from_csv  # noqa: E402


NON_FEATURE_COLUMNS = {
    "file_name",
    "csv_path",
    "subject_id",
    "condition",
    "trial_id",
    "foot_id",
    "foot_side",
    "label",
}

# Exclude clinical indices that may leak labels (Arch Index, CSI)
NON_FEATURE_COLUMNS.update({"arch_index", "csi", "paired_ai_gap"})


def build_dataset_from_folder(folder_path: str | Path, threshold: float = 5.0, min_size: int = 20) -> pd.DataFrame:
    """Scan a folder and build one row per detected foot."""

    folder_path = Path(folder_path)
    rows: list[dict] = []

    if not folder_path.exists():
        print("folder does not exist, check the path")
        return pd.DataFrame()

    for csv_path in sorted(folder_path.glob("*.csv")):
        try:
            rows.extend(extract_features_from_csv(csv_path, threshold=threshold, min_size=min_size))
        except Exception:
            print("skipping file because it could not be processed:", csv_path.name)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric columns used for ML."""

    feature_columns = []
    for column in df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)
    return feature_columns


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Subject-independent split when subject ids are available."""

    feature_columns = get_feature_columns(df)
    if not feature_columns:
        return None, None, None, None, feature_columns, None

    X = df[feature_columns].copy()
    y = df["label"].astype(int).copy()
    groups = df["subject_id"] if "subject_id" in df.columns else None

    if groups is not None and groups.notna().any() and groups.nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_index, test_index = next(splitter.split(X, y, groups=groups))
        return X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index], feature_columns, groups.iloc[train_index]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )
    return X_train, X_test, y_train, y_test, feature_columns, None


def get_cv_strategy(y: pd.Series, groups: pd.Series | None = None, n_splits: int = 5):
    """Pick a 5-fold CV strategy that fits the data."""

    if groups is not None and groups.nunique() >= 2:
        return GroupKFold(n_splits=min(n_splits, groups.nunique())), groups

    class_counts = y.value_counts()
    max_splits = int(class_counts.min()) if not class_counts.empty else 2
    return StratifiedKFold(n_splits=max(2, min(n_splits, max_splits)), shuffle=True, random_state=42), None


def feature_matrix_and_target(df: pd.DataFrame):
    feature_columns = get_feature_columns(df)
    X = df[feature_columns].copy()
    y = df["label"].astype(int).copy()
    return X, y, feature_columns
