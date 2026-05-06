#!/usr/bin/env python3
"""Shared evaluation utilities for the classical ML models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def _safe_auc(y_true, y_prob) -> float:
    if y_prob is None:
        return 0.0
    if len(set(y_true)) < 2:
        return 0.0
    try:
        return float(metrics.roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.0


def evaluate_predictions(y_true, y_pred, y_prob=None, model_name: str = "model", output_dir: str | Path | None = None):
    """Print metrics, save plots, and return a report dictionary."""

    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = None if y_prob is None else np.asarray(y_prob)

    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    accuracy = float(metrics.accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
    auc = _safe_auc(y_true, y_prob)

    matrix_path = output_dir / f"{model_name}_confusion_matrix.png"
    roc_path = output_dir / f"{model_name}_roc_curve.png"
    report_path = output_dir / f"{model_name}_evaluation.txt"

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{model_name} confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "flatfoot"])
    ax.set_yticklabels(["normal", "flatfoot"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    if y_prob is not None and len(set(y_true)) >= 2:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    else:
        ax.text(0.5, 0.5, "ROC unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{model_name} ROC curve")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        f"Model: {model_name}",
        f"Accuracy: {accuracy:.4f}",
        f"Sensitivity: {sensitivity:.4f}",
        f"Specificity: {specificity:.4f}",
        f"F1 score: {f1:.4f}",
        f"AUC: {auc:.4f}",
        f"Confusion matrix: {cm.tolist()}",
        f"Saved confusion matrix: {matrix_path}",
        f"Saved ROC curve: {roc_path}",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "matrix_path": str(matrix_path),
        "roc_path": str(roc_path),
        "report_path": str(report_path),
    }
