#!/usr/bin/env python3
"""Combine Phase 1 rule-based output with Phase 3 deep learning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1_rule_based.rule_classifier import classify_features
from phase1_rule_based.features import extract_features_from_csv


def hybrid_rule_dl_score(feature_row: dict, dl_probability: float, rule_weight: float = 0.4, dl_weight: float = 0.6) -> dict:
    """Combine rule-based AI score with deep learning probability."""

    rule_result = classify_features(feature_row)
    rule_score = float(rule_result["score"])

    combined_score = (rule_weight * rule_score) + (dl_weight * dl_probability)
    combined_score = float(np.clip(combined_score, 0.0, 1.0))

    if combined_score < 0.3:
        combined_label = "normal"
    elif combined_score < 0.4:
        combined_label = "borderline"
    elif combined_score < 0.5:
        combined_label = "mild flatfoot"
    elif combined_score < 0.7:
        combined_label = "moderate flatfoot"
    else:
        combined_label = "severe flatfoot"

    explanation = (
        f"Rule-based AI: {rule_score:.2f} ({rule_result['label']}), "
        f"Deep learning confidence: {dl_probability:.2f}, "
        f"Combined score: {combined_score:.2f} ({combined_label})"
    )

    return {
        "rule_score": rule_score,
        "rule_label": rule_result["label"],
        "dl_probability": dl_probability,
        "combined_score": combined_score,
        "combined_label": combined_label,
        "explanation": explanation,
        "arch_index": float(feature_row.get("arch_index", 0.0)),
        "key_features": rule_result.get("key_features", {}),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dl-prob", type=float, required=True)
    parser.add_argument("--rule-weight", type=float, default=0.4)
    parser.add_argument("--dl-weight", type=float, default=0.6)
    args = parser.parse_args()

    features = extract_features_from_csv(args.csv)
    if features:
        result = hybrid_rule_dl_score(features[0], args.dl_prob, rule_weight=args.rule_weight, dl_weight=args.dl_weight)
        print(result)


if __name__ == "__main__":
    main()
