#!/usr/bin/env python3
"""A fuzzy rule-based classifier for the Arch Index output."""

from __future__ import annotations

from math import exp


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _severity_label(ai_value: float) -> str:
    if ai_value < 0.21:
        return "normal"
    if ai_value < 0.24:
        return "borderline"
    if ai_value < 0.30:
        return "mild flatfoot"
    if ai_value < 0.36:
        return "moderate flatfoot"
    return "severe flatfoot"


def classify_features(feature_row: dict) -> dict:
    """Convert the feature vector into a structured screening result."""

    ai_value = float(feature_row.get("arch_index", 0.0))
    midfoot_area = float(feature_row.get("zone2_contact_area", 0.0))
    total_area = float(feature_row.get("total_contact_area", 0.0))
    midfoot_ratio = midfoot_area / total_area if total_area > 0 else 0.0

    score = _clamp(1.0 / (1.0 + exp(-(ai_value - 0.26) * 18.0)))
    label = _severity_label(ai_value)

    if ai_value <= 0.26:
        confidence_note = (
            f"AI of {ai_value:.2f} stays below the flatfoot threshold of 0.26, "
            f"and the midfoot contact area is only {midfoot_area:.0f} cells ({midfoot_ratio:.0%} of total contact)."
        )
    else:
        confidence_note = (
            f"AI of {ai_value:.2f} exceeds the 0.26 threshold, midfoot contact area is {midfoot_area:.0f} cells "
            f"which is {midfoot_ratio:.0%} of total contact area."
        )

    return {
        "score": score,
        "label": label,
        "arch_index": ai_value,
        "key_features": {
            "arch_index": ai_value,
            "zone2_contact_area": midfoot_area,
            "total_contact_area": total_area,
            "midfoot_contact_ratio": midfoot_ratio,
            "csi": float(feature_row.get("csi", 0.0)),
            "cop_x": float(feature_row.get("cop_x", 0.0)),
            "cop_y": float(feature_row.get("cop_y", 0.0)),
        },
        "confidence_note": confidence_note,
        "threshold_reference": 0.26,
        "flatfoot_probability_like": score,
    }


# API Alias for frontend compatibility
classify_flatfoot = classify_features


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", type=float, required=True)
    args = parser.parse_args()

    result = classify_features({"arch_index": args.ai, "zone2_contact_area": 0, "total_contact_area": 1})
    print(result)


if __name__ == "__main__":
    main()