#!/usr/bin/env python3
"""Feature extraction for one foot region.

This started as a messy note-to-self file and turned into the thing Phase 2
depends on, so I kept the structure plain and not too fancy.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .arch_index import compute_arch_metrics, orient_foot_region, split_into_three_zones
from .preprocess import load_and_split


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _peak_location(grid: np.ndarray) -> tuple[int, int]:
    if grid.size == 0 or np.max(grid) <= 0:
        return 0, 0
    flat_index = int(np.argmax(grid))
    row_index, col_index = np.unravel_index(flat_index, grid.shape)
    return int(row_index), int(col_index)


def parse_csv_metadata(csv_path: str | Path) -> dict:
    """Pull subject and condition info from the file name when possible."""

    csv_name = Path(csv_path).stem
    match = re.match(r"subject(?P<subject>\d+)_(?P<condition>[a-z-]+)_trial(?P<trial>\d+)_pressure", csv_name)
    if not match:
        return {
            "file_name": Path(csv_path).name,
            "subject_id": None,
            "condition": None,
            "trial_id": None,
        }

    return {
        "file_name": Path(csv_path).name,
        "subject_id": f"subject{match.group('subject')}",
        "condition": match.group("condition"),
        "trial_id": int(match.group("trial")),
    }


def extract_features_from_region(foot_region: dict, paired_ai: float | None = None) -> dict:
    """Turn a single detected foot into one feature dictionary."""

    clean_region = orient_foot_region(foot_region)
    grid = np.asarray(clean_region.get("grid", []), dtype=float)
    contact = np.asarray(clean_region.get("contact", []), dtype=np.uint8)
    bbox = clean_region.get("bbox", (0, -1, 0, -1))

    metrics = compute_arch_metrics(clean_region)
    zones = split_into_three_zones(clean_region)
    z1_top, z1_bottom = zones["zone1"]
    z2_top, z2_bottom = zones["zone2"]
    z3_top, z3_bottom = zones["zone3"]

    zone1_grid = grid[z1_top : z1_bottom + 1, :] if z1_bottom >= z1_top else np.zeros((0, 0))
    zone2_grid = grid[z2_top : z2_bottom + 1, :] if z2_bottom >= z2_top else np.zeros((0, 0))
    zone3_grid = grid[z3_top : z3_bottom + 1, :] if z3_bottom >= z3_top else np.zeros((0, 0))

    zone1_contact = contact[z1_top : z1_bottom + 1, :] if z1_bottom >= z1_top else np.zeros((0, 0))
    zone2_contact = contact[z2_top : z2_bottom + 1, :] if z2_bottom >= z2_top else np.zeros((0, 0))
    zone3_contact = contact[z3_top : z3_bottom + 1, :] if z3_bottom >= z3_top else np.zeros((0, 0))

    zone1_area = int(zone1_contact.sum())
    zone2_area = int(zone2_contact.sum())
    zone3_area = int(zone3_contact.sum())
    total_contact = zone1_area + zone2_area + zone3_area

    zone1_pressures = zone1_grid[zone1_contact == 1] if zone1_contact.size else np.array([])
    zone2_pressures = zone2_grid[zone2_contact == 1] if zone2_contact.size else np.array([])
    zone3_pressures = zone3_grid[zone3_contact == 1] if zone3_contact.size else np.array([])

    total_pressure = float(np.sum(grid)) if grid.size else 0.0
    forefoot_pressure = float(np.sum(zone3_grid[zone3_contact == 1])) if zone3_contact.size else 0.0
    hindfoot_pressure = float(np.sum(zone1_grid[zone1_contact == 1])) if zone1_contact.size else 0.0
    pressure_ratio = float(forefoot_pressure / hindfoot_pressure) if hindfoot_pressure > 0 else 0.0

    peak_row, peak_col = _peak_location(grid)

    side_guess = "left"
    if clean_region.get("source_shape") and clean_region.get("center_col") is not None:
        full_width = clean_region["source_shape"][1]
        side_guess = "left" if float(clean_region["center_col"]) < (full_width / 2.0) else "right"

    features = {
        "foot_id": int(clean_region.get("foot_id", 0)),
        "foot_side": side_guess,
        "bbox_top": int(bbox[0]),
        "bbox_bottom": int(bbox[1]),
        "bbox_left": int(bbox[2]),
        "bbox_right": int(bbox[3]),
        "arch_index": float(metrics["arch_index"]),
        "csi": float(metrics["csi"]),
        "cop_x": float(metrics["cop_x"]),
        "cop_y": float(metrics["cop_y"]),
        "total_contact_area": int(total_contact),
        "zone1_contact_area": zone1_area,
        "zone2_contact_area": zone2_area,
        "zone3_contact_area": zone3_area,
        "zone1_avg_pressure": _safe_mean(zone1_pressures),
        "zone2_avg_pressure": _safe_mean(zone2_pressures),
        "zone3_avg_pressure": _safe_mean(zone3_pressures),
        "zone1_max_pressure": _safe_max(zone1_pressures),
        "zone2_max_pressure": _safe_max(zone2_pressures),
        "zone3_max_pressure": _safe_max(zone3_pressures),
        "forefoot_to_hindfoot_pressure_ratio": pressure_ratio,
        "foot_length_cells": int(metrics["foot_length_cells"]),
        "foot_width_cells": int(metrics["foot_width_cells"]),
        "midfoot_min_width": int(metrics["midfoot_min_width"]),
        "peak_pressure_row": int(peak_row),
        "peak_pressure_col": int(peak_col),
        "mean_pressure": float(total_pressure / total_contact) if total_contact > 0 else 0.0,
        "paired_ai_gap": float(paired_ai) if paired_ai is not None else 0.0,
    }
    return features


def extract_features_from_csv(csv_path: str | Path, threshold: float = 5.0, min_size: int = 20) -> list[dict]:
    """Load a CSV and return one feature row per detected foot."""

    _, _, feet = load_and_split(csv_path, threshold=threshold, min_size=min_size)
    if not feet:
        return []

    arch_values = [compute_arch_metrics(foot)["arch_index"] for foot in feet]
    paired_gap = 0.0
    if len(arch_values) == 2:
        paired_gap = abs(float(arch_values[0]) - float(arch_values[1]))

    meta = parse_csv_metadata(csv_path)
    rows: list[dict] = []
    for foot in feet:
        row = extract_features_from_region(foot, paired_ai=paired_gap if len(feet) == 2 else None)
        row.update(meta)
        row["csv_path"] = str(csv_path)
        row["label"] = 1 if row["arch_index"] > 0.26 else 0
        rows.append(row)

    return rows


# API Alias for frontend compatibility
extract_features = extract_features_from_region


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    rows = extract_features_from_csv(args.csv)
    print("features extracted:", len(rows))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()