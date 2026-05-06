#!/usr/bin/env python3
"""Arch Index, CSI, and CoP helpers for Phase 1.

This is the clinical-ish math part. I kept it simple on purpose because the
main point is to make the pressure map measurable, not to overcomplicate it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocess import load_and_split


def orient_foot_region(foot_region: dict) -> dict:
    """Rotate/flip the foot crop so the heel is closer to the bottom.

    The scans are not always nicely oriented, and this little cleanup makes the
    three-zone split much more stable.
    """

    grid = np.asarray(foot_region.get("grid", []), dtype=float)
    contact = np.asarray(foot_region.get("contact", []), dtype=np.uint8)

    if grid.size == 0 or contact.size == 0:
        return foot_region

    grid = grid.copy()
    contact = contact.copy()

    if grid.shape[1] > grid.shape[0]:
        grid = np.rot90(grid, k=1)
        contact = np.rot90(contact, k=1)

    half_row = max(1, contact.shape[0] // 2)
    top_pressure = float(np.sum(grid[:half_row, :]))
    bottom_pressure = float(np.sum(grid[half_row:, :]))
    if top_pressure > bottom_pressure:
        grid = np.flipud(grid)
        contact = np.flipud(contact)

    new_region = dict(foot_region)
    new_region["grid"] = grid
    new_region["contact"] = contact
    return new_region


def split_into_three_zones(foot_region: dict) -> dict:
    """Split the foot crop into forefoot, midfoot, and heel zones."""

    contact = np.asarray(foot_region.get("contact", []), dtype=np.uint8)
    if contact.size == 0:
        return {"zone1": (0, -1), "zone2": (0, -1), "zone3": (0, -1)}

    rows = contact.shape[0]
    first = rows // 3
    second = rows // 3

    zone3 = (0, max(0, first - 1))
    zone2 = (zone3[1] + 1, min(rows - 1, zone3[1] + second))
    zone1 = (zone2[1] + 1, rows - 1)

    return {"zone1": zone1, "zone2": zone2, "zone3": zone3}


def compute_cop(grid: np.ndarray) -> tuple[float, float]:
    """Center of pressure using the pressure values as weights."""

    if grid.size == 0:
        return 0.0, 0.0

    total_pressure = float(np.sum(grid))
    if total_pressure <= 0:
        return 0.0, 0.0

    rows = np.arange(grid.shape[0], dtype=float)
    cols = np.arange(grid.shape[1], dtype=float)
    row_weights = np.sum(grid, axis=1)
    col_weights = np.sum(grid, axis=0)
    cop_y = float(np.sum(rows * row_weights) / total_pressure)
    cop_x = float(np.sum(cols * col_weights) / total_pressure)
    return cop_x, cop_y


def _row_width(row_contact: np.ndarray) -> int:
    indices = np.where(row_contact > 0)[0]
    if indices.size == 0:
        return 0
    return int(indices.max() - indices.min() + 1)


def compute_arch_metrics(foot_region: dict) -> dict:
    """Return Arch Index, CSI, and CoP together."""

    clean_region = orient_foot_region(foot_region)
    grid = np.asarray(clean_region.get("grid", []), dtype=float)
    contact = np.asarray(clean_region.get("contact", []), dtype=np.uint8)

    if grid.size == 0 or contact.size == 0:
        return {
            "arch_index": 0.0,
            "csi": 0.0,
            "cop_x": 0.0,
            "cop_y": 0.0,
            "foot_length_cells": 0,
            "foot_width_cells": 0,
            "midfoot_min_width": 0,
            "forefoot_max_width": 0,
            "zone1_contact_area": 0,
            "zone2_contact_area": 0,
            "zone3_contact_area": 0,
            "zone1_bounds": (0, -1),
            "zone2_bounds": (0, -1),
            "zone3_bounds": (0, -1),
        }

    zones = split_into_three_zones(clean_region)
    z1_top, z1_bottom = zones["zone1"]
    z2_top, z2_bottom = zones["zone2"]
    z3_top, z3_bottom = zones["zone3"]

    total_contact = int(contact.sum())
    zone2_contact = int(contact[z2_top : z2_bottom + 1, :].sum()) if z2_bottom >= z2_top else 0
    arch_index = float(zone2_contact / total_contact) if total_contact > 0 else 0.0

    widths = [_row_width(contact[row, :]) for row in range(contact.shape[0])]
    fore_widths = widths[z3_top : z3_bottom + 1] if z3_bottom >= z3_top else []
    mid_widths = widths[z2_top : z2_bottom + 1] if z2_bottom >= z2_top else []
    forefoot_max_width = max(fore_widths) if fore_widths else 0
    midfoot_positive = [width for width in mid_widths if width > 0]
    midfoot_min_width = min(midfoot_positive) if midfoot_positive else 0
    csi = float((midfoot_min_width / forefoot_max_width) * 100.0) if forefoot_max_width > 0 else 0.0

    cop_x, cop_y = compute_cop(grid)

    return {
        "arch_index": arch_index,
        "csi": csi,
        "cop_x": cop_x,
        "cop_y": cop_y,
        "foot_length_cells": int(grid.shape[0]),
        "foot_width_cells": int(grid.shape[1]),
        "midfoot_min_width": int(midfoot_min_width),
        "forefoot_max_width": int(forefoot_max_width),
        "zone1_contact_area": int(contact[z1_top : z1_bottom + 1, :].sum()) if z1_bottom >= z1_top else 0,
        "zone2_contact_area": int(zone2_contact),
        "zone3_contact_area": int(contact[z3_top : z3_bottom + 1, :].sum()) if z3_bottom >= z3_top else 0,
        "zone1_bounds": zones["zone1"],
        "zone2_bounds": zones["zone2"],
        "zone3_bounds": zones["zone3"],
    }


# API Aliases for frontend compatibility
def arch_index(foot_region: dict) -> float:
    """Get the Arch Index value from a foot region."""
    metrics = compute_arch_metrics(foot_region)
    return metrics.get("arch_index", 0.0)


def chippaux_smirak_index(foot_region: dict) -> float:
    """Get the Chippaux-Smirak Index from a foot region."""
    metrics = compute_arch_metrics(foot_region)
    return metrics.get("csi", 0.0)


def center_of_pressure(foot_region: dict) -> tuple[float, float]:
    """Get the Center of Pressure coordinates from a foot region."""
    metrics = compute_arch_metrics(foot_region)
    return (metrics.get("cop_x", 0.0), metrics.get("cop_y", 0.0))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()

    grid_data, _, feet = load_and_split(args.csv, threshold=args.threshold)
    print("grid shape:", grid_data.shape)
    print("feet found:", len(feet))
    for foot_region in feet:
        metrics = compute_arch_metrics(foot_region)
        print("foot", foot_region["foot_id"], metrics)


if __name__ == "__main__":
    main()