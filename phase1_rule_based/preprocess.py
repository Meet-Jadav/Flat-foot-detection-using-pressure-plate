#!/usr/bin/env python3
"""Phase 1 preprocessing helpers for pressure plate CSV files.

This file is the boring part, but it has to be solid because everything else
builds on it later.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_THRESHOLD = 5.0
DEFAULT_MIN_COMPONENT_SIZE = 20


def load_csv_grid(csv_path: str | Path) -> np.ndarray:
    """Load a CSV pressure grid and pad short rows with zeros.

    The OHM 3000 exports are usually clean, but a few files in the dataset have
    uneven row lengths or stray empty values, so I normalize them here instead
    of letting the whole pipeline crash.
    """

    rows: list[list[float]] = []

    try:
        if hasattr(csv_path, "getvalue"):
            raw_text = csv_path.getvalue().decode("utf-8", errors="ignore")
            handle = raw_text.splitlines()
            reader = csv.reader(handle)
            for raw_row in reader:
                if not raw_row:
                    continue

                cleaned_row: list[float] = []
                for cell in raw_row:
                    cell_text = str(cell).strip()
                    if not cell_text:
                        cleaned_row.append(0.0)
                        continue
                    try:
                        cleaned_row.append(float(cell_text))
                    except ValueError:
                        cleaned_row.append(0.0)
                rows.append(cleaned_row)
        else:
            csv_path = Path(csv_path)
            with csv_path.open("r", newline="") as handle:
                reader = csv.reader(handle)
                for raw_row in reader:
                    if not raw_row:
                        continue

                    cleaned_row: list[float] = []
                    for cell in raw_row:
                        cell_text = str(cell).strip()
                        if not cell_text:
                            cleaned_row.append(0.0)
                            continue
                        try:
                            cleaned_row.append(float(cell_text))
                        except ValueError:
                            cleaned_row.append(0.0)
                    rows.append(cleaned_row)
    except OSError:
        print("something went wrong with the CSV, check if it exists")
        return np.zeros((0, 0), dtype=float)

    if not rows:
        return np.zeros((0, 0), dtype=float)

    max_cols = max(len(row) for row in rows)
    if max_cols == 0:
        return np.zeros((len(rows), 0), dtype=float)

    fixed_rows = []
    for row in rows:
        if len(row) < max_cols:
            row = row + [0.0] * (max_cols - len(row))
        fixed_rows.append(row[:max_cols])

    return np.asarray(fixed_rows, dtype=float)


def contact_map(grid_data: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Convert pressure values into a binary contact map."""

    if grid_data.size == 0:
        return np.zeros_like(grid_data, dtype=np.uint8)
    return (grid_data > threshold).astype(np.uint8)


def _component_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _ in coords]
    cols = [col for _, col in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _crop_with_bbox(array_2d: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = bbox
    return array_2d[top : bottom + 1, left : right + 1].copy()


def connected_components(
    contact_grid: np.ndarray,
    min_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> list[dict]:
    """Find connected contact regions with a flood-fill style search.

    Only 4-neighbour connectivity is used because that was the easiest thing to
    reason about while I was testing the dataset. Tiny noise blobs are ignored.
    """

    if contact_grid.size == 0:
        return []

    rows, cols = contact_grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components: list[dict] = []

    for row in range(rows):
        for col in range(cols):
            if contact_grid[row, col] != 1 or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cur_row, cur_col = stack.pop()
                coords.append((cur_row, cur_col))

                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = cur_row + delta_row
                    next_col = cur_col + delta_col
                    if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                        continue
                    if visited[next_row, next_col] or contact_grid[next_row, next_col] != 1:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            if len(coords) < min_size:
                continue

            bbox = _component_bbox(coords)
            mask = np.zeros_like(contact_grid, dtype=np.uint8)
            for comp_row, comp_col in coords:
                mask[comp_row, comp_col] = 1

            components.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "area": len(coords),
                    "mask": mask,
                    "crop": _crop_with_bbox(contact_grid, bbox),
                }
            )

    components.sort(key=lambda item: (item["bbox"][2], item["bbox"][0]))
    return components


def split_feet_from_grid(
    grid_data: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    min_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> list[dict]:
    """Return each detected foot as its own region dictionary."""

    if grid_data.size == 0:
        return []

    contact = contact_map(grid_data, threshold=threshold)
    components = connected_components(contact, min_size=min_size)

    foot_regions: list[dict] = []
    for index, component in enumerate(components):
        top, bottom, left, right = component["bbox"]
        foot_grid = grid_data[top : bottom + 1, left : right + 1].copy()
        foot_contact = component["crop"].copy()
        foot_regions.append(
            {
                "foot_id": index,
                "grid": foot_grid,
                "contact": foot_contact,
                "bbox": component["bbox"],
                "area": component["area"],
                "mask": component["mask"],
                "source_shape": tuple(grid_data.shape),
                "center_col": float((left + right) / 2.0),
            }
        )

    return foot_regions


def load_and_split(
    csv_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load a CSV file, threshold it, and split it into separate feet."""

    grid_data = load_csv_grid(csv_path)
    if grid_data.size == 0:
        return grid_data, np.zeros_like(grid_data, dtype=np.uint8), []

    contact = contact_map(grid_data, threshold=threshold)
    feet = split_feet_from_grid(grid_data, threshold=threshold, min_size=min_size)
    return grid_data, contact, feet


def iter_csv_files(folder_path: str | Path) -> Iterable[Path]:
    """Yield CSV files in a folder in a predictable order."""

    folder_path = Path(folder_path)
    if not folder_path.exists():
        return []
    return sorted(folder_path.glob("*.csv"))


# API Aliases for consistency with frontend and other modules
load_csv = load_csv_grid
binarize_contact = contact_map
extract_components = connected_components
component_bbox = _component_bbox
crop_foot_with_padding = _crop_with_bbox


def main() -> None:
    """Tiny manual test for the loader."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_COMPONENT_SIZE)
    args = parser.parse_args()

    grid_data, contact, feet = load_and_split(args.csv, threshold=args.threshold, min_size=args.min_size)
    print("grid shape:", grid_data.shape)
    print("contact cells:", int(contact.sum()))
    print("feet found:", len(feet))
    for foot in feet:
        print("foot", foot["foot_id"], "bbox", foot["bbox"], "area", foot["area"])


if __name__ == "__main__":
    main()
