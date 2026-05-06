#!/usr/bin/env python3
"""Matplotlib plots for the pressure heatmaps and row profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from .arch_index import compute_arch_metrics, orient_foot_region
from .preprocess import load_and_split


def _plot_zones(ax, bbox, zone1, zone2, zone3, color="white"):
    top, bottom, left, right = bbox
    for row_start, row_end, label in ((zone3[0], zone3[1], "forefoot"), (zone2[0], zone2[1], "midfoot"), (zone1[0], zone1[1], "heel")):
        if row_end < row_start:
            continue
        ax.axhline(row_start - 0.5, color=color, linewidth=1.2, linestyle="--")
        ax.axhline(row_end + 0.5, color=color, linewidth=1.2, linestyle="--")
        ax.text(left, row_start, label, color=color, fontsize=9, va="bottom", ha="left", bbox=dict(facecolor="black", alpha=0.25, edgecolor="none"))


def _plot_one_heatmap(ax, grid, title, bbox=None, metrics=None):
    if grid.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "empty grid", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return None

    image = ax.imshow(grid, cmap="jet", aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("columns")
    ax.set_ylabel("rows")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if bbox is not None:
        top, bottom, left, right = bbox
        rect = patches.Rectangle((left - 0.5, top - 0.5), right - left + 1, bottom - top + 1, fill=False, edgecolor="white", linewidth=2)
        ax.add_patch(rect)

    if metrics is not None:
        _plot_zones(ax, bbox, metrics["zone1_bounds"], metrics["zone2_bounds"], metrics["zone3_bounds"])

    return image


def plot_heatmap_with_profile(grid: np.ndarray, foot_region: dict | None = None, save_path: str | Path | None = None, title: str = "pressure heatmap"):
    """Plot a heatmap and the row-sum profile side by side."""

    grid = np.asarray(grid, dtype=float)
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    layout = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.3])
    ax_heatmap = fig.add_subplot(layout[0, 0])
    ax_profile = fig.add_subplot(layout[0, 1])

    bbox = None
    metrics = None
    if foot_region is not None:
        clean_region = orient_foot_region(foot_region)
        bbox = clean_region.get("bbox")
        metrics = compute_arch_metrics(clean_region)

    _plot_one_heatmap(ax_heatmap, grid, title, bbox=bbox, metrics=metrics)

    row_profile = np.sum(grid, axis=1) if grid.size else np.array([])
    ax_profile.plot(row_profile, np.arange(len(row_profile)), color="#2c3e50", linewidth=2)
    ax_profile.invert_yaxis()
    ax_profile.set_title("row pressure profile")
    ax_profile.set_xlabel("sum of pressure")
    ax_profile.set_ylabel("row index")
    ax_profile.grid(True, alpha=0.25)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")

    return fig


def plot_full_grid_and_feet(csv_path: str | Path, threshold: float = 5.0, save_dir: str | Path | None = None):
    """Plot the raw full grid and each detected foot separately."""

    grid_data, _, feet = load_and_split(csv_path, threshold=threshold)
    save_dir = Path(save_dir) if save_dir else Path(__file__).resolve().parent / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_stem = Path(csv_path).stem
    figures = []

    full_path = save_dir / f"{csv_stem}_full_grid.png"
    figures.append(plot_heatmap_with_profile(grid_data, save_path=full_path, title=f"full grid - {csv_stem}"))

    for foot_region in feet:
        foot_grid = np.asarray(foot_region["grid"], dtype=float)
        foot_path = save_dir / f"{csv_stem}_foot_{foot_region['foot_id']}.png"
        figures.append(plot_heatmap_with_profile(foot_grid, foot_region=foot_region, save_path=foot_path, title=f"foot {foot_region['foot_id']}"))

    return figures


# API Alias for frontend compatibility
plot_grid_and_zones = plot_heatmap_with_profile


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    plot_full_grid_and_feet(args.csv, threshold=args.threshold, save_dir=args.save_dir)
    print("saved figures for", args.csv)


if __name__ == "__main__":
    main()