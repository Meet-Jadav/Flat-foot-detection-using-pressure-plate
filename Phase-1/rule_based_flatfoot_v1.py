#!/usr/bin/env python3
"""Simple Rule-Based Flatfoot Classifier (student version).

What it does:
1. Reads a pressure CSV.
2. Makes contact map using threshold.
3. Finds foot bounding box.
4. Splits box into 3 row zones (forefoot, midfoot, heel).
5. Computes Arch Index (AI) and classifies flatfoot or not.
6. Optionally prints a simple visual with box + partition lines.
"""

import argparse
import csv
from pathlib import Path


def read_grid(csv_path):
    grid = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.reader(f):
            if row:
                grid.append([float(x.strip()) if x.strip() else 0.0 for x in row])

    if not grid:
        raise ValueError("CSV is empty")

    cols = len(grid[0])
    for i, row in enumerate(grid):
        if len(row) != cols:
            raise ValueError(f"Row {i+1} has different column count")

    return grid


def binary_contact(grid, threshold):
    return [[1 if v > threshold else 0 for v in row] for row in grid]


def keep_largest_component(contact):
    rows = len(contact)
    cols = len(contact[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    best_component = []

    for r in range(rows):
        for c in range(cols):
            if contact[r][c] != 1 or visited[r][c]:
                continue

            stack = [(r, c)]
            visited[r][c] = True
            component = []

            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = cr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if visited[nr][nc] or contact[nr][nc] != 1:
                        continue

                    visited[nr][nc] = True
                    stack.append((nr, nc))

            if len(component) > len(best_component):
                best_component = component

    cleaned = [[0 for _ in range(cols)] for _ in range(rows)]
    for r, c in best_component:
        cleaned[r][c] = 1

    return cleaned


def find_bbox(contact):
    rows = len(contact)
    cols = len(contact[0])

    top, bottom = rows, -1
    left, right = cols, -1

    for r in range(rows):
        for c in range(cols):
            if contact[r][c] == 1:
                top = min(top, r)
                bottom = max(bottom, r)
                left = min(left, c)
                right = max(right, c)

    if bottom == -1:
        return None
    return (top, bottom, left, right)


def split_rows_into_3(top, bottom):
    length = bottom - top + 1
    p1 = length // 3
    p2 = length // 3
    p3 = length - p1 - p2

    z3_start = top
    z3_end = z3_start + p1 - 1

    z2_start = z3_end + 1
    z2_end = z2_start + p2 - 1

    z1_start = z2_end + 1
    z1_end = z1_start + p3 - 1

    return (z3_start, z3_end), (z2_start, z2_end), (z1_start, z1_end)


def count_contact(contact, left, right, row_start, row_end):
    total = 0
    for r in range(row_start, row_end + 1):
        for c in range(left, right + 1):
            total += contact[r][c]
    return total


def arch_index_and_class(contact, bbox):
    top, bottom, left, right = bbox
    z3, z2, z1 = split_rows_into_3(top, bottom)

    total_contact = count_contact(contact, left, right, top, bottom)
    mid_contact = count_contact(contact, left, right, z2[0], z2[1])

    ai = mid_contact / total_contact if total_contact > 0 else 0.0

    if ai > 0.26:
        result = "flatfoot"
    else:
        result = "not_flatfoot"

    return ai, result, z3, z2, z1


def print_visual(contact, bbox, z3, z2, z1):
    top, bottom, left, right = bbox
    width = right - left + 1

    border = "+" + ("-" * width) + "+"
    print("\nSimple boxed + partition view")
    print(border)

    for r in range(top, bottom + 1):
        if r == z2[0] or r == z1[0]:
            print(border)

        row = ""
        for c in range(left, right + 1):
            row += "#" if contact[r][c] == 1 else " "

        print("|" + row + "|")

    print(border)
    print("Legend: # = contact sensor")
    print(f"Zone 3 Forefoot rows: {z3[0]} to {z3[1]}")
    print(f"Zone 2 Midfoot rows : {z2[0]} to {z2[1]}")
    print(f"Zone 1 Heel rows    : {z1[0]} to {z1[1]}")


def analyze_one_file(csv_path, threshold, show_visual):
    grid = read_grid(csv_path)
    contact = binary_contact(grid, threshold)
    contact = keep_largest_component(contact)
    bbox = find_bbox(contact)

    if bbox is None:
        print(f"{Path(csv_path).name}: no contact detected")
        return

    ai, result, z3, z2, z1 = arch_index_and_class(contact, bbox)

    print(f"\nFile: {csv_path}")
    print(f"Shape: {len(grid)} x {len(grid[0])}")
    print(f"Bounding box (top,bottom,left,right): {bbox}")
    print(f"Arch Index (AI): {ai:.4f}")
    print(f"Final result: {result}")

    if show_visual:
        print_visual(contact, bbox, z3, z2, z1)


def main():
    parser = argparse.ArgumentParser(description="Simple flatfoot classifier")
    parser.add_argument("--csv", type=str, help="single CSV file path")
    parser.add_argument("--folder", type=str, help="folder with CSV files")
    parser.add_argument("--limit", type=int, default=5, help="max files from folder")
    parser.add_argument("--threshold", type=float, default=5.0, help="contact threshold")
    parser.add_argument("--show-visual", action="store_true", help="print box/partition view")
    args = parser.parse_args()

    if not args.csv and not args.folder:
        raise ValueError("Please provide --csv or --folder")

    if args.csv:
        analyze_one_file(args.csv, args.threshold, args.show_visual)

    if args.folder:
        folder = Path(args.folder)
        files = sorted(folder.glob("*.csv"))[: args.limit]
        for f in files:
            analyze_one_file(str(f), args.threshold, args.show_visual)


if __name__ == "__main__":
    main()
