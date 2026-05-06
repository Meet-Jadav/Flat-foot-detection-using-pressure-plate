#!/usr/bin/env python3
"""Build a tabular dataset from the raw pressure CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import build_dataset_from_folder


def build_dataset(folder_path: str | Path, threshold: float = 5.0, min_size: int = 20, output_csv: str | Path | None = None) -> pd.DataFrame:
    """Create a dataframe where each row is one detected foot."""

    df = build_dataset_from_folder(folder_path, threshold=threshold, min_size=min_size)
    if df.empty:
        print("no valid samples were found")
        return df

    normal_count = int((df["label"] == 0).sum())
    flatfoot_count = int((df["label"] == 1).sum())
    print(f"normal samples: {normal_count}")
    print(f"flatfoot samples: {flatfoot_count}")
    print(f"total samples: {len(df)}")

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print("saved dataset to", output_csv)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "output" / "features_dataset.csv"))
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--min-size", type=int, default=20)
    args = parser.parse_args()

    build_dataset(args.folder, threshold=args.threshold, min_size=args.min_size, output_csv=args.output)


if __name__ == "__main__":
    main()
