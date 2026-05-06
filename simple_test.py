#!/usr/bin/env python
"""Simple test that the models can be imported and loaded."""

import pickle
from pathlib import Path

model_dir = Path("models/saved")
print("Testing model loading...")

for pkl_file in model_dir.glob("*.pkl"):
    try:
        with open(pkl_file, "rb") as f:
            bundle = pickle.load(f)
        print(f"✓ {pkl_file.name}: {list(bundle.keys())}")
    except Exception as e:
        print(f"✗ {pkl_file.name}: {e}")

print("Done!")
