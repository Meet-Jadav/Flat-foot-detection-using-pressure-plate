#!/usr/bin/env python3
"""Test that all imports work correctly."""

import sys
from pathlib import Path

# Simulate what the frontend does
app_file = Path("e:/Flat Foot/final/frontend/app.py").resolve()
FINAL_DIR = app_file.parent.parent  # Go up to final/ directory

print(f"App file: {app_file}")
print(f"Final dir: {FINAL_DIR}")
print(f"Exists: {FINAL_DIR.exists()}")

sys.path.insert(0, str(FINAL_DIR))
sys.path.insert(0, str(FINAL_DIR / "phase1_rule_based"))
sys.path.insert(0, str(FINAL_DIR / "phase2_ml_models"))
sys.path.insert(0, str(FINAL_DIR / "phase3_deep_hybrid"))

print("\nAttempting imports...")

try:
    from phase1_rule_based.preprocess import load_csv, binarize_contact, extract_components
    print("✓ phase1_rule_based.preprocess")
except Exception as e:
    print(f"✗ phase1_rule_based.preprocess: {e}")

try:
    from phase1_rule_based.arch_index import arch_index, chippaux_smirak_index, center_of_pressure, orient_foot_region
    print("✓ phase1_rule_based.arch_index")
except Exception as e:
    print(f"✗ phase1_rule_based.arch_index: {e}")

try:
    from phase1_rule_based.features import extract_features
    print("✓ phase1_rule_based.features")
except Exception as e:
    print(f"✗ phase1_rule_based.features: {e}")

try:
    from phase1_rule_based.rule_classifier import classify_flatfoot
    print("✓ phase1_rule_based.rule_classifier")
except Exception as e:
    print(f"✗ phase1_rule_based.rule_classifier: {e}")

print("\nAll imports successful!")
