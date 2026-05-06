#!/usr/bin/env python
"""Quick test of Phase 2 model loading and prediction."""

import pickle
from pathlib import Path
import numpy as np

# Simulate loading models like the frontend does
model_dir = Path("models/saved")

print("=" * 60)
print("Testing Phase 2 Model Loading and Prediction")
print("=" * 60)

model_files = {
    "SVM (RBF Kernel)": "svm_rbf.pkl",
    "Random Forest": "random_forest.pkl",
    "KNN": "knn.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}

for model_name, filename in model_files.items():
    filepath = model_dir / filename
    print(f"\nLoading {model_name}...")
    
    if not filepath.exists():
        print(f"  ❌ File not found: {filename}")
        continue
    
    try:
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)
        
        print(f"  ✓ Loaded successfully")
        print(f"  Keys in bundle: {list(bundle.keys())}")
        
        # Check structure
        pipeline = bundle.get("pipeline") or bundle.get("model")
        feature_columns = bundle.get("feature_columns", [])
        
        if not pipeline:
            print(f"  ⚠ No pipeline/model found in bundle")
            continue
        
        if not feature_columns:
            print(f"  ⚠ No feature_columns found in bundle")
            continue
        
        print(f"  Model has {len(feature_columns)} features")
        
        # Create a dummy feature vector (all zeros)
        X_sample = np.zeros((1, len(feature_columns)))
        
        # Test prediction
        try:
            y_pred = pipeline.predict(X_sample)[0]
            print(f"  Prediction on zero vector: {y_pred}")
            
            # Try probability
            try:
                y_proba = pipeline.predict_proba(X_sample)[0]
                print(f"  Probability: {y_proba}")
            except:
                print(f"  Probability: Not available (may be KNN or RF without proba)")
        
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
        
    except Exception as e:
        print(f"  ❌ Error loading: {e}")

print("\n" + "=" * 60)
print("Phase 2 Model Test Complete")
print("=" * 60)
