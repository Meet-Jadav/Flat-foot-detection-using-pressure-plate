#!/usr/bin/env python3
"""Model comparison script - evaluate all Phase 2 models on test data."""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

MODEL_DIR = Path("models/saved")
DATASET_PATH = Path("phase2_ml_models/dataset.csv")

# Load dataset
df = pd.read_csv(DATASET_PATH)
print(f"Loaded {len(df)} samples from dataset")

# Split test set (using last 20% as test, like training did)
test_size = int(len(df) * 0.2)
df_test = df.iloc[-test_size:]
X_test = df_test.drop(["label"] + [col for col in df_test.columns if col in ["csv_path", "file_name", "subject_id", "condition", "trial_id"]], axis=1, errors="ignore")
y_test = df_test["label"]

print(f"Test set size: {len(df_test)}")
print(f"Test set class distribution: {y_test.value_counts().to_dict()}")
print()

# Load and evaluate each model
models_to_test = [
    ("svm_rbf.pkl", "SVM (RBF Kernel)"),
    ("random_forest.pkl", "Random Forest"),
    ("logistic_regression.pkl", "Logistic Regression"),
    ("knn.pkl", "KNN (k=5)"),
]

print("MODEL EVALUATION COMPARISON")
print("=" * 80)
print()

results = []
for model_file, model_name in models_to_test:
    model_path = MODEL_DIR / model_file
    if not model_path.exists():
        print(f"⚠️  {model_name}: Model file not found ({model_path})")
        continue
    
    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        
        # Handle different bundle structures
        if "pipeline" in bundle:
            pipeline = bundle["pipeline"]
        elif "model" in bundle:
            pipeline = bundle["model"]
        else:
            raise ValueError(f"Neither 'pipeline' nor 'model' key found in bundle. Keys: {bundle.keys()}")
        
        feature_columns = bundle["feature_columns"]
        
        # Predict
        X_test_subset = X_test[feature_columns]
        y_pred = pipeline.predict(X_test_subset)
        try:
            y_prob = pipeline.predict_proba(X_test_subset)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = np.nan
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1 Score": f1,
            "AUC": auc,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        })
        
        print(f"✅ {model_name}")
        print(f"   Accuracy:    {acc:.4f} ({int(acc*len(y_test))}/{len(y_test)})")
        print(f"   Sensitivity: {sensitivity:.4f} (True Pos Rate)")
        print(f"   Specificity: {specificity:.4f} (True Neg Rate)")
        print(f"   F1 Score:    {f1:.4f}")
        print(f"   AUC:         {auc:.4f}")
        print(f"   Confusion Matrix:")
        print(f"      TN={tn}, FP={fp}")
        print(f"      FN={fn}, TP={tp}")
        print()
        
    except Exception as e:
        print(f"❌ {model_name}: Error loading model - {e}")
        print()

# Summary table
if results:
    print()
    print("SUMMARY TABLE")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    results_df = results_df[["Model", "Accuracy", "Sensitivity", "Specificity", "F1 Score", "AUC"]]
    print(results_df.to_string(index=False))
    print()
    print(f"Best Model (Accuracy): {results_df.loc[results_df['Accuracy'].idxmax(), 'Model']}")
    print(f"Most Balanced: {results_df.loc[(results_df['Sensitivity'] - results_df['Specificity']).abs().idxmin(), 'Model']}")
