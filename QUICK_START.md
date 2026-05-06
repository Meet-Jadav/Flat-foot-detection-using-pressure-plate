# Quick Start Guide - Flatfoot Detection System

## ⚠️ IMPORTANT: PowerShell Path Syntax

When using paths with spaces in PowerShell, **ALWAYS USE QUOTES**:

### ❌ WRONG (will fail):
```powershell
cd e:\Flat Foot\final
```

### ✅ CORRECT (use quotes):
```powershell
cd "e:\Flat Foot\final"
```

OR use the venv activation script shortcut:

```powershell
& "e:\Flat Foot\.venv\Scripts\Activate.ps1"
cd "e:\Flat Foot\final"
```

---

## 🚀 Launch Options

### Option 1: Web Interface (Recommended)

```powershell
# Activate venv first
& "e:\Flat Foot\.venv\Scripts\Activate.ps1"

# Navigate to project
cd "e:\Flat Foot\final"

# Launch Streamlit
python -m streamlit run frontend/app.py
```

Then open browser: http://localhost:8501

### Option 2: Compare All Models

```powershell
& "e:\Flat Foot\.venv\Scripts\Activate.ps1"
cd "e:\Flat Foot\final"
python compare_models.py
```

### Option 3: Use Python API

```python
from pathlib import Path
import pickle

# Load best model (Logistic Regression)
with open("final/models/saved/logistic_regression.pkl", "rb") as f:
    bundle = pickle.load(f)
    model = bundle["pipeline"]
    features = bundle["feature_columns"]

# Make prediction
prediction = model.predict(X_test)
```

---

## 📊 Model Performance (After Latest Fix)

| Model | Accuracy | Best For |
|-------|----------|----------|
| **Logistic Regression** ⭐ | 97.78% | Production use |
| SVM (RBF) | 95.88% | Alternative |
| Random Forest | 100.00% | Watch for overfitting |
| KNN (k=5) | 82.61% | Not recommended |

---

## 🔧 Troubleshooting

### Problem: "The term 'streamlit' is not recognized"
**Solution:** Activate venv first: `& "e:\Flat Foot\.venv\Scripts\Activate.ps1"`

### Problem: "Set-Location: A positional parameter cannot be found"
**Solution:** Use quotes: `cd "e:\Flat Foot\final"`

### Problem: Models fail to load in compare_models.py
**Solution:** Already fixed! Run again: `python compare_models.py`

---

## 📁 Project Structure

```
e:\Flat Foot\final\
├── frontend/
│   ├── app.py                    # Streamlit web interface
│   └── README.md
├── phase1_rule_based/            # Rule-based detection (5 modules)
│   ├── preprocess.py
│   ├── arch_index.py
│   ├── features.py
│   ├── rule_classifier.py
│   └── visualize.py
├── phase2_ml_models/             # ML models (8 modules)
│   ├── svm_model.py
│   ├── random_forest_model.py
│   ├── knn_model.py
│   ├── logistic_model.py
│   ├── common.py
│   ├── evaluate.py
│   ├── dataset_builder.py
│   └── dataset.csv               # 4,739 training samples
├── phase3_deep_hybrid/           # Deep learning (optional)
│   ├── cnn_model.py
│   ├── transfer_learning.py
│   ├── hybrid_cnn_svm.py
│   ├── rule_dl_hybrid.py
│   └── grad_cam.py
├── models/saved/
│   ├── logistic_regression.pkl   # RECOMMENDED
│   ├── svm_rbf.pkl
│   ├── random_forest.pkl
│   └── knn.pkl
└── compare_models.py             # Model comparison tool
```

---

## 📞 Next Steps

1. **Test now**: `python -m streamlit run frontend/app.py`
2. **Upload a CSV** from e:\Flat Foot\Pressure_Data\
3. **Get predictions** with confidence scores
4. **Download report** (PDF/CSV)

---

**Status**: ✅ Phase 1+2 READY | 🔄 Phase 3 TRAINING NOW
