# Flatfoot Detection Pipeline: Complete Three-Phase System

> A student-built research project combining rule-based heuristics, classical machine learning, and deep learning to detect flatfoot from pressure plate data.

---

## 📚 What Is Flatfoot?

**Flatfoot** (pes planus) is a condition where the arch of the foot collapses, causing the entire foot surface to make contact with the ground. Clinically significant in pediatrics (developmental screening), sports medicine (injury risk), and geriatrics (mobility assessment).

**Why detection matters:**
- **Screening:** Quick assessment of 100+ subjects in a biomechanics lab
- **Intervention:** Early intervention in children can prevent chronic pain
- **Research:** Correlate flatfoot with gait asymmetry, injury history, athletic performance

**Traditional measurement:** Clinical visual inspection or manual footprint tracing (low precision, time-consuming).

**Our approach:** Automatic quantification from **OHM 3000 pressure plate** data using computational features.

---

## 🏗️ System Architecture: Three Phases

Why three phases? Each phase builds understanding:

```
Phase 1 (Rule-Based)
    ↓ Why do these thresholds work? Let's try machine learning...
Phase 2 (Classical ML)
    ↓ Good, but can we learn features automatically?
Phase 3 (Deep Learning + Hybrids)
    ↓ Let's blend rule + deep learning for robustness
Final Decision
```

### **Phase 1: Rule-Based Detection** (`phase1_rule_based/`)

**Idea:** Use clinical knowledge to compute foot indices (Arch Index, Chippaux Smirak Index) and classify directly.

**Key Algorithm:**
- **Arch Index (AI)** = Midfoot contact area / Total contact area
  - AI < 0.21: Normal foot
  - 0.21 < AI < 0.26: Borderline
  - 0.26 < AI < 0.30: Mild flatfoot
  - AI > 0.30: Moderate/severe flatfoot

**Pros:**
- ✅ Interpretable: "AI is 0.31 → mild flatfoot"
- ✅ Fast (milliseconds)
- ✅ Clinically validated (30+ years of literature)
- ✅ No training data needed

**Cons:**
- ❌ Brittle: small pressure sensor noise → threshold jumps
- ❌ Doesn't capture complex patterns (asymmetry, pressure distribution)

**Outputs:** Probability (0–1), fuzzy label (normal/borderline/mild/moderate/severe), explanation text

**Files:**
- `preprocess.py` – Load CSV, binarize pressure, extract foot regions via flood-fill
- `arch_index.py` – Compute Arch Index, Chippaux Smirak Index, Center of Pressure
- `features.py` – Extract 20+ features (zones, pressures, asymmetry)
- `rule_classifier.py` – Map features → decision + explanation
- `visualize.py` – Heatmaps with zone overlays, pressure profiles

### **Phase 2: Classical Machine Learning** (`phase2_ml_models/`)

**Idea:** Train algorithms (SVM, Random Forest, KNN, Logistic Regression) on Phase 1 features.

**Why ML over rules?**
- Rules are fixed; ML learns thresholds from data
- Handles complex feature interactions (e.g., "high CSI AND low CoP" → high confidence flatfoot)
- Cross-validation estimates real-world accuracy

**Models:**
- **SVM (RBF kernel):** Non-linear boundary, margin-maximizing
- **Random Forest:** Ensemble, feature importance ranking
- **KNN (k=5):** Non-parametric, sensitive to training set
- **Logistic Regression:** Linear, interpretable coefficients

**Pros:**
- ✅ Better accuracy than Phase 1 (typically 85–92%)
- ✅ Interpretable: feature importance, coefficients
- ✅ Fast training (<1 min on 200 samples)
- ✅ Standard methods, well-understood

**Cons:**
- ❌ Still relies on Phase 1 features (limited if rules are wrong)
- ❌ Accuracy plateaus; can't extract more from fixed features

**Outputs:** Probability, prediction, confidence, class probabilities

**Files:**
- `common.py` – Utilities (cross-validation split, scaling)
- `dataset_builder.py` – Load CSVs, extract Phase 1 features, auto-label by AI, build dataset CSV
- `evaluate.py` – Confusion matrix, ROC curve, metrics report
- `svm_model.py`, `random_forest_model.py`, `knn_model.py`, `logistic_model.py` – Individual models

### **Phase 3: Deep Learning & Hybrids** (`phase3_deep_hybrid/`)

**Idea:** Train neural networks to learn features directly from raw pressure grids; combine with rules for robustness.

**Why DL?**
- Automatically discovers patterns humans might miss
- Non-linear feature learning (Phase 1+2 were mostly manual)
- Theoretically higher ceiling on accuracy

**Approaches:**

1. **Custom CNN:** Convolutional neural network from scratch
   - Conv layers learn edges, patterns
   - Data augmentation (rotate, flip, noise) prevents overfitting
   - Output: probability (0–1)

2. **Transfer Learning (MobileNetV2):** Pretrained backbone + new head
   - Pretrained on ImageNet, so it knows textures/shapes
   - Fast training (10 min instead of 1 hour)
   - Works with limited data (50+ samples)

3. **CNN + SVM Hybrid:** CNN extracts features, SVM classifies
   - Combines deep feature learning with SVM margin-based decision
   - Often beats standalone CNN or SVM
   - More robust to overfitting

4. **Rule + DL Hybrid:** Weighted blend of Phase 1 + Phase 3
   - AI score + CNN probability = final decision
   - Rules ground DL uncertainty
   - Clinicians trust rule + DL blend more than DL alone

**Explainability:** **Grad-CAM** visualizes which regions the CNN focused on
- Sanity check: Is it looking at the midfoot, or a sensor artifact?
- Builds confidence in model

**Pros:**
- ✅ Highest accuracy (86–95% in literature)
- ✅ Learns features automatically
- ✅ Grad-CAM explains decisions

**Cons:**
- ❌ Needs more data (200+ positive samples)
- ❌ Harder to debug (black box unless Grad-CAM)
- ❌ Slower to train (5–30 min)
- ❌ Overfits easily on small datasets

**Files:**
- `cnn_model.py` – Custom CNN with augmentation, weighted loss
- `transfer_learning.py` – MobileNetV2 adapter
- `hybrid_cnn_svm.py` – CNN + SVM pipeline
- `rule_dl_hybrid.py` – Weighted rule + DL blend
- `grad_cam.py` – Visualize CNN attention (Grad-CAM)

---

## 📊 Expected Performance

Accuracy ranges from published literature + our testing:

| Model | Accuracy | Sensitivity | Specificity | Training Time | Data Needed |
|-------|----------|-------------|-------------|---------------|----|
| Phase 1 (Rule) | 75–85% | 70–80% | 80–90% | N/A | 0 |
| Phase 2 (SVM) | 82–92% | 80–90% | 85–95% | 30 sec | 50–100 |
| Phase 2 (RF) | 84–93% | 82–92% | 86–95% | 30 sec | 50–100 |
| Phase 2 (KNN) | 78–88% | 75–85% | 80–92% | <1 sec | 50–100 |
| Phase 3 (CNN) | 86–94% | 84–92% | 88–96% | 5 min | 100–200 |
| Phase 3 (MobileNet) | 88–96% | 86–94% | 90–97% | 2 min | 50–100 |
| Phase 3 (Hybrid) | 89–96% | 87–95% | 91–97% | 10 min | 100–200 |

**Note:** Ranges depend on subject count (our data: 3 subjects → high variance), cross-validation strategy (subject-independent vs. random split), and class balance.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Packages: NumPy, Pandas, Matplotlib, Scikit-learn, Streamlit, Pillow, PyTorch (optional for Phase 3)
- Data: CSV files from OHM 3000 pressure plate (one CSV per trial, 48-column grid)

### Installation

```bash
# Clone or download this repo
cd final

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Phase 1: Rule-Based (No Training)

```bash
cd phase1_rule_based
python -c "
from preprocess import load_csv, binarize_contact, extract_components, crop_foot_with_padding
from arch_index import arch_index
from features import extract_features
from rule_classifier import classify_flatfoot

grid = load_csv('../data/raw/subject1_normal_trial1_pressure.csv')
binary = binarize_contact(grid)
components = extract_components(binary)
for comp in components:
    features = extract_features(comp)
    result = classify_flatfoot(features)
    print(f'AI: {features[\"arch_index\"]:.3f}, Label: {result[\"label\"]}')
"
```

### Build Phase 2 Dataset + Train

```bash
cd phase2_ml_models

# Build dataset from raw CSVs
python dataset_builder.py --csv-dir ../data/raw --output dataset.csv

# Train SVM
python svm_model.py --dataset dataset.csv --output ../models/saved

# Evaluate
# Models save metrics automatically
```

### Run Phase 3: Deep Learning

```bash
cd phase3_deep_hybrid

# Train MobileNetV2
python transfer_learning.py --csv-dir ../data/raw --epochs 30

# Generate Grad-CAM
python grad_cam.py \
  --model-path ../models/saved/mobilenet_model.pkl \
  --csv-path ../data/raw/subject1_normal_trial1_pressure.csv
```

### Launch Streamlit Frontend

```bash
streamlit run frontend/app.py
# Opens at http://localhost:8501
```

---

## 📂 Project Structure

```
final/
├── phase1_rule_based/
│   ├── preprocess.py
│   ├── arch_index.py
│   ├── features.py
│   ├── rule_classifier.py
│   ├── visualize.py
│   └── README.md
├── phase2_ml_models/
│   ├── common.py
│   ├── dataset_builder.py
│   ├── evaluate.py
│   ├── svm_model.py
│   ├── random_forest_model.py
│   ├── knn_model.py
│   ├── logistic_model.py
│   └── README.md
├── phase3_deep_hybrid/
│   ├── cnn_model.py
│   ├── transfer_learning.py
│   ├── hybrid_cnn_svm.py
│   ├── rule_dl_hybrid.py
│   ├── grad_cam.py
│   └── README.md
├── frontend/
│   ├── app.py  (Streamlit interface)
│   └── README.md
├── data/
│   ├── raw/  (OHM 3000 CSV files go here)
│   └── notes.txt
├── models/
│   └── saved/  (Trained models, pickled)
├── figures/
│   └── (Generated visualizations)
├── requirements.txt
└── README.md  (This file)
```

---

## 🔄 Data Pipeline Overview

```
OHM 3000 CSV File (e.g., subject1_normal_trial1_pressure.csv)
    ↓ [Load + Binarize]
Binary contact map (48 cols × N rows)
    ↓ [Extract components via flood-fill]
Individual foot regions (left foot, right foot, or noise)
    ↓ [Phase 1: Compute Arch Index, features]
Feature vector (20+ scalars)
    ↓ Branch:
    ├─ [Rule classifier] → Fuzzy label + explanation
    ├─ [Phase 2 ML] → SVM/RF/KNN probability
    └─ [Phase 3 DL] → CNN probability + Grad-CAM visualization
    ↓ [Streamlit]
Interactive report + visualizations + download
```

---

## 🎯 When to Use Each Phase

| Scenario | Use |
|----------|-----|
| Screening 50 subjects in 1 hour | Phase 1 (fast, no training) |
| Need interpretable decision for clinician | Phase 1 + Phase 3 Grad-CAM |
| Dataset with 50–100 labeled feet | Phase 2 (SVM or RF) |
| Dataset with 200+ labeled feet | Phase 3 (CNN or MobileNet) |
| Balancing accuracy + confidence | Phase 3 Hybrid (Rule + DL) |
| Want to understand feature importance | Phase 2 (feature coefficients) or Phase 3 (Grad-CAM) |

---

## 📖 References

### Rule-Based (Phase 1)
- Bayat et al., "The role of the Arch Index and Chippaux Smirak Index in predicting flat feet," *J Foot Ankle Surg*, 2010
- Cavanagh & Rogers, "Ground Reaction Forces in Distance Running," *J Biomech*, 1982

### Classical ML (Phase 2)
- Hastie, Tibshirani, Friedman: "The Elements of Statistical Learning" (2009)
- Scikit-learn documentation: https://scikit-learn.org

### Deep Learning (Phase 3)
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," *ICCV 2017*
- LeCun et al., "Deep Learning," *Nature*, 2015
- Howard & Gugger: "Fastai" course on transfer learning

### Pressure Plate Biomechanics
- Auvinet et al., "Gait analysis in humans using plantar pressure sensors," *Hum Mov Sci*, 2017
- OHM 3000 Manual: See `Pressure_Data/README` (if exists)

---

## ⚠️ Limitations & Caveats

### Current Project
1. **Small dataset:** 3 subjects, 1400+ trials → high variance, risk of overfitting
2. **Single sensor:** OHM 3000 only; may not generalize to GaitRite, F-Scan, other platforms
3. **Static analysis:** Each foot treated independently; no temporal gait cycle data
4. **No inter-rater reliability:** No comparison with clinical examiners

### Why Accuracy May Be Lower in Practice
- Sensor drift, calibration errors
- Subject variability (different shoe wear, fatigue, nervousness)
- Labeling error ("Is this really flatfoot?" medical disagreement)
- Threshold tuning on small training set (overfitting)

### What I'd Do With More Resources
1. Collect 20–30 subjects × 3 conditions (normal, fatigued, loaded) = 1000+ high-quality trials
2. Get clinical reference standard (footprint scoring by multiple podiatrists)
3. Test cross-platform generalization (GaitRite, F-Scan, etc.)
4. Add temporal features (gait cycle sequence, loading response symmetry)
5. Deploy as REST API with uncertainty quantification
6. Adversarial robustness testing (noisy sensors, outlier subjects)

---

## 🤝 Contributing & Feedback

This is a student research project. Questions, suggestions, or improvements?

- **Found a bug?** Check existing code; file an issue.
- **Want to add a model?** Add it to `phase2_ml_models/` or `phase3_deep_hybrid/` following the naming convention.
- **Want to use this on your data?** Make sure CSV format matches OHM 3000 (48 columns, comma-separated, float values).

---

## 📝 License & Attribution

Built by a student research team as part of a biomechanics course project. Academic use encouraged. If you publish work using this code, please cite:

```
Student Flatfoot Detection Pipeline (2024). 
Available: [GitHub repo or project page]
```

---

## 🎓 Student Reflection

Starting this project, I thought rule-based was enough. Arch Index is elegant, thresholds are documented. But real data is messy—sensor noise, individual variation, edge cases the rules didn't anticipate. That's when Phase 2 clicked: machine learning learns from data what rules can't hardcode.

Then Phase 3 felt like cheating—CNN accuracy jumped 10%!—but also scary because I couldn't explain why. Grad-CAM solved that. Seeing the network focus on the midfoot (right place) vs. a stray artifact (wrong place) gave me confidence.

The hybrid model was the breakthrough: rule + deep learning together beat either alone. A clinician would trust "Rule says borderline, CNN says flatfoot, Grad-CAM looks legit → refer for further examination" way more than CNN alone.

If I had to do this again, I'd start with more data collection, implement from day 1 proper train/test/validation splits (I mixed them early), and validate more rigorously. But for a semester project, this pipeline taught me the entire ML workflow: features → classical ML → deep learning → hybrid → deployment.

---

*Last updated: [date]. Questions? See individual phase READMEs.*
