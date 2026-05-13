# AI Flatfoot Detection System: Project Documentation

**Project Title:** Multi-Phase Computational Pipeline for Flatfoot Detection using Plantar Pressure Data  
**Student Name:** U24CS063 - Meet Jadav , U24CS080 - Aryan Mori
**Date:** May 2024  
**Subject:** Biomechanics / Artificial Intelligence  

---

## 1. Project Overview
This project presents a comprehensive, three-phase system for the detection and analysis of flatfoot (pes planus) using data from an **OHM 3000 pressure plate**. The system bridges the gap between traditional clinical methods and modern AI approaches by integrating rule-based heuristics, classical machine learning, and deep learning architectures.

### Key Objectives:
- **Automate Screening:** Reduce manual effort in analyzing footprint tracings.
- **Enhance Accuracy:** Use machine learning to identify patterns beyond simple thresholds.
- **Provide Explainability:** Use clinical indices and Grad-CAM visualizations to ensure clinicians can trust the AI's decisions.

---

## 2. Methodology & Architecture
The system is designed in three distinct phases, each offering different trade-offs between interpretability and raw performance.

### Phase 1: Rule-Based Detection
*   **Concept:** Utilizes established clinical literature to compute foot indices.
*   **Key Metrics:**
    *   **Arch Index (AI):** Ratio of midfoot contact area to total contact area (excluding toes).
    *   **Chippaux-Smirak Index (CSI):** Ratio of the narrowest part of the midfoot to the widest part of the forefoot.
*   **Classification:** Uses fixed thresholds (e.g., AI > 0.26 indicates flatfoot).
*   **Pros:** 100% interpretable, clinically validated, requires no training data.

### Phase 2: Classical Machine Learning
*   **Concept:** Trains statistical models on 26 extracted features (pressure distribution, geometric ratios, asymmetry).
*   **Models Implemented:**
    *   **Logistic Regression:** Most reliable and interpretable linear model.
    *   **Support Vector Machine (SVM):** Excellent for high-dimensional feature spaces.
    *   **Random Forest:** Ensemble method to capture non-linear relationships.
    *   **K-Nearest Neighbors (KNN):** Pattern-based classification.
*   **Performance:** Achieved up to **97.78% accuracy** with Logistic Regression.

### Phase 3: Deep Learning & Hybrids
*   **Concept:** Uses Convolutional Neural Networks (CNN) to learn features directly from raw 2D pressure maps.
*   **Approaches:**
    *   **Custom CNN:** Built with PyTorch for spatial feature extraction.
    *   **Transfer Learning:** Utilizes MobileNetV2 architecture for high performance on smaller datasets.
    *   **Rule-DL Hybrid:** A weighted ensemble that combines clinical rules with neural network confidence.
*   **Explainability:** Integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) to highlight which regions of the foot the model focused on during prediction.

---

## 3. Data Processing Pipeline
The "raw" data consists of CSV files from the OHM 3000 pressure plate (48 columns representing a grid).

1.  **Loading & Binarization:** Data is cleaned and converted to a binary contact map using Otsu's thresholding.
2.  **Foot Extraction:** A flood-fill algorithm identifies connected components to separate left and right feet from background noise.
3.  **Alignment & Normalization:** Feet are rotated to a vertical principal axis and cropped to a standard orientation (forefoot at the top).
4.  **Zoning:** Each foot is automatically divided into four clinical zones: **Toe, Metatarsal, Midfoot, and Heel**.
5.  **Feature Extraction:** 26 scalar features are computed per foot, including pressure-weighted indices and symmetry scores.

---

## 4. Technology Stack
*   **Programming Language:** Python 3.10+
*   **Data Science:** NumPy, Pandas (Matrix manipulation and dataframes).
*   **Machine Learning:** Scikit-learn (Classical ML models, scaling, and metrics).
*   **Deep Learning:** PyTorch & Torchvision (Neural network architecture and training).
*   **Visualization:** Matplotlib & Seaborn (Heatmaps, ROC curves, confusion matrices).
*   **Frontend/Deployment:** Streamlit (Interactive web application for real-time analysis).

---

## 5. Results & Performance Comparison

| Model | Accuracy | Sensitivity | Specificity | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1 (Rules)** | ~80-85% | 75% | 85% | Initial Screening |
| **Logistic Regression** | **97.78%** | 97.29% | 100% | **Clinical Production** |
| **SVM (RBF Kernel)** | 95.88% | 95.23% | 98.83% | Robustness |
| **Random Forest** | 94.72% | 97.68% | 81.29% | Feature Importance |
| **Deep CNN (Hybrid)** | ~92-96% | 90% | 94% | Complex Patterns |

---

## 6. Project Structure
```
Flat Foot Detection/
├── phase1_rule_based/      # Core logic for Arch Index and Preprocessing
├── phase2_ml_models/       # Training scripts and dataset builder for ML
├── phase3_deep_hybrid/     # Neural network architectures and Grad-CAM
├── models/saved/           # Serialized (.pkl and .pt) trained models
├── frontend/app.py         # Streamlit web interface source code
├── data/raw/               # Storage for OHM 3000 CSV files
├── compare_models.py       # Benchmarking script for performance evaluation
└── Pipeline.py             # Integrated processing pipeline
```

---

## 7. How to Use the System
1.  **Launch Interface:** Run `streamlit run frontend/app.py` in the terminal.
2.  **Upload Data:** Select an OHM 3000 pressure CSV file.
3.  **Select Phase:** Choose between Rule-Based (Phase 1), Machine Learning (Phase 2), or Deep Learning (Phase 3).
4.  **Analyze:** Review the generated heatmap, zone-wise breakdown, and final classification with confidence scores.
5.  **Export:** Download the analysis report as a PDF/CSV.

---

## 8. Conclusion & Future Work
This project successfully demonstrates that a hybrid approach—combining clinical domain knowledge with advanced machine learning—provides the most reliable results for flatfoot detection. 

**Future Enhancements:**
- Integration of temporal (dynamic) gait analysis.
- Expansion of the dataset to include a wider range of clinical pathologies.
- Mobile application deployment for remote screening in rural areas.

---

## 9. References
- Bayat et al. (2010). "The role of the Arch Index and Chippaux Smirak Index in predicting flat feet." *Journal of Foot and Ankle Surgery*.
- Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV*.
- Scikit-learn Documentation: https://scikit-learn.org
- PyTorch Documentation: https://pytorch.org

---

## 10. Video Demonstration
- Project Demo: https://youtu.be/2j4ahmSphjM


