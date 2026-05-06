# Streamlit Frontend: Flatfoot Detection Web Interface

## Quick Start

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Features

### 📤 File Upload
- Select one or more CSV files from OHM 3000 pressure plate
- Auto-detects number of feet per file
- Processes each foot independently

### 📊 Visualization
- Pressure heatmap (original grid)
- Pressure map with zone overlay (forefoot/midfoot/hindfoot)
- Pressure profile (sum across rows)
- Feature extraction table

### 🔍 Phase 1: Rule-Based
- Arch Index computed and displayed
- Fuzzy classification (normal/borderline/mild/moderate/severe)
- Explanation text: "AI of 0.31 exceeds moderate threshold..."
- Clinical grounding

### 🤖 Phase 2: Classical ML
- Model selector (SVM, Random Forest, KNN, Logistic Regression)
- Probability bar
- Model comparison table (if multiple models available)
- Feature importance (for Random Forest, Logistic Regression)

### 🧠 Phase 3: Deep Learning
- Model selector (Custom CNN, MobileNetV2, CNN+SVM Hybrid)
- Probability + confidence score
- Grad-CAM visualization (shows which regions CNN focused on)
- CNN decision explanation

### 📋 Summary & Report
- Comparison table across all uploaded files
- Download report as TXT file
- Quick statistics (% flatfoot, mean AI, etc.)

---

## Configuration Sidebar

**Select Phases to Run:**
- Phase 1: Rule-Based (always fast, always available)
- Phase 2: Classical ML (requires trained models)
- Phase 3: Deep Learning (requires trained models)

**Model Selection:**
- Choose specific ML model to use for Phase 2
- Choose specific DL architecture for Phase 3

---

## File Format

Expected CSV structure:
- 48 columns (OHM 3000 grid width)
- Variable rows (20-120 typical)
- Comma-separated floats
- 0 = no contact, positive = pressure in Pa

Example: `subject1_normal_trial1_pressure.csv`

---

## Limitations (Current)

- **Model loading:** Currently placeholder; models need to be implemented
- **Single uploads:** Works best with 1-5 files at a time (large batch may be slow)
- **Preprocessing:** Uses Phase 1 pipeline; assumes standard OHM 3000 format
- **No session persistence:** Refresh page to reset

---

## Troubleshooting

**Q: "Error processing file"**
A: Check CSV format (48 columns, comma-separated floats). Run Phase 1 preprocessing manually to debug.

**Q: "Could not load SVM"**
A: Model not found at `models/saved/svm_model.pkl`. Train Phase 2 models first.

**Q: Page is slow with large files**
A: 48×100 grids are manageable. If >150 rows, app may lag on replot. Reduce batch size or optimize Matplotlib.

---

## Future Enhancements

- [ ] Add confidence threshold slider (only show predictions > confidence)
- [ ] Batch upload with queue processing
- [ ] Export visualizations as high-res PNG
- [ ] Compare predictions across phases (rule vs. ML vs. DL) side-by-side
- [ ] Real-time model comparison: toggle between models to see differences
- [ ] Bilateral asymmetry analysis (if left + right foot detected)
- [ ] Historical comparison (upload previous patient's data to compare)
- [ ] PDF report generation

---

## Architecture Notes

**Why Streamlit?**
- Rapid prototyping: write Python, auto-generate UI
- No frontend framework needed (no React, Vue, etc.)
- Deploy anywhere: AWS, Heroku, Streamlit Cloud
- Rerun-based architecture: clean code, less state management

**Integration with Phases:**
- Phase 1: Always available (no training required)
- Phase 2: Requires trained models in `models/saved/*.pkl`
- Phase 3: Requires trained models + PyTorch/TensorFlow

**Data Flow:**
```
CSV Upload
    ↓
Phase 1: Preprocess (extract components, features) → Display
    ↓
Phase 2 (if selected): Load ML model, predict → Display
    ↓
Phase 3 (if selected): Load DL model, predict, Grad-CAM → Display
    ↓
Summary Table + Download Report
```

---

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repo at https://share.streamlit.io
3. Select branch and app file
4. Done! App auto-deploys on push

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD streamlit run frontend/app.py --server.port 8501
```

```bash
docker build -t flatfoot-app .
docker run -p 8501:8501 flatfoot-app
```

---

*Last updated: [date]*
