#!/usr/bin/env python3
"""Streamlit frontend for unified flatfoot detection across all three phases."""

import sys
from pathlib import Path
from io import BytesIO
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for phase imports
FINAL_DIR = Path(__file__).resolve().parent.parent  # Go up to final/ directory
sys.path.insert(0, str(FINAL_DIR))
sys.path.insert(0, str(FINAL_DIR / "phase1_rule_based"))
sys.path.insert(0, str(FINAL_DIR / "phase2_ml_models"))
sys.path.insert(0, str(FINAL_DIR / "phase3_deep_hybrid"))

from phase1_rule_based.preprocess import load_csv, binarize_contact, extract_components
from phase1_rule_based.arch_index import arch_index, chippaux_smirak_index, center_of_pressure, orient_foot_region
from phase1_rule_based.features import extract_features
from phase1_rule_based.rule_classifier import classify_flatfoot


st.set_page_config(page_title="Flatfoot Detection System", layout="wide")


def crop_with_padding(array_2d, bbox, pad=3):
    """Crop a 2D array around bbox with optional zero padding."""
    top, bottom, left, right = bbox
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(array_2d.shape[0] - 1, bottom + pad)
    right = min(array_2d.shape[1] - 1, right + pad)
    return array_2d[top : bottom + 1, left : right + 1].copy()


@st.cache_resource
def load_models():
    """Load all available models (cached to avoid reload)."""
    models = {
        "phase1_rule": None,  # Rule-based is always available
        "phase2_svm": None,
        "phase2_rf": None,
        "phase2_knn": None,
        "phase2_lr": None,
        "phase3_cnn": None,
        "phase3_mobilenet": None,
        "phase3_hybrid": None,
    }

    try:
        import pickle
        model_dir = FINAL_DIR / "models" / "saved"
        if (model_dir / "svm_model.pkl").exists():
            with open(model_dir / "svm_model.pkl", "rb") as f:
                models["phase2_svm"] = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load SVM: {e}")

    return models


def get_phase2_models():
    """Load all Phase 2 ML models with proper error handling."""
    import pickle
    model_dir = FINAL_DIR / "models" / "saved"
    models_dict = {}
    
    model_files = {
        "SVM (RBF Kernel)": "svm_rbf.pkl",
        "Random Forest": "random_forest.pkl",
        "KNN": "knn.pkl",
        "Logistic Regression": "logistic_regression.pkl",
    }
    
    for model_name, filename in model_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "rb") as f:
                    bundle = pickle.load(f)
                models_dict[model_name] = bundle
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
        else:
            st.warning(f"{model_name} model file not found: {filename}")
    
    return models_dict


def predict_with_phase2(model_bundle, feature_dict):
    """Make a prediction with a Phase 2 ML model."""
    if not model_bundle:
        return None
    
    try:
        # Get the pipeline and feature columns from the bundle
        pipeline = model_bundle.get("pipeline") or model_bundle.get("model")
        feature_columns = model_bundle.get("feature_columns", [])
        
        if not pipeline or not feature_columns:
            st.error("Model bundle is incomplete (missing pipeline or feature_columns)")
            return None
        
        # Prepare the feature vector (exclude non-numeric and leaked features)
        feature_vector = []
        missing_features = []
        for col in feature_columns:
            if col in feature_dict:
                val = feature_dict[col]
                if isinstance(val, (int, float)):
                    feature_vector.append(float(val))
                else:
                    missing_features.append(col)
            else:
                missing_features.append(col)
        
        if missing_features and len(missing_features) < len(feature_columns):
            st.warning(f"Missing {len(missing_features)} features")
        
        # Convert to numpy array and reshape for sklearn
        X_sample = np.array([feature_vector])
        
        # Make prediction
        y_pred = pipeline.predict(X_sample)[0]
        
        # Try to get probability if available
        y_prob = None
        try:
            y_proba = pipeline.predict_proba(X_sample)[0]
            y_prob = y_proba[1] if len(y_proba) > 1 else y_proba[0]
        except:
            pass
        
        return {
            "prediction": "Flatfoot" if y_pred == 1 else "Normal",
            "probability": float(y_prob) if y_prob is not None else None,
            "confidence": float(y_prob) if y_prob is not None else None,
            "raw_prediction": int(y_pred),
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)[:100]}")
        return None


def process_csv_file(uploaded_file):
    """Load and preprocess a single CSV file."""
    try:
        grid_data = load_csv(uploaded_file)
        binary = binarize_contact(grid_data, threshold=5.0)
        components = extract_components(binary)

        feet = []
        for comp_id, component in enumerate(components, 1):
            bbox = component["bbox"]
            cropped = crop_with_padding(grid_data, bbox, pad=3)
            contact_crop = crop_with_padding(binary, bbox, pad=3)
            oriented = orient_foot_region({
                "foot_id": comp_id,
                "grid": cropped,
                "contact": contact_crop,
                "bbox": bbox,
                "area": component["area"],
                "mask": component["mask"],
                "source_shape": tuple(grid_data.shape),
                "center_col": float((bbox[2] + bbox[3]) / 2.0),
            })
            feet.append({
                "component_id": comp_id,
                "grid": oriented,
                "bbox": bbox,
            })

        return feet, grid_data, binary

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None


def display_pressure_map(grid_data, title="Pressure Distribution"):
    """Display a pressure map heatmap."""
    grid_data = np.asarray(grid_data, dtype=float)
    height, width = grid_data.shape if grid_data.ndim == 2 else (1, 1)
    fig_width = max(7.0, min(12.0, width / 4.0))
    fig_height = max(8.0, min(14.0, height / 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    im = ax.imshow(grid_data, cmap="hot", origin="upper", aspect="auto", interpolation="bicubic")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pressure (Pa)")
    plt.tight_layout()
    return fig


def display_features_table(features_dict):
    """Display extracted features in a nice table."""
    df = pd.DataFrame({
        "Feature": list(features_dict.keys()),
        "Value": [f"{v:.2f}" if isinstance(v, float) else str(v) for v in features_dict.values()],
    })
    st.dataframe(df, use_container_width=True)


def main():
    st.title("🦶 Flatfoot Detection System")
    st.markdown("**Three-Phase Analysis: Rule-Based → Classical ML → Deep Learning**")

    # Sidebar: Configuration
    st.sidebar.header("⚙️ Configuration")
    phase_selection = st.sidebar.multiselect(
        "Which phases to run?",
        ["Phase 1: Rule-Based", "Phase 2: Classical ML", "Phase 3: Deep Learning"],
        default=["Phase 1: Rule-Based"],
    )

    # Load Phase 2 models once
    phase2_models = None
    if "Phase 2: Classical ML" in phase_selection:
        ml_model = st.sidebar.selectbox(
            "Select ML model",
            ["SVM (RBF Kernel)", "Random Forest", "KNN", "Logistic Regression"],
        )
        phase2_models = get_phase2_models()
    else:
        ml_model = None

    if "Phase 3: Deep Learning" in phase_selection:
        dl_model = st.sidebar.selectbox(
            "Select DL architecture",
            ["Custom CNN", "MobileNetV2 Transfer", "CNN+SVM Hybrid"],
        )
    else:
        dl_model = None

    # Main interface
    st.header("📊 Analysis")

    uploaded_files = st.file_uploader(
        "Upload pressure plate CSV file(s)",
        type="csv",
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("👆 Upload one or more CSV files to begin analysis")
        return

    results_summary = []

    for file_idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"File {file_idx + 1}: {uploaded_file.name}")

        feet, grid_data, binary = process_csv_file(uploaded_file)
        if feet is None:
            continue

        st.write(f"**Detected {len(feet)} foot region(s)**")

        for foot_idx, foot in enumerate(feet, 1):
            st.write(f"**Foot {foot_idx} of {len(feet)}**")
            foot_region = foot["grid"]
            pressure_grid = foot_region.get("grid", foot_region)

            col1, col2 = st.columns(2)

            with col1:
                fig_pressure = display_pressure_map(np.asarray(pressure_grid, dtype=float), title=f"Foot {foot_idx} Pressure Map")
                st.pyplot(fig_pressure, use_container_width=True)

            with col2:
                # Extract features
                features = extract_features(foot_region)
                st.markdown("#### Extracted Features")
                display_features_table(features)

            # Phase 1: Rule-Based
            if "Phase 1: Rule-Based" in phase_selection:
                st.markdown("### Phase 1: Rule-Based Classification")
                rule_result = classify_flatfoot(features)
                col_prob, col_label, col_explain = st.columns(3)

                with col_prob:
                    st.metric("Probability", f"{rule_result.get('flatfoot_probability_like', rule_result.get('score', 0.0)):.2%}")

                with col_label:
                    st.metric("Label", rule_result["label"])

                with col_explain:
                    st.markdown(f"**Explanation**\n\n{rule_result.get('confidence_note', 'No explanation available.')}")

            # Phase 2: Classical ML
            if "Phase 2: Classical ML" in phase_selection:
                st.markdown("### Phase 2: Classical Machine Learning")
                if ml_model and phase2_models and ml_model in phase2_models:
                    model_bundle = phase2_models[ml_model]
                    ml_result = predict_with_phase2(model_bundle, features)
                    
                    if ml_result:
                        col_pred, col_conf = st.columns(2)
                        with col_pred:
                            st.metric("Prediction", ml_result["prediction"])
                        with col_conf:
                            if ml_result["probability"] is not None:
                                st.metric("Confidence", f"{ml_result['probability']:.2%}")
                            else:
                                st.metric("Confidence", "N/A")
                        st.markdown(f"**Model:** {ml_model}")
                    else:
                        st.warning(f"Could not make prediction with {ml_model}")
                else:
                    st.warning(f"Model {ml_model} not loaded or not available")

            # Phase 3: Deep Learning
            if "Phase 3: Deep Learning" in phase_selection:
                st.markdown("### Phase 3: Deep Learning & Hybrids")
                st.info(f"Would use {dl_model} here (model loading in development)")
                # TODO: load and predict with actual model, show Grad-CAM

            results_summary.append({
                "File": uploaded_file.name,
                "Foot": foot_idx,
                "AI": f"{features.get('arch_index', 0):.3f}",
                "Rule Label": rule_result.get("label", "N/A") if "Phase 1: Rule-Based" in phase_selection else "N/A",
            })

            st.divider()

    # Summary table
    if results_summary:
        st.header("📋 Summary")
        summary_df = pd.DataFrame(results_summary)
        st.dataframe(summary_df, use_container_width=True)

        # Download report
        report_text = "FLATFOOT DETECTION REPORT\n" + "=" * 50 + "\n\n"
        for row in results_summary:
            report_text += f"File: {row['File']}\n"
            report_text += f"Foot: {row['Foot']}\n"
            report_text += f"Arch Index: {row['AI']}\n"
            report_text += f"Classification: {row['Rule Label']}\n\n"

        st.download_button(
            label="📥 Download Report (TXT)",
            data=report_text,
            file_name="flatfoot_detection_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
