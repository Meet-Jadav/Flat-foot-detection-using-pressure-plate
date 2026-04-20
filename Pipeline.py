import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.ndimage import label, rotate

try:
    from skimage.filters import threshold_otsu  # type: ignore[import-not-found]
except ImportError:
    threshold_otsu = None

def load_pressure_data(filename="subject8_stiff-legged_trial10_pressure.csv"):
    """Load a pressure CSV and flip it for display/analysis orientation."""
    script_dir = Path(__file__).resolve().parent
    candidate_dirs = [
        script_dir / "Pressure_Data",
        script_dir.parent / "Pressure_Data",
    ]
    data_dir = next((path for path in candidate_dirs if path.exists()), candidate_dirs[0])
    file_path = data_dir / filename

    if not file_path.exists():
        fallback_files = sorted(data_dir.glob("*stiff-legged*_pressure.csv"))
        if fallback_files:
            file_path = fallback_files[0]
        else:
            raise FileNotFoundError(
                f"Could not find {file_path.name} in {data_dir.resolve()}"
            )

    data = pd.read_csv(file_path, header=None)
    pressure = data.iloc[::-1].to_numpy()
    return pressure, file_path


def parse_trial_metadata(csv_path):
    """Extract subject, condition, and trial from a pressure CSV filename."""
    parts = csv_path.stem.split("_")
    subject = parts[0] if len(parts) > 0 else "unknown"
    condition = parts[1] if len(parts) > 1 else "unknown"
    trial = parts[2] if len(parts) > 2 else "unknown"
    return subject, condition, trial


def create_binary_map(pressure, threshold_ratio=0.05):
    """Create a binary contact map from the pressure image."""
    positive_pressure = pressure[pressure > 0]

    if positive_pressure.size == 0:
        return np.zeros_like(pressure, dtype=int), 0.0

    if threshold_otsu is not None:
        threshold = float(threshold_otsu(pressure))
    else:
        threshold = float(pressure.max() * threshold_ratio)

    return (pressure > threshold).astype(int), threshold


def find_foot_regions(binary, max_regions=2):
    """Find the largest connected components corresponding to feet."""
    labeled, num_features = label(binary)
    region_sizes = []

    for i in range(1, num_features + 1):
        region = labeled == i
        size = np.sum(region)
        region_sizes.append((size, region))

    region_sizes.sort(reverse=True, key=lambda x: x[0])
    regions = [region for _, region in region_sizes[:max_regions]]

    return labeled, num_features, regions


def crop_pressure_region(region, pressure_map):
    """Crop a labeled foot region and keep original pressure values inside it."""
    rows = np.any(region, axis=1)
    cols = np.any(region, axis=0)
    masked_pressure = np.where(region, pressure_map, 0)
    return masked_pressure[rows][:, cols]


def extract_feet(pressure, regions):
    """Extract cropped pressure images for each detected foot."""
    return [crop_pressure_region(region, pressure) for region in regions]


def crop_nonzero_region(image):
    """Crop away zero-only borders from a pressure image."""
    mask = image > 0
    if not np.any(mask):
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return image[rows][:, cols]


def get_region_centroid_x(region):
    """Return the centroid column index for a full-size foot region mask."""
    cols = np.where(np.any(region, axis=0))[0]
    if cols.size == 0:
        return None

    return float(cols.mean())


def get_foot_labels(regions, full_width):
    """Assign foot labels using relative centroid order in the original image."""
    if not regions:
        return []

    centroids = [get_region_centroid_x(region) for region in regions]

    if len(regions) == 1:
        centroid = centroids[0]
        if centroid is None:
            return ["unknown"]
        if centroid < full_width / 2:
            return ["left"]
        return ["right"]

    labels = ["unknown"] * len(regions)
    sortable = [
        (index, centroid)
        for index, centroid in enumerate(centroids)
        if centroid is not None
    ]

    sortable.sort(key=lambda item: item[1])

    if sortable:
        labels[sortable[0][0]] = "left"
    if len(sortable) > 1:
        labels[sortable[1][0]] = "right"
    if len(sortable) > 2:
        for index, _ in sortable[2:]:
            labels[index] = "unknown"

    return labels


def append_missing_foot_row(results, subject, condition, trial, missing_foot, csv_path):
    """Append a placeholder row when one expected foot is missing."""
    results.append(
        {
            "subject": subject,
            "condition": condition,
            "trial": trial,
            "foot": missing_foot,
            "arch_index": np.nan,
            "peak_midfoot_pressure": np.nan,
            "heel_forefoot_ratio": np.nan,
            "symmetry_score": np.nan,
            "source_file": csv_path.name,
        }
    )


def compute_principal_axis_angle(binary_mask):
    """Compute the principal-axis angle in degrees from a binary foot mask."""
    coords = np.column_stack(np.nonzero(binary_mask))
    if len(coords) < 2:
        return 0.0

    centered = coords - coords.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]

    row_component, col_component = principal_vector
    angle_from_horizontal = np.degrees(np.arctan2(row_component, col_component))
    return float(angle_from_horizontal)


def orient_foot_top_to_forefoot(foot):
    """Flip vertically so the wider forefoot is at the top of the image."""
    if foot.size == 0:
        return foot

    mask = foot > 0
    h = mask.shape[0]
    section = max(1, int(h * 0.3))

    top_width = np.count_nonzero(np.any(mask[:section, :], axis=0))
    bottom_width = np.count_nonzero(np.any(mask[-section:, :], axis=0))

    if top_width < bottom_width:
        return np.flipud(foot)
    return foot


def align_foot_to_vertical(foot):
    """Rotate the foot so its principal axis is vertical, then crop tightly."""
    binary_mask = foot > 0
    if np.count_nonzero(binary_mask) < 2:
        return foot

    angle = compute_principal_axis_angle(binary_mask)
    rotation_angle = 90.0 - angle
    rotated_foot = rotate(
        foot,
        rotation_angle,
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
    )
    rotated_foot = np.clip(rotated_foot, 0, None)

    rotated_foot = crop_nonzero_region(rotated_foot)
    rotated_foot = orient_foot_top_to_forefoot(rotated_foot)
    return crop_nonzero_region(rotated_foot)

# (0-15) toe , (15-40) metatrancial , (40-65) midfoot ,(65-100) hill

def split_foot_zones(foot):
    """Split an aligned foot into toe, metatarsal, midfoot, and heel zones."""
    h = foot.shape[0]
    toe_end = int(h * 0.15)
    metatarsal_end = int(h * 0.40)
    midfoot_end = int(h * 0.65)

    toe = foot[:toe_end, :]
    metatarsal = foot[toe_end:metatarsal_end, :]
    midfoot = foot[metatarsal_end:midfoot_end, :]
    heel = foot[midfoot_end:, :]

    return toe, metatarsal, midfoot, heel

def compute_arch_index(foot):
    """Calculate pressure-weighted arch index using metatarsal+midfoot+heel."""
    _, metatarsal, midfoot, heel = split_foot_zones(foot)

    heel_area = np.sum(heel)
    mid_area = np.sum(midfoot)
    metatarsal_area = np.sum(metatarsal)

    total = heel_area + mid_area + metatarsal_area
    if total == 0:
        return 0

    return mid_area / total


def compute_peak_midfoot_pressure(foot):
    """Return the peak pressure in the midfoot zone."""
    _, _, midfoot, _ = split_foot_zones(foot)
    return float(np.max(midfoot)) if midfoot.size else 0.0


def compute_heel_forefoot_ratio(foot):
    """Return the ratio of heel pressure to forefoot pressure."""
    toe, metatarsal, _, heel = split_foot_zones(foot)
    heel_pressure = np.sum(heel)
    forefoot_pressure = np.sum(toe) + np.sum(metatarsal)

    if forefoot_pressure == 0:
        return 0.0

    return float(heel_pressure / forefoot_pressure)


def compute_symmetry_score(foot):
    """Measure left-right pressure symmetry within one cropped foot."""
    if foot.size == 0 or foot.shape[1] < 2:
        return 1.0

    midpoint = foot.shape[1] // 2
    left_half = foot[:, :midpoint]
    right_half = foot[:, midpoint:]

    if left_half.shape[1] == 0 or right_half.shape[1] == 0:
        return 1.0

    min_width = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_width]
    right_half = np.fliplr(right_half[:, -min_width:])

    total_pressure = np.sum(np.abs(left_half)) + np.sum(np.abs(right_half))
    if total_pressure == 0:
        return 1.0

    difference = np.sum(np.abs(left_half - right_half))
    return float(1.0 - (difference / total_pressure))


def analyze_foot(foot):
    """Compute zones and reference-model features for one foot."""
    aligned_foot = align_foot_to_vertical(foot)
    toe, metatarsal, midfoot, heel = split_foot_zones(aligned_foot)
    arch_index = compute_arch_index(aligned_foot)
    peak_midfoot_pressure = compute_peak_midfoot_pressure(aligned_foot)
    heel_forefoot_ratio = compute_heel_forefoot_ratio(aligned_foot)
    symmetry_score = compute_symmetry_score(aligned_foot)

    return {
        "aligned_foot": aligned_foot,
        "toe": toe,
        "metatarsal": metatarsal,
        "heel": heel,
        "midfoot": midfoot,
        "arch_index": arch_index,
        "peak_midfoot_pressure": peak_midfoot_pressure,
        "heel_forefoot_ratio": heel_forefoot_ratio,
        "symmetry_score": symmetry_score,
    }


def show_binary_map(binary):
    plt.imshow(binary, cmap="gray")
    plt.title("Binary Map")
    plt.axis("off")
    plt.show()


def show_foot_analysis(foot, analysis, foot_number):
    aligned_foot = analysis["aligned_foot"]

    plt.imshow(
        aligned_foot,
        cmap="jet",
        vmax=aligned_foot.max() if aligned_foot.size else None,
    )
    plt.title(f"Foot {foot_number} | AI: {analysis['arch_index']:.3f}")
    plt.axis("off")
    plt.colorbar(label="Pressure (arb. units)")
    plt.show()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    zones = [
        analysis["toe"],
        analysis["metatarsal"],
        analysis["midfoot"],
        analysis["heel"],
    ]
    zone_names = ["Toe", "Metatarsal", "Midfoot", "Heel"]

    for ax, zone_name, zone in zip(axes, zone_names, zones):
        zone_image = ax.imshow(
            zone,
            cmap="jet",
            vmax=aligned_foot.max() if aligned_foot.size else None,
        )
        ax.set_title(zone_name)
        ax.axis("off")

    fig.colorbar(zone_image, ax=axes, label="Pressure (arb. units)")
    plt.suptitle(f"Foot {foot_number} Divided Into Four Zones")
    plt.tight_layout()
    plt.show()


def process_trial(csv_path, show_plots=False):
    """Run the full analysis pipeline for one pressure CSV file."""
    data = pd.read_csv(csv_path, header=None)
    pressure = data.iloc[::-1].to_numpy()
    binary, threshold = create_binary_map(pressure)
    _, num_features, regions = find_foot_regions(binary)
    feet = extract_feet(pressure, regions)
    subject, condition, trial = parse_trial_metadata(csv_path)

    if show_plots:
        print(f"Loaded file: {csv_path.name}")
        print(f"Threshold used: {threshold:.3f}")
        print(f"Number of regions found: {num_features}")
        print(f"Valid foot regions: {len(regions)}")
        show_binary_map(binary)

    if len(regions) < 2:
        print(f"Warning: only one foot detected in {csv_path.name}")

    results = []
    foot_labels = get_foot_labels(regions, pressure.shape[1])

    for foot_index, foot in enumerate(feet):
        analysis = analyze_foot(foot)
        foot_label = foot_labels[foot_index] if foot_index < len(foot_labels) else "unknown"

        if show_plots:
            show_foot_analysis(foot, analysis, foot_index + 1)
            print(f"Foot {foot_index + 1} Arch Index: {analysis['arch_index']:.3f}")

        results.append(
            {
                "subject": subject,
                "condition": condition,
                "trial": trial,
                "foot": foot_label,
                "arch_index": analysis["arch_index"],
                "peak_midfoot_pressure": analysis["peak_midfoot_pressure"],
                "heel_forefoot_ratio": analysis["heel_forefoot_ratio"],
                "symmetry_score": analysis["symmetry_score"],
                "source_file": csv_path.name,
            }
        )

    detected_feet = {row["foot"] for row in results}
    if len(regions) == 1:
        if "left" in detected_feet and "right" not in detected_feet:
            append_missing_foot_row(results, subject, condition, trial, "right", csv_path)
        elif "right" in detected_feet and "left" not in detected_feet:
            append_missing_foot_row(results, subject, condition, trial, "left", csv_path)
        else:
            append_missing_foot_row(results, subject, condition, trial, "unknown", csv_path)

    return results


def process_all_trials(data_dir):
    """Compute arch index results for all pressure CSV files in the dataset."""
    results = []

    for csv_path in sorted(data_dir.glob("*_pressure.csv")):
        results.extend(process_trial(csv_path, show_plots=False))

    return pd.DataFrame(results)


def build_normal_reference_profile(features_df):
    """Build a normal-foot reference profile from a single-class dataset."""
    feature_columns = [
        "arch_index",
        "peak_midfoot_pressure",
        "heel_forefoot_ratio",
        "symmetry_score",
    ]
    valid_df = features_df.dropna(subset=feature_columns).copy()

    summary_rows = []
    for column in feature_columns:
        values = valid_df[column]
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=0))
        summary_rows.append(
            {
                "feature": column,
                "mean": mean_value,
                "std": std_value,
                "min": float(values.min()),
                "max": float(values.max()),
                "lower_2std": mean_value - 2 * std_value,
                "upper_2std": mean_value + 2 * std_value,
            }
        )

    return pd.DataFrame(summary_rows)


def fit_normal_learning_model(features_df):
    """Fit a simple normal-only reference model using z-scores."""
    feature_columns = [
        "arch_index",
        "peak_midfoot_pressure",
        "heel_forefoot_ratio",
        "symmetry_score",
    ]
    valid_df = features_df.dropna(subset=feature_columns).copy()

    model = {"feature_columns": feature_columns, "stats": {}}
    for column in feature_columns:
        values = valid_df[column]
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=0))
        if std_value == 0:
            std_value = 1e-8
        model["stats"][column] = {
            "mean": mean_value,
            "std": std_value,
        }

    return model


def apply_normal_learning_model(features_df, model):
    """Score each sample by how far it deviates from the learned normal profile."""
    scored_df = features_df.copy()
    zscore_columns = []

    for column in model["feature_columns"]:
        mean_value = model["stats"][column]["mean"]
        std_value = model["stats"][column]["std"]
        z_column = f"{column}_zscore"
        scored_df[z_column] = (scored_df[column] - mean_value) / std_value
        zscore_columns.append(z_column)

    scored_df["normality_score"] = np.sqrt(
        np.nansum(np.square(scored_df[zscore_columns]), axis=1)
    )
    scored_df["normality_status"] = np.where(
        scored_df["normality_score"] <= 2.0,
        "within_normal_range",
        np.where(
            scored_df["normality_score"] <= 3.0,
            "borderline",
            "outside_normal_range",
        ),
    )

    missing_mask = scored_df[model["feature_columns"]].isna().any(axis=1)
    scored_df.loc[missing_mask, "normality_score"] = np.nan
    scored_df.loc[missing_mask, "normality_status"] = "missing_features"

    return scored_df


def save_dataframe_safely(dataframe, output_path):
    """Save a DataFrame, falling back to a timestamped filename if locked."""
    try:
        dataframe.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        dataframe.to_csv(fallback_path, index=False)
        print(
            f"Warning: could not write to {output_path.name} because the file is in use. "
            f"Saved to {fallback_path.name} instead."
        )
        return fallback_path


def plot_arch_index_histogram(features_df, labels_df=None):
    """Plot Arch Index distributions and compare them with literature cutoffs."""
    plt.figure(figsize=(9, 5))

    valid_ai = features_df["arch_index"].dropna()
    plt.hist(valid_ai, bins=30, alpha=0.75, label="Normal reference samples")

    plt.axvline(0.21, color="orange", linestyle="--", linewidth=2, label="Literature normal cutoff (0.21)")
    plt.axvline(0.26, color="red", linestyle="--", linewidth=2, label="Literature flat-foot cutoff (0.26)")
    plt.xlabel("Arch Index")
    plt.ylabel("Count")
    plt.title("Arch Index Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    pressure, file_path = load_pressure_data()
    data_dir = file_path.parent

    single_trial_results = process_trial(file_path, show_plots=True)
    print("\nSingle-trial summary:")
    print(pd.DataFrame(single_trial_results))

    all_results_df = process_all_trials(data_dir)
    normal_model = fit_normal_learning_model(all_results_df)
    all_results_df = apply_normal_learning_model(all_results_df, normal_model)
    all_results_df["assumed_class"] = "Normal"
    reference_profile_df = build_normal_reference_profile(all_results_df)
    model_summary_df = pd.DataFrame(
        [
            {
                "feature": feature_name,
                "mean": stats["mean"],
                "std": stats["std"],
            }
            for feature_name, stats in normal_model["stats"].items()
        ]
    )
    profile_path = Path(__file__).resolve().parent / "normal_reference_profile.csv"
    model_path = Path(__file__).resolve().parent / "normal_learning_model.csv"
    results_path = Path(__file__).resolve().parent / "normal_foot_features.csv"
    profile_path = save_dataframe_safely(reference_profile_df, profile_path)
    model_path = save_dataframe_safely(model_summary_df, model_path)
    results_path = save_dataframe_safely(all_results_df, results_path)

    print("\nBatch results preview:")
    print(all_results_df.head())
    print(f"\nTotal analyzed feet: {len(all_results_df)}")
    print("\nAll trials are being treated as normal-foot reference samples.")
    print(f"Saved feature table: {results_path.name}")
    print(f"Saved reference profile: {profile_path.name}")
    print(f"Saved normal learning model: {model_path.name}")
    print("\nNormal reference profile:")
    print(reference_profile_df)
    print("\nNormal-only learning model summary:")
    print(model_summary_df)

    plot_arch_index_histogram(all_results_df)
    print(
        "\nNormal-only learning model enabled. Each sample now gets a "
        "deviation-based normality score instead of a supervised flat-foot label."
    )


if __name__ == "__main__":
    main()
