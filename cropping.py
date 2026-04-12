import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label, rotate

try:
    from skimage.filters import threshold_otsu  # type: ignore[import-not-found]
except ImportError:
    threshold_otsu = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
except ImportError:
    RandomForestClassifier = None
    classification_report = None
    train_test_split = None


def load_pressure_data(filename="subject8_stiff-legged_trial10_pressure.csv"):
    """Load a pressure CSV and flip it for display/analysis orientation."""
    data_dir = Path(__file__).resolve().parent / "Pressure_Data"
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
            "flatness_class": "Missing",
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


def split_foot_zones(foot, heel_pct=0.3, mid_pct=0.4):
    """Divide an aligned foot into forefoot, midfoot, and heel by height."""
    h = foot.shape[0]
    forefoot_end = int(h * (1.0 - heel_pct - mid_pct))
    mid_end = forefoot_end + int(h * mid_pct)

    forefoot = foot[:forefoot_end, :]
    midfoot = foot[forefoot_end:mid_end, :]
    heel = foot[mid_end:, :]

    return heel, midfoot, forefoot


def compute_arch_index(foot):
    """Calculate pressure-weighted arch index from the three foot zones."""
    heel, midfoot, forefoot = split_foot_zones(foot)

    heel_area = np.sum(heel)
    mid_area = np.sum(midfoot)
    fore_area = np.sum(forefoot)

    total = heel_area + mid_area + fore_area
    if total == 0:
        return 0

    return mid_area / total


def compute_peak_midfoot_pressure(foot):
    """Return the peak pressure in the midfoot zone."""
    _, midfoot, _ = split_foot_zones(foot)
    return float(np.max(midfoot)) if midfoot.size else 0.0


def compute_heel_forefoot_ratio(foot):
    """Return the ratio of heel pressure to forefoot pressure."""
    heel, _, forefoot = split_foot_zones(foot)
    heel_pressure = np.sum(heel)
    forefoot_pressure = np.sum(forefoot)

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


def classify_flatness(ai):
    """Classify foot flatness from arch index thresholds."""
    if ai < 0.21:
        return "Normal"
    if ai < 0.26:
        return "Mild flat foot"
    return "Severe flat foot"


def analyze_foot(foot):
    """Compute zones, arch index, and flat-foot classification for one foot."""
    aligned_foot = align_foot_to_vertical(foot)
    heel, midfoot, forefoot = split_foot_zones(aligned_foot)
    arch_index = compute_arch_index(aligned_foot)
    peak_midfoot_pressure = compute_peak_midfoot_pressure(aligned_foot)
    heel_forefoot_ratio = compute_heel_forefoot_ratio(aligned_foot)
    symmetry_score = compute_symmetry_score(aligned_foot)
    classification = classify_flatness(arch_index)

    return {
        "aligned_foot": aligned_foot,
        "heel": heel,
        "midfoot": midfoot,
        "forefoot": forefoot,
        "arch_index": arch_index,
        "peak_midfoot_pressure": peak_midfoot_pressure,
        "heel_forefoot_ratio": heel_forefoot_ratio,
        "symmetry_score": symmetry_score,
        "classification": classification,
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
    plt.title(
        f"Foot {foot_number} | AI: {analysis['arch_index']:.3f} | "
        f"{analysis['classification']}"
    )
    plt.axis("off")
    plt.colorbar(label="Pressure (arb. units)")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    zones = [analysis["heel"], analysis["midfoot"], analysis["forefoot"]]
    zone_names = ["Heel", "Midfoot", "Forefoot"]

    for ax, zone_name, zone in zip(axes, zone_names, zones):
        zone_image = ax.imshow(
            zone,
            cmap="jet",
            vmax=aligned_foot.max() if aligned_foot.size else None,
        )
        ax.set_title(zone_name)
        ax.axis("off")

    fig.colorbar(zone_image, ax=axes, label="Pressure (arb. units)")
    plt.suptitle(f"Foot {foot_number} Divided Into Three Zones")
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
            print(
                f"Foot {foot_index + 1} Arch Index: {analysis['arch_index']:.3f} | "
                f"Classification: {analysis['classification']}"
            )

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
                "flatness_class": analysis["classification"],
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


def find_best_arch_index_cutoff(merged_df):
    """Find the cutoff that best separates normal vs flat-foot labels."""
    valid_df = merged_df.dropna(subset=["arch_index", "grade"]).copy()
    valid_df = valid_df[valid_df["grade"].isin([0, 1, 2])]

    if valid_df.empty:
        return None, None

    binary_target = (valid_df["grade"] > 0).astype(int)
    candidate_thresholds = np.sort(valid_df["arch_index"].unique())

    if candidate_thresholds.size == 0:
        return None, None

    best_threshold = None
    best_accuracy = -1.0

    for threshold in candidate_thresholds:
        predictions = (valid_df["arch_index"] >= threshold).astype(int)
        accuracy = float(np.mean(predictions == binary_target))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def plot_arch_index_histogram(features_df, labels_df=None):
    """Plot Arch Index distributions and compare them with literature cutoffs."""
    plt.figure(figsize=(9, 5))

    if labels_df is not None:
        merged_df = features_df.merge(
            labels_df,
            on=["subject", "condition", "trial", "foot"],
            how="inner",
        )
        merged_df = merged_df.dropna(subset=["arch_index"])

        normal_df = merged_df[merged_df["grade"] == 0]
        flat_df = merged_df[merged_df["grade"].isin([1, 2])]

        if not normal_df.empty:
            plt.hist(
                normal_df["arch_index"],
                bins=20,
                alpha=0.6,
                label="Grade 0 (Normal)",
            )
        if not flat_df.empty:
            plt.hist(
                flat_df["arch_index"],
                bins=20,
                alpha=0.6,
                label="Grade 1-2 (Flat foot)",
            )

        best_cutoff, best_accuracy = find_best_arch_index_cutoff(merged_df)
        if best_cutoff is not None:
            plt.axvline(
                best_cutoff,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Best cutoff: {best_cutoff:.3f} (acc={best_accuracy:.2f})",
            )
    else:
        valid_ai = features_df["arch_index"].dropna()
        plt.hist(valid_ai, bins=30, alpha=0.75, label="All detected feet")

    plt.axvline(0.21, color="orange", linestyle="--", linewidth=2, label="Literature normal cutoff (0.21)")
    plt.axvline(0.26, color="red", linestyle="--", linewidth=2, label="Literature flat-foot cutoff (0.26)")
    plt.xlabel("Arch Index")
    plt.ylabel("Count")
    plt.title("Arch Index Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_flat_foot_classifier(features_df, labels_path):
    """Train a grade classifier when ground-truth labels are available."""
    if RandomForestClassifier is None:
        raise ImportError(
            "scikit-learn is required to train the classifier. "
            "Install it before running training."
        )

    labels_df = pd.read_csv(labels_path)
    merged_df = features_df.merge(
        labels_df,
        on=["subject", "condition", "trial", "foot"],
        how="inner",
    )
    merged_df = merged_df.dropna(
        subset=[
            "arch_index",
            "peak_midfoot_pressure",
            "heel_forefoot_ratio",
            "symmetry_score",
            "grade",
        ]
    )

    if merged_df.empty:
        raise ValueError(
            "No matching rows were found between extracted features and labels."
        )

    feature_columns = [
        "arch_index",
        "peak_midfoot_pressure",
        "heel_forefoot_ratio",
        "symmetry_score",
    ]
    X = merged_df[feature_columns]
    y = merged_df["grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)

    return model, report, merged_df


def main():
    pressure, file_path = load_pressure_data()
    data_dir = file_path.parent

    single_trial_results = process_trial(file_path, show_plots=True)
    print("\nSingle-trial summary:")
    print(pd.DataFrame(single_trial_results))

    all_results_df = process_all_trials(data_dir)
    print("\nBatch results preview:")
    print(all_results_df.head())
    print(f"\nTotal analyzed feet: {len(all_results_df)}")

    labels_path = Path(__file__).resolve().parent / "flat_foot_labels.csv"
    if labels_path.exists():
        try:
            labels_df = pd.read_csv(labels_path)
            plot_arch_index_histogram(all_results_df, labels_df)

            _, report, merged_df = train_flat_foot_classifier(
                all_results_df,
                labels_path,
            )
            print("\nTraining rows used:")
            print(len(merged_df))
            print("\nClassifier report:")
            print(report)

            best_cutoff, best_accuracy = find_best_arch_index_cutoff(merged_df)
            if best_cutoff is not None:
                print(
                    f"\nBest Arch Index cutoff from labeled data: "
                    f"{best_cutoff:.3f} (accuracy={best_accuracy:.2f})"
                )
        except Exception as exc:
            print(f"\nClassifier training skipped: {exc}")
    else:
        plot_arch_index_histogram(all_results_df)
        print(
            "\nClassifier training skipped: add "
            "'flat_foot_labels.csv' with columns "
            "subject, condition, trial, foot, grade"
        )


if __name__ == "__main__":
    main()
