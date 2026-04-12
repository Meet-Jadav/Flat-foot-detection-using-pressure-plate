import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label


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


def create_binary_map(pressure, threshold_ratio=0.05):
    """Create a binary contact map from the pressure image."""
    threshold = pressure.max() * threshold_ratio
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


def split_foot_zones(foot, heel_pct=0.3, mid_pct=0.4):
    """Divide the foot into heel, midfoot, and forefoot by percentage."""
    h = foot.shape[0]
    heel_end = int(h * heel_pct)
    mid_end = heel_end + int(h * mid_pct)

    heel = foot[:heel_end, :]
    midfoot = foot[heel_end:mid_end, :]
    forefoot = foot[mid_end:, :]

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


def classify_flatness(ai):
    """Classify foot flatness from arch index thresholds."""
    if ai < 0.21:
        return "Normal"
    if ai < 0.26:
        return "Mild flat foot"
    return "Severe flat foot"


def analyze_foot(foot):
    """Compute zones, arch index, and flat-foot classification for one foot."""
    heel, midfoot, forefoot = split_foot_zones(foot)
    arch_index = compute_arch_index(foot)
    classification = classify_flatness(arch_index)

    return {
        "heel": heel,
        "midfoot": midfoot,
        "forefoot": forefoot,
        "arch_index": arch_index,
        "classification": classification,
    }


def show_binary_map(binary):
    plt.imshow(binary, cmap="gray")
    plt.title("Binary Map")
    plt.axis("off")
    plt.show()


def show_foot_analysis(foot, analysis, foot_number):
    plt.imshow(foot, cmap="hot")
    plt.title(
        f"Foot {foot_number} | AI: {analysis['arch_index']:.3f} | "
        f"{analysis['classification']}"
    )
    plt.axis("off")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    zones = [analysis["heel"], analysis["midfoot"], analysis["forefoot"]]
    zone_names = ["Heel", "Midfoot", "Forefoot"]

    for ax, zone_name, zone in zip(axes, zone_names, zones):
        ax.imshow(zone, cmap="hot")
        ax.set_title(zone_name)
        ax.axis("off")

    plt.suptitle(f"Foot {foot_number} Divided Into Three Zones")
    plt.tight_layout()
    plt.show()


def main():
    pressure, file_path = load_pressure_data()
    binary, threshold = create_binary_map(pressure)
    _, num_features, regions = find_foot_regions(binary)
    feet = extract_feet(pressure, regions)

    print(f"Loaded file: {file_path.name}")
    print(f"Threshold used: {threshold:.3f}")
    print(f"Number of regions found: {num_features}")
    print(f"Valid foot regions: {len(regions)}")

    show_binary_map(binary)

    for i, foot in enumerate(feet, start=1):
        analysis = analyze_foot(foot)
        show_foot_analysis(foot, analysis, i)
        print(
            f"Foot {i} Arch Index: {analysis['arch_index']:.3f} | "
            f"Classification: {analysis['classification']}"
        )


if __name__ == "__main__":
    main()
