import yaml
import numpy as np
import os
from tqdm import tqdm
from skimage import measure, morphology
from scipy.ndimage import label
import warnings

# Suppress specific warning from scikit-image
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
pixel_size_km = config.get("PIXEL_SIZE_KM", 1.0)


# --- Core Geometric Calculation Functions (copied from original script) ---
def compute_A_P_CC_single_threshold_numpy(prec_2d_np, threshold, pixel_size_km=1.0):
    """Computes Area, Perimeter, and Connected Components for a single threshold."""
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold

    # Area
    area_km2 = mask.sum() * (pixel_size_km**2)

    # Perimeter
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = sum(
        np.linalg.norm(np.diff(contour, axis=0), axis=1).sum() for contour in contours
    )
    perimeter_km = perimeter_pixels * pixel_size_km

    # Connected Components
    structure = morphology.disk(1)
    _, num_features = label(mask, structure=structure)

    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km=1.0):
    """Computes the 3xN_quantiles Gamma matrix for a single precipitation image."""
    gamma_matrix = np.zeros((3, len(thresholds)), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix


# --- Main Processing Function ---
def process_and_save_gamma_targets(data_split):
    """
    Loads precipitation data for a split (e.g., 'train'),
    computes all gamma matrices, and saves them to a new file.
    """
    print(f"--- Processing data split: {data_split} ---")

    # Define paths
    input_dir = os.path.join(PREPROCESSED_DATA_DIR, data_split)
    precip_path = os.path.join(input_dir, "original_precip.npz")
    output_path = os.path.join(input_dir, "gamma_targets.npz")

    if not os.path.exists(precip_path):
        print(f"Precipitation file not found at {precip_path}. Skipping.")
        return

    print(f"Loading precipitation data from {precip_path}...")
    # Use mmap_mode for memory efficiency, though we still iterate through all
    precip_data = np.load(precip_path, mmap_mode="r")["data"]
    num_samples = precip_data.shape[0]

    all_gamma_matrices = []

    print("Computing Gamma matrices for all samples...")
    for i in tqdm(range(num_samples), desc=f"Calculating Gamma for {data_split}"):
        prec_field = precip_data[i]
        gamma_matrix = compute_gamma_matrix_for_image(
            prec_field, QUANTILE_LEVELS, pixel_size_km=1.0
        )
        all_gamma_matrices.append(gamma_matrix)

    # Convert list of matrices to a single NumPy array
    gamma_targets_array = np.array(all_gamma_matrices, dtype=np.float32)

    print(f"Saving computed Gamma matrices to {output_path}...")
    np.savez_compressed(output_path, data=gamma_targets_array)

    print(
        f"Finished processing for {data_split}. Shape of saved data: {gamma_targets_array.shape}"
    )


if __name__ == "__main__":
    data_splits = ["train", "validation", "test"]
    for split in data_splits:
        process_and_save_gamma_targets(split)
    print("\nAll pre-computation is complete.")
