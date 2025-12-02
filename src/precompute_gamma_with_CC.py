import yaml
import numpy as np
import os
from tqdm import tqdm
from skimage import measure, morphology
from scipy.ndimage import label
import warnings
import tempfile

# Suppress specific warning from scikit-image
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)

# --- Config ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
PIXEL_SIZE_KM = config["PIXEL_SIZE_KM"]


# --- Core Geometric Calculation Functions ---
def compute_A_P_CC_single_threshold_numpy(
    prec_2d_np,
    threshold,
    pixel_size_km=1.0,
):
    """Computes Area, Perimeter using skimage, and Connected Components using TDA."""
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold

    # --- Area ---
    area_km2 = mask.sum() * (pixel_size_km**2)

    # --- Perimeter ---
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = sum(
        np.linalg.norm(np.diff(contour, axis=0), axis=1).sum() for contour in contours
    )
    perimeter_km = perimeter_pixels * pixel_size_km

    # --- Connected Components ---
    structure = morphology.disk(1)  # Define connectivity (8-connectivity for disk(1))
    _, num_features = label(
        mask, structure=structure
    )  # Label connected regions in the mask

    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km):
    """Computes the 3xN_quantiles Gamma matrix for a single precipitation image."""
    gamma_matrix = np.zeros((3, len(thresholds)), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix


# --- Main Processing Function (Generic) ---
def process_single_file_topology(data_split, input_filename, output_filename):
    """
    Generic function to compute topology for a specific input file within a data split.
    """
    input_dir = os.path.join(PREPROCESSED_DATA_DIR, data_split)
    input_path = os.path.join(input_dir, input_filename)
    output_path = os.path.join(input_dir, output_filename)

    if not os.path.exists(input_path):
        print(f"Warning: Input file not found at {input_path}. Skipping.")
        return

    print(f"Processing {input_filename} for split: {data_split}...")

    # Create a temporary file path for the memmap
    temp_fd, temp_output_path = tempfile.mkstemp(suffix=".npy", dir=input_dir)
    os.close(temp_fd)

    print(f"Using temporary memmap file at: {temp_output_path}")

    print(f"Loading data from {input_path}...")
    # Load input data (using mmap to save RAM)
    precip_data = np.load(input_path, mmap_mode="r")["data"]
    num_samples = precip_data.shape[0]

    # 1. Define the shape of the final array
    output_shape = (num_samples, 3, len(QUANTILE_LEVELS))
    output_dtype = np.float32

    # 2. Create a memory-mapped array on disk in 'write' mode
    gamma_targets_array_mmap = np.memmap(
        temp_output_path, dtype=output_dtype, mode="w+", shape=output_shape
    )

    print(f"Computing Gamma matrices for {input_filename}...")
    for i in tqdm(range(num_samples), desc=f"Calculating Topology ({input_filename})"):
        prec_field = precip_data[i]
        gamma_matrix = compute_gamma_matrix_for_image(
            prec_field, QUANTILE_LEVELS, PIXEL_SIZE_KM
        )
        gamma_targets_array_mmap[i] = gamma_matrix

    # 3. Flush changes to disk
    print("Flushing computed data to disk...")
    gamma_targets_array_mmap.flush()
    del gamma_targets_array_mmap  # Closes the file mapping

    # --- Step 2: Compress the file ---
    print(f"Reloading from memmap and compressing to {output_path}...")
    final_data_mmap = np.memmap(
        temp_output_path, dtype=output_dtype, mode="r", shape=output_shape
    )

    np.savez_compressed(output_path, data=final_data_mmap)

    # 5. Clean up the temporary file
    print(f"Cleaning up temporary file: {temp_output_path}")
    os.remove(temp_output_path)
    print(f"Saved {output_path}. Shape: {output_shape}")


def main():
    data_splits = ["train", "validation", "test"]

    # Define the pairs of (Input Filename, Output Filename)
    processing_tasks = [
        # ("physical_precip.npz", "gamma_targets.npz"),
        (
            "interpolated_physical_precip.npz",
            "gamma_targets_persistence_interpolated.npz",
        ),
    ]

    for split in data_splits:
        print(f"\n{'='*40}")
        print(f"--- DATA SPLIT: {split.upper()} ---")
        print(f"{'='*40}")

        for input_name, output_name in processing_tasks:
            process_single_file_topology(split, input_name, output_name)

    print("\nAll pre-computation is complete.")


if __name__ == "__main__":
    main()
