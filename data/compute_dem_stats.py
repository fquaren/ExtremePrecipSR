import os
import glob
import numpy as np
import json
import multiprocessing
from tqdm import tqdm
import warnings

# Suppress runtime warnings from numpy (e.g., from loading)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURATION ---
DEM_PATCH_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/extremes/OPERA/patches/dem/"

OUTPUT_FILE = "dem_stats.json"
NUM_WORKERS = os.cpu_count()  # Use all available cores
# ---------------------


def compute_patch_stats(file_path):
    """
    Worker function to compute statistics for a single .npy file.
    Returns the sum, sum of squares, and number of pixels.
    """
    try:
        patch = np.load(file_path)

        # Ensure data is float64 for high precision during sum
        patch = patch.astype(np.float64)

        count = patch.size
        patch_sum = np.sum(patch)
        patch_sum_sq = np.sum(patch**2)

        return patch_sum, patch_sum_sq, count
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0.0, 0.0, 0


def main():
    """
    Main function to orchestrate parallel statistics computation.
    """
    # Ensure numpy/MKL/OpenBLAS are not multi-threading in the background
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    print(f"Starting DEM stats computation using {NUM_WORKERS} workers.")
    print(f"Scanning for patches in: {DEM_PATCH_DIR}")

    file_list = glob.glob(os.path.join(DEM_PATCH_DIR, "dem_patch_*.npy"))

    if not file_list:
        print(f"Error: No 'dem_patch_*.npy' files found in {DEM_PATCH_DIR}")
        return

    print(f"Found {len(file_list)} DEM patch files.")

    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # Use imap_unordered for efficiency, as file order doesn't matter
        results_iter = pool.imap_unordered(compute_patch_stats, file_list)

        pbar = tqdm(results_iter, total=len(file_list), desc="Processing Patches")
        for patch_sum, patch_sum_sq, count in pbar:
            if count > 0:
                total_sum += patch_sum
                total_sum_sq += patch_sum_sq
                total_count += count

    if total_count == 0:
        print("Error: No valid data was processed.")
        return

    # --- Final Statistics Calculation ---

    # 1. Global Mean
    global_mean = total_sum / total_count

    # 2. Global Variance
    # Var = E[X^2] - (E[X])^2
    global_variance = (total_sum_sq / total_count) - (global_mean**2)

    # 3. Global Standard Deviation
    global_std = np.sqrt(global_variance)

    print("\n--- Computation Complete ---")
    print(f"Global Mean (DEM):  {global_mean:.6f}")
    print(f"Global Std (DEM):   {global_std:.6f}")

    # --- Save Results to JSON ---
    stats = {
        "dem_mean": global_mean,
        "dem_std": global_std,
    }

    output_path = os.path.join(os.path.dirname(DEM_PATCH_DIR), OUTPUT_FILE)
    try:
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Successfully saved stats to: {output_path}")
    except IOError as e:
        print(f"Error saving stats file: {e}")


if __name__ == "__main__":
    # Set the start method to 'spawn' for safety with C-extension
    # libraries like numpy
    multiprocessing.set_start_method("spawn", force=True)
    main()
