import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import zoom
import random  # For splitting data

# --- Parameters (Consolidated from both scripts) ---
# Patch Extraction Parameters
patch_size = 128
stride = 128
min_valid_fraction_precip = 1.0
min_valid_fraction_dem = 1.0
n_workers_patch_extraction = 8  # Number of parallel processes for Zarr folders

# Paths for Patch Extraction
base_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/OPERA"
output_base_dir = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/patches"
)
dem_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/DEM"
reprojected_dem_filename = "reproj_OPERA_1km_europe_dem.nc"
reprojected_dem_path = os.path.join(dem_data_dir, reprojected_dem_filename)

# Only DEM patches are saved here directly from Phase 1
DEM_PATCH_OUTPUT_DIR = os.path.join(output_base_dir, "dem")

# Data Preprocessing Parameters
downscaling_factor = 6

# Directory where the preprocessed data (NPZ files) will be stored.
PREPROCESSED_DATA_DIR = os.path.join(output_base_dir, "preprocessed_precip")

# Create necessary directories
os.makedirs(DEM_PATCH_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

# --- (Patch Extraction) ---


def extract_valid_patches_with_coords(
    array_2d, patch_size, stride, min_valid_pixels_count
):
    """
    Extracts valid patches (as numpy arrays) and their top-left (y, x) coordinates from a 2D array.
    A patch is considered valid if it contains at least 'min_valid_pixels_count' non-NaN pixels.
    """
    y_dim, x_dim = array_2d.shape
    valid_patches_info = []

    for i in range(0, y_dim - patch_size + 1, stride):
        for j in range(0, x_dim - patch_size + 1, stride):
            patch = array_2d[i : i + patch_size, j : j + patch_size]
            valid_pixels = np.count_nonzero(~np.isnan(patch))

            if valid_pixels >= min_valid_pixels_count:
                valid_patches_info.append((patch, (i, j)))
    return valid_patches_info


def process_single_zarr_folder(zarr_folder_name):
    """
    Processes a single Zarr folder containing precipitation data.
    It extracts valid patches and their metadata, but does NOT save them as .npy files.
    Instead, it returns the patch data directly in memory.

    Returns:
        tuple: (list of unique (y, x) coordinates,
                list of (patch_data_array, y_start, x_start, time_str) tuples)
    """
    folder_path = os.path.join(base_path, zarr_folder_name)
    all_valid_coords_in_folder = set()
    all_raw_precip_patch_info_in_folder = []  # Store (patch_data, y, x, time_str)
    precip_var_name = "TOT_PREC"

    try:
        ds = xr.open_zarr(folder_path, chunks={})
        precip_data_array = ds[precip_var_name]

        print(
            f"Processing precipitation folder: {zarr_folder_name}, shape: {precip_data_array.shape}"
        )

        min_valid_pixels = int(min_valid_fraction_precip * patch_size * patch_size)

        for t in tqdm(
            range(precip_data_array.sizes["time"]),
            desc=f"Extracting patches from {zarr_folder_name}",
        ):
            slice_2d = precip_data_array.isel(time=t).values.astype(np.float32)

            try:
                time_str = (
                    precip_data_array.time.isel(time=t)
                    .dt.strftime("%Y%m%d%H%M%S")
                    .item()
                )
            except AttributeError:
                time_str = f"time{t:04d}"

            patches_info = extract_valid_patches_with_coords(
                slice_2d, patch_size, stride, min_valid_pixels
            )

            for patch_data, (y_start, x_start) in patches_info:
                # Add the (y, x) coordinates to our set of unique locations for DEM
                all_valid_coords_in_folder.add((y_start, x_start))
                # Store the raw patch data and its metadata for later preprocessing
                all_raw_precip_patch_info_in_folder.append(
                    (patch_data, y_start, x_start, time_str, zarr_folder_name)
                )

    except Exception as e:
        print(f"Error processing {zarr_folder_name}: {e}")

    return list(all_valid_coords_in_folder), all_raw_precip_patch_info_in_folder


def process_dem_data(
    dem_array_2d, all_unique_precip_coords, patch_size, min_valid_pixels_dem
):
    """
    Extracts and saves DEM patches only for the coordinates where valid precipitation
    patches were found. This ensures spatial alignment.
    Returns:
        list: Paths to saved .npy DEM patches. (These are saved as they are fixed per coordinate)
    """
    print(
        f"\nExtracting and saving DEM patches for {len(all_unique_precip_coords)} unique locations..."
    )
    saved_dem_patch_paths = []
    dem_y_dim, dem_x_dim = dem_array_2d.shape

    for y_start, x_start in tqdm(all_unique_precip_coords, desc="Saving DEM patches"):
        if (y_start + patch_size <= dem_y_dim) and (x_start + patch_size <= dem_x_dim):
            dem_patch = dem_array_2d[
                y_start : y_start + patch_size, x_start : x_start + patch_size
            ].astype(np.float32)

            valid_pixels_dem = np.count_nonzero(~np.isnan(dem_patch))
            if valid_pixels_dem >= min_valid_pixels_dem:
                patch_filename = f"dem_patch_y{y_start:04d}_x{x_start:04d}.npy"
                full_patch_path = os.path.join(DEM_PATCH_OUTPUT_DIR, patch_filename)
                np.save(full_patch_path, dem_patch)
                saved_dem_patch_paths.append(full_patch_path)
    return saved_dem_patch_paths


# --- (Preprocessing) ---


def _compute_stats_for_single_patch_data(patch_data):
    """
    Helper function to compute sums, sums_sq, and count for a single patch data array.
    Designed for parallel execution.
    """
    log_data = np.log1p(patch_data)
    return log_data.sum(), (log_data**2).sum(), log_data.size


def compute_global_stats_parallel(list_of_patch_data):
    """
    Computes global mean and standard deviation of log-transformed data in parallel
    directly from a list of patch data arrays.
    """
    sums_total = 0
    sums_sq_total = 0
    count_total = 0

    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Using {num_cpus} CPU workers for computing global statistics.")

    # Use a generator expression to avoid creating an intermediate list of arguments
    # (patch_data for _compute_stats_for_single_patch_data) if it's too large
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(
            tqdm(
                executor.map(_compute_stats_for_single_patch_data, list_of_patch_data),
                total=len(list_of_patch_data),
                desc="Computing global statistics in parallel... ",
            )
        )

    for sums, sums_sq, count in results:
        sums_total += sums
        sums_sq_total += sums_sq
        count_total += count

    mean = sums_total / count_total
    # Prevent division by zero if count_total is 0 or sums_sq_total / count_total - mean**2
    # is negative due to float precision
    variance = sums_sq_total / count_total - mean**2
    std = np.sqrt(max(0, variance))  # Ensure non-negative argument to sqrt
    return mean, std


def normalize_precip(arr, train_mean=None, train_std=None):
    """
    Normalizes precipitation array using log1p transformation and provided mean/std.
    If mean/std are not provided, computes them from the current array.
    """
    log_arr = np.log1p(arr)
    if train_mean is None or train_std is None:
        mean = np.mean(log_arr)
        std = np.std(log_arr)
    else:
        mean = train_mean
        std = train_std
    # Handle case where std might be zero (e.g., all values are same after log1p)
    return (log_arr - mean) / (std if std != 0 else 1.0)


def coarsen_array(arr, factor):
    """
    Coarsens an array by a given factor using simple averaging.
    """
    m, n = arr.shape
    m_new = m // factor
    n_new = n // factor
    # Ensure array dimensions are multiples of the factor
    arr = arr[: m_new * factor, : n_new * factor]
    return arr.reshape(m_new, factor, n_new, factor).mean(axis=(1, 3))


def interpolate_array(arr, factor):
    """
    Interpolates an array by a given factor using cubic spline interpolation (order=3).
    """
    return zoom(arr, zoom=factor, order=3)


def process_and_save_transformed_patch(args):
    """
    Wrapper function to process a single patch: normalization, coarsening, interpolation.
    Saves all outputs in a single .npz file, constructing the path based on metadata.
    """
    (
        raw_patch_data,
        y_start,
        x_start,
        time_str,
        zarr_folder_name,
        factor,
        train_mean,
        train_std,
        output_base_dir,
    ) = args

    norm = normalize_precip(raw_patch_data, train_mean, train_std)
    coarse = coarsen_array(norm, factor)
    interp = interpolate_array(norm, factor)

    # Construct the output directory structure:
    # PREPROCESSED_DATA_DIR / zarr_folder_name / time_str / patch_yXX_xYY.npz
    output_folder_path = os.path.join(output_base_dir, zarr_folder_name, time_str)
    os.makedirs(output_folder_path, exist_ok=True)

    output_npz_filename = f"patch_y{y_start:04d}_x{x_start:04d}.npz"
    output_npz_path = os.path.join(output_folder_path, output_npz_filename)

    np.savez(
        output_npz_path,
        original=raw_patch_data,  # Save original for reference if needed
        normalized=norm,
        coarsened=coarse,
        interpolated=interp,
    )
    return output_npz_path


# --- Main ---


def main_merged_workflow():
    print("--- Starting Merged Data Processing Workflow ---")

    # --- Phase 1: Patch Extraction (collecting data in memory) ---
    print(
        "\nPhase 1: Starting patch extraction for precipitation data and identifying valid patch locations..."
    )

    zarr_folders = sorted(
        [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    )

    all_precip_coords = (
        set()
    )  # Master set to collect all unique (y, x) coordinates for DEM
    # Master list to collect all raw precip patch data and their metadata
    all_raw_precip_patch_info = (
        []
    )  # (patch_data, y_start, x_start, time_str, zarr_folder_name)

    with ProcessPoolExecutor(max_workers=n_workers_patch_extraction) as executor:
        futures = {
            executor.submit(process_single_zarr_folder, folder): folder
            for folder in zarr_folders
        }
        for future in as_completed(futures):
            folder = futures[future]
            try:
                coords_from_folder, patch_info_from_folder = future.result()
                if coords_from_folder:
                    all_precip_coords.update(coords_from_folder)
                if patch_info_from_folder:
                    all_raw_precip_patch_info.extend(patch_info_from_folder)
            except Exception as e:
                print(f"Exception processing folder {folder}: {e}")

    # Process DEM Data (still requires saving .npy files as they are fixed per coordinate)
    print(
        "\nAll precipitation patches extracted to memory. Now loading and processing DEM data..."
    )
    try:
        dem_da = xr.open_dataarray(reprojected_dem_path)
        # Ensure the y-axis (latitude) is oriented consistently (e.g., north-up).
        if dem_da.y.values[0] > dem_da.y.values[-1]:
            dem_da = dem_da.isel(y=slice(None, None, -1))

        dem_array_2d = dem_da.values
        min_valid_pixels_dem = int(min_valid_fraction_dem * patch_size * patch_size)
        saved_dem_patch_paths = process_dem_data(
            dem_array_2d, all_precip_coords, patch_size, min_valid_pixels_dem
        )
        print(
            f"Saved {len(saved_dem_patch_paths)} DEM patches to {DEM_PATCH_OUTPUT_DIR}."
        )

    except FileNotFoundError:
        print(
            f"Error: DEM file not found at {reprojected_dem_path}. Cannot process DEM patches."
        )
    except Exception as e:
        print(f"Error loading or processing DEM: {e}")

    print("Phase 1 Complete: Patch extraction finished and DEMs saved.")

    # --- Phase 2: Data Preprocessing (Normalization, Coarsening, Interpolation) ---
    print(
        "\nPhase 2: Starting data preprocessing (normalization, coarsening, interpolation)..."
    )

    # How to define train/val/test splits:
    # Here, we randomly shuffle the collected patch info.
    # In a real-world scenario, you might want to split based on time ranges
    # for the `zarr_folder_name` to avoid data leakage (e.g., all of 2017 is train, 2018 is val, etc.)
    random.seed(42)  # For reproducibility of the split
    random.shuffle(all_raw_precip_patch_info)

    total_patches = len(all_raw_precip_patch_info)
    train_split = int(0.7 * total_patches)
    val_split = int(0.15 * total_patches)

    train_patch_info = all_raw_precip_patch_info[:train_split]
    val_patch_info = all_raw_precip_patch_info[train_split : train_split + val_split]
    test_patch_info = all_raw_precip_patch_info[train_split + val_split :]

    print(f"Total precip patches extracted: {total_patches}")
    print(
        f"Train: {len(train_patch_info)}, Val: {len(val_patch_info)}, Test: {len(test_patch_info)}"
    )

    # 2.1 Compute global statistics (only from training data)
    print("\nComputing global statistics (parallelized) from training patches...")
    # Extract only the actual patch data for statistic computation
    train_patch_data_for_stats = [info[0] for info in train_patch_info]
    train_mean, train_std = compute_global_stats_parallel(train_patch_data_for_stats)
    stats = {"train_mean": train_mean, "train_std": train_std}
    print(f"Global Mean (log1p): {train_mean}, Global Std (log1p): {train_std}")
    np.save(os.path.join(PREPROCESSED_DATA_DIR, "train_stat.npy"), stats)

    # 2.2 Process and save transformed patches for all sets
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"\nUsing {num_cpus} CPU workers for file transformations.")

    for file_list_name, current_patch_info_list in [
        ("train", train_patch_info),
        ("val", val_patch_info),
        ("test", test_patch_info),
    ]:
        print(
            f"\nProcessing {file_list_name} set (total {len(current_patch_info_list)} patches)..."
        )

        # Prepare arguments for each call to process_and_save_transformed_patch
        # (raw_patch_data, y_start, x_start, time_str, zarr_folder_name, factor, train_mean, train_std, output_base_dir)
        tasks = [
            (
                info[0],
                info[1],
                info[2],
                info[3],
                info[4],
                downscaling_factor,
                train_mean,
                train_std,
                PREPROCESSED_DATA_DIR,
            )
            for info in current_patch_info_list
        ]

        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            for idx, completed_path in enumerate(
                tqdm(
                    executor.map(process_and_save_transformed_patch, tasks),
                    total=len(tasks),
                    desc=f"Transforming and saving {file_list_name} patches",
                )
            ):
                if (idx + 1) % 5000 == 0:  # Print progress periodically
                    print(f"Processed {idx + 1} patches (last saved: {completed_path})")
        print(
            f"Finished processing all {len(current_patch_info_list)} files in {file_list_name} set."
        )

    print("\nPhase 2 Complete: Data preprocessing finished and NPZ files saved.")
    print(
        "\n--- All Done. Precipitation patches preprocessed directly, and DEM patches saved. ---"
    )


if __name__ == "__main__":
    main_merged_workflow()
