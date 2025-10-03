import os
import numpy as np
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
from datetime import datetime, timedelta
import xarray as xr


# --- Global Parameters ---
# Patch Extraction Parameters
PATCH_SIZE = 128
STRIDE = 128
MIN_VALID_FRACTION_PRECIP = 1.0
MIN_VALID_FRACTION_DEM = 1.0
N_WORKERS_PATCH_EXTRACTION = 8

# Paths
BASE_DATA_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/"
RAW_OPERA_DATA_PATH = os.path.join(BASE_DATA_DIR, "raw_data", "OPERA")
PROCESSED_DATA_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "fquareng", "data", "OPERA")

PATCHES_BASE_OUTPUT_DIR = os.path.join(PROCESSED_DATA_OUTPUT_DIR, "patches")
DEM_DATA_DIR = os.path.join(PROCESSED_DATA_OUTPUT_DIR, "dem")
REPROJECTED_DEM_FILENAME = "reproj_OPERA_1km_europe_dem.nc"
REPROJECTED_DEM_PATH = os.path.join(DEM_DATA_DIR, REPROJECTED_DEM_FILENAME)

# Output directories for the *final* .npz files and for DEM patches
PRECIP_FINAL_BASE_OUTPUT_DIR = os.path.join(PATCHES_BASE_OUTPUT_DIR, "precip")
DEM_PATCH_OUTPUT_DIR = os.path.join(PATCHES_BASE_OUTPUT_DIR, "dem")

# Preprocessing Parameters
DOWNSCALING_FACTOR = 6
# Directory to save train_files.txt, val_files.txt, test_files.txt, and train_stat.npy
FINAL_FILE_LISTS_AND_STATS_DIR = os.path.join(
    BASE_DATA_DIR, "fquareng", "data", "OPERA"
)

# Decluttering Threshold
DECLUTTER_THRESHOLD = (
    150.0  # Define your decluttering threshold here (e.g., 150 mm/m^2)
)

# Quantiles to compute (still relevant for raw data analysis)
QUANTILE_LEVELS = [
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
]

# Create necessary directories
os.makedirs(
    PRECIP_FINAL_BASE_OUTPUT_DIR, exist_ok=True
)  # Only the base dir, YYYYMMDD will be added
os.makedirs(DEM_PATCH_OUTPUT_DIR, exist_ok=True)  # For the DEM .npy outputs
os.makedirs(FINAL_FILE_LISTS_AND_STATS_DIR, exist_ok=True)


# --- Functions from File 4: Patch Extraction ---
def extract_valid_patches_with_coords_and_data(
    array_2d, patch_size, stride, min_valid_pixels_count, original_coords=None
):
    """
    Extracts valid patches (as numpy arrays) and their top-left (y, x) coordinates from a 2D array.
    A patch is considered valid if it contains at least 'min_valid_pixels_count' non-NaN pixels.
    Returns patch data along with their global coordinates.
    """
    y_dim, x_dim = array_2d.shape
    valid_patches_info = []  # (patch_data, (start_y, start_x))

    for i in range(0, y_dim - patch_size + 1, stride):
        for j in range(0, x_dim - patch_size + 1, stride):
            patch = array_2d[i : i + patch_size, j : j + patch_size]
            valid_pixels = np.count_nonzero(~np.isnan(patch))

            if valid_pixels >= min_valid_pixels_count:
                # If original_coords are provided (for mapping back to source filename info),
                # use them to derive the global Y/X, otherwise use local patch coords
                global_y = i + (original_coords[0] if original_coords else 0)
                global_x = j + (original_coords[1] if original_coords else 0)
                valid_patches_info.append((patch, (global_y, global_x)))
    return valid_patches_info


def process_single_zarr_folder_in_memory(zarr_folder_name):
    """
    Processes a single Zarr folder containing precipitation data.
    It extracts valid patches for each time step and returns their data along with
    their derived file paths, but DOES NOT save them to disk at this stage.
    It also collects the unique (y, x) coordinates of all extracted patches for DEM processing.

    Args:
        zarr_folder_name (str): The name of the Zarr folder (e.g., "20170101T000000").

    Returns:
        tuple: A tuple containing:
            - list: A list of tuples, where each inner tuple is (file_path_suggestion_base, patch_data)
            - list: A list of unique (y, x) coordinates found in this Zarr folder.
    """
    folder_path = os.path.join(RAW_OPERA_DATA_PATH, zarr_folder_name)
    all_valid_coords_in_folder = set()
    all_precip_patch_data_for_later = (
        []
    )  # List of (suggested_path_base_without_ext, patch_data)
    precip_var_name = "TOT_PREC"

    try:
        ds = xr.open_zarr(folder_path, chunks={})
        precip_data_array = ds[precip_var_name]

        print(
            f"Processing precipitation folder: {zarr_folder_name}, shape: {precip_data_array.shape}"
        )

        # The base output path for the *final* .npz files for this specific Zarr folder's data
        folder_output_path = os.path.join(
            PRECIP_FINAL_BASE_OUTPUT_DIR, zarr_folder_name
        )

        # Create both the YYYYMMDD directory and the YYYYMMDDTHHMMSS directory
        os.makedirs(folder_output_path, exist_ok=True)

        min_valid_pixels = int(MIN_VALID_FRACTION_PRECIP * PATCH_SIZE * PATCH_SIZE)

        for t in tqdm(
            range(precip_data_array.sizes["time"]),
            desc=f"Extracting patches from {zarr_folder_name}",
        ):
            slice_2d = precip_data_array.isel(time=t).values.astype(np.float32)

            patches_info = extract_valid_patches_with_coords_and_data(
                slice_2d, PATCH_SIZE, STRIDE, min_valid_pixels
            )

            if patches_info:
                for patch_data, (y_start, x_start) in patches_info:
                    suggested_patch_filename_base = os.path.join(
                        folder_output_path, f"patch_y{y_start:04d}_x{x_start:04d}"
                    )
                    all_precip_patch_data_for_later.append(
                        (suggested_patch_filename_base, patch_data)
                    )
                    all_valid_coords_in_folder.add((y_start, x_start))

    except Exception as e:
        print(f"Error processing {zarr_folder_name}: {e}")

    return all_precip_patch_data_for_later, list(all_valid_coords_in_folder)


def process_dem_data(
    dem_array_2d, all_unique_precip_coords, patch_size, min_valid_pixels_dem
):
    """
    Extracts and saves DEM patches only for the coordinates where valid precipitation
    patches were found. This ensures spatial alignment.
    """
    print(
        f"\nExtracting and saving DEM patches for {len(all_unique_precip_coords)} unique locations..."
    )

    dem_y_dim, dem_x_dim = dem_array_2d.shape

    for y_start, x_start in tqdm(all_unique_precip_coords, desc="Saving DEM patches"):
        if (y_start + patch_size <= dem_y_dim) and (x_start + patch_size <= dem_x_dim):
            dem_patch = dem_array_2d[
                y_start : y_start + patch_size, x_start : x_start + patch_size
            ].astype(np.float32)

            valid_pixels_dem = np.count_nonzero(~np.isnan(dem_patch))
            if valid_pixels_dem >= min_valid_pixels_dem:
                patch_filename = f"dem_patch_y{y_start:04d}_x{x_start:04d}.npy"
                np.save(os.path.join(DEM_PATCH_OUTPUT_DIR, patch_filename), dem_patch)


# --- Functions from File 2: Date-Based Data Splitting Logic ---
def get_dated_directories(base_path):
    """
    Reads directories with YYYYMMDDTHHMMSS structure from a given base path.
    (Note: The actual split logic uses the YYYYMMDD part of these directory names)
    """
    # The glob pattern still targets YYYYMMDDTHHMMSS directories
    pattern = os.path.join(
        base_path,
        "[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]T[0-9][0-9]_[0-9][0-9]_[0-9][0-9]",
    )
    all_dirs = glob.glob(pattern)

    dated_dirs = []
    for d in all_dirs:
        if os.path.isdir(d):
            try:
                dir_name = os.path.basename(d)  # This will be YYYYMMDDTHHMMSS
                # We need to parse just the date part for date comparison
                date_str = dir_name[:8]  # Extract YYYYMMDD
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                dated_dirs.append(
                    (date_obj, dir_name)
                )  # Store YYYYMMDDTHHMMSS as dirname
            except ValueError:
                continue

    dated_dirs.sort(key=lambda x: x[0])
    return dated_dirs


def get_date_ranges(all_directories):
    """
    Splits dated directories (YYYYMMDDTHHMMSS) into train, validation, and test sets
    based on specific date ranges and a 3-weeks-on/1-week-off pattern,
    using only the YYYYMMDD part for the split logic.
    """
    list1_selected_weeks = (
        []
    )  # For training/validation (contains YYYYMMDDTHHMMSS folder names)
    list2_skipped_weeks = []  # Skipped weeks (contains YYYYMMDDTHHMMSS folder names)
    list3_aug_oct_2024 = []  # For testing (contains YYYYMMDDTHHMMSS folder names)

    # Define the date ranges (these are YYYYMMDD dates)
    start_date_aug_2023 = datetime(2023, 8, 1)
    end_date_aug_2024_excluded = datetime(2024, 8, 1)
    start_date_aug_2024 = datetime(2024, 8, 1)
    end_date_oct_2024 = datetime(2024, 10, 30)

    # Filter directories relevant to the first two lists (August 2023 to August 2024 excluded)
    dirs_for_list1_2 = sorted(
        [
            (date_obj, dirname)  # date_obj is YYYYMMDD, dirname is YYYYMMDDTHHMMSS
            for date_obj, dirname in all_directories
            if start_date_aug_2023 <= date_obj < end_date_aug_2024_excluded
        ]
    )

    # Process for List 1 and List 2 (August 2023 to August 2024 excluded)
    if dirs_for_list1_2:
        first_date = dirs_for_list1_2[0][0]  # This is a YYYYMMDD datetime object
        current_week_monday = first_date - timedelta(
            days=first_date.weekday()
        )  # Find the Monday of the first week

        week_count = 0
        dates_in_current_week = (
            []
        )  # Stores YYYYMMDDTHHMMSS folder names for the current week

        for date_obj, dirname_full in dirs_for_list1_2:
            # Check if we've crossed into a new week based on the YYYYMMDD part
            if date_obj >= current_week_monday + timedelta(weeks=1):
                if week_count % 4 < 3:  # 0, 1, 2 (first three weeks of the cycle)
                    list1_selected_weeks.extend(dates_in_current_week)
                else:  # 3 (fourth week of the cycle, skipped)
                    list2_skipped_weeks.extend(dates_in_current_week)

                # Reset for the new week, using the YYYYMMDD part of the current file's date
                current_week_monday = date_obj - timedelta(days=date_obj.weekday())
                week_count += 1
                dates_in_current_week = []

            dates_in_current_week.append(
                dirname_full
            )  # Add the full YYYYMMDDTHHMMSS name

        # Process any remaining dates from the last week
        if dates_in_current_week:
            if week_count % 4 < 3:
                list1_selected_weeks.extend(dates_in_current_week)
            else:
                list2_skipped_weeks.extend(dates_in_current_week)

    # Process for List 3 (August 1, 2024 to October 30, 2024)
    for date_obj, dirname_full in all_directories:
        if start_date_aug_2024 <= date_obj <= end_date_oct_2024:
            list3_aug_oct_2024.append(dirname_full)  # Add the full YYYYMMDDTHHMMSS name

    return list1_selected_weeks, list2_skipped_weeks, list3_aug_oct_2024


def get_train_val_test_dates_orchestrator(base_path):
    """Orchestrates getting dated directories and splitting them into ranges."""
    all_directories = get_dated_directories(base_path)
    return get_date_ranges(all_directories)


# --- Functions from File 3: File Path Generation and Saving ---
def save_to_txt(
    train_files_metadata, val_files_metadata, test_files_metadata, save_path
):
    """
    Saves lists of file paths (extracted from metadata) to separate .txt files.
    Each item in metadata is (filepath_base, patch_data). We only save the filepath_base.
    """
    train_files_path = os.path.join(save_path, "train_files.txt")
    with open(train_files_path, "w") as f:
        for filepath_base, _ in train_files_metadata:
            f.write(filepath_base + ".npz" + "\n")
    val_files_path = os.path.join(save_path, "val_files.txt")
    with open(val_files_path, "w") as f:
        for filepath_base, _ in val_files_metadata:
            f.write(filepath_base + ".npz" + "\n")
    test_files_path = os.path.join(save_path, "test_files.txt")
    with open(test_files_path, "w") as f:
        for filepath_base, _ in test_files_metadata:
            f.write(filepath_base + ".npz" + "\n")
    print(f"File lists saved at {save_path}.")


# --- Functions for Data Preprocessing ---


def declutter_precip(arr, threshold):
    """
    Sets pixel values in the array that are above the given threshold to zero.
    """
    arr_copy = arr.copy()
    arr_copy[arr_copy > threshold] = 0
    return arr_copy


def _get_flattened_data_for_stats(file_info):
    """
    Helper function to return the flattened original data for quantile computation.
    Designed for parallel execution.
    file_info is (filepath_base, data_array)
    """
    _, data = file_info
    return data.flatten()


def compute_global_quantiles_parallel_in_memory(files_info, quantile_levels):
    """
    Computes global quantiles of raw data in parallel from in-memory data.
    """
    all_data_flat = []

    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Using {num_cpus} CPU workers for collecting data for quantiles.")

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(
            tqdm(
                executor.map(_get_flattened_data_for_stats, files_info),
                total=len(files_info),
                desc="Collecting data for quantiles... ",
            )
        )

    for data_flat in results:
        all_data_flat.append(data_flat)

    # Concatenate all flattened data into a single array for quantile computation
    # Filter out empty arrays that might arise from empty patches
    all_data_flat_combined = np.concatenate(
        [arr for arr in all_data_flat if arr.size > 0]
    )

    # Compute quantiles on the combined and flattened raw data
    quantiles = np.nanquantile(all_data_flat_combined, quantile_levels)

    return quantiles


def coarsen_array(arr, factor):
    """
    Coarsens an array by a given factor using simple averaging.
    """
    m, n = arr.shape
    m_new = m // factor
    n_new = n // factor
    arr = arr[: m_new * factor, : n_new * factor]
    return arr.reshape(m_new, factor, n_new, factor).mean(axis=(1, 3))


def interpolate_array(arr, factor):
    """
    Interpolates an array by a given factor using cubic spline interpolation (order=3).
    """
    return zoom(arr, zoom=factor, order=3)


def process_file_wrapper_in_memory(args):
    """
    Wrapper function to process a single file's data (in-memory): decluttering, coarsening, interpolation.
    Saves all outputs in a single .npz file.
    args is (filepath_base, data_array, factor, declutter_threshold)
    """
    filepath_base, data, factor, declutter_threshold = args

    # Apply decluttering directly to the original data
    decluttered_data = declutter_precip(data, declutter_threshold)

    # Coarsening and interpolation are now applied to the decluttered data
    coarse = coarsen_array(decluttered_data, factor)
    interp = interpolate_array(decluttered_data, factor)

    npz_output_path = filepath_base + ".npz"
    np.savez(
        npz_output_path,
        original=data,  # Original data before any processing
        decluttered=decluttered_data,  # Data after decluttering
        coarsened=coarse,
        interpolated=interp,
    )
    return npz_output_path


# --- Main Orchestration Function ---
def main_preprocessing_pipeline():
    """
    Orchestrates the entire data preprocessing workflow:
    1. Extracts precipitation and DEM patches (precipitation kept in memory).
    2. Splits data into train/val/test sets by date.
    3. Generates file lists for each set.
    4. Performs decluttering, coarsening, and interpolation on in-memory precipitation data,
       saving the final .npz files.
    5. Computes and saves quantiles for the raw (undecluttered) training data.
    """
    print("--- Starting Full Data Preprocessing Pipeline ---")

    # --- Step 1: Extract Precipitation and DEM Patches (Precipitation in-memory) ---
    print("\n## Step 1: Extracting Precipitation (in-memory) and DEM Patches")
    print(
        "Starting patch extraction for precipitation data and identifying valid patch locations..."
    )

    # The raw data Zarr folders are still YYYYMMDDTHHMMSS
    zarr_folders = sorted(
        [
            f
            for f in os.listdir(RAW_OPERA_DATA_PATH)
            if os.path.isdir(os.path.join(RAW_OPERA_DATA_PATH, f))
        ]
    )

    all_precip_coords = set()  # For DEM mapping
    all_precip_data_for_later_processing = []  # List of (filepath_base, patch_data)

    with ProcessPoolExecutor(max_workers=N_WORKERS_PATCH_EXTRACTION) as executor:
        futures = {
            executor.submit(process_single_zarr_folder_in_memory, folder): folder
            for folder in zarr_folders
        }
        for future in as_completed(futures):
            folder = futures[future]
            try:
                precip_data_from_folder, coords_from_folder = future.result()
                if precip_data_from_folder:
                    all_precip_data_for_later_processing.extend(precip_data_from_folder)
                if coords_from_folder:
                    all_precip_coords.update(coords_from_folder)
            except Exception as e:
                print(f"Exception processing folder {folder}: {e}")

    print(
        f"Total precipitation patches extracted (in-memory): {len(all_precip_data_for_later_processing)}"
    )
    print(
        "\nAll precipitation patch metadata collected. Now loading and processing DEM data..."
    )
    try:
        dem_da = xr.open_dataarray(REPROJECTED_DEM_PATH)
        # Ensure 'y' coordinate is increasing for consistent slicing
        if dem_da.y.values[0] > dem_da.y.values[-1]:
            dem_da = dem_da.isel(y=slice(None, None, -1))
        dem_array_2d = dem_da.values
        min_valid_pixels_dem = int(MIN_VALID_FRACTION_DEM * PATCH_SIZE * PATCH_SIZE)
        process_dem_data(
            dem_array_2d, all_precip_coords, PATCH_SIZE, min_valid_pixels_dem
        )
    except FileNotFoundError:
        print(
            f"Error: DEM file not found at {REPROJECTED_DEM_PATH}. Cannot process DEM patches."
        )
        return  # Exit if DEM is critical and not found
    except Exception as e:
        print(f"Error loading or processing DEM: {e}")
        return  # Exit if DEM processing fails

    print("\nFinished patch extraction for precipitation (in-memory) and DEM data.")

    # --- Step 2: Split Data into Train/Val/Test Sets by Date ---
    # And Step 3: Generate File Lists
    print("\n## Step 2 & 3: Splitting Data and Generating File Lists")
    # train_dates_list, val_dates_list, test_dates_list will contain YYYYMMDDTHHMMSS folder names
    train_dates_list, val_dates_list, test_dates_list = (
        get_train_val_test_dates_orchestrator(RAW_OPERA_DATA_PATH)
    )

    print(f"Train dates (YYYYMMDDTHHMMSS folders): {len(train_dates_list)} entries")
    print(f"Validation dates (YYYYMMDDTHHMMSS folders): {len(val_dates_list)} entries")
    print(f"Test dates (YYYYMMDDTHHMMSS folders): {len(test_dates_list)} entries")

    # Filter the in-memory patch data based on the determined dates
    train_files_metadata = []  # List of (filepath_base, patch_data)
    val_files_metadata = []
    test_files_metadata = []

    # Create sets for faster lookup
    train_dates_set = set(train_dates_list)  # These are YYYYMMDDTHHMMSS
    val_dates_set = set(val_dates_list)
    test_dates_set = set(test_dates_list)

    # Use a single loop to categorize the collected patch data
    for filepath_base, patch_data in tqdm(
        all_precip_data_for_later_processing, desc="Categorizing patches by date"
    ):
        # Extract the YYYYMMDDTHHMMSS part from the filepath_base
        path_parts = filepath_base.split(os.sep)
        date_time_dir_from_path = path_parts[-2]  # e.g., '20230801T000000'

        if date_time_dir_from_path in train_dates_set:
            train_files_metadata.append((filepath_base, patch_data))
        elif date_time_dir_from_path in val_dates_set:
            val_files_metadata.append((filepath_base, patch_data))
        elif date_time_dir_from_path in test_dates_set:
            test_files_metadata.append((filepath_base, patch_data))
        # Note: Patches whose dates don't fall into train/val/test lists are implicitly discarded here.

    print(f"Total training files (in-memory): {len(train_files_metadata)}")
    print(f"Total validation files (in-memory): {len(val_files_metadata)}")
    print(f"Total testing files (in-memory): {len(test_files_metadata)}")

    # Save only the *paths* to the .txt files. The .npy extension is added here for consistency
    save_to_txt(
        train_files_metadata,
        val_files_metadata,
        test_files_metadata,
        FINAL_FILE_LISTS_AND_STATS_DIR,
    )
    print("Finished splitting data and generating file lists.")

    # --- Step 4: Data Processing (Decluttering, Coarsening, Interpolation) ---
    print("\n## Step 4: Applying Decluttering and Transformations")

    data_sets_to_process = {
        "train_files.txt": train_files_metadata,
        "val_files.txt": val_files_metadata,
        "test_files.txt": test_files_metadata,
    }

    for file_list_name, files_info_list in data_sets_to_process.items():
        print(f"Processing data for {file_list_name}...")
        print(f"Start processing {len(files_info_list)} data entries...")

        if "train" in file_list_name:
            print("Computing quantiles for raw training data...")
            # Compute quantiles on the *combined and flattened* raw data
            train_quantiles = compute_global_quantiles_parallel_in_memory(
                files_info_list, QUANTILE_LEVELS
            )

            stats = {
                "quantile_levels": QUANTILE_LEVELS,
                "train_quantiles": train_quantiles,
            }
            print(f"Training Quantiles ({QUANTILE_LEVELS}): {train_quantiles}")
            np.save(
                os.path.join(FINAL_FILE_LISTS_AND_STATS_DIR, "train_stat.npy"), stats
            )
        else:
            # For validation/test, we don't need to recompute quantiles or stats
            # We're not applying any statistics from training to these sets,
            # but loading train_stat.npy might still be useful for other contexts later.
            print("Skipping quantile computation for non-training data.")

        print("Starting parallel processing for data transformations...")
        # Prepare arguments for each call to process_file_wrapper_in_memory
        # Now passing DECLUTTER_THRESHOLD instead of mean/std
        tasks = [
            (filepath_base, data, DOWNSCALING_FACTOR, DECLUTTER_THRESHOLD)
            for filepath_base, data in files_info_list
        ]

        num_cpus = 32  # int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        print(f"Using {num_cpus} CPU workers for data transformations.")

        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            for idx, completed_path_npz in enumerate(
                tqdm(
                    executor.map(process_file_wrapper_in_memory, tasks),
                    total=len(tasks),
                    desc=f"Processing {file_list_name} data",
                )
            ):
                if (idx + 1) % 1000 == 0:
                    print(
                        f"Processed {idx + 1} data entries (last saved: {completed_path_npz})"
                    )
        print(
            f"Finished processing all {len(files_info_list)} data entries in {file_list_name}."
        )

    print("\n--- Full Data Preprocessing Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main_preprocessing_pipeline()
