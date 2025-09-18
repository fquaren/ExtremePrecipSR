import os
import numpy as np
from scipy.ndimage import zoom  # For interpolate_array
import xarray as xr
from datetime import datetime, timedelta
import glob
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import json


# --- Re-define parameters and preprocessing functions from the previous script ---
# These would typically be imported or defined in a config file accessible by your training script.

# Patch Extraction Parameters
PATCH_SIZE = 128
STRIDE = 128  # Though STRIDE is implicitly handled by how metadata is generated
DOWNSCALING_FACTOR = 6
DECLUTTER_THRESHOLD = 150.0
MIN_VALID_FRACTION_PRECIP = (
    1.0  # Consider lowering if minor NaNs are acceptable in patches
)
MIN_VALID_FRACTION_DEM = (
    1.0  # Consider lowering if minor NaNs are acceptable in DEM patches
)

N_WORKERS_PATCH_EXTRACTION = 8

# Paths (adjust these to your actual paths)
BASE_DATA_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/"
# Now this points to the directory containing all the YYYYMMDD Zarr sub-directories
RAW_OPERA_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw_data", "OPERA")


PROCESSED_DATA_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "fquareng", "data", "OPERA")
DEM_PATCH_OUTPUT_DIR = os.path.join(PROCESSED_DATA_OUTPUT_DIR, "patches", "dem")
FINAL_FILE_LISTS_AND_STATS_DIR = os.path.join(
    BASE_DATA_DIR, "fquareng", "data", "OPERA"
)
DEM_DATA_DIR = os.path.join(PROCESSED_DATA_OUTPUT_DIR, "DEM")
REPROJECTED_DEM_FILENAME = "reproj_OPERA_1km_europe_dem.nc"
REPROJECTED_DEM_PATH = os.path.join(DEM_DATA_DIR, REPROJECTED_DEM_FILENAME)

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


# Preprocessing Functions (copied for completeness, assume they are available)
def declutter_precip(arr, threshold):
    """Sets pixel values in the array that are above the given threshold to zero."""
    arr_copy = arr.copy()
    arr_copy[arr_copy > threshold] = 0
    return arr_copy


def coarsen_array(arr, factor):
    """Coarsens an array by a given factor using simple averaging."""
    m, n = arr.shape
    m_new = m // factor
    n_new = n // factor
    arr = arr[: m_new * factor, : n_new * factor]
    return arr.reshape(m_new, factor, n_new, factor).mean(axis=(1, 3))


def interpolate_array(arr, factor):
    """Interpolates an array by a given factor using cubic spline interpolation (order=3)."""
    # Ensure input is float32 for zoom
    return zoom(arr.astype(np.float32), zoom=factor, order=3)


# --- Helper functions for the preprocessing script (modified) ---


def extract_valid_patches_with_coords(
    array_2d, patch_size, stride, min_valid_pixels_count
):
    """
    Extracts top-left (y, x) coordinates of valid patches from a 2D array.
    A patch is considered valid if it contains at least 'min_valid_pixels_count' non-NaN pixels.
    """
    y_dim, x_dim = array_2d.shape
    valid_coords = []

    for i in range(0, y_dim - patch_size + 1, stride):
        for j in range(0, x_dim - patch_size + 1, stride):
            patch = array_2d[i : i + patch_size, j : j + patch_size]
            valid_pixels = np.count_nonzero(~np.isnan(patch))

            if valid_pixels >= min_valid_pixels_count:
                valid_coords.append(
                    (i, j)
                )  # These are local coords within the 2D slice
    return valid_coords


def process_single_zarr_folder_for_metadata(zarr_folder_path, precip_var_name):
    """
    Processes a single Zarr folder (e.g., '20170101') to identify valid patch coordinates.
    Returns metadata for each valid patch: (full_timestamp_str, y_start, x_start)
    """
    all_valid_coords_in_folder = set()
    all_precip_patch_metadata = []  # (full_timestamp_str, y_start, x_start)

    try:
        # Open the specific Zarr folder as an xarray dataset using a 'with' statement
        with xr.open_zarr(
            zarr_folder_path, chunks={}
        ) as ds:  # ADDED 'with' statement here
            precip_data_array = ds[precip_var_name]

            # The folder name itself often contains the date, but the 'time' dimension
            # in the Zarr array holds the exact timestamps.
            # We need to iterate through time steps in this Zarr dataset.
            min_valid_pixels = int(MIN_VALID_FRACTION_PRECIP * PATCH_SIZE * PATCH_SIZE)

            for t_idx in range(precip_data_array.sizes["time"]):
                slice_2d = precip_data_array.isel(time=t_idx).values.astype(np.float32)

                # Get the exact timestamp for this slice from the DataArray
                # timestamp_obj = precip_data_array.isel(time=t_idx)["time"].item()
                # if isinstance(timestamp_obj, np.datetime64):
                #     timestamp_obj = timestamp_obj.astype(datetime)
                # full_timestamp_str = timestamp_obj.strftime(
                #     "%Y%m%d%H%M%S"
                # )  # e.g., 20170101000000

                try:
                    full_timestamp_str = (
                        precip_data_array.isel(time=t_idx)["time"]
                        .dt.strftime("%Y%m%d%H%M%S")
                        .item()
                    )
                except AttributeError:
                    full_timestamp_str = f"time index {t_idx}"  # Fallback if 'dt' accessor is not available

                patches_coords = extract_valid_patches_with_coords(
                    slice_2d, PATCH_SIZE, STRIDE, min_valid_pixels
                )

                if patches_coords:
                    for y_start, x_start in patches_coords:
                        all_precip_patch_metadata.append(
                            (full_timestamp_str, y_start, x_start)
                        )
                        all_valid_coords_in_folder.add((y_start, x_start))  # For DEM

    except Exception as e:
        print(f"Error processing {zarr_folder_path}: {e}")

    return all_precip_patch_metadata, list(all_valid_coords_in_folder)


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
                # Ensure DEM output directory exists
                os.makedirs(DEM_PATCH_OUTPUT_DIR, exist_ok=True)
                patch_filename = f"dem_patch_y{y_start:04d}_x{x_start:04d}.npy"
                np.save(os.path.join(DEM_PATCH_OUTPUT_DIR, patch_filename), dem_patch)


# --- Functions for Date-Based Data Splitting Logic (adapted for timestamps) ---


def get_all_dated_zarr_folders(base_path, return_datetimes=False):
    """
    Reads directories with YYYYMMDD structure from a given base path,
    and optionally extracts unique timestamps (dates only from folder names).

    Args:
        base_path (str): The base directory to search within.
        return_datetimes (bool): If True, returns a list of unique datetime.date objects
                                  parsed from the folder names. Otherwise, returns
                                  a list of unique folder paths.

    Returns:
        list: A sorted list of unique Zarr folder paths or datetime.date objects.
    """
    # Regex for YYYYMMDD
    # The parentheses create capturing groups for year, month, day.
    date_regex = r"(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])"

    # The glob pattern matches YYYYMMDD folder names
    glob_pattern = os.path.join(base_path, "[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]")

    all_zarr_folders = sorted(glob.glob(glob_pattern))

    if return_datetimes:
        unique_dates = (
            set()
        )  # Changed to unique_dates as only date is parsed from folder name
        for folder_path in all_zarr_folders:
            match = re.search(date_regex, os.path.basename(folder_path))
            if match:
                year, month, day = match.groups()  # Only year, month, day are captured
                try:
                    dt_object = datetime(
                        int(year), int(month), int(day)
                    ).date()  # Store as date object for consistency with parsing from folder names
                    unique_dates.add(dt_object)
                except ValueError:
                    # Handle cases where the regex matches but datetime conversion fails
                    print(
                        f"Warning: Could not parse date from {folder_path} (e.g., Feb 30th)."
                    )
        return sorted(list(unique_dates))
    else:
        return all_zarr_folders


def get_date_ranges_from_timestamps(all_unique_timestamps):
    """
    Splits unique timestamps into train, validation, and test sets
    based on specific date ranges and a 3-weeks-on/1-week-off pattern.
    """
    list1_selected_weeks = []  # For training/validation
    list2_skipped_weeks = []  # Skipped weeks
    list3_aug_oct_2024 = []  # For testing

    # Define the date ranges
    # Note: Using .date() to ensure comparison with date objects from folder names
    start_date_aug_2023 = datetime(2023, 8, 1).date()
    end_date_aug_2024_excluded = datetime(2024, 8, 1).date()
    start_date_aug_2024 = datetime(2024, 8, 1).date()
    end_date_oct_2024 = datetime(2024, 10, 30).date()

    # Filter timestamps relevant to the first two lists (August 2023 to August 2024 excluded)
    timestamps_for_list1_2 = sorted(
        [
            ts
            for ts in all_unique_timestamps
            if start_date_aug_2023 <= ts.date() < end_date_aug_2024_excluded
        ]
    )

    # Process for List 1 and List 2 (August 2023 to August 2024 excluded)
    if timestamps_for_list1_2:
        first_date = timestamps_for_list1_2[0]
        # Ensure current_week_monday is a date object for consistent comparison
        current_week_monday = first_date - timedelta(days=first_date.weekday())

        week_count = 0
        timestamps_in_current_week = []

        for ts in timestamps_for_list1_2:
            # Compare date objects directly
            if ts >= current_week_monday + timedelta(weeks=1):
                if week_count % 4 < 3:  # 0, 1, 2 (first three weeks of the cycle)
                    list1_selected_weeks.extend(timestamps_in_current_week)
                else:  # 3 (fourth week of the cycle, skipped)
                    list2_skipped_weeks.extend(timestamps_in_current_week)

                current_week_monday = ts - timedelta(days=ts.weekday())
                week_count += 1
                timestamps_in_current_week = []

            timestamps_in_current_week.append(ts)

        # Process any remaining dates from the last week
        if timestamps_in_current_week:
            if week_count % 4 < 3:
                list1_selected_weeks.extend(timestamps_in_current_week)
            else:
                list2_skipped_weeks.extend(timestamps_in_current_week)

    # Process for List 3 (August 1, 2024 to October 30, 2024)
    for ts in all_unique_timestamps:
        if start_date_aug_2024 <= ts.date() <= end_date_oct_2024:
            list3_aug_oct_2024.append(ts)

    # Convert datetime objects to YYYYMMDDHHMMSS strings for storage/lookup
    # Note: These will have 000000 for time, as only date was parsed from folder names
    return (
        [dt.strftime("%Y%m%d%H%M%S") for dt in list1_selected_weeks],
        [dt.strftime("%Y%m%d%H%M%S") for dt in list2_skipped_weeks],
        [dt.strftime("%Y%m%d%H%M%S") for dt in list3_aug_oct_2024],
    )


def get_train_val_test_dates_orchestrator(base_data_dir):
    """Orchestrates getting all unique dates from YYYYMMDD Zarr folders and splitting them into ranges."""
    # Call get_all_dated_zarr_folders to return datetime.date objects
    all_dates = get_all_dated_zarr_folders(base_data_dir, return_datetimes=True)
    return get_date_ranges_from_timestamps(all_dates)


# --- Functions for File Path Generation and Saving (adapted for metadata) ---
def save_metadata_to_txt(train_metadata, val_metadata, test_metadata, save_path):
    """
    Saves lists of metadata (timestamp_str, y_start, x_start) to separate .txt files.
    Each line will represent a unique patch: 'YYYYMMDDHHMMSS,Y_COORD,X_COORD'.
    """
    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)

    train_files_path = os.path.join(save_path, "train_patches_metadata.txt")
    with open(train_files_path, "w") as f:
        for ts, y, x in train_metadata:
            f.write(f"{ts},{y},{x}\n")

    val_files_path = os.path.join(save_path, "val_patches_metadata.txt")
    with open(val_files_path, "w") as f:
        for ts, y, x in val_metadata:
            f.write(f"{ts},{y},{x}\n")

    test_files_path = os.path.join(save_path, "test_patches_metadata.txt")
    with open(test_files_path, "w") as f:
        for ts, y, x in test_metadata:
            f.write(f"{ts},{y},{x}\n")
    print(f"Patch metadata lists saved at {save_path}.")


def _get_flattened_data_for_stats_from_zarr(args):
    """
    Helper function to return the flattened original data for quantile computation
    from a specific Zarr patch. Designed for parallel execution.
    Uses a per-process cache for opened Zarr datasets to reduce I/O overhead.

    args is (zarr_folder_path, time_idx_in_folder, y_start, x_start, precip_var_name, patch_size)
    """
    (
        zarr_folder_path,
        time_idx_in_folder,
        y_start,
        x_start,
        precip_var_name,
        patch_size,
    ) = args

    try:
        # Check if the Zarr dataset is already in this process's cache
        if zarr_folder_path not in _zarr_dataset_cache:
            # If not, open it and store it in the cache
            # chunks={} loads the full data for that Zarr group into memory (or creates Dask arrays if chunked)
            _zarr_dataset_cache[zarr_folder_path] = xr.open_zarr(
                zarr_folder_path, chunks={}
            )
            # print(f"DEBUG (PID: {os.getpid()}): Opened Zarr: {os.path.basename(zarr_folder_path)}")
            # Optional: for debugging cache hits

        # Retrieve the dataset from the cache
        ds = _zarr_dataset_cache[zarr_folder_path]
        precip_data_array = ds[precip_var_name]

        # Load the specific patch from Zarr
        # .values forces immediate computation and loading if Dask array
        patch_data = (
            precip_data_array.isel(time=time_idx_in_folder)
            .values[y_start : y_start + patch_size, x_start : x_start + patch_size]
            .astype(np.float32)
        )

        return patch_data.flatten()
    except Exception as e:
        print(
            f"Error in _get_flattened_data_for_stats_from_zarr for \
                  {zarr_folder_path} (time_idx {time_idx_in_folder}, {y_start},{x_start}): {e}"
        )
        return np.array([])  # Return empty array to be filtered later


def compute_global_quantiles_parallel_from_zarr(
    all_zarr_folder_info, train_patch_metadata, quantile_levels, precip_var_name
):
    """
    Computes global quantiles of raw data in parallel directly from Zarr folders.
    all_zarr_folder_info: A dict mapping full_timestamp_str to (zarr_folder_path, time_idx_in_folder)
    """
    all_data_flat = []

    # Prepare tasks: (zarr_folder_path, time_idx_in_folder, y_start, x_start, precip_var_name, patch_size)
    tasks = []
    for ts_str, y_start, x_start in train_patch_metadata:
        if ts_str in all_zarr_folder_info:
            zarr_path, time_idx_in_folder = all_zarr_folder_info[ts_str]
            tasks.append(
                (
                    zarr_path,
                    time_idx_in_folder,
                    y_start,
                    x_start,
                    precip_var_name,
                    PATCH_SIZE,
                )
            )

    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Using {num_cpus} CPU workers for collecting data for quantiles.")

    # Define a function to clean up the cache in each worker (called after completion)
    def _worker_cleanup():
        global _zarr_dataset_cache
        for ds in _zarr_dataset_cache.values():
            if hasattr(ds, "close"):
                ds.close()
        _zarr_dataset_cache = {}  # Clear the cache

    with ProcessPoolExecutor(
        max_workers=num_cpus, initializer=_worker_cleanup
    ) as executor:
        results = list(
            tqdm(
                executor.map(_get_flattened_data_for_stats_from_zarr, tasks),
                total=len(tasks),
                desc="Collecting data for quantiles... ",
            )
        )
        # Ensure all Zarr datasets opened by worker processes are closed
        # This can be tricky with ProcessPoolExecutor.
        # Often, datasets opened by workers are automatically closed when the worker process terminates.
        # However, for explicit control and resource management, you might want to consider
        # passing the _worker_cleanup function to the initializer argument of ProcessPoolExecutor
        # or relying on __del__ methods of xarray/zarr (less reliable).
        # For this specific case, once the 'with' block exits, the worker processes are terminated,
        # which usually handles resource cleanup. The cache helps only within the life of a worker.

    for data_flat in results:
        all_data_flat.append(data_flat)

    # Concatenate all flattened data into a single array for quantile computation
    # Add a check for empty list to prevent ValueError if no data was collected
    if not all_data_flat:
        print(
            "Warning: No valid data collected for quantile computation. Returning empty array."
        )
        return np.array([])  # Or raise an error, depending on desired behavior

    all_data_flat_combined = np.concatenate(
        [arr for arr in all_data_flat if arr.size > 0]
    )

    # Compute quantiles on the combined and flattened raw data
    quantiles = np.nanquantile(all_data_flat_combined, quantile_levels)

    return quantiles


# --- Main Orchestration Function (modified) ---
def main_preprocessing_pipeline():
    """
    Orchestrates the entire data preprocessing workflow:
    1. Extracts precipitation patch metadata (timestamps, y, x) by scanning individual Zarr folders.
    2. Extracts and saves static DEM patches for all identified locations.
    3. Splits data into train/val/test sets by date (based on timestamps).
    4. Generates metadata files (text files) for each set.
    5. Computes and saves quantiles for the raw training data.
    """
    print("--- Starting Full Data Preprocessing Pipeline ---")

    # --- Step 0: Identify all Zarr folders and their contained timestamps ---
    print(
        f"\n## Step 0: Identifying all individual Zarr folders in {RAW_OPERA_DATA_DIR}"
    )
    print(
        "This step builds a comprehensive map of timestamps to Zarr folder paths and internal time indices."
    )

    # This dictionary will map 'YYYYMMDDHHMMSS' timestamp string to (zarr_folder_path, time_idx_within_that_folder)
    # This is crucial for both quantile calculation and the Dataset class.
    all_zarr_folder_info = {}
    precip_var_name = "TOT_PREC"  # Assuming this is your precipitation variable

    # Modified glob pattern to match YYYYMMDD folders
    zarr_folders = sorted(
        glob.glob(
            os.path.join(
                RAW_OPERA_DATA_DIR,
                "[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]",  # Updated pattern here
            )
        )
    )

    # Added debug print to confirm found folders
    print(
        f"DEBUG: glob.glob found {len(zarr_folders)} folders. First 5: {zarr_folders[:5]}"
    )
    if not zarr_folders:
        print(
            f"DEBUG: No Zarr folders found matching the YYYYMMDD pattern in {RAW_OPERA_DATA_DIR}."
        )
        print(
            "Please verify the 'RAW_OPERA_DATA_DIR' path and the naming convention of Zarr subdirectories."
        )
        return  # Exit if no folders are found, as subsequent steps will fail

    for folder_path in tqdm(
        zarr_folders, desc="Mapping Zarr folders and internal timestamps"
    ):
        try:
            with xr.open_zarr(
                folder_path, chunks={}
            ) as ds:  # Added 'with' statement here
                if "time" in ds.coords and precip_var_name in ds.data_vars:
                    # Iterate through each time step in the current folder's Zarr
                    for t_idx, t_val in enumerate(ds[precip_var_name]["time"].values):
                        dt_obj = t_val.astype("datetime64[s]").astype(datetime)
                        full_timestamp_str = dt_obj.strftime("%Y%m%d%H%M%S")
                        all_zarr_folder_info[full_timestamp_str] = (folder_path, t_idx)
                else:
                    print(
                        f"Warning: Zarr dataset {folder_path} does not contain 'time' coordinate \
                            or '{precip_var_name}' variable."
                    )
        except Exception as e:
            print(
                f"Warning: Could not process {folder_path} for timestamp mapping: {e}"
            )
            continue

    print(
        f"Total unique timestamps identified across all Zarr folders: {len(all_zarr_folder_info)}"
    )

    if not all_zarr_folder_info:
        print("No timestamps could be extracted from Zarr folders. Exiting pipeline.")
        return  # Exit if no timestamps are extracted

    print(
        f"\nSaving all_zarr_folder_info map to {FINAL_FILE_LISTS_AND_STATS_DIR}/zarr_info_map.json"
    )
    # Convert tuples in values to lists for JSON serialization
    serializable_zarr_info = {k: list(v) for k, v in all_zarr_folder_info.items()}
    zarr_info_map_path = os.path.join(
        FINAL_FILE_LISTS_AND_STATS_DIR, "zarr_info_map.json"
    )
    try:
        with open(zarr_info_map_path, "w") as f:
            json.dump(serializable_zarr_info, f, indent=4)
        print("Successfully saved zarr_info_map.json.")
    except Exception as e:
        print(f"Error saving zarr_info_map.json: {e}")
        # Decide if you want to exit here or continue with a warning
        # For a critical map, it might be better to exit.
        return

    # --- Step 1: Extract Precipitation Patch Metadata ---
    print("\n## Step 1: Extracting Precipitation Patch Metadata (timestamps, y, x)")
    print(
        "Scanning Zarr time slices to identify valid patch locations and their timestamps..."
    )

    all_precip_coords_for_dem = set()  # For DEM mapping (unique (y, x) pairs)
    all_precip_patch_metadata = []  # List of (timestamp_str, y_start, x_start)

    # Process each identified Zarr folder for patch metadata
    # The 'futures' structure ensures parallel processing across Zarr folders
    num_workers_folders = min(N_WORKERS_PATCH_EXTRACTION, len(zarr_folders))

    with ProcessPoolExecutor(max_workers=num_workers_folders) as executor:
        futures = {
            executor.submit(
                process_single_zarr_folder_for_metadata, folder_path, precip_var_name
            ): folder_path
            for folder_path in zarr_folders
        }
        for future in as_completed(futures):
            folder_path = futures[future]
            try:
                metadata_from_folder, coords_from_folder = future.result()
                if metadata_from_folder:
                    all_precip_patch_metadata.extend(metadata_from_folder)
                if coords_from_folder:
                    all_precip_coords_for_dem.update(coords_from_folder)
            except Exception as e:
                print(f"Exception processing folder {folder_path}: {e}")

    print(
        f"Total precipitation patch metadata collected: {len(all_precip_patch_metadata)}"
    )
    if not all_precip_patch_metadata:
        print("No precipitation patches found. Exiting pipeline.")
        return  # Exit if no patches are found

    print(
        "\nAll precipitation patch metadata collected. Now loading and processing DEM data..."
    )

    # --- Step 2: Process and Save DEM data ---
    try:
        dem_da = xr.open_dataarray(REPROJECTED_DEM_PATH)
        # Ensure 'y' coordinate is increasing for consistent slicing
        if dem_da.y.values[0] > dem_da.y.values[-1]:
            dem_da = dem_da.isel(y=slice(None, None, -1))
        dem_array_2d = dem_da.values
        min_valid_pixels_dem = int(MIN_VALID_FRACTION_DEM * PATCH_SIZE * PATCH_SIZE)
        process_dem_data(
            dem_array_2d, all_precip_coords_for_dem, PATCH_SIZE, min_valid_pixels_dem
        )
    except FileNotFoundError:
        print(
            f"Error: DEM file not found at {REPROJECTED_DEM_PATH}. Cannot process DEM patches. \
                Please ensure the DEM file exists."
        )
        return
    except Exception as e:
        print(f"Error loading or processing DEM: {e}")
        return

    print("\nFinished extracting precipitation patch metadata and processing DEM data.")

    # --- Step 3: Split Data into Train/Val/Test Sets by Date ---
    # --- Step 4: Generate Metadata Files ---
    print("\n## Step 3 & 4: Splitting Data and Generating Metadata Files")

    # Get unique datetime objects from the earlier built map for splitting
    # These are full timestamps (YYYYMMDDHHMMSS) from the Zarr internal data
    unique_timestamps_dt = sorted(
        list(
            set(
                datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                for ts_str in all_zarr_folder_info.keys()
            )
        )
    )

    train_dates_list, val_dates_list, test_dates_list = get_date_ranges_from_timestamps(
        unique_timestamps_dt
    )

    print(f"Train timestamps (YYYYMMDDHHMMSS): {len(train_dates_list)} entries")
    print(f"Validation timestamps (YYYYMMDDHHMMSS): {len(val_dates_list)} entries")
    print(f"Test timestamps (YYYYMMDDHHMMSS): {len(test_dates_list)} entries")

    # Filter the collected patch metadata based on the determined date splits
    train_metadata = []  # List of (timestamp_str, y_start, x_start)
    val_metadata = []
    test_metadata = []

    # Create sets for faster lookup
    train_dates_set = set(train_dates_list)
    val_dates_set = set(val_dates_list)
    test_dates_set = set(test_dates_list)

    for ts_str, y_start, x_start in tqdm(
        all_precip_patch_metadata, desc="Categorizing patches by date"
    ):
        if ts_str in train_dates_set:
            train_metadata.append((ts_str, y_start, x_start))
        elif ts_str in val_dates_set:
            val_metadata.append((ts_str, y_start, x_start))
        elif ts_str in test_dates_set:
            test_metadata.append((ts_str, y_start, x_start))

    print(f"Total training patches: {len(train_metadata)}")
    print(f"Total validation patches: {len(val_metadata)}")
    print(f"Total testing patches: {len(test_metadata)}")

    # Save the metadata lists to text files
    save_metadata_to_txt(
        train_metadata,
        val_metadata,
        test_metadata,
        FINAL_FILE_LISTS_AND_STATS_DIR,
    )
    print("Finished splitting data and generating metadata files.")

    # --- Step 5: Compute Quantiles for Raw Training Data ---
    # print("\n## Step 5: Computing Quantiles for Raw Training Data")

    # # Compute quantiles on the *combined and flattened* raw data directly from Zarr
    # train_quantiles = compute_global_quantiles_parallel_from_zarr(
    #     all_zarr_folder_info, train_metadata, QUANTILE_LEVELS, precip_var_name
    # )

    # stats = {
    #     "quantile_levels": QUANTILE_LEVELS,
    #     "train_quantiles": train_quantiles,
    #     "declutter_threshold": DECLUTTER_THRESHOLD,
    #     "downscaling_factor": DOWNSCALING_FACTOR,
    # }
    # print(f"Training Quantiles ({QUANTILE_LEVELS}): {train_quantiles}")
    # np.save(os.path.join(FINAL_FILE_LISTS_AND_STATS_DIR, "train_stat.npy"), stats)

    # print("\n--- Full Data Preprocessing Pipeline Completed Successfully ---")


if __name__ == "__main__":
    # Create necessary output directories here, before calling main_preprocessing_pipeline
    # Redundant here as they are also created in save_metadata_to_txt and process_dem_data
    # But keeping them explicitly doesn't hurt.
    os.makedirs(DEM_PATCH_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_FILE_LISTS_AND_STATS_DIR, exist_ok=True)
    main_preprocessing_pipeline()
