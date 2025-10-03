import os
import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import zoom
from tqdm import tqdm
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing  # For Manager

# Load configuration
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PATCH_SIZE = config["PATCH_SIZE"]
DOWNSCALING_FACTOR = config["DOWNSCALING_FACTOR"]
DECLUTTER_THRESHOLD = config["DECLUTTER_THRESHOLD"]
RAW_OPERA_DATA_DIR = config["RAW_OPERA_DATA_DIR"]
PREPROCESSED_DATA_DIR = config.get("PREPROCESSED_DATA_DIR")
METADATA_DIR = config["METADATA_DIR"]


# --- Preprocessing Utility Functions (unchanged) ---
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


def interpolate_array(arr, factor, target_shape=None):
    """Interpolates an array by a given factor using cubic spline interpolation (order=3).
    If target_shape is given, resizes exactly to that shape after interpolation.
    """
    # Interpolate with zoom
    interpolated = zoom(arr.astype(np.float32), zoom=factor, order=3)

    if target_shape is not None:
        # Resize using slicing or padding to enforce exact dimensions
        current_shape = interpolated.shape
        pad_y = max(0, target_shape[0] - current_shape[0])
        pad_x = max(0, target_shape[1] - current_shape[1])

        # Pad if too small
        if pad_y > 0 or pad_x > 0:
            interpolated = np.pad(
                interpolated,
                ((0, pad_y), (0, pad_x)),
                mode="constant",
                constant_values=0.0,
            )

        # Crop if too large
        interpolated = interpolated[: target_shape[0], : target_shape[1]]

    return interpolated


# --- Global Zarr Cache for each process ---
# Each process will have its own _process_zarr_cache
_process_zarr_cache = {}


def _get_zarr_dataset_for_process(zarr_folder_path):
    """
    Manages a Zarr dataset cache specific to the current process.
    """
    global _process_zarr_cache
    if zarr_folder_path not in _process_zarr_cache:
        _process_zarr_cache[zarr_folder_path] = xr.open_zarr(
            zarr_folder_path, chunks={}, decode_times=True
        )
    return _process_zarr_cache[zarr_folder_path]


def _close_process_zarr_cache():
    """
    Closes all Zarr datasets in the current process's cache.
    Called before a process exits.
    """
    global _process_zarr_cache
    for ds in _process_zarr_cache.values():
        if hasattr(ds, "close"):
            ds.close()
    _process_zarr_cache = {}  # Clear the cache


def _initializer():
    """
    Initializer function for ProcessPoolExecutor to set up resources per process.
    Currently, just ensuring the cache is fresh.
    """
    global _process_zarr_cache
    _process_zarr_cache = {}  # Ensure each process starts with an empty cache


# --- New function to process a single patch ---
def process_single_patch(
    patch_metadata,
    all_zarr_folder_info,
    preprocessed_data_dir,
    precip_var_name,
    patch_size,
    downscaling_factor,
    declutter_threshold,
    transform_input_precip,
):
    """
    Processes and saves data for a single patch.
    This function is designed to be run in parallel.
    """
    timestamp_str, y_start, x_start = patch_metadata

    # Define file paths for the current patch
    original_precip_filename = (
        f"original_precip_{timestamp_str}_y{y_start:04d}_x{x_start:04d}.npy"
    )
    interpolated_precip_filename = (
        f"interpolated_precip_{timestamp_str}_y{y_start:04d}_x{x_start:04d}.npy"
    )
    coarse_precip_filename = (
        f"coarse_precip_{timestamp_str}_y{y_start:04d}_x{x_start:04d}.npy"
    )

    original_precip_path = os.path.join(
        preprocessed_data_dir, "original_precip", original_precip_filename
    )
    interpolated_precip_path = os.path.join(
        preprocessed_data_dir, "interpolated_precip", interpolated_precip_filename
    )
    coarse_precip_path = os.path.join(
        preprocessed_data_dir, "coarse_precip", coarse_precip_filename
    )

    # Check if files already exist. If so, skip processing for this patch.
    if (
        os.path.exists(original_precip_path)
        and os.path.exists(interpolated_precip_path)
        and os.path.exists(coarse_precip_path)
    ):
        return f"Skipped existing: {timestamp_str}, Y:{y_start}, X:{x_start}"

    try:
        # Load Zarr dataset using the process-specific cache
        zarr_folder_path, time_idx_in_folder = all_zarr_folder_info[timestamp_str]
        ds = _get_zarr_dataset_for_process(zarr_folder_path)
        precip_data_array = ds[precip_var_name]

        # High-resolution output (original)
        output_precip = (
            precip_data_array.isel(
                time=time_idx_in_folder,
                y=slice(y_start, y_start + patch_size),
                x=slice(x_start, x_start + patch_size),
            )
            .load()
            .values.astype(np.float32)
        )
        output_precip[np.isnan(output_precip)] = 0.0

        # Save original precipitation
        # np.save(original_precip_path, output_precip)

        # Low-resolution input and interpolated
        if transform_input_precip:
            decluttered = declutter_precip(output_precip, declutter_threshold)
            low_res = coarsen_array(decluttered, downscaling_factor)
            interpolated = interpolate_array(
                low_res,
                downscaling_factor,
                target_shape=(patch_size, patch_size),
            )
        else:
            # If no transformation, input is just the original precip
            interpolated = output_precip.copy()
            low_res = coarsen_array(
                output_precip, downscaling_factor
            )  # Still calculate for saving

        # Save interpolated and coarse precipitation
        # np.save(interpolated_precip_path, interpolated)
        # np.save(
        #    coarse_precip_path, low_res
        # )

        print(f"Processed: {timestamp_str}, Y:{y_start}, X:{x_start}")
        return output_precip, interpolated, low_res
    except Exception as e:
        return f"Error processing {timestamp_str}, Y:{y_start}, X:{x_start}: {e}"


# --- Main preprocessing orchestration function (modified for parallelism) ---
def parallel_preprocess_and_save_data(
    metadata_file_path,
    all_zarr_folder_info,
    preprocessed_data_dir,
    precip_var_name="TOT_PREC",
    patch_size=PATCH_SIZE,
    downscaling_factor=DOWNSCALING_FACTOR,
    declutter_threshold=DECLUTTER_THRESHOLD,
    transform_input_precip=True,
    max_workers=32,  # Number of parallel processes
):
    """
    Preprocesses Zarr precipitation data in parallel and saves it as .npz files.
    """
    metadata = []
    with open(metadata_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                timestamp_str, y_str, x_str = parts
                metadata.append((timestamp_str, int(y_str), int(x_str)))
    print(
        f"Starting preprocessing for {len(metadata)} entries from {metadata_file_path}"
    )

    original_patches = []
    interpolated_patches = []
    coarse_patches = []

    # Prepare arguments for each task
    tasks = [
        (
            patch,
            all_zarr_folder_info,
            preprocessed_data_dir,
            precip_var_name,
            patch_size,
            downscaling_factor,
            declutter_threshold,
            transform_input_precip,
        )
        for patch in metadata
    ]

    # Use ProcessPoolExecutor for parallel execution
    # Set max_workers to os.cpu_count() or a sensible number
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_initializer
    ) as executor:
        futures = [
            executor.submit(process_single_patch, *task_args) for task_args in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Parallel Preprocessing ({max_workers or os.cpu_count()} workers)",
            leave=True,
        ):
            result = future.result()
            if result is None:
                continue
            # Unpack the arrays
            original_arr, interpolated_arr, coarse_arr = result
            original_patches.append(original_arr)
            interpolated_patches.append(interpolated_arr)
            coarse_patches.append(coarse_arr)

    original_patches = np.stack(original_patches, axis=0)
    interpolated_patches = np.stack(interpolated_patches, axis=0)
    coarse_patches = np.stack(coarse_patches, axis=0)

    # Save them as .npz files
    np.savez_compressed(
        os.path.join(preprocessed_data_dir, "original_precip.npz"),
        data=original_patches,
    )
    np.savez_compressed(
        os.path.join(preprocessed_data_dir, "interpolated_precip.npz"),
        data=interpolated_patches,
    )
    np.savez_compressed(
        os.path.join(preprocessed_data_dir, "coarse_precip.npz"), data=coarse_patches
    )

    # IMPORTANT: Manually close Zarr datasets in each worker process when done
    # This isn't strictly necessary with ProcessPoolExecutor as processes exit,
    # but it's good practice for long-running processes or if you were using threads.
    # For ProcessPoolExecutor, the `_close_process_zarr_cache` is implicitly handled
    # as processes shut down, but the `initializer` helps ensure a clean start.
    print(f"Preprocessing for {metadata_file_path} completed.")


# --- Main execution logic (modified to use parallel_preprocess_and_save_data) ---
def main():
    all_zarr_folder_info = {}
    precip_var_name = "TOT_PREC"

    # Modified glob pattern to match YYYYMMDD folders
    zarr_folders = sorted(
        glob.glob(
            os.path.join(
                RAW_OPERA_DATA_DIR,
                "[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]",
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
        return

    # Use a Manager to share all_zarr_folder_info across processes if needed for this phase
    # However, for simply reading metadata, a standard dict is fine if populated before multiprocessing.
    # If the _mapping_ of timestamps to zarr folders itself is parallelized,
    # then you'd need a Manager. For now, it's sequential.

    for folder_path in tqdm(
        zarr_folders, desc="Mapping Zarr folders and internal timestamps"
    ):
        try:
            # Use decode_times=True explicitly
            with xr.open_zarr(folder_path, chunks={}, decode_times=True) as ds:
                if "time" in ds.coords and precip_var_name in ds.data_vars:
                    for t_idx, t_val in enumerate(ds[precip_var_name]["time"].values):
                        # Ensure conversion to Python datetime before strftime
                        dt_obj = t_val.astype("datetime64[s]").item()
                        full_timestamp_str = dt_obj.strftime("%Y%m%d%H%M%S")
                        all_zarr_folder_info[full_timestamp_str] = (folder_path, t_idx)
                else:
                    print(
                        f"Warning: Zarr dataset {folder_path} does not contain 'time' coordinate "
                        f"or '{precip_var_name}' variable. Skipping."
                    )
        except Exception as e:
            print(
                f"Warning: Could not process {folder_path} for timestamp mapping: {e}. Skipping."
            )
            continue

    print(
        f"Total unique timestamps identified across all Zarr folders: {len(all_zarr_folder_info)}"
    )

    train_metadata_path = os.path.join(METADATA_DIR, "train_patches_metadata.txt")
    val_metadata_path = os.path.join(METADATA_DIR, "val_patches_metadata.txt")
    test_metadata_path = os.path.join(METADATA_DIR, "test_patches_metadata.txt")

    base_preprocessed_dir = PREPROCESSED_DATA_DIR
    train_preprocessed_dir = os.path.join(base_preprocessed_dir, "train")
    os.makedirs(train_preprocessed_dir, exist_ok=True)
    val_preprocessed_dir = os.path.join(base_preprocessed_dir, "validation")
    os.makedirs(val_preprocessed_dir, exist_ok=True)
    test_preprocessed_dir = os.path.join(base_preprocessed_dir, "test")
    os.makedirs(test_preprocessed_dir, exist_ok=True)

    num_workers = 32
    print(f"Using {num_workers} worker processes for preprocessing.")

    print("\n--- Starting Parallel Preprocessing for Training Data ---")
    parallel_preprocess_and_save_data(
        metadata_file_path=train_metadata_path,
        all_zarr_folder_info=all_zarr_folder_info,
        preprocessed_data_dir=train_preprocessed_dir,
        max_workers=num_workers,
    )

    print("\n--- Starting Parallel Preprocessing for Validation Data ---")
    parallel_preprocess_and_save_data(
        metadata_file_path=val_metadata_path,
        all_zarr_folder_info=all_zarr_folder_info,
        preprocessed_data_dir=val_preprocessed_dir,
        max_workers=num_workers,
    )

    print("\n--- Starting Parallel Preprocessing for Test Data ---")
    parallel_preprocess_and_save_data(
        metadata_file_path=test_metadata_path,
        all_zarr_folder_info=all_zarr_folder_info,
        preprocessed_data_dir=test_preprocessed_dir,
        max_workers=num_workers,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
