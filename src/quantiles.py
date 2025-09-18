import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
import yaml
import glob


config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
PATCH_SIZE = config["PATCH_SIZE"]
RAW_OPERA_DATA_DIR = config["RAW_OPERA_DATA_DIR"]
METADATA_DIR = config["METADATA_DIR"]


# Function to read file paths from a list file
def get_file_paths_from_list(list_filepath):
    if not os.path.exists(list_filepath):
        raise FileNotFoundError(
            f"File list not found: {list_filepath}. Please ensure the merged preprocessing script ran."
        )
    with open(list_filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


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

    train_patch_metadata = []
    train_metadata_path = os.path.join(METADATA_DIR, "train_patches_metadata.txt")
    with open(train_metadata_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                timestamp_str, y_str, x_str = parts
                train_patch_metadata.append((timestamp_str, int(y_str), int(x_str)))
    print(
        f"Starting preprocessing for {len(train_patch_metadata)} entries from {train_metadata_path}"
    )

    quantile_levels = config["QUANTILE_LEVELS"]
    quantiles = compute_global_quantiles_parallel_from_zarr(
        all_zarr_folder_info, train_patch_metadata, quantile_levels, precip_var_name
    )
    base_preprocessed_dir = config["PREPROCESSED_DATA_DIR"]
    train_preprocessed_dir = os.path.join(base_preprocessed_dir, "train")
    np.savez_compressed(
        os.path.join(train_preprocessed_dir, "train_quantiles.npz"), data=quantiles
    )

    return print("Done.")


if __name__ == "__main__":
    main()
