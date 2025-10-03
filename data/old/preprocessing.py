import os
import numpy as np
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def _compute_stats_for_single_file(path):
    """
    Helper function to compute sums, sums_sq, and count for a single file.
    Designed for parallel execution.
    """
    data = np.load(path)
    log_data = np.log1p(data)
    return log_data.sum(), (log_data**2).sum(), log_data.size


def compute_global_stats_parallel(file_paths):
    """
    Computes global mean and standard deviation of log-transformed data in parallel.
    """
    sums_total = 0
    sums_sq_total = 0
    count_total = 0

    # Determine the number of CPU workers based on SLURM allocation or default
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Using {num_cpus} CPU workers for computing global statistics.")

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(
            tqdm(
                executor.map(_compute_stats_for_single_file, file_paths),
                total=len(file_paths),
                desc="Computing global statistics in parallel... ",
            )
        )

    # Aggregate results from all processes
    for sums, sums_sq, count in results:
        sums_total += sums
        sums_sq_total += sums_sq
        count_total += count

    mean = sums_total / count_total
    std = np.sqrt(sums_sq_total / count_total - mean**2)
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
    return (log_arr - mean) / std


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


def process_file_wrapper(args):
    """
    Wrapper function to process a single file: normalization, coarsening, interpolation.
    Saves all outputs in a single .npz file.
    """
    path, factor, train_mean, train_std = args
    data = np.load(path)
    norm = normalize_precip(data, train_mean, train_std)
    coarse = coarsen_array(norm, factor)
    interp = interpolate_array(norm, factor)

    # Save all arrays in a single .npz file
    base, _ = os.path.splitext(path)
    np.savez(
        base + ".npz",
        original=data,
        normalized=norm,
        coarsened=coarse,
        interpolated=interp,
    )
    return path


def main():
    """
    Main function to orchestrate the data preprocessing workflow.
    """
    data_dir = (
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/data/"
    )
    train_files = os.path.join(data_dir, "train_files.txt")
    val_files = os.path.join(data_dir, "val_files.txt")
    test_files = os.path.join(data_dir, "test_files.txt")

    downscaling_factor = 6

    # Process files for training, validation, and testing sets
    for file_list_path in [train_files, val_files, test_files]:
        print(f"Preprocessing files at {file_list_path}...")

        # Read file paths from the list
        with open(file_list_path, "r") as f:
            files_paths = [line.strip() for line in f if line.strip()]

        print(f"Start processing {len(files_paths)} files...")

        # Handle global statistics calculation (only for training set)
        if "train" in file_list_path:
            print("Computing global statistics (parallelized)...")
            train_mean, train_std = compute_global_stats_parallel(files_paths)
            stats = {"train_mean": train_mean, "train_std": train_std}
            print(f"Global Mean: {train_mean}, Global Std: {train_std}")
            np.save(os.path.join(data_dir, "train_stat.npy"), stats)
        else:
            print("Using statistics from training...")
            stats = np.load(
                os.path.join(data_dir, "train_stat.npy"), allow_pickle=True
            ).item()  # Use .item() for dictionaries
            train_mean = stats["train_mean"]
            train_std = stats["train_std"]
            print(f"Global Mean: {train_mean}, Global Std: {train_std}")

        print("Starting parallel processing for file transformations...")
        # Prepare arguments for each call to process_file_wrapper
        tasks = [
            (path, downscaling_factor, train_mean, train_std) for path in files_paths
        ]

        # Determine the number of CPU workers for file processing
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        print(f"Using {num_cpus} CPU workers for file transformations.")

        # Use ProcessPoolExecutor for parallel processing of each file
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            for idx, completed_path in enumerate(
                tqdm(
                    executor.map(process_file_wrapper, tasks),
                    total=len(tasks),
                    desc=f"Processing {os.path.basename(file_list_path)} files",
                )
            ):
                # Optional: print progress periodically
                if (idx + 1) % 1000 == 0:
                    print(
                        f"Processed {idx + 1} files (last processed: {completed_path})"
                    )
        print(
            f"Finished processing all {len(files_paths)} files in {os.path.basename(file_list_path)}."
        )


if __name__ == "__main__":
    main()
