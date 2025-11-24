import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- CONFIGURATION ---
# Adjust these paths if necessary, or use command line args
DEFAULT_PREPROCESSED_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/extremes/OPERA/patches/precip"
DEFAULT_METADATA_FILE = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/extremes/OPERA/train_patches_metadata.txt"


def compute_chunk_stats(indices, npy_path):
    """
    Worker function to compute max precip for a chunk of indices.
    Uses mmap to read disk directly without loading full file to RAM.
    """
    data_mmap = np.load(npy_path, mmap_mode="r")
    max_vals = []
    for i in indices:
        # Compute max of the single patch
        max_vals.append(np.max(data_mmap[i]))
    return max_vals


def main(data_dir, metadata_file, output_file, split_name="train"):
    print(f"--- Generating Stats for {split_name} ---")

    # 1. Load the existing metadata (to get timestamps/coords)
    print(f"Reading metadata: {metadata_file}")
    # Your file has no header, so we assign names manually
    df = pd.read_csv(metadata_file, header=None, names=["timestamp", "row", "col"])
    print(f"Found {len(df)} entries in metadata.")

    # 2. Locate the physical precip file
    npy_path = os.path.join(data_dir, split_name, "physical_precip.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Could not find precipitation data at: {npy_path}")

    # 3. Verify shapes
    data_mmap = np.load(npy_path, mmap_mode="r")
    print(f"Found .npy data with shape: {data_mmap.shape}")

    if len(df) != data_mmap.shape[0]:
        print(
            f"WARNING: Metadata length ({len(df)}) != Data length ({data_mmap.shape[0]})"
        )
        # Usually safer to truncate to the shorter one or error out.
        # Assuming 1-to-1 mapping for now.

    # 4. Parallel Computation of Max Precip
    num_samples = data_mmap.shape[0]
    num_workers = 16
    chunk_size = 10000

    indices = list(range(num_samples))
    # Create chunks
    chunks = [indices[i : i + chunk_size] for i in range(0, num_samples, chunk_size)]

    print(f"Computing statistics using {num_workers} workers...")

    pool_func = partial(compute_chunk_stats, npy_path=npy_path)

    results = []
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap(pool_func, chunks), total=len(chunks)):
            results.extend(res)

    # 5. Add to DataFrame and Save
    df["max_precip"] = results

    print(f"Saving enriched metadata to: {output_file}")
    # Save with header so pandas can read it easily next time
    df.to_csv(output_file, index=False, sep=" ")

    print("Done! Sample of new data:")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument(
        "--meta_dir",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/extremes/OPERA",
    )
    args = parser.parse_args()

    # List of splits to process
    splits = [
        ("train", "train_patches_metadata.txt"),
        ("validation", "val_patches_metadata.txt"),
        ("test", "test_patches_metadata.txt"),
    ]

    for split_name, meta_filename in splits:
        meta_path = os.path.join(args.meta_dir, meta_filename)

        # Check if file exists before trying
        if os.path.exists(meta_path):
            output_filename = meta_path.replace(".txt", "_with_stats.txt")
            try:
                main(args.base_dir, meta_path, output_filename, split_name=split_name)
            except Exception as e:
                print(f"Failed to process {split_name}: {e}")
        else:
            print(f"Skipping {split_name}: Metadata file not found at {meta_path}")
