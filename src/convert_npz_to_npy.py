import numpy as np
import os
import sys


def convert_file(file_path):
    if not file_path.endswith(".npz"):
        print(f"Skipping {file_path}, not a .npz")
        return

    npy_path = file_path.replace(".npz", ".npy")
    if os.path.exists(npy_path):
        print(f"Skipping {file_path}, .npy file already exists.")
        return

    print(f"Loading {file_path}...")
    try:
        with np.load(file_path) as loader:
            data = loader["data"]
            print(f"  Loaded data with shape: {data.shape}")
            print(f"  Saving to {npy_path}...")
            np.save(npy_path, data)
            print(f"  Successfully saved {npy_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_npz_to_npy.py <data_split_directory>")
        print("Example: python convert_npz_to_npy.py /path/to/precip/train")
        sys.exit(1)

    data_dir = sys.argv[1]
    input_npz = os.path.join(data_dir, "physical_precip.npz")

    if os.path.exists(input_npz):
        convert_file(input_npz)
    else:
        print(f"File not found: {input_npz}")
