import os
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from scipy.ndimage import zoom


config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PATCH_SIZE = config["PATCH_SIZE"]
DOWNSCALING_FACTOR = config["DOWNSCALING_FACTOR"]
DECLUTTER_THRESHOLD = config["DECLUTTER_THRESHOLD"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]


# Preprocessing functions (from your script)
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


class ZarrPatchDataset(Dataset):
    def __init__(
        self,
        metadata_file_path,
        dem_patch_dir,
        preprocessed_data_dir=PREPROCESSED_DATA_DIR,
        patch_size=PATCH_SIZE,
    ):
        """
        Args:
            metadata_file_path (str): Path to the .txt file containing patch metadata
                                      (e.g., 'train_patches_metadata.txt').
                                      Each line: 'YYYYMMDDHHMMSS,Y_COORD,X_COORD'
            dem_patch_dir (str): Directory where DEM patches are saved as .npy files.
            preprocessed_data_dir (str): Directory where preprocessed precipitation data
                                         (coarse, interpolated, original) are saved as .npy files.
            patch_size (int): Size of the square patches.
        """
        self.metadata = self._load_metadata(metadata_file_path)
        self.dem_patch_dir = dem_patch_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.patch_size = patch_size

    def _load_metadata(self, metadata_file_path):
        metadata = []
        with open(metadata_file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    timestamp_str, y_str, x_str = parts
                    metadata.append((timestamp_str, int(y_str), int(x_str)))
        print(f"Loaded {len(metadata)} metadata entries from {metadata_file_path}")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        timestamp_str, y_start, x_start = self.metadata[idx]

        # Construct file paths for preprocessed precipitation data
        # The naming convention must match the preprocessing script
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
            self.preprocessed_data_dir, "original_precip", original_precip_filename
        )
        interpolated_precip_path = os.path.join(
            self.preprocessed_data_dir,
            "interpolated_precip",
            interpolated_precip_filename,
        )
        coarse_precip_path = os.path.join(
            self.preprocessed_data_dir, "coarse_precip", coarse_precip_filename
        )

        # Load preprocessed precipitation data
        output_precip = torch.from_numpy(np.load(original_precip_path)).float()
        input_precip = torch.from_numpy(np.load(interpolated_precip_path)).float()
        coarse_precip = torch.from_numpy(np.load(coarse_precip_path)).float()

        # Load DEM data
        dem_filename = f"dem_patch_y{y_start:04d}_x{x_start:04d}.npy"
        dem_path = os.path.join(self.dem_patch_dir, dem_filename)
        dem_patch = torch.from_numpy(np.load(dem_path)).float()
        dem_patch[torch.isnan(dem_patch)] = 0.0

        # Add channel dimension if not already present
        if input_precip.ndim == 2:
            input_precip = input_precip.unsqueeze(0)
        if output_precip.ndim == 2:
            output_precip = output_precip.unsqueeze(0)
        if coarse_precip.ndim == 2:
            coarse_precip = coarse_precip.unsqueeze(0)
        if dem_patch.ndim == 2:
            dem_patch = dem_patch.unsqueeze(0)

        return input_precip, coarse_precip, output_precip, dem_patch
