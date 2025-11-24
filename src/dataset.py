import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# TODO: Consistency between outputs of SRDataset and PreprocessedNpzDataset


# --- Data Handling & Augmentation (using log1p transform) ---
class AddGaussianNoise(object):
    """Adds Gaussian noise to a tensor."""

    def __init__(self, mean=0.0, std=0.01):
        self.std, self.mean = std, mean

    def __call__(self, tensor):
        return (
            tensor
            + torch.randn(tensor.size(), device=tensor.device) * self.std
            + self.mean
        )

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


class SRDataset(Dataset):
    """
    Loads data for the super-resolution task.
    - X: Interpolated low-res precip + DEM (2 channels)
    - Y_target: The original high-res precip (1 channel)
    - Y_gamma: The LOG-TRANSFORMED gamma targets for the surrogate loss
    """

    def __init__(
        self,
        preprocessed_data_dir,
        metadata_file,
        dem_patches_dir,
        dem_stats,
        split="train",
        noise_std=0.01,
    ):
        print(f"Loading {split} data from {preprocessed_data_dir}...")
        self.preprocessed_data_dir = preprocessed_data_dir
        self.dem_patches_dir = dem_patches_dir
        self.dem_mean, self.dem_std = dem_stats

        with open(metadata_file, "r") as f:
            lines = f.readlines()

        try:
            float(lines[0].split()[0])
            start_idx = 0
        except ValueError:
            print("Header detected in metadata file. Skipping first row.")
            start_idx = 1

        self.metadata = [line.strip().split() for line in lines[start_idx:]]

        # Load data via memory-mapping
        precip_path = os.path.join(preprocessed_data_dir, split, "physical_precip.npz")
        interp_path = os.path.join(
            preprocessed_data_dir, split, "interpolated_physical_precip.npz"
        )
        gamma_path = os.path.join(preprocessed_data_dir, split, "gamma_targets.npz")

        self.original_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.interpolated_patches = np.load(interp_path, mmap_mode="r")["data"]

        # Note: These are likely stored in physical space in the NPZ
        self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]

        self.is_train = split == "train"

        # Augmentations
        rotations = T.RandomChoice(
            [
                T.RandomRotation([0, 0]),
                T.RandomRotation([90, 90]),
                T.RandomRotation([180, 180]),
                T.RandomRotation([270, 270]),
            ]
        )
        self.geom_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                rotations,
            ]
        )
        self.noise_transform = T.RandomApply([AddGaussianNoise(0.0, noise_std)], p=0.2)

        if self.is_train:
            print("Data augmentation (flips, rotations, noise) is enabled.")

        # Sanity Checks
        if not (
            len(self.metadata)
            == self.original_patches.shape[0]
            == self.interpolated_patches.shape[0]
            == self.gamma_targets.shape[0]
        ):
            raise ValueError("Data array lengths mismatch.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Load precipitation data
        original_precip = self.original_patches[idx]
        interp_precip = self.interpolated_patches[idx]
        target_gamma_phys = self.gamma_targets[idx]  # Physical units

        # 2. Load corresponding DEM patch
        meta_line = self.metadata[idx]
        y_coord, x_coord = int(meta_line[1]), int(meta_line[2])
        dem_filename = f"dem_patch_y{y_coord:04d}_x{x_coord:04d}.npy"
        dem_path = os.path.join(self.dem_patches_dir, dem_filename)

        try:
            dem_patch = np.load(dem_path)
        except FileNotFoundError:
            dem_patch = np.zeros_like(original_precip)

        # 3. Normalize and convert to Tensors
        interp_tensor = torch.from_numpy(interp_precip).float().unsqueeze(0)
        target_tensor = torch.from_numpy(original_precip).float().unsqueeze(0)

        dem_patch_norm = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        dem_tensor = torch.from_numpy(dem_patch_norm).float().unsqueeze(0)

        # 4. Stack inputs
        input_stack = torch.cat([interp_tensor, dem_tensor], dim=0)

        # 5. Apply Augmentations
        if self.is_train:
            state = torch.get_rng_state()
            input_stack = self.geom_transform(input_stack)
            torch.set_rng_state(state)
            target_tensor = self.geom_transform(target_tensor)

            interp_aug, dem_aug = torch.chunk(input_stack, 2, dim=0)
            interp_aug = self.noise_transform(interp_aug)
            input_stack = torch.cat([interp_aug, dem_aug], dim=0)

        # 6. LOG TRANSFORM GAMMA TARGETS
        # We apply log1p (log(1+x)) to handle zeros and match emulator training
        target_gamma_log = np.log1p(target_gamma_phys)
        target_gamma_tensor = torch.from_numpy(target_gamma_log).float()

        return (
            input_stack,  # [2, H, W]
            target_tensor,  # [1, H, W]
            target_gamma_tensor,  # [3, NQ] (Log Space)
        )


# --- Emulator Training Dataset Class ---
class PreprocessedNpzDataset(Dataset):
    def __init__(
        self, preprocessed_data_dir, metadata_file, augment=False, noise_std=0.01
    ):
        print(f"Loading data from {preprocessed_data_dir}...")
        with open(metadata_file, "r") as f:
            lines = f.readlines()

        # Heuristic to detect header.
        # If the first line contains text headers (e.g. "max_precip") and
        # the second line contains numbers, skip the first.
        try:
            float(lines[0].split()[0])  # Try converting first token to float
            start_idx = 0
        except ValueError:
            print("Header detected in metadata file. Skipping first row.")
            start_idx = 1

        self.metadata = [line.strip().split() for line in lines[start_idx:]]

        precip_path = os.path.join(preprocessed_data_dir, "physical_precip.npz")
        gamma_path = os.path.join(
            preprocessed_data_dir, "gamma_targets_persistence.npz"
        )
        self.original_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]
        self.augment = augment

        if self.augment:
            rotations = T.RandomChoice(
                [
                    T.RandomRotation([0, 0]),
                    T.RandomRotation([90, 90]),
                    T.RandomRotation([180, 180]),
                    T.RandomRotation([270, 270]),
                ]
            )
            self.transform = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    rotations,
                    T.RandomApply([AddGaussianNoise(0.0, noise_std)], p=0.2),
                ]
            )
            print("Data augmentation is enabled for this dataset.")
        if len(self.metadata) != self.original_patches.shape[0]:
            raise ValueError("Metadata/patch count mismatch.")
        if self.original_patches.shape[0] != self.gamma_targets.shape[0]:
            raise ValueError("Precipitation/Gamma count mismatch.")
        print(f"Loaded {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        original_precip = self.original_patches[idx]
        target_gamma_phys = self.gamma_targets[idx]
        input_tensor = torch.from_numpy(original_precip).float().unsqueeze(0)
        original_precip_tensor = torch.from_numpy(original_precip).float().unsqueeze(0)
        target_gamma_phys_tensor = torch.from_numpy(target_gamma_phys).float()
        log_target_gamma_tensor = torch.log1p(
            target_gamma_phys_tensor
        )  # Apply log transform

        if self.augment:
            input_tensor = self.transform(input_tensor)

        return (
            input_tensor,
            log_target_gamma_tensor,
            original_precip_tensor,
            target_gamma_phys_tensor,
        )
