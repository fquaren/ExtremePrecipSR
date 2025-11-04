import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from torch.utils.data import Sampler
import itertools


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
    - Y_gamma: The gamma targets for the surrogate loss

    This class has been modified to:
    1. Include DEM data as a static input channel.
    2. Correctly apply geometric augmentations to both input and target.
    """

    def __init__(
        self,
        preprocessed_data_dir,
        metadata_file,
        dem_patches_dir,  # Path to the 'patches/dem/' directory
        dem_stats,  # A (mean, std) tuple for the DEM data
        split="train",
        noise_std=0.01,
    ):
        print(f"Loading {split} data from {preprocessed_data_dir}...")
        self.preprocessed_data_dir = preprocessed_data_dir
        self.dem_patches_dir = dem_patches_dir

        # DEM normalization stats
        self.dem_mean, self.dem_std = dem_stats

        with open(metadata_file, "r") as f:
            # Metadata format: "timestamp,y_start,x_start"
            self.metadata = [line.strip().split(",") for line in f]

        # Load data via memory-mapping
        precip_path = os.path.join(preprocessed_data_dir, split, "original_precip.npz")
        interp_path = os.path.join(
            preprocessed_data_dir, split, "interpolated_precip.npz"
        )
        gamma_path = os.path.join(preprocessed_data_dir, split, "gamma_targets.npz")

        self.original_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.interpolated_patches = np.load(interp_path, mmap_mode="r")["data"]
        self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]

        self.is_train = split == "train"

        # --- Define Augmentations ---
        # 1. Geometric transforms (must be applied to input AND target)
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

        # 2. Noise transform (applied ONLY to precip input)
        self.noise_transform = T.RandomApply([AddGaussianNoise(0.0, noise_std)], p=0.2)

        if self.is_train:
            print("Data augmentation (flips, rotations, noise) is enabled.")

        # --- Sanity Checks ---
        if not (
            len(self.metadata)
            == self.original_patches.shape[0]
            == self.interpolated_patches.shape[0]
            == self.gamma_targets.shape[0]
        ):
            raise ValueError(
                f"Data array lengths or metadata mismatch. "
                f"Metadata: {len(self.metadata)}, "
                f"Original: {self.original_patches.shape[0]}, "
                f"Interp: {self.interpolated_patches.shape[0]}, "
                f"Gamma: {self.gamma_targets.shape[0]}"
            )

        if self.original_patches.shape != self.interpolated_patches.shape:
            raise ValueError("Original/Interpolated patch shape mismatch.")
        print(f"Loaded {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Load precipitation data
        original_precip = self.original_patches[idx]
        interp_precip = self.interpolated_patches[idx]
        target_gamma = self.gamma_targets[idx]

        # 2. Load corresponding DEM patch
        meta_line = self.metadata[idx]
        y_coord, x_coord = int(meta_line[1]), int(meta_line[2])

        # Construct filename, e.g., dem_patch_y1408_x1280.npy
        dem_filename = f"dem_patch_y{y_coord:04d}_x{x_coord:04d}.npy"
        dem_path = os.path.join(self.dem_patches_dir, dem_filename)

        try:
            dem_patch = np.load(dem_path)
        except FileNotFoundError:
            print(f"Error: DEM file not found at {dem_path}")
            # Return empty/zero DEM as a fallback
            dem_patch = np.zeros_like(original_precip)

        # 3. Normalize and convert to Tensors
        interp_tensor = (
            torch.from_numpy(interp_precip).float().unsqueeze(0)
        )  # [1, H, W]
        target_tensor = (
            torch.from_numpy(original_precip).float().unsqueeze(0)
        )  # [1, H, W]

        # Normalize DEM
        dem_patch_norm = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        dem_tensor = torch.from_numpy(dem_patch_norm).float().unsqueeze(0)  # [1, H, W]

        # 4. Stack inputs
        # (Channel 0: Interp. Precip, Channel 1: DEM)
        input_stack = torch.cat([interp_tensor, dem_tensor], dim=0)  # [2, H, W]

        # 5. Apply Augmentations (if training)
        if self.is_train:
            # Apply geometric transforms to input and target identically
            state = torch.get_rng_state()
            input_stack = self.geom_transform(input_stack)
            torch.set_rng_state(state)  # Ensure same random transform
            target_tensor = self.geom_transform(target_tensor)

            # Unpack, apply noise *only* to precip, and re-stack
            interp_aug, dem_aug = torch.chunk(input_stack, 2, dim=0)
            interp_aug = self.noise_transform(interp_aug)
            input_stack = torch.cat([interp_aug, dem_aug], dim=0)

        # 6. Prepare outputs
        target_gamma_tensor = torch.from_numpy(target_gamma).float()

        return (
            input_stack,  # [2, H, W]
            target_tensor,  # [1, H, W]
            target_gamma_tensor,  # [3, NQ]
        )


class PreprocessedNpzDataset(Dataset):
    def __init__(
        self, preprocessed_data_dir, metadata_file, augment=False, noise_std=0.01
    ):
        print(f"Loading data from {preprocessed_data_dir}...")
        with open(metadata_file, "r") as f:
            self.metadata = [line.strip().split(",") for line in f]
        precip_path = os.path.join(preprocessed_data_dir, "original_precip.npz")
        gamma_path = os.path.join(preprocessed_data_dir, "gamma_targets.npz")
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


# --- Stratified Sampler ---
class StratifiedBatchSampler(Sampler):
    def __init__(self, indices_dry, indices_normal, indices_extreme, batch_composition):
        self.indices_dry, self.indices_normal, self.indices_extreme = (
            indices_dry,
            indices_normal,
            indices_extreme,
        )
        self.batch_composition = batch_composition
        self.batch_size = sum(batch_composition.values())
        if not self.indices_extreme or self.batch_composition.get("extreme", 0) == 0:
            self.num_batches = 0
        else:
            self.num_batches = (
                len(self.indices_extreme) // self.batch_composition["extreme"]
            )

    def __iter__(self):
        dry_iter = iter(
            itertools.cycle(random.sample(self.indices_dry, len(self.indices_dry)))
        )
        normal_iter = iter(
            itertools.cycle(
                random.sample(self.indices_normal, len(self.indices_normal))
            )
        )
        extreme_iter = iter(
            random.sample(self.indices_extreme, len(self.indices_extreme))
        )
        for _ in range(self.num_batches):
            batch = []
            try:
                batch.extend(
                    [
                        next(extreme_iter)
                        for _ in range(self.batch_composition["extreme"])
                    ]
                )
                batch.extend(
                    [next(normal_iter) for _ in range(self.batch_composition["normal"])]
                )
                batch.extend(
                    [next(dry_iter) for _ in range(self.batch_composition["dry"])]
                )
            except StopIteration:
                break
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
