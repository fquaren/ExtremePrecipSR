import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SRDataset(Dataset):
    """
    Super-Resolution Dataset for UNet and DDPM Training.
    """

    def __init__(
        self,
        preprocessed_data_dir,
        metadata_file,
        dem_patches_dir,
        dem_stats,
        split="train",
        subset_fraction=1.0,
    ):
        self.preprocessed_data_dir = preprocessed_data_dir
        self.dem_patches_dir = dem_patches_dir
        self.dem_mean, self.dem_std = dem_stats
        self.split = split
        self.is_train = split == "train"

        # Load Metadata
        with open(metadata_file, "r") as f:
            lines = f.readlines()

        try:
            float(lines[0].split()[0])
            start_idx = 0
        except ValueError:
            start_idx = 1
        self.metadata = [line.strip().split() for line in lines[start_idx:]]

        precip_path = os.path.join(preprocessed_data_dir, split, "original_precip.npz")
        interp_path = os.path.join(
            preprocessed_data_dir, split, "interpolated_original_precip.npz"
        )
        gamma_path = os.path.join(preprocessed_data_dir, split, "gamma_targets.npz")

        print(f"Loading {split} dataset components...")

        if not os.path.exists(precip_path):
            raise FileNotFoundError(f"Missing normalized ground truth: {precip_path}")
        self.original_patches = np.load(precip_path, mmap_mode="r")["data"]

        if not os.path.exists(interp_path):
            raise FileNotFoundError(f"Missing normalized interpolation: {interp_path}")
        self.interpolated_patches = np.load(interp_path, mmap_mode="r")["data"]

        if not os.path.exists(gamma_path):
            print(f"Warning: Gamma targets not found at {gamma_path}. Using zeros.")
            self.gamma_targets = np.zeros((len(self.metadata), 3), dtype=np.float32)
        else:
            self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]

        # --- OPTIMIZATION: Zero-Filtering (Modified) ---
        if self.is_train:
            print("Filtering patches...")
            max_vals = np.max(self.original_patches, axis=(1, 2))
            wet_indices = np.where(max_vals > 1e-6)[0]
            dry_indices = np.where(max_vals <= 1e-6)[0]

            # Keep ALL wet patches
            # Keep randomly selected dry patches (e.g., 20% ratio)
            n_dry_to_keep = int(len(wet_indices) * 0.2)

            if len(dry_indices) > n_dry_to_keep:
                keep_dry = np.random.choice(
                    dry_indices, size=n_dry_to_keep, replace=False
                )
            else:
                keep_dry = dry_indices

            self.valid_indices = np.concatenate([wet_indices, keep_dry])
            np.random.shuffle(self.valid_indices)
            print(
                f"Dataset Balanced (Pre-subset): {len(wet_indices)} Wet, {len(keep_dry)} Dry."
            )

        else:
            # For validation/test, we MUST keep everything to evaluate performance honestly.
            self.valid_indices = np.arange(len(self.original_patches))

        # --- SUBSET LOGIC ---
        # We apply this AFTER the wet/dry balance logic to ensure we are subsetting
        # the distribution intended for training.
        if 0.0 < subset_fraction < 1.0:
            total_samples = len(self.valid_indices)
            subset_size = int(total_samples * subset_fraction)
            if subset_size < 1:
                subset_size = 1  # Ensure at least one sample

            print(
                f"Subsetting {split} dataset to {subset_fraction*100:.1f}% ({subset_size}/{total_samples} samples)."
            )

            # Use a fixed seed for subsetting to ensure runs are comparable
            rng = np.random.default_rng(seed=42)
            self.valid_indices = rng.choice(
                self.valid_indices, size=subset_size, replace=False
            )
        elif subset_fraction <= 0.0 or subset_fraction > 1.0:
            raise ValueError(
                f"subset_fraction must be in (0, 1]. Received {subset_fraction}"
            )

        self.geom_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # NOTE: This index is 'virtual' (0 to len(valid_indices))
        # We map it to the 'real' index in the .npz files
        real_idx = self.valid_indices[idx]

        # 1. Fetch Tensors
        target_img = self.original_patches[real_idx].copy()
        interp_img = self.interpolated_patches[real_idx].copy()

        # 2. Fetch DEM
        meta_line = self.metadata[real_idx]
        y_coord, x_coord = int(meta_line[1]), int(meta_line[2])
        dem_filename = f"dem_patch_y{y_coord:04d}_x{x_coord:04d}.npy"
        dem_path = os.path.join(self.dem_patches_dir, dem_filename)

        try:
            dem_patch = np.load(dem_path)
            dem_patch = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        except FileNotFoundError:
            dem_patch = np.zeros_like(target_img)

        # 3. Convert to Tensors
        target_tensor = torch.from_numpy(target_img).float().unsqueeze(0)
        interp_tensor = torch.from_numpy(interp_img).float().unsqueeze(0)
        dem_tensor = torch.from_numpy(dem_patch).float().unsqueeze(0)

        # 4. Construct Stack
        input_stack = torch.cat([interp_tensor, dem_tensor], dim=0)

        # 5. Augmentations
        if self.is_train:
            state = torch.get_rng_state()
            input_stack = self.geom_transform(input_stack)
            torch.set_rng_state(state)
            target_tensor = self.geom_transform(target_tensor)

        # 6. Prepare Gamma Targets (Log Transform)
        gamma_phys = self.gamma_targets[real_idx]
        target_gamma_tensor = torch.from_numpy(np.log1p(gamma_phys)).float()

        return input_stack, target_tensor, target_gamma_tensor


class PrecomputedMixupDataset(Dataset):
    def __init__(
        self,
        preprocessed_data_dir,
        metadata_file,
        augment=True,
        include_original=True,
        include_mixup=True,
        subset_fraction=1.0,
    ):
        # Metadata loading (assumes metadata aligns with 'physical_precip.npz')
        with open(metadata_file, "r") as f:
            lines = f.readlines()
        try:
            float(lines[0].split()[0])
            start_idx = 0
        except ValueError:
            start_idx = 1
        self.metadata_raw = [line.strip().split() for line in lines[start_idx:]]

        self.data_sources = []
        self.target_sources = []

        # 1. Load Original Real Data
        if include_original:
            print("Loading Original Real Data...")
            self.data_sources.append(
                np.load(
                    os.path.join(preprocessed_data_dir, "physical_precip.npz"),
                    mmap_mode="r",
                )["data"]
            )
            self.target_sources.append(
                np.load(
                    os.path.join(
                        preprocessed_data_dir, "gamma_targets_persistence.npz"
                    ),
                    mmap_mode="r",
                )["data"]
            )

        # 2. Load Pre-computed MixUp Data
        self.include_mixup = include_mixup
        if self.include_mixup:
            print("Loading Pre-computed MixUp Data...")
            mix_p_path = os.path.join(
                preprocessed_data_dir, "mixup_augmented_precip.npz"
            )
            mix_t_path = os.path.join(
                preprocessed_data_dir, "mixup_augmented_targets_persistence.npz"
            )

            if os.path.exists(mix_p_path) and os.path.exists(mix_t_path):
                self.data_sources.append(np.load(mix_p_path, mmap_mode="r")["data"])
                self.target_sources.append(np.load(mix_t_path, mmap_mode="r")["data"])
            else:
                print(
                    f"Warning: MixUp files not found at {mix_p_path}. Training without them."
                )

        # Create indexing map
        self.cumulative_sizes = np.cumsum([len(d) for d in self.data_sources])
        self.total_len = self.cumulative_sizes[-1]

        # --- SUBSET LOGIC ---
        # For Mixup dataset, since we don't have an explicit 'valid_indices' list
        # mapping to files, we create a virtual index mapper.
        self.indices_map = np.arange(self.total_len)

        if 0.0 < subset_fraction < 1.0:
            subset_size = int(self.total_len * subset_fraction)
            if subset_size < 1:
                subset_size = 1
            print(
                f"Subsetting MixupDataset to {subset_fraction*100:.1f}% ({subset_size}/{self.total_len})."
            )
            rng = np.random.default_rng(seed=42)
            self.indices_map = rng.choice(
                self.indices_map, size=subset_size, replace=False
            )
        elif subset_fraction <= 0.0 or subset_fraction > 1.0:
            raise ValueError(
                f"subset_fraction must be in (0, 1]. Received {subset_fraction}"
            )

        self.augment = augment
        if self.augment:
            self.transform = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomChoice(
                        [T.RandomRotation([d, d]) for d in [0, 90, 180, 270]]
                    ),
                ]
            )

        print(f"Total Dataset Size: {len(self.indices_map)} samples.")

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, idx):
        # Map the virtual subset index to the real dataset index
        real_idx = self.indices_map[idx]

        # Resolve Index to Source
        source_idx = np.searchsorted(self.cumulative_sizes, real_idx, side="right")
        if source_idx == 0:
            local_idx = real_idx
        else:
            local_idx = real_idx - self.cumulative_sizes[source_idx - 1]

        patch = self.data_sources[source_idx][local_idx]
        target_phys = self.target_sources[source_idx][local_idx]

        # To Tensor
        input_tensor = torch.from_numpy(patch).float().unsqueeze(0)
        target_phys_tensor = torch.from_numpy(target_phys).float()

        # Augmentation (Spatial Only - No MixUp here)
        if self.augment:
            input_tensor = self.transform(input_tensor)

        # Log Transform for training
        log_target_gamma = torch.log1p(target_phys_tensor)

        # Return same signature as before
        return input_tensor, log_target_gamma, input_tensor, target_phys_tensor
