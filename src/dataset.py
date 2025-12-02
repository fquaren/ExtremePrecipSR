import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from loss import compute_gamma_matrix_for_image

# TODO: Consistency between outputs of SRDataset and PreprocessedNpzDataset


class SRDataset(Dataset):
    """
    Super-Resolution Dataset for UNet and DDPM Training.

    NOTE:
    This class loads 'original_precip.npz' and 'interpolated_original_precip.npz'.
    These files must contain data strictly scaled to [0, 1] (or Log-Transformed [0, 1]).
    Loading raw physical units (mm/h) here would break the Diffusion Signal-to-Noise ratio.
    """

    def __init__(
        self,
        preprocessed_data_dir,
        metadata_file,
        dem_patches_dir,
        dem_stats,
        split="train",
    ):
        self.preprocessed_data_dir = preprocessed_data_dir
        self.dem_patches_dir = dem_patches_dir
        self.dem_mean, self.dem_std = dem_stats
        self.split = split
        self.is_train = split == "train"

        # Load Metadata
        with open(metadata_file, "r") as f:
            lines = f.readlines()

        # Robust header check
        try:
            float(lines[0].split()[0])
            start_idx = 0
        except ValueError:
            start_idx = 1
        self.metadata = [line.strip().split() for line in lines[start_idx:]]

        # --- 1. Load Normalized Ground Truth (High Res) ---
        # Contains [0, 1] data
        precip_path = os.path.join(preprocessed_data_dir, split, "original_precip.npz")

        # --- 2. Load Normalized Interpolated Input (Low Res) ---
        # Contains [0, 1] data
        interp_path = os.path.join(
            preprocessed_data_dir, split, "interpolated_original_precip.npz"
        )

        # --- 3. Load Auxiliary Targets ---
        gamma_path = os.path.join(preprocessed_data_dir, split, "gamma_targets.npz")

        print(f"Loading {split} dataset components...")

        # Use mmap_mode='r' to keep RAM usage low on the GH200
        if not os.path.exists(precip_path):
            raise FileNotFoundError(f"Missing normalized ground truth: {precip_path}")
        self.original_patches = np.load(precip_path, mmap_mode="r")["data"]

        if not os.path.exists(interp_path):
            raise FileNotFoundError(f"Missing normalized interpolation: {interp_path}")
        self.interpolated_patches = np.load(interp_path, mmap_mode="r")["data"]

        if not os.path.exists(gamma_path):
            # Fail gracefully or warn if you haven't run Script 6 yet
            print(f"Warning: Gamma targets not found at {gamma_path}. Using zeros.")
            self.gamma_targets = np.zeros((len(self.metadata), 3), dtype=np.float32)
        else:
            self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]

        # --- OPTIMIZATION: Zero-Filtering ---
        if self.is_train:
            print("Filtering dry patches for training...")
            # We assume if the Max Intensity is 0 (or very close), the patch is dry.
            # Using the pre-loaded mmap is fast enough for this check.

            # Compute max along spatial dimensions (H, W) for every sample
            # This returns a shape (N,) array
            max_vals = np.max(self.original_patches, axis=(1, 2))

            # Keep indices where max > 0
            # Note: Since data is floats, use a small epsilon instead of strict 0
            wet_indices = np.where(max_vals > 1e-6)[0]

            # Filter metadata and arrays
            self.valid_indices = wet_indices
            print(
                f"Retained {len(self.valid_indices)}/{len(self.original_patches)} wet patches."
            )
        else:
            # Keep validation/test intact to evaluate the 'hard-coded zero' logic
            self.valid_indices = np.arange(len(self.original_patches))

        # --- Augmentations ---
        # Geometric transformations are safe for physical fields.
        self.geom_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ]
        )

    def __len__(self):
        # We now return the length of valid (wet) indices
        return len(self.valid_indices)

    def __getitem__(self, idx):

        real_idx = self.valid_indices[idx]

        # 1. Fetch Tensors (Already Normalized [0, 1])
        # Clone is necessary if using mmap to ensure we have a writable copy in memory
        target_img = self.original_patches[real_idx].copy()
        interp_img = self.interpolated_patches[real_idx].copy()

        # 2. Fetch DEM (Physical -> Normalize on the fly)
        meta_line = self.metadata[real_idx]
        y_coord, x_coord = int(meta_line[1]), int(meta_line[2])
        dem_filename = f"dem_patch_y{y_coord:04d}_x{x_coord:04d}.npy"
        dem_path = os.path.join(self.dem_patches_dir, dem_filename)

        try:
            # Load DEM and normalize z-score: (x - u) / sigma
            dem_patch = np.load(dem_path)
            dem_patch = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        except FileNotFoundError:
            # Fallback for missing DEMs
            dem_patch = np.zeros_like(target_img)

        # 3. Convert to Tensors [C, H, W]
        target_tensor = torch.from_numpy(target_img).float().unsqueeze(0)
        interp_tensor = torch.from_numpy(interp_img).float().unsqueeze(0)
        dem_tensor = torch.from_numpy(dem_patch).float().unsqueeze(0)

        # 4. Construct Conditioning Stack
        # Channel 0: Interpolated LR
        # Channel 1: DEM
        input_stack = torch.cat([interp_tensor, dem_tensor], dim=0)

        # 5. Apply Synchronized Augmentations (Train only)
        if self.is_train:
            # We must apply the exact same geometric transform to Input and Target
            state = torch.get_rng_state()
            input_stack = self.geom_transform(input_stack)

            torch.set_rng_state(state)
            target_tensor = self.geom_transform(target_tensor)

        # 6. Prepare Gamma Targets
        # These are likely raw values, so we apply log1p to compress dynamic range
        gamma_phys = self.gamma_targets[real_idx]
        target_gamma_tensor = torch.from_numpy(np.log1p(gamma_phys)).float()

        return input_stack, target_tensor, target_gamma_tensor


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


class PairedInterpolationDataset(Dataset):
    def __init__(
        self,
        preprocessed_data_dir,
        metadata_file,
        quantile_levels,
        pixel_size_km,
        augment=False,
        noise_std=0.05,
        mixup_alpha=0.4,
        mixup_prob=0.8,
        use_physics_consistent_mixup=True,
    ):
        self.quantile_levels = quantile_levels
        self.pixel_size_km = pixel_size_km
        self.use_physics_consistent_mixup = use_physics_consistent_mixup

        # --- Metadata Loading ---
        with open(metadata_file, "r") as f:
            lines = f.readlines()
        try:
            float(lines[0].split()[0])
            start_idx = 0
        except ValueError:
            start_idx = 1
        self.metadata = [line.strip().split() for line in lines[start_idx:]]

        # --- Load Data (Memory Mapped) ---
        self.real_patches = np.load(
            os.path.join(preprocessed_data_dir, "physical_precip.npz"), mmap_mode="r"
        )["data"]
        self.real_targets = np.load(
            os.path.join(preprocessed_data_dir, "gamma_targets.npz"), mmap_mode="r"
        )["data"]
        self.interp_patches = np.load(
            os.path.join(preprocessed_data_dir, "interpolated_physical_precip.npz"),
            mmap_mode="r",
        )["data"]

        # Only load interpolated targets if we rely on linear approximation (Option B)
        if not self.use_physics_consistent_mixup:
            self.interp_targets = np.load(
                os.path.join(preprocessed_data_dir, "gamma_targets_interpolated.npz"),
                mmap_mode="r",
            )["data"]

        self.augment = augment
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        # Augmentation Pipeline
        if self.augment:
            self.spatial_transform = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomChoice(
                        [T.RandomRotation([d, d]) for d in [0, 90, 180, 270]]
                    ),
                ]
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Load Data
        real_img = torch.from_numpy(self.real_patches[idx]).float().unsqueeze(0)
        interp_img = torch.from_numpy(self.interp_patches[idx]).float().unsqueeze(0)

        # 2. Augmentations (Spatial + Noise)
        if self.augment:
            # Spatial: Transform both identically
            combined = torch.cat([real_img, interp_img], dim=0)
            combined = self.spatial_transform(combined)
            real_img = combined[0:1]
            interp_img = combined[1:2]

            # Noise: Only on Interpolated
            noise = torch.randn_like(interp_img) * self.noise_std
            interp_img_noisy = torch.clamp(interp_img + noise, min=0.0)
        else:
            interp_img_noisy = interp_img

        # 3. MixUp Logic
        mixed_input = real_img
        mixed_target_phys = None

        if self.augment and (torch.rand(1).item() < self.mixup_prob):
            lambda_val = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_input = lambda_val * real_img + (1 - lambda_val) * interp_img_noisy

            # --- CRITICAL BRANCH ---
            if self.use_physics_consistent_mixup:
                # OPTION A: Compute TRUE topology of the mixed mess
                # Using the imported function from loss.py
                img_np = mixed_input.squeeze(0).numpy()

                gamma_matrix = compute_gamma_matrix_for_image(
                    img_np, self.quantile_levels, self.pixel_size_km
                )
                mixed_target_phys = torch.from_numpy(gamma_matrix).float()
            else:
                # OPTION B: Linear Approximation
                real_t = torch.from_numpy(self.real_targets[idx]).float()
                interp_t = torch.from_numpy(self.interp_targets[idx]).float()
                mixed_target_phys = lambda_val * real_t + (1 - lambda_val) * interp_t

        else:
            # No Mixup -> Just Real Data
            mixed_input = real_img
            mixed_target_phys = torch.from_numpy(self.real_targets[idx]).float()

        # Final Log Transform
        log_target_gamma = torch.log1p(mixed_target_phys)

        return mixed_input, log_target_gamma, real_img, mixed_target_phys
