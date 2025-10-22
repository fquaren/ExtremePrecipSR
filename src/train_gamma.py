import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import random
from torch.utils.data import Sampler
import itertools

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
N = N_QUANTILES * 3
PATCH_SIZE = config["PATCH_SIZE"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]
VAL_METADATA_FILE = config["VAL_METADATA_FILE"]
TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
BATCH_SIZE = config.get("BATCH_SIZE", 16)
LEARNING_RATE = config.get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = config.get("WEIGHT_DECAY", 1e-5)
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 10)
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "Debugging")
LAMBDA_MONOTONICITY = config.get("LAMBDA_MONOTONICITY", 1.0)
LAMBDA_PLAUSIBILITY = config.get("LAMBDA_PLAUSIBILITY", 1.0)
PLAUSIBILITY_THRESHOLD = config.get(
    "PLAUSIBILITY_THRESHOLD", 12.0
)  # Heuristic value > 4*pi
LAMBDA_BOUND = config.get("LAMBDA_BOUND", 1.0)
PIXEL_AREA_KM2 = config.get("PIXEL_AREA_KM2", 1.0)


# --- Model Definition ---
class GammaPredictor(nn.Module):
    def __init__(
        self,
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        num_output_features_flat=N,
        n_quantiles=N_QUANTILES,
        activation_fn=F.gelu,
    ):
        super(GammaPredictor, self).__init__()
        self.n_quantiles, self.activation = n_quantiles, activation_fn
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_input_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_output_features_flat)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(-1, self.fc_input_size)
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.view(-1, 3, self.n_quantiles)
        return F.relu(x)


# --- Data Handling & Augmentation ---
class AddGaussianNoise(object):
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
        target_gamma = self.gamma_targets[idx]
        input_tensor = torch.from_numpy(original_precip).float().unsqueeze(0)
        target_gamma_tensor = torch.from_numpy(target_gamma).float()
        if self.augment:
            input_tensor = self.transform(input_tensor)
        return input_tensor, target_gamma_tensor


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


# New loss class for homoscedastic uncertainty
class ComponentWiseCDFLoss(nn.Module):
    def __init__(self, quantile_levels):
        super(ComponentWiseCDFLoss, self).__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )

    def forward(self, gamma_pred_3d, gamma_target_3d):
        abs_diff = torch.abs(gamma_pred_3d - gamma_target_3d)
        integrand = abs_diff * self.quantiles
        integral_per_component = torch.trapezoid(integrand, self.quantiles, dim=2)
        return (
            torch.mean(integral_per_component[:, 0]),
            torch.mean(integral_per_component[:, 1]),
            torch.mean(integral_per_component[:, 2]),
        )


# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda")  # if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir = os.path.join("experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment artifacts to: {output_dir}")
    try:
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
    except IOError as e:
        print(f"Error saving config file: {e}")

    # --- 1. Prepare Datasets ---
    train_dataset_full = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
        metadata_file=TRAIN_METADATA_FILE,
        augment=True,
    )
    val_dataset_full = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "validation"),
        metadata_file=VAL_METADATA_FILE,
        augment=False,
    )

    def subsample_dataset(dataset, fraction=0.1, seed=42):
        num_samples = int(fraction * len(dataset))
        if num_samples == 0 and len(dataset) > 0:
            num_samples = 1
        subset_indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )[:num_samples]
        return Subset(dataset, subset_indices)

    val_dataset = subsample_dataset(
        val_dataset_full, config["SUBSAMPLE_FRACTION"], seed=0
    )

    # --- Oversampling Strategy ---
    print("Stratifying dataset using MAX precipitation...")
    wet_event_metrics = [
        np.max(patch)
        for patch in train_dataset_full.original_patches
        if np.max(patch) > 1e-6
    ]
    if wet_event_metrics:
        extreme_threshold = np.percentile(wet_event_metrics, 95)
        print(
            f"Data-driven threshold (95th percentile of MAX precip): {extreme_threshold:.4f} mm/hr"
        )
    else:
        extreme_threshold = float("inf")
    indices_dry, indices_normal, indices_extreme = [], [], []
    for i, patch in enumerate(train_dataset_full.original_patches):
        metric = np.max(patch)
        if metric <= 1e-6:
            indices_dry.append(i)
        elif metric < extreme_threshold:
            indices_normal.append(i)
        else:
            indices_extreme.append(i)
    print(
        f"Stratification complete: {len(indices_dry)} Dry, {len(indices_normal)} Normal, {len(indices_extreme)} High."
    )
    batch_composition = {
        "dry": int(BATCH_SIZE / 4),
        "normal": int(BATCH_SIZE / 2),
        "extreme": int(BATCH_SIZE / 4),
    }
    stratified_sampler = StratifiedBatchSampler(
        indices_dry, indices_normal, indices_extreme, batch_composition
    )

    # --- Prepare DataLoaders ---
    train_loader = DataLoader(
        train_dataset_full,
        batch_sampler=stratified_sampler,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )

    # --- 3. Initialize Model, Optimizer, and Loss ---
    # Define learnable parameters for homoscedastic uncertainty
    log_var_A = nn.Parameter(torch.zeros((1,), device=device))
    log_var_P = nn.Parameter(torch.zeros((1,), device=device))
    log_var_CC = nn.Parameter(torch.zeros((1,), device=device))
    model = GammaPredictor(activation_fn=F.mish).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [log_var_A, log_var_P, log_var_CC],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS).to(device)
    LOSS_LAMBDA = config.get("LOSS_LAMBDA", 0.5)  # Weight for zero penalty vs main loss
    print(f"Using weighted average zero penalty with lambda = {LOSS_LAMBDA}")
    print(
        f"Using physics penalties: Mono={LAMBDA_MONOTONICITY}, Plaus={LAMBDA_PLAUSIBILITY}, Bound={LAMBDA_BOUND}"
    )

    log_file_path = os.path.join(output_dir, "training_log.csv")
    try:
        with open(log_file_path, "w") as log_file:
            # MODIFICATION: Add penalty columns to log header
            log_file.write(
                "epoch,train_loss_total,train_loss_main,train_loss_zero_penalty,"
                "train_penalty_mono,train_penalty_plaus,train_penalty_bound,"
                "val_loss_total,val_loss_main,val_loss_zero_penalty,"
                "val_penalty_mono,val_penalty_plaus,val_penalty_bound,"
                "sigma_A,sigma_P,sigma_CC\n"
            )
        print(f"Log file will be saved to {log_file_path}")
    except IOError as e:
        print(f"Error creating log file: {e}")
        exit()

    # --- 4. Training & Validation Loop ---
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss, running_main_loss, running_zero_penalty = 0.0, 0.0, 0.0
        running_penalty_mono, running_penalty_plaus, running_penalty_bound = (
            0.0,
            0.0,
            0.0,
        )

        for input_data, target_gamma in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"
        ):
            input_data, target_gamma = input_data.to(device), target_gamma.to(device)
            optimizer.zero_grad()
            predicted_gamma_3d = model(input_data)

            # --- Calculate Main Homoscedastic Loss ---
            loss_A, loss_P, loss_CC = criterion(predicted_gamma_3d, target_gamma)
            term_A = torch.exp(-log_var_A) * loss_A + log_var_A
            term_P = torch.exp(-log_var_P) * loss_P + log_var_P
            term_CC = torch.exp(-log_var_CC) * loss_CC + log_var_CC
            main_loss_homo = (
                term_A + term_P + term_CC
            ) * 0.5  # Mean over batch already in criterion

            # --- Calculate Zero Penalty ---
            is_dry_mask = input_data.sum(dim=(1, 2, 3)) <= 1e-6
            zero_penalty = torch.tensor(0.0, device=device)
            if is_dry_mask.sum() > 0:
                predictions_for_dry_inputs = predicted_gamma_3d[is_dry_mask]
                zero_penalty = torch.mean(torch.abs(predictions_for_dry_inputs))

            # --- Calculate Physics Penalties (on predictions) ---
            pred_A = predicted_gamma_3d[:, 0, :]  # Shape [B, NQ]
            pred_P = predicted_gamma_3d[:, 1, :]
            pred_CC = predicted_gamma_3d[:, 2, :]

            # 1. Monotonicity Penalty (Area)
            # Difference between adjacent elements along the quantile dim
            diff_A_mono = pred_A[:, 1:] - pred_A[:, :-1]
            # Penalize only positive differences (violations)
            penalty_mono = torch.mean(F.relu(diff_A_mono))

            # 2. Plausibility Penalty (Perimeter vs Area)
            # P^2 / (A + eps) should not be too small
            epsilon = 1e-6
            ratio_plaus = (pred_P**2) / (pred_A + epsilon)
            # Penalize if ratio is below threshold. Average over batch and quantiles.
            penalty_plaus = torch.mean(F.relu(PLAUSIBILITY_THRESHOLD - ratio_plaus))

            # 3. Upper Bound Penalty (CC vs Area)
            # CC should not exceed Area (assuming pixel area = 1)
            # Penalize if CC > A. Average over batch and quantiles.
            penalty_bound = torch.mean(F.relu(pred_CC - (pred_A / PIXEL_AREA_KM2)))

            # --- Combine ALL losses ---
            # Main homoscedastic loss + Weighted Zero Penalty + Weighted Physics Penalties
            total_loss = (
                main_loss_homo
                + LOSS_LAMBDA * zero_penalty
                + LAMBDA_MONOTONICITY * penalty_mono
                + LAMBDA_PLAUSIBILITY * penalty_plaus
                + LAMBDA_BOUND * penalty_bound
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- Accumulate losses for logging ---
            running_loss += total_loss.item()
            running_main_loss += (
                main_loss_homo.item()
            )  # Log the combined homoscedastic main loss
            running_zero_penalty += zero_penalty.item()
            running_penalty_mono += penalty_mono.item()
            running_penalty_plaus += penalty_plaus.item()
            running_penalty_bound += penalty_bound.item()

        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_main_loss = running_main_loss / num_batches if num_batches > 0 else 0
        avg_zero_penalty = running_zero_penalty / num_batches if num_batches > 0 else 0
        avg_penalty_mono = running_penalty_mono / num_batches if num_batches > 0 else 0
        avg_penalty_plaus = (
            running_penalty_plaus / num_batches if num_batches > 0 else 0
        )
        avg_penalty_bound = (
            running_penalty_bound / num_batches if num_batches > 0 else 0
        )

        print(
            f"Epoch {epoch+1} Train Loss: Total={avg_train_loss:.4f} (Main={avg_main_loss:.4f}, ZeroPen={avg_zero_penalty:.4f})"
        )
        print(
            f"             Penalties: Mono={avg_penalty_mono:.4f}, Plaus={avg_penalty_plaus:.4f}, Bound={avg_penalty_bound:.4f}"
        )

        model.eval()

        val_running_loss, val_running_main_loss, val_running_zero_penalty = (
            0.0,
            0.0,
            0.0,
        )
        (
            val_running_penalty_mono,
            val_running_penalty_plaus,
            val_running_penalty_bound,
        ) = (0.0, 0.0, 0.0)

        with torch.no_grad():
            for input_data, target_gamma in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"
            ):
                input_data, target_gamma = input_data.to(device), target_gamma.to(
                    device
                )
                predicted_gamma_3d = model(input_data)

                # --- Calculate Main Homoscedastic Loss ---
                loss_A, loss_P, loss_CC = criterion(predicted_gamma_3d, target_gamma)
                term_A = torch.exp(-log_var_A) * loss_A + log_var_A
                term_P = torch.exp(-log_var_P) * loss_P + log_var_P
                term_CC = torch.exp(-log_var_CC) * loss_CC + log_var_CC
                main_loss_homo = (term_A + term_P + term_CC) * 0.5

                # --- Calculate Zero Penalty ---
                is_dry_mask = input_data.sum(dim=(1, 2, 3)) <= 1e-6
                zero_penalty = torch.tensor(0.0, device=device)
                if is_dry_mask.sum() > 0:
                    predictions_for_dry_inputs = predicted_gamma_3d[is_dry_mask]
                    zero_penalty = torch.mean(torch.abs(predictions_for_dry_inputs))

                # --- MODIFICATION: Calculate Physics Penalties ---
                pred_A = predicted_gamma_3d[:, 0, :]
                pred_P = predicted_gamma_3d[:, 1, :]
                pred_CC = predicted_gamma_3d[:, 2, :]
                diff_A_mono = pred_A[:, 1:] - pred_A[:, :-1]
                penalty_mono = torch.mean(F.relu(diff_A_mono))
                epsilon = 1e-6
                ratio_plaus = (pred_P**2) / (pred_A + epsilon)
                penalty_plaus = torch.mean(F.relu(PLAUSIBILITY_THRESHOLD - ratio_plaus))
                penalty_bound = torch.mean(F.relu(pred_CC - pred_A))

                # --- Combine ALL losses ---
                total_loss = (
                    main_loss_homo
                    + LOSS_LAMBDA * zero_penalty
                    + LAMBDA_MONOTONICITY * penalty_mono
                    + LAMBDA_PLAUSIBILITY * penalty_plaus
                    + LAMBDA_BOUND * penalty_bound
                )

                # --- Accumulate validation losses ---
                val_running_loss += total_loss.item()
                val_running_main_loss += main_loss_homo.item()
                val_running_zero_penalty += zero_penalty.item()
                val_running_penalty_mono += penalty_mono.item()
                val_running_penalty_plaus += penalty_plaus.item()
                val_running_penalty_bound += penalty_bound.item()

        # Calculate average validation losses
        num_val_batches = len(val_loader)
        avg_val_loss = val_running_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_main_loss = (
            val_running_main_loss / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_zero_penalty = (
            val_running_zero_penalty / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_penalty_mono = (
            val_running_penalty_mono / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_penalty_plaus = (
            val_running_penalty_plaus / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_penalty_bound = (
            val_running_penalty_bound / num_val_batches if num_val_batches > 0 else 0
        )

        scheduler.step(avg_val_loss)

        sigma_A = torch.sqrt(torch.exp(log_var_A)).item()
        sigma_P = torch.sqrt(torch.exp(log_var_P)).item()
        sigma_CC = torch.sqrt(torch.exp(log_var_CC)).item()

        print(
            f"Epoch {epoch+1} Val Loss: Total={avg_val_loss:.4f} (Main={avg_val_main_loss:.4f}, ZeroPen={avg_val_zero_penalty:.4f}) | Sigmas: A={sigma_A:.3f}, P={sigma_P:.3f}, CC={sigma_CC:.3f}"
        )
        print(
            f"           Val Penalties: Mono={avg_val_penalty_mono:.4f}, Plaus={avg_val_penalty_plaus:.4f}, Bound={avg_val_penalty_bound:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "log_var_A": log_var_A,
                "log_var_P": log_var_P,
                "log_var_CC": log_var_CC,
            }
            model_save_path = os.path.join(output_dir, "best_model_checkpoint.pth")
            torch.save(checkpoint, model_save_path)
            print(f"Model checkpoint saved to {model_save_path}.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement."
                )
                break

        try:
            with open(log_file_path, "a") as log_file:
                # MODIFICATION: Add penalty values to log record
                log_file.write(
                    f"{epoch+1},{avg_train_loss:.6f},{avg_main_loss:.6f},{avg_zero_penalty:.6f},"
                    f"{avg_penalty_mono:.6f},{avg_penalty_plaus:.6f},{avg_penalty_bound:.6f},"
                    f"{avg_val_loss:.6f},{avg_val_main_loss:.6f},{avg_val_zero_penalty:.6f},"
                    f"{avg_val_penalty_mono:.6f},{avg_val_penalty_plaus:.6f},{avg_val_penalty_bound:.6f},"
                    f"{sigma_A:.6f},{sigma_P:.6f},{sigma_CC:.6f}\n"
                )
        except IOError as e:
            print(f"Error writing to log file: {e}")

    print("Training complete.")
