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

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrePrecipSR/config.yaml"
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
        self.n_quantiles = n_quantiles
        self.activation = activation_fn

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
        return x


# --- Custom Transform for Data Augmentation ---
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return (
            tensor
            + torch.randn(tensor.size(), device=tensor.device) * self.std
            + self.mean
        )

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


# --- Data Handling with Augmentation ---
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
            raise ValueError("Metadata count does not match precipitation patch count.")
        if self.original_patches.shape[0] != self.gamma_targets.shape[0]:
            raise ValueError("Precipitation patches and Gamma targets count mismatch.")
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


# --- CDF-Weighted Integral Loss Function ---
# class CDFWeightedIntegralLoss(nn.Module):
#     def __init__(self, quantile_levels):
#         super(CDFWeightedIntegralLoss, self).__init__()
#         self.register_buffer(
#             "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
#         )

#     def forward(self, gamma_pred_3d, gamma_target_3d):
#         abs_diff = torch.abs(gamma_pred_3d - gamma_target_3d)
#         integrand = abs_diff * self.quantiles
#         integral_per_component = torch.trapezoid(integrand, self.quantiles, dim=2)
#         total_integral_per_sample = torch.sum(integral_per_component, dim=1)
#         return torch.mean(total_integral_per_sample)


class ComponentWiseCDFLoss(nn.Module):
    def __init__(self, quantile_levels):
        super(ComponentWiseCDFLoss, self).__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )

    def forward(self, gamma_pred_3d, gamma_target_3d):
        abs_diff = torch.abs(gamma_pred_3d - gamma_target_3d)
        integrand = abs_diff * self.quantiles
        # Shape: (B, 3) -> each column is the integrated loss for A, P, or CC
        integral_per_component = torch.trapezoid(integrand, self.quantiles, dim=2)

        # Return the mean loss for each component separately
        return (
            torch.mean(integral_per_component[:, 0]),  # Loss for Area
            torch.mean(integral_per_component[:, 1]),  # Loss for Perimeter
            torch.mean(integral_per_component[:, 2]),  # Loss for CC
        )


# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a unique directory for this experiment run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir = os.path.join("experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment artifacts to: {output_dir}")

    # Save the config file for reproducibility
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

    train_dataset = subsample_dataset(train_dataset_full, config["SUBSAMPLE_FRACTION"])
    val_dataset = subsample_dataset(
        val_dataset_full, config["SUBSAMPLE_FRACTION"], seed=0
    )

    # --- 2. Prepare DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- 3. Initialize Model, Optimizer, and Loss ---
    # We learn the log variance for numerical stability.
    log_var_A = torch.zeros((1,), requires_grad=True, device=device)
    log_var_P = torch.zeros((1,), requires_grad=True, device=device)
    log_var_CC = torch.zeros((1,), requires_grad=True, device=device)

    model = GammaPredictor(activation_fn=F.mish).to(device)
    optimizer = torch.optim.Adam(
        model.parameters() + [log_var_A, log_var_P, log_var_CC],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # criterion = CDFWeightedIntegralLoss(quantile_levels=QUANTILE_LEVELS).to(device)
    criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS)
    lambda_zero_penalty = config.get("LAMBDA_ZERO_PENALTY", 10.0)
    print(f"Using zero penalty with lambda = {lambda_zero_penalty}")

    # Update log file path to be inside the experiment directory
    log_file_path = os.path.join(output_dir, "training_log.csv")
    try:
        with open(log_file_path, "w") as log_file:
            log_file.write(
                "epoch,train_loss_total,train_loss_main,train_loss_penalty,"
                "val_loss_total,val_loss_main,val_loss_penalty\n"
            )
        print(f"Log file will be saved to {log_file_path}")
    except IOError as e:
        print(f"Error creating log file: {e}")
        exit()

    # --- 4. Training & Validation Loop ---
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    print(f"Early stopping patience set to {EARLY_STOPPING_PATIENCE} epochs.")

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        running_loss, running_main_loss, running_penalty = 0.0, 0.0, 0.0
        for input_data, target_gamma in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"
        ):
            input_data, target_gamma = input_data.to(device), target_gamma.to(device)
            optimizer.zero_grad()
            predicted_gamma_3d = model(input_data)

            # 1. Get the individual loss for each component
            loss_A, loss_P, loss_CC = criterion(predicted_gamma_3d, target_gamma)
            # 2. Apply the homoscedastic uncertainty formula
            # Note: A more stable form is exp(-s)*L + s, where s = log(sigma^2)
            term_A = torch.exp(-log_var_A) * loss_A + log_var_A
            term_P = torch.exp(-log_var_P) * loss_P + log_var_P
            term_CC = torch.exp(-log_var_CC) * loss_CC + log_var_CC
            # The main loss
            main_loss = (term_A + term_P + term_CC) * 0.5

            is_dry_mask = input_data.sum(dim=(1, 2, 3)) <= 1e-6
            zero_penalty = torch.tensor(0.0, device=device)
            if is_dry_mask.sum() > 0:
                predictions_for_dry_inputs = predicted_gamma_3d[is_dry_mask]
                zero_penalty = torch.sum(torch.abs(predictions_for_dry_inputs))

            total_loss = main_loss + lambda_zero_penalty * zero_penalty
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()
            running_main_loss += main_loss.item()
            running_penalty += zero_penalty.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_main_loss = running_main_loss / len(train_loader)
        avg_penalty = running_penalty / len(train_loader)
        print(
            f"Epoch {epoch+1} Train Loss: Total={avg_train_loss:.4f} (Main={avg_main_loss:.4f}, Penalty={avg_penalty:.4f})"
        )

        # --- Validation ---
        model.eval()
        val_running_loss, val_running_main_loss, val_running_penalty = 0.0, 0.0, 0.0
        with torch.no_grad():
            for input_data, target_gamma in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"
            ):
                input_data, target_gamma = input_data.to(device), target_gamma.to(
                    device
                )
                predicted_gamma_3d = model(input_data)

                main_loss = criterion(predicted_gamma_3d, target_gamma)
                is_dry_mask = input_data.sum(dim=(1, 2, 3)) <= 1e-6
                zero_penalty = torch.tensor(0.0, device=device)
                if is_dry_mask.sum() > 0:
                    predictions_for_dry_inputs = predicted_gamma_3d[is_dry_mask]
                    zero_penalty = torch.sum(torch.abs(predictions_for_dry_inputs))

                total_loss = main_loss + lambda_zero_penalty * zero_penalty
                val_running_loss += total_loss.item()
                val_running_main_loss += main_loss.item()
                val_running_penalty += zero_penalty.item()

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_main_loss = val_running_main_loss / len(val_loader)
        avg_val_penalty = val_running_penalty / len(val_loader)

        scheduler.step(avg_val_loss)
        print(
            f"Epoch {epoch+1} Val Loss: Total={avg_val_loss:.4f} (Main={avg_val_main_loss:.4f}, Penalty={avg_val_penalty:.4f})"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Update model save path
            model_save_path = os.path.join(output_dir, "best_gamma_predictor_model.pth")
            torch.save(model.state_dict(), model_save_path)
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
                log_file.write(
                    f"{epoch+1},{avg_train_loss:.6f},{avg_main_loss:.6f},{avg_penalty:.6f},"
                    f"{avg_val_loss:.6f},{avg_val_main_loss:.6f},{avg_val_penalty:.6f}\n"
                )
        except IOError as e:
            print(f"Error writing to log file: {e}")

    print("Training complete.")
