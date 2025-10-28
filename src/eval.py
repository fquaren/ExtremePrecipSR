import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from torch.utils.data import Dataset
import argparse
import pandas as pd
import math  # Needed for hard constraint model
from sklearn.metrics import r2_score


# --- Model Definition ---
# MODIFICATION: Use the hard constraint model class (must match training script)
class GammaPredictorHardConstraints(nn.Module):
    def __init__(
        self,
        input_shape,  # e.g., (1, PATCH_SIZE, PATCH_SIZE)
        num_output_features_flat,
        n_quantiles,
        activation_fn=F.gelu,
        quantile_levels=[0.0],  # Default needed, loaded from config later
        pixel_area_km2=1.0,  # Default needed, loaded from config later
    ):
        super(GammaPredictorHardConstraints, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        # Keep a buffer for quantiles for calculating A_total
        self.register_buffer(
            "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.pixel_area_km2 = pixel_area_km2

        # --- Convolutional Body ---
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

        # --- FC Layers ---
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
        # --- Feature Extraction ---
        x_conv = self._forward_conv(x)
        x_flat = x_conv.view(-1, self.fc_input_size)
        x_fc = self.activation(self.fc1(x_flat))
        x_fc = self.dropout1(x_fc)
        x_fc = self.activation(self.fc2(x_fc))
        x_fc = self.dropout2(x_fc)
        raw_output = self.fc3(x_fc)  # Shape [B, 3 * NQ]

        # --- Split into Raw Outputs ---
        raw_A_logits = raw_output[
            :, 0 * self.n_quantiles : 1 * self.n_quantiles
        ]  # [B, NQ]
        raw_P_logits = raw_output[
            :, 1 * self.n_quantiles : 2 * self.n_quantiles
        ]  # [B, NQ]
        raw_CC_pred = raw_output[
            :, 2 * self.n_quantiles : 3 * self.n_quantiles
        ]  # [B, NQ]

        # --- Calculate A_total Directly from Input ---
        with torch.no_grad():
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x, nan=-1.0) >= threshold
            A_total = (
                mask.sum(dim=(2, 3), keepdim=True).float() * self.pixel_area_km2 + 1e-6
            )  # Shape [B, 1, 1, 1] -> [B, 1]
            A_total = A_total.squeeze()  # Make it [B]
            if A_total.dim() == 0:  # Handle batch size 1 case
                A_total = A_total.unsqueeze(0)
            A_total = A_total.unsqueeze(1)  # Make it [B, 1]

        # --- Constrain Area (Monotonicity) ---
        probs_A = torch.softmax(raw_A_logits, dim=1)  # [B, NQ]
        scaled_probs_A = probs_A * A_total  # Broadcasting [B, NQ] * [B, 1] -> [B, NQ]
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )  # [B, NQ]

        # --- Constrain Perimeter (Plausibility) ---
        epsilon = 1e-6
        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        P_excess = F.relu(raw_P_logits)  # Or F.softplus(raw_P_logits)
        pred_P = P_min + P_excess  # [B, NQ]

        # --- Constrain CC (Non-negativity) ---
        pred_CC = F.relu(raw_CC_pred)  # [B, NQ]

        # --- Stack Components Back Together ---
        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)  # Shape [B, 3, NQ]

        return final_output


# Evaluation Metric using learned sigmas for normalization (compares in LOG-SPACE)
class NormalizedErrorMetric(nn.Module):
    def __init__(self, quantile_levels, sigmas):
        super(NormalizedErrorMetric, self).__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.register_buffer("precision", 1 / (sigmas**2 + 1e-6))
        print(
            f"Evaluation metric (in log-space) using precisions (1/sigma^2): A={self.precision[0]:.4f}, P={self.precision[1]:.4f}, CC={self.precision[2]:.4f}"
        )

    def forward(self, log_gamma_pred_3d, log_gamma_target_3d):
        abs_diff_log = torch.abs(log_gamma_pred_3d - log_gamma_target_3d)
        integrand = abs_diff_log * self.quantiles
        integral_per_component = torch.trapezoid(integrand, self.quantiles, dim=2)
        normalized_error = integral_per_component * self.precision
        total_normalized_error = torch.sum(normalized_error, dim=1)
        return total_normalized_error


# --- Dataset Definition ---
# Dataset returns log_target and original physical target
class PreprocessedNpzDataset(Dataset):
    def __init__(self, preprocessed_data_dir, metadata_file):
        print(f"Loading data from {preprocessed_data_dir}...")
        with open(metadata_file, "r") as f:
            self.metadata = [line.strip().split(",") for line in f]
        precip_path = os.path.join(preprocessed_data_dir, "original_precip.npz")
        gamma_path = os.path.join(preprocessed_data_dir, "gamma_targets.npz")
        self.input_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.original_precip_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.gamma_targets = np.load(gamma_path, mmap_mode="r")[
            "data"
        ]  # Physical targets
        if not (
            len(self.metadata)
            == self.input_patches.shape[0]
            == self.gamma_targets.shape[0]
            == self.original_precip_patches.shape[0]
        ):
            raise ValueError("Data array lengths or metadata mismatch.")
        print(f"Loaded {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        input_precip = self.input_patches[idx]
        target_gamma_phys = self.gamma_targets[idx]  # Physical gamma
        original_precip = self.original_precip_patches[idx]
        input_tensor = torch.from_numpy(input_precip).float().unsqueeze(0)
        target_gamma_phys_tensor = torch.from_numpy(
            target_gamma_phys
        ).float()  # Physical gamma tensor
        original_precip_tensor = torch.from_numpy(original_precip).float().unsqueeze(0)
        # Log-transformed target for loss calculation consistency
        log_target_gamma_tensor = torch.log1p(target_gamma_phys_tensor)
        # Return all needed components
        return (
            input_tensor,
            log_target_gamma_tensor,
            original_precip_tensor,
            target_gamma_phys_tensor,
        )


# --- Plotting Functions ---
# Expect PHYSICAL SPACE values
def _plot_single_gamma_comparison(
    sample_idx,
    all_preds_phys,
    all_targets_phys,
    all_images,
    all_losses_log,
    quantiles,
    title_prefix,
    sub_folder,
    output_dir,
):
    pred_gamma = all_preds_phys[sample_idx]
    target_gamma = all_targets_phys[sample_idx]
    target_image = all_images[sample_idx]
    loss = all_losses_log[sample_idx]  # Loss calculated in log space
    mean_precip = np.mean(target_image)
    gamma_types = ["Area (km²)", "Perimeter (km)", "CCs"]
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, wspace=0.4)
    ax_img = fig.add_subplot(gs[0, 0])
    im = ax_img.imshow(target_image, cmap="Blues", origin="lower", vmin=0)
    ax_img.set_title(f"Target Image (Mean: {mean_precip:.2f})")
    fig.colorbar(im, ax=ax_img, shrink=0.7, label="Precipitation (mm/hr)")
    for j in range(3):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.plot(quantiles, target_gamma[j], "o-", label="Target", color="royalblue")
        ax.plot(quantiles, pred_gamma[j], "x--", label="Prediction", color="salmon")
        ax.set_title(gamma_types[j])
        ax.set_xlabel("Precip. Threshold (mm/hr)")
        ax.grid(True, linestyle="--", alpha=0.6)
        if j == 0:
            ax.legend()
    fig.suptitle(
        f"{title_prefix} | Sample {sample_idx} | Normalized Log-Space Loss: {loss:.4f}",
        fontsize=16,
        y=1.03,
    )  # Clarify loss is log-space
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", sub_folder)
    os.makedirs(plot_save_dir, exist_ok=True)
    save_path = os.path.join(
        plot_save_dir,
        f"{title_prefix.replace(' ', '_').lower()}_sample_{sample_idx}.png",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def plot_gamma_performance_by_quantile(
    predictions_phys,
    targets_gamma_phys,
    target_images,
    losses_log,
    quantiles,
    output_dir,
    n_samples=5,
):
    print(
        "\nGenerating plots for best and worst samples based on loss (calculated in log-space)..."
    )
    all_means = np.mean(target_images, axis=(1, 2))
    sorted_indices_by_mean = np.argsort(all_means)
    n_total = len(target_images)
    quantile_groups = {
        "Low_Precip (0-33%)": sorted_indices_by_mean[: int(n_total * 0.33)],
        "Mid_Precip (33-67%)": sorted_indices_by_mean[
            int(n_total * 0.33) : int(n_total * 0.67)
        ],
        "High_Precip (67-100%)": sorted_indices_by_mean[int(n_total * 0.67) :],
    }
    for group_name, candidate_indices in quantile_groups.items():
        print(f"\n--- Processing Group: {group_name} ---")
        if len(candidate_indices) == 0:
            continue
        candidate_losses = losses_log[candidate_indices]
        sorted_loss_indices_in_group = np.argsort(candidate_losses)
        best_in_group_indices = candidate_indices[
            sorted_loss_indices_in_group[:n_samples]
        ]
        worst_in_group_indices = candidate_indices[
            sorted_loss_indices_in_group[-n_samples:]
        ]
        print(f"Plotting {len(best_in_group_indices)} best samples...")
        for rank, sample_idx in enumerate(best_in_group_indices):
            _plot_single_gamma_comparison(
                sample_idx,
                predictions_phys,
                targets_gamma_phys,
                target_images,
                losses_log,
                quantiles,
                f"Best Sample #{rank+1}",
                group_name,
                output_dir,
            )
        print(f"Plotting {len(worst_in_group_indices)} worst samples...")
        for rank, sample_idx in enumerate(worst_in_group_indices):
            _plot_single_gamma_comparison(
                sample_idx,
                predictions_phys,
                targets_gamma_phys,
                target_images,
                losses_log,
                quantiles,
                f"Worst Sample #{rank+1}",
                group_name,
                output_dir,
            )


# MODIFICATION: Updated plot function for reduced number of penalties
def plot_training_log(log_path, output_dir):
    if not os.path.exists(log_path):
        print(
            f"\nWarning: Log file not found at {log_path}. Skipping training history plot."
        )
        return
    print("\nGenerating training history plot...")
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error reading log file with pandas: {e}. Skipping plot.")
        return

    # Check for expected columns
    required_cols = [
        "epoch",
        "train_loss_total",
        "val_loss_total",
        "sigma_A",
        "sigma_P",
        "sigma_CC",
        "train_penalty_bound",
        "val_penalty_bound",
    ]  # Removed mono, plaus
    if not all(col in df.columns for col in required_cols):
        print(
            "Warning: Log file columns mismatch (expected hard constraint log). Skipping training history plot."
        )
        # print(f"Missing: {[c for c in required_cols if c not in df.columns]}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Training History (Hard Constraints)", fontsize=16)

    # --- Plot 1: Total Loss ---
    ax1.plot(
        df["epoch"],
        df["train_loss_total"],
        "o-",
        label="Train Total Loss",
        color="royalblue",
    )
    ax1.plot(
        df["epoch"],
        df["val_loss_total"],
        "o-",
        label="Validation Total Loss",
        color="salmon",
    )
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Total Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")

    # --- Plot 2: Learned Sigmas ---
    ax2.plot(df["epoch"], df["sigma_A"], "o-", label="Sigma A", color="green")
    ax2.plot(df["epoch"], df["sigma_P"], "o-", label="Sigma P", color="purple")
    ax2.plot(df["epoch"], df["sigma_CC"], "o-", label="Sigma CC", color="orange")
    ax2.set_ylabel("Learned Std. Dev. (σ)")
    ax2.set_title("Learned Uncertainty Parameters")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")

    # --- Plot 3: Remaining Physics Penalty (Bound) ---
    ax3.plot(
        df["epoch"],
        df["train_penalty_bound"],
        "d--",
        label="Train Bound Pen.",
        color="wheat",
        alpha=0.8,
    )
    ax3.plot(
        df["epoch"],
        df["val_penalty_bound"],
        "d-",
        label="Val Bound Pen.",
        color="orange",
        alpha=0.8,
    )
    # You might also want to plot the zero penalty if it's informative
    ax3.plot(
        df["epoch"],
        df["train_loss_zero_penalty"],
        "p:",
        label="Train Zero Pen.",
        color="grey",
        alpha=0.6,
    )
    ax3.plot(
        df["epoch"],
        df["val_loss_zero_penalty"],
        "p:",
        label="Val Zero Pen.",
        color="black",
        alpha=0.6,
    )

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Penalty Value")
    ax3.set_title("Remaining Soft Penalty Terms")
    ax3.legend(ncol=2)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved training history plot to: {save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GammaPredictor model."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the timestamped experiment run directory.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"Error: Run directory not found at '{args.run_dir}'")
    print(f"Evaluating experiment from: {args.run_dir}")

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    N_QUANTILES = len(QUANTILE_LEVELS)
    N = N_QUANTILES * 3
    PATCH_SIZE = config["PATCH_SIZE"]
    PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
    TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
    BATCH_SIZE = config.get("BATCH_SIZE", 16)
    # MODIFICATION: Get pixel area needed for the hard constraint model init
    PIXEL_AREA_KM2 = config.get(
        "PIXEL_AREA_KM2", 1.0
    )  # Default if missing in old config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MODIFICATION: Instantiate the correct Hard Constraint model
    model = GammaPredictorHardConstraints(
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        num_output_features_flat=N,
        n_quantiles=N_QUANTILES,
        activation_fn=F.mish,  # Make sure this matches training
        quantile_levels=QUANTILE_LEVELS,
        pixel_area_km2=PIXEL_AREA_KM2,
    ).to(device)

    checkpoint_path = os.path.join(args.run_dir, "best_model_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Error: Checkpoint file not found: '{checkpoint_path}'"
        )

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    log_var_A = checkpoint["log_var_A"]
    log_var_P = checkpoint["log_var_P"]
    log_var_CC = checkpoint["log_var_CC"]
    model.eval()
    print("Model and uncertainty parameters loaded successfully.")

    sigma_A = torch.sqrt(torch.exp(log_var_A))
    sigma_P = torch.sqrt(torch.exp(log_var_P))
    sigma_CC = torch.sqrt(torch.exp(log_var_CC))
    sigmas = torch.cat([sigma_A, sigma_P, sigma_CC]).squeeze().to(device)

    # MODIFICATION: Dataset now returns log_target and original physical target
    test_dataset = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "test"),
        metadata_file=TEST_METADATA_FILE,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loaded {len(test_dataset)} samples for evaluation.")

    evaluation_metric = NormalizedErrorMetric(
        quantile_levels=QUANTILE_LEVELS, sigmas=sigmas
    ).to(device)

    # Store predictions and targets in PHYSICAL space for plotting, losses in LOG space
    all_preds_phys, all_targets_phys = [], []
    all_original_images, all_losses_log = [], []

    with torch.no_grad():
        # MODIFICATION: Update loop to handle 4 outputs from dataset
        for input_data, log_target_gamma, original_precip, target_gamma_phys in tqdm(
            test_loader, desc="Generating predictions and calculating losses"
        ):
            input_data, log_target_gamma = input_data.to(device), log_target_gamma.to(
                device
            )

            # Model prediction is in PHYSICAL space due to hard constraints
            predicted_gamma_phys = model(input_data)

            # Transform prediction to LOG-SPACE for loss calculation
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            # Calculate loss in LOG-SPACE
            per_sample_losses = evaluation_metric(predicted_gamma_log, log_target_gamma)

            # Store values
            all_losses_log.append(per_sample_losses.cpu().numpy())
            all_preds_phys.append(predicted_gamma_phys.cpu().numpy())
            all_targets_phys.append(
                target_gamma_phys.cpu().numpy()
            )  # Store original physical target
            all_original_images.append(original_precip.squeeze(1).cpu().numpy())

    # Concatenate results
    all_preds_phys = np.concatenate(all_preds_phys, axis=0)
    all_targets_phys = np.concatenate(all_targets_phys, axis=0)
    all_original_images = np.concatenate(all_original_images, axis=0)
    all_losses_log = np.concatenate(all_losses_log, axis=0)
    print(f"Generated predictions and losses for {all_preds_phys.shape[0]} samples.")

    # MODIFICATION: Calculate and print R^2 scores
    print("\nCalculating R^2 scores (coefficient of determination)...")
    # Reshape predictions and targets to [n_samples, n_features] for sklearn
    n_samples = all_preds_phys.shape[0]
    n_features = 3 * N_QUANTILES
    preds_flat = all_preds_phys.reshape(n_samples, n_features)
    targets_flat = all_targets_phys.reshape(n_samples, n_features)

    # Calculate R^2 for each feature (component at each quantile)
    r2_scores_raw = r2_score(targets_flat, preds_flat, multioutput="raw_values")

    # Reshape back to [3, NQ] for easier interpretation
    r2_scores_matrix = r2_scores_raw.reshape(3, N_QUANTILES)

    # Calculate and print average R^2 per component
    avg_r2_A = np.mean(r2_scores_matrix[0, :])
    avg_r2_P = np.mean(r2_scores_matrix[1, :])
    avg_r2_CC = np.mean(r2_scores_matrix[2, :])
    print(f"Average R^2 Score - Area:      {avg_r2_A:.4f}")
    print(f"Average R^2 Score - Perimeter: {avg_r2_P:.4f}")
    print(f"Average R^2 Score - CC:        {avg_r2_CC:.4f}")

    # Optionally, save all R^2 scores
    r2_save_path = os.path.join(args.run_dir, "r2_scores.npz")
    np.savez_compressed(
        r2_save_path, r2_matrix=r2_scores_matrix, quantiles=QUANTILE_LEVELS
    )
    print(f"Detailed R^2 scores saved to: {r2_save_path}")

    # --- Plotting ---
    plot_gamma_performance_by_quantile(
        predictions_phys=all_preds_phys,
        targets_gamma_phys=all_targets_phys,
        target_images=all_original_images,
        losses_log=all_losses_log,
        quantiles=QUANTILE_LEVELS,
        output_dir=args.run_dir,
        n_samples=5,
    )

    log_file_path = os.path.join(args.run_dir, "training_log.csv")
    plot_training_log(log_file_path, args.run_dir)

    print("\n✅ Evaluation script finished.")
