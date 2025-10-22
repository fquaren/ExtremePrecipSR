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


# --- Model Definition ---
class GammaPredictor(nn.Module):
    # This class must be identical to the one used for training.
    def __init__(
        self, input_shape, num_output_features_flat, n_quantiles, activation_fn
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


# Renamed and updated the metric to use learned sigmas for normalization
class NormalizedErrorMetric(nn.Module):
    def __init__(self, quantile_levels, sigmas):
        super(NormalizedErrorMetric, self).__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        # Use sigmas to create precision (1/variance) terms
        self.register_buffer("precision", 1 / (sigmas**2 + 1e-6))
        print(
            f"Evaluation metric using precisions (1/sigma^2): A={self.precision[0]:.4f}, P={self.precision[1]:.4f}, CC={self.precision[2]:.4f}"
        )

    def forward(self, gamma_pred_3d, gamma_target_3d):
        """Returns the normalized total error for each sample in the batch."""
        abs_diff = torch.abs(gamma_pred_3d - gamma_target_3d)
        integrand = abs_diff * self.quantiles
        # integral_per_component has shape [B, 3]
        integral_per_component = torch.trapezoid(integrand, self.quantiles, dim=2)

        # Normalize the error of each component by its learned precision
        normalized_error = integral_per_component * self.precision

        # Return the sum of normalized errors per-sample, shape: (B,)
        total_normalized_error = torch.sum(normalized_error, dim=1)
        return total_normalized_error


# --- Dataset Definition ---
class PreprocessedNpzDataset(Dataset):
    def __init__(self, preprocessed_data_dir, metadata_file):
        print(f"Loading data from {preprocessed_data_dir}...")
        with open(metadata_file, "r") as f:
            self.metadata = [line.strip().split(",") for line in f]
        precip_path = os.path.join(preprocessed_data_dir, "original_precip.npz")
        gamma_path = os.path.join(preprocessed_data_dir, "gamma_targets.npz")
        self.input_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.original_precip_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]
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
        target_gamma = self.gamma_targets[idx]
        original_precip = self.original_precip_patches[idx]
        input_tensor = torch.from_numpy(input_precip).float().unsqueeze(0)
        target_gamma_tensor = torch.from_numpy(target_gamma).float()
        original_precip_tensor = torch.from_numpy(original_precip).float().unsqueeze(0)
        return input_tensor, target_gamma_tensor, original_precip_tensor


# --- Plotting Functions (unchanged from your previous script) ---
def _plot_single_gamma_comparison(
    sample_idx,
    all_preds,
    all_targets,
    all_images,
    all_losses,
    quantiles,
    title_prefix,
    sub_folder,
    output_dir,
):
    pred_gamma = all_preds[sample_idx]
    target_gamma = all_targets[sample_idx]
    target_image = all_images[sample_idx]
    loss = all_losses[sample_idx]
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
        f"{title_prefix} | Sample {sample_idx} | Normalized Loss: {loss:.4f}",
        fontsize=16,
        y=1.03,
    )
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
    predictions,
    targets_gamma,
    target_images,
    losses,
    quantiles,
    output_dir,
    n_samples=5,
):
    print("\nGenerating plots for best and worst samples based on loss...")
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
        candidate_losses = losses[candidate_indices]
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
                predictions,
                targets_gamma,
                target_images,
                losses,
                quantiles,
                f"Best Sample #{rank+1}",
                group_name,
                output_dir,
            )
        print(f"Plotting {len(worst_in_group_indices)} worst samples...")
        for rank, sample_idx in enumerate(worst_in_group_indices):
            _plot_single_gamma_comparison(
                sample_idx,
                predictions,
                targets_gamma,
                target_images,
                losses,
                quantiles,
                f"Worst Sample #{rank+1}",
                group_name,
                output_dir,
            )


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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Training History", fontsize=16)
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
    ax1.set_ylabel("Loss")
    ax1.set_title("Total Training & Validation Loss Over Epochs")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")
    ax2.plot(df["epoch"], df["sigma_A"], "o-", label="Sigma A", color="green")
    ax2.plot(df["epoch"], df["sigma_P"], "o-", label="Sigma P", color="purple")
    ax2.plot(df["epoch"], df["sigma_CC"], "o-", label="Sigma CC", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learned Std. Dev. (σ)")
    ax2.set_title("Learned Uncertainty Parameters Over Epochs")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GammaPredictor(
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        num_output_features_flat=N,
        n_quantiles=N_QUANTILES,
        activation_fn=F.mish,
    ).to(device)

    # Load the entire checkpoint which now includes log_vars
    checkpoint_path = os.path.join(args.run_dir, "best_model_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Error: Checkpoint file not found in run directory: '{checkpoint_path}'"
        )

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load the log_vars to calculate the sigmas for the evaluation metric
    log_var_A = checkpoint["log_var_A"]
    log_var_P = checkpoint["log_var_P"]
    log_var_CC = checkpoint["log_var_CC"]

    model.eval()
    print("Model and uncertainty parameters loaded successfully.")

    # Calculate sigmas from the loaded log_vars
    sigma_A = torch.sqrt(torch.exp(log_var_A))
    sigma_P = torch.sqrt(torch.exp(log_var_P))
    sigma_CC = torch.sqrt(torch.exp(log_var_CC))
    sigmas = torch.cat([sigma_A, sigma_P, sigma_CC]).to(device)

    test_dataset = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "test"),
        metadata_file=TEST_METADATA_FILE,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loaded {len(test_dataset)} samples for evaluation.")

    # Instantiate the normalized error metric with the loaded sigmas
    evaluation_metric = NormalizedErrorMetric(
        quantile_levels=QUANTILE_LEVELS, sigmas=sigmas
    ).to(device)

    all_preds, all_targets_gamma, all_original_images, all_losses = [], [], [], []

    with torch.no_grad():
        for input_data, target_gamma, original_precip in tqdm(
            test_loader, desc="Generating predictions and calculating losses"
        ):
            input_data, target_gamma = input_data.to(device), target_gamma.to(device)
            predicted_gamma_3d = model(input_data)
            per_sample_losses = evaluation_metric(predicted_gamma_3d, target_gamma)

            all_losses.append(per_sample_losses.cpu().numpy())
            all_preds.append(predicted_gamma_3d.cpu().numpy())
            all_targets_gamma.append(target_gamma.cpu().numpy())
            all_original_images.append(original_precip.squeeze(1).cpu().numpy())

    all_preds, all_targets_gamma = np.concatenate(all_preds, axis=0), np.concatenate(
        all_targets_gamma, axis=0
    )
    all_original_images, all_losses = np.concatenate(
        all_original_images, axis=0
    ), np.concatenate(all_losses, axis=0)
    print(f"Generated predictions and losses for {all_preds.shape[0]} samples.")

    plot_gamma_performance_by_quantile(
        predictions=all_preds,
        targets_gamma=all_targets_gamma,
        target_images=all_original_images,
        losses=all_losses,
        quantiles=QUANTILE_LEVELS,
        output_dir=args.run_dir,
        n_samples=5,
    )

    log_file_path = os.path.join(args.run_dir, "training_log.csv")
    plot_training_log(log_file_path, args.run_dir)

    print("\n✅ Evaluation script finished.")
