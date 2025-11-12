import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from skimage import measure, morphology
from scipy.ndimage import label
import numpy as np
from tqdm import tqdm
import yaml


config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
N_QUANTILES = len(config["QUANTILE_LEVELS"])


# --- Utililty functions ---


def calculate_monotonicity_penalty(pred_A_phys):
    diffs = pred_A_phys[:, :-1] - pred_A_phys[:, 1:]
    return torch.mean(F.relu(-diffs), dim=1)  # Shape [B]


def calculate_plausibility_penalty(pred_A_phys, pred_P_phys):
    epsilon = 1e-6
    P_min = torch.sqrt(4 * math.pi * (pred_A_phys + epsilon))
    return torch.mean(F.relu(P_min - pred_P_phys), dim=1)  # Shape [B]


def calculate_bound_penalty(pred_A_phys, pred_CC_phys, pixel_area_km2):
    epsilon = 1e-6
    CC_max = (pred_A_phys / pixel_area_km2) + epsilon
    return torch.mean(F.relu(pred_CC_phys - CC_max), dim=1)  # Shape [B]


def calculate_zero_penalty(input_data, predicted_gamma_phys):
    with torch.no_grad():
        is_dry_mask = input_data.sum(dim=(1, 2, 3)) <= 1e-6

    # Create a full-batch tensor for per-sample penalties
    full_penalty = torch.zeros(input_data.shape[0], device=input_data.device)
    if not is_dry_mask.any():
        return full_penalty

    dry_predictions = predicted_gamma_phys[is_dry_mask]
    # Calculate per-sample penalty
    penalty = torch.mean(
        torch.abs(dry_predictions), dim=(1, 2)
    )  # Shape [num_dry_samples]
    full_penalty[is_dry_mask] = penalty
    return full_penalty  # Shape [B]


# --- Add Analytical Gamma Functions ---
def compute_A_P_CC_single_threshold_numpy(prec_2d_np, threshold, pixel_size_km=1.0):
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold
    area_km2 = mask.sum() * (pixel_size_km**2)
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = sum(
        np.linalg.norm(np.diff(contour, axis=0), axis=1).sum() for contour in contours
    )
    perimeter_km = perimeter_pixels * pixel_size_km
    structure = morphology.disk(1)
    _, num_features = label(mask, structure=structure)
    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km):
    gamma_matrix = np.zeros((3, len(thresholds)), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix


# --- Add helper to estimate S_inv from the dataset ---
def estimate_s_inv_from_dataset(dataset, num_samples, device):
    print(f"Estimating S_inv from {num_samples} training samples...")
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    all_gamma_A, all_gamma_P, all_gamma_CC = [], [], []
    for idx in tqdm(indices, desc="Collecting gamma targets"):
        _, _, _, Y_gamma = dataset[idx]
        all_gamma_A.append(Y_gamma[0, :].numpy())
        all_gamma_P.append(Y_gamma[1, :].numpy())
        all_gamma_CC.append(Y_gamma[2, :].numpy())
    all_gamma_A_np = np.array(all_gamma_A)
    all_gamma_P_np = np.array(all_gamma_P)
    all_gamma_CC_np = np.array(all_gamma_CC)
    S_A = np.cov(all_gamma_A_np, rowvar=False) + np.eye(N_QUANTILES) * 1e-6
    S_P = np.cov(all_gamma_P_np, rowvar=False) + np.eye(N_QUANTILES) * 1e-6
    S_CC = np.cov(all_gamma_CC_np, rowvar=False) + np.eye(N_QUANTILES) * 1e-6
    S_A_inv = np.linalg.inv(S_A)
    S_P_inv = np.linalg.inv(S_P)
    S_CC_inv = np.linalg.inv(S_CC)
    print("S_inv estimation complete.")
    return (
        torch.from_numpy(S_A_inv).float().to(device),
        torch.from_numpy(S_P_inv).float().to(device),
        torch.from_numpy(S_CC_inv).float().to(device),
    )


# --- Loss Classes ---


class ComponentWiseCDFLoss(nn.Module):
    def __init__(self, quantile_levels):
        super(ComponentWiseCDFLoss, self).__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )

    def forward(self, gamma_pred_log, gamma_target_log):  # Expects LOG SPACE
        abs_diff_log = torch.abs(gamma_pred_log - gamma_target_log)
        integrand = abs_diff_log * self.quantiles
        # Return per-sample loss, not batch mean
        integral_per_component = torch.trapezoid(
            integrand, self.quantiles, dim=2
        )  # Shape [B, 3]
        return (
            integral_per_component[:, 0],
            integral_per_component[:, 1],
            integral_per_component[:, 2],
        )  # Shape [B], [B], [B]


class TotalErrorMetric(nn.Module):
    def __init__(self, quantile_levels, config):
        super(TotalErrorMetric, self).__init__()
        self.component_criterion = ComponentWiseCDFLoss(quantile_levels)
        self.config = config
        self.pixel_area_km2 = config.get("PIXEL_SIZE_KM", 1.0) ** 2

    def forward(self, input_data, predicted_gamma_phys, log_target_gamma):
        # 1. Calculate main loss in log space
        predicted_gamma_log = torch.log1p(predicted_gamma_phys)
        loss_A, loss_P, loss_CC = self.component_criterion(
            predicted_gamma_log, log_target_gamma
        )

        main_loss_per_sample = (
            (self.config.get("WEIGHT_A", 1.0) * loss_A)
            + (self.config.get("WEIGHT_P", 1.0) * loss_P)
            + (self.config.get("WEIGHT_CC", 1.0) * loss_CC)
        )

        total_loss_per_sample = main_loss_per_sample

        # 2. Add soft penalties if in that mode
        constraint_mode = self.config.get("CONSTRAINT_MODE", "hybrid")

        if constraint_mode == "soft":
            pred_A_phys = predicted_gamma_phys[:, 0, :]
            pred_P_phys = predicted_gamma_phys[:, 1, :]
            pred_CC_phys = predicted_gamma_phys[:, 2, :]

            penalty_zero = calculate_zero_penalty(input_data, predicted_gamma_phys)
            penalty_mono = calculate_monotonicity_penalty(pred_A_phys)
            penalty_plaus = calculate_plausibility_penalty(pred_A_phys, pred_P_phys)
            penalty_bound = calculate_bound_penalty(
                pred_A_phys, pred_CC_phys, self.pixel_area_km2
            )

            total_loss_per_sample = (
                main_loss_per_sample
                + self.config.get("LOSS_LAMBDA", 0.0) * penalty_zero
                + self.config.get("LAMBDA_MONOTONICITY", 0.0) * penalty_mono
                + self.config.get("LAMBDA_PLAUSIBILITY", 0.0) * penalty_plaus
                + self.config.get("LAMBDA_BOUND", 0.0) * penalty_bound
            )

        elif constraint_mode == "hybrid":
            pred_A_phys = predicted_gamma_phys[:, 0, :]
            pred_CC_phys = predicted_gamma_phys[:, 2, :]
            penalty_bound = calculate_bound_penalty(
                pred_A_phys, pred_CC_phys, self.pixel_area_km2
            )
            total_loss_per_sample = (
                main_loss_per_sample
                + self.config.get("LAMBDA_BOUND", 0.0) * penalty_bound
            )

        return total_loss_per_sample


# --- GeometricLoss class ---
class GeometricLossSeparate(nn.Module):
    def __init__(self, S_inv_tensors, reduction="mean"):
        """
        Initializes the Geometric (Mahalanobis) Loss module.

        Args:
            S_inv_tensors (tuple or list): A tuple/list of three tensors
                                           [S_A_inv, S_P_inv, S_CC_inv],
                                           representing the inverse covariance
                                           matrices for Area, Perimeter, and CCs.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean'. Default: 'mean'.
                             'none': returns per-sample losses.
                             'mean': returns the mean of per-sample losses.
        """
        super(GeometricLossSeparate, self).__init__()

        if not hasattr(S_inv_tensors, "__len__") or len(S_inv_tensors) != 3:
            raise ValueError("S_inv_tensors must be a sequence of 3 tensors.")

        self.register_buffer("S_A_inv", S_inv_tensors[0])
        self.register_buffer("S_P_inv", S_inv_tensors[1])
        self.register_buffer("S_CC_inv", S_inv_tensors[2])

        if reduction not in ["mean", "none"]:
            raise ValueError(
                f"Invalid reduction type: {reduction}. Must be 'mean' or 'none'."
            )
        self.reduction = reduction

    def forward(self, gamma_pred_3d, gamma_target_3d):
        """
        Computes the geometric loss.

        Args:
            gamma_pred_3d (torch.Tensor): Predictions, shape (B, 3, N_quantiles)
            gamma_target_3d (torch.Tensor): Targets, shape (B, 3, N_quantiles)

        Returns:
            torch.Tensor: The loss (scalar if reduction='mean',
                          tensor of shape (B,) if reduction='none').
        """
        # Decompose predictions and targets
        # Shapes go from (B, 3, Nq) -> (B, Nq)
        pred_A = gamma_pred_3d[:, 0, :]
        pred_P = gamma_pred_3d[:, 1, :]
        pred_CC = gamma_pred_3d[:, 2, :]

        target_A = gamma_target_3d[:, 0, :]
        target_P = gamma_target_3d[:, 1, :]
        target_CC = gamma_target_3d[:, 2, :]

        # Calculate difference vectors, shape (B, Nq)
        diff_A = pred_A - target_A
        diff_P = pred_P - target_P
        diff_CC = pred_CC - target_CC

        # Calculate squared Mahalanobis distance for each component
        # (diff @ S_inv) * diff -> (B, Nq) * (B, Nq) -> sum(dim=1) -> (B,)
        loss_A_sq = torch.sum((diff_A @ self.S_A_inv) * diff_A, dim=1)
        loss_P_sq = torch.sum((diff_P @ self.S_P_inv) * diff_P, dim=1)
        loss_CC_sq = torch.sum((diff_CC @ self.S_CC_inv) * diff_CC, dim=1)

        # Calculate the geometric distance (L2 norm of the squared distance)
        # Add epsilon for numerical stability during sqrt
        loss_A = torch.sqrt(loss_A_sq + 1e-8)
        loss_P = torch.sqrt(loss_P_sq + 1e-8)
        loss_CC = torch.sqrt(loss_CC_sq + 1e-8)

        # The total loss for each sample is the sum of component distances
        # Shape: (B,)
        per_sample_loss = loss_A + loss_P + loss_CC

        # Apply the specified reduction
        if self.reduction == "mean":
            return torch.mean(per_sample_loss)
        elif self.reduction == "none":
            return per_sample_loss

        # This line should not be reachable due to __init__ check
        return per_sample_loss
