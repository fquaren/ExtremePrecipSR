import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def calculate_monotonicity_penalty(pred_A_phys):
    diffs = pred_A_phys[:, :-1] - pred_A_phys[:, 1:]
    penalty = torch.mean(F.relu(-diffs), dim=1)  # Per-sample
    return penalty


def calculate_plausibility_penalty(pred_A_phys, pred_P_phys):
    epsilon = 1e-6
    P_min = torch.sqrt(4 * math.pi * (pred_A_phys + epsilon))
    penalty = torch.mean(F.relu(P_min - pred_P_phys), dim=1)  # Per-sample
    return penalty


def calculate_bound_penalty(pred_A_phys, pred_CC_phys, pixel_area_km2):
    epsilon = 1e-6
    CC_max = (pred_A_phys / pixel_area_km2) + epsilon
    penalty = torch.mean(F.relu(pred_CC_phys - CC_max), dim=1)  # Per-sample
    return penalty


def calculate_zero_penalty(input_data, predicted_gamma_phys):
    with torch.no_grad():
        is_dry_mask = input_data.sum(dim=(1, 2, 3)) <= 1e-6  # Shape [B]
    if not is_dry_mask.any():
        return torch.tensor(0.0, device=input_data.device)
    # Return per-sample penalty
    dry_predictions = predicted_gamma_phys[is_dry_mask]
    penalty = torch.mean(
        torch.abs(dry_predictions), dim=(1, 2)
    )  # Mean over components and quantiles

    # Create a full-batch tensor, zero for wet samples
    full_penalty = torch.zeros(input_data.shape[0], device=input_data.device)
    full_penalty[is_dry_mask] = penalty
    return full_penalty


# --- Loss Function ---
class ComponentWiseCDFLoss(nn.Module):
    def __init__(self, quantile_levels, reduction="mean"):
        super(ComponentWiseCDFLoss, self).__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.reduction = reduction

    def forward(self, gamma_pred_3d, gamma_target_3d):  # Expects LOG SPACE values
        abs_diff_log = torch.abs(gamma_pred_3d - gamma_target_3d)
        integrand = abs_diff_log * self.quantiles
        integral_per_component = torch.trapezoid(integrand, self.quantiles, dim=2)
        if self.reduction is None:
            return (
                integral_per_component[:, 0],
                integral_per_component[:, 1],
                integral_per_component[:, 2],
            )
        else:
            return (
                torch.mean(integral_per_component[:, 0]),
                torch.mean(integral_per_component[:, 1]),
                torch.mean(integral_per_component[:, 2]),
            )


# --- New Loss Metric for Evaluation ---
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

        main_loss = (
            (self.config.get("WEIGHT_A", 1.0) * loss_A)
            + (self.config.get("WEIGHT_P", 1.0) * loss_P)
            + (self.config.get("WEIGHT_CC", 1.0) * loss_CC)
        )

        total_loss = main_loss

        # 2. Add soft penalties if in that mode
        # These penalties are calculated PER-SAMPLE
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

            total_loss = (
                main_loss
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
            total_loss = (
                main_loss + self.config.get("LAMBDA_BOUND", 0.0) * penalty_bound
            )

        return total_loss
