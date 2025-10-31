import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Utility functions for penalties
def calculate_monotonicity_penalty(pred_A):
    """
    Penalizes non-monotonic (increasing) values in the Area prediction.
    pred_A [B, NQ] should be monotonically decreasing.
    """
    diffs = pred_A[:, :-1] - pred_A[:, 1:]
    penalty = torch.mean(F.relu(-diffs), dim=1)  # Per-sample penalty
    return penalty


def calculate_p_min_penalty(pred_A, pred_P):
    """
    Penalizes predictions where P < P_min.
    P_min = sqrt(4 * pi * A)
    """
    epsilon = 1e-6
    # We detach pred_A: backprop penalty only through P head.
    P_min = torch.sqrt(4 * math.pi * (pred_A.detach() + epsilon))
    penalty = torch.mean(F.relu(P_min - pred_P), dim=1)  # Per-sample penalty
    return penalty


def calculate_zero_penalty(input_data, predicted_gamma_phys):
    """
    Penalizes non-zero predictions for dry (all-zero) input patches.
    """
    with torch.no_grad():
        is_dry_mask = input_data.sum(dim=(1, 2, 3)) <= 1e-6  # Shape [B]

    # Calculate penalty for all samples (will be 0 for wet ones)
    # Sum over components [3] and quantiles [NQ]
    total_prediction_sum = predicted_gamma_phys.sum(dim=(1, 2))

    # Only apply penalty where is_dry_mask is True
    penalty = total_prediction_sum * is_dry_mask.float()
    return penalty


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
class SimpleCDFLossMetric(nn.Module):
    """
    Calculates the per-sample loss, matching the simple loss
    (loss_A + loss_P + loss_CC) from train.py.
    """

    def __init__(self, quantile_levels):
        super(SimpleCDFLossMetric, self).__init__()
        # Use the base loss, but ensure it does *not* reduce (mean)
        self.criterion = ComponentWiseCDFLoss(
            quantile_levels=quantile_levels, reduction=None
        )

    def forward(self, log_gamma_pred_3d, log_gamma_target_3d):
        # criterion returns [B, 3] tensor of losses (A, P, CC)
        loss_A, loss_P, loss_CC = self.criterion(log_gamma_pred_3d, log_gamma_target_3d)

        # Return the sum of component losses for each sample
        # [B]
        total_loss_per_sample = loss_A + loss_P + loss_CC
        return total_loss_per_sample
