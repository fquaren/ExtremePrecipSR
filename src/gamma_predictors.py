import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Load configuration
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Length of gamma
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N = len(QUANTILE_LEVELS) * 3
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config["PATCH_SIZE"]
PIXEL_SIZE_KM = config["PIXEL_SIZE_KM"]


class InputNormalization(nn.Module):
    """
    Scales physical input (mm/hr) to [0, 1] range for numerical stability
    inside the network.
    """

    def __init__(self, max_precip_value):
        super().__init__()
        # Register as a buffer so it's saved with the state_dict
        # but is not a learnable parameter.
        self.register_buffer(
            "scale_factor", torch.tensor(max_precip_value, dtype=torch.float32)
        )

    def forward(self, x):
        # Safety: Avoid division by zero if config is wrong
        return x / (self.scale_factor + 1e-8)


class GammaPredictorHierarchicalHardGated(nn.Module):
    def __init__(
        self,
        input_shape,
        n_quantiles,
        activation_fn=F.gelu,
        quantile_levels=[0.0],
        pixel_area_km2=1.0,
        max_precip_value=150.0,
    ):
        super(GammaPredictorHierarchicalHardGated, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        self.register_buffer(
            "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.pixel_area_km2 = pixel_area_km2

        # --- 1. Internal Normalizer ---
        self.normalizer = InputNormalization(max_precip_value)

        # --- 2. CNN Trunk ---
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(2, 2)

        self.fc_input_size = self._get_conv_output_size(input_shape)

        # --- 3. Gated Interaction Heads ---

        # Shared latent representation
        self.shared_fc = nn.Linear(self.fc_input_size, 256)

        # Branch A (Independent)
        self.fc_A = nn.Linear(256, 128)
        self.out_A = nn.Linear(128, self.n_quantiles)

        # Branch P (Dependent on A)
        self.fc_P = nn.Linear(256, 128)
        self.gate_A_to_P = nn.Linear(128, 128)  # Learns how A modifies P
        self.out_P = nn.Linear(128, self.n_quantiles)

        # Branch CC (Dependent on P)
        self.fc_CC = nn.Linear(256, 128)
        self.gate_P_to_CC = nn.Linear(128, 128)  # Learns how P modifies CC
        self.out_CC = nn.Linear(128, self.n_quantiles)

        self.dropout = nn.Dropout(0.3)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input_tensor = torch.rand(1, *shape)
            output = self._forward_conv(input_tensor)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x_phys):
        epsilon = 1e-6

        # --- A. Backbone ---
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = torch.flatten(x_conv, 1)

        # Shared dense layer
        shared = self.activation(self.shared_fc(x_flat))
        shared = self.dropout(shared)

        # --- B. Gated Forward Pass ---

        # 1. Path A
        feat_A = self.activation(self.fc_A(shared))
        raw_A_logits = self.out_A(self.dropout(feat_A))

        # 2. Path P (Gated by A)
        feat_P_raw = self.activation(self.fc_P(shared))
        # Gate: sigmoidal map from A's features [0, 1]
        gate_A = torch.sigmoid(self.gate_A_to_P(feat_A))
        # Apply Gate: Element-wise multiplication (Attention)
        feat_P_gated = feat_P_raw * gate_A
        raw_P_logits = self.out_P(self.dropout(feat_P_gated))

        # 3. Path CC (Gated by P)
        feat_CC_raw = self.activation(self.fc_CC(shared))
        # Gate: sigmoidal map from P's *gated* features
        gate_P = torch.sigmoid(self.gate_P_to_CC(feat_P_gated))
        feat_CC_gated = feat_CC_raw * gate_P
        raw_CC_logits = self.out_CC(self.dropout(feat_CC_gated))

        # --- C. Constraints (Same as before) ---
        with torch.no_grad():
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x_phys, nan=-1.0) >= threshold
            A_total = mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + epsilon

        probs_A = torch.softmax(raw_A_logits, dim=1)
        scaled_probs_A = probs_A * A_total
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )

        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        R_P = 1.0 + F.softplus(raw_P_logits)
        pred_P = P_min * R_P

        pred_CC_continuous = F.softplus(raw_CC_logits)
        pred_CC = (
            pred_CC_continuous if self.training else torch.round(pred_CC_continuous)
        )

        constrained_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)

        with torch.no_grad():
            is_dry_mask = x_phys.sum(dim=(1, 2, 3)) <= epsilon
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)

        return constrained_output * wet_factor


class GammaPredictorHierarchicalSoftGated(nn.Module):
    def __init__(
        self,
        input_shape=(1, 128, 128),
        n_quantiles=9,
        activation_fn=F.gelu,
        max_precip_value=150.0,
    ):
        super(GammaPredictorHierarchicalSoftGated, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn

        self.normalizer = InputNormalization(max_precip_value)

        # CNN Trunk
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(2, 2)

        self.fc_input_size = self._get_conv_output_size(input_shape)

        # Gated Heads Setup
        self.shared_fc = nn.Linear(self.fc_input_size, 256)

        self.fc_A = nn.Linear(256, 128)
        self.out_A = nn.Linear(128, self.n_quantiles)

        self.fc_P = nn.Linear(256, 128)
        self.gate_A_to_P = nn.Linear(128, 128)
        self.out_P = nn.Linear(128, self.n_quantiles)

        self.fc_CC = nn.Linear(256, 128)
        self.gate_P_to_CC = nn.Linear(128, 128)
        self.out_CC = nn.Linear(128, self.n_quantiles)

        self.dropout = nn.Dropout(0.5)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input_tensor = torch.rand(1, *shape)
            output = self._forward_conv(input_tensor)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x_phys):
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = torch.flatten(x_conv, 1)

        shared = self.activation(self.shared_fc(x_flat))
        shared = self.dropout(shared)

        # 1. Path A
        feat_A = self.activation(self.fc_A(shared))
        raw_A = self.out_A(self.dropout(feat_A))

        # 2. Path P (Gated by A)
        feat_P_raw = self.activation(self.fc_P(shared))
        gate_A = torch.sigmoid(self.gate_A_to_P(feat_A))
        feat_P_gated = feat_P_raw * gate_A
        raw_P = self.out_P(self.dropout(feat_P_gated))

        # 3. Path CC (Gated by P)
        feat_CC_raw = self.activation(self.fc_CC(shared))
        gate_P = torch.sigmoid(self.gate_P_to_CC(feat_P_gated))
        feat_CC_gated = feat_CC_raw * gate_P
        raw_CC = self.out_CC(self.dropout(feat_CC_gated))

        pred_A = F.softplus(raw_A)
        pred_P = F.softplus(raw_P)
        pred_CC = F.softplus(raw_CC)

        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)
        return final_output


class GammaPredictorSeparateHeadsHard(nn.Module):
    def __init__(
        self,
        input_shape,
        n_quantiles,
        activation_fn=F.gelu,
        quantile_levels=[0.0],
        pixel_area_km2=1.0,
        max_precip_value=150.0,
    ):
        super(GammaPredictorSeparateHeadsHard, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        self.register_buffer(
            "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.pixel_area_km2 = pixel_area_km2

        # --- 1. Internal Normalizer ---
        self.normalizer = InputNormalization(max_precip_value)

        # --- Shared CNN Trunk ---
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(2, 2)

        self.fc_input_size = self._get_conv_output_size(input_shape)

        # --- Separate Regression Heads ---
        self.head_A = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_P = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_CC = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            # Normalizer doesn't change shape, but we call it for correctness
            input = self.normalizer(input)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x_phys):
        epsilon = 1e-6

        # Normalize for the Trunk
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = x_conv.view(-1, self.fc_input_size)

        raw_A_logits = self.head_A(x_flat)
        raw_P_logits = self.head_P(x_flat)
        raw_CC_logits = self.head_CC(x_flat)

        # Physical constraints use x_phys
        with torch.no_grad():
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x_phys, nan=-1.0) >= threshold
            A_total = mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + epsilon

        probs_A = torch.softmax(raw_A_logits, dim=1)
        scaled_probs_A = probs_A * A_total
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )

        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        R_P = 1.0 + F.softplus(raw_P_logits)
        pred_P = P_min * R_P

        pred_CC = F.softplus(raw_CC_logits)

        constrained_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)

        with torch.no_grad():
            is_dry_mask = x_phys.sum(dim=(1, 2, 3)) <= epsilon
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)

        final_output = constrained_output * wet_factor
        return final_output


class GammaPredictorSeparateHeadsSoft(nn.Module):
    def __init__(
        self,
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        n_quantiles=N_QUANTILES,
        activation_fn=F.gelu,
        max_precip_value=150.0,  # <--- NEW
    ):
        super(GammaPredictorSeparateHeadsSoft, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn

        # --- 1. Internal Normalizer ---
        self.normalizer = InputNormalization(max_precip_value)

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(2, 2)

        self.fc_input_size = self._get_conv_output_size(input_shape)

        self.head_A = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_P = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_CC = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            input = self.normalizer(input)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x_phys):
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = x_conv.view(-1, self.fc_input_size)

        raw_A = self.head_A(x_flat)
        raw_P = self.head_P(x_flat)
        raw_CC = self.head_CC(x_flat)

        pred_A = F.softplus(raw_A)
        pred_P = F.softplus(raw_P)
        pred_CC = F.softplus(raw_CC)

        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)
        return final_output
