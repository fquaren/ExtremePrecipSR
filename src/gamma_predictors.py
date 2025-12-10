import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

# --- Config ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
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
        pixel_area_km2=4.0,
        max_precip_value=150.0,
        q_embedding_dim=32,
    ):
        super(GammaPredictorHierarchicalHardGated, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        self.register_buffer(
            "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.pixel_area_km2 = pixel_area_km2
        self.q_emb = q_embedding_dim

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

        # --- 3. Shared Latent ---
        self.shared_fc = nn.Linear(self.fc_input_size, 256)
        self.dropout = nn.Dropout(0.3)

        # --- 4. Convolutional Heads (IQ-CGN) ---
        total_latent_size = self.n_quantiles * self.q_emb

        # Expansion
        self.expand_A = nn.Linear(256, total_latent_size)
        self.expand_P = nn.Linear(256, total_latent_size)
        self.expand_CC = nn.Linear(256, total_latent_size)

        # Mixing (Spectral Smoothing)
        self.mix_A = nn.Conv1d(self.q_emb, self.q_emb, kernel_size=3, padding=1)
        self.mix_P = nn.Conv1d(self.q_emb, self.q_emb, kernel_size=3, padding=1)
        self.mix_CC = nn.Conv1d(self.q_emb, self.q_emb, kernel_size=3, padding=1)

        # Output Heads
        self.head_A = nn.Conv1d(self.q_emb, 1, kernel_size=1)
        self.head_P = nn.Conv1d(self.q_emb, 1, kernel_size=1)
        self.head_CC = nn.Conv1d(self.q_emb, 1, kernel_size=1)

        # --- Gating Mechanisms ---

        # Gate A -> P
        self.gate_A_to_P = nn.Conv1d(self.q_emb, self.q_emb, kernel_size=1)

        # Gate (A + P) -> CC (MODIFIED)
        # Concatenates A and P features to inform CC
        self.gate_AP_to_CC = nn.Conv1d(self.q_emb * 2, self.q_emb, kernel_size=1)

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
        batch_size = x_phys.size(0)

        # --- A. Backbone ---
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = torch.flatten(x_conv, 1)

        shared = self.activation(self.shared_fc(x_flat))
        shared = self.dropout(shared)

        # Helper: Reshape to (Batch, Channels, Length)
        def to_sequence(tensor):
            return tensor.view(batch_size, self.n_quantiles, self.q_emb).permute(
                0, 2, 1
            )

        # --- B. Gated Convolutional Pass ---

        # 1. Path A
        raw_A_flat = self.expand_A(shared)
        feat_A_seq = to_sequence(raw_A_flat)
        feat_A_mixed = self.activation(self.mix_A(feat_A_seq))  # [B, q_emb, n_q]

        raw_A_logits = self.head_A(self.dropout(feat_A_mixed)).squeeze(1)

        # 2. Path P (Gated by A)
        raw_P_flat = self.expand_P(shared)
        feat_P_seq = to_sequence(raw_P_flat)
        feat_P_mixed = self.activation(self.mix_P(feat_P_seq))

        # Gate Calculation (A -> P)
        gate_A = torch.sigmoid(self.gate_A_to_P(feat_A_mixed))
        feat_P_gated = feat_P_mixed * gate_A

        raw_P_logits = self.head_P(self.dropout(feat_P_gated)).squeeze(1)

        # 3. Path CC (Gated by A + P)
        raw_CC_flat = self.expand_CC(shared)
        feat_CC_seq = to_sequence(raw_CC_flat)
        feat_CC_mixed = self.activation(self.mix_CC(feat_CC_seq))

        # --- FUSED GATING LOGIC ---
        # Concatenate A features and Gated P features
        feat_combined = torch.cat(
            [feat_A_mixed, feat_P_gated], dim=1
        )  # [B, 2*q_emb, n_q]

        # Calculate Gate
        gate_AP = torch.sigmoid(self.gate_AP_to_CC(feat_combined))

        feat_CC_gated = feat_CC_mixed * gate_AP
        raw_CC_logits = self.head_CC(self.dropout(feat_CC_gated)).squeeze(1)

        # --- C. Hard Constraints ---

        # 1. Area: Softmax + Cumsum (Monotonicity)
        with torch.no_grad():
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x_phys, nan=-1.0) >= threshold
            A_total = mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + epsilon

        probs_A = torch.softmax(raw_A_logits, dim=1)
        scaled_probs_A = probs_A * A_total

        # Enforce monotonicity: A(q_i) >= A(q_{i+1})
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )

        # 2. Perimeter: Geometric Lower Bound
        # P must be at least the perimeter of a circle with Area A
        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        # Network predicts deviation factor (R_P >= 1.0)
        R_P = 1.0 + F.softplus(raw_P_logits)
        pred_P = P_min * R_P

        # 3. CC: Rounding for inference
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
        q_embedding_dim=32,
    ):
        super(GammaPredictorHierarchicalSoftGated, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        self.q_emb = q_embedding_dim

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

        # --- 3. Shared Latent ---
        self.shared_fc = nn.Linear(self.fc_input_size, 256)
        self.dropout = nn.Dropout(0.5)

        # --- 4. Convolutional Heads (IQ-CGN) ---
        total_latent_size = self.n_quantiles * self.q_emb

        # Expansion
        self.expand_A = nn.Linear(256, total_latent_size)
        self.expand_P = nn.Linear(256, total_latent_size)
        self.expand_CC = nn.Linear(256, total_latent_size)

        # Mixing (Spectral Smoothing)
        self.mix_A = nn.Conv1d(self.q_emb, self.q_emb, kernel_size=3, padding=1)
        self.mix_P = nn.Conv1d(self.q_emb, self.q_emb, kernel_size=3, padding=1)
        self.mix_CC = nn.Conv1d(self.q_emb, self.q_emb, kernel_size=3, padding=1)

        # Output Projection
        self.head_A = nn.Conv1d(self.q_emb, 1, kernel_size=1)
        self.head_P = nn.Conv1d(self.q_emb, 1, kernel_size=1)
        self.head_CC = nn.Conv1d(self.q_emb, 1, kernel_size=1)

        # --- Gating Mechanisms ---

        # Gate A -> P (Unchanged)
        # Input: Features of A (dim: q_emb) -> Output: Gate for P (dim: q_emb)
        self.gate_A_to_P = nn.Conv1d(self.q_emb, self.q_emb, kernel_size=1)

        # Gate (A + P) -> CC (MODIFIED)
        # We concatenate features of A and P, so input dim is 2 * q_emb.
        # The network learns how to weight A vs P for the CC gate.
        self.gate_AP_to_CC = nn.Conv1d(self.q_emb * 2, self.q_emb, kernel_size=1)

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
        batch_size = x_phys.size(0)

        # 1. Backbone
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = torch.flatten(x_conv, 1)

        shared = self.activation(self.shared_fc(x_flat))
        shared = self.dropout(shared)

        # Helper: Reshape to (Batch, Channels, Length)
        def to_sequence(tensor):
            return tensor.view(batch_size, self.n_quantiles, self.q_emb).permute(
                0, 2, 1
            )

        # 2. Path A
        raw_A_flat = self.expand_A(shared)
        feat_A_seq = to_sequence(raw_A_flat)
        feat_A_mixed = self.activation(self.mix_A(feat_A_seq))  # [B, q_emb, n_q]

        raw_A_logits = self.head_A(self.dropout(feat_A_mixed)).squeeze(1)

        # 3. Path P (Gated by A)
        raw_P_flat = self.expand_P(shared)
        feat_P_seq = to_sequence(raw_P_flat)
        feat_P_mixed = self.activation(self.mix_P(feat_P_seq))

        # Gate Calculation (A -> P)
        gate_A = torch.sigmoid(self.gate_A_to_P(feat_A_mixed))
        feat_P_gated = feat_P_mixed * gate_A

        raw_P_logits = self.head_P(self.dropout(feat_P_gated)).squeeze(1)

        # 4. Path CC (Gated by A AND P)
        raw_CC_flat = self.expand_CC(shared)
        feat_CC_seq = to_sequence(raw_CC_flat)
        feat_CC_mixed = self.activation(self.mix_CC(feat_CC_seq))

        # --- FUSED GATING LOGIC ---
        # Concatenate features along the channel dimension (dim=1)
        # feat_A_mixed: [B, q_emb, n_q]
        # feat_P_gated: [B, q_emb, n_q] (Using the gated P features passes the 'refined' info)
        feat_combined = torch.cat(
            [feat_A_mixed, feat_P_gated], dim=1
        )  # [B, 2*q_emb, n_q]

        # Generate Gate from combined info
        gate_AP = torch.sigmoid(self.gate_AP_to_CC(feat_combined))

        feat_CC_gated = feat_CC_mixed * gate_AP
        raw_CC_logits = self.head_CC(self.dropout(feat_CC_gated)).squeeze(1)

        # 5. Activations & Constraints
        pred_A = F.softplus(raw_A_logits)
        pred_P = F.softplus(raw_P_logits)

        pred_CC_continuous = F.softplus(raw_CC_logits)
        if self.training:
            pred_CC = pred_CC_continuous
        else:
            pred_CC = torch.round(pred_CC_continuous)

        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)

        # 6. Global Dry Mask
        with torch.no_grad():
            epsilon = 1e-6
            is_dry_mask = x_phys.sum(dim=(1, 2, 3)) <= epsilon
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)

        return final_output * wet_factor


class GammaPredictorSeparateHeadsHard(nn.Module):
    def __init__(
        self,
        input_shape,
        n_quantiles,
        activation_fn=F.gelu,
        quantile_levels=[0.0],
        pixel_area_km2=4.0,
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

        pred_CC_continuous = F.softplus(raw_CC_logits)

        if self.training:
            # TRAIN: Pure float. Can go down to 0.0.
            pred_CC = pred_CC_continuous
        else:
            # EVAL: Round to nearest integer.
            pred_CC = torch.round(pred_CC_continuous)

        constrained_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)

        with torch.no_grad():
            is_dry_mask = x_phys.sum(dim=(1, 2, 3)) <= epsilon
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)

        final_output = constrained_output * wet_factor
        return final_output


class GammaPredictorSeparateHeadsSoft(nn.Module):
    def __init__(
        self,
        input_shape=(1, 128, 128),
        n_quantiles=9,
        activation_fn=F.gelu,
        max_precip_value=150.0,
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

        # --- Separate Heads (Unchanged) ---
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
        # 1. Backbone
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = x_conv.view(-1, self.fc_input_size)

        # 2. Raw Logits
        raw_A = self.head_A(x_flat)
        raw_P = self.head_P(x_flat)
        raw_CC = self.head_CC(x_flat)

        # 3. Activations
        pred_A = F.softplus(raw_A)
        pred_P = F.softplus(raw_P)

        # --- CC Handling (Scientific Logic) ---
        pred_CC_continuous = F.softplus(raw_CC)

        if self.training:
            # Differentiable: [0, inf)
            pred_CC = pred_CC_continuous
        else:
            # Interpretable: Integers {0, 1, 2...}
            pred_CC = torch.round(pred_CC_continuous)

        # 4. Stack
        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)

        # 5. Apply Global Dry Mask
        # Even in "Soft" mode, if the input is empty, output MUST be zero.
        # This helps the Zero Penalty converge instantly.
        with torch.no_grad():
            epsilon = 1e-6
            # Sum pixels to see if patch is empty
            is_dry_mask = x_phys.sum(dim=(1, 2, 3)) <= epsilon
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)

        return final_output * wet_factor
