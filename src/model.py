import torch
import torch.nn as nn


class SoftmaxConstraintLayer(nn.Module):
    """
    Numerically stable implementation of Softmax Constraint Layer (SmCL).
    """

    def __init__(self):
        super(SoftmaxConstraintLayer, self).__init__()

    def forward(self, logits, x_phys):
        """
        logits: (B, 1, H, W)
        x_phys: (B, 1, H, W)
        """
        # 1. Numerical Stability: Subtract max logit per sample/patch
        # We detach the max to avoid affecting gradients of the shift itself,
        # though mathematically it cancels out.
        logit_max = torch.amax(logits, dim=(2, 3), keepdim=True).detach()
        logits_shifted = logits - logit_max

        # 2. Numerator: exp(y_tilde - max)
        exp_y = torch.exp(logits_shifted)

        # 3. Denominator: Mean of the shifted exponential
        mean_exp_y = exp_y.mean(dim=(2, 3), keepdim=True)

        # 4. Constraint Target
        mean_x = x_phys.mean(dim=(2, 3), keepdim=True)

        # 5. Apply Constraint
        # The exp(max) factor mathematically cancels out in numerator/denominator ratio
        # Ratio = (exp(l-m) * e^m) / (mean(exp(l-m)) * e^m) = exp(l-m)/mean(exp(l-m))
        out_phys = exp_y * (mean_x / (mean_exp_y + 1e-8))

        return out_phys


class DoubleConv(nn.Module):
    """(CONV -> BN -> Mish) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Mish(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetSR(nn.Module):
    def __init__(self, in_channels, out_channels, n_features_base=64):
        super(UNetSR, self).__init__()

        # --- Encoder ---
        self.inc = DoubleConv(in_channels, n_features_base)
        self.down1 = nn.Sequential(
            nn.AvgPool2d(2), DoubleConv(n_features_base, n_features_base * 2)
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2), DoubleConv(n_features_base * 2, n_features_base * 4)
        )
        self.down3 = nn.Sequential(
            nn.AvgPool2d(2), DoubleConv(n_features_base * 4, n_features_base * 8)
        )
        self.down4 = nn.Sequential(
            nn.AvgPool2d(2), DoubleConv(n_features_base * 8, n_features_base * 16)
        )

        # --- Decoder ---
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(n_features_base * 16, n_features_base * 8, kernel_size=1),
        )
        self.conv1 = DoubleConv(
            n_features_base * 8 + n_features_base * 8, n_features_base * 8
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(n_features_base * 8, n_features_base * 4, kernel_size=1),
        )
        self.conv2 = DoubleConv(
            n_features_base * 4 + n_features_base * 4, n_features_base * 4
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(n_features_base * 4, n_features_base * 2, kernel_size=1),
        )
        self.conv3 = DoubleConv(
            n_features_base * 2 + n_features_base * 2, n_features_base * 2
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(n_features_base * 2, n_features_base, kernel_size=1),
        )
        self.conv4 = DoubleConv(n_features_base + n_features_base, n_features_base)

        # --- Heads ---

        # 1. Regression Head
        # Changed: Outputs 'logits' (tilde_y) for the SmCL, not the final value.
        # No residual connection here, the SmCL handles the mapping.
        self.head_reg = nn.Conv2d(n_features_base, out_channels, kernel_size=1)

        # 2. Constraint Layer (SmCL)
        self.smcl = SoftmaxConstraintLayer()

    def forward(self, x, x_phys_constraint):
        """
        x: Input stack (interpolated + DEM)
        x_phys_constraint: The physical values of the input precipitation
                           used for mass conservation.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        # Generate Logits (y_tilde)
        logits = self.head_reg(x)

        # Apply Softmax Constraint Layer
        # This returns physical units
        out_phys = self.smcl(logits, x_phys_constraint)

        return out_phys


# class DoubleConv(nn.Module):
#     """(CONV -> BN -> Mish) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.Mish(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.Mish(),
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class UNetSR(nn.Module):
#     def __init__(self, in_channels, out_channels, n_features_base=64):
#         super(UNetSR, self).__init__()

#         # --- Encoder ---
#         self.inc = DoubleConv(in_channels, n_features_base)
#         self.down1 = nn.Sequential(
#             nn.AvgPool2d(2), DoubleConv(n_features_base, n_features_base * 2)
#         )
#         self.down2 = nn.Sequential(
#             nn.AvgPool2d(2), DoubleConv(n_features_base * 2, n_features_base * 4)
#         )
#         self.down3 = nn.Sequential(
#             nn.AvgPool2d(2), DoubleConv(n_features_base * 4, n_features_base * 8)
#         )
#         self.down4 = nn.Sequential(
#             nn.AvgPool2d(2), DoubleConv(n_features_base * 8, n_features_base * 16)
#         )

#         # --- Decoder ---
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
#             nn.Conv2d(n_features_base * 16, n_features_base * 8, kernel_size=1),
#         )
#         self.conv1 = DoubleConv(
#             n_features_base * 8 + n_features_base * 8, n_features_base * 8
#         )

#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
#             nn.Conv2d(n_features_base * 8, n_features_base * 4, kernel_size=1),
#         )
#         self.conv2 = DoubleConv(
#             n_features_base * 4 + n_features_base * 4, n_features_base * 4
#         )

#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
#             nn.Conv2d(n_features_base * 4, n_features_base * 2, kernel_size=1),
#         )
#         self.conv3 = DoubleConv(
#             n_features_base * 2 + n_features_base * 2, n_features_base * 2
#         )

#         self.up4 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
#             nn.Conv2d(n_features_base * 2, n_features_base, kernel_size=1),
#         )
#         self.conv4 = DoubleConv(n_features_base + n_features_base, n_features_base)

#         # --- Dual Heads ---

#         # 1. Regression Head (Intensity)
#         # Predicts the residual depth.
#         self.head_reg = nn.Conv2d(n_features_base, out_channels, kernel_size=1)

#         # 2. Classification Head (Rain Probability)
#         # Predicts Logits (use sigmoid later).
#         self.head_clf = nn.Conv2d(n_features_base, 1, kernel_size=1)

#         self.final_activation = nn.ReLU()

#     def forward(self, x):
#         baseline = x[:, 0:1, :, :]

#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         x = self.up1(x5)
#         x = torch.cat([x4, x], dim=1)
#         x = self.conv1(x)

#         x = self.up2(x)
#         x = torch.cat([x3, x], dim=1)
#         x = self.conv2(x)

#         x = self.up3(x)
#         x = torch.cat([x2, x], dim=1)
#         x = self.conv3(x)

#         x = self.up4(x)
#         x = torch.cat([x1, x], dim=1)
#         x = self.conv4(x)

#         # --- Head 1: Regression ---
#         residual = self.head_reg(x)
#         out_reg = baseline + residual
#         out_reg = self.final_activation(out_reg)  # Intensity is always >= 0

#         # --- Head 2: Classification ---
#         # We return logits (unnormalized scores) for numerical stability with BCEWithLogitsLoss
#         out_clf_logits = self.head_clf(x)

#         return out_reg, out_clf_logits
