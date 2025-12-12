import torch
import torch.nn as nn


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

        # --- Dual Heads ---

        # 1. Regression Head (Intensity)
        # Predicts the residual depth.
        self.head_reg = nn.Conv2d(n_features_base, out_channels, kernel_size=1)

        # 2. Classification Head (Rain Probability)
        # Predicts Logits (use sigmoid later).
        self.head_clf = nn.Conv2d(n_features_base, 1, kernel_size=1)

        self.final_activation = nn.ReLU()

    def forward(self, x):
        baseline = x[:, 0:1, :, :]

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

        # --- Head 1: Regression ---
        residual = self.head_reg(x)
        out_reg = baseline + residual
        out_reg = self.final_activation(out_reg)  # Intensity is always >= 0

        # --- Head 2: Classification ---
        # We return logits (unnormalized scores) for numerical stability with BCEWithLogitsLoss
        out_clf_logits = self.head_clf(x)

        return out_reg, out_clf_logits
