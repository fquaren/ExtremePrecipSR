import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    A block of two sequential 3x3 convolutions, each followed by
    Batch Normalization and a Mish activation.
    (CONV -> BN -> Mish) * 2
    """

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
    """
    A UNet architecture modified for residual super-resolution using
    Bicubic Interpolation for upscaling.
    """

    def __init__(self, in_channels, out_channels, n_features_base=64):
        super(UNetSR, self).__init__()

        # --- Encoder (Downsampling Path) ---
        self.inc = DoubleConv(in_channels, n_features_base)  # 128x128
        self.down1 = nn.Sequential(
            nn.AvgPool2d(2), DoubleConv(n_features_base, n_features_base * 2)  # 64x64
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(n_features_base * 2, n_features_base * 4),  # 32x32
        )
        self.down3 = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(n_features_base * 4, n_features_base * 8),  # 16x16
        )
        self.down4 = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(n_features_base * 8, n_features_base * 16),  # 8x8 (Bottleneck)
        )

        # --- Decoder (Upsampling Path) ---
        # Note: We replace ConvTranspose2d with Upsample + Conv2d(1x1).
        # Upsample expands spatial dim, Conv2d reduces channel dim to match skip connection.

        # Block 1: 8x8 -> 16x16
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(n_features_base * 16, n_features_base * 8, kernel_size=1),
        )
        self.conv1 = DoubleConv(
            n_features_base * 8 + n_features_base * 8, n_features_base * 8
        )

        # Block 2: 16x16 -> 32x32
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(n_features_base * 8, n_features_base * 4, kernel_size=1),
        )
        self.conv2 = DoubleConv(
            n_features_base * 4 + n_features_base * 4, n_features_base * 4
        )

        # Block 3: 32x32 -> 64x64
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(n_features_base * 4, n_features_base * 2, kernel_size=1),
        )
        self.conv3 = DoubleConv(
            n_features_base * 2 + n_features_base * 2, n_features_base * 2
        )

        # Block 4: 64x64 -> 128x128
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(n_features_base * 2, n_features_base, kernel_size=1),
        )
        self.conv4 = DoubleConv(n_features_base + n_features_base, n_features_base)

        # Final 1x1 convolution to produce the residual
        self.outc = nn.Conv2d(n_features_base, out_channels, kernel_size=1)

        # Final activation to enforce non-negativity *after* adding the residual
        self.final_activation = nn.ReLU()

    def forward(self, x):
        # x shape: (B, C_in, 128, 128)

        # Extract the baseline (upscaled low-res precip)
        # We assume this is channel 0.
        baseline = x[:, 0:1, :, :]

        # --- Encoder ---
        x1 = self.inc(x)  # (B, 64, 128, 128)
        x2 = self.down1(x1)  # (B, 128, 64, 64)
        x3 = self.down2(x2)  # (B, 256, 32, 32)
        x4 = self.down3(x3)  # (B, 512, 16, 16)
        x5 = self.down4(x4)  # (B, 1024, 8, 8)

        # --- Decoder ---
        # Up 1
        x = self.up1(x5)  # Upsample + Reduce Channels -> (B, 512, 16, 16)
        x = torch.cat(
            [x4, x], dim=1
        )  # Concat with skip connection -> (B, 1024, 16, 16)
        x = self.conv1(x)  # DoubleConv -> (B, 512, 16, 16)

        # Up 2
        x = self.up2(x)  # (B, 256, 32, 32)
        x = torch.cat([x3, x], dim=1)  # (B, 512, 32, 32)
        x = self.conv2(x)  # (B, 256, 32, 32)

        # Up 3
        x = self.up3(x)  # (B, 128, 64, 64)
        x = torch.cat([x2, x], dim=1)  # (B, 256, 64, 64)
        x = self.conv3(x)  # (B, 128, 64, 64)

        # Up 4
        x = self.up4(x)  # (B, 64, 128, 128)
        x = torch.cat([x1, x], dim=1)  # (B, 128, 128, 128)
        x = self.conv4(x)  # (B, 64, 128, 128)

        # Final prediction
        residual = self.outc(x)  # (B, C_out, 128, 128)

        output = baseline + residual
        output = self.final_activation(output)

        return output
