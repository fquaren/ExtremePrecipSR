import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    A block of two sequential 3x3 convolutions, each followed by
    Batch Normalization and a LeakyReLU activation.
    (CONV -> BN -> LeakyReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetSR(nn.Module):
    """
    A standard UNet architecture modified for a residual super-resolution task.
    It accepts a multi-channel input (e.g., precip + DEM)
    and predicts a single-channel residual, which is added to the input precip.

    This architecture assumes the input patch size is a power of 2 (e.g., 128).
    """

    def __init__(self, in_channels, out_channels, n_features_base=64):
        super(UNetSR, self).__init__()

        # Encoder (Downsampling Path)
        self.inc = DoubleConv(in_channels, n_features_base)  # 128x128
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(n_features_base, n_features_base * 2)  # 64x64
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(n_features_base * 2, n_features_base * 4),  # 32x32
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(n_features_base * 4, n_features_base * 8),  # 16x16
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(n_features_base * 8, n_features_base * 16),  # 8x8 (Bottleneck)
        )

        # Decoder (Upsampling Path)
        self.up1 = nn.ConvTranspose2d(
            n_features_base * 16, n_features_base * 8, kernel_size=2, stride=2
        )
        self.conv1 = DoubleConv(
            n_features_base * 8 + n_features_base * 8, n_features_base * 8
        )  # 16x16

        self.up2 = nn.ConvTranspose2d(
            n_features_base * 8, n_features_base * 4, kernel_size=2, stride=2
        )
        self.conv2 = DoubleConv(
            n_features_base * 4 + n_features_base * 4, n_features_base * 4
        )  # 32x32

        self.up3 = nn.ConvTranspose2d(
            n_features_base * 4, n_features_base * 2, kernel_size=2, stride=2
        )
        self.conv3 = DoubleConv(
            n_features_base * 2 + n_features_base * 2, n_features_base * 2
        )  # 64x64

        self.up4 = nn.ConvTranspose2d(
            n_features_base * 2, n_features_base, kernel_size=2, stride=2
        )
        self.conv4 = DoubleConv(
            n_features_base + n_features_base, n_features_base
        )  # 128x128

        # Final 1x1 convolution to produce the residual
        self.outc = nn.Conv2d(n_features_base, out_channels, kernel_size=1)

        # Final activation to enforce non-negativity *after* adding the residual
        self.final_activation = nn.ReLU()

    def forward(self, x):
        # x shape: (B, C_in, 128, 128)

        # Extract the baseline (upscaled low-res precip)
        # We assume this is the first channel.
        # We must detach it if it's not part of the loss's target
        # or if we only want to backprop through the residual.
        # For a simple residual model, just select it.
        baseline = x[:, 0:1, :, :]  # Shape: (B, 1, 128, 128)

        # --- Encoder ---
        x1 = self.inc(x)  # (B, 64, 128, 128)
        x2 = self.down1(x1)  # (B, 128, 64, 64)
        x3 = self.down2(x2)  # (B, 256, 32, 32)
        x4 = self.down3(x3)  # (B, 512, 16, 16)
        x5 = self.down4(x4)  # (B, 1024, 8, 8)

        # --- Decoder ---
        x = self.up1(x5)  # (B, 512, 16, 16)
        x = torch.cat([x4, x], dim=1)  # (B, 1024, 16, 16)
        x = self.conv1(x)  # (B, 512, 16, 16)

        x = self.up2(x)  # (B, 256, 32, 32)
        x = torch.cat([x3, x], dim=1)  # (B, 512, 32, 32)
        x = self.conv2(x)  # (B, 256, 32, 32)

        x = self.up3(x)  # (B, 128, 64, 64)
        x = torch.cat([x2, x], dim=1)  # (B, 256, 64, 64)
        x = self.conv3(x)  # (B, 128, 64, 64)

        x = self.up4(x)  # (B, 64, 128, 128)
        x = torch.cat([x1, x], dim=1)  # (B, 128, 128, 128)
        x = self.conv4(x)  # (B, 64, 128, 128)

        # This is now the predicted residual (can be positive or negative)
        residual = self.outc(x)  # (B, C_out, 128, 128)

        # Add the predicted residual to the baseline
        output = baseline + residual

        # Enforce non-negativity *after* the addition
        output = self.final_activation(output)

        return output
