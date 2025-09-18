import torch
import torch.nn as nn
import torch.nn.init as init


def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            init.zeros_(m.bias)


class UNet(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(UNet, self).__init__()

        self.dropout_p = dropout_p

        # Now expecting 3 input channels: interpolated, coarse, elevation
        self.encoder1 = self.conv_block(3, 64)
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256, dropout_p=self.dropout_p)

        self.bottleneck = self.conv_block(256, 512, dropout_p=self.dropout_p)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256, dropout_p=self.dropout_p)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_p=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
        )

    def forward(self, interpolated_precip, coarse_precip, elevation):
        # Add channel dimensions if needed
        if interpolated_precip.ndim == 3:
            interpolated_precip = interpolated_precip.unsqueeze(1)
        if coarse_precip.ndim == 3:
            coarse_precip = coarse_precip.unsqueeze(1)
        if elevation.ndim == 3:
            elevation = elevation.unsqueeze(1)

        # Validate input shapes
        expected_shape = (1, 128, 128)
        for tensor, name in zip(
            [interpolated_precip, coarse_precip, elevation],
            ["interpolated_precip", "coarse_precip", "elevation"],
        ):
            assert (
                tensor.shape[1:] == expected_shape
            ), f"{name} must have shape (B, 1, 128, 128), got {tensor.shape}"

        # Concatenate along channel dimension: [B, 3, H, W]
        x = torch.cat((interpolated_precip, coarse_precip, elevation), dim=1)

        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)
