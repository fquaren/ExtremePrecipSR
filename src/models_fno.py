import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    """
    2D Fourier Layer.
    It performs FFT, mixes low-frequency modes in the frequency domain, and performs IFFT.
    This creates the 'Resolution Invariance' and 'Global Receptive Field'.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        # Complex weights for the top-left corner (Low Freqs)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        # Complex weights for the bottom-left corner (High Vertical / Low Horizontal Freqs)
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def complex_mul2d(self, input, weights):
        # (batch, in_channel, x, y) * (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 1. Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # 2. Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Upper block (Low frequencies)
        out_ft[:, :, : self.modes1, : self.modes2] = self.complex_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )

        # Lower block
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.complex_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # 3. Inverse FFT back to physical domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class ProbabilisticFNO(nn.Module):
    """
    Probabilistic FNO Emulator.

    Inputs:
        x: [Batch, 1, H, W] Precipitation Field

    Outputs:
        mu:  [Batch, 3, Q] Predicted Mean of Metrics (Area, Perim, CC)
        var: [Batch, 3, Q] Predicted Variance (Uncertainty)
    """

    def __init__(self, n_quantiles=10, modes=12, width=32):
        super(ProbabilisticFNO, self).__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.n_quantiles = n_quantiles

        # 1. Lifting Layer: Projects (x, y, precip) -> Latent Width
        self.fc0 = nn.Linear(3, self.width)

        # 2. Spectral Layers (Global Mixing)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # 3. Residual Bypasses (Crucial for High-Freq Storm Texture)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # 4. Projection to Output
        # We sum-pool over spatial dimensions to get global metrics
        self.fc1 = nn.Linear(self.width, 128)

        # Output dim is 2 * (3 metrics * Q quantiles) -> [Mean, LogVar]
        self.output_dim = 3 * n_quantiles
        self.fc2 = nn.Linear(128, self.output_dim * 2)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    def forward(self, x):
        # x shape: [B, 1, H, W]

        # --- 1. Lifting & Grids ---
        grid = self.get_grid(x.shape, x.device)
        x_in = torch.cat((x, grid), dim=1)  # [B, 3, H, W]
        x_in = x_in.permute(0, 2, 3, 1)  # [B, H, W, 3]
        x_lifted = self.fc0(x_in)
        x = x_lifted.permute(0, 3, 1, 2)  # [B, C, H, W]

        # --- 2. FNO Blocks ---
        # Layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        # Layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        # Layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        # Layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)

        # --- 3. Topological Pooling ---
        # Sum-pool to approximate extensive properties
        x_pool = x.mean(dim=(2, 3))

        # --- 4. Latent Projection ---
        x_latent = F.gelu(self.fc1(x_pool))
        raw_out = self.fc2(x_latent)

        # --- 5. Split Mean / Variance ---
        B = raw_out.shape[0]
        # Reshape to [B, 2, 3, Q] -> 2=(mu,var), 3=(A,P,CC)
        raw_reshaped = raw_out.view(B, 2, 3, -1)

        mu = raw_reshaped[:, 0, :, :]  # [B, 3, Q]
        log_var = raw_reshaped[:, 1, :, :]  # [B, 3, Q]

        # --- 6. Constraints ---
        # Enforce positive variance
        var = F.softplus(log_var) + 1e-6

        # Gating for "Dry" inputs (Manifold Projection)
        input_mass = x.view(B, -1).sum(dim=1).view(B, 1, 1)
        gate = torch.sigmoid((input_mass - 0.1) * 10.0)

        mu = mu * gate

        return mu, var
