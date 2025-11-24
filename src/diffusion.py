import torch
import torch.nn as nn
from tqdm import tqdm


class Diffusion(nn.Module):
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=128,
        device="cuda",
    ):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Define linear beta schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Diffuses images x to timestep t.
        Returns x_t and the noise used.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """Samples n random timesteps for training."""
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    def sample(self, model, n, conditions):
        """
        Generates new images from noise using the model and conditional input.
        conditions: The low-res/upsampled images [B, C, H, W]
        """
        model.eval()
        print(f"Sampling {n} images....")

        # Start from pure noise
        x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)

        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                # Predict noise
                predicted_noise = model(x, t, conditions)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # DDPM Sampling equation
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                ) + torch.sqrt(beta) * noise

        model.train()

        # Clamp output to physical range [0, 1] (optional but good for images)
        # Since our data is normalized [0,1], diffusion operates in roughly [-1, 1] or N(0,1).
        # Usually DDPM data is scaled to [-1, 1].
        # Assuming your data loader keeps [0, 1], we might clamp [0, 1].
        x = x.clamp(0, 1)
        return x
