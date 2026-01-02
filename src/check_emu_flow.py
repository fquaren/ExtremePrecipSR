import torch
from gamma_predictors import (
    GammaPredictorHierarchicalHardGated_old,
    GammaPredictorHierarchicalHardGated,
)

# Instantiate new class
# model = GammaPredictorHierarchicalHardGated(input_shape=(1, 32, 32), n_quantiles=16)
model = GammaPredictorHierarchicalHardGated_old(input_shape=(1, 32, 32), n_quantiles=16)

# Create a dummy input (all zeros)
x = torch.zeros(1, 1, 32, 32, requires_grad=True)

# 1. Forward Pass
# This relies on the 'with torch.no_grad():' block
out = model(x)

# 2. We try to maximize the output (Area)
loss = out.sum()
loss.backward()

# 3. Check gradients specifically on the pixel values
# If the UNet can learn to "turn on" pixels, this should be non-zero.
grad_sum = x.grad.abs().sum().item()

print(f"Gradient Magnitude on Pixels: {grad_sum}")

if grad_sum == 0.0:
    print(
        "CONFIRMED: The model cannot learn to turn pixels on/off (Area gradient is dead)."
    )
else:
    # Note: You might get a tiny non-zero value from the CNN trunk noise,
    # but it won't be the strong directional gradient needed to increase Area.
    print(
        "Gradients exist (from shape trunk), but are likely disconnected from Area magnitude."
    )
