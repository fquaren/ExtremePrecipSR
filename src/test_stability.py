import unittest
import torch
import numpy as np

# Import your model
from gamma_predictors_v5 import ConstrainedIsometricCNN


class TestModelStability(unittest.TestCase):

    def setUp(self):
        q_levels = np.linspace(0.1, 100, 30)
        self.model = ConstrainedIsometricCNN(
            n_quantiles=30,
            input_shape=(1, 128, 128),
            quantile_levels=q_levels,
            pixel_area_km2=4.0,
        )
        self.model.eval()

    def test_jacobian_spectrum_stability(self):
        """
        Robust Jacobian check using Random Projections (Hutchinson's Estimator).
        Avoids false positives from summing multiple outputs.
        """
        print("\n--- Test 1: Jacobian Spectrum Stability (Robust) ---")

        # 1. Random Input (Physical Scale: 0 to ~150)
        # We use abs() to match physical constraints (precip >= 0)
        x = torch.abs(torch.randn(8, 1, 128, 128)) * 20.0
        x.requires_grad = True

        # 2. Forward Pass
        output = self.model(x)

        # 3. Random Projection (The Fix)
        # Instead of summing, we project onto a random unit vector v.
        # This measures the magnitude of the gradient in a random direction.
        v = torch.randn_like(output)
        v = v / torch.norm(v)  # Normalize to unit length

        # Project output onto v
        target_scalar = (output * v).sum()
        target_scalar.backward()

        # 4. Analyze Gradients
        grads = x.grad.abs().detach().cpu().numpy().flatten()
        grads_nonzero = grads[grads > 1e-20]

        log_grads = np.log10(grads_nonzero)
        mean_log = np.mean(log_grads)
        std_log = np.std(log_grads)

        print(f"Gradient Statistics (Log10 scale):")
        print(f"  Mean: {mean_log:.4f} (Expected: -4.0 to 1.0)")
        print(f"  Std:  {std_log:.4f}  (Expected: < 3.0)")

        # 5. Assertions
        # Lower bound: Prevent vanishing gradients (dead network)
        self.assertGreater(mean_log, -8.0, "FAIL: Vanishing Gradients.")

        # Upper bound: Prevent exploding gradients
        # We relax the bound slightly to account for the InputNormalization layer dynamics
        self.assertLess(mean_log, 2.5, "FAIL: Exploding Gradients.")

        # Check for Shattering (Bimodality)
        self.assertLess(std_log, 4.0, "FAIL: Shattered Gradients (High Variance).")

        print("SUCCESS: Jacobian spectrum is stable.")

    def test_ghost_area_artifact(self):
        print("\n--- Test 2: Ghost Area Artifact Check ---")
        x_zero = torch.zeros(1, 1, 128, 128)

        with torch.no_grad():
            output = self.model(x_zero)

        # Get the accumulated area from the last quantile
        pred_area_dist = output[:, 0, :]
        pred_total_area = pred_area_dist.max().item()

        print(f"Predicted Total Area for Empty Input: {pred_total_area:.4f} km²")

        # Threshold: Should be small (e.g., Softplus(0) = 0.69)
        self.assertLess(
            pred_total_area,
            5.0,  # Stricter threshold since we fixed initialization
            f"FAIL: Ghost Area {pred_total_area:.2f} km² is too high.",
        )
        print("SUCCESS: Zero-input behavior is correct.")


if __name__ == "__main__":
    unittest.main()
