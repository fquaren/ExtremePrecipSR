import os
import torch
from tqdm import tqdm


# --- Evaluation Loop Function ---
def evaluate_model(model, test_loader, loss_fn, device, model_path=None):
    """
    Evaluates the trained UNet model on the test dataset.

    Args:
        model (nn.Module): The UNet model to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        loss_fn (nn.Module): Loss function (e.g., nn.MSELoss).
        device (torch.device): Device to evaluate on (e.g., 'cuda' or 'cpu').
        model_path (str, optional): Path to the saved model checkpoint (.pth file).
                                    If None, the provided model object is used as is.
    Returns:
        float: The average test loss.
    """
    model.to(device)

    # Load model state if a path is provided
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        print(f"Loading model state from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")

    model.eval()  # Set model to evaluation mode
    running_test_loss = 0.0
    test_batches = 0

    print(f"Starting evaluation on device: {device}")
    with torch.no_grad():  # Disable gradient calculation
        test_pbar = tqdm(test_loader, desc="Evaluation [Test]", leave=True)
        for batch_idx, batch_data in enumerate(test_pbar):
            coarse_precip = batch_data["coarse_precip"].to(device)
            interpolated_precip = batch_data["interpolated_precip"].to(device)
            elevation = batch_data["elevation"].to(device)
            target_normalized_precip = batch_data["target_normalized_precip"].to(device)

            outputs = model(coarse_precip, interpolated_precip, elevation)
            loss = loss_fn(outputs, target_normalized_precip)

            running_test_loss += loss.item()
            test_batches += 1
            test_pbar.set_postfix({"test_loss": running_test_loss / (batch_idx + 1)})

    avg_test_loss = running_test_loss / test_batches if test_batches > 0 else 0.0
    print(f"\nFinal Test Loss: {avg_test_loss:.6f}")
    return avg_test_loss
