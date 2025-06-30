import os
import torch
from tqdm import tqdm


# --- Training Loop Function ---
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs,
    patience,
    checkpoint_dir="checkpoints",
    early_stopping=True,
):
    """
    Trains and validates the UNet model.

    Args:
        model (nn.Module): The UNet model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        loss_fn (nn.Module): Loss function (e.g., nn.MSELoss).
        device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
        checkpoint_dir (str): Directory to save model checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    model.to(device)

    print(f"Starting training on device: {device}")

    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        train_batches = 0

        # Training Phase
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )
        for batch_idx, batch_data in enumerate(train_pbar):
            coarse_precip = batch_data["coarse_precip"].to(device)
            interpolated_precip = batch_data["interpolated_precip"].to(device)
            elevation = batch_data["elevation"].to(device)
            target_normalized_precip = batch_data["target_normalized_precip"].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(coarse_precip, interpolated_precip, elevation)

            # Calculate loss
            loss = loss_fn(outputs, target_normalized_precip)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({"loss": running_train_loss / (batch_idx + 1)})

        avg_train_loss = (
            running_train_loss / train_batches if train_batches > 0 else 0.0
        )

        # Validation Phase
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Validation]",
                leave=False,
            )
            for batch_idx, batch_data in enumerate(val_pbar):
                coarse_precip = batch_data["coarse_precip"].to(device)
                interpolated_precip = batch_data["interpolated_precip"].to(device)
                elevation = batch_data["elevation"].to(device)
                target_normalized_precip = batch_data["target_normalized_precip"].to(
                    device
                )

                outputs = model(coarse_precip, interpolated_precip, elevation)
                loss = loss_fn(outputs, target_normalized_precip)

                running_val_loss += loss.item()
                val_batches += 1
                val_pbar.set_postfix({"val_loss": running_val_loss / (batch_idx + 1)})

        avg_val_loss = running_val_loss / val_batches if val_batches > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"unet_best_model_epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Saved best model checkpoint to {checkpoint_path} with Val Loss: {best_val_loss:.6f}"
            )
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stopping and early_stop_counter >= patience:
            break

    print("Training finished.")

    return checkpoint_path
