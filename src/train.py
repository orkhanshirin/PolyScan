import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from architectures import FCN, UNet

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model_name, data_img, data_seg, idx_train, idx_val, epochs=100, batch_size=16, learning_rate=0.001):
    """Trains the selected model (FCN or UNet) and saves the best model."""
    
    # Select model dynamically
    if model_name.lower() == "fcn":
        model = FCN().to(device)
    elif model_name.lower() == "unet":
        model = UNet().to(device)
    else:
        raise ValueError("Invalid model name! Choose either 'fcn' or 'unet'.")
    
    best_model = type(model)().to(device)  # Create a fresh model to store best state

    # Loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.9 * epochs), gamma=0.1)

    # Track losses
    train_losses, val_losses = [], []
    best_loss = float('inf')

    print(f"Training {model_name.upper()} for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Mini-batch gradient descent
        for i in range(0, len(idx_train), batch_size):
            batch_indices = idx_train[i:i+batch_size]
            source_image, target_label = data_img[batch_indices], data_seg[batch_indices]

            # Forward pass
            pred_label = model(source_image)
            loss = loss_function(pred_label, target_label)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / (len(idx_train) // batch_size)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(idx_val), batch_size):
                batch_indices = idx_val[i:i+batch_size]
                source_image, target_label = data_img[batch_indices], data_seg[batch_indices]
                pred_label = model(source_image)
                loss = loss_function(pred_label, target_label)
                val_loss += loss.item()

        avg_val_loss = val_loss / (len(idx_val) // batch_size)
        val_losses.append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model.load_state_dict(model.state_dict())

        # Step the learning rate scheduler
        scheduler.step()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Time Elapsed = {elapsed_time:.2f}s")

    # Save the best model
    model_save_path = f"models/best_{model_name.lower()}.pth"
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Training complete. Best model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model (FCN or UNet).")
    parser.add_argument("model", type=str, default="unet", help="Model to train: 'fcn' or 'unet'")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")

    args = parser.parse_args()

    # Load dataset (from data_loader.py)
    from data_loader import load_dataset
    data_img, data_seg, idx_train, idx_val, _ = load_dataset()

    train_model(args.model, data_img, data_seg, idx_train, idx_val, args.epochs, args.batch_size, args.lr)
