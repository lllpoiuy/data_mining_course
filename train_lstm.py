from dataset import load_FY_Dataset
import torch
from model import OptimizedLSTM
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataloaders = load_FY_Dataset(csv_files=["datasets/test.csv"], batch_size=32)

# Create LSTM model
model = OptimizedLSTM(
    seq_len=12,          # Same as MaskedLinear
    input_dim=3,         # Same as MaskedLinear
    hidden_dim=8,        # Increased hidden dimension for better expressiveness
    output_dim=1,        # Same as MaskedLinear
    num_layers=2         # Increased to 2 layers to better capture complex relationships
).to(device)

# Print model parameters count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LSTM model total trainable parameters: {total_params}")

# Use the same loss function as MaskedLinear for comparability
criterion = nn.L1Loss()

# Use AdamW optimizer with dynamic weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=2e-5, betas=(0.9, 0.999))

# Create learning rate scheduler for dynamic learning rate adjustment
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=3000,
    steps_per_epoch=1,
    pct_start=0.2,        # 20% time for warmup
    div_factor=25.0,      # initial_lr = max_lr/div_factor
    final_div_factor=10000.0  # final_lr = max_lr/final_div_factor
)

# Arrays to store training curves
val_losses = []          # Validation losses
train_losses = []        # Training losses
march_errors = []        # March prediction errors
may_errors = []          # May prediction errors
july_errors = []         # July prediction errors

# Checkpoint file path
CHECKPOINT_FILE = "lstm_training_checkpoint.pt"

def eval():
    """
    Evaluate the model on the validation set.
    Uses the same evaluation methodology as MaskedLinear.
    """
    model.eval()
    total_loss = 0.0
    sum_outputs = torch.zeros(12)
    sum_targets = torch.zeros(12)
    with torch.no_grad():
        for batch in dataloaders['val']:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.reshape(-1, 12)
            sum_outputs += outputs.sum(dim=0).cpu()
            sum_targets += targets.sum(dim=0).cpu()
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloaders['val'])
    eval_outputs = (sum_outputs - sum_targets) / sum_targets
    
    # Store metrics for plotting
    val_losses.append(avg_loss)
    march_errors.append(eval_outputs[2].cpu().item())  # March
    may_errors.append(eval_outputs[4].cpu().item())    # May
    july_errors.append(eval_outputs[6].cpu().item())   # July
    
    return avg_loss

def train():
    """
    Train the model for one epoch.
    Uses the same training methodology as MaskedLinear.
    """
    model.train()
    total_loss = 0.0
    for batch in dataloaders['train']:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.reshape(-1, 12)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloaders['train'])
    train_losses.append(avg_loss)  # Record training loss
    return avg_loss

def save_checkpoint(epoch, best_val_loss, patience_counter):
    """Save checkpoint for resumable training"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'curves': {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'march_error': march_errors,
            'may_error': may_errors,
            'july_error': july_errors
        }
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint():
    """Load checkpoint to resume training"""
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading checkpoint from {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        
        # Load curve data
        global train_losses, val_losses, march_errors, may_errors, july_errors
        train_losses = checkpoint['curves'].get('train_loss', [])
        val_losses = checkpoint['curves']['val_loss']
        march_errors = checkpoint['curves']['march_error']
        may_errors = checkpoint['curves']['may_error']
        july_errors = checkpoint['curves']['july_error']
        
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch, best_val_loss, patience_counter
    else:
        print("No checkpoint found, starting training from scratch")
        return 0, float('inf'), 0

def plot_curves(smoothed=True):
    """
    Plot training curves and evaluation curves
    Display both training and validation loss as well as prediction errors
    """
    # Apply smoothing
    if smoothed and len(train_losses) > 10:
        train_losses_plot = gaussian_filter1d(np.array(train_losses), sigma=3).tolist()
        val_losses_plot = gaussian_filter1d(np.array(val_losses), sigma=3).tolist()
        march_errors_plot = gaussian_filter1d(np.array(march_errors), sigma=3).tolist()
        may_errors_plot = gaussian_filter1d(np.array(may_errors), sigma=3).tolist()
        july_errors_plot = gaussian_filter1d(np.array(july_errors), sigma=3).tolist()
    else:
        train_losses_plot = train_losses
        val_losses_plot = val_losses
        march_errors_plot = march_errors
        may_errors_plot = may_errors
        july_errors_plot = july_errors
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # First subplot: Training and validation loss
    ax1.plot(train_losses_plot, label='Training Loss', color='blue')
    ax1.plot(val_losses_plot, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('LSTM Training Curves - Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Second subplot: Prediction errors
    ax2.plot(march_errors_plot, label='March', color='green')
    ax2.plot(may_errors_plot, label='May', color='orange')
    ax2.plot(july_errors_plot, label='July', color='purple')
    ax2.axhline(y=0.05, color='r', linestyle='--', label='Target Range')
    ax2.axhline(y=-0.05, color='r', linestyle='--')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Prediction Error')
    ax2.set_title('LSTM Evaluation Curves - Prediction Errors')
    ax2.legend()
    ax2.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("lstm_training_curves.png")
    plt.close()

if __name__ == "__main__":
    # If checkpoint exists, load it
    start_epoch, best_val_loss, patience_counter = load_checkpoint()
    
    # Training settings
    num_epochs = 3000
    patience = 250  # Early stopping patience value
    min_epochs = 500  # Minimum training rounds to ensure sufficient training
    checkpoint_interval = 50  # Save checkpoint every 50 epochs
    
    # Track recent validation loss history for more robust early stopping
    val_loss_history = []
    history_size = 10  # Consider recent 10 epochs
    
    # Training loop with progress bar
    try:
        for epoch in tqdm(range(start_epoch, num_epochs)):
            train_loss = train()
            val_loss = eval()
            
            # Update learning rate scheduler
            scheduler.step()
            
            # Update validation loss history
            val_loss_history.append(val_loss)
            if len(val_loss_history) > history_size:
                val_loss_history.pop(0)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "lstm_model.pth")
                patience_counter = 0
                print(f"New best model saved at epoch {epoch} with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Periodic checkpoint saving
            if epoch % checkpoint_interval == 0:
                save_checkpoint(epoch, best_val_loss, patience_counter)
                # Plot current training curve
                plot_curves()
            
            # Early stopping check (must exceed minimum training rounds)
            if epoch > min_epochs and patience_counter >= patience:
                # Check recent validation loss trend
                recent_min = min(val_loss_history)
                recent_avg = sum(val_loss_history) / len(val_loss_history)
                # If recent average is worse than best by more than 3%, early stop
                if recent_avg > recent_min * 1.03:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                else:
                    # Reset patience counter to give model more training time
                    print(f"Recent validation loss is stable, resetting patience counter")
                    patience_counter = patience // 2
            
            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
                print(f"Patience counter: {patience_counter}/{patience}")
        
        # Save final checkpoint
        save_checkpoint(epoch, best_val_loss, patience_counter)
        
        print("\nTraining completed successfully!")
        print("Final best validation loss:", best_val_loss)
        print("Model saved as lstm_model.pth")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(epoch, best_val_loss, patience_counter)
        print("You can resume training later by running this script again.")
        exit(0)
    except Exception as e:
        print(f"\nError occurred during training: {e}")
        save_checkpoint(epoch, best_val_loss, patience_counter)
        print("Checkpoint saved. You can resume after fixing the issue.")
        raise
    
    # Only create visualizations if training is successful
    if os.path.exists("lstm_model.pth"):
        # Create final training curve plot
        plot_curves(smoothed=True)
        
        print("Training visualizations saved.") 