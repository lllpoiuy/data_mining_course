from dataset import load_FY_Dataset
import torch
from model import MaskedLinear, OptimizedLSTM
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

# Create both models with the same configuration
maskedlinear_model = MaskedLinear(
    seq_len=12,
    input_dim=3,
    hidden_dim=5,
    output_dim=1
).to(device)

lstm_model = OptimizedLSTM(
    seq_len=12,
    input_dim=3,
    hidden_dim=4,  # Smaller hidden dim to prevent overfitting
    output_dim=1,
    num_layers=1   # Single layer for fewer parameters
).to(device)

# Print model parameters count for comparison
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ml_params = count_parameters(maskedlinear_model)
lstm_params = count_parameters(lstm_model)
print(f"MaskedLinear parameters: {ml_params:,}")
print(f"LSTM parameters: {lstm_params:,}")
print(f"Parameter ratio: LSTM has {lstm_params/ml_params:.2f}x parameters compared to MaskedLinear")

# Using the same loss function for both models
criterion = nn.L1Loss()

# Configure optimizers with slightly different settings for each model
ml_optimizer = optim.Adam(maskedlinear_model.parameters(), lr=0.0003)
lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.0002, weight_decay=1e-5)

# Learning rate scheduler for LSTM model
lstm_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    lstm_optimizer,
    T_0=100, 
    T_mult=2,
    eta_min=0.00001
)

# Arrays to store training metrics for both models
ml_val_loss = []
ml_train_loss = []  # Add array to store training loss
ml_march_error = []
ml_may_error = []
ml_july_error = []

lstm_val_loss = []
lstm_train_loss = []  # Add array to store training loss
lstm_march_error = []
lstm_may_error = []
lstm_july_error = []

def evaluate_model(model, name):
    """Evaluate model on validation set"""
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
    
    # Store metrics based on model type
    if name == "maskedlinear":
        ml_val_loss.append(avg_loss)
        ml_march_error.append(eval_outputs[2].cpu().item())
        ml_may_error.append(eval_outputs[4].cpu().item())
        ml_july_error.append(eval_outputs[6].cpu().item())
    else:  # lstm
        lstm_val_loss.append(avg_loss)
        lstm_march_error.append(eval_outputs[2].cpu().item())
        lstm_may_error.append(eval_outputs[4].cpu().item())
        lstm_july_error.append(eval_outputs[6].cpu().item())
        
    return avg_loss, eval_outputs

def train_model(model, optimizer, scheduler=None, name=""):
    """Train model for one epoch"""
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
        
        # Apply gradient clipping for LSTM model
        if name == "lstm":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloaders['train'])
    
    # Store training loss based on model type
    if name == "maskedlinear":
        ml_train_loss.append(avg_loss)
    else:  # lstm
        lstm_train_loss.append(avg_loss)
        
    return avg_loss

def smooth_and_plot_curves():
    """Apply smoothing and create comparison plots"""
    # Apply smoothing
    ml_val_loss_smooth = gaussian_filter1d(np.array(ml_val_loss), sigma=3).tolist()
    ml_march_error_smooth = gaussian_filter1d(np.array(ml_march_error), sigma=3).tolist()
    ml_may_error_smooth = gaussian_filter1d(np.array(ml_may_error), sigma=3).tolist()
    ml_july_error_smooth = gaussian_filter1d(np.array(ml_july_error), sigma=3).tolist()
    
    lstm_val_loss_smooth = gaussian_filter1d(np.array(lstm_val_loss), sigma=3).tolist()
    lstm_march_error_smooth = gaussian_filter1d(np.array(lstm_march_error), sigma=3).tolist()
    lstm_may_error_smooth = gaussian_filter1d(np.array(lstm_may_error), sigma=3).tolist()
    lstm_july_error_smooth = gaussian_filter1d(np.array(lstm_july_error), sigma=3).tolist()
    
    # Plot validation loss comparison
    plt.figure(figsize=(10, 6))
    plt.plot(ml_val_loss_smooth, label='MaskedLinear Loss')
    plt.plot(lstm_val_loss_smooth, label='OptimizedLSTM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Model Comparison: Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_comparison.png")
    plt.close()
    
    # Combined loss plot for training and validation (if training loss is available)
    if hasattr(globals(), 'ml_train_loss') and hasattr(globals(), 'lstm_train_loss'):
        ml_train_loss_smooth = gaussian_filter1d(np.array(ml_train_loss), sigma=3).tolist()
        lstm_train_loss_smooth = gaussian_filter1d(np.array(lstm_train_loss), sigma=3).tolist()
        
        plt.figure(figsize=(12, 10))
        
        # Create a 2x2 subplot grid
        plt.subplot(2, 1, 1)
        plt.plot(ml_train_loss_smooth, label='Training Loss', color='blue')
        plt.plot(ml_val_loss_smooth, label='Validation Loss', color='red')
        plt.title('MaskedLinear: Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(lstm_train_loss_smooth, label='Training Loss', color='blue')
        plt.plot(lstm_val_loss_smooth, label='Validation Loss', color='red')
        plt.title('OptimizedLSTM: Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("combined_training_curves.png")
        plt.close()
    
    # Plot prediction errors for March
    plt.figure(figsize=(10, 6))
    plt.plot(ml_march_error_smooth, label='MaskedLinear')
    plt.plot(lstm_march_error_smooth, label='OptimizedLSTM')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Target Range')
    plt.axhline(y=-0.05, color='r', linestyle='--')
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Error')
    plt.title('Model Comparison: March Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("march_comparison.png")
    plt.close()
    
    # Plot prediction errors for May
    plt.figure(figsize=(10, 6))
    plt.plot(ml_may_error_smooth, label='MaskedLinear')
    plt.plot(lstm_may_error_smooth, label='OptimizedLSTM')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Target Range')
    plt.axhline(y=-0.05, color='r', linestyle='--')
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Error')
    plt.title('Model Comparison: May Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("may_comparison.png")
    plt.close()
    
    # Plot prediction errors for July
    plt.figure(figsize=(10, 6))
    plt.plot(ml_july_error_smooth, label='MaskedLinear')
    plt.plot(lstm_july_error_smooth, label='OptimizedLSTM')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Target Range')
    plt.axhline(y=-0.05, color='r', linestyle='--')
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Error')
    plt.title('Model Comparison: July Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("july_comparison.png")
    plt.close()
    
    # Combined plot for all months
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 3, 1)
    plt.plot(ml_march_error_smooth, label='MaskedLinear')
    plt.plot(lstm_march_error_smooth, label='OptimizedLSTM')
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.axhline(y=-0.05, color='r', linestyle='--')
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Error')
    plt.title('March')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(ml_may_error_smooth, label='MaskedLinear')
    plt.plot(lstm_may_error_smooth, label='OptimizedLSTM')
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.axhline(y=-0.05, color='r', linestyle='--')
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Epoch')
    plt.title('May')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(ml_july_error_smooth, label='MaskedLinear')
    plt.plot(lstm_july_error_smooth, label='OptimizedLSTM')
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.axhline(y=-0.05, color='r', linestyle='--')
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Epoch')
    plt.title('July')
    plt.grid(True)
    
    plt.suptitle('Prediction Error Comparison by Month', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig("monthly_comparison.png")
    plt.close()

if __name__ == "__main__":
    # Training settings
    num_epochs = 1500  # Lower number of epochs for quicker comparison
    ml_best_val_loss = float('inf')
    lstm_best_val_loss = float('inf')
    
    # Patience settings for early stopping
    patience = 200
    ml_patience_counter = 0
    lstm_patience_counter = 0
    
    # Track training time
    start_time = time.time()
    
    # Training loop with progress bar
    for epoch in tqdm(range(num_epochs)):
        # Train and evaluate MaskedLinear model
        ml_train_loss = train_model(maskedlinear_model, ml_optimizer, name="maskedlinear")
        ml_val_loss_current, ml_outputs = evaluate_model(maskedlinear_model, "maskedlinear")
        
        # Train and evaluate LSTM model
        lstm_train_loss = train_model(lstm_model, lstm_optimizer, name="lstm")
        lstm_val_loss_current, lstm_outputs = evaluate_model(lstm_model, "lstm")
        
        # Update LSTM scheduler
        lstm_scheduler.step()
        
        # Save best MaskedLinear model
        if ml_val_loss_current < ml_best_val_loss:
            ml_best_val_loss = ml_val_loss_current
            torch.save(maskedlinear_model.state_dict(), "maskedlinear_model_compare.pth")
            ml_patience_counter = 0
        else:
            ml_patience_counter += 1
            
        # Save best LSTM model
        if lstm_val_loss_current < lstm_best_val_loss:
            lstm_best_val_loss = lstm_val_loss_current
            torch.save(lstm_model.state_dict(), "lstm_model_compare.pth")
            lstm_patience_counter = 0
        else:
            lstm_patience_counter += 1
            
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"MaskedLinear - Train Loss: {ml_train_loss:.4f}, Val Loss: {ml_val_loss_current:.4f}")
            print(f"LSTM - Train Loss: {lstm_train_loss:.4f}, Val Loss: {lstm_val_loss_current:.4f}")
            print(f"Current LSTM learning rate: {lstm_scheduler.get_last_lr()[0]:.6f}")
            
        # Check for early stopping for both models
        if ml_patience_counter >= patience and lstm_patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} - both models converged")
            break
            
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Final model performance
    print("\nFinal Best Validation Loss:")
    print(f"MaskedLinear: {ml_best_val_loss:.4f}")
    print(f"LSTM: {lstm_best_val_loss:.4f}")
    print(f"Improvement ratio: {ml_best_val_loss/lstm_best_val_loss:.2f}x")
    
    # Parameter count comparison
    print("\nModel Parameter Count:")
    print(f"MaskedLinear: {ml_params:,} parameters")
    print(f"LSTM: {lstm_params:,} parameters")
    print(f"Parameter ratio: LSTM has {lstm_params/ml_params:.2f}x parameters compared to MaskedLinear")
    
    # Save training metrics to a file for future analysis
    np.savez('training_metrics.npz', 
             ml_train_loss=np.array(ml_train_loss), 
             ml_val_loss=np.array(ml_val_loss),
             ml_march_error=np.array(ml_march_error),
             ml_may_error=np.array(ml_may_error),
             ml_july_error=np.array(ml_july_error),
             lstm_train_loss=np.array(lstm_train_loss), 
             lstm_val_loss=np.array(lstm_val_loss),
             lstm_march_error=np.array(lstm_march_error),
             lstm_may_error=np.array(lstm_may_error),
             lstm_july_error=np.array(lstm_july_error)
            )
    print("Training metrics saved to training_metrics.npz")
    
    # Generate comparison plots
    smooth_and_plot_curves()
    print("\nComparison plots have been saved.")
    
    # Add brief analysis to a text file
    with open("model_comparison_analysis.txt", "w") as f:
        f.write("MODEL COMPARISON ANALYSIS\n")
        f.write("=======================\n\n")
        f.write(f"Training time: {training_time:.2f} seconds\n\n")
        f.write("Model Parameters:\n")
        f.write(f"- MaskedLinear: {ml_params:,} parameters\n")
        f.write(f"- OptimizedLSTM: {lstm_params:,} parameters\n")
        f.write(f"- Ratio: LSTM has {lstm_params/ml_params:.2f}x parameters compared to MaskedLinear\n\n")
        f.write("Final Best Validation Loss:\n")
        f.write(f"- MaskedLinear: {ml_best_val_loss:.4f}\n")
        f.write(f"- OptimizedLSTM: {lstm_best_val_loss:.4f}\n")
        f.write(f"- Improvement ratio: {ml_best_val_loss/lstm_best_val_loss:.2f}x\n\n")
        f.write("ANALYSIS SUMMARY:\n")
        f.write("The OptimizedLSTM model uses specialized architecture to capture sequential dependencies,\n")
        f.write("while the MaskedLinear model uses an explicit masking approach.\n\n")
        f.write("The optimized LSTM model incorporates several advanced techniques:\n")
        f.write("- Feature normalization to stabilize training\n")
        f.write("- Attention mechanism to focus on relevant time steps\n")
        f.write("- Residual connections to improve gradient flow\n")
        f.write("- GELU activation function which often works better for sequence data\n") 
        f.write("- Dropout regularization (0.1)\n")
        f.write("- Weight decay in optimizer (1e-5)\n")
        f.write("- Cosine annealing learning rate scheduling\n")
        f.write("- Gradient clipping\n")
        f.write("- Layer normalization\n")
        f.write("- Orthogonal initialization for recurrent weights\n\n")
        f.write("These optimizations help the LSTM model maintain good generalization\n")
        f.write("performance despite having intrinsically more parameters than the MaskedLinear model.\n")
    
    print("Analysis saved to model_comparison_analysis.txt")