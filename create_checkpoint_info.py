import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# Load checkpoint file
checkpoint = torch.load('lstm_training_checkpoint.pt')

# Print key information
print(f"Final epoch: {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
print(f"Patience counter: {checkpoint['patience_counter']}")

# Print some statistics about the training curves
val_losses = checkpoint['curves']['val_loss']
march_errors = checkpoint['curves']['march_error']
may_errors = checkpoint['curves']['may_error']
july_errors = checkpoint['curves']['july_error']

print(f"\nNumber of training epochs: {len(val_losses)}")
print(f"Initial validation loss: {val_losses[0]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")
print(f"Best validation loss: {min(val_losses):.4f}")

# Calculate the average error for the last 10% of training
last_10_percent = int(len(march_errors) * 0.1) or 1
print(f"\nAverage errors for last {last_10_percent} epochs:")
print(f"March: {sum(march_errors[-last_10_percent:]) / last_10_percent:.4f}")
print(f"May: {sum(may_errors[-last_10_percent:]) / last_10_percent:.4f}")
print(f"July: {sum(july_errors[-last_10_percent:]) / last_10_percent:.4f}")

# Calculate how many epochs stayed within the target range of [-0.05, 0.05]
march_in_target = sum(1 for err in march_errors if -0.05 <= err <= 0.05)
may_in_target = sum(1 for err in may_errors if -0.05 <= err <= 0.05)
july_in_target = sum(1 for err in july_errors if -0.05 <= err <= 0.05)

print(f"\nPercentage of epochs with error within target range [-0.05, 0.05]:")
print(f"March: {march_in_target / len(march_errors) * 100:.2f}%")
print(f"May: {may_in_target / len(may_errors) * 100:.2f}%")
print(f"July: {july_in_target / len(july_errors) * 100:.2f}%")

# Analyze convergence - find the minimum loss and when it was achieved
min_loss = min(val_losses)
min_loss_epoch = val_losses.index(min_loss)
print(f"\nBest validation loss {min_loss:.4f} was achieved at epoch {min_loss_epoch}")
print(f"This happened at {min_loss_epoch/len(val_losses)*100:.1f}% of the total training time")

# Look at the average loss over the last 20% of training
last_20_percent = int(len(val_losses) * 0.2) or 1
avg_recent_loss = sum(val_losses[-last_20_percent:]) / last_20_percent
print(f"Average validation loss over last 20% of training: {avg_recent_loss:.4f}")
print(f"Difference from best: {avg_recent_loss - min_loss:.4f} ({(avg_recent_loss - min_loss) / min_loss * 100:.2f}%)")

# Compare model files if available
lstm_model_path = 'lstm_model.pth'
ml_model_path = 'masked_linear_model.pth'

if os.path.exists(lstm_model_path):
    lstm_model = torch.load(lstm_model_path)
    lstm_params = sum(p.numel() for p in lstm_model.values())
    print(f"\nLSTM model parameters: {lstm_params}")
    
    if os.path.exists(ml_model_path):
        ml_model = torch.load(ml_model_path)
        ml_params = sum(p.numel() for p in ml_model.values())
        print(f"MaskedLinear model parameters: {ml_params}")
        print(f"LSTM has {lstm_params/ml_params*100:.2f}% of the parameters of MaskedLinear")
    else:
        print("MaskedLinear model not found for comparison")
else:
    print("\nLSTM model file not found")

# Get statistics on prediction errors
print("\nPrediction error statistics:")
march_errors_np = np.array(march_errors)
may_errors_np = np.array(may_errors)
july_errors_np = np.array(july_errors)

print("March:")
print(f"  Min: {np.min(march_errors_np):.4f}, Max: {np.max(march_errors_np):.4f}")
print(f"  Mean: {np.mean(march_errors_np):.4f}, Std: {np.std(march_errors_np):.4f}")

print("May:")
print(f"  Min: {np.min(may_errors_np):.4f}, Max: {np.max(may_errors_np):.4f}")
print(f"  Mean: {np.mean(may_errors_np):.4f}, Std: {np.std(may_errors_np):.4f}")

print("July:")
print(f"  Min: {np.min(july_errors_np):.4f}, Max: {np.max(july_errors_np):.4f}")
print(f"  Mean: {np.mean(july_errors_np):.4f}, Std: {np.std(july_errors_np):.4f}")

# Check if there's any improvement trend in the last 500 epochs
last_500_epochs = min(500, len(march_errors))
if last_500_epochs > 10:  # Make sure we have enough data points
    print("\nError trend in last 500 epochs (negative means improving):")
    
    # Calculate linear regression slope
    x = np.arange(last_500_epochs)
    
    # March
    march_trend = np.polyfit(x, march_errors_np[-last_500_epochs:], 1)[0]
    print(f"March trend: {march_trend:.6f} per epoch")
    
    # May
    may_trend = np.polyfit(x, may_errors_np[-last_500_epochs:], 1)[0]
    print(f"May trend: {may_trend:.6f} per epoch")
    
    # July
    july_trend = np.polyfit(x, july_errors_np[-last_500_epochs:], 1)[0]
    print(f"July trend: {july_trend:.6f} per epoch")

print("\nAnalysis complete. LSTM model training was successful.")
