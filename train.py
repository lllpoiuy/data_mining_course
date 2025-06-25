from dataset import load_FY_Dataset
from torch import nn, optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

# Define model hyperparameters
INPUT_SIZE = 36  # Based on dataset (12 months x 3 features per month)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 1  # Predicting the next real number
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RealNumberPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Model for predicting the next real number in a sequence
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            output_size: Size of output (1 for next token prediction)
        """
        super(RealNumberPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(
            input_size=1,  # One number at a time
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 1)
            
        Returns:
            Predicted next token
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply layer normalization
        normalized = self.layer_norm(lstm_out)
        
        # Self-attention mechanism (reshape for attention)
        attn_input = normalized.permute(1, 0, 2)  # seq_len, batch, hidden
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.permute(1, 0, 2)  # batch, seq_len, hidden
        
        # Get the last time step output
        out = attn_output[:, -1, :]
        
        # Fully connected layers with ReLU activation
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def prepare_sequence_data(batch_data):
    """
    Prepare sequence data for training by creating input-target pairs
    
    Args:
        batch_data: Batch of sequences from the dataset
        
    Returns:
        inputs: Input sequences
        targets: Target next tokens
    """
    inputs = []
    targets = []
    device = batch_data.device  # Get the device of input data
    
    for sequence in batch_data:
        # Split into input and target
        for i in range(len(sequence) - 1):
            # Use the first i+1 elements as input
            input_seq = sequence[:i+1].unsqueeze(-1)  # Add feature dimension
            target = sequence[i+1]  # Next element is the target
            
            inputs.append(input_seq)
            targets.append(target)
    
    # Pad sequences to the same length
    max_length = max(len(seq) for seq in inputs)
    padded_inputs = []
    
    for seq in inputs:
        pad_length = max_length - seq.size(0)
        if pad_length > 0:
            # Pad with zeros at the beginning (ensure it's on the same device)
            padding = torch.zeros(pad_length, 1, device=device)
            padded_seq = torch.cat([padding, seq], dim=0)
        else:
            padded_seq = seq
        padded_inputs.append(padded_seq)
    
    # Stack to create batched tensors
    inputs_tensor = torch.stack(padded_inputs)
    targets_tensor = torch.tensor(targets, device=device)
    
    return inputs_tensor, targets_tensor

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """
    Train the model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    model.to(device)
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            
            # Prepare sequence data
            inputs, targets = prepare_sequence_data(batch_data)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                
                # Prepare sequence data
                inputs, targets = prepare_sequence_data(batch_data)
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return history

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Dictionary with training history
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # Load data
    data_loaders = load_FY_Dataset(
        csv_files=["datasets/FY1920.csv", "datasets/FY2021.csv"],
        batch_size=BATCH_SIZE
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # Initialize model
    model = RealNumberPredictor(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=DEVICE
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    torch.save(model.state_dict(), 'real_number_predictor.pth')
    print("Model saved as 'real_number_predictor.pth'")

if __name__ == "__main__":
    main()

