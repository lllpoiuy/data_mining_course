import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedLinear(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim=None):
        """
        Initialize the MaskedLinear layer.
        
        Args:
            seq_len (int): Length of the sequence.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int, optional): Dimension of the output features. If None, defaults to input_dim.
        """
        super(MaskedLinear, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # First linear layer: seq_len * input_dim -> seq_len * hidden_dim
        self.linear1_weight = nn.Parameter(torch.Tensor(seq_len * input_dim, seq_len * hidden_dim))
        self.linean1_bias = nn.Parameter(torch.Tensor(seq_len * hidden_dim))
        
        # Second linear layer: seq_len * hidden_dim -> seq_len * output_dim
        self.linear2_weight = nn.Parameter(torch.Tensor(seq_len * hidden_dim, seq_len * self.output_dim))
        self.linear2_bias = nn.Parameter(torch.Tensor(seq_len * self.output_dim))

        # self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU()
        self.activation = nn.ReLU()

        # Create mask for the first layer (input_dim -> hidden_dim)
        self.mask1 = torch.zeros(self.seq_len * self.input_dim, self.seq_len * self.hidden_dim)
        for i in range(self.seq_len):
            for j in range(i, self.seq_len):
                self.mask1[i * self.input_dim:(i + 1) * self.input_dim, j * self.hidden_dim:(j + 1) * self.hidden_dim] = 1.0
        # for i in range(self.seq_len):
            # self.mask1[i * self.input_dim:(i + 1) * self.input_dim, i * self.hidden_dim:(i + 1) * self.hidden_dim] = 1.0
        self.mask1 = nn.Parameter(self.mask1, requires_grad=False)

        # Create mask for the second layer (hidden_dim -> output_dim)
        self.mask2 = torch.zeros(self.seq_len * self.hidden_dim, self.seq_len * self.output_dim)
        for i in range(self.seq_len):
            for j in range(i, self.seq_len):
                self.mask2[i * self.hidden_dim:(i + 1) * self.hidden_dim, j * self.output_dim:(j + 1) * self.output_dim] = 1.0
        self.mask2 = nn.Parameter(self.mask2, requires_grad=False)

        # print(self.mask1)
        # print(self.mask2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear1_weight, a=math.sqrt(5))
        nn.init.zeros_(self.linean1_bias)
        nn.init.kaiming_uniform_(self.linear2_weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear2_bias)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.seq_len * self.input_dim)
        
        masked_weight1 = self.linear1_weight * self.mask1  # (seq_len*input_dim, seq_len*hidden_dim)
        x = torch.matmul(x, masked_weight1) + self.linean1_bias  # (batch_size, seq_len*hidden_dim)
        x = self.activation(x)

        masked_weight2 = self.linear2_weight * self.mask2  # (seq_len*hidden_dim, seq_len*output_dim)
        x = torch.matmul(x, masked_weight2) + self.linear2_bias  # (batch_size, seq_len*output_dim)

        x = x.reshape(batch_size, self.seq_len, self.output_dim)
        return x


class FeatureNormalization(nn.Module):
    """
    Feature normalization layer that normalizes each feature independently.
    Helps stabilize training and potentially improve model performance.
    """
    def __init__(self, eps=1e-8):
        """
        Initialize the feature normalization layer.
        
        Args:
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        """
        Normalize features across time dimension.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and std across time dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        return (x - mean) / std


class SimpleAttention(nn.Module):
    """
    Simple attention mechanism for sequence data.
    Allows the model to focus on relevant parts of the input sequence.
    """
    def __init__(self, hidden_dim):
        """
        Initialize attention layer.
        
        Args:
            hidden_dim (int): Dimension of the hidden features
        """
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, sequence):
        """
        Apply attention to sequence data.
        
        Args:
            sequence: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Weighted sequence tensor
        """
        # Calculate attention weights
        attn_weights = F.softmax(self.attn(sequence), dim=1)
        # Apply attention weights to sequence
        return sequence * attn_weights


class OptimizedLSTM(nn.Module):
    """
    Optimized LSTM model with carefully tuned parameters to prevent overfitting.
    
    This model follows the same input/output format as MaskedLinear to ensure comparability,
    but utilizes LSTM architecture with advanced optimization techniques like:
    - Feature normalization
    - Attention mechanism
    - Residual connections
    - Improved activation functions
    - Careful weight initialization
    - Layer normalization
    - Dropout regularization
    """
    
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim=None, num_layers=2):
        """
        Initialize the OptimizedLSTM model.
        
        Args:
            seq_len (int): Length of the sequence.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden LSTM state.
            output_dim (int, optional): Dimension of the output features. If None, defaults to input_dim.
            num_layers (int): Number of LSTM layers. Default is 2 for better feature extraction.
        """
        super(OptimizedLSTM, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_layers = num_layers
        
        # Feature normalization to stabilize training
        self.feature_norm = FeatureNormalization(eps=1e-6)
        
        # Input projection to increase dimensionality before LSTM
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layer with dropout to prevent overfitting
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # (batch_size, seq_len, input_dim)
            dropout=0.25 if num_layers > 1 else 0,  # Dropout between LSTM layers
            bidirectional=False  # Unidirectional to match the causality in MaskedLinear
        )
        
        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Attention mechanism to focus on important parts of the sequence
        self.attention = SimpleAttention(hidden_dim)
        
        # Output projection with two layers for better expressiveness
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, self.output_dim)
        
        # Additional dropout before output to reduce overfitting
        self.dropout = nn.Dropout(0.2)
        
        # SiLU activation (Swish) often works better than ReLU/GELU for complex relationships
        self.activation = nn.SiLU()
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Custom parameter initialization with constraints to prevent overfitting"""
        # Input projection initialization
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity='linear')
        nn.init.zeros_(self.input_projection.bias)
        
        # LSTM parameters - use orthogonal initialization for recurrent weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input weights
                nn.init.xavier_uniform_(param, gain=0.6)  # Adjusted gain
            elif 'weight_hh' in name:  # Recurrent weights
                nn.init.orthogonal_(param, gain=0.9)  # Increased gain for better gradient flow
            elif 'bias' in name:
                nn.init.zeros_(param)  # Initialize biases to zero
                
        # Output layers with careful initialization
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        """
        Forward pass of the OptimizedLSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size = x.size(0)
        
        # Apply feature normalization
        x_norm = self.feature_norm(x)
        
        # Project input to higher dimension
        x_proj = self.input_projection(x_norm)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x_proj, (h0, c0))
        
        # Apply layer normalization
        normalized = self.layer_norm(lstm_out)
        
        # Apply attention mechanism
        attended = self.attention(normalized)
        
        # Apply residual connection 
        residual = attended + x_proj
        
        # Apply dropout
        dropped = self.dropout(residual)
        
        # Apply first fully connected layer and activation
        fc1_out = self.activation(self.fc1(dropped))
        
        # Apply second dropout
        dropped2 = self.dropout(fc1_out)
        
        # Apply second fully connected layer
        output = self.fc2(dropped2)
        
        return output


if __name__ == "__main__":
    # Example usage
    seq_len = 4
    input_dim = 2
    hidden_dim = 3
    output_dim = 3
    batch_size = 16

    # Example 1: Using with default output_dim (same as input_dim)
    model1 = MaskedLinear(seq_len, input_dim, hidden_dim)
    x1 = torch.ones(batch_size, seq_len, input_dim)
    output1 = model1(x1)
    print("Model 1 output shape:", output1.shape)  # Should be (batch_size, seq_len, input_dim)
    
    # Example 2: Using with custom output_dim
    model2 = MaskedLinear(seq_len, input_dim, hidden_dim, output_dim)
    x2 = torch.ones(batch_size, seq_len, input_dim)
    output2 = model2(x2)
    print("Model 2 output shape:", output2.shape)  # Should be (batch_size, seq_len, output_dim)
    
    # Example 3: Using OptimizedLSTM with custom output_dim
    lstm_model = OptimizedLSTM(seq_len, input_dim, hidden_dim, output_dim)
    x3 = torch.ones(batch_size, seq_len, input_dim)
    output3 = lstm_model(x3)
    print("LSTM Model output shape:", output3.shape)  # Should be (batch_size, seq_len, output_dim)