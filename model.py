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