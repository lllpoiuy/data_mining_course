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


class GaussianVariationalLinear(nn.Module):
    def __init__(self, in_features, out_features, mask=None, prior_scale=1.0):
        super(GaussianVariationalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight mean parameters
        self.weight_mu = nn.Parameter(torch.Tensor(in_features, out_features))
        
        # Weight variance parameters (using log-space for numerical stability)
        self.weight_rho = nn.Parameter(torch.Tensor(in_features, out_features))
        
        # Bias mean parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        
        # Bias variance parameters
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Masking matrix
        self.mask = mask
        
        # Prior scale for regularization
        self.prior_scale = prior_scale
        
        # Initialize parameters
        self.reset_parameters()
        
        # Record KL divergence
        self.kl_divergence = 0
        
    def reset_parameters(self):
        # Initialize means with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        
        # Initialize rhos for appropriate scale of posterior variance
        nn.init.constant_(self.weight_rho, -3.0)  # Starting with smaller variance
        nn.init.constant_(self.bias_rho, -3.0)
        
    def _sample_weights(self):
        # Convert rho to sigma using softplus for numerical stability
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # Sample from N(0,1)
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)
        
        # Reparameterization trick to sample from N(mu, sigma)
        weight = self.weight_mu + weight_epsilon * weight_sigma
        bias = self.bias_mu + bias_epsilon * bias_sigma
        
        # Apply mask if provided
        if self.mask is not None:
            weight = weight * self.mask
        
        # Calculate KL divergence between posterior and prior
        # Using scaled prior to control regularization strength
        weight_prior = torch.distributions.Normal(
            torch.zeros_like(self.weight_mu), 
            torch.ones_like(self.weight_mu) * self.prior_scale
        )
        bias_prior = torch.distributions.Normal(
            torch.zeros_like(self.bias_mu), 
            torch.ones_like(self.bias_mu) * self.prior_scale
        )
        
        weight_posterior = torch.distributions.Normal(self.weight_mu, weight_sigma)
        bias_posterior = torch.distributions.Normal(self.bias_mu, bias_sigma)
        
        self.kl_divergence = torch.distributions.kl_divergence(weight_posterior, weight_prior).sum() + \
                             torch.distributions.kl_divergence(bias_posterior, bias_prior).sum()
        
        return weight, bias
    
    def forward(self, x):
        weight, bias = self._sample_weights()
        return F.linear(x, weight, bias)


class BayesianMaskedLinear(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim=None, 
                 num_mc_samples=5, prior_scale=1.0, activation='relu', dropout_rate=0.0):
        """
        Bayesian version of MaskedLinear with variational inference.
        
        Args:
            seq_len (int): Length of the sequence.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int, optional): Dimension of the output features. If None, defaults to input_dim.
            num_mc_samples (int): Number of Monte Carlo samples for prediction.
            prior_scale (float): Scale of the prior distribution (controls regularization strength).
            activation (str): Activation function ('relu', 'tanh', 'leaky_relu').
            dropout_rate (float): Dropout rate for regularization.
        """
        super(BayesianMaskedLinear, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        
        # Create mask for the first layer (input_dim -> hidden_dim)
        self.mask1 = torch.zeros(self.seq_len * self.input_dim, self.seq_len * self.hidden_dim)
        for i in range(self.seq_len):
            for j in range(i, self.seq_len):
                self.mask1[i * self.input_dim:(i + 1) * self.input_dim, j * self.hidden_dim:(j + 1) * self.hidden_dim] = 1.0
        self.mask1 = nn.Parameter(self.mask1, requires_grad=False)

        # Create mask for the second layer (hidden_dim -> output_dim)
        self.mask2 = torch.zeros(self.seq_len * self.hidden_dim, self.seq_len * self.output_dim)
        for i in range(self.seq_len):
            for j in range(i, self.seq_len):
                self.mask2[i * self.hidden_dim:(i + 1) * self.hidden_dim, j * self.output_dim:(j + 1) * self.output_dim] = 1.0
        self.mask2 = nn.Parameter(self.mask2, requires_grad=False)
        
        # First Bayesian layer
        self.layer1 = GaussianVariationalLinear(
            seq_len * input_dim, 
            seq_len * hidden_dim, 
            self.mask1,
            prior_scale=prior_scale
        )
        
        # Second Bayesian layer
        self.layer2 = GaussianVariationalLinear(
            seq_len * hidden_dim, 
            seq_len * self.output_dim, 
            self.mask2,
            prior_scale=prior_scale
        )
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
            
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sample=True):
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
                              or (batch_size, seq_len * input_dim)
            sample (bool): Whether to sample multiple predictions for uncertainty estimation
        
        Returns:
            mean (torch.Tensor): Mean predictions
            std (torch.Tensor): Standard deviation of predictions (uncertainty)
        """
        batch_size = x.size(0)
        
        # Debug dimensions
        # print(f"Input shape: {x.shape}, Expected: [batch_size, {self.seq_len}, {self.input_dim}]")
        
        # Handle different input shapes
        if x.dim() == 3:  # (batch_size, seq_len, input_dim)
            if x.size(2) != self.input_dim:
                raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(2)}")
            x_flat = x.reshape(batch_size, self.seq_len * self.input_dim)
        elif x.dim() == 2:  # (batch_size, seq_len * input_dim)
            if x.size(1) != self.seq_len * self.input_dim:
                raise ValueError(f"Expected flattened input dimension {self.seq_len * self.input_dim}, got {x.size(1)}")
            x_flat = x
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, got shape: {x.shape}")
        
        # Debug dimensions
        # print(f"Flattened shape: {x_flat.shape}, Expected: [batch_size, {self.seq_len * self.input_dim}]")
        # print(f"Layer1 weight shape: {self.layer1.weight_mu.shape}, Expected: [{self.seq_len * self.input_dim}, {self.seq_len * self.hidden_dim}]")
        
        if not sample:
            # Single forward pass
            h = self.dropout(self.activation(self.layer1(x_flat)))
            out = self.layer2(h)
            out = out.reshape(batch_size, self.seq_len, self.output_dim)
            return out, torch.zeros_like(out)  # No uncertainty estimation when sample=False
        
        # Multiple forward passes for MC dropout
        mc_samples = []
        for _ in range(self.num_mc_samples):
            h = self.dropout(self.activation(self.layer1(x_flat)))
            out = self.layer2(h)
            out = out.reshape(batch_size, self.seq_len, self.output_dim)
            mc_samples.append(out)
        
        # Stack samples along a new dimension
        mc_samples = torch.stack(mc_samples, dim=0)  # [num_samples, batch_size, seq_len, output_dim]
        
        # Calculate mean and standard deviation across MC samples
        mean = mc_samples.mean(dim=0)
        std = mc_samples.std(dim=0)
        
        return mean, std
    
    def kl_loss(self):
        """
        Calculate KL divergence loss for variational inference.
        """
        return self.layer1.kl_divergence + self.layer2.kl_divergence


class BayesianTimeSeriesModel(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim=1, 
                 num_mc_samples=5, prior_scale=1.0, activation='relu', 
                 dropout_rate=0.0, hidden_layers=1):
        """
        Bayesian time series model for prediction with uncertainty.
        
        Args:
            seq_len (int): Length of the sequence.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output features.
            num_mc_samples (int): Number of Monte Carlo samples for prediction.
            prior_scale (float): Scale of the prior distribution (controls regularization strength).
            activation (str): Activation function ('relu', 'tanh', 'leaky_relu').
            dropout_rate (float): Dropout rate for regularization.
            hidden_layers (int): Number of hidden layers (1 or 2).
        """
        super(BayesianTimeSeriesModel, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_mc_samples = num_mc_samples
        self.hidden_layers = hidden_layers
        
        if hidden_layers == 1:
            self.bayesian_masked_linear = BayesianMaskedLinear(
                seq_len=seq_len,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_mc_samples=num_mc_samples,
                prior_scale=prior_scale,
                activation=activation,
                dropout_rate=dropout_rate
            )
        else:
            # Multi-layer implementation
            self.layer1 = BayesianMaskedLinear(
                seq_len=seq_len,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,  # Output same dim as input to make it compatible with second layer
                num_mc_samples=num_mc_samples,
                prior_scale=prior_scale,
                activation=activation,
                dropout_rate=dropout_rate
            )
            
            self.layer2 = BayesianMaskedLinear(
                seq_len=seq_len,
                input_dim=input_dim,  # Input from previous layer
                hidden_dim=hidden_dim // 2,  # Smaller hidden dim
                output_dim=output_dim,
                num_mc_samples=num_mc_samples,
                prior_scale=prior_scale * 1.5,  # Different prior for second layer
                activation=activation,
                dropout_rate=dropout_rate
            )
        
    def forward(self, x, sample=True):
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            sample (bool): Whether to sample multiple predictions for uncertainty estimation
            
        Returns:
            tuple: (mean, std) tensors representing predictions and uncertainty
        """
        # Print input shape for debugging
        # print(f"BayesianTimeSeriesModel input: {x.shape}")
        
        if self.hidden_layers == 1:
            return self.bayesian_masked_linear(x, sample)
        else:
            if not sample:
                # Single forward pass
                mean1, _ = self.layer1(x, sample=False)
                mean2, _ = self.layer2(mean1, sample=False)
                return mean2, torch.zeros_like(mean2)
            
            # Multiple forward passes for MC sampling
            mc_samples = []
            for _ in range(self.num_mc_samples):
                mean1, _ = self.layer1(x, sample=True)
                # print(f"Layer 1 output: {mean1.shape}")
                out, _ = self.layer2(mean1, sample=True)
                # print(f"Layer 2 output: {out.shape}")
                mc_samples.append(out)
            
            # Stack samples along a new dimension
            mc_samples = torch.stack(mc_samples, dim=0)
            
            # Calculate mean and standard deviation across MC samples
            mean = mc_samples.mean(dim=0)
            std = mc_samples.std(dim=0)
            
            return mean, std
    
    def kl_loss(self):
        """
        Calculate KL divergence loss for variational inference.
        """
        if self.hidden_layers == 1:
            return self.bayesian_masked_linear.kl_loss()
        else:
            return self.layer1.kl_loss() + self.layer2.kl_loss()
    
    def loss_function(self, y_pred, y_true, kl_weight=1.0, mse_weight=1.0):
        """
        Calculate loss combining NLL and KL divergence.
        
        Args:
            y_pred (tuple): (mean, std) tuple from forward pass
            y_true (torch.Tensor): Target values
            kl_weight (float): Weight for KL divergence term
            mse_weight (float): Weight for MSE term
            
        Returns:
            tuple: (total_loss, mse_loss, kl_loss) tuple
        """
        mean, std = y_pred
        
        # Print shapes for debugging
        # print(f"Mean shape: {mean.shape}, Target shape: {y_true.shape}")
        
        # Reshape target if needed
        if y_true.shape != mean.shape:
            try:
                # If target is [batch, seq_len], we need to reshape it to match [batch, seq_len, output_dim]
                if y_true.dim() == 2 and mean.dim() == 3:
                    # Reshape to [batch, seq_len, 1]
                    y_true = y_true.unsqueeze(-1)
            except RuntimeError as e:
                print(f"Error reshaping target: {e}")
                print(f"Target shape: {y_true.shape}, Mean shape: {mean.shape}")
        
        # MSE loss
        mse_loss = F.mse_loss(mean, y_true)
        
        # Gaussian NLL loss - use only if std is meaningful
        # nll = 0.5 * torch.mean(torch.log(2 * math.pi * std**2 + 1e-8) + 
        #                       (mean - y_true)**2 / (2 * std**2 + 1e-8))
        
        # KL divergence
        kl = self.kl_loss()
        
        # Combined loss - using MSE + KL
        loss = mse_weight * mse_loss + kl_weight * kl
        
        return loss, mse_loss, kl


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
    
    # Example 3: Using Bayesian model
    bayes_model = BayesianTimeSeriesModel(
        seq_len=seq_len,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_mc_samples=5,
        prior_scale=1.0,
        dropout_rate=0.1
    )
    x3 = torch.ones(batch_size, seq_len, input_dim)
    mean, std = bayes_model(x3)
    print("Bayesian model output shape:", mean.shape, std.shape)