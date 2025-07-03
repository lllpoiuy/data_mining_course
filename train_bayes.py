import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
import glob
import math
from dataset import FY_Dataset, load_FY_Dataset
import sys
import traceback
from model import MaskedLinear

def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define a simple Bayesian model
class SimpleBayesianNetwork(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim=1, dropout_rate=0.2):
        super(SimpleBayesianNetwork, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = seq_len * input_dim
        self.output_size = seq_len * output_dim
        self.hidden_dim = hidden_dim
        
        print(f"Network dimensions: input_size={self.input_size}, hidden_dim={hidden_dim}, output_size={self.output_size}")
        
        # First layer with weight uncertainty
        self.fc1_mu = nn.Linear(self.input_size, hidden_dim)
        # 初始化 sigma 参数为较小的值，以减少 KL 散度
        self.fc1_sigma = nn.Parameter(torch.ones(hidden_dim) * -6.0)
        
        # Second layer
        self.fc2_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_sigma = nn.Parameter(torch.ones(hidden_dim) * -6.0)
        
        # Output layer
        self.fc3_mu = nn.Linear(hidden_dim, self.output_size)
        self.fc3_sigma = nn.Parameter(torch.ones(self.output_size) * -6.0)
        
        # 初始化权重
        self._init_weights()
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # For tracking KL divergence
        self.kl = 0
    
    def _init_weights(self):
        """初始化网络权重"""
        # 使用 Kaiming 初始化均值参数
        nn.init.kaiming_normal_(self.fc1_mu.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1_mu.bias, 0)
        
        nn.init.kaiming_normal_(self.fc2_mu.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc2_mu.bias, 0)
        
        nn.init.kaiming_normal_(self.fc3_mu.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc3_mu.bias, 0)
        
    def forward(self, x, sample=True):
        try:
            # Reshape input to (batch_size, seq_len * input_dim)
            batch_size = x.size(0)
            
            # 只在调试模式下打印形状信息
            debug = False
            if debug:
                print(f"Input shape: {x.shape}")
            
            x = x.view(batch_size, -1)
            
            if debug:
                print(f"Flattened input shape: {x.shape}")
            
            # Layer 1 with uncertainty
            mu1 = self.fc1_mu(x)
            
            if debug:
                print(f"Layer 1 output shape: {mu1.shape}")
            
            if sample:
                # 使用 softplus 将 rho 转换为 sigma，确保方差为正
                sigma1 = torch.nn.functional.softplus(self.fc1_sigma)
                epsilon = torch.randn_like(sigma1)
                h1 = mu1 + epsilon * sigma1
                # 计算 KL 散度，使用更稳定的公式
                self.kl = 0.5 * torch.sum(
                    sigma1**2 + mu1**2 - torch.log(sigma1**2 + 1e-8) - 1
                ) / batch_size  # 除以 batch_size 使其与 MSE 损失尺度相当
            else:
                h1 = mu1
                self.kl = 0
            
            h1 = self.relu(h1)
            h1 = self.dropout(h1)
            
            # Layer 2 with uncertainty
            mu2 = self.fc2_mu(h1)
            
            if debug:
                print(f"Layer 2 output shape: {mu2.shape}")
            
            if sample:
                sigma2 = torch.nn.functional.softplus(self.fc2_sigma)
                epsilon = torch.randn_like(sigma2)
                h2 = mu2 + epsilon * sigma2
                # 计算 KL 散度
                self.kl += 0.5 * torch.sum(
                    sigma2**2 + mu2**2 - torch.log(sigma2**2 + 1e-8) - 1
                ) / batch_size
            else:
                h2 = mu2
            
            h2 = self.relu(h2)
            h2 = self.dropout(h2)
            
            # Output layer with uncertainty
            mu_out = self.fc3_mu(h2)
            
            if debug:
                print(f"Output layer shape: {mu_out.shape}, expected output_size: {self.output_size}")
            
            if sample:
                sigma_out = torch.nn.functional.softplus(self.fc3_sigma)
                epsilon = torch.randn_like(sigma_out)
                output = mu_out + epsilon * sigma_out
                # 计算 KL 散度
                self.kl += 0.5 * torch.sum(
                    sigma_out**2 + mu_out**2 - torch.log(sigma_out**2 + 1e-8) - 1
                ) / batch_size
            else:
                output = mu_out
                sigma_out = torch.zeros_like(mu_out)
            
            # 修复维度重塑问题
            if debug:
                print(f"Raw output shape: {output.shape}")
            
            # 如果输出是 [batch_size, seq_len]，直接添加输出维度
            if output.shape[1] == self.seq_len:
                output = output.unsqueeze(-1)  # 变成 [batch_size, seq_len, 1]
                # 为 sigma_out 创建相同形状的零张量
                sigma_out = torch.zeros_like(output)
            # 如果输出是 [batch_size, seq_len * output_dim]，尝试重塑
            elif output.shape[1] == self.seq_len * self.output_dim:
                try:
                    output = output.view(batch_size, self.seq_len, self.output_dim)
                    sigma_out = sigma_out.view(batch_size, self.seq_len, self.output_dim)
                except RuntimeError:
                    # 如果重塑失败，使用原始输出并添加新维度
                    output = output.unsqueeze(-1)
                    sigma_out = sigma_out.unsqueeze(-1)
            else:
                # 如果形状不匹配预期，只添加新维度
                output = output.unsqueeze(-1)
                sigma_out = sigma_out.unsqueeze(-1)
            
            if debug:
                print(f"Final output shape: {output.shape}, sigma shape: {sigma_out.shape}")
            
            return output, sigma_out
        except Exception as e:
            print(f"Exception in forward pass: {e}")
            traceback.print_exc()
            raise e
    
    def kl_loss(self):
        return self.kl
    
    def loss_function(self, y_pred, y_true, kl_weight=1.0, mse_weight=1.0):
        try:
            mean, std = y_pred
            
            # 只在调试模式下打印形状信息
            debug = False
            if debug:
                print(f"Target shape: {y_true.shape}, Prediction shape: {mean.shape}")
            
            # 确保维度匹配
            # 如果预测是 [batch_size, seq_len, output_dim]
            if mean.dim() == 3:
                # 如果目标是 [batch_size, seq_len]
                if y_true.dim() == 2:
                    y_true = y_true.unsqueeze(-1)  # 变成 [batch_size, seq_len, 1]
                    if debug:
                        print(f"Reshaped target shape: {y_true.shape}")
            
            # 如果预测是 [batch_size, seq_len, 1, 1] 或其他额外维度
            if mean.dim() > 3:
                # 移除额外的维度
                mean = mean.squeeze()
                std = std.squeeze()
                # 确保至少保留三个维度 [batch_size, seq_len, output_dim]
                if mean.dim() < 3:
                    mean = mean.unsqueeze(-1)
                    std = std.unsqueeze(-1)
                if debug:
                    print(f"Adjusted prediction shape: {mean.shape}")
            
            # 最终检查维度匹配
            if mean.shape != y_true.shape and debug:
                print(f"Warning: Shapes still don't match. Mean: {mean.shape}, Target: {y_true.shape}")
                # 尝试广播
                if mean.dim() == y_true.dim():
                    # 如果维度数相同，我们可以尝试广播
                    pass
                else:
                    # 否则，调整目标维度
                    while mean.dim() > y_true.dim():
                        y_true = y_true.unsqueeze(-1)
                    while mean.dim() < y_true.dim():
                        mean = mean.unsqueeze(-1)
                        std = std.unsqueeze(-1)
                    print(f"After dimension adjustment: Mean: {mean.shape}, Target: {y_true.shape}")
            
            # MSE loss
            mse_loss = F.mse_loss(mean, y_true)
            
            # KL divergence
            kl_loss = self.kl_loss()
            
            # Combined loss
            total_loss = mse_weight * mse_loss + kl_weight * kl_loss
            
            return total_loss, mse_loss, kl_loss
        except Exception as e:
            print(f"Exception in loss function: {e}")
            traceback.print_exc()
            raise e

def train_epoch(model, train_loader, optimizer, device, epoch, kl_annealing_factor=1.0, clip_value=1.0):
    """Train for one epoch with KL annealing and gradient clipping"""
    model.train()
    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_kl = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            data, target = data.to(device), target.to(device)
            
            # 只在第一个批次打印调试信息
            if batch_idx == 0:
                print(f"Batch {batch_idx}, data shape: {data.shape}, target shape: {target.shape}")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(data)
            
            # Calculate loss with annealed KL
            loss, mse, kl = model.loss_function(pred, target, kl_weight=kl_annealing_factor)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            # Update parameters
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_mse += mse.item()
            epoch_kl += kl.item()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:  # 每 10 个批次打印一次
                print(f"  Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"Exception in train_epoch: {e}")
            traceback.print_exc()
            raise e
    
    avg_loss = epoch_loss / len(train_loader)
    avg_mse = epoch_mse / len(train_loader)
    avg_kl = epoch_kl / len(train_loader)
    
    return avg_loss, avg_mse, avg_kl

def validate(model, val_loader, device, epoch, kl_annealing_factor=1.0):
    """Validate model performance"""
    model.eval()
    val_loss = 0.0
    val_mse = 0.0
    val_kl = 0.0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            pred = model(data)
            
            # Calculate loss
            loss, mse, kl = model.loss_function(pred, target, kl_weight=kl_annealing_factor)
            
            # Accumulate metrics
            val_loss += loss.item()
            val_mse += mse.item()
            val_kl += kl.item()
    
    avg_loss = val_loss / len(val_loader)
    avg_mse = val_mse / len(val_loader)
    avg_kl = val_kl / len(val_loader)
    
    return avg_loss, avg_mse, avg_kl

def test(model, test_loader, device):
    """Test model performance with uncertainty estimation"""
    model.eval()
    predictions = []
    ground_truth = []
    uncertainties = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass with uncertainty
            mean, std = model(data, sample=True)
            
            # Store results
            predictions.append(mean.cpu().numpy())
            uncertainties.append(std.cpu().numpy())
            ground_truth.append(target.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(predictions, axis=0)
    uncertainties = np.concatenate(uncertainties, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    # Ensure ground truth has correct shape for comparison
    if ground_truth.ndim == 2:  # If shape is [batch, seq_len]
        ground_truth = ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1], 1)
    
    # Calculate metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - ground_truth))
    
    # Calculate calibration metrics - coverage probability
    z_score = 1.96  # 95% confidence interval
    lower_bound = predictions - z_score * uncertainties
    upper_bound = predictions + z_score * uncertainties
    
    in_interval = np.logical_and(ground_truth >= lower_bound, ground_truth <= upper_bound)
    coverage = np.mean(in_interval)
    
    # Average uncertainty
    avg_uncertainty = np.mean(uncertainties)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'coverage': coverage,
        'avg_uncertainty': avg_uncertainty
    }
    
    return metrics, predictions, uncertainties, ground_truth

def plot_results(train_losses, val_losses, train_mses, val_mses, train_kls, val_kls):
    """Plot training curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot total loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    
    # Plot MSE
    plt.subplot(1, 3, 2)
    plt.plot(train_mses, label='Train MSE')
    plt.plot(val_mses, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE Loss')
    plt.legend()
    
    # Plot KL
    plt.subplot(1, 3, 3)
    plt.plot(train_kls, label='Train KL')
    plt.plot(val_kls, label='Validation KL')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bayesian_training_curves.png')
    plt.close()

def plot_predictions(predictions, uncertainties, ground_truth, n_samples=3, seq_len=12):
    """Plot predictions with uncertainty for random samples"""
    plt.figure(figsize=(15, 10))
    
    # Ensure ground_truth has the right shape for comparison
    if ground_truth.ndim == 2:
        ground_truth = ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1], 1)
    
    # Process predictions to make them more interpretable
    # Take predictions for a few time steps to show the sequence
    n_timesteps = min(5, seq_len)  # Show up to 5 time steps
    
    for i in range(n_samples):
        # Randomly select a sample
        idx = np.random.randint(0, len(predictions))
        
        plt.subplot(n_samples, 1, i+1)
        
        # Plot time steps on x-axis
        x = np.arange(n_timesteps)
        
        # Plot ground truth
        plt.plot(x, ground_truth[idx, :n_timesteps, 0], 'ko-', label='True')
        
        # Plot prediction
        plt.plot(x, predictions[idx, :n_timesteps, 0], 'bo-', label='Prediction')
        
        # Plot uncertainty
        plt.fill_between(
            x,
            predictions[idx, :n_timesteps, 0] - 2 * uncertainties[idx, :n_timesteps, 0],
            predictions[idx, :n_timesteps, 0] + 2 * uncertainties[idx, :n_timesteps, 0],
            color='b', alpha=0.2, label='95% Confidence'
        )
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Sample {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('bayesian_predictions.png')
    plt.close()

def main():
    try:
        parser = argparse.ArgumentParser(description='Train Bayesian Time Series Model')
        parser.add_argument('--data_dir', type=str, default='datasets', help='Path to datasets directory')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
        parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for L2 regularization')
        parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--kl_annealing', type=int, default=50, help='KL annealing epochs')
        parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler (plateau or cosine)')
        parser.add_argument('--early_stopping', action='store_true', help='Use early stopping')
        parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
        parser.add_argument('--clip_value', type=float, default=1.0, help='Gradient clipping value')
        parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training')
        parser.add_argument('--kl_weight', type=float, default=0.01, help='Base weight for KL divergence term')
        
        args = parser.parse_args()
        
        # Configuration from arguments
        config = vars(args)
        config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config['save_dir'] = './'
        
        print(f"Using device: {config['device']}")
        print("Loading and preparing data...")
        
        # Set seeds for reproducibility
        seed_everything()
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(config['data_dir'], '*.csv'))
        
        # Load dataset using the project's dataset module
        data_loaders = load_FY_Dataset(
            csv_files=csv_files,
            batch_size=config['batch_size'],
            train_ratio=config['train_ratio']
        )
        
        # Create test dataset (from the validation set for this example)
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        # Get sample batch to determine dimensions
        for sample_batch in train_loader:
            if sample_batch[0] is not None and len(sample_batch[0].shape) >= 2:
                break
        
        # Analyze the structure of the data
        print(f"Sample batch shapes: Input {sample_batch[0].shape}, Target {sample_batch[1].shape}")
        
        # Determine dimensions based on the data structure from the FY_Dataset
        # From dataset.py, we know that x shape is [batch_size, seq_len=12, input_dim=3]
        # and target shape is [batch_size, output_dim=12] (one value per time step)
        seq_len = sample_batch[0].shape[1]  # Should be 12
        input_dim = sample_batch[0].shape[2]  # Should be 3
        output_dim = 1  # Output dimension is 1 value per sequence position
        
        print(f"Data shapes: seq_len={seq_len}, input_dim={input_dim}, output_dim={output_dim}")
        
        # Initialize model (using our simplified Bayesian model)
        model = SimpleBayesianNetwork(
            seq_len=seq_len,
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=output_dim,
            dropout_rate=config['dropout']
        ).to(config['device'])
        
        # Print model summary
        print(f"Model: {model}")
        
        # Initialize optimizer with weight decay
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize scheduler
        if config['scheduler'] == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                verbose=True
            )
        else:  # cosine
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # Restart every 10 epochs
                T_mult=2,  # Double period after each restart
                eta_min=1e-6  # Minimum learning rate
            )
        
        # Training loop
        train_losses, val_losses = [], []
        train_mses, val_mses = [], []
        train_kls, val_kls = [], []
        curve_march, curve_may, curve_july = [], [], []

        best_val_loss = float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        
        print("Starting training...")
        start_time = time.time()
        
        # 运行多个 epoch
        for epoch in range(300):  # 运行 300 个 epoch 进行公平比较
            # Calculate KL annealing factor
            kl_factor = min(1.0, epoch / config['kl_annealing']) * config['kl_weight']
            
            # Train
            train_loss, train_mse, train_kl = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                config['device'], 
                epoch,
                kl_annealing_factor=kl_factor,
                clip_value=config['clip_value']
            )
            
            # Validate
            val_loss, val_mse, val_kl = validate(
                model, 
                val_loader, 
                config['device'], 
                epoch,
                kl_annealing_factor=kl_factor
            )
            # validate后，插入如下代码
            with torch.no_grad():
                sum_pred = np.zeros(seq_len)
                sum_true = np.zeros(seq_len)
                for data, target in val_loader:
                    data, target = data.to(config['device']), target.to(config['device'])
                    pred, _ = model(data, sample=False)
                    pred = pred.squeeze(-1).cpu().numpy()  # [batch, seq_len]
                    target = target.cpu().numpy()
                    sum_pred += pred.sum(axis=0)
                    sum_true += target.sum(axis=0)
                eval_outputs = (sum_pred - sum_true) / (sum_true + 1e-8)
                curve_march.append(eval_outputs[2])
                curve_may.append(eval_outputs[4])
                curve_july.append(eval_outputs[6])
            # Print progress
            print(f"Epoch: {epoch+1}/{300} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | "
                f"Train KL: {train_kl:.4f} | Val KL: {val_kl:.4f} | "
                f"KL Factor: {kl_factor:.4f}")
            
            # Update scheduler
            if config['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_mses.append(train_mse)
            val_mses.append(val_mse)
            train_kls.append(train_kl)
            val_kls.append(val_kl)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                
                # Save best model
                torch.save(model.state_dict(), os.path.join(config['save_dir'], 'bayesian_best_model.pth'))
                print(f"New best model saved at epoch {epoch+1}")
            else:
                epochs_no_improve += 1
                
            # Early stopping
            if config['early_stopping'] and epochs_no_improve >= config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        training_time = time.time() - start_time
        print(f"Training finished in {training_time:.2f} seconds.")
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'bayesian_best_model.pth')))
        
        # Evaluate on test set (using validation set as test set for this example)
        print("Evaluating Bayesian model on test set...")
        metrics, predictions, uncertainties, ground_truth = test(
            model, val_loader, config['device']
        )
        
        # Print metrics
        print("Bayesian Model Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 评估MaskedLinear模型
        print("\nEvaluating MaskedLinear model on test set...")

        masked_model = MaskedLinear(seq_len=seq_len, input_dim=input_dim, output_dim=output_dim, hidden_dim=5)
        masked_model.load_state_dict(torch.load("masked_linear_model.pth", map_location=config['device']))
        masked_model = masked_model.to(config['device'])
        masked_model.eval()
        masked_preds = []
        masked_targets = []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(config['device'])
                target = target.to(config['device'])
                out = masked_model(data)
                masked_preds.append(out.cpu().numpy())
                masked_targets.append(target.cpu().numpy())
        masked_preds = np.concatenate(masked_preds, axis=0)
        masked_targets = np.concatenate(masked_targets, axis=0)
        # 保证形状一致
        if masked_preds.shape[-1] == 1:
            masked_preds = masked_preds.squeeze(-1)
        if masked_targets.shape[-1] == 1:
            masked_targets = masked_targets.squeeze(-1)
        masked_mse = np.mean((masked_preds - masked_targets) ** 2)
        masked_mae = np.mean(np.abs(masked_preds - masked_targets))
        print(f"MaskedLinear Model Test Metrics:\nMSE: {masked_mse:.4f}\nMAE: {masked_mae:.4f}")
        # 绘制对比图
        plt.figure(figsize=(12, 6))
        for i in range(5):
            plt.subplot(2, 3, i+1)
            x = np.arange(seq_len)
            plt.plot(x, ground_truth[i, :, 0], 'ko-', label='True')
            plt.plot(x, predictions[i, :, 0], 'bo-', label='Bayesian')
            plt.fill_between(
                x,
                predictions[i, :, 0] - 2 * uncertainties[i, :, 0],
                predictions[i, :, 0] + 2 * uncertainties[i, :, 0],
                color='b', alpha=0.2, label='Bayesian 95% CI'
            )
            plt.plot(x, masked_preds[i], 'ro-', label='MaskedLinear')
            plt.title(f'Sample {i+1}')
            if i == 0:
                plt.legend()
        plt.suptitle('Model Prediction Comparison')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('model_comparison.png')
        plt.close()

        # ========================
        # 训练结束后统一绘制曲线
        # ========================
        
        # 平滑处理，风格与MaskedLinear一致
        curve_march_smooth = gaussian_filter1d(np.array(curve_march), sigma=3)
        curve_may_smooth = gaussian_filter1d(np.array(curve_may), sigma=3)
        curve_july_smooth = gaussian_filter1d(np.array(curve_july), sigma=3)
        train_losses_smooth = gaussian_filter1d(np.array(train_losses), sigma=3)
        val_losses_smooth = gaussian_filter1d(np.array(val_losses), sigma=3)

        # March/May/July曲线
        plt.figure(figsize=(10, 6))
        plt.plot(curve_march_smooth, label='March')
        plt.plot(curve_may_smooth, label='May')
        plt.plot(curve_july_smooth, label='July')
        plt.axhline(y=0.05, color='r', linestyle='--', label='target Line')
        plt.axhline(y=-0.05, color='r', linestyle='--')
        plt.ylim(-0.5, 0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('bayesian_training_curves_custom.png')
        plt.close()

        # Loss曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_smooth, label='Training Loss')
        plt.plot(val_losses_smooth, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('bayesian_training_history_custom.png')
        plt.close()

        # ========================
        # 保留原有的plot_results和plot_predictions
        # ========================
        plot_results(train_losses, val_losses, train_mses, val_mses, train_kls, val_kls)
        plot_predictions(predictions, uncertainties, ground_truth, n_samples=3, seq_len=seq_len)
        
        # Save configuration and metrics
        import json
        with open(os.path.join(config['save_dir'], 'bayesian_model_results.json'), 'w') as f:
            results = {
                'config': {k: str(v) if isinstance(v, torch.device) else v for k, v in config.items()},
                'metrics': {k: float(v) for k, v in metrics.items()},
                'best_epoch': best_epoch,
                'training_time': training_time
            }
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {os.path.join(config['save_dir'], 'bayesian_model_results.json')}")
        
    except Exception as e:
        print(f"Exception in main: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 