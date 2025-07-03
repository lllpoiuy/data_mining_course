import argparse
from utils.dataset import train_get_dataloader
from models.MLP import MLP
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random

def train_single_run(path, num, seed):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model_name = f"MLP_{num}_seed_{seed}"
    print(f"Training model: {model_name} with data from {path}")

    train_dataset, eval_dataset, shape = train_get_dataloader(path, num, use_region=True, use_site=True, use_category=True, batch_size=128)

    model = MLP(input_dim=shape, hidden_dim=15, output_dim=1)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    error_rates = []

    def validate():
        outputs_sum = 0
        targets_sum = 0
        for batch in eval_dataset:
            inputs, targets = batch
            outputs = model(inputs)
            targets = targets.view(-1, 1)
            outputs_sum += outputs.sum().item()
            targets_sum += targets.sum().item()
        
        error_rate = (outputs_sum - targets_sum) * 100 / (targets_sum + 1e-8)
        if error_rate < 0:
            error_rate = -error_rate
        error_rates.append(error_rate)
        
        print(f"Seed {seed} - Validation: Outputs sum = {outputs_sum}, Targets sum = {targets_sum}, Error rate = {error_rate:.4f}")
        return error_rate

    for epoch in range(1, 1000):
        print(f"Seed {seed} - Epoch {epoch}: Training {model_name}...")
        for batch in train_dataset:
            inputs, targets = batch
            outputs = model(inputs)
            targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validate()
    
    return error_rates

def train(path, num):
    num_seeds = 10
    all_error_rates = []
    
    # Train with 10 different seeds
    for seed in range(1, num_seeds + 1):
        error_rates = train_single_run(path, num, seed)
        all_error_rates.append(error_rates)
    
    max_len = max(len(rates) for rates in all_error_rates)
    
    padded_rates = []
    for rates in all_error_rates:
        if len(rates) < max_len:
            padded = rates + [rates[-1]] * (max_len - len(rates))
        else:
            padded = rates
        padded_rates.append(padded)
    
    error_rates_array = np.array(padded_rates)
    
    mean_error = np.mean(error_rates_array, axis=0)
    std_error = np.std(error_rates_array, axis=0)
    upper_bound = mean_error + std_error
    lower_bound = mean_error - std_error
    # upper_bound = np.percentile(error_rates_array, 100, axis=0)
    # lower_bound = np.percentile(error_rates_array, 0, axis=0)
    
    epochs = np.arange(1, max_len + 1)
    
    plt.figure(figsize=(12, 8))
    
    # for i, rates in enumerate(padded_rates):
    #     plt.plot(epochs, rates, alpha=0.2, linewidth=1, label=f'Seed {i+1}' if i == 0 else None)
    
    smoothed_mean = gaussian_filter1d(mean_error, sigma=3)
    plt.plot(epochs, smoothed_mean, 'b-', linewidth=2, label='Mean (Smoothed)')
    
    plt.plot(epochs, mean_error, 'r-', linewidth=1, alpha=0.7, label='Mean')
    
    plt.fill_between(epochs, lower_bound, upper_bound, color='blue', alpha=0.2, label='±1 Std Dev')
    
    plt.title(f'Error Rate Curve for Model MLP_{num} (10 Seeds)')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate (%)')
    plt.grid(True)
    plt.legend()

    ylim = np.percentile(np.abs(mean_error), 90) * 2
    plt.ylim(0, ylim)
    
    plt.savefig(f'error_rate_curve_MLP_{num}_10seeds.png')
    plt.close()
    
    print(f"Error rate curve saved as error_rate_curve_MLP_{num}_10seeds.png")
    
    return mean_error[-1]

