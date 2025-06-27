from dataset import load_FY_Dataset
import torch
from model import MaskedLinear
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = load_FY_Dataset(csv_files=["datasets/test.csv"], batch_size=16)

model = MaskedLinear(seq_len=12, input_dim=3, output_dim=1, hidden_dim=4).to(device)
criterion = nn.MSELoss().to(device)
# criterion = nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

curve_1 = []
curve_2 = []
curve_3 = []
curve_4 = []

def eval():
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
    curve_1.append(avg_loss)
    curve_2.append(eval_outputs[2].cpu().item())
    curve_3.append(eval_outputs[4].cpu().item())
    curve_4.append(eval_outputs[6].cpu().item())
    # print(f"Validation Loss: {avg_loss:.4f} | {eval_outputs}")

def train():
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
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloaders['train'])
    # print(f"Training Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    num_epochs = 1500
    for epoch in tqdm(range(num_epochs)):
        train()
        eval()
    
    # Save the model
    torch.save(model.state_dict(), "masked_linear_model.pth")
    print("Model saved as masked_linear_model.pth")

    curve_1 = gaussian_filter1d(np.array(curve_1), sigma=3).tolist()
    curve_2 = gaussian_filter1d(np.array(curve_2), sigma=3).tolist()
    curve_3 = gaussian_filter1d(np.array(curve_3), sigma=3).tolist()
    curve_4 = gaussian_filter1d(np.array(curve_4), sigma=3).tolist()


    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(curve_1, label='Validation Loss')
    # plt.plot(curve_2, label='March')
    # plt.plot(curve_3, label='May')
    # plt.plot(curve_4, label='July')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    
    plt.figure(figsize=(10, 10))
    # plt.plot(curve_1, label='Validation Loss')
    plt.plot(curve_2, label='March')
    plt.plot(curve_3, label='May')
    plt.plot(curve_4, label='July')
    plt.axhline(y=0.05, color='r', linestyle='--', label='target Line')
    plt.axhline(y=-0.05, color='r', linestyle='--', label='target Line')
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()