from dataset import load_FY_Dataset
import torch
from model import MaskedLinear
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

dataloaders = load_FY_Dataset(csv_files=["datasets/test.csv"], batch_size=32)

model = MaskedLinear(seq_len=12, input_dim=3, output_dim=1, hidden_dim=3)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def eval():
    model.eval()
    total_loss = 0.0
    sum_outputs = torch.zeros(12)
    sum_targets = torch.zeros(12)
    with torch.no_grad():
        for batch in tqdm(dataloaders['val'], desc="Evaluating"):
            inputs, targets = batch
            outputs = model(inputs)
            outputs = outputs.reshape(-1, 12)
            sum_outputs += outputs.sum(dim=0)
            sum_targets += targets.sum(dim=0)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloaders['val'])
    eval_outputs = (sum_outputs - sum_targets) / sum_targets
    print(f"Validation Loss: {avg_loss:.4f} | {eval_outputs}")

def train():
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloaders['train'], desc="Training"):
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.reshape(-1, 12)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloaders['train'])
    print(f"Training Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    num_epochs = 1000
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train()
        eval()
        print("-" * 30)
    
    # Save the model
    torch.save(model.state_dict(), "masked_linear_model.pth")
    print("Model saved as masked_linear_model.pth")