# Data Mining Course - Sequence Prediction

This project implements a simple neural network for sequence prediction using PyTorch.

## Environment Setup

```python
conda create -n env_data_mining python==3.12
conda activate env_data_mining
pip install -r requirements.txt
```

## Project Structure

- `dataset.py`: Contains dataset loading and preprocessing functions
- `train.py`: Implements the neural network model and training process
- `datasets/`: Directory containing the CSV and Excel data files
- `requirements.txt`: Required Python packages

## Model Architecture

The sequence prediction model uses an LSTM-based architecture:

1. Input: Time series data of shape [batch_size, 3, 12] where:
   - First dimension is the batch size
   - Second dimension represents 3 different sequences
   - Third dimension represents 12 time steps (months)

2. Model Components:
   - LSTM layers for sequence modeling
   - Fully connected layer for output projection
   - Linear layer for prediction length adjustment

3. Output: Predicted sequence of shape [batch_size, 3, 6] representing the next 6 months of data

## Usage

To train the model:

```bash
python train.py
```

This will:
1. Load the data from CSV files
2. Create and train the sequence prediction model
3. Save the trained model as `sequence_predictor_model.pth`
4. Generate a training history plot as `training_history.png`

## Customization

You can modify the model parameters in the `train.py` file:
- `input_dim`: Input dimension (default: 3)
- `hidden_dim`: Hidden dimension of the LSTM (default: 64)
- `num_layers`: Number of LSTM layers (default: 2)
- `output_dim`: Output dimension (default: 3)
- `pred_len`: Prediction length (default: 6)