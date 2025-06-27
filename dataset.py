import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

class FY_Dataset(Dataset):

    def __init__(
        self, 
        csv_files: List[str], 
    ):
        """
        Initialize the dataset
        
        Args:
            excel_files: List of paths to Excel files
            sheet_name: Sheet name or index to read (default: 0, first sheet)
        """
        dfs = []
        for file_path in csv_files:
            df = pd.read_csv(file_path, index_col=None, header=0)
            dfs.append(df)
        self.data = pd.concat(dfs, ignore_index=True)
        self.data = self.data.fillna(0)

    def test_print(self):
        """For testing purposes, print the first row of the dataset"""
        # print(self.data.to_csv(index=False))
        print(self.data.columns)
        print(self.data.iloc[0])
        print(self.data.iloc[0].values)
        print(self.data.iloc[0].values[7:19])
        print(self.data.iloc[0].values[19:31])

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample1 = self.data.iloc[idx].values[7:19].astype('float32')
        sample2 = self.data.iloc[idx].values[19:31].astype('float32')
        sample3 = sample2.cumsum()

        # Alternate elements from samples
        # alternated = [val for pair in zip(sample1, sample2, sample3) for val in pair]
        # sample = torch.tensor(alternated, dtype=torch.float32)

        # concat sample1, sample2, sample3 to [batch_size, 12, 3]
        sample = torch.stack([torch.tensor(sample1), torch.tensor(sample2), torch.tensor(sample3)], dim=1)

        target = torch.tensor(sample3[-1], dtype=torch.float32).repeat(12)

        return sample, target

def load_FY_Dataset(
    csv_files: List[str],
    batch_size: int = 32,
    train_ratio: float = 0.7
) -> Dict[str, DataLoader]:
    """
    Load csv FY files into PyTorch DataLoaders
    
    Args:
        csv_files: List of paths to CSV files
        batch_size: Batch size for DataLoader
        train_ratio: Ratio of training data to total data (default: 0.8)
        
    Returns:
        Dictionary with 'train' and 'val' DataLoaders
    """

    dataset = FY_Dataset(csv_files=csv_files)
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


if __name__ == "__main__":
    # Example usage
    dataset = FY_Dataset(csv_files=["datasets/test.csv"])
    dataset.test_print()
    print(dataset[0])
    print(len(dataset))
    loaders = load_FY_Dataset(csv_files=["datasets/test.csv"], batch_size=2)
    for batch in loaders['train']:
        print(batch)
    for batch in loaders['val']:
        print(batch)
    print("Done")
