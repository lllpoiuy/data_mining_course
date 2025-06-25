import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from typing import List, Union, Callable, Optional, Dict, Any

class ExcelDataset(Dataset):
    """
    A Dataset for reading data from Excel files
    """

    def __init__(
        self, 
        excel_files: List[str], 
        sheet_name: List[Union[str, int]],
    ):
        """
        Initialize the Excel dataset
        
        Args:
            excel_files: List of paths to Excel files
            sheet_name: Sheet name or index to read (default: 0, first sheet)
        """
        dfs = []
        assert len(excel_files) == len(sheet_name), "Number of excel files must match number of sheet names"
        for file_path, sheet in zip(excel_files, sheet_name):
            df = pd.read_excel(file_path, sheet_name=sheet)
            dfs.append(df)
        self.data = pd.concat(dfs, ignore_index=True)

    def print(self):
        """
        Print the first few rows of the dataset
        """
        print(self.data.head())
        print(self.data)

if __name__ == "__main__":
    # Example usage
    dataset = ExcelDataset(excel_files=["datasets/FY2021.xlsx"], sheet_name=[0])
    # dataset = ExcelDataset(excel_files=["datasets/FY1920.xlsx"], sheet_name=[0])
    dataset.print()
    # dataset = ExcelDataset(excel_files=["datasets/FY1920.xlsx"], sheet_name=["Capital Forecast"])
