import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import os

REGIONS = ["Market E", "Market F"]
SITES = ["Plant A", "Plant B", "Plant H", "Plant D", "Plant E", "Plant F"]
CATEGORIES = ["Category A", "Category B", "Category C", "Category D", "Category E", "Category F"]

MONTHS_1 = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
MONTHS_2 = ["J", "A", "S", "O", "N", "D", "J.1", "F", "M", "A.1", "M.1", "J.2"]

class FY_Dataset(Dataset):

    def __init__(
        self, 
        excel_files: List[str],
        num: int,
        use_region: bool = False,
        use_site: bool = False,
        use_category: bool = False,
    ):
        """
        Initialize the dataset
        
        Args:
            excel_files (List[str]): List of paths to the CSV files
            num (int): Number of months to process
            use_region (bool): Whether to include region in the processed data
            use_site (bool): Whether to include site in the processed data
            use_category (bool): Whether to include category in the processed data
        """
        super(FY_Dataset, self).__init__()

        print("Reading data from files:", excel_files)

        self.use_region = use_region
        self.use_site = use_site
        self.use_category = use_category
        self.num = num
        dfs = []
        for file_path in excel_files:
            df = pd.read_excel(file_path)
            dfs.append(df)
        self.data = pd.concat(dfs, ignore_index=True)
        self.data = self.data.fillna(0)
        # print(self.data)

        self.processed_data = []
        for _, row in self.data.iterrows():
            month_values_1 = []
            month_values_2 = []
            for i in range(num):
                month_values_1.append(row[MONTHS_1[i]])
                month_values_2.append(row[MONTHS_2[i]])
            target = 0
            for i in range(0, 12):
                target += row[MONTHS_1[i]]
            try:
                processed_row = {
                    "region": REGIONS.index(row["Region"]),
                    "site": SITES.index(row["Site"]),
                    "category": CATEGORIES.index(row["Business Category"]),
                    "Firm": torch.tensor(row["Firm"], dtype=torch.float32),
                    "month_values_1": torch.tensor(month_values_1, dtype=torch.float32),
                    "month_values_2": torch.tensor(month_values_2, dtype=torch.float32),
                    "target": torch.tensor(target, dtype=torch.float32)
                }
            except ValueError as e:
                print(f"Error: {e}")
                continue
            self.processed_data.append(processed_row)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.processed_data)

    def get_shape(self):
        return self.num * 2 + 1 + len(REGIONS) * int(self.use_region) + len(SITES) * int(self.use_site) + len(CATEGORIES) * int(self.use_category)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        if idx >= len(self.processed_data):
            raise IndexError("Index out of range")
        row = self.processed_data[idx]
        features = []
        # v = 1
        v = row["Firm"]
        if self.use_region:
            region_onehot = torch.zeros(len(REGIONS))
            region_onehot[row["region"]] = v
            features.append(region_onehot)
        if self.use_site:
            site_onehot = torch.zeros(len(SITES))
            site_onehot[row["site"]] = v
            features.append(site_onehot)
        if self.use_category:
            category_onehot = torch.zeros(len(CATEGORIES))
            category_onehot[row["category"]] = v
            features.append(category_onehot)
        features.append(row["Firm"].unsqueeze(0))
        features.append(row["month_values_1"])
        features.append(row["month_values_2"])
        sample = torch.cat(features)
        assert sample.shape[0] == self.get_shape(), f"Sample shape mismatch: {sample.shape[0]} != {self.get_shape()}"
        target = row["target"]
        return sample, target

def test_get_dataloader(
    path: str,
    num: int,
    batch_size: int = 32,
    use_region: bool = False,
    use_site: bool = False,
    use_category: bool = False,
) -> DataLoader:
    """    Create a DataLoader for the FY_Dataset
    Args:
        path (str): Path to the dataset file / dictionary of files
        num (int): Number of months to process
        batch_size (int): Batch size for the DataLoader
        use_region (bool): Whether to include region in the processed data
        use_site (bool): Whether to include site in the processed data
        use_category (bool): Whether to include category in the processed data
    Returns:
        DataLoader: DataLoader for the FY_Dataset
    """
    
    
    assert not os.path.isdir(path), "Path should be a file, not a directory for testing."
    excel_files = [path]
    dataset = FY_Dataset(
        excel_files=excel_files,
        num=num,
        use_region=use_region,
        use_site=use_site,
        use_category=use_category
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.get_shape()

def train_get_dataloader(
    path: str,
    num: int,
    batch_size: int = 32,
    use_region: bool = False,
    use_site: bool = False,
    use_category: bool = False,
) -> DataLoader:
    """    Create a DataLoader for the FY_Dataset
    Args:
        path (str): Path to the dataset file / dictionary of files
        num (int): Number of months to process
        batch_size (int): Batch size for the DataLoader
        use_region (bool): Whether to include region in the processed data
        use_site (bool): Whether to include site in the processed data
        use_category (bool): Whether to include category in the processed data
    Returns:
        DataLoader: DataLoader for the FY_Dataset
    """
    
    
    assert os.path.isdir(path), "Path should be a directory containing Excel files."
    excel_files = list(paths for paths in os.listdir(path) if paths.endswith('.xlsx'))
    excel_files = [os.path.join(path, file) for file in excel_files]
    train_dataset = FY_Dataset(
        excel_files=excel_files[:-1],
        num=num,
        use_region=use_region,
        use_site=use_site,
        use_category=use_category
    )
    eval_dataset = FY_Dataset(
        excel_files=[excel_files[-1]],
        num=num,
        use_region=use_region,
        use_site=use_site,
        use_category=use_category
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloaderm = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, eval_dataloaderm, train_dataset.get_shape()

# fy = FY_Dataset(
#     csv_files=[
#         "datasets/Traindata/FY1920.xlsx",
#     ],
#     num = 6
# )
# fy.__getitem__(1)
# get_dataloader(
#     path="datasets/Traindata",
#     num=6,
#     use_region=True,
#     use_site=True,
#     use_category=True
# )