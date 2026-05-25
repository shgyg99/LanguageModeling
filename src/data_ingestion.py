import os
import sys
from pathlib import Path
from datasets import load_dataset
from utils.config_manager import config_manager


class DataIngestion:
    def __init__(self):
        self.config = config_manager
        
        data_config = self.config.get('data', {})
        dataset_config = data_config.get('dataset_config')
        dataset_name = data_config.get('dataset_name')
        
        if isinstance(dataset_name, list) and len(dataset_name) == 2:
            self.dataset_name = dataset_name[0]
            self.dataset_config = dataset_name[1]
        else:
            self.dataset_name = dataset_name
            self.dataset_config = dataset_config
        
        self.data_dir = Path(self.config.get('paths', {}).get('data_raw', "./data/raw"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self):
        """Download and cache dataset"""
        
        # Use datasets library's native caching
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            cache_dir=str(self.data_dir)
            )
        
        return dataset


if __name__ == "__main__":
    dt = DataIngestion()
    dataset = dt.download_dataset()
    