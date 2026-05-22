import os
import sys
import warnings
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict
from transformers import AutoTokenizer
from utils.config_manager import config_manager

warnings.filterwarnings('ignore')

class WikiDataset:
    def __init__(self):
        self.config = config_manager
        self.data_dir = Path(self.config.get('paths', {}).get('data_raw', "./data/raw"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('model', {}).get('tokenizer_name', "bert-base-uncased")
        )
        
        self.seq_len = self.config.get('data', {}).get('processing', {}).get('seq_len', 70)
        self.batch_size = self.config.get('training', {}).get('batch_size', 32)
        
        self.arrow_files = {
            "train": "wikitext-train.arrow",
            "validation": "wikitext-validation.arrow",
            "test": "wikitext-test.arrow"
        }
        
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def load_dataset(self):
        """Loading data from arrow files"""
        dataset = DatasetDict()
        
        for split, filename in self.arrow_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                dataset[split] = HFDataset.from_file(str(file_path))  # استفاده از HFDataset
        
        return dataset if dataset else None
    
    def _tokenize_and_prepare(self, split):
        """Tokenize and prepare data for a given split""" 

        cache_file = self.cache_dir / f"{split}_token_ids.pt"
        
        if cache_file.exists():
            print(f"✅ Loading cached token ids for {split}")
            all_ids = torch.load(cache_file)
        else:
            print(f"🔄 Tokenizing {split}...")
            dataset = self.load_dataset()
            if not dataset or split not in dataset:
                return None
            
            all_ids = []
            for line in dataset[split]['text']:
                if line.strip():
                    token_ids = self.tokenizer.encode(line, add_special_tokens=False)
                    all_ids.extend(token_ids)
            
            torch.save(all_ids, cache_file)
            print(f"✅ Cached token ids for {split}")
        
        return WikiTextDataset(all_ids, self.seq_len)
    
    def prepare_dataloaders(self):
        """Preparing dataloaders"""
        
        print("\n📊 Preparing datasets...")
        self.train_dataset = self._tokenize_and_prepare("train")
        self.val_dataset = self._tokenize_and_prepare("validation")
        self.test_dataset = self._tokenize_and_prepare("test")
        
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True
            )
            print(f"✅ Train loader: {len(self.train_loader)} batches")
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
            print(f"✅ Validation loader: {len(self.val_loader)} batches")
        
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
            print(f"✅ Test loader: {len(self.test_loader)} batches")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataloaders(self):
        """Get dataloaders (prepares them if not ready)"""
        if self.train_loader is None:
            self.prepare_dataloaders()
        return self.train_loader, self.val_loader, self.test_loader


class WikiTextDataset(Dataset):
    """Custom dataset for tokenized data"""
    
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.token_data = torch.LongTensor(token_ids) 
        self.num_sequences = len(self.token_data) // seq_len
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.token_data[start:end]
        y = self.token_data[start+1:end+1]
        return x, y


if __name__ == "__main__":
    wiki = WikiDataset()
    
    train_loader, val_loader, test_loader = wiki.prepare_dataloaders()
    
    if train_loader:
        for batch_x, batch_y in train_loader:
            print(f"Batch X shape: {batch_x.shape}")
            print(f"Batch Y shape: {batch_y.shape}")
            break