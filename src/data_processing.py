import os
import sys
import warnings
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from transformers import AutoTokenizer
from utils.config_manager import config_manager

warnings.filterwarnings('ignore')

class WikiDataset:
    def __init__(self):
        self.config = config_manager
        self.data_dir = Path(self.config.get('paths', {}).get('data_raw', "./data/raw"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('model', {}).get('tokenizer_name', "bert-base-uncased")
        )
        
        self.seq_len = self.config.get('data', {}).get('processing', {}).get('seq_len', 70)
        self.batch_size = self.config.get('training', {}).get('batch_size', 32)
        
        # دریافت تنظیمات دیتاست از config
        data_config = self.config.get('data', {})
        dataset_names = data_config.get('dataset_name')
        
        if isinstance(dataset_names, list) and len(dataset_names) == 2:
            self.dataset_name = dataset_names[0]
            self.dataset_config = dataset_names[1]
        else:
            self.dataset_name = dataset_names
            self.dataset_config = data_config.get('dataset_config', 'wikitext-103-raw-v1')
        
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def download_and_prepare(self):
        """Download dataset from Hugging Face and prepare it"""
        
        print(f"📥 Loading {self.dataset_name}/{self.dataset_config}...")
        
        raw_dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            cache_dir=str(self.data_dir)
        )
        
        print(f"✅ Dataset loaded with splits: {list(raw_dataset.keys())}")
        print(f"📊 Train size: {len(raw_dataset['train'])}")
        print(f"📊 Validation size: {len(raw_dataset['validation'])}")
        print(f"📊 Test size: {len(raw_dataset['test'])}")
        
        return raw_dataset
    
    def _tokenize_and_prepare(self, split, raw_dataset):
        """Tokenize and prepare data for a given split""" 

        cache_file = self.cache_dir / f"{split}_token_ids.pt"
        
        if cache_file.exists():
            print(f"✅ Loading cached token ids for {split}")
            all_ids = torch.load(cache_file)
        else:
            print(f"🔄 Tokenizing {split}...")
            
            all_ids = []
            max_length = self.tokenizer.model_max_length  # 512 for BERT
            
            for idx, line in enumerate(raw_dataset[split]['text']):
                if line and line.strip():
                    token_ids = self.tokenizer.encode(
                        line, 
                        add_special_tokens=False,
                        truncation=True, 
                        max_length=max_length 
                    )
                    all_ids.extend(token_ids)
                
                if (idx + 1) % 50000 == 0:
                    print(f"  Processed {idx + 1:,} lines...")
            
            if not all_ids:
                print(f"⚠️ Warning: No tokens generated for {split}")
                return None
                
            torch.save(all_ids, cache_file)
            print(f"✅ Cached token ids for {split} (total tokens: {len(all_ids):,})")
        
        return WikiTextDataset(all_ids, self.seq_len)
    
    def prepare_dataloaders(self):
        """Preparing dataloaders"""
        
        raw_dataset = self.download_and_prepare()
        
        print("\n📊 Tokenizing and preparing datasets...")
        self.train_dataset = self._tokenize_and_prepare("train", raw_dataset)
        self.val_dataset = self._tokenize_and_prepare("validation", raw_dataset)
        self.test_dataset = self._tokenize_and_prepare("test", raw_dataset)
        
        if self.train_dataset and len(self.train_dataset) > 0:
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True
            )
            print(f"✅ Train loader: {len(self.train_loader)} batches")
        
        if self.val_dataset and len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
            print(f"✅ Validation loader: {len(self.val_loader)} batches")
        
        if self.test_dataset and len(self.test_dataset) > 0:
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
        print(f"📊 Created dataset with {self.num_sequences:,} sequences from {len(token_ids):,} tokens")
        
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
            print(f"\n📦 Batch X shape: {batch_x.shape}")
            print(f"📦 Batch Y shape: {batch_y.shape}")
            print(f"📝 Sample input: {batch_x[0][:10]}...")
            print(f"📝 Sample target: {batch_y[0][:10]}...")
            break