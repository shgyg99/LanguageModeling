import os
import sys
import shutil
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_manager import config_manager


class DataIngestion:
    def __init__(self):
        self.config = config_manager
        
        data_config = self.config.get('data', {})
        self.dataset_config =  data_config.get('dataset_config')
        self.dataset_name = data_config.get('dataset_name')
        
        self.target_dir = Path(self.config.get('paths', {}).get('data_raw', "./data/raw"))
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        self.arrow_files = ["wikitext-train.arrow", "wikitext-validation.arrow", "wikitext-test.arrow"]
    
    def _move_files_from_cache(self):
        cache_base = Path(self.target_dir) / self.dataset_name
        if self.dataset_config:
            cache_base = cache_base / self.dataset_config
        
        if not cache_base.exists():
            return False
        
        moved = False
        for arrow_file in self.arrow_files:
            src = list(cache_base.rglob(arrow_file))
            if src:
                shutil.move(str(src[0]), str(self.target_dir / arrow_file))
                moved = True
                print(f"✅ Moved: {arrow_file}")
        
        if moved:
            shutil.rmtree(Path(self.target_dir) / self.dataset_name)
            print("🗑️ Removed cache")
        
        return moved
    
    def _load_local(self):
        dataset = DatasetDict()
        
        for split, filename in [("train", "wikitext-train.arrow"), 
                                  ("validation", "wikitext-validation.arrow"), 
                                  ("test", "wikitext-test.arrow")]:
            file_path = self.target_dir / filename
            if file_path.exists():
                dataset[split] = Dataset.from_file(str(file_path))
                print(f"✅ Loaded {split}")
        
        return dataset if dataset else None
    
    def download_dataset(self):
        
        if all((self.target_dir / f).exists() for f in self.arrow_files):
            print("✅ Using existing dataset")
            return self._load_local()
        
        if self._move_files_from_cache():
            print("✅ Found in cache, moved to target")
            return self._load_local()
        
        print(f"📥 Downloading {self.dataset_name}/{self.dataset_config}...")
        dataset = load_dataset(self.dataset_name, self.dataset_config, 
                               cache_dir=str(self.target_dir), trust_remote_code=True)
        
        self._move_files_from_cache()
        return dataset


if __name__ == "__main__":
    dt = DataIngestion()
    dataset = dt.download_dataset()
    
    if dataset:
        print(f"\n🎉 Success! Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}, Test: {len(dataset['test'])}")