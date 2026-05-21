import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ConfigManager:
    config: Dict[str, Any]
    config_path: Path
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml"):
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        cls._create_directories(config)
        
        return cls(config=config, config_path=path)
    
    @staticmethod
    def _create_directories(config: Dict):
        paths = config.get('paths', {})
        for key, path in paths.items():
            if isinstance(path, dict):
                for subpath in path.values():
                    Path(subpath).mkdir(parents=True, exist_ok=True)
            else:
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_data_path(self, subpath: str = "") -> Path:
        base = Path(self.get('paths.data.raw'))
        return base / subpath if subpath else base
    
    def get_results_path(self, subject_id: Optional[int] = None) -> Path:
        base = Path(self.get('paths.output.plots'))
        if subject_id:
            base = base / f"subject_{subject_id}"
        base.mkdir(parents=True, exist_ok=True)
        return base


config_manager = ConfigManager.from_yaml()