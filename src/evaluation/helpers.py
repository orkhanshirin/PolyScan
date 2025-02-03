from pathlib import Path
import os
import yaml
import torch
from  typing import Any

def find_latest_model(experiment_name: str) -> str:
    """Finds latest model checkpoint for an experiment"""
    runs_dir = Path("runs")
    experiment_dirs = [
        d for d in runs_dir.iterdir() 
        if d.is_dir() and d.name.startswith(experiment_name)
    ]
    
    if not experiment_dirs:
        raise FileNotFoundError(f"No runs found for experiment {experiment_name}")
    
    # sort by creation time (newest first)
    sorted_dirs = sorted(
        experiment_dirs,
        key=lambda d: os.path.getctime(d),
        reverse=True
    )
    
    # look for checkpoints in latest directory
    latest_dir = sorted_dirs[0]
    model_path = latest_dir / "checkpoints" / "best_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {latest_dir}")
    
    return str(model_path)

def resolve_model_path(config: dict) -> str:
    """Resolves model path from config options"""
    if config.get('model_path'):
        return config['model_path']
    
    if config.get('use_latest') and config.get('experiment_name'):
        return find_latest_model(config['experiment_name'])
    
    raise ValueError("Could not resolve model path from config")

def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate evaluation configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # set device
    config['device'] = torch.device(
        config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # resolve model path
    config['model_path'] = resolve_model_path(config)
    
    return config