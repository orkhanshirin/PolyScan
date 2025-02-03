import yaml
from typing import Any
import torch

def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate training configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = torch.device(device)
    
    # set default values
    defaults = {
        'seed': 42,
        'grad_clip': None,
        'output_dir': 'runs',
        'early_stopping': {'patience': 5}
    }
    
    for key, value in defaults.items():
        config.setdefault(key, value)
    
    # validate required parameters
    required = ['model', 'epochs', 'lr', 'batch_size']
    for param in required:
        if param not in config:
            raise ValueError(f"Missing required config parameter: {param}")
    
    return config