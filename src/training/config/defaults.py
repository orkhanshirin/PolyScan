from pathlib import Path
from typing import Any

def get_default_config() -> dict[str, Any]:
    """Return default configuration values for PolyScan training"""
    return {
        # experiment setup
        'seed': 42,
        'device': 'cuda',  # auto-resolved later
        'output_dir': str(Path('runs') / 'experiments'),
        'experiment_name': 'unnamed',
        
        # model architecture defaults
        'model_params': {
            'fcn': {
                'in_channels': 3,
                'init_filters': 32,
                'num_classes': 2
            },
            'unet': {
                'in_channels': 3,
                'init_filters': 64,
                'num_classes': 2,
                'deep_supervision': False
            }
        },
        
        # training hyperparameters
        'epochs': 100,
        'batch_size': 16,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'lr_scheduler': {
            'name': 'reduce_on_plateau',
            'mode': 'min',
            'factor': 0.1,
            'patience': 5
        },
        
        # data configuration
        'data': {
            'root_dir': 'data',
            'metadata_file': 'metadata.csv',
            'split_ratios': [0.7, 0.15, 0.15],
            'augmentations': [
                {
                    'name': 'horizontal_flip',
                    'probability': 0.5
                },
                {
                    'name': 'vertical_flip',
                    'probability': 0.5
                }
            ]
        },
        
        # callbacks
        'checkpoint': {
            'monitor': 'val_loss',
            'mode': 'min',
            'save_top_k': 3
        },
        'early_stopping': {
            'enabled': True,
            'monitor': 'val_loss',
            'mode': 'min',
            'patience': 10,
            'min_delta': 0.001
        },
        
        # medical imaging specific
        'medical': {
            'normalization': 'zscore',  # zscore or minmax
            'resample_spacing': [1.0, 1.0, 1.0],  # For volumetric data
            'cache_rate': 0.2  # For large datasets
        }
    }