#!/usr/bin/env python
import argparse
from .core.trainer import Trainer
from .config.parser import load_config
from .tracking.mlflow import MLFlowTracker
from architectures import FCN, UNet
from data_tools.loader import load_data

def main(config_path: str):
    # load configuration
    config = load_config(config_path)
    
    # init model
    model_map = {
        'fcn': FCN,
        'unet': UNet
    }
    model_class = model_map[config['model'].lower()]
    model = model_class(**config.get('model_params', {}))
    
    # load data
    train_loader, val_loader, _ = load_data(
        batch_size=config['batch_size'],
        augmentations=config.get('augmentations', [])
    )
    
    # init experiment tracking
    with MLFlowTracker(config) as tracker:
        # init trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        # run training
        best_metric = trainer.run()
        tracker.log_metrics({'best_val_loss': best_metric}, step=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train medical image segmentation model")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)