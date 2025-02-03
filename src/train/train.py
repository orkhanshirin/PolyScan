import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml  # type: ignore
from architectures import FCN, UNet
from data_tools.loader import load_data


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        self.best_metric = float("inf")
        self.experiment_name = (
            f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.checkpoint_dir = Path("runs") / self.experiment_name / "checkpoints"
        self._create_checkpoint_dir()

    def _create_checkpoint_dir(self):
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        except PermissionError:
            raise RuntimeError(
                f"Permission denied: Could not create directory {self.checkpoint_dir}"
            )
        except Exception as e:
            raise RuntimeError(f"Error creating checkpoint directory: {str(e)}")

    def _create_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get("weight_decay", 0),
        )

    def _create_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for images, masks in self.train_loader:
            images = images.to(device)
            masks = masks.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, metric):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "metric": metric,
        }

        # always keep latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")

        # keep top-k checkpoints
        if metric < self.best_metric:
            self.best_metric = metric
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")

        # save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch:04d}.pth")

    def run(self):
        print(
            f"Training {self.config['model_name']} on {len(self.train_loader.dataset)} samples"
        )
        for epoch in range(self.config["epochs"]):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)

            # save best model
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.save_checkpoint(epoch, val_loss)
                print(
                    f"New best model at epoch {epoch + 1} with val loss: {val_loss:.4f}"
                )

            # report progress
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1:03d}/{self.config['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.2f}s"
            )


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main(config_path):
    config = load_config(config_path)

    # init model
    if config["model_name"].lower() == "fcn":
        model = FCN(**config["model_params"])
    elif config["model_name"].lower() == "unet":
        model = UNet(**config["model_params"])
    else:
        raise ValueError(f"Unsupported model: {config['model_name']}")

    # load data
    train_loader, val_loader, _ = load_data(
        batch_size=config["batch_size"], transform=config.get("augmentations")
    )

    # init trainer
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader, config=config
    )

    # start training
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train medical image segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main(args.config)
