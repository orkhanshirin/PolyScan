import time
from pathlib import Path
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .callbacks import CheckpointCallback, EarlyStopping
from src.helpers import get_logger 

logger = get_logger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """Main training orchestrator for medical image segmentation models."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        callbacks: Optional[list] = None,
    ):
        self.model = model.to(config["device"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.epoch = 0
        self.best_metric = float("inf")

        # init components
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterion = torch.nn.CrossEntropyLoss()

        # callbacks
        self.callbacks = callbacks or []
        self._register_default_callbacks()

        # reproducibility
        if config.get("seed"):
            set_seed(config["seed"])

    def _init_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get("weight_decay", 0),
        )

    def _init_scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.get("lr_factor", 0.1),
            patience=self.config.get("lr_patience", 5),
        )

    def _register_default_callbacks(self):
        """Register essential training callbacks"""
        self.callbacks.append(
            CheckpointCallback(
                save_dir=Path(self.config["output_dir"]), monitor="val_loss", mode="min"
            )
        )

        if self.config.get("early_stopping"):
            self.callbacks.append(
                EarlyStopping(
                    patience=self.config["early_stopping"]["patience"],
                    monitor="val_loss",
                    mode="min",
                )
            )

    def train_epoch(self) -> float:
        """Execute one training epoch"""
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.config["device"])
            masks = masks.to(self.config["device"])

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()

            if self.config.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )

            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def validate(self) -> float:
        """Run validation on the current model"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.config["device"])
                masks = masks.to(self.config["device"])

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def run(self):
        """Execute full training run"""
        logger.info(f"Starting training for {self.config['model']}")
        start_time = time.time()

        for self.epoch in range(1, self.config["epochs"] + 1):
            epoch_start = time.time()

            # training
            train_loss = self.train_epoch()

            # validation
            val_loss = self.validate()
            self.scheduler.step(val_loss)

            # metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": time.time() - epoch_start,
            }

            # handle callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(
                    epoch=self.epoch, model=self.model, metrics=metrics
                )

            # early stopping check
            if any(callback.stop_training for callback in self.callbacks):
                logger.info("Early stopping triggered")
                break

        # finalize training
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        return self.best_metric
