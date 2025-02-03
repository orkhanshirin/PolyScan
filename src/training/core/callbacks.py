from pathlib import Path
import torch

class Callback:
    """Base callback class for training hooks"""
    def on_epoch_end(self, epoch: int, model: torch.nn.Module, metrics: dict[str, float]):
        pass

class CheckpointCallback(Callback):
    """Saves model checkpoints based on monitored metric"""
    def __init__(self, save_dir: Path, monitor: str = 'val_loss', mode: str = 'min'):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, metrics: dict[str, float]):
        current = metrics.get(self.monitor)
        if current is None:
            return

        # save latest
        torch.save(model.state_dict(), self.save_dir / "latest.pth")
        
        # save best
        if (self.mode == 'min' and current < self.best_metric) or \
           (self.mode == 'max' and current > self.best_metric):
            self.best_metric = current
            torch.save(model.state_dict(), self.save_dir / "best.pth")

class EarlyStopping(Callback):
    """Stops training when metric stops improving"""
    def __init__(self, patience: int = 5, monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, metrics: dict[str, float]):
        current = metrics.get(self.monitor)
        if current is None:
            return

        if (self.mode == 'min' and current < self.best_metric) or \
           (self.mode == 'max' and current > self.best_metric):
            self.best_metric = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True