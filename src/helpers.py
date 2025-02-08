import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml


def get_logger(name: str) -> logging.Logger:
    """Get configured logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def find_latest_model(experiment_name: str) -> str:
    """Finds latest model checkpoint for an experiment"""
    runs_dir = Path("runs")
    experiment_dirs = [
        d
        for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(experiment_name)
    ]

    if not experiment_dirs:
        raise FileNotFoundError(f"No runs found for experiment {experiment_name}")

    # sort by creation time (newest first)
    sorted_dirs = sorted(
        experiment_dirs, key=lambda d: os.path.getctime(d), reverse=True
    )

    # look for checkpoints in latest directory
    latest_dir = sorted_dirs[0]
    model_path = latest_dir / "checkpoints" / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {latest_dir}")

    return str(model_path)


def resolve_model_path(config: dict) -> str:
    """find model path based on experiment configuration"""
    if config["model"].get("path"):
        return config["model"]["path"]

    if config["model"].get("use_latest") and config["model"].get("experiment_name"):
        experiment_name = config["model"]["experiment_name"]
        runs_dir = Path("runs")
        experiment_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.name.startswith(experiment_name)],
            key=lambda d: d.stat().st_ctime,
            reverse=True,
        )

        if not experiment_dirs:
            raise FileNotFoundError(f"no runs found for experiment {experiment_name}")

        latest_run = experiment_dirs[0]
        model_path = latest_run / "checkpoints" / "best_model.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"model not found in {latest_run}")

        return str(model_path)

    raise ValueError("could not resolve model path from config")


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate evaluation configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # set device
    config["device"] = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    # resolve model path
    config["model_path"] = resolve_model_path(config)

    return config
