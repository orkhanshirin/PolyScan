from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from .overlay import tensor_to_numpy


def plot_predictions(
    images: list[Union[np.ndarray, torch.Tensor]],
    masks: list[Union[np.ndarray, torch.Tensor]],
    overlays: list[np.ndarray],
    save_path: Optional[Path] = None,
    figsize: tuple = (18, 12),
    alpha: float = 0.4,
    n_samples: int = 8,
):
    """
    Plots medical image segmentation results with standardized overlays

    Args:
        images: List of original images in numpy array or torch tensor format
        masks: List of ground truth masks
        overlays: List of overlay images from overlay_mask()
        save_path: Optional path to save visualization
        figsize: Figure dimensions
        alpha: Transparency level for overlays
        n_samples: Number of samples to display
    """
    # convert all inputs to numpy arrays
    images = [tensor_to_numpy(img) for img in images]
    masks = [tensor_to_numpy(mask, is_mask=True) for mask in masks]

    # validate input lengths
    n_samples = min(n_samples, len(images))
    images = images[:n_samples]
    masks = masks[:n_samples]
    overlays = overlays[:n_samples]

    # create figure and axes
    fig, axes = plt.subplots(n_samples, 3, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # configure plot elements
    titles = ["Original", "Ground Truth", "Prediction Overlay"]
    cmap = plt.cm.viridis  # medical-friendly colormap

    for idx in range(n_samples):
        # original image column
        axes[idx, 0].imshow(images[idx])

        # ground truth mask column
        axes[idx, 1].imshow(masks[idx], cmap=cmap)

        # prediction overlay column
        axes[idx, 2].imshow(overlays[idx])

        # set titles for first row only
        if idx == 0:
            for col, title in enumerate(titles):
                axes[idx, col].set_title(title)

        # remove axis decorations
        for col in range(3):
            axes[idx, col].axis("off")

    # add color scale reference
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap), ax=axes, orientation="vertical", fraction=0.02
    )

    # handle output
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    metrics: dict[str, list[float]],
    save_path: Optional[Path] = None,
):
    """
    Plots training history for medical image segmentation models

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        metrics: Dictionary of metric values per epoch
        save_path: Optional path to save visualization
    """
    plt.figure(figsize=(18, 6))

    # loss curves subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training History - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # metrics subplot
    plt.subplot(1, 2, 2)
    for name, values in metrics.items():
        plt.plot(values, label=name.replace("_", " ").title())
    plt.title("Training History - Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()

    # handle output
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
