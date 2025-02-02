import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def dice_coeff(target, pred, n_classes=2):
    """Computes Dice coefficient per class."""
    smooth = 1e-6
    dice = torch.zeros(n_classes, device=target.device)
    for c in range(n_classes):
        intersection = ((pred == c) & (target == c)).float().sum()
        union = (pred == c).float().sum() + (target == c).float().sum()
        dice[c] = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def overlay_segmentation(image, mask, alpha=0.4):
    """Overlays segmentation mask on the original image."""
    image = np.array(image, dtype=np.uint8)
    mask = np.array(mask, dtype=np.uint8) * 255
    overlay = np.zeros_like(image)
    overlay[mask == 255] = [255, 0, 0]  # Red color for segmentation
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

def visualize_predictions(imgs_testset, prediction, idx_test, data_dir="data", n_samples=8, output_dir="outputs", save_outputs=False):
    """Visualizes and optionally saves segmentation predictions."""
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 2.5))
    all_filenames = sorted(os.listdir(os.path.join(data_dir, "PNG/Original")))
    test_filenames = [all_filenames[i] for i in idx_test]

    for i in range(n_samples):
        img_filename = test_filenames[i]
        img_path = os.path.join(data_dir, "PNG/Original", img_filename)
        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        overlayed_image = overlay_segmentation(img_original, prediction[i].cpu().numpy())
        axes[i, 0].imshow(img_original)
        axes[i, 1].imshow(overlayed_image)
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
