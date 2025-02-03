import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PolypDataset(Dataset):
    def __init__(self, metadata_path, img_dir, mask_dir, transform=None):
        """
        Args:
            metadata_path (str): Path to metadata CSV file.
            img_dir (str): Directory containing images.
            mask_dir (str): Directory containing masks.
            transform (callable): Optional transform to apply to image-mask pairs.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.img_dir, row["png_image_path"])
        mask_path = os.path.join(self.mask_dir, row["png_mask_path"])

        # load image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        img = img.astype(np.float32) / 255.0

        # load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
        mask = (mask > 127).astype(np.uint8)  # binarize mask

        # apply transforms (if any)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        # convert to PyTorch tensors
        img_t = (
            torch.from_numpy(np.transpose(img, (2, 0, 1)))
            if len(img.shape) == 3
            else torch.from_numpy(img).unsqueeze(0)
        )
        mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t
