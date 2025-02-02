import os
import pandas as pd
import torch
import cv2
import numpy as np

def load_dataset(data_dir="data", metadata_file="metadata.csv", train_split=0.8, val_split=0.1):
    """
    Loads and preprocesses dataset.

    Args:
        data_dir (str): Path to dataset folder.
        metadata_file (str): CSV file with image and mask paths.
        train_split (float): Fraction of data for training.
        val_split (float): Fraction of data for validation.

    Returns:
        Tuple (data_img, data_seg, idx_train, idx_val, idx_test)
    """
    csv_path = os.path.join(data_dir, metadata_file)
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} rows in {metadata_file}")

    all_imgs = []
    all_segs = []

    for i, row in df.iterrows():
        img_path = os.path.join(data_dir, row["png_image_path"])
        mask_path = os.path.join(data_dir, row["png_mask_path"])

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: couldn't read {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: couldn't read {mask_path}")
            continue
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
        mask_bin = (mask_gray > 127).astype(np.uint8)

        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))) if len(img.shape) == 3 else torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask_bin).long().unsqueeze(0)

        all_imgs.append(img_t)
        all_segs.append(mask_t)

    imgs = torch.stack(all_imgs)
    segs = torch.stack(all_segs)
    print(f"Loaded {imgs.shape[0]} image-mask pairs.")

    # Split dataset
    data_size = len(imgs)
    idx = torch.randperm(data_size)
    num_train = int(train_split * data_size)
    num_val = int(val_split * data_size)
    idx_train = idx[:num_train]
    idx_val = idx[num_train:num_train + num_val]
    idx_test = idx[num_train + num_val:]

    return imgs, segs, idx_train, idx_val, idx_test
