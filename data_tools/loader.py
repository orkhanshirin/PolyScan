import os

from torch.utils.data import DataLoader, Subset

from .dataset import PolypDataset
from .split import create_splits
from .validate import validate_dataset


def load_data(
    data_dir="data",
    metadata_file="metadata.csv",
    train_split=0.8,
    val_split=0.1,
    batch_size=16,
    transform=None,
):
    """
    Loads and splits the dataset into train/val/test DataLoaders.

    Args:
        data_dir (str): Path to dataset folder.
        metadata_file (str): CSV file with image and mask paths.
        train_split (float): Fraction of data for training.
        val_split (float): Fraction of data for validation.
        batch_size (int): Batch size for DataLoader.
        transform (callable): Transform to apply to image-mask pairs.

    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """
    metadata_path = os.path.join(data_dir, metadata_file)
    img_dir = os.path.join(data_dir, "PNG/Original")
    mask_dir = os.path.join(data_dir, "PNG/Ground Truth")

    # init dataset
    dataset = PolypDataset(metadata_path, img_dir, mask_dir, transform=transform)

    # validate dataset
    validate_dataset(dataset)

    # create splits
    idx_train, idx_val, idx_test = create_splits(len(dataset), train_split, val_split)

    # create DataLoaders
    train_loader = DataLoader(
        Subset(dataset, idx_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, idx_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, idx_test), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader
