import torch


def create_splits(dataset_size, train_split=0.8, val_split=0.1, seed=42):
    """
    Creates train/val/test splits.

    Args:
        dataset_size (int): Total number of samples.
        train_split (float): Fraction of data for training.
        val_split (float): Fraction of data for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple (idx_train, idx_val, idx_test)
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator)

    num_train = int(train_split * dataset_size)
    num_val = int(val_split * dataset_size)

    idx_train = indices[:num_train]
    idx_val = indices[num_train : num_train + num_val]
    idx_test = indices[num_train + num_val :]

    return idx_train, idx_val, idx_test
