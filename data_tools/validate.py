def validate_dataset(dataset):
    """
    Validates dataset for consistency.

    Args:
        dataset (PolypDataset): Dataset to validate.

    Raises:
        ValueError: If dataset is invalid.
    """
    for i in range(len(dataset)):
        img, mask = dataset[i]
        if img.shape[1:] != mask.shape:
            raise ValueError(
                f"Image and mask shapes mismatch at index {i}: {img.shape} vs {mask.shape}"
            )
        if mask.max() > 1:
            raise ValueError(
                f"Mask at index {i} is not binary: max value = {mask.max()}"
            )
