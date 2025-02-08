import torch
from monai.metrics import compute_hausdorff_distance


def hausdorff_distance_95(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    n_classes: int = 2,
    percentile: float = 95.0,
) -> torch.Tensor:
    """
    Computes 95th percentile Hausdorff Distance for medical image segmentation.

    Requires MONAI library: pip install monai

    Args:
        y_pred: Predicted masks [N, C, H, W] (probabilities)
        y_true: Ground truth masks [N, C, H, W]
        n_classes: Number of classes
        percentile: Percentile to compute (default: 95)

    Returns:
        Tensor: HD95 scores per class [n_classes]
    """
    scores = torch.zeros(n_classes, device=y_true.device)

    for class_idx in range(n_classes):
        # convert to binary masks for current class
        pred_class = y_pred[:, class_idx].unsqueeze(1)  # [N, 1, H, W]
        true_class = y_true[:, class_idx].unsqueeze(1)

        # compute HD95 using MONAI
        hd = compute_hausdorff_distance(
            pred_class, true_class, percentile=percentile, include_background=False
        )

        scores[class_idx] = hd.mean()

    return scores
