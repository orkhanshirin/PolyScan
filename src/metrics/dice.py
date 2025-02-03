import torch

def dice_score(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    n_classes: int = 2,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Computes Dice Similarity Coefficient for medical image segmentation.
    
    Args:
        y_pred: Predicted masks [N, H, W] (class indices)
        y_true: Ground truth masks [N, H, W]
        n_classes: Number of classes
        epsilon: Smoothing factor to avoid division by zero
    
    Returns:
        Tensor: Dice scores per class [n_classes]
    
    Example:
        >>> preds = torch.randint(0, 2, (4, 256, 256))
        >>> targets = torch.randint(0, 2, (4, 256, 256))
        >>> dice = dice_score(preds, targets)
    """
    scores = torch.zeros(n_classes, device=y_true.device)
    
    for class_idx in range(n_classes):
        pred_inds = (y_pred == class_idx)
        true_inds = (y_true == class_idx)
        
        intersection = (pred_inds & true_inds).sum().float()
        union = pred_inds.sum() + true_inds.sum()
        
        scores[class_idx] = (2. * intersection + epsilon) / (union + epsilon)
        
    return scores