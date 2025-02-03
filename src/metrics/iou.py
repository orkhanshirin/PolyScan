import torch

def iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    n_classes: int = 2,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Computes Intersection over Union (Jaccard Index) for medical segmentation.
    
    Args:
        y_pred: Predicted masks [N, H, W] (class indices)
        y_true: Ground truth masks [N, H, W]
        n_classes: Number of classes
        epsilon: Smoothing factor
    
    Returns:
        Tensor: IoU scores per class [n_classes]
    """
    scores = torch.zeros(n_classes, device=y_true.device)
    
    for class_idx in range(n_classes):
        pred_inds = (y_pred == class_idx)
        true_inds = (y_true == class_idx)
        
        intersection = (pred_inds & true_inds).sum().float()
        union = (pred_inds | true_inds).sum().float()
        
        scores[class_idx] = (intersection + epsilon) / (union + epsilon)
        
    return scores