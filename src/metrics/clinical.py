import torch

def sensitivity_specificity(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    n_classes: int = 2,
    epsilon: float = 1e-6
) -> tuple:
    """
    Computes sensitivity and specificity for medical image analysis.
    
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    
    Args:
        y_pred: Predicted masks [N, H, W] (class indices)
        y_true: Ground truth masks [N, H, W]
        n_classes: Number of classes
        epsilon: Smoothing factor
    
    Returns:
        (sensitivity, specificity) tensors [n_classes]
    """
    sens = torch.zeros(n_classes, device=y_true.device)
    spec = torch.zeros(n_classes, device=y_true.device)
    
    for class_idx in range(n_classes):
        # true positives, false negatives
        tp = ((y_pred == class_idx) & (y_true == class_idx)).sum().float()
        fn = ((y_pred != class_idx) & (y_true == class_idx)).sum().float()
        sens[class_idx] = tp / (tp + fn + epsilon)
        
        # true negatives, false positives
        tn = ((y_pred != class_idx) & (y_true != class_idx)).sum().float()
        fp = ((y_pred == class_idx) & (y_true != class_idx)).sum().float()
        spec[class_idx] = tn / (tn + fp + epsilon)
    
    return sens, spec