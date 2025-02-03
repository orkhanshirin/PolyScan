import cv2
import numpy as np
import torch
from typing import Union

def overlay_mask(
    image: np.ndarray, 
    mask: np.ndarray, 
    color: tuple = (0, 0, 255),  # BGR format (default: red)
    alpha: float = 0.4,
    resize_mask: bool = True
) -> np.ndarray:
    """
    Overlays a segmentation mask on an image with specified color and transparency.
    
    Args:
        image: Input image (H, W, 3) in BGR or RGB format
        mask: Segmentation mask (H, W) with values 0-1 or 0-255
        color: BGR color tuple for overlay
        alpha: Transparency level (0.0-1.0)
        resize_mask: Resize mask to match image dimensions if needed
    
    Returns:
        Blended image with mask overlay in RGB format
    """
    # validate inputs
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be HWC BGR/RGB format")
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D array")
        
    # convert image to RGB if needed
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # normalize mask to 0-255
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    mask = mask.astype(np.uint8)
    
    # resize mask if needed
    if resize_mask and mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # create color overlay
    overlay = np.zeros_like(image)
    overlay[mask > 127] = color  # threshold at 50% probability
    
    # blend images
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

def tensor_to_numpy(
    tensor: Union[torch.Tensor, np.ndarray], 
    is_mask: bool = False
) -> np.ndarray:
    """
    Converts torch tensor to numpy array for visualization.
    
    Args:
        tensor: Input tensor (C, H, W) or (H, W)
        is_mask: Whether the tensor represents a mask
    
    Returns:
        Numpy array in HWC format
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = tensor.copy()
    
    if arr.ndim == 3:  # channel-first format
        arr = arr.transpose(1, 2, 0)
    
    if is_mask:
        return arr.squeeze().astype(np.uint8)
    return arr