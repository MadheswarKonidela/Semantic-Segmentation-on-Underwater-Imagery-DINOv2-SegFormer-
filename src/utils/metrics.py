import torch
import numpy as np

def calculate_iou(pred, target, num_classes):
    """
    Calculates Intersection over Union (IoU) for semantic segmentation.
    Args:
        pred (torch.Tensor): Predicted segmentation masks (logits or class indices).
                             Shape: (B, H, W) for class indices or (B, C, H, W) for logits.
        target (torch.Tensor): Ground truth segmentation masks (class indices).
                               Shape: (B, H, W)
        num_classes (int): Total number of classes.
    Returns:
        tuple: (list of IoU per class, mean IoU)
    """
    if pred.dim() == 4: # If logits are passed, convert to class indices
        pred = torch.argmax(pred, dim=1) # (B, H, W)

    pred = pred.view(-1)
    target = target.view(-1)

    iou_per_class = []
    for cls_id in range(num_classes):
        # Intersection: (pred == cls_id) AND (target == cls_id)
        intersection = ((pred == cls_id) & (target == cls_id)).sum().item()
        
        # Union: (pred == cls_id) OR (target == cls_id)
        union = ((pred == cls_id) | (target == cls_id)).sum().item()

        if union == 0:
            iou_per_class.append(np.nan)  # Avoid division by zero, means class not present in union
        else:
            iou_per_class.append(intersection / union)

    # Filter out NaNs for mean IoU (classes not present in batch)
    mean_iou = np.nanmean(iou_per_class) if iou_per_class else 0.0
    return iou_per_class, mean_iou

def calculate_pixel_accuracy(pred, target):
    """
    Calculates pixel accuracy for semantic segmentation.
    Args:
        pred (torch.Tensor): Predicted segmentation masks (logits or class indices).
                             Shape: (B, H, W) for class indices or (B, C, H, W) for logits.
        target (torch.Tensor): Ground truth segmentation masks (class indices).
                               Shape: (B, H, W)
    Returns:
        float: Pixel accuracy.
    """
    if pred.dim() == 4: # If logits are passed, convert to class indices
        pred = torch.argmax(pred, dim=1) # (B, H, W)

    correct_pixels = (pred == target).sum().item()
    total_pixels = target.numel()
    
    return correct_pixels / total_pixels if total_pixels > 0 else 0.0