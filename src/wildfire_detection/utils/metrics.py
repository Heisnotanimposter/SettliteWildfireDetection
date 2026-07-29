import torch
import numpy as np
from typing import Dict, Union


def calculate_segmentation_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes binary segmentation evaluation metrics: IoU, Dice/F1, Precision, Recall, Accuracy.
    
    Args:
        pred: Predicted probabilities or binary mask (tensor or numpy array).
        target: Target ground truth mask (tensor or numpy array).
        threshold: Binarization threshold.
        eps: Small constant to avoid division by zero.
        
    Returns:
        Dict[str, float]: Dictionary containing metric values.
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
        
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    total_pred = pred_binary.sum()
    total_target = target_binary.sum()
    union = total_pred + total_target - intersection
    
    iou = (intersection + eps) / (union + eps) if union > 0 else torch.tensor(1.0 if total_target == 0 else 0.0)
    dice = (2.0 * intersection + eps) / (total_pred + total_target + eps)
    precision = (intersection) / (total_pred + eps) if total_pred > 0 else torch.tensor(0.0)
    recall = (intersection) / (total_target + eps) if total_target > 0 else torch.tensor(0.0)
    accuracy = (pred_binary == target_binary).float().mean()
    
    return {
        "iou": float(iou.item()),
        "dice": float(dice.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "accuracy": float(accuracy.item())
    }
