import torch
import numpy as np
from typing import Dict, Union, List


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


def calculate_classification_metrics(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-7
) -> Dict[str, Union[float, List[List[int]]]]:
    """
    Computes binary/multiclass classification evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
    
    Args:
        preds: Predicted class indices (N,) or logits/probabilities (N, C).
        targets: Target class indices (N,).
        
    Returns:
        Dict: Dictionary containing accuracy, precision, recall, f1_score, confusion_matrix.
    """
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    if preds.ndim > 1:
        preds = torch.argmax(preds, dim=1)

    preds = preds.long()
    targets = targets.long()

    correct = (preds == targets).float().sum()
    accuracy = float((correct / max(1, len(targets))).item())

    # Binary classification specifics (assuming positive class is 1)
    tp = ((preds == 1) & (targets == 1)).float().sum()
    fp = ((preds == 1) & (targets == 0)).float().sum()
    fn = ((preds == 0) & (targets == 1)).float().sum()
    tn = ((preds == 0) & (targets == 0)).float().sum()

    precision = float((tp / (tp + fp + eps)).item())
    recall = float((tp / (tp + fn + eps)).item())
    f1_score = float((2 * precision * recall / (precision + recall + eps)))

    cm = [[int(tn.item()), int(fp.item())], [int(fn.item()), int(tp.item())]]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": cm
    }
