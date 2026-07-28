import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Builds a Faster R-CNN PyTorch model with ResNet50-FPN backbone for Wildfire Bounding Box Detection.
    
    Args:
        num_classes (int): Number of classes including background. Default is 2 (Background + Fire).
        pretrained (bool): Whether to use ImageNet/COCO pre-trained weights.
        
    Returns:
        nn.Module: PyTorch Faster R-CNN model.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # Replace the box predictor head for fine-tuning
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
