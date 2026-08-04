import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)


def build_classifier(
    model_name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True
) -> nn.Module:
    """
    Builds a transfer learning image classifier (Wildfire vs No Wildfire).
    
    Args:
        model_name (str): Backbone architecture name ('resnet18', 'resnet50', 'efficientnet_b0').
        num_classes (int): Number of output classes (default 2).
        pretrained (bool): Whether to use ImageNet pre-trained weights.
        
    Returns:
        nn.Module: PyTorch classification model.
    """
    model_name_lower = model_name.lower()

    if "resnet18" in model_name_lower:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif "resnet50" in model_name_lower:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif "efficientnet" in model_name_lower:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Choose from 'resnet18', 'resnet50', 'efficientnet_b0'."
        )

    return model
