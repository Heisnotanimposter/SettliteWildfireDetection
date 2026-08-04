import torch
import pytest
import numpy as np
from wildfire_detection.models.classifier import build_classifier
from wildfire_detection.utils.metrics import calculate_classification_metrics
from wildfire_detection.dataset import WildfireClassificationDataset


def test_classifier_forward_pass():
    for model_name in ["resnet18", "resnet50", "efficientnet_b0"]:
        model = build_classifier(model_name=model_name, num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 2)


def test_classification_metrics_perfect():
    preds = torch.tensor([1, 0, 1, 0])
    targets = torch.tensor([1, 0, 1, 0])
    metrics = calculate_classification_metrics(preds, targets)
    assert pytest.approx(metrics["accuracy"], abs=1e-3) == 1.0
    assert pytest.approx(metrics["precision"], abs=1e-3) == 1.0
    assert pytest.approx(metrics["recall"], abs=1e-3) == 1.0
    assert pytest.approx(metrics["f1_score"], abs=1e-3) == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_classification_metrics_mixed():
    preds = torch.tensor([1, 1, 0, 0])
    targets = torch.tensor([1, 0, 1, 0])
    metrics = calculate_classification_metrics(preds, targets)
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_classification_dataset_empty_init():
    dataset = WildfireClassificationDataset(image_paths=[], labels=[])
    assert len(dataset) == 0
