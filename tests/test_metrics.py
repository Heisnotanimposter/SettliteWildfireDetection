import torch
import pytest
from wildfire_detection.utils.metrics import calculate_segmentation_metrics


def test_segmentation_metrics_perfect_match():
    pred = torch.ones((1, 1, 10, 10))
    target = torch.ones((1, 1, 10, 10))
    
    metrics = calculate_segmentation_metrics(pred, target)
    assert pytest.approx(metrics["iou"], abs=1e-3) == 1.0
    assert pytest.approx(metrics["dice"], abs=1e-3) == 1.0
    assert pytest.approx(metrics["precision"], abs=1e-3) == 1.0
    assert pytest.approx(metrics["recall"], abs=1e-3) == 1.0
    assert pytest.approx(metrics["accuracy"], abs=1e-3) == 1.0


def test_segmentation_metrics_no_overlap():
    pred = torch.zeros((1, 1, 10, 10))
    target = torch.ones((1, 1, 10, 10))
    
    metrics = calculate_segmentation_metrics(pred, target)
    assert pytest.approx(metrics["iou"], abs=1e-3) == 0.0
    assert pytest.approx(metrics["dice"], abs=1e-3) == 0.0
    assert pytest.approx(metrics["recall"], abs=1e-3) == 0.0
