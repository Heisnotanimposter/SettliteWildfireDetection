import numpy as np
import torch
import pytest
from wildfire_detection.dataset import mask_to_bboxes, WildfireSegmentationDataset


def test_mask_to_bboxes_empty():
    mask = np.zeros((100, 100), dtype=np.uint8)
    boxes, labels = mask_to_bboxes(mask)
    assert len(boxes) == 0
    assert len(labels) == 0


def test_mask_to_bboxes_single_object():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:60] = 255
    boxes, labels = mask_to_bboxes(mask)
    assert len(boxes) == 1
    assert len(labels) == 1
    assert list(boxes[0]) == [30, 20, 60, 50]
    assert labels[0] == 1
