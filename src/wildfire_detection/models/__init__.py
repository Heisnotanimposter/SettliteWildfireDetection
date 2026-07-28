"""
Model Package for Satellite Wildfire Detection & Segmentation
"""

from wildfire_detection.models.unet import UNet
from wildfire_detection.models.faster_rcnn import build_faster_rcnn

__all__ = ["UNet", "build_faster_rcnn"]
