import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2


def mask_to_bboxes(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts bounding boxes [xmin, ymin, xmax, ymax] and labels from a binary mask.
    
    Args:
        mask (np.ndarray): Binary mask image (H, W).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (boxes, labels)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    boxes = []
    class_labels = []
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 4:  # Filter out tiny noise dots
            boxes.append([x, y, x + w, y + h])
            class_labels.append(1)  # Fire class
            
    if not boxes:
        boxes = np.zeros((0, 4), dtype=np.float32)
        class_labels = np.zeros((0,), dtype=np.int64)
    else:
        boxes = np.array(boxes, dtype=np.float32)
        class_labels = np.array(class_labels, dtype=np.int64)
        
    return boxes, class_labels


class WildfireSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Wildfire Image Segmentation (U-Net style).
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        img_size: Tuple[int, int] = (256, 256),
        is_train: bool = True
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size, Image.BILINEAR)
        img_array = np.array(image, dtype=np.float32) / 255.0
        # Transpose to (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        result = {"image": img_tensor, "path": img_path}

        if self.mask_paths and idx < len(self.mask_paths):
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(self.img_size, Image.NEAREST)
            mask_array = np.array(mask, dtype=np.float32)
            mask_array = (mask_array > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # (1, H, W)
            result["mask"] = mask_tensor

        return result


class WildfireDetectionDataset(Dataset):
    """
    PyTorch Dataset for Object Detection (Faster R-CNN style).
    Converts binary masks into bounding boxes dynamically.
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        img_size: Tuple[int, int] = (256, 256)
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize(self.img_size, Image.BILINEAR)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        target = {
            "image_id": torch.tensor([idx]),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64)
        }

        if self.mask_paths and idx < len(self.mask_paths):
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(self.img_size, Image.NEAREST)
            mask_array = np.array(mask, dtype=np.uint8)
            boxes, labels = mask_to_bboxes(mask_array)
            target["boxes"] = torch.from_numpy(boxes)
            target["labels"] = torch.from_numpy(labels)

        return img_tensor, target


def build_dataset_splits(
    image_dir: str,
    mask_dir: Optional[str] = None,
    val_size: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str], Optional[List[str]], Optional[List[str]]]:
    """
    Scans directory for matching images and masks, returning train/val path splits.
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]) if os.path.exists(image_dir) else []

    mask_files = None
    if mask_dir and os.path.exists(mask_dir):
        mask_files = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])

    if not image_files:
        return [], [], None, None

    if mask_files and len(mask_files) == len(image_files):
        train_img, val_img, train_mask, val_mask = train_test_split(
            image_files, mask_files, test_size=val_size, random_state=seed
        )
        return train_img, val_img, train_mask, val_mask
    else:
        train_img, val_img = train_test_split(
            image_files, test_size=val_size, random_state=seed
        )
        return train_img, val_img, None, None
