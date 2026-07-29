#!/usr/bin/env python3
"""
YOLOv9 Dataset Preparation and Training Pipeline for Satellite Wildfire Detection
"""

import os
import sys
import argparse
import shutil
import yaml
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wildfire_detection.dataset import mask_to_bboxes


def convert_mask_to_yolo_labels(mask_path: str, output_txt_path: str):
    """
    Converts a binary mask image to YOLO object detection text format (class x_center y_center width height).
    """
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img, dtype=np.uint8)
    h, w = mask.shape
    boxes, labels = mask_to_bboxes(mask)

    lines = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))


def prepare_yolo_dataset(image_dir: str, mask_dir: str, output_dir: str, val_size: float = 0.2, seed: int = 42):
    """
    Organizes images and labels into standard YOLO dataset layout:
    output_dir/
      train/images/
      train/labels/
      val/images/
      val/labels/
      data.yaml
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]) if os.path.exists(image_dir) else []

    if not image_files:
        print(f"No image files found in {image_dir}")
        return None

    train_files, val_files = train_test_split(image_files, test_size=val_size, random_state=seed)

    for split, files in [("train", train_files), ("val", val_files)]:
        img_out = os.path.join(output_dir, split, "images")
        lbl_out = os.path.join(output_dir, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fname in files:
            src_img = os.path.join(image_dir, fname)
            dst_img = os.path.join(img_out, fname)
            shutil.copy2(src_img, dst_img)

            if mask_dir and os.path.exists(mask_dir):
                base_name = os.path.splitext(fname)[0]
                mask_path = os.path.join(mask_dir, fname)
                if not os.path.exists(mask_path):
                    for ext in [".png", ".jpg", ".tif"]:
                        alt = os.path.join(mask_dir, base_name + ext)
                        if os.path.exists(alt):
                            mask_path = alt
                            break

                txt_out = os.path.join(lbl_out, base_name + ".txt")
                if os.path.exists(mask_path):
                    convert_mask_to_yolo_labels(mask_path, txt_out)

    yaml_config = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["Fire"]
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f)

    print(f"YOLO dataset prepared at: {output_dir}")
    print(f"Dataset config written to: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLOv9 dataset config and training pipeline")
    parser.add_argument("--image-dir", type=str, default="data/train_img", help="Path to raw training images")
    parser.add_argument("--mask-dir", type=str, default="data/train_mask", help="Path to raw training masks")
    parser.add_argument("--output-dir", type=str, default="data/yolo_dataset", help="Output directory for YOLO dataset")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    if os.path.exists(args.image_dir) and os.path.exists(args.mask_dir):
        prepare_yolo_dataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            val_size=args.val_size
        )
    else:
        print(f"Image directory '{args.image_dir}' or mask directory '{args.mask_dir}' does not exist.")
        print("Run this script with valid --image-dir and --mask-dir paths to generate YOLO labels.")


if __name__ == "__main__":
    main()
