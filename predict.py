#!/usr/bin/env python3
"""
Inference & Submission Generator for Satellite Wildfire Detection, Segmentation, and Classification
"""

import os
import sys
import argparse
import joblib
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from wildfire_detection.models.unet import UNet
from wildfire_detection.models.classifier import build_classifier
from wildfire_detection.utils.device import get_device


def predict_segmentation(
    weights_path: str,
    image_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    img_size: int = 256,
    device_str: str = "auto"
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(device_str)
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights path '{weights_path}' not found. Using default initialized weights.")

    model.eval()

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]) if os.path.exists(image_dir) else []

    if not image_files:
        print(f"No target test images found in '{image_dir}'.")
        return

    print(f"Processing {len(image_files)} test images for segmentation...")
    predictions = {}

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Segmentation Inference"):
            filename = os.path.basename(img_path)
            orig_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = orig_img.size

            resized = orig_img.resize((img_size, img_size), Image.BILINEAR)
            img_arr = np.array(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

            output = model(tensor)
            prob_mask = output.squeeze().cpu().numpy()
            binary_mask = (prob_mask > threshold).astype(np.uint8)

            mask_img = Image.fromarray(binary_mask * 255).resize((orig_w, orig_h), Image.NEAREST)
            final_mask = (np.array(mask_img) > 127).astype(np.uint8)

            out_img_path = os.path.join(output_dir, f"pred_{os.path.splitext(filename)[0]}.png")
            Image.fromarray(final_mask * 255).save(out_img_path)

            predictions[filename] = final_mask

    pkl_path = os.path.join(output_dir, "submission_predictions.pkl")
    joblib.dump(predictions, pkl_path)
    print(f"Segmentation completed. Predictions serialized to: {pkl_path}")


def predict_classification(
    weights_path: str,
    image_dir: str,
    output_dir: str,
    backbone: str = "resnet18",
    img_size: int = 256,
    device_str: str = "auto"
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(device_str)
    print(f"Using device: {device}")

    model = build_classifier(model_name=backbone, num_classes=2, pretrained=False).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights path '{weights_path}' not found. Using default initialized weights.")

    model.eval()

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = []
    if os.path.exists(image_dir):
        for root, _, files in os.walk(image_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    image_files.append(os.path.join(root, f))
    image_files.sort()

    if not image_files:
        print(f"No target classification images found in '{image_dir}'.")
        return

    print(f"Processing {len(image_files)} images for classification...")
    results = {}
    class_names = ["nowildfire", "wildfire"]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Classification Inference"):
            rel_path = os.path.relpath(img_path, image_dir)
            orig_img = Image.open(img_path).convert("RGB")
            resized = orig_img.resize((img_size, img_size), Image.BILINEAR)

            img_arr = np.array(resized, dtype=np.float32) / 255.0
            img_arr = (img_arr - mean) / std
            tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_class_idx = int(np.argmax(probs))

            results[rel_path] = {
                "predicted_class": class_names[pred_class_idx],
                "confidence": float(probs[pred_class_idx]),
                "probabilities": {
                    "nowildfire": float(probs[0]),
                    "wildfire": float(probs[1])
                }
            }

    out_file = os.path.join(output_dir, "classification_results.json")
    import json
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Classification completed. Results written to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Perform Wildfire Prediction & Inference")
    parser.add_argument("--task", type=str, choices=["segmentation", "classification"], default="segmentation", help="Inference task mode")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Classifier backbone architecture")
    parser.add_argument("--weights", type=str, default="weights/best_unet_model.pth", help="Path to trained model weights")
    parser.add_argument("--image-dir", type=str, default="data/test_img", help="Path to test images folder")
    parser.add_argument("--output-dir", type=str, default="predictions", help="Path to save predictions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for segmentation")
    parser.add_argument("--img-size", type=int, default=256, help="Input resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, mps, cpu)")
    args = parser.parse_args()

    if args.task == "segmentation":
        predict_segmentation(
            weights_path=args.weights,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            img_size=args.img_size,
            device_str=args.device
        )
    elif args.task == "classification":
        predict_classification(
            weights_path=args.weights,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            backbone=args.backbone,
            img_size=args.img_size,
            device_str=args.device
        )


if __name__ == "__main__":
    main()
