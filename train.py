#!/usr/bin/env python3
"""
Unified Training Launcher for Satellite Wildfire Detection & Segmentation Models
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wildfire_detection.dataset import (
    WildfireSegmentationDataset,
    WildfireDetectionDataset,
    build_dataset_splits
)
from wildfire_detection.models.unet import UNet
from wildfire_detection.models.faster_rcnn import build_faster_rcnn
from wildfire_detection.utils.device import get_device
from wildfire_detection.utils.metrics import calculate_segmentation_metrics


def train_unet(args, device):
    print("--- Training U-Net Segmentation Model ---")
    train_imgs, val_imgs, train_masks, val_masks = build_dataset_splits(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        val_size=args.val_size
    )

    if args.dry_run or not train_imgs:
        print("[Dry Run / No Data Mode] Initializing U-Net synthetic forward pass check...")
        model = UNet(in_channels=3, out_channels=1).to(device)
        dummy_in = torch.randn(2, 3, 256, 256).to(device)
        dummy_out = model(dummy_in)
        print(f"Model Input Shape: {dummy_in.shape} | Output Shape: {dummy_out.shape}")
        print("Dry run completed successfully.")
        return

    train_dataset = WildfireSegmentationDataset(train_imgs, train_masks, img_size=(args.img_size, args.img_size))
    val_dataset = WildfireSegmentationDataset(val_imgs, val_masks, img_size=(args.img_size, args.img_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_iou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                metrics = calculate_segmentation_metrics(outputs, masks)
                for k in val_metrics:
                    val_metrics[k] += metrics[k]

        avg_val_loss = val_loss / max(1, len(val_loader))
        for k in val_metrics:
            val_metrics[k] /= max(1, len(val_loader))

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | Val Dice: {val_metrics['dice']:.4f}"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_path = os.path.join(args.output_dir, "best_unet_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model weights to {save_path}")


def train_fasterrcnn(args, device):
    print("--- Training Faster R-CNN Detection Model ---")
    train_imgs, val_imgs, train_masks, val_masks = build_dataset_splits(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        val_size=args.val_size
    )

    if args.dry_run or not train_imgs:
        print("[Dry Run / No Data Mode] Initializing Faster R-CNN synthetic check...")
        model = build_faster_rcnn(num_classes=2, pretrained=True).to(device)
        dummy_in = [torch.randn(3, 256, 256).to(device)]
        dummy_target = [{"boxes": torch.tensor([[10., 10., 50., 50.]]).to(device), "labels": torch.tensor([1]).to(device)}]
        model.train()
        loss_dict = model(dummy_in, dummy_target)
        print(f"Faster R-CNN constructed successfully. Synthetic loss dict: {loss_dict}")
        return

    from AirportSat.fasterRCNN import train_one_epoch, collate_fn

    train_dataset = WildfireDetectionDataset(train_imgs, train_masks, img_size=(args.img_size, args.img_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = build_faster_rcnn(num_classes=2, pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        save_path = os.path.join(args.output_dir, f"faster_rcnn_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch+1}/{args.epochs}] Avg Loss: {avg_loss:.4f} | Weights saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Trainer for Satellite Wildfire Detection")
    parser.add_argument("--model", type=str, choices=["unet", "fasterrcnn"], default="unet", help="Model architecture")
    parser.add_argument("--image-dir", type=str, default="data/train_img", help="Path to training images")
    parser.add_argument("--mask-dir", type=str, default="data/train_mask", help="Path to training masks")
    parser.add_argument("--output-dir", type=str, default="weights", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--img-size", type=int, default=256, help="Square image resolution")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Perform synthetic forward-pass test without dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)
    print(f"Active device: {device}")

    if args.model == "unet":
        train_unet(args, device)
    elif args.model == "fasterrcnn":
        train_fasterrcnn(args, device)


if __name__ == "__main__":
    main()
