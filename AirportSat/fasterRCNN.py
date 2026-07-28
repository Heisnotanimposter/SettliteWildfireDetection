#!/usr/bin/env python3
"""
Faster R-CNN Wildfire Bounding Box Detection Training Script
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import sys

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wildfire_detection.dataset import WildfireDetectionDataset, build_dataset_splits
from wildfire_detection.models.faster_rcnn import build_faster_rcnn
from wildfire_detection.utils.device import get_device


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    for step, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        if (step + 1) % 5 == 0 or step == len(data_loader) - 1:
            print(f"Epoch [{epoch + 1}] Step [{step + 1}/{len(data_loader)}] Loss: {losses.item():.4f}")

    return total_loss / max(1, len(data_loader))


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for Wildfire Detection")
    parser.add_argument("--image-dir", type=str, default="data/train_img", help="Path to training images")
    parser.add_argument("--mask-dir", type=str, default="data/train_mask", help="Path to training masks")
    parser.add_argument("--output-dir", type=str, default="weights", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cuda, mps, cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Build dataset splits
    train_imgs, val_imgs, train_masks, val_masks = build_dataset_splits(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        val_size=0.2
    )

    if not train_imgs:
        print(f"Warning: No images found at '{args.image_dir}'. Initializing model only.")
        model = build_faster_rcnn(num_classes=2, pretrained=True)
        print("Faster R-CNN model successfully constructed.")
        return

    train_dataset = WildfireDetectionDataset(train_imgs, train_masks)
    val_dataset = WildfireDetectionDataset(val_imgs, val_masks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = build_faster_rcnn(num_classes=2, pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        save_path = os.path.join(args.output_dir, f"faster_rcnn_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Completed | Avg Loss: {avg_loss:.4f} | Model saved to {save_path}")


if __name__ == "__main__":
    main()
