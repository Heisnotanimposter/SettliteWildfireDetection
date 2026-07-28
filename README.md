# Satellite Wildfire Detection & Segmentation Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Modular machine learning and computer vision framework for **Satellite Wildfire Detection & Image Segmentation** developed for the 2024 AISPARK Competition.

---

## 🌟 Key Features

- **Architectures Supported**:
  - **U-Net**: High-resolution binary semantic segmentation for pinpointing wildfire burned areas.
  - **Faster R-CNN**: Deep object detection pipeline for wildfire region localized bounding boxes.
  - **YOLOv9**: Optimized single-stage object detection dataset formatting & training preparation pipeline.
- **Hardware Acceleration**: Automatic device selection supporting NVIDIA CUDA, Apple Silicon (`mps`), and CPU.
- **Validation Metrics**: Evaluation suite measuring Intersection over Union (**IoU**), **Dice/F1 Score**, **Precision**, **Recall**, and **Pixel Accuracy**.
- **Unified CLI**: Standardized scripts for modular training (`train.py`), inference (`predict.py`), and dataset conversion (`yolov9_test1.py`).

---

## 📁 Repository Directory Structure

```
SettliteWildfireDetection/
├── AirportSat/
│   └── fasterRCNN.py             # Faster R-CNN detection trainer script
├── Detection/
│   ├── yolov9_test1.py           # YOLOv9 dataset converter & setup pipeline
│   └── yolov9_test1.ipynb        # Exploratory notebook
├── Prediction/
│   └── baseline/
│       └── 24AI_SPARK_baseline.ipynb # Contest baseline reference notebook
├── src/
│   └── wildfire_detection/
│       ├── __init__.py
│       ├── dataset.py            # Dataset loaders, splitters & mask-to-bbox utils
│       ├── models/
│       │   ├── __init__.py
│       │   ├── unet.py           # PyTorch U-Net segmentation network
│       │   └── faster_rcnn.py    # PyTorch Faster R-CNN detection builder
│       └── utils/
│           ├── device.py         # Hardware device auto-selector
│           └── metrics.py        # IoU, Dice, Precision, Recall metrics
├── tests/                        # Unit test suite (pytest)
├── train.py                      # Primary training entrypoint
├── predict.py                    # Inference & submission generator script
├── pyproject.toml                # Package configuration
└── requirements.txt              # Dependency specifications
```

---

## 🚀 Installation

1. **Clone repository**:
   ```bash
   git clone https://github.com/Heisnotanimposter/SettliteWildfireDetection.git
   cd SettliteWildfireDetection
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install package in editable mode**:
   ```bash
   pip install -e .
   ```

---

## 💡 Usage Guide

### 1. Model Training (`train.py`)

Train a U-Net model on your dataset:
```bash
python train.py --model unet --image-dir data/train_img --mask-dir data/train_mask --epochs 20 --batch-size 8
```

Train a Faster R-CNN object detection model:
```bash
python train.py --model fasterrcnn --image-dir data/train_img --mask-dir data/train_mask --epochs 10 --batch-size 2
```

Perform a synthetic dry-run test without downloading datasets:
```bash
python train.py --model unet --dry-run
python train.py --model fasterrcnn --dry-run
```

---

### 2. Inference & Submission Generation (`predict.py`)

Generate binary mask predictions on test satellite images:
```bash
python predict.py --weights weights/best_unet_model.pth --image-dir data/test_img --output-dir predictions
```

Output includes:
- Individual prediction PNG mask images (`predictions/pred_*.png`)
- Serialized pickle file (`predictions/submission_predictions.pkl`) containing binary mask arrays for contest submission.

---

### 3. YOLOv9 Dataset Conversion (`Detection/yolov9_test1.py`)

Convert raw satellite images and binary masks into standard YOLO text annotations and `data.yaml`:
```bash
python Detection/yolov9_test1.py --image-dir data/train_img --mask-dir data/train_mask --output-dir data/yolo_dataset
```

---

## 🧪 Running Unit Tests

Verify dataset loader, metrics calculation, and PyTorch model architectures:
```bash
pytest tests/
```

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
