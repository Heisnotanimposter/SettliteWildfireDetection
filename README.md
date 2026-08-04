# Satellite Wildfire Detection, Segmentation & Classification Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Modular machine learning and computer vision framework for **Satellite Wildfire Detection, Image Segmentation, and Classification** developed for the AISPARK Competition and Kaggle datasets.

---

## 🌟 Key Features

- **Supported Tasks**:
  - **Image Classification**: Predict whether a satellite image region is at risk of a wildfire (`wildfire` vs `nowildfire`) using pretrained `resnet18`, `resnet50`, or `efficientnet_b0`.
  - **Semantic Segmentation**: U-Net architecture for high-resolution pixel-level wildfire burned area segmentation.
  - **Object Detection**: Faster R-CNN & YOLOv9 pipelines for bounding-box wildfire localization.
- **Automatic Kaggle Dataset Downloader**: Seamless integration with `kagglehub` to fetch datasets like [`abdelghaniaaba/wildfire-prediction-dataset`](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset).
- **Hardware Acceleration**: Auto-device selector for NVIDIA CUDA, Apple Silicon (`mps`), and CPU.
- **Evaluation Suite**: Includes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, IoU, and Dice score.

---

## 📁 Directory Structure

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
│       ├── data_downloader.py    # Kaggle dataset downloader (kagglehub)
│       ├── dataset.py            # Classification, Segmentation & Detection loaders
│       ├── models/
│       │   ├── __init__.py
│       │   ├── classifier.py     # ResNet / EfficientNet classification backbones
│       │   ├── unet.py           # PyTorch U-Net segmentation network
│       │   └── faster_rcnn.py    # PyTorch Faster R-CNN detection builder
│       └── utils/
│           ├── device.py         # Hardware device auto-selector
│           └── metrics.py        # Classification & Segmentation metrics
├── tests/                        # Unit test suite (pytest)
├── train.py                      # Unified training launcher
├── predict.py                    # Inference launcher script
├── pyproject.toml                # Package configuration
└── requirements.txt              # Dependency specifications
```

---

## 🚀 Quick Start & Installation

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Install package in editable mode**:
   ```bash
   pip install -e .
   ```

---

## 💡 Usage Guide

### 1. Automatic Dataset Download & Image Classification (`train.py`)

Automatically download the Kaggle Wildfire Prediction Dataset and train a ResNet18 classifier:
```bash
python train.py --model classifier --download-kaggle --kaggle-handle abdelghaniaaba/wildfire-prediction-dataset --epochs 10 --batch-size 32
```

Train a classifier on a local dataset directory containing `wildfire/` and `nowildfire/` subfolders:
```bash
python train.py --model classifier --image-dir path/to/dataset --backbone resnet50 --epochs 15
```

### 2. Semantic Segmentation (`train.py`)

Train a U-Net model:
```bash
python train.py --model unet --image-dir data/train_img --mask-dir data/train_mask --epochs 20
```

### 3. Dry-Run Check (Synthetic Data Test)

```bash
python train.py --model classifier --dry-run
python train.py --model unet --dry-run
python train.py --model fasterrcnn --dry-run
```

---

### 4. Classification & Segmentation Inference (`predict.py`)

Run classification inference on test satellite images:
```bash
python predict.py --task classification --weights weights/best_classifier_resnet18.pth --image-dir data/test --output-dir predictions
```
*Outputs `predictions/classification_results.json` containing predicted classes (`wildfire` / `nowildfire`) and confidence scores.*

Run segmentation inference:
```bash
python predict.py --task segmentation --weights weights/best_unet_model.pth --image-dir data/test_img --output-dir predictions
```

---

## 🧪 Running Unit Tests

Verify the codebase (dataset loaders, models, classification metrics, segmentation metrics):
```bash
pytest tests/
```

---

## 📜 License

Licensed under the MIT License - see [LICENSE](LICENSE) for details.
