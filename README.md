# SettliteWildfireDetection
 2024 AISPARK Mountain fire detection contest

# U-Net for Image Segmentation

This repository contains the implementation of a U-Net model for image segmentation, designed to work with satellite imagery. It uses TensorFlow and Keras to train a deep convolutional network to segment images based on training data provided.

## Prerequisites

- Python 3.6+
- TensorFlow 2.x
- Keras
- Rasterio (for image handling)
- NumPy
- Pandas
- Joblib

## Installation

1. Clone this repository to your local machine.
2. Ensure you have all the required libraries:
    ```bash
    pip install tensorflow keras rasterio numpy pandas joblib
    ```

3. Download the dataset:
    - The dataset is hosted on Google Drive. You will need access to the drive to download it.
    - Mount your Google Drive in the environment where you're running this notebook:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```
    - Navigate to the shared folder and download the dataset and pre-trained weights.

## Directory Structure

Ensure your directory is structured as follows:
/content/drive/MyDrive/twofouraispark/
│
├── train_img/ # Training images
├── train_mask/ # Training masks (if available, otherwise use unsupervised techniques as mentioned)
├── test_img/ # Test images
├── train_meta.csv # Metadata for training images
├── test_meta.csv # Metadata for test images
└── weights/ # Folder for trained model weights

markdown
Copy code

## Usage

### Training

1. Open the training script:
    - Adjust parameters such as `EPOCHS`, `BATCH_SIZE`, and paths to data if necessary.
    - Configure GPU settings as required by your setup.

2. To train the model, run:
    ```bash
    python train.py
    ```
    This script will train a U-Net model on the specified training data, save checkpoints, and save the final model weights in the `train_output` directory.

### Inference

1. To perform inference using the trained model, run:
    ```bash
    python inference.py
    ```
    - This script will load the model from the saved weights and perform segmentation on the test set.
    - Predictions will be saved as `.pkl` files containing binary masks of the segmented areas.

## Model Details

- **Architecture**: U-Net
- **Input size**: Configurable, default is 256x256 pixels
- **Output**: Binary mask of the segmented regions

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


