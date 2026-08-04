import os
from typing import Optional


def download_kaggle_dataset(dataset_handle: str = "abdelghaniaaba/wildfire-prediction-dataset") -> str:
    """
    Downloads a Kaggle dataset using kagglehub and returns the local dataset directory path.
    
    Args:
        dataset_handle (str): Kaggle dataset identifier (owner/dataset-name).
        
    Returns:
        str: Absolute path to the downloaded dataset directory.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub package is required for downloading Kaggle datasets. "
            "Install it via `pip install kagglehub`."
        )

    print(f"Downloading dataset '{dataset_handle}' via kagglehub...")
    dataset_path = kagglehub.dataset_download(dataset_handle)
    print(f"Dataset successfully downloaded to: {dataset_path}")
    return dataset_path
