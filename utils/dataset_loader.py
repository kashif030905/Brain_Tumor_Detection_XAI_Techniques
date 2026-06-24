"""
Dataset loader for NIH ChestX-ray14 dataset.

Handles:
- Scanning image directories (images_001 to images_012)
- Loading and parsing Data_Entry_2017.csv
- Creating image-to-label mappings
- Supporting multi-label classification with 14 disease classes
- Train/val/test splits
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from .preprocessing import preprocess_image, get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

# 14 disease classes in ChestX-ray14
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


class NIHChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14.
    
    Returns:
        tuple: (image_tensor, label_tensor) where label_tensor is multi-hot encoded.
    """
    
    def __init__(
        self,
        dataset_dir: str,
        image_list: List[str],
        labels_df: pd.DataFrame,
        image_size: int = 224,
        augment: bool = False,
        transform=None,
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_dir: Root directory containing images_001 to images_012.
            image_list: List of image filenames (e.g., '00000001_000.png').
            labels_df: DataFrame with image names and disease labels.
            image_size: Size to resize images to.
            augment: Whether to apply data augmentation.
            transform: Optional custom transform.
        """
        self.dataset_dir = dataset_dir
        self.image_list = image_list
        self.labels_df = labels_df
        self.image_size = image_size
        self.augment = augment
        self.transform = transform
        
        # Build image path mapping
        self.image_paths = self._build_image_paths()
        
        logger.info(f"Initialized dataset with {len(self.image_list)} images")
    
    def _build_image_paths(self) -> Dict[str, str]:
        """
        Build mapping from image filename to full path.
        
        Scans images_001 to images_012 directories.
        
        Returns:
            dict: Mapping {filename: full_path}.
        """
        image_paths = {}
        
        # Scan all image folders
        for folder_idx in range(1, 13):  # images_001 to images_012
            folder_name = f"images_{folder_idx:03d}"
            folder_path = os.path.join(self.dataset_dir, folder_name, "images")
            
            if not os.path.exists(folder_path):
                logger.warning(f"Image folder not found: {folder_path}")
                continue
            
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(folder_path, img_file)
                    image_paths[img_file] = full_path
        
        logger.info(f"Found {len(image_paths)} total images in dataset")
        return image_paths
    
    def _labels_to_multihot(self, labels_str: str) -> np.ndarray:
        """
        Convert disease labels string to multi-hot vector.
        
        Args:
            labels_str: String with disease labels separated by '|'.
                       E.g., 'Infiltration|Pneumothorax' or 'No Finding'.
        
        Returns:
            np.ndarray: Multi-hot vector of shape (14,).
        """
        multi_hot = np.zeros(len(DISEASE_CLASSES), dtype=np.float32)
        
        if pd.isna(labels_str) or labels_str == 'No Finding':
            return multi_hot
        
        diseases = [d.strip() for d in str(labels_str).split('|')]
        
        for disease in diseases:
            if disease in DISEASE_CLASSES:
                idx = DISEASE_CLASSES.index(disease)
                multi_hot[idx] = 1.0
        
        return multi_hot
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            tuple: (image_tensor, label_tensor)
        """
        img_filename = self.image_list[idx]
        
        # Get image path
        if img_filename not in self.image_paths:
            raise FileNotFoundError(f"Image not found: {img_filename}")
        
        img_path = self.image_paths[img_filename]
        
        # Load and preprocess image
        try:
            if self.transform is not None:
                image = self.transform(img_path)
            else:
                image = preprocess_image(
                    img_path,
                    image_size=self.image_size,
                    augment=self.augment
                )
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
        
        # Get labels
        row = self.labels_df[self.labels_df['Image Index'] == img_filename]
        if len(row) == 0:
            logger.warning(f"No label found for {img_filename}, using all zeros")
            labels = np.zeros(len(DISEASE_CLASSES), dtype=np.float32)
        else:
            labels_str = row.iloc[0]['Finding Labels']
            labels = self._labels_to_multihot(labels_str)
        
        return image, torch.tensor(labels, dtype=torch.float32)


class ChestXrayDataLoader:
    """Helper class to create dataloaders for ChestX-ray dataset."""
    
    def __init__(self, dataset_dir: str, csv_path: str, image_size: int = 224):
        """
        Initialize dataloader manager.
        
        Args:
            dataset_dir: Root directory containing images_001 to images_012.
            csv_path: Path to Data_Entry_2017.csv.
            image_size: Size to resize images to.
        """
        self.dataset_dir = dataset_dir
        self.csv_path = csv_path
        self.image_size = image_size
        self.labels_df = None
        self.all_images = None
        
        self._load_labels_and_images()
    
    def _load_labels_and_images(self):
        """Load labels from CSV and build image list."""
        logger.info(f"Loading labels from {self.csv_path}")
        
        if not os.path.exists(self.csv_path):
            error_msg = (
                f"\n{'='*70}\n"
                f"❌ DATASET NOT FOUND\n"
                f"{'='*70}\n"
                f"CSV file not found: {self.csv_path}\n\n"
                f"To download the NIH ChestX-ray14 dataset:\n"
                f"  1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC\n"
                f"  2. Download and extract to: {self.dataset_dir}/\n"
                f"  3. The directory should contain:\n"
                f"     - Data_Entry_2017.csv\n"
                f"     - train_val_list.txt\n"
                f"     - test_list.txt\n"
                f"     - BBox_List_2017.csv\n"
                f"     - images_001/ through images_012/ (image folders)\n"
                f"{'='*70}\n"
            )
            raise FileNotFoundError(error_msg)
        
        self.labels_df = pd.read_csv(self.csv_path)
        self.all_images = self.labels_df['Image Index'].tolist()
        
        logger.info(f"Loaded {len(self.all_images)} image labels from CSV")
    
    def _load_split_list(self, split_file: str) -> List[str]:
        """
        Load image names from split file (train_val_list.txt or test_list.txt).
        
        Args:
            split_file: Path to the split file.
            
        Returns:
            list: Image names in the split.
        """
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            images = [line.strip() for line in f.readlines()]
        
        return images
    
    def get_train_val_loaders(
        self,
        train_val_list_path: str,
        batch_size: int = 32,
        val_split: float = 0.1,
        num_workers: int = 4,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders.
        
        Args:
            train_val_list_path: Path to train_val_list.txt.
            batch_size: Batch size.
            val_split: Fraction of training data to use for validation.
            num_workers: Number of workers for data loading.
            shuffle_train: Whether to shuffle training data.
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        train_val_images = self._load_split_list(train_val_list_path)
        
        # Split into train and val
        n_train = int(len(train_val_images) * (1 - val_split))
        train_images = train_val_images[:n_train]
        val_images = train_val_images[n_train:]
        
        logger.info(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
        
        # Create datasets
        train_dataset = NIHChestXrayDataset(
            self.dataset_dir,
            train_images,
            self.labels_df,
            image_size=self.image_size,
            augment=True,
            transform=None,
        )
        
        val_dataset = NIHChestXrayDataset(
            self.dataset_dir,
            val_images,
            self.labels_df,
            image_size=self.image_size,
            augment=False,
            transform=None,
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader
    
    def get_test_loader(
        self,
        test_list_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> DataLoader:
        """
        Create test dataloader.
        
        Args:
            test_list_path: Path to test_list.txt.
            batch_size: Batch size.
            num_workers: Number of workers for data loading.
            
        Returns:
            DataLoader: Test dataloader.
        """
        test_images = self._load_split_list(test_list_path)
        
        logger.info(f"Test images: {len(test_images)}")
        
        test_dataset = NIHChestXrayDataset(
            self.dataset_dir,
            test_images,
            self.labels_df,
            image_size=self.image_size,
            augment=False,
            transform=None,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return test_loader
    
    def get_background_loader(
        self,
        train_val_list_path: str,
        num_samples: int = 50,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> DataLoader:
        """
        Create background dataloader for SHAP explainer.
        
        Args:
            train_val_list_path: Path to train_val_list.txt.
            num_samples: Number of background samples to use.
            batch_size: Batch size.
            num_workers: Number of workers for data loading.
            
        Returns:
            DataLoader: Background dataloader.
        """
        train_val_images = self._load_split_list(train_val_list_path)
        
        # Sample background images
        if len(train_val_images) > num_samples:
            np.random.seed(42)
            bg_images = list(np.random.choice(train_val_images, num_samples, replace=False))
        else:
            bg_images = train_val_images
        
        logger.info(f"Background images: {len(bg_images)}")
        
        bg_dataset = NIHChestXrayDataset(
            self.dataset_dir,
            bg_images,
            self.labels_df,
            image_size=self.image_size,
            augment=False,
            transform=None,
        )
        
        bg_loader = DataLoader(
            bg_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return bg_loader


def get_disease_classes() -> List[str]:
    """Get list of 14 disease classes."""
    return DISEASE_CLASSES.copy()
