"""
Dataset loader for MRI Brain Tumor Classification.

Handles:
- Loading MRI brain images from Training/Testing folders
- Multi-class classification (4 tumor types)
- Data augmentation and preprocessing
- Train/validation/test splits
"""

import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path

from .preprocessing import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class MRIBrainTumorDataset(data.Dataset):
    """PyTorch Dataset for MRI brain tumor classification."""
    
    def __init__(
        self,
        folder_path: str,
        image_size: int = 224,
        classes: Optional[List[str]] = None,
        transform=None,
        augment: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            folder_path: Path to folder containing class subfolders
            image_size: Size to resize images to
            classes: List of class names
            transform: Torchvision transforms to apply
            augment: Whether to apply augmentation
        """
        self.folder_path = folder_path
        self.image_size = image_size
        self.classes = classes or ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.augment = augment
        
        if transform is None:
            if augment:
                self.transform = get_train_transforms(image_size)
            else:
                self.transform = get_val_transforms(image_size)
        else:
            self.transform = transform
        
        # Build image list
        self.images = []
        self.labels = []
        self._build_image_list()
        
        logger.info(f"Initialized MRI dataset with {len(self.images)} images from {folder_path}")
    
    def _build_image_list(self):
        """Build list of (image_path, label) pairs."""
        for class_name in self.classes:
            class_dir = os.path.join(self.folder_path, class_name)
            
            if not os.path.exists(class_dir):
                logger.warning(f"Class folder not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in the class directory
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    img_path = os.path.join(class_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        logger.info(f"Found {len(self.images)} images in {len(self.classes)} classes")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get single image and label.
        
        Args:
            idx: Index
            
        Returns:
            tuple: (image_tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MRIDataLoader:
    """Data loader manager for MRI brain tumor dataset."""
    
    def __init__(
        self,
        dataset_dir: str = "data/raw/MRI",
        image_size: int = 224,
        classes: Optional[List[str]] = None,
    ):
        """
        Initialize data loader.
        
        Args:
            dataset_dir: Path to MRI dataset directory
            image_size: Image size for preprocessing
            classes: List of class names
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.classes = classes or ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        self.train_folder = os.path.join(dataset_dir, 'Training')
        self.test_folder = os.path.join(dataset_dir, 'Testing')
        
        # Verify folders exist
        if not os.path.exists(self.train_folder):
            raise FileNotFoundError(f"Training folder not found: {self.train_folder}")
        if not os.path.exists(self.test_folder):
            raise FileNotFoundError(f"Testing folder not found: {self.test_folder}")
        
        logger.info(f"Initialized MRI DataLoader")
        logger.info(f"  Training folder: {self.train_folder}")
        logger.info(f"  Testing folder: {self.test_folder}")
        logger.info(f"  Classes: {', '.join(self.classes)}")
    
    def get_train_val_loaders(
        self,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get training and validation dataloaders.
        
        Args:
            batch_size: Batch size
            val_split: Validation split ratio
            num_workers: Number of workers
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        logger.info("Loading training dataset...")
        
        # Load training dataset
        train_dataset = MRIBrainTumorDataset(
            folder_path=self.train_folder,
            image_size=self.image_size,
            classes=self.classes,
            augment=True,
        )
        
        # Split into train/val
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Training set: {len(train_subset)} images")
        logger.info(f"Validation set: {len(val_subset)} images")
        
        return train_loader, val_loader
    
    def get_test_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> DataLoader:
        """
        Get test dataloader.
        
        Args:
            batch_size: Batch size
            num_workers: Number of workers
            
        Returns:
            DataLoader: Test dataloader
        """
        logger.info("Loading test dataset...")
        
        test_dataset = MRIBrainTumorDataset(
            folder_path=self.test_folder,
            image_size=self.image_size,
            classes=self.classes,
            augment=False,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Test set: {len(test_dataset)} images")
        
        return test_loader
    
    def get_background_loader(
        self,
        batch_size: int = 32,
        num_samples: int = 50,
        num_workers: int = 4,
    ) -> DataLoader:
        """
        Get background loader for SHAP explanations.
        
        Args:
            batch_size: Batch size
            num_samples: Number of background samples
            num_workers: Number of workers
            
        Returns:
            DataLoader: Background dataloader
        """
        # Use training data as background
        dataset = MRIBrainTumorDataset(
            folder_path=self.train_folder,
            image_size=self.image_size,
            classes=self.classes,
            augment=False,
        )
        
        # Sample subset for background
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        background_dataset = data.Subset(dataset, indices)
        
        background_loader = DataLoader(
            background_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Background set: {len(background_dataset)} images")
        
        return background_loader
