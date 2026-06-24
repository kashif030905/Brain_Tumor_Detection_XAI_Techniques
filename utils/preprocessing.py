"""
Image preprocessing utilities for ChestX-ray data.

Provides consistent preprocessing for both training and inference,
including normalization, resizing, and augmentation.
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import logging

logger = logging.getLogger(__name__)


class ImageNetNormalizer:
    """ImageNet normalization statistics."""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=224):
    """
    Get augmented transforms for training.
    
    Args:
        image_size (int): Size to resize images to.
        
    Returns:
        transforms.Compose: Composition of transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ImageNetNormalizer.MEAN,
            std=ImageNetNormalizer.STD
        ),
    ])


def get_val_transforms(image_size=224):
    """
    Get deterministic transforms for validation and testing.
    
    Args:
        image_size (int): Size to resize images to.
        
    Returns:
        transforms.Compose: Composition of transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ImageNetNormalizer.MEAN,
            std=ImageNetNormalizer.STD
        ),
    ])


def preprocess_image(image_path, image_size=224, augment=False):
    """
    Load and preprocess a single image from file.
    
    Args:
        image_path (str): Path to the image file.
        image_size (int): Size to resize to. Default: 224.
        augment (bool): Whether to apply augmentation. Default: False.
        
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (3, image_size, image_size).
        
    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If image cannot be loaded.
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Cannot load image {image_path}: {str(e)}")
    
    transforms_fn = get_train_transforms(image_size) if augment else get_val_transforms(image_size)
    return transforms_fn(image)


def preprocess_tensor(image_tensor, image_size=224, augment=False):
    """
    Preprocess an already-loaded image tensor.
    
    Args:
        image_tensor (torch.Tensor): Image tensor or PIL Image.
        image_size (int): Size to resize to. Default: 224.
        augment (bool): Whether to apply augmentation. Default: False.
        
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if not isinstance(image_tensor, (torch.Tensor, Image.Image)):
        raise TypeError(f"Expected torch.Tensor or PIL.Image, got {type(image_tensor)}")
    
    if isinstance(image_tensor, torch.Tensor):
        # Convert back to PIL if tensor
        if image_tensor.max() > 1:
            image_tensor = image_tensor / 255.0
        image_tensor = transforms.ToPILImage()(image_tensor)
    
    transforms_fn = get_train_transforms(image_size) if augment else get_val_transforms(image_size)
    return transforms_fn(image_tensor)


def denormalize_image(image_tensor):
    """
    Denormalize an image tensor using ImageNet statistics.
    
    Args:
        image_tensor (torch.Tensor): Normalized image tensor of shape (C, H, W) or (B, C, H, W).
        
    Returns:
        torch.Tensor: Denormalized image tensor in range [0, 1].
    """
    if len(image_tensor.shape) == 3:
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    mean = torch.tensor(ImageNetNormalizer.MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(ImageNetNormalizer.STD, device=image_tensor.device).view(1, 3, 1, 1)
    
    denormalized = image_tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    if squeeze_output:
        denormalized = denormalized.squeeze(0)
    
    return denormalized


def get_image_stats(image_tensor):
    """
    Get statistics (mean, std) of an image tensor.
    
    Args:
        image_tensor (torch.Tensor): Image tensor.
        
    Returns:
        dict: Statistics including mean, std, min, max.
    """
    return {
        'mean': image_tensor.mean().item(),
        'std': image_tensor.std().item(),
        'min': image_tensor.min().item(),
        'max': image_tensor.max().item(),
    }


def convert_to_uint8(image_tensor):
    """
    Convert a tensor to uint8 format for visualization.
    
    Args:
        image_tensor (torch.Tensor): Image tensor in range [0, 1] or [0, 255].
        
    Returns:
        np.ndarray: Image array in uint8 format.
    """
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.cpu().numpy()
    
    # Handle channel-first format (C, H, W)
    if image_tensor.ndim == 3 and image_tensor.shape[0] in [1, 3]:
        image_tensor = np.transpose(image_tensor, (1, 2, 0))
    
    # Squeeze if single channel
    if image_tensor.ndim == 3 and image_tensor.shape[2] == 1:
        image_tensor = image_tensor.squeeze(2)
    
    # Convert to uint8
    if image_tensor.max() <= 1.0:
        image_array = (image_tensor * 255).astype(np.uint8)
    else:
        image_array = image_tensor.astype(np.uint8)
    
    return image_array
