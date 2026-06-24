"""
SHAP explainer for chest X-ray model.

Uses DeepExplainer with gradient-based attribution to explain
model predictions on medical images.
"""

import torch
import torch.nn as nn
import numpy as np
import shap
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
from pathlib import Path

from .base_explainer import Explainer, ExplainerFactory
from utils.preprocessing import denormalize_image, convert_to_uint8
from utils.visualization import overlay_heatmap

logger = logging.getLogger(__name__)


class SHAPCompatibleWrapper(nn.Module):
    """
    Wrapper to make models SHAP-compatible by handling gradient computation.
    Disables eval mode's no_grad context during SHAP computation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Ensure gradients are enabled for SHAP
        return self.model(x)


class SHAPExplainer(Explainer):
    """
    SHAP explainer for neural networks.
    
    Implements DeepExplainer which uses gradient-based attribution
    to compute Shapley values for image classification.
    """
    
    def __init__(
        self,
        background_loader: Optional[torch.utils.data.DataLoader] = None,
        num_samples: int = 100,
        device: str = 'cuda',
        batch_size: int = 32,
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            background_loader: DataLoader for background images used to establish
                             baseline for Shapley value computation.
                             If None, background will be created during explain().
            num_samples: Number of background samples to use.
            device: Device to use ('cuda' or 'cpu').
            batch_size: Batch size for explanation computation.
        """
        super().__init__('SHAP')
        
        self.background_loader = background_loader
        self.num_samples = num_samples
        self.device = device
        self.batch_size = batch_size
        
        self.explainer = None
        self.background = None
        
        logger.info(f"Initialized SHAP explainer on device: {device}")
    
    def _prepare_background(self, background_loader) -> torch.Tensor:
        """
        Prepare background images for SHAP.
        
        Args:
            background_loader: DataLoader with background images.
            
        Returns:
            torch.Tensor: Background images of shape (num_samples, 3, H, W).
        """
        if self.background is not None:
            return self.background
        
        background_images = []
        num_collected = 0
        
        logger.info(f"Collecting {self.num_samples} background images...")
        
        for images, _ in background_loader:
            background_images.append(images)
            num_collected += images.shape[0]
            
            if num_collected >= self.num_samples:
                break
        
        background = torch.cat(background_images, dim=0)[:self.num_samples]
        background = background.to(self.device)
        
        self.background = background
        logger.info(f"Prepared background with {len(background)} images")
        
        return background
    
    def _create_explainer(
        self,
        model: nn.Module,
        background: torch.Tensor,
    ):
        """
        Create SHAP GradientExplainer (simpler than DeepExplainer, no autograd issues).
        
        Args:
            model: PyTorch model to explain.
            background: Background images for baseline.
            
        Returns:
            shap.GradientExplainer: SHAP explainer instance.
        """
        logger.info("Creating SHAP GradientExplainer...")
        
        # Use GradientExplainer instead of DeepExplainer
        # GradientExplainer computes gradients of outputs w.r.t. inputs
        explainer = shap.GradientExplainer(
            model,
            background,
        )
        
        self.explainer = explainer
        logger.info("Created SHAP GradientExplainer")
        
        return explainer
    
    def explain(
        self,
        image: torch.Tensor,
        model: nn.Module,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single image.
        
        Args:
            image: Input image of shape (1, 3, H, W) or (3, H, W).
            model: Model to explain.
            target_class: Target class for explanation.
                         If None, average over all classes.
        
        Returns:
            dict: Explanation with keys:
                - 'attributions': Heatmap of shape (H, W) or (num_classes, H, W)
                - 'predictions': Model predictions
                - 'shap_values': Raw SHAP values
                - 'method': 'SHAP'
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Prepare background if needed
        if self.background_loader is not None and self.background is None:
            background = self._prepare_background(self.background_loader)
        elif self.background is not None:
            background = self.background
        else:
            raise ValueError(
                "Must provide background_loader or set background images"
            )
        
        # Create explainer if needed
        if self.explainer is None:
            self._create_explainer(model, background)
        
        # Compute SHAP values
        logger.info("Computing SHAP values...")
        model.eval()
        
        # Enable gradients for SHAP computation (required by DeepExplainer)
        image = image.requires_grad_(True)
        
        shap_values = self.explainer.shap_values(image)
        
        # Get predictions without gradients
        with torch.no_grad():
            predictions = torch.sigmoid(model(image.detach()))
        
        # shap_values is a list [num_classes] of arrays with shape (1, 3, H, W)
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=0)  # (num_classes, 1, 3, H, W)
            shap_values = shap_values.squeeze(1)  # (num_classes, 3, H, W)
        
        # Convert to numpy (detach first since image requires grad)
        image_np = image.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()[0]
        
        # Create attribution heatmap by averaging across channels
        if shap_values.ndim == 4:
            # (num_classes, 3, H, W) -> average across channels -> (num_classes, H, W)
            attributions = np.mean(np.abs(shap_values), axis=1)
        else:
            attributions = np.abs(shap_values)
        
        # If target class specified, select only that class
        if target_class is not None:
            if isinstance(target_class, int):
                attributions = attributions[target_class]
            elif isinstance(target_class, list):
                attributions = attributions[target_class]
                attributions = attributions.mean(axis=0)
        else:
            # Average across all classes
            attributions = attributions.mean(axis=0)
        
        return {
            'attributions': attributions,
            'predictions': predictions_np,
            'shap_values': shap_values,
            'method': 'SHAP',
            'image': image_np,
            'target_class': target_class,
        }
    
    def explain_batch(
        self,
        images: torch.Tensor,
        model: nn.Module,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a batch of images.
        
        Args:
            images: Batch of images of shape (B, 3, H, W).
            model: Model to explain.
            target_class: Target class for explanation.
        
        Returns:
            dict: Batch explanations.
        """
        images = images.to(self.device)
        
        # Prepare background if needed
        if self.background_loader is not None and self.background is None:
            background = self._prepare_background(self.background_loader)
        elif self.background is not None:
            background = self.background
        else:
            raise ValueError("Must provide background_loader")
        
        # Create explainer if needed
        if self.explainer is None:
            self._create_explainer(model, background)
        
        # Compute SHAP values
        logger.info(f"Computing SHAP values for {len(images)} images...")
        model.eval()
        
        all_shap_values = []
        all_attributions = []
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            with torch.no_grad():
                shap_vals = self.explainer.shap_values(batch)
                preds = torch.sigmoid(model(batch))
            
            if isinstance(shap_vals, list):
                shap_vals = np.stack(shap_vals, axis=0).squeeze(1)
            
            attributions = np.mean(np.abs(shap_vals), axis=1)
            
            if target_class is not None:
                if isinstance(target_class, int):
                    attributions = attributions[:, target_class]
                else:
                    attributions = attributions[:, target_class].mean(axis=1)
            else:
                attributions = attributions.mean(axis=1)
            
            all_shap_values.append(shap_vals)
            all_attributions.append(attributions)
            all_predictions.append(preds.cpu().numpy())
        
        return {
            'attributions': np.concatenate(all_attributions, axis=0),
            'predictions': np.concatenate(all_predictions, axis=0),
            'shap_values': np.concatenate(all_shap_values, axis=0),
            'method': 'SHAP',
            'batch_size': len(images),
            'target_class': target_class,
        }
    
    def visualize_explanation(
        self,
        image: np.ndarray,
        attributions: np.ndarray,
        save_path: str,
        title: str = "",
    ) -> None:
        """
        Visualize explanation by overlaying heatmap on image.
        
        Args:
            image: Original image array.
            attributions: Attribution heatmap.
            save_path: Path to save visualization.
            title: Title for the figure.
        """
        overlay_heatmap(
            image=image,
            heatmap=attributions,
            alpha=0.5,
            cmap='jet',
            title=title,
            save_path=save_path,
        )
        logger.info(f"Saved explanation visualization to {save_path}")


# Register SHAP explainer in factory
ExplainerFactory.register('shap', SHAPExplainer)
