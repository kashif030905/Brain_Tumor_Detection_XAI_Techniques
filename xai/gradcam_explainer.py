"""
GradCAM explainer for medical image model.

Uses Gradient-weighted Class Activation Mapping (GradCAM) to generate
visual explanations of model predictions. Much faster than SHAP and
works reliably with ResNet models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path
import cv2

from .base_explainer import Explainer, ExplainerFactory
from utils.preprocessing import denormalize_image, convert_to_uint8

logger = logging.getLogger(__name__)


class GradCAMExplainer(Explainer):
    """
    GradCAM explainer for neural networks.
    
    Uses gradient-based Class Activation Mapping to generate heatmaps
    showing which regions of the image are important for the model's
    predictions.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
    ):
        """
        Initialize GradCAM explainer.
        
        Args:
            device: Device to use ('cuda' or 'cpu').
        """
        super().__init__('GradCAM')
        
        self.device = device
        self.gradients = None
        self.activations = None
        
        logger.info(f"Initialized GradCAM explainer on device: {device}")
    
    def _register_hooks(self, model: nn.Module):
        """Register hooks to capture gradients and activations."""
        
        # Get the last convolutional layer (layer4)
        target_layer = model.resnet.layer4[-1]
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def explain(
        self,
        image: torch.Tensor,
        model: nn.Module,
        predicted_class: Optional[int] = None,
        class_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Generate GradCAM explanation for a single image.
        
        Args:
            image: Input image of shape (1, 3, H, W) or (3, H, W).
            model: Model to explain.
            predicted_class: Target class for explanation.
                           If None, use the model's prediction.
            class_names: List of class names.
        
        Returns:
            dict: Explanation with keys:
                - 'heatmap': GradCAM heatmap of shape (H, W)
                - 'predictions': Model predictions
                - 'method': 'GradCAM'
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad_(True)
        
        # Register hooks
        self._register_hooks(model)
        
        # Forward pass
        model.eval()
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)
        
        # Get target class
        if predicted_class is None:
            predicted_class = probabilities.argmax(dim=1).item()
        
        # Get the score for target class
        score = logits[0, predicted_class]
        
        # Backward pass
        model.zero_grad()
        score.backward()
        
        # Compute GradCAM
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)
        
        # Compute channel weights (average gradient over spatial dimensions)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted combination of activations
        heatmap = np.zeros(activations.shape[1:])  # (H, W)
        for c, w in enumerate(weights):
            heatmap += w * activations[c]
        
        # ReLU to keep only positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Upsample to input image size (224x224)
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Convert to numpy
        predictions_np = probabilities.cpu().detach().numpy()[0]
        image_np = image.cpu().detach().numpy()[0]
        
        return {
            'heatmap': heatmap,
            'predictions': predictions_np,
            'predicted_class': predicted_class,
            'method': 'GradCAM',
            'logits': logits[0].cpu().detach().numpy(),
        }
    
    def explain_batch(
        self,
        images: torch.Tensor,
        model: nn.Module,
        target_class: int = None,
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanations for a batch of images.
        
        Args:
            images: Batch of input images.
            model: CNN model.
            target_class: Target class for explanation.
            
        Returns:
            dict: Batch explanations.
        """
        # For batch, just process each image individually
        batch_size = images.shape[0]
        explanations = []
        
        for i in range(batch_size):
            exp = self.explain(
                image=images[i:i+1],
                model=model,
                target_class=target_class,
            )
            explanations.append(exp)
        
        return {
            'explanations': explanations,
            'method': 'GradCAM',
            'batch_size': batch_size,
        }


def create_gradcam_explainer(
    device: str = 'cuda',
    **kwargs
) -> GradCAMExplainer:
    """Factory method for GradCAM explainer."""
    return GradCAMExplainer(device=device)
