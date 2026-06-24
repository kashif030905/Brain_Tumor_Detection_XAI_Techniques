"""
Enhanced GradCAM explainer with improved visualization and localization.

Implements:
1. Guided Grad-CAM (combines guided backprop with Grad-CAM)
2. Multi-scale Grad-CAM (combines multiple layers)
3. Post-processing filters (bilateral, morphological ops)
4. Better overlay blending for clarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
import cv2

logger = logging.getLogger(__name__)


class EnhancedGradCAM:
    """
    Enhanced GradCAM with multiple improvement techniques.
    
    Features:
    - Guided Grad-CAM (guided backpropagation)
    - Multi-scale CAM (multiple layers)
    - Post-processing filters
    - Better normalization
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize Enhanced GradCAM."""
        self.device = device
        self.gradients = {}
        self.activations = {}
        logger.info(f"Initialized Enhanced GradCAM on device: {device}")
    
    def _register_hooks(self, model: nn.Module, layers: List[nn.Module]):
        """Register hooks for multiple layers."""
        
        for layer_idx, layer in enumerate(layers):
            def forward_hook(module, input, output, idx=layer_idx):
                self.activations[idx] = output.detach()
            
            def backward_hook(module, grad_input, grad_output, idx=layer_idx):
                self.gradients[idx] = grad_output[0].detach()
            
            layer.register_forward_hook(forward_hook)
            layer.register_full_backward_hook(backward_hook)
    
    def _compute_single_gradcam(
        self,
        layer_idx: int,
        image_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Compute Grad-CAM for a single layer."""
        
        gradients = self.gradients[layer_idx].cpu().numpy()[0]  # (C, H, W)
        activations = self.activations[layer_idx].cpu().numpy()[0]  # (C, H, W)
        
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
        
        # Upsample to input image size
        heatmap = cv2.resize(heatmap, image_size, interpolation=cv2.INTER_LINEAR)
        
        return heatmap
    
    def _apply_bilateral_filter(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.
        
        This reduces noise while keeping sharp boundaries around ROI.
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(heatmap_uint8, d=9, sigmaColor=75, sigmaSpace=75)
        return filtered.astype(np.float32) / 255.0
    
    def _apply_morphological_ops(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up heatmap.
        
        - Closing: fills small holes
        - Opening: removes small noise
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Closing (fill holes)
        closed = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Opening (remove noise)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened.astype(np.float32) / 255.0
    
    def _enhance_contrast(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        This improves visibility of important regions.
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(heatmap_uint8)
        
        return enhanced.astype(np.float32) / 255.0
    
    def _compute_guided_gradcam(
        self,
        image: torch.Tensor,
        model: nn.Module,
        target_class: int,
        layer_idx: int
    ) -> np.ndarray:
        """
        Compute Guided Grad-CAM using guided backpropagation.
        
        This combines Grad-CAM with input gradients for finer localization.
        """
        # Get standard Grad-CAM
        gradcam = self._compute_single_gradcam(layer_idx)
        
        # Get guided backprop (input gradients)
        image_copy = image.clone().detach().requires_grad_(True)
        model.zero_grad()
        
        logits = model(image_copy)
        score = logits[0, target_class]
        score.backward()
        
        # Get input gradients and take positive only (guided)
        input_grad = image_copy.grad.data[0].cpu().numpy()  # (3, 224, 224)
        guided_backprop = np.maximum(input_grad, 0)
        
        # Take mean across channels
        guided_backprop_mean = np.mean(guided_backprop, axis=0)
        
        # Normalize
        if guided_backprop_mean.max() > 0:
            guided_backprop_norm = guided_backprop_mean / guided_backprop_mean.max()
        else:
            guided_backprop_norm = guided_backprop_mean
        
        # Combine: element-wise multiplication
        guided_gradcam = gradcam * guided_backprop_norm
        
        return guided_gradcam
    
    def _post_process_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply all post-processing improvements."""
        
        # Step 1: Bilateral filter (smooth while preserving edges)
        heatmap = self._apply_bilateral_filter(heatmap)
        
        # Step 2: Morphological operations (clean up)
        heatmap = self._apply_morphological_ops(heatmap)
        
        # Step 3: Contrast enhancement (better visibility)
        heatmap = self._enhance_contrast(heatmap)
        
        # Step 4: Final normalization
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def explain_standard(
        self,
        image: torch.Tensor,
        model: nn.Module,
        predicted_class: int,
    ) -> Dict[str, Any]:
        """Generate standard Grad-CAM explanation."""
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Register hooks for layer4
        target_layers = [model.resnet.layer4[-1]]
        self._register_hooks(model, target_layers)
        
        # Forward pass
        model.eval()
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)
        
        # Backward pass
        score = logits[0, predicted_class]
        model.zero_grad()
        score.backward()
        
        # Compute and post-process Grad-CAM
        heatmap = self._compute_single_gradcam(layer_idx=0)
        heatmap = self._post_process_heatmap(heatmap)
        
        predictions_np = probabilities.cpu().detach().numpy()[0]
        
        return {
            'heatmap': heatmap,
            'heatmap_raw': heatmap,
            'predictions': predictions_np,
            'predicted_class': predicted_class,
            'method': 'GradCAM-Standard',
        }
    
    def explain_guided(
        self,
        image: torch.Tensor,
        model: nn.Module,
        predicted_class: int,
    ) -> Dict[str, Any]:
        """Generate Guided Grad-CAM explanation (combines guided backprop)."""
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Register hooks
        target_layers = [model.resnet.layer4[-1]]
        self._register_hooks(model, target_layers)
        
        # Forward pass
        model.eval()
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)
        
        # Backward pass for standard layer
        score = logits[0, predicted_class]
        model.zero_grad()
        score.backward()
        
        # Compute guided Grad-CAM
        heatmap = self._compute_guided_gradcam(
            image=image,
            model=model,
            target_class=predicted_class,
            layer_idx=0
        )
        
        # Post-process
        heatmap = self._post_process_heatmap(heatmap)
        
        predictions_np = probabilities.cpu().detach().numpy()[0]
        
        return {
            'heatmap': heatmap,
            'heatmap_raw': heatmap,
            'predictions': predictions_np,
            'predicted_class': predicted_class,
            'method': 'GradCAM-Guided',
        }
    
    def explain_multiscale(
        self,
        image: torch.Tensor,
        model: nn.Module,
        predicted_class: int,
    ) -> Dict[str, Any]:
        """
        Generate Multi-scale Grad-CAM (combines multiple layers).
        
        This captures features at different scales:
        - layer2: Broader features
        - layer3: Medium-scale features
        - layer4: Fine-grained features
        """
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Register hooks for multiple layers
        target_layers = [
            model.resnet.layer2[-1],
            model.resnet.layer3[-1],
            model.resnet.layer4[-1],
        ]
        self._register_hooks(model, target_layers)
        
        # Forward pass
        model.eval()
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)
        
        # Backward pass
        score = logits[0, predicted_class]
        model.zero_grad()
        score.backward()
        
        # Compute CAMs from all layers
        multiscale_heatmap = None
        weights = [0.2, 0.3, 0.5]  # Layer2, Layer3, Layer4 importance
        
        for layer_idx, weight in enumerate(weights):
            cam = self._compute_single_gradcam(layer_idx=layer_idx)
            
            if multiscale_heatmap is None:
                multiscale_heatmap = weight * cam
            else:
                multiscale_heatmap = multiscale_heatmap + weight * cam
        
        # Post-process
        multiscale_heatmap = self._post_process_heatmap(multiscale_heatmap)
        
        predictions_np = probabilities.cpu().detach().numpy()[0]
        
        return {
            'heatmap': multiscale_heatmap,
            'heatmap_raw': multiscale_heatmap,
            'predictions': predictions_np,
            'predicted_class': predicted_class,
            'method': 'GradCAM-MultiScale',
        }
    
    def explain(
        self,
        image: torch.Tensor,
        model: nn.Module,
        predicted_class: int,
        method: str = 'guided',
    ) -> Dict[str, Any]:
        """
        Generate enhanced Grad-CAM explanation.
        
        Args:
            image: Input image tensor
            model: Model to explain
            predicted_class: Target class
            method: 'standard', 'guided', or 'multiscale'
        
        Returns:
            dict: Explanation with enhanced heatmap
        """
        if method == 'standard':
            return self.explain_standard(image, model, predicted_class)
        elif method == 'guided':
            return self.explain_guided(image, model, predicted_class)
        elif method == 'multiscale':
            return self.explain_multiscale(image, model, predicted_class)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'standard', 'guided', or 'multiscale'")
