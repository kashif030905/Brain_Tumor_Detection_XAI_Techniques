"""
LIME (Local Interpretable Model-agnostic Explanations) explainer for medical images.

LIME is a model-agnostic explanation technique that explains individual predictions
by approximating the model locally with an interpretable model.

Key advantages:
- Model-agnostic (works with any model)
- Local explanations (focuses on individual predictions)
- Interpretable (uses simple linear models)
- Different from Grad-CAM (captures different aspects)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Callable
import cv2
from skimage.segmentation import felzenszwalb, mark_boundaries
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for image classification.
    
    LIME explains a model's prediction by:
    1. Perturbing the input (turning superpixels on/off)
    2. Getting predictions for perturbed inputs
    3. Fitting a linear model to approximate local behavior
    4. Using the linear model weights as explanations
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        num_samples: int = 150,
        num_features: int = 50,
    ):
        """
        Initialize LIME explainer.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            num_samples: Number of perturbed samples to generate
            num_features: Number of superpixels for segmentation
        """
        self.device = device
        self.num_samples = num_samples
        self.num_features = num_features
        
        logger.info(
            f"Initialized LIME explainer (num_samples={num_samples}, "
            f"num_features={num_features}, device={device})"
        )
    
    def _get_image_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Segment image into superpixels using Felzenszwalb algorithm.
        
        Args:
            image: Image array (H, W, 3) in [0, 1]
            
        Returns:
            Tuple of (segmentation_mask, num_segments)
        """
        # Felzenszwalb requires image in [0, 1]
        if image.max() > 1.0:
            image_norm = image / 255.0
        else:
            image_norm = image
        
        # Apply Felzenszwalb segmentation
        segments = felzenszwalb(
            image_norm,
            scale=100,
            sigma=0.5,
            min_size=50
        )
        
        return segments, segments.max() + 1
    
    def _generate_perturbed_samples(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        num_segments: int,
    ) -> np.ndarray:
        """
        Generate perturbed samples by randomly turning superpixels on/off.
        
        Args:
            image: Original image (H, W, 3)
            segments: Segmentation mask
            num_segments: Number of segments
            
        Returns:
            Perturbation matrix (num_samples, num_segments) with binary values
        """
        # Generate random perturbations
        perturbed = np.random.binomial(1, 0.5, (self.num_samples, num_segments))
        
        # Ensure first sample is the original (all ones)
        perturbed[0, :] = 1
        
        return perturbed
    
    def _perturbed_image_to_tensor(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        perturbation: np.ndarray,
    ) -> torch.Tensor:
        """
        Create perturbed image from perturbation mask.
        
        Args:
            image: Original image (H, W, 3)
            segments: Segmentation mask
            perturbation: Binary mask for segments (num_segments,)
            
        Returns:
            Tensor of perturbed image
        """
        # Create perturbed image by masking out superpixels
        perturbed_image = image.copy()
        
        for segment_id in range(segments.max() + 1):
            if perturbation[segment_id] == 0:
                # Turn off this segment (set to mean color)
                mask = segments == segment_id
                perturbed_image[mask] = image[mask].mean(axis=0, keepdims=True)
        
        return perturbed_image
    
    def _preprocess_for_model(
        self,
        image: np.ndarray,
        image_size: int = 224,
    ) -> torch.Tensor:
        """
        Preprocess image for model inference (ImageNet normalization).
        
        Args:
            image: Image array (H, W, 3) in [0, 1]
            image_size: Target image size
            
        Returns:
            Preprocessed tensor
        """
        # Resize if needed
        if image.shape[:2] != (image_size, image_size):
            image = cv2.resize(image, (image_size, image_size))
        
        # Normalize with ImageNet stats
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        image_normalized = (image - imagenet_mean) / imagenet_std
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()
        
        return tensor.unsqueeze(0).to(self.device)
    
    def _get_model_prediction(
        self,
        model: nn.Module,
        perturbed_images: np.ndarray,
    ) -> np.ndarray:
        """
        Get model predictions for perturbed images.
        
        Args:
            model: Neural network model
            perturbed_images: Array of perturbed images (num_samples, H, W, 3)
            
        Returns:
            Predictions for target class (num_samples,)
        """
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for image in perturbed_images:
                tensor = self._preprocess_for_model(image)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy()[0])
        
        return np.array(predictions)
    
    def _fit_linear_model(
        self,
        perturbations: np.ndarray,
        predictions: np.ndarray,
        target_class: int,
        distances: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Fit linear model to explain predictions.
        
        Uses weighted Ridge regression where weights are based on
        distance from original image (closer = higher weight).
        
        Args:
            perturbations: Binary perturbations (num_samples, num_features)
            predictions: Model predictions (num_samples, num_classes)
            target_class: Class to explain
            distances: Distance from original for each sample
            
        Returns:
            Tuple of (feature_weights, model_score)
        """
        # Get predictions for target class
        y = predictions[:, target_class]
        
        # Compute weights: closer samples get higher weight
        weights = np.exp(-distances)
        
        # Fit Ridge regression
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(perturbations, y, sample_weight=weights)
        
        # Get feature weights
        feature_weights = model.coef_
        
        # Compute R² score
        score = model.score(perturbations, y, sample_weight=weights)
        
        return feature_weights, score
    
    def _weights_to_heatmap(
        self,
        feature_weights: np.ndarray,
        segments: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert feature importance weights to spatial heatmap.
        
        Args:
            feature_weights: Weight for each superpixel
            segments: Segmentation mask
            image: Original image
            
        Returns:
            Heatmap (H, W) with importance scores
        """
        heatmap = np.zeros(segments.shape, dtype=np.float32)
        
        # Map superpixel weights to heatmap
        for segment_id, weight in enumerate(feature_weights):
            mask = segments == segment_id
            # Use ReLU to focus on positive contributions
            heatmap[mask] = max(weight, 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def explain(
        self,
        image: np.ndarray,
        model: nn.Module,
        target_class: int,
        image_size: int = 224,
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for an image.
        
        Args:
            image: Input image (H, W, 3) in [0, 1] or [0, 255]
            model: Neural network model
            target_class: Class to explain
            image_size: Target image size for model
            
        Returns:
            dict: Explanation with keys:
                - 'heatmap': Spatial heatmap (H, W)
                - 'weights': Feature importance weights
                - 'segments': Superpixel segments
                - 'score': Linear model R² score
                - 'method': 'LIME'
        """
        logger.info(f"Generating LIME explanation for class {target_class}...")
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image_norm = image / 255.0
        else:
            image_norm = image.astype(np.float32)
        
        # Get image segmentation
        logger.info(f"Segmenting image into superpixels...")
        segments, num_segments = self._get_image_segmentation(image_norm)
        
        # Generate perturbations
        logger.info(f"Generating {self.num_samples} perturbed samples...")
        perturbations = self._generate_perturbed_samples(image_norm, segments, num_segments)
        
        # Create perturbed images
        logger.info(f"Creating perturbed images...")
        perturbed_images = []
        for perturbation_mask in perturbations:
            perturbed_img = image_norm.copy()
            for seg_id in range(num_segments):
                if perturbation_mask[seg_id] == 0:
                    mask = segments == seg_id
                    perturbed_img[mask] = image_norm[mask].mean(axis=0, keepdims=True)
            perturbed_images.append(perturbed_img)
        perturbed_images = np.array(perturbed_images)
        
        # Get predictions
        logger.info(f"Getting model predictions for perturbed samples...")
        predictions = self._get_model_prediction(model, perturbed_images)
        
        # Compute distances (for weighting)
        distances = np.array([
            cosine(pert, perturbations[0])
            for pert in perturbations
        ])
        
        # Fit linear model
        logger.info(f"Fitting linear model...")
        feature_weights, score = self._fit_linear_model(
            perturbations,
            predictions,
            target_class,
            distances,
        )
        
        logger.info(f"LIME model R² score: {score:.4f}")
        
        # Convert to heatmap
        heatmap = self._weights_to_heatmap(feature_weights, segments, image_norm)
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (image_norm.shape[1], image_norm.shape[0]))
        
        return {
            'heatmap': heatmap_resized,
            'weights': feature_weights,
            'segments': segments,
            'score': score,
            'method': 'LIME',
            'num_superpixels': num_segments,
        }
    
    def explain_batch(
        self,
        images: torch.Tensor,
        model: nn.Module,
        target_class: int = None,
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for a batch of images.
        
        Args:
            images: Batch of input images
            model: Neural network model
            target_class: Target class for explanation
            
        Returns:
            dict: Batch explanations
        """
        batch_size = images.shape[0]
        explanations = []
        
        for i in range(batch_size):
            image = images[i].cpu().numpy()
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            exp = self.explain(
                image=image,
                model=model,
                target_class=target_class,
            )
            explanations.append(exp)
        
        return {
            'explanations': explanations,
            'method': 'LIME',
            'batch_size': batch_size,
        }


def create_lime_explainer(
    device: str = 'cpu',
    num_samples: int = 150,
    num_features: int = 50,
    **kwargs
) -> LIMEExplainer:
    """Factory method for LIME explainer."""
    return LIMEExplainer(
        device=device,
        num_samples=num_samples,
        num_features=num_features,
    )
