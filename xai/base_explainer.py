"""
Base class for XAI explainers.

Defines the interface that all explainers must implement.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Explainer(ABC):
    """
    Abstract base class for XAI explainers.
    
    All explainer implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize explainer.
        
        Args:
            name: Name of the explainer (e.g., 'SHAP', 'Grad-CAM', 'LIME').
        """
        self.name = name
        logger.info(f"Initialized {name} explainer")
    
    @abstractmethod
    def explain(
        self,
        image: torch.Tensor,
        model: nn.Module,
        target_class: int = None,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single image.
        
        Args:
            image: Input image tensor of shape (1, 3, H, W) or (3, H, W).
            model: Neural network model to explain.
            target_class: If specified, focus explanation on this class.
                         For multi-label, can be a list of indices.
                         If None, explain all classes.
        
        Returns:
            dict: Explanation containing:
                - 'attributions': Heatmap of shape (H, W) or (num_classes, H, W)
                - 'predictions': Model predictions
                - 'method': Name of the explainer
                - Additional method-specific keys
        """
        pass
    
    @abstractmethod
    def explain_batch(
        self,
        images: torch.Tensor,
        model: nn.Module,
        target_class: int = None,
    ) -> Dict[str, Any]:
        """
        Generate explanations for a batch of images.
        
        Args:
            images: Batch of input tensors of shape (B, 3, H, W).
            model: Neural network model to explain.
            target_class: Target class for explanation.
        
        Returns:
            dict: Batch explanations.
        """
        pass
    
    @staticmethod
    def _ensure_batch_dim(tensor: torch.Tensor, has_batch: bool = False) -> torch.Tensor:
        """
        Ensure tensor has batch dimension.
        
        Args:
            tensor: Input tensor.
            has_batch: Whether tensor already has batch dimension.
            
        Returns:
            torch.Tensor: Tensor with batch dimension.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor
    
    @staticmethod
    def _remove_batch_dim(tensor: torch.Tensor, original_shape: int) -> torch.Tensor:
        """
        Remove batch dimension if original tensor didn't have it.
        
        Args:
            tensor: Tensor with batch dimension.
            original_shape: Original number of dimensions.
            
        Returns:
            torch.Tensor: Tensor without batch dimension (if applicable).
        """
        if original_shape == 3:
            tensor = tensor.squeeze(0)
        return tensor
    
    def get_info(self) -> str:
        """
        Get information about the explainer.
        
        Returns:
            str: Explainer information.
        """
        return f"{self.name} Explainer"


class ExplainerFactory:
    """Factory for creating explainer instances."""
    
    _explainers = {}
    
    @classmethod
    def register(cls, name: str, explainer_class):
        """
        Register an explainer class.
        
        Args:
            name: Name of the explainer.
            explainer_class: Explainer class (must inherit from Explainer).
        """
        cls._explainers[name.lower()] = explainer_class
        logger.info(f"Registered explainer: {name}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Explainer:
        """
        Create an explainer instance.
        
        Args:
            name: Name of the explainer to create.
            **kwargs: Arguments to pass to the explainer constructor.
            
        Returns:
            Explainer: Instance of the requested explainer.
            
        Raises:
            ValueError: If explainer name is not registered.
        """
        name_lower = name.lower()
        if name_lower not in cls._explainers:
            raise ValueError(
                f"Unknown explainer: {name}. "
                f"Available: {list(cls._explainers.keys())}"
            )
        
        explainer_class = cls._explainers[name_lower]
        return explainer_class(**kwargs)
    
    @classmethod
    def get_available(cls) -> list:
        """Get list of available explainers."""
        return list(cls._explainers.keys())
