"""
Visualization utilities for medical images and XAI explanations.

Provides functions for plotting images, overlaying heatmaps,
and visualizing model predictions and explanations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
from typing import Optional, Tuple, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_image(
    image: np.ndarray,
    title: str = "",
    figsize: Tuple[int, int] = (6, 6),
    cmap: str = "gray",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single medical image.
    
    Args:
        image: Image array (H, W) for grayscale or (H, W, 3) for RGB.
        title: Title for the plot.
        figsize: Figure size as (width, height).
        cmap: Colormap to use for grayscale images.
        save_path: If provided, save figure to this path.
        
    Returns:
        plt.Figure: Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(image.shape) == 2:
        # Grayscale
        ax.imshow(image, cmap=cmap)
    else:
        # RGB
        ax.imshow(image)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay a heatmap on top of an image.
    
    Args:
        image: Image array (H, W) for grayscale or (H, W, 3) for RGB.
        heatmap: Heatmap array (H, W) with values typically in [-1, 1] or [0, 1].
        alpha: Transparency of the heatmap overlay (0-1).
        cmap: Colormap for the heatmap.
        title: Title for the plot.
        figsize: Figure size as (width, height).
        save_path: If provided, save figure to this path.
        
    Returns:
        plt.Figure: Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))
    
    # Original image
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap="gray")
    else:
        axes[0].imshow(image)
    axes[0].set_title("Original Image", fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im1 = axes[1].imshow(heatmap, cmap=cmap)
    axes[1].set_title("Heatmap", fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    if len(image.shape) == 2:
        axes[2].imshow(image, cmap="gray")
    else:
        axes[2].imshow(image)
    
    # Normalize heatmap for overlay
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    im2 = axes[2].imshow(heatmap_normalized, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay", fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def plot_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    disease_classes: List[str],
    top_k: int = 5,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot model predictions as a bar chart.
    
    Args:
        predictions: Prediction probabilities (num_classes,).
        labels: Ground truth labels (num_classes,).
        disease_classes: List of disease class names.
        top_k: Number of top predictions to show.
        figsize: Figure size as (width, height).
        save_path: If provided, save figure to this path.
        
    Returns:
        plt.Figure: Figure object.
    """
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_predictions = predictions[top_indices]
    top_labels = [disease_classes[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if labels[i] == 1 else 'orange' for i in top_indices]
    bars = ax.barh(top_labels, top_predictions, color=colors, alpha=0.7)
    
    ax.set_xlabel('Probability', fontsize=11, fontweight='bold')
    ax.set_title('Top-k Predictions', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_predictions)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    disease_classes: List[str],
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix for multi-label classification.
    
    Args:
        cm: Confusion matrix (num_classes, 2, 2) for binary classification per class.
        disease_classes: List of disease class names.
        figsize: Figure size as (width, height).
        save_path: If provided, save figure to this path.
        
    Returns:
        plt.Figure: Figure object.
    """
    num_classes = len(disease_classes)
    ncols = 4
    nrows = (num_classes + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, disease_name in enumerate(disease_classes):
        ax = axes[idx]
        
        if idx < len(cm):
            matrix = cm[idx]
            im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(int(matrix[i, j])),
                           ha='center', va='center', color='white' if matrix[i, j] > matrix.max() / 2 else 'black',
                           fontsize=12, fontweight='bold')
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            ax.set_title(disease_name, fontsize=10, fontweight='bold')
        else:
            ax.axis('off')
    
    # Remove extra subplots
    for idx in range(len(disease_classes), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def plot_metrics(
    metrics_history: dict,
    metric_names: List[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training metrics over epochs.
    
    Args:
        metrics_history: Dictionary with metric histories {metric_name: [values]}.
        metric_names: List of metric names to plot. If None, plot all.
        figsize: Figure size as (width, height).
        save_path: If provided, save figure to this path.
        
    Returns:
        plt.Figure: Figure object.
    """
    if metric_names is None:
        metric_names = list(metrics_history.keys())
    
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(1, num_metrics, figsize=(figsize[0], figsize[1]))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric_name in zip(axes, metric_names):
        if metric_name in metrics_history:
            values = metrics_history[metric_name]
            ax.plot(values, marker='o', linewidth=2)
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(metric_name, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved figure to {save_path}")
    
    plt.close(fig)
    return fig


def tensor_to_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> np.ndarray:
    """
    Convert a PyTorch tensor to numpy image array.
    
    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W).
        denormalize: Whether to denormalize using ImageNet stats.
        
    Returns:
        np.ndarray: Image array.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    tensor = tensor.cpu().detach()
    
    if denormalize:
        from .preprocessing import denormalize_image
        tensor = denormalize_image(tensor)
    
    # Convert to numpy and transpose
    image = tensor.numpy()
    if image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    
    # Convert to uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Handle single channel
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(2)
    
    return image
