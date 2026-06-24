"""
LIME-specific visualization utilities.

Provides high-quality visualizations for LIME explanations including
superpixel overlays and feature importance maps.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from skimage.segmentation import mark_boundaries
from typing import Tuple, Optional


def create_lime_visualization(
    image: np.ndarray,
    heatmap: np.ndarray,
    segments: np.ndarray,
    predictions: np.ndarray,
    class_names: list,
    predicted_class_idx: int,
    lime_score: float,
    figsize: Tuple[int, int] = (18, 6),
) -> plt.Figure:
    """
    Create comprehensive LIME visualization.
    
    Args:
        image: Original image (H, W, 3) in [0, 1]
        heatmap: LIME importance heatmap (H, W) in [0, 1]
        segments: Superpixel segmentation
        predictions: Class predictions
        class_names: List of class names
        predicted_class_idx: Index of predicted class
        lime_score: LIME model R² score
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Superpixel Segmentation
    ax2 = fig.add_subplot(gs[0, 1])
    boundaries = mark_boundaries(image, segments, color=(1, 0, 0), mode='thick')
    ax2.imshow(boundaries)
    ax2.set_title(f'Superpixel Segments\n({segments.max() + 1} superpixels)', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. LIME Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(heatmap, cmap='hot')
    ax3.set_title('LIME Importance Map', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. LIME Overlay on Original
    ax4 = fig.add_subplot(gs[1, 0])
    # Create colored heatmap
    heatmap_colored = plt.cm.hot(heatmap)[:, :, :3]
    overlay = 0.6 * image + 0.4 * heatmap_colored
    ax4.imshow(overlay)
    ax4.set_title('LIME Overlay', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Important Regions (threshold)
    ax5 = fig.add_subplot(gs[1, 1])
    important_mask = (heatmap > 0.3).astype(np.uint8)
    ax5.imshow(important_mask, cmap='binary')
    ax5.set_title('Important Regions\n(threshold=0.3)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. Predictions and LIME Score
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    pred_text = f"Model Predictions\n{'='*25}\n"
    pred_text += f"R² Score: {lime_score:.3f}\n\n"
    for idx, (class_name, prob) in enumerate(zip(class_names, predictions)):
        marker = "→ PREDICTED" if idx == predicted_class_idx else ""
        pred_text += f"{class_name:15s}: {prob:.1%} {marker}\n"
    
    ax6.text(0.1, 0.5, pred_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.3))
    
    return fig


def create_lime_comparison_visualization(
    image: np.ndarray,
    heatmap_lime: np.ndarray,
    segments: np.ndarray,
    predictions: np.ndarray,
    class_names: list,
    predicted_class_idx: int,
    lime_score: float,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Create detailed LIME visualization with multiple perspectives.
    
    Args:
        image: Original image
        heatmap_lime: LIME importance heatmap
        segments: Superpixel segmentation
        predictions: Class predictions
        class_names: List of class names
        predicted_class_idx: Index of predicted class
        lime_score: LIME model R² score
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Row 1
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Superpixel boundaries
    boundaries = mark_boundaries(image, segments, color=(1, 0, 0), mode='thick')
    axes[0, 1].imshow(boundaries)
    axes[0, 1].set_title(f'Superpixels ({segments.max() + 1})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # LIME heatmap
    im = axes[0, 2].imshow(heatmap_lime, cmap='hot')
    axes[0, 2].set_title('LIME Importance', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2
    # Overlay 1: Soft overlay
    heatmap_colored = plt.cm.hot(heatmap_lime)[:, :, :3]
    overlay_soft = 0.5 * image + 0.5 * heatmap_colored
    axes[1, 0].imshow(overlay_soft)
    axes[1, 0].set_title('Soft Overlay (50-50)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay 2: Hard overlay
    overlay_hard = 0.7 * image + 0.3 * heatmap_colored
    axes[1, 1].imshow(overlay_hard)
    axes[1, 1].set_title('Hard Overlay (70-30)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Important regions mask with boundaries
    axes[1, 2].axis('off')
    
    # Create detailed info
    info_text = f"LIME Analysis\n{'='*30}\n\n"
    info_text += f"Model R² Score: {lime_score:.4f}\n"
    info_text += f"Predicted: {class_names[predicted_class_idx]}\n"
    info_text += f"Confidence: {predictions[predicted_class_idx]:.1%}\n\n"
    info_text += "Class Probabilities:\n"
    for idx, (class_name, prob) in enumerate(zip(class_names, predictions)):
        marker = " ← PRED" if idx == predicted_class_idx else ""
        info_text += f"{class_name:12s}: {prob:.1%}{marker}\n"
    
    axes[1, 2].text(0.5, 0.5, info_text, fontsize=10, family='monospace',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4))
    
    plt.tight_layout()
    return fig


def visualize_lime_superpixels(
    image: np.ndarray,
    segments: np.ndarray,
    weights: np.ndarray,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Visualize superpixels colored by their importance weights.
    
    Args:
        image: Original image
        segments: Superpixel segmentation
        weights: Importance weight for each superpixel
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original with boundaries
    boundaries = mark_boundaries(image, segments, color=(1, 0, 0), mode='thick')
    axes[0].imshow(boundaries)
    axes[0].set_title('Superpixel Boundaries', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Positive weights (contributing to prediction)
    positive_weights = np.maximum(weights, 0)
    weight_map = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in enumerate(positive_weights):
        weight_map[segments == seg_id] = weight
    
    if weight_map.max() > 0:
        weight_map = weight_map / weight_map.max()
    
    im1 = axes[1].imshow(weight_map, cmap='Greens')
    axes[1].set_title('Positive Contributions', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Negative weights (against prediction)
    negative_weights = -np.minimum(weights, 0)
    weight_map_neg = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in enumerate(negative_weights):
        weight_map_neg[segments == seg_id] = weight
    
    if weight_map_neg.max() > 0:
        weight_map_neg = weight_map_neg / weight_map_neg.max()
    
    im2 = axes[2].imshow(weight_map_neg, cmap='Reds')
    axes[2].set_title('Negative Contributions', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig
