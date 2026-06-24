"""
Enhanced visualization utilities for Grad-CAM explanations.

Provides high-quality visualizations with:
- Professional color schemes
- Better blending
- Region highlighting
- Uncertainty visualization
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from typing import Optional, Tuple


def get_optimal_colormap(heatmap: np.ndarray) -> str:
    """
    Select optimal colormap based on heatmap characteristics.
    
    Args:
        heatmap: Input heatmap
        
    Returns:
        str: Colormap name
    """
    # For medical imaging, jet is generally good, but 'hot' works better for fine details
    return 'hot'


def create_professional_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'hot',
) -> np.ndarray:
    """
    Create professional-quality overlay with better blending.
    
    Args:
        image: Original image (H, W, 3) in [0, 255] or [0, 1]
        heatmap: Heatmap (H, W) in [0, 1]
        alpha: Transparency of overlay (0-1)
        colormap: Colormap name
        
    Returns:
        np.ndarray: Blended image (H, W, 3)
    """
    # Normalize image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert grayscale to RGB if needed
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=2)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Apply colormap to heatmap
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Blend: I_blend = (1-α) * I_original + α * I_heatmap
    blended = (1 - alpha) * image + alpha * heatmap_colored
    
    return np.clip(blended, 0, 1)


def create_attention_mask(
    heatmap: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Create binary attention mask from heatmap.
    
    Args:
        heatmap: Input heatmap (H, W) in [0, 1]
        threshold: Threshold for binary mask
        
    Returns:
        np.ndarray: Binary mask
    """
    return (heatmap > threshold).astype(np.uint8)


def create_uncertainty_visualization(
    heatmap: np.ndarray,
    variance: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create visualization showing uncertainty regions.
    
    Args:
        heatmap: Primary heatmap
        variance: Optional variance map
        
    Returns:
        np.ndarray: Uncertainty-enhanced heatmap
    """
    if variance is None:
        # Use heatmap gradient as proxy for uncertainty
        gx = cv2.Sobel(heatmap, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(heatmap, cv2.CV_32F, 0, 1, ksize=3)
        variance = np.sqrt(gx**2 + gy**2)
        variance = variance / (variance.max() + 1e-8)
    
    # Combine: higher uncertainty reduces confidence
    enhanced = heatmap * (1 - 0.3 * variance)
    
    return enhanced


def visualize_multiscale_gradcam(
    image: np.ndarray,
    heatmaps: dict,  # {'layer2': hm2, 'layer3': hm3, 'layer4': hm4}
    predictions: dict,
    predicted_class: str,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """
    Visualize multi-scale Grad-CAM from different layers.
    
    Args:
        image: Original image
        heatmaps: Dict of heatmaps from different layers
        predictions: Class predictions
        predicted_class: Name of predicted class
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(heatmaps) + 1, figsize=figsize)
    
    # Original image
    if image.max() > 1.0:
        image = image / 255.0
    
    if image.ndim == 3 and image.shape[2] == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Multi-scale heatmaps
    for idx, (layer_name, heatmap) in enumerate(heatmaps.items(), 1):
        im = axes[idx].imshow(heatmap, cmap='hot')
        axes[idx].set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def create_enhanced_gradcam_visualization(
    image: np.ndarray,
    heatmap: np.ndarray,
    predictions: np.ndarray,
    class_names: list,
    predicted_class_idx: int,
    method: str = 'guided',
    figsize: Tuple[int, int] = (18, 6),
) -> plt.Figure:
    """
    Create enhanced Grad-CAM visualization with 5 subplots.
    
    Args:
        image: Original image (H, W, 3) already denormalized for display
        heatmap: Grad-CAM heatmap (H, W) in [0, 1]
        predictions: Class predictions
        class_names: List of class names
        predicted_class_idx: Index of predicted class
        method: 'guided', 'standard', or 'multiscale'
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Ensure image is in proper format for display
    if image.ndim == 3:
        # Already in (H, W, C) format
        display_image = image
    elif image.ndim == 2:
        # Grayscale, convert to 3 channels for consistency
        display_image = np.stack([image] * 3, axis=-1)
    else:
        # Handle (C, H, W) format
        if image.shape[0] == 3:
            display_image = np.transpose(image, (1, 2, 0))
        else:
            display_image = image
    
    # Ensure image is in [0, 1] range for display
    if display_image.max() > 1.0:
        display_image = display_image / 255.0
    
    display_image = np.clip(display_image, 0, 1)
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(display_image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Raw Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(heatmap, cmap='hot')
    ax2.set_title(f'Grad-CAM Heatmap\n({method})', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Enhanced Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    overlay = create_professional_overlay(display_image, heatmap, alpha=0.4)
    ax3.imshow(overlay)
    ax3.set_title(f'Overlay\n({class_names[predicted_class_idx]})', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Attention Mask
    ax4 = fig.add_subplot(gs[1, 0])
    mask = create_attention_mask(heatmap, threshold=0.3)
    ax4.imshow(mask, cmap='binary')
    ax4.set_title('Attention Mask\n(threshold=0.3)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Uncertainty Map
    ax5 = fig.add_subplot(gs[1, 1])
    uncertainty = create_uncertainty_visualization(heatmap)
    im5 = ax5.imshow(uncertainty, cmap='RdYlGn')
    ax5.set_title('Confidence Map\n(High=confident)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # 6. Predictions
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create prediction text
    pred_text = f"Model Predictions\n{'='*25}\n\n"
    for idx, (class_name, prob) in enumerate(zip(class_names, predictions)):
        marker = "→ PREDICTED" if idx == predicted_class_idx else ""
        pred_text += f"{class_name:15s}: {prob:.1%} {marker}\n"
    
    ax6.text(0.1, 0.5, pred_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    return fig


def save_comparison_visualization(
    image: np.ndarray,
    heatmap_standard: np.ndarray,
    heatmap_guided: np.ndarray,
    heatmap_multiscale: Optional[np.ndarray],
    predictions: np.ndarray,
    class_names: list,
    predicted_class_idx: int,
    save_path: str,
    figsize: Tuple[int, int] = (20, 12),
):
    """
    Create side-by-side comparison of different Grad-CAM methods.
    
    Args:
        image: Original image (H, W, 3) already denormalized for display
        heatmap_standard: Standard Grad-CAM
        heatmap_guided: Guided Grad-CAM
        heatmap_multiscale: Multi-scale Grad-CAM (optional)
        predictions: Class predictions
        class_names: List of class names
        predicted_class_idx: Index of predicted class
        save_path: Path to save figure
        figsize: Figure size
    """
    num_cols = 3 if heatmap_multiscale is not None else 2
    fig, axes = plt.subplots(3, num_cols + 1, figsize=figsize)
    
    # Ensure image is properly formatted for display
    if image.ndim == 3 and image.shape[0] == 3:
        display_image = np.transpose(image, (1, 2, 0))
    else:
        display_image = image
    
    # Ensure [0, 1] range
    if display_image.max() > 1.0:
        display_image = display_image / 255.0
    display_image = np.clip(display_image, 0, 1)
    
    methods = [
        ('Standard Grad-CAM', heatmap_standard),
        ('Guided Grad-CAM', heatmap_guided),
        ('Multi-scale Grad-CAM', heatmap_multiscale),
    ]
    
    # Row 1: Heatmaps
    for col, (method_name, heatmap) in enumerate(methods):
        if heatmap is None:
            continue
        ax = axes[0, col]
        im = ax.imshow(heatmap, cmap='hot')
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 2: Overlays
    for col, (method_name, heatmap) in enumerate(methods):
        if heatmap is None:
            continue
        ax = axes[1, col]
        overlay = create_professional_overlay(display_image, heatmap, alpha=0.35)
        ax.imshow(overlay)
        ax.set_title(f'{method_name} Overlay', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Row 3: Attention masks
    for col, (method_name, heatmap) in enumerate(methods):
        if heatmap is None:
            continue
        ax = axes[2, col]
        mask = create_attention_mask(heatmap, threshold=0.3)
        ax.imshow(mask, cmap='binary')
        ax.set_title(f'{method_name} Mask', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Last column: Original image and predictions
    for row in range(3):
        ax = axes[row, num_cols]
        if row == 0:
            ax.imshow(display_image)
            ax.set_title('Original Image', fontsize=12, fontweight='bold')
        elif row == 1:
            ax.axis('off')
        else:
            ax.axis('off')
            pred_text = f"Predictions\n{'='*20}\n"
            for idx, (class_name, prob) in enumerate(zip(class_names, predictions)):
                marker = " ← PRED" if idx == predicted_class_idx else ""
                pred_text += f"{class_name:12s}: {prob:.1%}{marker}\n"
            ax.text(0.5, 0.5, pred_text, fontsize=10, family='monospace',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
