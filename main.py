"""
Main training pipeline for MRI brain tumor classification model.

Usage:
    python -m main train --config configs/config.yaml
    python -m main explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path <image_path>
"""

import argparse
import yaml
import logging
import os
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.cnn_model import get_model, load_model
from models.train import train_model
from utils.mri_dataset_loader import MRIDataLoader
from utils.preprocessing import preprocess_image, convert_to_uint8
from utils.visualization import plot_image, overlay_heatmap, plot_predictions
from utils.gradcam_visualization import create_enhanced_gradcam_visualization, save_comparison_visualization
from utils.lime_visualization import create_lime_visualization, create_lime_comparison_visualization
from xai.shap_explainer import SHAPExplainer
from xai.gradcam_enhanced import EnhancedGradCAM
from xai.lime_explainer import LIMEExplainer
from evaluation.metrics import MultiLabelMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict) -> None:
    """Create necessary directories."""
    dirs = [
        config['paths']['models_dir'],
        config['paths']['data_processed_dir'],
        config['paths']['data_annotations_dir'],
        config['paths']['logs_dir'],
        config['paths']['results_dir'],
        config['shap']['save_dir'],
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def train_command(config_path: str) -> None:
    """Train the model."""
    logger.info("=" * 70)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Load config
    config = load_config(config_path)
    setup_directories(config)
    
    # Set random seed
    seed = config['training'].get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = get_model(
        num_classes=config['dataset']['num_classes'],
        dropout_rate=config['model'].get('dropout_rate', 0.5),
        device=device,
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_dir = config['dataset']['path']
    classes = config['dataset'].get('classes', ['glioma', 'meningioma', 'notumor', 'pituitary'])
    
    data_loader = MRIDataLoader(
        dataset_dir=dataset_dir,
        image_size=config['dataset']['image_size'],
        classes=classes,
    )
    
    train_loader, val_loader = data_loader.get_train_val_loaders(
        batch_size=config['training']['batch_size'],
        val_split=config['training']['val_split'],
        num_workers=config['training']['num_workers'],
    )
    
    logger.info(f"Loaded train/val dataloaders")
    
    # Train model
    logger.info("Training model...")
    trainer = train_model(
        config={
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'warmup_epochs': config['training']['warmup_epochs'],
            'num_epochs': config['training']['num_epochs'],
            'models_dir': config['paths']['models_dir'],
        },
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    # Save training history
    history_path = os.path.join(config['paths']['results_dir'], 'training_history.json')
    trainer.save_history(history_path)
    logger.info(f"Saved training history to {history_path}")
    
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best model saved at: {config['paths']['models_dir']}/best_model.pth")
    logger.info("=" * 70)


def explain_command(config_path: str, model_path: str, image_path: str, output_dir: str = None, method: str = 'shap') -> None:
    """Generate explanation for an image using specified method (SHAP or Grad-CAM)."""
    logger.info("=" * 70)
    logger.info(f"STARTING {method.upper()} EXPLANATION PIPELINE")
    logger.info("=" * 70)
    
    # Load config
    config = load_config(config_path)
    setup_directories(config)
    
    # Set output directory based on method
    if output_dir is None:
        if method == 'gradcam':
            output_dir = 'experiments/results/Grad-CAM'
        elif method == 'lime':
            output_dir = 'experiments/results/LIME'
        else:
            output_dir = config['shap']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(
        model_path=model_path,
        num_classes=config['dataset']['num_classes'],
        device=device,
    )
    
    # Load and preprocess image
    logger.info(f"Loading image from {image_path}")
    image_tensor = preprocess_image(
        image_path,
        image_size=config['dataset']['image_size'],
        augment=False,
    )
    
    # Get class names
    class_names = config['dataset'].get('classes', ['glioma', 'meningioma', 'notumor', 'pituitary'])
    
    # ====================================================================
    # GRAD-CAM EXPLANATION (ENHANCED)
    # ====================================================================
    if method == 'gradcam':
        logger.info("Creating Enhanced Grad-CAM explainer...")
        
        # Get predictions first to determine target class
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            predictions = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        predicted_class = predictions.argmax()
        
        logger.info(f"Predicted class: {class_names[predicted_class]} ({predictions[predicted_class]:.4f})")
        
        # Initialize Enhanced Grad-CAM
        explainer = EnhancedGradCAM(device=device)
        
        # Generate explanations with different methods
        logger.info("Generating Guided Grad-CAM (with post-processing)...")
        explanation = explainer.explain(
            image=image_tensor,
            model=model,
            predicted_class=predicted_class,
            method='guided',  # Options: 'standard', 'guided', 'multiscale'
        )
        
        # Save Grad-CAM visualizations
        logger.info("\n" + "=" * 70)
        logger.info("PREDICTIONS")
        logger.info("=" * 70)
        
        for class_name, pred in zip(class_names, predictions):
            marker = " ← PREDICTED" if pred == predictions[predicted_class] else ""
            logger.info(f"{class_name:25s}: {pred:.4f}{marker}")
        
        # Create enhanced visualization with 5+ subplots
        logger.info("Creating enhanced visualization...")
        image_np = image_tensor.squeeze().cpu().numpy()
        
        # Properly denormalize the image for display
        # The image comes normalized with ImageNet stats
        # We need to denormalize: image = (image * std) + mean
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            # Denormalize
            image_np_display = (image_np * imagenet_std + imagenet_mean)
            # Clip to valid range
            image_np_display = np.clip(image_np_display, 0, 1)
            # Convert to (H, W, C) for display
            image_np_display = np.transpose(image_np_display, (1, 2, 0))
        else:
            image_np_display = image_np
        
        heatmap = explanation['heatmap']
        
        gradcam_path = os.path.join(output_dir, f'gradcam_enhanced_{class_names[predicted_class]}.png')
        Path(gradcam_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig = create_enhanced_gradcam_visualization(
            image=image_np_display,
            heatmap=heatmap,
            predictions=predictions,
            class_names=class_names,
            predicted_class_idx=predicted_class,
            method='guided',
            figsize=(18, 10),
        )
        
        fig.savefig(gradcam_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"✓ Saved enhanced Grad-CAM visualization to {gradcam_path}")
        
        # Save predictions chart
        pred_path = os.path.join(output_dir, 'predictions.png')
        plot_predictions(
            predictions=predictions,
            labels=np.zeros_like(predictions),
            disease_classes=class_names,
            top_k=len(class_names),
            save_path=pred_path,
        )
        logger.info(f"✓ Saved predictions plot to {pred_path}")
        
        # Optional: Generate multi-scale comparison
        logger.info("\nGenerating multi-scale Grad-CAM for comparison...")
        explanation_multiscale = explainer.explain(
            image=image_tensor,
            model=model,
            predicted_class=predicted_class,
            method='multiscale',
        )
        
        # Save both methods comparison
        comparison_path = os.path.join(output_dir, f'gradcam_comparison_{class_names[predicted_class]}.png')
        
        # Generate standard for comparison
        explanation_standard = explainer.explain(
            image=image_tensor,
            model=model,
            predicted_class=predicted_class,
            method='standard',
        )
        
        save_comparison_visualization(
            image=image_np_display,
            heatmap_standard=explanation_standard['heatmap'],
            heatmap_guided=explanation['heatmap'],
            heatmap_multiscale=explanation_multiscale['heatmap'],
            predictions=predictions,
            class_names=class_names,
            predicted_class_idx=predicted_class,
            save_path=comparison_path,
            figsize=(20, 12),
        )
        
        logger.info(f"✓ Saved method comparison to {comparison_path}")
    
    # ====================================================================
    # LIME EXPLANATION
    # ====================================================================
    elif method == 'lime':
        logger.info("Creating LIME explainer...")
        
        # Get predictions first to determine target class
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            predictions = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        predicted_class = predictions.argmax()
        
        logger.info(f"Predicted class: {class_names[predicted_class]} ({predictions[predicted_class]:.4f})")
        
        # Initialize LIME explainer
        explainer = LIMEExplainer(device=device, num_samples=150, num_features=50)
        
        # Get image for LIME (denormalized)
        image_np = image_tensor.squeeze().cpu().numpy()
        
        # Denormalize for LIME
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np_display = (image_np * imagenet_std + imagenet_mean)
            image_np_display = np.clip(image_np_display, 0, 1)
            image_np_display = np.transpose(image_np_display, (1, 2, 0))
        else:
            image_np_display = image_np
        
        # Generate LIME explanation
        logger.info("Generating LIME explanation (this may take a minute)...")
        explanation = explainer.explain(
            image=image_np_display,
            model=model,
            target_class=predicted_class,
        )
        
        # Save LIME visualizations
        logger.info("\n" + "=" * 70)
        logger.info("PREDICTIONS")
        logger.info("=" * 70)
        
        for class_name, pred in zip(class_names, predictions):
            marker = " ← PREDICTED" if pred == predictions[predicted_class] else ""
            logger.info(f"{class_name:25s}: {pred:.4f}{marker}")
        
        # Create main LIME visualization
        logger.info("Creating LIME visualization...")
        lime_path = os.path.join(output_dir, f'lime_{class_names[predicted_class]}.png')
        Path(lime_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig = create_lime_visualization(
            image=image_np_display,
            heatmap=explanation['heatmap'],
            segments=explanation['segments'],
            predictions=predictions,
            class_names=class_names,
            predicted_class_idx=predicted_class,
            lime_score=explanation['score'],
            figsize=(18, 10),
        )
        
        fig.savefig(lime_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"✓ Saved LIME visualization to {lime_path}")
        
        # Save detailed comparison visualization
        logger.info("Creating detailed LIME analysis...")
        lime_detail_path = os.path.join(output_dir, f'lime_detailed_{class_names[predicted_class]}.png')
        
        fig = create_lime_comparison_visualization(
            image=image_np_display,
            heatmap_lime=explanation['heatmap'],
            segments=explanation['segments'],
            predictions=predictions,
            class_names=class_names,
            predicted_class_idx=predicted_class,
            lime_score=explanation['score'],
            figsize=(16, 10),
        )
        
        fig.savefig(lime_detail_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"✓ Saved detailed LIME analysis to {lime_detail_path}")
        
        # Save predictions chart
        pred_path = os.path.join(output_dir, 'predictions.png')
        plot_predictions(
            predictions=predictions,
            labels=np.zeros_like(predictions),
            disease_classes=class_names,
            top_k=len(class_names),
            save_path=pred_path,
        )
        logger.info(f"✓ Saved predictions plot to {pred_path}")
        
        # Log LIME model quality
        logger.info(f"\nLIME Model Quality (R² score): {explanation['score']:.4f}")
        logger.info(f"Number of superpixels: {explanation['num_superpixels']}")
    
    # ====================================================================
    # SHAP EXPLANATION
    # ====================================================================
    else:
        logger.info("Creating SHAP explainer...")
        
        # Load background data for SHAP
        logger.info("Loading background data for SHAP...")
        data_loader = MRIDataLoader(
            dataset_dir=config['dataset']['path'],
            image_size=config['dataset']['image_size'],
        )
        background_loader = data_loader.get_background_loader(
            num_samples=config['shap']['background_size'],
            batch_size=config['validation']['batch_size'],
            num_workers=config['validation']['num_workers'],
        )
        
        # Create SHAP explainer
        logger.info("Creating SHAP explainer...")
        explainer = SHAPExplainer(
            background_loader=background_loader,
            num_samples=config['shap']['background_size'],
            device=device,
            batch_size=config['shap'].get('batch_size', 32),
        )
        
        # Generate explanation
        logger.info("Generating SHAP explanation...")
        explanation = explainer.explain(
            image=image_tensor,
            model=model,
            target_class=None,  # Average over all classes
        )
        
        # Get predictions
        predictions = explanation['predictions']
        
        logger.info("\n" + "=" * 70)
        logger.info("PREDICTIONS")
        logger.info("=" * 70)
        
        for class_name, pred in zip(class_names, predictions):
            logger.info(f"{class_name:25s}: {pred:.4f}")
        
        # Prepare images for visualization
        image_denorm = preprocess_image(image_path, config['dataset']['image_size']).cpu().numpy()
        image_denorm = np.transpose(image_denorm, (1, 2, 0))
        if image_denorm.shape[2] == 3:
            # Keep as is for color
            pass
        else:
            image_denorm = image_denorm.squeeze(2)
        
        image_display = convert_to_uint8(image_denorm if image_denorm.ndim == 3 else np.expand_dims(image_denorm, 2))
        attributions = explanation['attributions']
        
        # Save visualizations
        # 1. Plot predictions
        pred_path = os.path.join(output_dir, 'predictions.png')
        plot_predictions(
            predictions=predictions,
            labels=np.zeros_like(predictions),
            disease_classes=class_names,
            top_k=len(class_names),
            save_path=pred_path,
        )
        logger.info(f"Saved predictions plot to {pred_path}")
        
        # 2. Plot SHAP explanation
        shap_path = os.path.join(output_dir, 'shap_explanation.png')
        explainer.visualize_explanation(
            image=image_display,
            attributions=attributions,
            save_path=shap_path,
            title='SHAP Explanation',
        )
        logger.info(f"Saved SHAP explanation to {shap_path}")
        
        # 3. Save original image
        img_save_path = os.path.join(output_dir, 'original_image.png')
        plot_image(
            image=image_display,
            title='Original Image',
            save_path=img_save_path,
        )
        logger.info(f"Saved original image to {img_save_path}")
    
    logger.info("=" * 70)
    logger.info("EXPLANATION COMPLETED")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='XAI Techniques Analysis for Chest X-ray Classification'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Generate explanations')
    explain_parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    explain_parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    explain_parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to image to explain'
    )
    explain_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    explain_parser.add_argument(
        '--method',
        type=str,
        choices=['shap', 'gradcam', 'lime'],
        default='shap',
        help='Explanation method: shap, gradcam, or lime'
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    if args.command == 'train':
        train_command(args.config)
    elif args.command == 'explain':
        explain_command(
            args.config,
            args.model_path,
            args.image_path,
            args.output_dir,
            args.method,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
