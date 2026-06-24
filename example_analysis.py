"""
Example analysis notebook showing how to use the XAI project.

This demonstrates:
1. Loading a trained model
2. Performing inference
3. Generating SHAP explanations
4. Computing evaluation metrics
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# For Jupyter notebook compatibility
try:
    from IPython.display import Image, display
except ImportError:
    pass

from models import load_model
from utils.dataset_loader import ChestXrayDataLoader, get_disease_classes
from utils.preprocessing import preprocess_image
from xai.shap_explainer import SHAPExplainer
from evaluation.metrics import MultiLabelMetrics, get_prediction_confidence
from utils.visualization import plot_predictions, overlay_heatmap
import yaml


def load_config(config_path='configs/config.yaml'):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device():
    """Setup GPU device."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_trained_model(model_path, config, device):
    """Load trained model."""
    print(f"\nLoading model from {model_path}...")
    model = load_model(
        model_path=model_path,
        num_classes=config['dataset']['num_classes'],
        device=device,
    )
    print("✓ Model loaded")
    return model


def prepare_background_data(config, device):
    """Prepare background data for SHAP."""
    print("\nPreparing background data...")
    
    dataset_dir = config['dataset']['path']
    csv_path = os.path.join(dataset_dir, config['dataset']['data_entry_csv'])
    train_val_list_path = os.path.join(dataset_dir, config['dataset']['train_val_list'])
    
    data_loader = ChestXrayDataLoader(
        dataset_dir=dataset_dir,
        csv_path=csv_path,
        image_size=config['dataset']['image_size'],
    )
    
    background_loader = data_loader.get_background_loader(
        train_val_list_path=train_val_list_path,
        num_samples=config['shap']['background_size'],
        batch_size=16,
    )
    
    print("✓ Background data prepared")
    return background_loader


def predict_single_image(model, image_path, config, device):
    """Make prediction on a single image."""
    print(f"\nPredicting on: {image_path}")
    
    # Load and preprocess
    image = preprocess_image(image_path, config['dataset']['image_size'])
    image = image.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    
    disease_classes = get_disease_classes()
    
    print("\nPredictions:")
    print("-" * 50)
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices[:5]:
        print(f"{disease_classes[idx]:25s} {probs[idx]:.4f}")
    
    return probs, image.cpu()


def explain_with_shap(model, image, background_loader, config, device):
    """Generate SHAP explanation."""
    print("\nGenerating SHAP explanation...")
    
    explainer = SHAPExplainer(
        background_loader=background_loader,
        num_samples=config['shap']['background_size'],
        device=device,
    )
    
    explanation = explainer.explain(image, model, target_class=None)
    
    print("✓ SHAP explanation generated")
    
    return explanation, explainer


def analyze_predictions(probs, config):
    """Analyze prediction statistics."""
    print("\nPrediction Analysis:")
    print("-" * 50)
    
    confidence = get_prediction_confidence(probs.reshape(1, -1))[0]
    
    print(f"Max confidence: {confidence:.4f}")
    print(f"Min confidence: {probs.min():.4f}")
    print(f"Mean confidence: {probs.mean():.4f}")
    print(f"Std confidence: {probs.std():.4f}")
    
    # Count positive predictions
    threshold = 0.5
    num_positive = (probs >= threshold).sum()
    print(f"\nPositive predictions (threshold={threshold}): {num_positive}")


def main():
    """Main example workflow."""
    print("=" * 70)
    print("XAI PROJECT - EXAMPLE ANALYSIS")
    print("=" * 70)
    
    # Setup
    config = load_config()
    device = setup_device()
    
    # Load model
    model_path = 'models/checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print(f"\n✗ Model not found at {model_path}")
        print("Please train the model first: python main.py train")
        return
    
    model = load_trained_model(model_path, config, device)
    
    # Prepare background data
    background_loader = prepare_background_data(config, device)
    
    # Find a sample image
    dataset_dir = config['dataset']['path']
    sample_image = None
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        folder_path = os.path.join(dataset_dir, folder, "images")
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
            if images:
                sample_image = os.path.join(folder_path, images[0])
                break
    
    if sample_image is None:
        print("\n✗ No sample images found")
        return
    
    # Predict
    probs, image_tensor = predict_single_image(model, sample_image, config, device)
    
    # Analyze
    analyze_predictions(probs, config)
    
    # Explain
    explanation, explainer = explain_with_shap(model, image_tensor, background_loader, config, device)
    
    # Visualize
    print("\nVisualizing results...")
    disease_classes = get_disease_classes()
    
    # Create output directory
    output_dir = 'experiments/results/example'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot predictions
    plot_predictions(
        predictions=probs,
        labels=np.zeros_like(probs),
        disease_classes=disease_classes,
        top_k=14,
        save_path=os.path.join(output_dir, 'predictions.png'),
    )
    print(f"✓ Saved predictions to {output_dir}/predictions.png")
    
    # Plot SHAP explanation
    from utils.preprocessing import convert_to_uint8
    image_display = convert_to_uint8(image_tensor[0].cpu().numpy())
    
    explainer.visualize_explanation(
        image=image_display,
        attributions=explanation['attributions'],
        save_path=os.path.join(output_dir, 'shap_explanation.png'),
        title='SHAP Explanation',
    )
    print(f"✓ Saved SHAP explanation to {output_dir}/shap_explanation.png")
    
    print("\n" + "=" * 70)
    print("✓ EXAMPLE ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
