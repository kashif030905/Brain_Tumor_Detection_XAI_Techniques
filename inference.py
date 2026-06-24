"""
Simple inference script to get predictions from the trained model on test images.

Usage:
    python inference.py --model_path models/checkpoints/best_model.pth --image_path <image_path>
    python inference.py --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/glioma/Te-gl_1.jpg
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os
import logging
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.cnn_model import load_model
from utils.preprocessing import get_val_transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_preprocess_image(image_path: str, image_size: int = 224):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to image file
        image_size: Size to resize image to
        
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, H, W)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply preprocessing transforms
    transform = get_val_transforms(image_size)
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def predict(model_path: str, image_path: str, num_classes: int = 4, class_names: list = None):
    """
    Generate predictions for a single image.
    
    Args:
        model_path: Path to trained model checkpoint
        image_path: Path to input image
        num_classes: Number of output classes
        class_names: Names of classes
        
    Returns:
        dict: Predictions and model output
    """
    if class_names is None:
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    logger.info("=" * 70)
    logger.info("INFERENCE - MODEL PREDICTION")
    logger.info("=" * 70)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = load_model(
        model_path=model_path,
        num_classes=num_classes,
        device=device,
    )
    model.eval()
    logger.info("Model loaded successfully")
    
    # Load and preprocess image
    logger.info(f"Loading image from: {image_path}")
    image_tensor = load_and_preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    logger.info(f"Image preprocessed: shape {image_tensor.shape}")
    
    # Generate predictions
    logger.info("Generating predictions...")
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
    
    # Extract results
    logits_np = logits.cpu().numpy()[0]
    probs_np = probabilities.cpu().numpy()[0]
    predicted_class_idx = np.argmax(probs_np)
    predicted_class_name = class_names[predicted_class_idx]
    predicted_confidence = probs_np[predicted_class_idx]
    
    # Log results
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 70)
    
    logger.info(f"\nPredicted Class: {predicted_class_name} (index: {predicted_class_idx})")
    logger.info(f"Confidence: {predicted_confidence:.4f} ({predicted_confidence*100:.2f}%)")
    
    logger.info("\nClass Probabilities:")
    logger.info("-" * 70)
    for i, (class_name, prob) in enumerate(zip(class_names, probs_np)):
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        logger.info(f"  {class_name:15s}: {prob:.4f} ({prob*100:6.2f}%) │{bar}│")
    
    logger.info("\nRaw Logits:")
    logger.info("-" * 70)
    for class_name, logit in zip(class_names, logits_np):
        logger.info(f"  {class_name:15s}: {logit:.4f}")
    
    logger.info("=" * 70 + "\n")
    
    # Return results
    return {
        'predicted_class': predicted_class_name,
        'predicted_class_idx': int(predicted_class_idx),
        'confidence': float(predicted_confidence),
        'probabilities': {class_name: float(prob) for class_name, prob in zip(class_names, probs_np)},
        'logits': {class_name: float(logit) for class_name, logit in zip(class_names, logits_np)},
    }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Run inference on a test image using trained model'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=4,
        help='Number of output classes (default: 4)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=['glioma', 'meningioma', 'notumor', 'pituitary'],
        help='Class names (default: glioma meningioma notumor pituitary)'
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.image_path):
        logger.error(f"Image not found: {args.image_path}")
        sys.exit(1)
    
    # Run prediction
    try:
        results = predict(
            model_path=args.model_path,
            image_path=args.image_path,
            num_classes=args.num_classes,
            class_names=args.classes,
        )
        
        print("\n" + "=" * 70)
        print("FINAL PREDICTION")
        print("=" * 70)
        print(f"Class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']*100:.2f}%")
        print("=" * 70 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
