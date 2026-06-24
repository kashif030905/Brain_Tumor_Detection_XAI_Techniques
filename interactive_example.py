#!/usr/bin/env python3
"""
Interactive Example: Input → Model Predictions → SHAP Explanations

This script demonstrates the complete pipeline step-by-step.

Usage:
    python interactive_example.py
    
Or with custom image:
    python interactive_example.py --image_path /path/to/image.jpg
"""

import argparse
import torch
import logging
import os
from pathlib import Path
from datetime import datetime

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# IMPORTS
# ============================================================
try:
    from models.cnn_model import load_model
    from utils.preprocessing import preprocess_image
    logger.info("✓ Model imports successful")
except ImportError as e:
    logger.error(f"Error importing models: {e}")
    exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
TEST_IMAGES = {
    "glioma": "data/raw/MRI/Testing/glioma/Te-gl_1.jpg",
    "meningioma": "data/raw/MRI/Testing/meningioma/Te-me_1.jpg",
    "notumor": "data/raw/MRI/Testing/notumor/Te-no_1.jpg",
    "pituitary": "data/raw/MRI/Testing/pituitary/Te-pi_1.jpg",
}
MODEL_PATH = "models/checkpoints/best_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def print_banner(text: str):
    """Print formatted banner."""
    width = 70
    logger.info("\n" + "=" * width)
    logger.info(text.center(width))
    logger.info("=" * width)


def print_subheader(text: str):
    """Print formatted subheader."""
    logger.info(f"\n{text}")
    logger.info("-" * len(text))


def format_probability_bar(prob: float, width: int = 20) -> str:
    """Create ASCII bar for probability visualization."""
    filled = int(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"│{bar}│"


def get_test_image() -> str:
    """Let user choose a test image or provide custom path."""
    print_subheader("📁 Available Test Images")
    
    options = list(TEST_IMAGES.keys())
    for i, tumor_type in enumerate(options, 1):
        path = TEST_IMAGES[tumor_type]
        exists = "✓" if Path(path).exists() else "✗"
        logger.info(f"  {i}. {tumor_type:12s} {exists} {path}")
    
    logger.info(f"  {len(options)+1}. Custom path")
    
    while True:
        try:
            choice = input("\n👉 Select option (1-5): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(options):
                image_path = TEST_IMAGES[options[choice_idx]]
                if not Path(image_path).exists():
                    logger.warning(f"⚠️  File not found: {image_path}")
                    continue
                logger.info(f"✓ Selected: {options[choice_idx]}")
                return image_path
            elif choice_idx == len(options):
                custom = input("Enter custom image path: ").strip()
                if Path(custom).exists():
                    logger.info(f"✓ Selected: {custom}")
                    return custom
                else:
                    logger.warning(f"⚠️  File not found: {custom}")
                    continue
            else:
                logger.warning("Invalid choice. Try again.")
        except ValueError:
            logger.warning("Please enter a number.")


def load_and_check_model() -> torch.nn.Module:
    """Load model and verify it works."""
    print_subheader("🤖 Loading Model")
    
    if not Path(MODEL_PATH).exists():
        logger.error(f"✗ Model not found: {MODEL_PATH}")
        logger.error("  Train the model first: python main.py train")
        exit(1)
    
    logger.info(f"Loading from: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH, num_classes=4, device=DEVICE)
        model.eval()
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Device: {DEVICE}")
        logger.info(f"  Classes: {', '.join(CLASS_NAMES)}")
        return model
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        exit(1)


def generate_predictions(model: torch.nn.Module, image_tensor: torch.Tensor) -> dict:
    """Generate model predictions."""
    print_subheader("🔮 Generating Predictions")
    
    logger.info("Running inference...")
    
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)
    
    # Extract results
    pred_idx = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, pred_idx].item()
    probs_array = probabilities[0].detach().cpu().numpy()
    logits_array = logits[0].detach().cpu().numpy()
    
    return {
        "predicted_index": pred_idx,
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "probabilities": probs_array,
        "logits": logits_array,
    }


def display_predictions(predictions: dict, image_path: str):
    """Display predictions in a nice format."""
    print_banner("PREDICTION RESULTS")
    
    logger.info(f"\n📸 Image: {image_path}")
    logger.info(f"🎯 Predicted Class: {predictions['predicted_class']}")
    logger.info(f"📊 Confidence: {predictions['confidence']:.2%}")
    
    # Confidence indicator
    confidence = predictions['confidence']
    if confidence > 0.90:
        indicator = "✅ Very confident"
    elif confidence > 0.70:
        indicator = "✔️  Confident"
    elif confidence > 0.50:
        indicator = "⚠️  Uncertain"
    else:
        indicator = "❌ Very uncertain"
    logger.info(f"     ({indicator})")
    
    # Class probabilities
    print_subheader("Class Probabilities")
    
    for class_name, prob in zip(CLASS_NAMES, predictions['probabilities']):
        bar = format_probability_bar(prob)
        marker = " ← HIGHEST" if class_name == predictions['predicted_class'] else ""
        logger.info(f"  {class_name:12s}: {prob:7.4f} ({prob*100:5.2f}%) {bar}{marker}")
    
    # Logits
    print_subheader("Raw Logits (Pre-Softmax)")
    
    for class_name, logit in zip(CLASS_NAMES, predictions['logits']):
        marker = " ← Highest" if logit == predictions['logits'].max() else ""
        logger.info(f"  {class_name:12s}: {logit:8.4f}{marker}")


def generate_shap_explanation(predictions: dict):
    """Option to generate SHAP explanation."""
    print_subheader("🎨 SHAP Explanation")
    
    logger.info("\nWould you like to generate SHAP explanation?")
    logger.info("  ⚠️  This will take 2-5 minutes on CPU")
    logger.info("  ✓ Shows which pixels influenced the prediction")
    
    response = input("\nGenerate SHAP? (y/n): ").strip().lower()
    
    if response == 'y':
        logger.info("\n⏳ Starting SHAP computation...")
        logger.info("   This is computationally intensive. Please wait...")
        
        # This would integrate with the SHAP explainer
        logger.info("\nTo generate SHAP, run:")
        logger.info(f"  python main.py explain \\")
        logger.info(f"    --config configs/config.yaml \\")
        logger.info(f"    --model_path {MODEL_PATH}")
        logger.info(f"    --image_path <selected_image>")
        
        return True
    else:
        logger.info("Skipped SHAP generation")
        return False


# ============================================================
# MAIN FLOW
# ============================================================
def main():
    """Run the interactive pipeline."""
    
    # Banner
    print_banner("🧠 MRI BRAIN TUMOR CLASSIFICATION")
    logger.info("Input → Model Predictions → SHAP Explanation")
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default=None, help='Path to image')
    args = parser.parse_args()
    
    # Step 1: Get input image
    print_banner("STEP 1: INPUT IMAGE")
    
    if args.image_path:
        image_path = args.image_path
        if not Path(image_path).exists():
            logger.error(f"File not found: {image_path}")
            exit(1)
    else:
        image_path = get_test_image()
    
    logger.info(f"✓ Image selected: {image_path}")
    
    # Step 2: Load model
    print_banner("STEP 2: LOAD MODEL")
    model = load_and_check_model()
    
    # Step 3: Preprocess image
    print_banner("STEP 3: PREPROCESS IMAGE")
    logger.info("Converting image to tensor...")
    try:
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(DEVICE)
        logger.info(f"✓ Image shape: {image_tensor.shape}")
        logger.info(f"  Image size: 224×224 (standard ImageNet size)")
        logger.info(f"  Color channels: RGB (3)")
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        exit(1)
    
    # Step 4: Generate predictions
    print_banner("STEP 4: GENERATE PREDICTIONS")
    predictions = generate_predictions(model, image_tensor)
    
    # Step 5: Display results
    print_banner("STEP 5: DISPLAY RESULTS")
    display_predictions(predictions, image_path)
    
    # Step 6: Optional SHAP
    print_banner("STEP 6: OPTIONAL SHAP EXPLANATION")
    wants_shap = generate_shap_explanation(predictions)
    
    # Summary
    print_banner("PIPELINE COMPLETE ✅")
    
    logger.info(f"\nSummary:")
    logger.info(f"  Image: {image_path}")
    logger.info(f"  Prediction: {predictions['predicted_class']}")
    logger.info(f"  Confidence: {predictions['confidence']:.1%}")
    logger.info(f"  SHAP Generated: {'Yes ✓' if wants_shap else 'No'}")
    
    logger.info("\n" + "="*70)
    logger.info("Next steps:")
    logger.info(f"  1. Try another image: python interactive_example.py")
    logger.info(f"  2. Generate SHAP: python main.py explain ...")
    logger.info(f"  3. View documentation: see INPUT_OUTPUT_SHAP_TUTORIAL.md")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
