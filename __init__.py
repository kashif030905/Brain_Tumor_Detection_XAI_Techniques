"""
XAI Techniques Analysis for Chest X-ray Classification

A complete, production-quality Python project for evaluating explainable AI
techniques on the NIH ChestX-ray14 dataset using SHAP and CNNs.

Version: 1.0.0
Author: XAI Project Team
Date: April 9, 2026

Quick Start:
    1. python test_setup.py              # Verify setup
    2. python main.py train              # Train model
    3. python main.py explain ...        # Generate explanations

Documentation:
    - README.md: Full documentation
    - SETUP_GUIDE.md: Setup instructions
    - QUICK_REFERENCE.md: Quick lookup
    - PROJECT_SUMMARY.md: Project overview
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Version
__version__ = "1.0.0"
__author__ = "XAI Project Team"
__all__ = [
    'models',
    'xai',
    'utils',
    'evaluation',
    'PROJECT_ROOT',
    '__version__',
]

# Import main modules
from models import (
    ResNet50ChestXray,
    get_model,
    load_model,
    save_model,
)

from xai import (
    Explainer,
    ExplainerFactory,
    SHAPExplainer,
)

from utils import (
    ChestXrayDataLoader,
    DISEASE_CLASSES,
    preprocess_image,
    overlay_heatmap,
    plot_image,
)

from evaluation import (
    MultiLabelMetrics,
    get_prediction_confidence,
)


def print_welcome():
    """Print welcome message with version info."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   XAI TECHNIQUES ANALYSIS FOR CHEST X-RAY                ║
    ║   Explainable AI with SHAP and CNNs                      ║
    ║                                                           ║
    ║   Version: 1.0.0                                          ║
    ║   Status: Production Ready                               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    📚 Documentation:
        - README.md: Full documentation
        - SETUP_GUIDE.md: Setup instructions
        - QUICK_REFERENCE.md: Quick commands
        - PROJECT_SUMMARY.md: Project overview
    
    🚀 Quick Start:
        1. python test_setup.py              # Verify setup
        2. python main.py train              # Train model
        3. python main.py explain ...        # Generate explanations
    
    📊 Support:
        - 14 disease classes (ChestX-ray14)
        - ResNet50 model
        - SHAP explanations
        - Comprehensive metrics
    """)


if __name__ == "__main__":
    print_welcome()
