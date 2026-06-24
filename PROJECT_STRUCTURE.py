"""
PROJECT STRUCTURE SUMMARY

This document provides a complete overview of the XAI project organization.
"""

PROJECT_STRUCTURE = {
    "XAI/": {
        "description": "Root project directory",
        "subdirs": {
            "data/": {
                "description": "Dataset directory (ignored by git)",
                "note": "Only this directory is ignored, not tracked by git",
                "subdirs": {
                    "raw/NIH_ChestXray/": "Downloaded dataset from NIH",
                    "processed/": "Processed dataset and features",
                    "annotations/": "Additional annotations and metadata",
                }
            },
            "models/": {
                "description": "CNN models and training",
                "files": {
                    "__init__.py": "Package initialization",
                    "cnn_model.py": "ResNet50 architecture",
                    "train.py": "Training pipeline",
                },
                "subdirs": {
                    "checkpoints/": "Saved model weights (.pth files)",
                }
            },
            "xai/": {
                "description": "Explainability modules",
                "files": {
                    "__init__.py": "Package initialization",
                    "base_explainer.py": "Abstract base class for explainers",
                    "shap_explainer.py": "SHAP implementation",
                },
                "note": "Extensible design for LIME, Grad-CAM, etc."
            },
            "utils/": {
                "description": "Utility modules",
                "files": {
                    "__init__.py": "Package initialization",
                    "dataset_loader.py": "NIH dataset loading and preprocessing",
                    "preprocessing.py": "Image preprocessing utilities",
                    "visualization.py": "Visualization tools",
                }
            },
            "evaluation/": {
                "description": "Evaluation metrics",
                "files": {
                    "__init__.py": "Package initialization",
                    "metrics.py": "Classification metrics (AUROC, AUPRC, F1, etc.)",
                }
            },
            "experiments/": {
                "description": "Experiment outputs",
                "subdirs": {
                    "results/": "Training history, explanations, visualizations",
                    "results/shap/": "SHAP explanation outputs",
                }
            },
            "configs/": {
                "description": "Configuration files",
                "files": {
                    "config.yaml": "Main configuration (hyperparameters, paths, etc.)",
                }
            },
            "logs/": {
                "description": "Training and execution logs",
                "files": {
                    "main.log": "Main training/execution log",
                }
            }
        },
        "root_files": {
            "main.py": {
                "description": "Entry point for training and explanation",
                "commands": [
                    "python main.py train",
                    "python main.py explain --model_path ... --image_path ...",
                ]
            },
            "test_setup.py": {
                "description": "Setup verification script",
                "tests": [
                    "Directory structure",
                    "Module imports",
                    "Dataset availability",
                    "Configuration loading",
                    "Model creation",
                ]
            },
            "example_analysis.py": {
                "description": "Example usage and analysis workflow",
                "demonstrates": [
                    "Loading trained model",
                    "Making predictions",
                    "Generating SHAP explanations",
                    "Computing metrics",
                ]
            },
            "requirements.txt": {
                "description": "Python dependencies with versions",
                "packages": [
                    "torch==2.1.2",
                    "torchvision==0.16.2",
                    "shap==0.43.1",
                    "numpy, pandas, scikit-learn",
                    "matplotlib, seaborn",
                    "yaml, pillow, tqdm",
                ]
            },
            ".gitignore": {
                "description": "Git ignore rules",
                "ignores": [
                    "data/ (dataset)",
                    "__pycache__/",
                    "*.pyc",
                    ".venv/",
                    "models/checkpoints/",
                    "*.log",
                ]
            },
            "README.md": {
                "description": "Main project documentation",
                "sections": [
                    "Overview",
                    "Setup instructions",
                    "Quick start",
                    "API reference",
                    "Extending the project",
                ]
            },
            "SETUP_GUIDE.md": {
                "description": "Detailed setup instructions",
                "sections": [
                    "System requirements",
                    "Step-by-step setup",
                    "GPU configuration",
                    "Troubleshooting",
                    "Performance benchmarks",
                ]
            }
        }
    }
}

# Module Dependencies Map
DEPENDENCIES = {
    "models.cnn_model": {
        "depends_on": ["torch", "torchvision"],
        "used_by": ["models.train", "main.py", "example_analysis.py"]
    },
    "models.train": {
        "depends_on": ["models.cnn_model", "torch", "utils.dataset_loader"],
        "used_by": ["main.py"]
    },
    "utils.dataset_loader": {
        "depends_on": ["torch", "pandas", "utils.preprocessing"],
        "used_by": ["models.train", "main.py", "example_analysis.py"]
    },
    "utils.preprocessing": {
        "depends_on": ["torch", "torchvision", "PIL"],
        "used_by": ["utils.dataset_loader", "xai.shap_explainer", "main.py"]
    },
    "utils.visualization": {
        "depends_on": ["matplotlib", "numpy"],
        "used_by": ["main.py", "example_analysis.py", "xai.shap_explainer"]
    },
    "xai.base_explainer": {
        "depends_on": ["torch", "abc"],
        "used_by": ["xai.shap_explainer"]
    },
    "xai.shap_explainer": {
        "depends_on": ["xai.base_explainer", "shap", "torch", "numpy"],
        "used_by": ["main.py", "example_analysis.py"]
    },
    "evaluation.metrics": {
        "depends_on": ["numpy", "sklearn"],
        "used_by": ["models.train", "main.py", "example_analysis.py"]
    }
}

# Data Flow
DATA_FLOW = """
┌─────────────────────────────────────────────────────────────────┐
│                      DATA FLOW DIAGRAM                          │
└─────────────────────────────────────────────────────────────────┘

TRAINING PIPELINE:
─────────────────
  Dataset (NIH ChestX-ray14)
         ↓
  ChestXrayDataLoader
         ↓
  Image Preprocessing
  ├─ Resize (224×224)
  ├─ Augmentation
  └─ Normalization
         ↓
  DataLoader (batches)
         ↓
  ResNet50ChestXray
         ↓
  BCEWithLogitsLoss
         ↓
  Training & Validation
         ↓
  Model Checkpoint (best_model.pth)


EXPLANATION PIPELINE:
────────────────────
  Trained Model
         ↓
  Test Image
         ↓
  Preprocessing
         ↓
  Background Dataset (50 samples)
         ↓
  SHAPExplainer
         ↓
  SHAP Values
         ↓
  Visualization (Heatmap + Overlay)
         ↓
  Results (experiments/results/shap/)


EVALUATION PIPELINE:
───────────────────
  Model
  Predictions
  Ground Truth
         ↓
  MultiLabelMetrics
  ├─ AUROC
  ├─ AUPRC
  ├─ F1 Score
  ├─ Accuracy
  └─ Per-class Metrics
         ↓
  Evaluation Results
"""

# Configuration Options
CONFIGURATION_OPTIONS = """
╔═══════════════════════════════════════════════════════════════════╗
║               CONFIGURATION FILE (config.yaml)                    ║
╚═══════════════════════════════════════════════════════════════════╝

dataset:
  path: "data/raw/NIH_ChestXray"           # Dataset location
  image_size: 224                          # Input image size
  num_classes: 14                          # Number of disease classes
  train_val_list: "train_val_list.txt"     # Train/val split file
  test_list: "test_list.txt"               # Test split file

model:
  architecture: "resnet50"                 # Model architecture
  pretrained: true                         # Use ImageNet weights
  dropout_rate: 0.5                        # Dropout before FC layer

training:
  batch_size: 32                           # Training batch size
  num_epochs: 10                           # Number of epochs
  learning_rate: 0.001                     # Learning rate
  weight_decay: 1e-5                       # L2 regularization
  warmup_epochs: 2                         # Warmup epochs
  val_split: 0.1                           # Validation split ratio

shap:
  background_size: 50                      # Background samples for SHAP
  num_samples: 100                         # SHAP sampling iterations
  device: "auto"                           # "auto", "cuda", or "cpu"

evaluation:
  metrics:                                 # Metrics to compute
    - "auroc"
    - "auprc"
    - "f1_score"
    - "accuracy"
"""

# API Quick Reference
API_REFERENCE = """
╔═══════════════════════════════════════════════════════════════════╗
║                      API QUICK REFERENCE                          ║
╚═══════════════════════════════════════════════════════════════════╝

MODELS
──────
from models import get_model, load_model, save_model

model = get_model(num_classes=14, device='cuda')
model = load_model('path/to/model.pth', device='cuda')
save_model(model, 'path/to/model.pth', optimizer=opt, epoch=10)


DATASET LOADING
───────────────
from utils.dataset_loader import ChestXrayDataLoader

loader = ChestXrayDataLoader('data/raw/NIH_ChestXray', 'csv_path')
train_loader, val_loader = loader.get_train_val_loaders(...)
test_loader = loader.get_test_loader(...)
background_loader = loader.get_background_loader(...)


PREPROCESSING
──────────────
from utils.preprocessing import preprocess_image, denormalize_image

image = preprocess_image('path.png', image_size=224, augment=False)
denorm = denormalize_image(image)


VISUALIZATION
──────────────
from utils.visualization import plot_image, overlay_heatmap

plot_image(image, title="Title", save_path="path.png")
overlay_heatmap(image, heatmap, alpha=0.5, save_path="path.png")


XAI - SHAP
──────────
from xai.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(background_loader=bg_loader, device='cuda')
explanation = explainer.explain(image, model, target_class=None)


METRICS
───────
from evaluation.metrics import MultiLabelMetrics

auroc = MultiLabelMetrics.compute_auroc(predictions, labels)
auprc = MultiLabelMetrics.compute_auprc(predictions, labels)
f1 = MultiLabelMetrics.compute_f1_score(predictions, labels)
metrics = MultiLabelMetrics.compute_per_class_metrics(pred, labels, classes)
"""

# Disease Classes
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# Key Files
KEY_FILES = """
╔═══════════════════════════════════════════════════════════════════╗
║                    KEY PROJECT FILES                              ║
╚═══════════════════════════════════════════════════════════════════╝

MAIN ENTRY POINTS:
──────────────────
  main.py                    Main CLI for training and explanations
  test_setup.py             Setup verification
  example_analysis.py       Example workflow


CORE MODULES:
─────────────
  models/cnn_model.py        ResNet50 architecture
  models/train.py            Training pipeline
  utils/dataset_loader.py    Dataset handling
  utils/preprocessing.py     Image preprocessing
  xai/shap_explainer.py      SHAP explanations
  evaluation/metrics.py      Evaluation metrics


CONFIGURATION:
───────────────
  configs/config.yaml        Main configuration
  requirements.txt           Python dependencies
  .gitignore                Git ignore rules


DOCUMENTATION:
────────────────
  README.md                 Main documentation
  SETUP_GUIDE.md           Setup instructions
  PROJECT_STRUCTURE.py     This file
"""

if __name__ == "__main__":
    print("XAI PROJECT STRUCTURE SUMMARY")
    print("=" * 70)
    print(DATA_FLOW)
    print("\n" + "=" * 70)
    print(CONFIGURATION_OPTIONS)
    print("\n" + "=" * 70)
    print(API_REFERENCE)
    print("\n" + "=" * 70)
    print(KEY_FILES)
    print("\n" + "=" * 70)
    print(f"\nTotal Disease Classes: {len(DISEASE_CLASSES)}")
    for i, disease in enumerate(DISEASE_CLASSES, 1):
        print(f"  {i:2d}. {disease}")
