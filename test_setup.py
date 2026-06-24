"""
Quick test script to verify the project setup and dataset.

Usage:
    python test_setup.py
"""

import os
import sys
from pathlib import Path

def test_directory_structure():
    """Test that required directories exist."""
    print("Testing directory structure...")
    
    required_dirs = [
        'configs',
        'models',
        'xai',
        'utils',
        'evaluation',
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            return False
    
    # Check data directories (can be missing)
    optional_dirs = ['data', 'data/raw', 'data/raw/NIH_ChestXray']
    for dir_path in optional_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"⚠ {dir_path} - OPTIONAL (dataset can be downloaded later)")
    
    return True


def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting imports...")
    
    modules = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'sklearn',
        'shap',
        'yaml',
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - {str(e)}")
            return False
    
    return True


def test_dataset():
    """Test dataset structure."""
    print("\nTesting dataset structure...")
    
    dataset_path = 'data/raw/NIH_ChestXray'
    
    if not os.path.exists(dataset_path):
        print(f"⚠ Dataset directory not found: {dataset_path}")
        print(f"  Note: Dataset must be downloaded separately from NIH")
        print(f"  See README.md for download instructions")
        return True  # Don't fail - dataset download is optional for setup verification
    
    print(f"✓ Dataset directory exists: {dataset_path}")
    
    # Check for required files
    required_files = [
        'Data_Entry_2017.csv',
        'train_val_list.txt',
        'test_list.txt',
        'BBox_List_2017.csv',
    ]
    
    for file in required_files:
        file_path = os.path.join(dataset_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file} - NOT FOUND")
            return False
    
    # Check for image folders
    image_folders_found = 0
    for i in range(1, 13):
        folder_name = f'images_{i:03d}'
        folder_path = os.path.join(dataset_path, folder_name, 'images')
        
        if os.path.exists(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
            print(f"✓ {folder_name}: {num_images} images")
            image_folders_found += 1
        else:
            print(f"✗ {folder_name} - NOT FOUND")
    
    if image_folders_found == 12:
        print(f"\n✓ All 12 image folders found")
        return True
    else:
        print(f"\n✗ Only {image_folders_found}/12 image folders found")
        return False


def test_config():
    """Test configuration file."""
    print("\nTesting configuration...")
    
    config_path = 'configs/config.yaml'
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config file loaded")
        print(f"  - Batch size: {config['training']['batch_size']}")
        print(f"  - Learning rate: {config['training']['learning_rate']}")
        print(f"  - Epochs: {config['training']['num_epochs']}")
        print(f"  - Image size: {config['dataset']['image_size']}")
        print(f"  - Num classes: {config['dataset']['num_classes']}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading config: {str(e)}")
        return False


def test_model():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from models import get_model
        
        print("Creating ResNet50 model...")
        model = get_model(num_classes=14, device='cpu')
        
        print(f"✓ Model created successfully")
        print(f"  - Architecture: ResNet50ChestXray")
        print(f"  - Output classes: 14")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error creating model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("XAI PROJECT SETUP TEST")
    print("=" * 70)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("Configuration", test_config),
        ("Model", test_model),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED - READY TO TRAIN!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Train model: python main.py train --config configs/config.yaml")
        print("2. Generate explanations: python main.py explain ...")
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ SOME TESTS FAILED - PLEASE FIX ISSUES")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
