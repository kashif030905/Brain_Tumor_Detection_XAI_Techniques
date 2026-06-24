# 🧠 XAI Brain Tumor Classification - Complete Guide

## Overview

This guide covers how to use the Brain Tumor Classification model with three different explainability techniques:
- **Grad-CAM** (Fast, Visual)
- **LIME** (Local, Interpretable)
- **SHAP** (Detailed, Comprehensive)

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Running Model Inference Only](#running-model-inference-only)
3. [Using Grad-CAM](#using-grad-cam)
4. [Using LIME](#using-lime)
5. [Using SHAP](#using-shap)
6. [Comparison of Methods](#comparison-of-methods)
7. [Output Descriptions](#output-descriptions)

---

## Quick Start

### Prerequisites
Ensure you're in the correct virtual environment:
```bash
# Activate environment if not already active
source xai-env/bin/activate
```

### Fastest Way to Get Started (Grad-CAM)
```bash
python3 main.py explain \
  --config configs/config.yaml \
  --model_path models/checkpoints/checkpoint_epoch_10.pth \
  --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg \
  --method gradcam
```

**Time: ~2 seconds** ⚡
**Output Location**: `experiments/results/Grad-CAM/`

---

## Running Model Inference Only

If you only want predictions without explanations, use Grad-CAM (it's the fastest explainer):

```bash
python3 main.py explain \
  --config configs/config.yaml \
  --model_path models/checkpoints/best_model.pth \
  --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg \
  --method gradcam
```

**What you get:**
- Class predictions with confidence scores
- Visual explanation (Grad-CAM heatmap)
- Execution time: ~2 seconds

**Fastest option for production predictions**: Use Grad-CAM as it completes in ~2 seconds with visual explanations included.

---

## Using Grad-CAM

### What is Grad-CAM?
Grad-CAM (Gradient-weighted Class Activation Mapping) shows which regions of the image are most important for the model's prediction.

**Characteristics:**
- ✅ **Speed**: ~2 seconds per image
- ✅ **Visual**: Easy to understand heatmaps
- ✅ **Multiple methods**: Standard, Guided, Multi-scale
- ❌ **Limited detail**: Doesn't show feature interactions

### Basic Usage
```bash
python3 main.py explain \
  --config configs/config.yaml \
  --model_path models/checkpoints/best_model.pth \
  --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg \
  --method gradcam
```

### Example Images
```bash
# Test on different tumor types
# Pituitary
python3 main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg --method gradcam

# Glioma
python3 main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/glioma/Te-gl_1.jpg --method gradcam

# Meningioma
python3 main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/meningioma/Te-men_1.jpg --method gradcam

# No Tumor
python3 main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/notumor/Te-no_1.jpg --method gradcam
```

### Output Files
Generated in `experiments/results/Grad-CAM/`:

1. **gradcam_enhanced_[class].png** (562 KB)
   - 5-panel visualization:
     - Original image
     - Raw Grad-CAM heatmap
     - Professional overlay
     - Attention mask (binary)
     - Confidence map
     - Model predictions

2. **gradcam_comparison_[class].png** (1.2 MB)
   - Side-by-side comparison of three methods:
     - Standard Grad-CAM
     - Guided Grad-CAM
     - Multi-scale Grad-CAM

3. **predictions.png** (31 KB)
   - Bar chart of class probabilities

### Interpreting Results
- **Red regions** = High importance (model focused here)
- **Blue regions** = Low importance
- **Overlay** = Combines original image with heatmap for context
- **Confidence Map** = Shows model certainty by region

### When to Use Grad-CAM
✅ Quick checks and fast prototyping
✅ Visual exploration of model behavior
✅ Production environments (speed critical)
✅ When you need multiple layers analyzed
✅ Medical imaging interpretability

---

## Using LIME

### What is LIME?
LIME (Local Interpretable Model-agnostic Explanations) explains predictions by approximating the model locally with an interpretable model.

**Characteristics:**
- ✅ **Model-agnostic**: Works with any model
- ✅ **Local explanations**: Focuses on specific predictions
- ✅ **Interpretable**: Uses simple linear models
- ⏱️ **Speed**: ~1-2 minutes per image
- ✅ **Superpixel analysis**: Explains important regions

### Basic Usage
```bash
python main.py explain \
  --config configs/config.yaml \
  --model_path models/checkpoints/best_model.pth \
  --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg \
  --method lime
```

### Example Images
```bash
# Pituitary
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg --method lime

# Glioma
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/glioma/Te-gl_1.jpg --method lime

# Meningioma
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/meningioma/Te-men_1.jpg --method lime
```

### Output Files
Generated in `experiments/results/LIME/`:

1. **lime_[class].png** 
   - 6-panel visualization:
     - Original image
     - Superpixel segmentation
     - LIME importance map
     - Soft overlay (50-50)
     - Important regions mask
     - Predictions & R² score

2. **lime_detailed_[class].png**
   - Detailed analysis with:
     - Original + boundaries
     - Positive contributions (supporting prediction)
     - Negative contributions (against prediction)
     - Multiple overlay styles

3. **predictions.png**
   - Bar chart of class probabilities

### Interpreting Results
- **Red regions** = High importance
- **Blue regions** = Low importance
- **R² Score** = Model quality (higher is better, max 1.0)
- **Superpixels** = Image segments analyzed
- **Positive/Negative contributions** = What supports/opposes the prediction

### When to Use LIME
✅ Understanding why specific decisions were made
✅ When you need both supporting and opposing evidence
✅ Research and detailed analysis
✅ Model debugging and validation
✅ When model-agnosticism is important
❌ When speed is critical (takes 1-2 minutes)

---

## Using SHAP

### What is SHAP?
SHAP (SHapley Additive exPlanations) uses game theory to explain model predictions. It shows the contribution of each feature to the final prediction.

**Characteristics:**
- ✅ **Theoretically sound**: Based on Shapley values
- ✅ **Feature-level explanations**: Precise attribution
- ✅ **Comprehensive**: Shows all feature interactions
- ⏱️ **Speed**: ~90 seconds per image on CPU
- ✅ **High quality**: Most detailed explanations

### Basic Usage
```bash
python main.py explain \
  --config configs/config.yaml \
  --model_path models/checkpoints/best_model.pth \
  --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg \
  --method shap
```

### Example Images
```bash
# Pituitary
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg --method shap

# Glioma
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/glioma/Te-gl_1.jpg --method shap

# Meningioma
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path data/raw/MRI/Testing/meningioma/Te-men_1.jpg --method shap
```

### Output Files
Generated in `experiments/results/shap/`:

1. **original_image.png**
   - Original MRI scan for reference

2. **shap_explanation.png**
   - SHAP heatmap showing feature importance
   - Red = Positive contribution
   - Blue = Negative contribution

3. **predictions.png**
   - Bar chart of class probabilities

### Interpreting Results
- **Red areas** = Pixels supporting the prediction
- **Blue areas** = Pixels opposing the prediction
- **Intensity** = Strength of contribution
- **Overall pattern** = How model makes decisions

### When to Use SHAP
✅ Research and publications
✅ Detailed feature importance analysis
✅ Model validation and debugging
✅ When accuracy of explanation matters most
✅ Regulatory/compliance requirements
❌ When speed is critical (slow on CPU)
✅ When GPU available (much faster)

---

## Comparison of Methods

| Aspect | Grad-CAM | LIME | SHAP |
|--------|----------|------|------|
| **Speed (CPU)** | ~2 sec ⚡⚡⚡ | ~90 sec ⚡ | ~90 sec ⚡ |
| **Visual Quality** | Excellent | Good | Good |
| **Interpretability** | High | Very High | Very High |
| **Detail Level** | Medium | Medium | High |
| **Model-agnostic** | No | Yes | No |
| **Best For** | Quick analysis | Understanding decisions | Research |
| **Learning Curve** | Easy | Medium | Medium |
| **Output Complexity** | Simple | Medium | Complex |

---

## Output Descriptions

### Directory Structure
```
experiments/results/
├── Grad-CAM/
│   ├── gradcam_enhanced_[class].png
│   ├── gradcam_comparison_[class].png
│   └── predictions.png
├── LIME/
│   ├── lime_[class].png
│   ├── lime_detailed_[class].png
│   └── predictions.png
└── shap/
    ├── original_image.png
    ├── shap_explanation.png
    └── predictions.png
```

### Understanding Predictions
All methods output a predictions chart showing:
```
Class Probabilities:
  glioma     : 44.49% ← PREDICTED
  meningioma : 33.70%
  notumor    : 01.45%
  pituitary  : 20.37%
```

The **PREDICTED** marker shows which class the model chose.

---

## Tips & Tricks

### Batch Processing
Process multiple images:
```bash
# Using a loop
for img in data/raw/MRI/Testing/pituitary/*.jpg; do
  python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path "$img" --method gradcam
done
```

### Custom Output Directory
Specify where to save results:
```bash
python main.py explain \
  --config configs/config.yaml \
  --model_path models/checkpoints/best_model.pth \
  --image_path data/raw/MRI/Testing/pituitary/Te-pi_1.jpg \
  --method gradcam \
  --output_dir my_results/
```

### Comparing Methods on Same Image
Run all three methods on one image to compare:
```bash
IMG="data/raw/MRI/Testing/pituitary/Te-pi_1.jpg"

echo "Running Grad-CAM..."
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path "$IMG" --method gradcam

echo "Running LIME..."
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path "$IMG" --method lime

echo "Running SHAP..."
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path "$IMG" --method shap
```

---

## Troubleshooting

### Issue: "Command not found: python"
**Solution**: Use full path or activate virtual environment:
```bash
source xai-env/bin/activate
python main.py explain ...
```

### Issue: Model weights not found
**Solution**: Ensure checkpoint exists:
```bash
ls -lh models/checkpoints/best_model.pth
```

### Issue: Image path not found
**Solution**: Verify image exists:
```bash
ls -lh data/raw/MRI/Testing/pituitary/Te-pi_1.jpg
```

### Issue: LIME taking too long
**Solution**: This is normal! LIME requires 150 model evaluations. Grab a coffee ☕

### Issue: Memory error on SHAP
**Solution**: Reduce background samples in config.yaml:
```yaml
shap:
  background_size: 10  # Reduce from 20
  num_samples: 15      # Reduce from 30
```

---

## Performance Metrics

### Tested on MacBook Pro (CPU only):

**Grad-CAM**
- Inference time: ~2 seconds
- File size: ~1.8 MB (all outputs)
- Quality: Excellent

**LIME** (150 samples, 50 superpixels)
- Inference time: ~90 seconds
- File size: ~2 MB
- Quality: Very good

**SHAP** (30 background samples, 20 test samples)
- Inference time: ~90 seconds
- File size: ~1.5 MB
- Quality: Excellent

**With GPU (CUDA available):**
- All methods: 5-10x faster

---

## Advanced Usage

### Modifying Grad-CAM Parameters
Edit `main.py` line 216:
```python
explanation = explainer.explain(
    ...
    method='guided',  # Try: 'standard', 'guided', 'multiscale'
)
```

### Modifying LIME Parameters
Edit `main.py` line 329:
```python
explainer = LIMEExplainer(
    device=device,
    num_samples=150,      # Increase for more accuracy
    num_features=50,      # Increase for finer segmentation
)
```

### Modifying SHAP Parameters
Edit `configs/config.yaml`:
```yaml
shap:
  background_size: 20      # Number of background samples
  num_samples: 30          # Number of test samples
  save_dir: experiments/results/shap
```

---

## Citation

If you use these explanation methods, please cite:

**Grad-CAM:**
```
Selvaraju et al. (2016)
"Grad-CAM: Visual Explanations from Deep Networks via 
Gradient-based Localization"
```

**LIME:**
```
Ribeiro et al. (2016)
"'Why Should I Trust You?': Explaining the Predictions 
of Any Classifier"
```

**SHAP:**
```
Lundberg & Lee (2017)
"A Unified Approach to Interpreting Model Predictions"
```

---

## Getting Help

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `logs/`
3. Examine output images for visual clues
4. Check model predictions match expected behavior

---

## Summary

| Task | Command | Time |
|------|---------|------|
| **Quick test** | `--method gradcam` | 2 sec |
| **Understanding why** | `--method lime` | 90 sec |
| **Deep analysis** | `--method shap` | 90 sec |
| **Predictions only** | `infer` command | <1 sec |

**Recommended workflow:**
1. Start with Grad-CAM for quick feedback
2. Use LIME for specific decision analysis
3. Apply SHAP for research/publication quality
4. Compare all three for comprehensive understanding

---

Happy analyzing! 🚀
