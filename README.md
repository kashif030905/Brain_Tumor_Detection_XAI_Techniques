# XAI Techniques Analysis — MRI Brain Tumour Classification

This repository provides a working PyTorch pipeline for MRI brain tumour
classification (ResNet50-based) and integrated explainability tools with a
modular evaluation system that compares SHAP, Grad-CAM and LIME.

This README is a concise, actionable reference for the current code in this
workspace (training, inference, explainers, and evaluation).

Prerequisites
- Python 3.8+ with required packages installed from `requirements.txt`.
- Data placed under `data/raw/MRI/` with `Training/` and `Testing/` subfolders.

Repository layout (key folders)
- `models/` — model architecture and training utilities.
- `xai/` — explainers: `shap_explainer.py`, `gradcam_enhanced.py`, `lime_explainer.py`.
- `evaluation/` — evaluation utilities: `xai_explainers_eval.py`, `xai_comparison.py`.
- `scripts/` — helper scripts; `evaluate_xai.py` runs the evaluator.
- `utils/` — preprocessing, dataset loader, visualization helpers.
- `configs/config.yaml` — main configuration for dataset/model/training/explainers.
- `experiments/results/` — where visual outputs and evaluation plots are saved.

Quick commands
All commands assume your working directory is the repository root.

- Train (CPU):

```bash
CUDA_VISIBLE_DEVICES="" python main.py train --config configs/config.yaml
```

- Explain a single image (method: `shap`, `gradcam`, `lime`):

```bash
python main.py explain --config configs/config.yaml --model_path models/checkpoints/best_model.pth --image_path <IMAGE_PATH> --method gradcam
```

- Quick evaluator (smoke test using small subset):

```bash
PYTHONPATH=. python3 scripts/evaluate_xai.py --config configs/config.yaml --model_path models/checkpoints/best_model.pth --subset_size 1 --device cpu
```

- Full comparison (larger subset; slow):

```bash
PYTHONPATH=. python3 scripts/evaluate_xai.py --config configs/config.yaml --model_path models/checkpoints/best_model.pth --subset_size 100 --device cpu
```

Notes
- For local imports when running scripts, use `PYTHONPATH=.`.
- Use `--device cuda` when running on a machine with a GPU to speed up SHAP/LIME.
- SHAP needs a background set; the evaluator will attempt to build one from training data.

Explainability methods (implemented)
- SHAP (GradientExplainer): pixel-level attributions using gradients.
- Grad-CAM (Enhanced): guided & multi-scale Grad-CAM with post-processing and overlays.
- LIME: superpixel-based perturbation + weighted linear model; evaluator rasterizes superpixel weights to pixels for consistent comparison.

Evaluation metrics (implemented)
- Fidelity: whether top-1 prediction is preserved when keeping only top-k important pixels.
- Faithfulness: relative drop in prediction confidence when iteratively removing most important pixels.
- Stability: similarity (cosine) between heatmaps for small input perturbations.
- Sparsity: fraction of pixels considered important; encourages compact explanations.
- Runtime: average runtime per explanation (normalized; lower is better before inversion).

All metrics are normalized to [0,1] across methods for easy comparison. The evaluation code
converts all explanations to pixel-space heatmaps (rasterizing LIME superpixel weights when necessary)
so metrics are computed consistently.

Visualizations
- `experiments/results/xai_comparison/` contains the generated comparison plots:
    - `bar_chart.png` — per-metric bar chart
    - `radar_chart.png` — radar plot across metrics
    - `heatmap.png` — methods vs metrics matrix
    - `results.json` — numeric values used for plotting

Where per-explainer outputs are saved
- Grad-CAM images: `experiments/results/Grad-CAM/`
- LIME images: `experiments/results/LIME/`
- SHAP images: `experiments/results/shap/`

Extending the project
- Add a new explainer: implement an explainer under `xai/` exposing `explain(image, model, target_class)`.
    Return either a pixel heatmap (HxW) or `weights`+`segments` for superpixel explainers; the evaluator will rasterize as needed.
- Add metrics: extend `evaluation/xai_comparison.py` — each metric should accept a pixel-heatmap and return a scalar.

Troubleshooting
- If imports fail when running scripts, prefix the command with `PYTHONPATH=.` and run from the repository root.
- SHAP and LIME are computationally expensive; test with small `--subset_size` and use GPU when available.

If you'd like, I can also add a short changelog summarizing the recent code and README changes, or commit these README edits for you.
# F1 Score
f1 = MultiLabelMetrics.compute_f1_score(predictions, labels, threshold=0.5)

# Accuracy
accuracy = MultiLabelMetrics.compute_accuracy(predictions, labels)

# Per-class metrics
per_class = MultiLabelMetrics.compute_per_class_metrics(
    predictions, labels, disease_classes
)
```

### Sensitivity Analysis

```python
from evaluation.metrics import sensitivity_analysis

sensitivity = sensitivity_analysis(
    model=model,
    image=image_tensor,
    target_class=0,
    perturbation=0.1,
    num_perturbations=100
)

print(f"Mean sensitivity: {sensitivity['mean_sensitivity']:.4f}")
```

## 🔄 Extending the Project

### Adding LIME Explainer

1. Create `xai/lime_explainer.py`:

```python
from xai.base_explainer import Explainer, ExplainerFactory

class LIMEExplainer(Explainer):
    def __init__(self, **kwargs):
        super().__init__('LIME')
    
    def explain(self, image, model, target_class=None):
        # LIME implementation
        pass
    
    def explain_batch(self, images, model, target_class=None):
        # Batch LIME implementation
        pass

ExplainerFactory.register('lime', LIMEExplainer)
```

2. Use in pipeline:

```python
from xai.base_explainer import ExplainerFactory

explainer = ExplainerFactory.create('lime', num_samples=1000)
explanation = explainer.explain(image, model)
```

### Adding Grad-CAM

Similar process in `xai/gradcam_explainer.py`

## 📝 Logging

Training logs are saved to `logs/main.log`:

```
2024-04-09 10:23:45,123 - __main__ - INFO - Starting training...
2024-04-09 10:24:12,456 - models.train - INFO - Epoch 1/10
2024-04-09 10:24:45,789 - models.train - INFO - Training loss: 0.1234
```

Access logs during training:
```bash
tail -f logs/main.log
```

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in `configs/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Slow Data Loading

**Solution**: Increase number of workers:
```yaml
training:
  num_workers: 8  # Increase from 4
```

### Issue: Image Not Found

**Solution**: Verify image path:
```bash
find data/raw/NIH_ChestXray -name "*.png" | head -5
```

### Issue: Model Training Stalled

**Solution**: Check if GPU is being used:
```python
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 📚 API Reference

### Models

```python
from models import get_model, load_model, save_model

# Create new model
model = get_model(num_classes=14, device='cuda')

# Load trained model
model = load_model('models/checkpoints/best_model.pth', device='cuda')

# Save model
save_model(model, 'path/to/model.pth', optimizer=optimizer, epoch=10)
```

### Dataset

```python
from utils.dataset_loader import ChestXrayDataLoader

loader = ChestXrayDataLoader('data/raw/NIH_ChestXray', 'data_entry.csv')
train_loader, val_loader = loader.get_train_val_loaders(...)
test_loader = loader.get_test_loader(...)
background_loader = loader.get_background_loader(...)
```

### Preprocessing

```python
from utils.preprocessing import preprocess_image, denormalize_image

image = preprocess_image('path.png', augment=True)
denorm = denormalize_image(image)
```

### XAI

```python
from xai.base_explainer import ExplainerFactory

explainer = ExplainerFactory.create('shap', background_loader=bg_loader)
explanation = explainer.explain(image, model, target_class=None)
```

## 🎓 References

- **ChestX-ray14**: https://nihcc.app.box.com/v/ChestXray-NIHCC
- **ResNet**: He et al., 2015 - https://arxiv.org/abs/1512.03385
- **SHAP**: Lundberg & Lee, 2017 - https://arxiv.org/abs/1705.07874

## 📄 License

This project is provided for educational and research purposes.

## ✨ Features

- ✅ Production-quality code with proper error handling
- ✅ Modular, extensible architecture
- ✅ Comprehensive logging and monitoring
- ✅ GPU support (CUDA)
- ✅ Multi-label classification (14 diseases)
- ✅ SHAP explanations with visualization
- ✅ Extensive evaluation metrics
- ✅ Configuration-driven pipeline
- ✅ Clean, well-documented code
- ✅ Easy to extend (LIME, Grad-CAM ready)

## 🚀 Next Steps

1. **Train the model**: `python main.py train`
2. **Generate explanations**: `python main.py explain --model_path ... --image_path ...`
3. **Analyze results**: Check `experiments/results/` for outputs
4. **Extend**: Add LIME or Grad-CAM using the modular architecture

## 📧 Support

For issues or questions, refer to:
- Configuration: `configs/config.yaml`
- Logs: `logs/main.log`
- Results: `experiments/results/`

