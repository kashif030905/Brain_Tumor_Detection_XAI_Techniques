# 🧠 Brain Tumor Detection with Explainable AI (XAI)

A PyTorch pipeline for MRI brain tumor classification using a ResNet50 backbone, integrated with three explainability techniques — **SHAP**, **Grad-CAM**, and **LIME** — plus a modular evaluation framework to compare them quantitatively.

This project directly supports the co-authored research paper **"Comparative Analysis of Explainable AI Techniques: LIME, SHAP, and Grad-CAM for Medical Imaging"** (VNRVJIET).

---

## 📌 Project Overview

Deep learning models for medical imaging are often accurate but opaque. This project builds a brain tumor MRI classifier and then asks: *which explainability method actually explains the model's decisions best?* Three popular XAI techniques are implemented on the same model and evaluated against a common set of metrics, rather than judged only by visual inspection.

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Model | PyTorch, ResNet50 (transfer learning) |
| Explainability | SHAP (GradientExplainer), Grad-CAM (enhanced, multi-scale), LIME (superpixel-based) |
| Evaluation | Custom metrics — fidelity, faithfulness, stability, sparsity, runtime |
| Data | MRI scans, `Training/` and `Testing/` split |

## 📁 Project Structure

```
Brain_Tumor_Detection_XAI_Techniques/
├── configs/
│   └── config.yaml              # Dataset, model, training, explainer config
├── models/                      # Model architecture + training utilities
├── xai/                         # Explainer implementations
│   ├── shap_explainer.py
│   ├── gradcam_enhanced.py
│   └── lime_explainer.py
├── evaluation/                  # Evaluation framework
│   ├── xai_explainers_eval.py
│   └── xai_comparison.py
├── scripts/
│   └── evaluate_xai.py          # Runs the evaluator
├── utils/                       # Preprocessing, dataset loader, visualization
├── main.py                      # Train / explain entry point
├── inference.py
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

1. **Train** a ResNet50 classifier on MRI scans (`Training/` / `Testing/` folders under `data/raw/MRI/`).
2. **Explain** individual predictions using SHAP, Grad-CAM, or LIME.
3. **Evaluate** all three explainers on a shared metric suite so results are comparable, not just visually similar.

### Explainability Methods

- **SHAP (GradientExplainer)** — pixel-level attributions using gradients.
- **Grad-CAM (Enhanced)** — guided, multi-scale Grad-CAM with post-processing and overlays.
- **LIME** — superpixel-based perturbation with a weighted linear model; outputs are rasterized to pixel-level heatmaps for fair comparison.

### Evaluation Metrics

- **Fidelity** — does the top-1 prediction survive when only the top-k important pixels are kept?
- **Faithfulness** — how much does confidence drop as important pixels are removed?
- **Stability** — cosine similarity between heatmaps under small input perturbations.
- **Sparsity** — fraction of pixels flagged as important (more compact = better).
- **Runtime** — average time per explanation, normalized across methods.

All metrics are normalized to [0, 1] so the three methods can be ranked directly.

## 🚀 Setup & Usage

```bash
# Train (CPU)
CUDA_VISIBLE_DEVICES="" python main.py train --config configs/config.yaml

# Explain a single image (method: shap | gradcam | lime)
python main.py explain --config configs/config.yaml \
  --model_path models/checkpoints/best_model.pth \
  --image_path <IMAGE_PATH> --method gradcam

# Quick evaluator (smoke test)
PYTHONPATH=. python3 scripts/evaluate_xai.py \
  --config configs/config.yaml \
  --model_path models/checkpoints/best_model.pth \
  --subset_size 1 --device cpu

# Full comparison (slower, larger subset)
PYTHONPATH=. python3 scripts/evaluate_xai.py \
  --config configs/config.yaml \
  --model_path models/checkpoints/best_model.pth \
  --subset_size 100 --device cpu
```

## 📊 Outputs

Generated comparison plots are saved to `experiments/results/xai_comparison/`:
- `bar_chart.png` — per-metric comparison across methods
- `radar_chart.png` — radar plot across all metrics
- `heatmap.png` — methods vs. metrics matrix
- `results.json` — raw numeric results

Per-explainer visualizations are saved separately under `experiments/results/Grad-CAM/`, `experiments/results/LIME/`, and `experiments/results/shap/`.

## 🔄 Extending the Project

- **New explainer**: implement it under `xai/`, expose an `explain(image, model, target_class)` method, and register it with `ExplainerFactory`.
- **New metric**: extend `evaluation/xai_comparison.py` — each metric should accept a pixel-heatmap and return a scalar.

## 📚 References

- **ResNet** — He et al., 2015 — https://arxiv.org/abs/1512.03385
- **SHAP** — Lundberg & Lee, 2017 — https://arxiv.org/abs/1705.07874

## 📄 License

Provided for educational and research purposes.

## 👤 Author

**Syed Kashif Uddin**
B.Tech CSE | VNRVJIET, Hyderabad