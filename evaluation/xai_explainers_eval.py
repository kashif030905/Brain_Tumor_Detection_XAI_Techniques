"""
Evaluate XAI explainers (SHAP, Grad-CAM, LIME) by measuring prediction accuracy
on masked images produced by each explainer. The evaluation masks out the least
important pixels/superpixels and measures whether the model's top prediction
remains the same (or not). Produces numeric scores per-method.

Usage:
    from evaluation.xai_explainers_eval import XAIExplainersEvaluator

The evaluator expects a PyTorch model (loaded), a callable preprocess function
that returns a normalized tensor (C,H,W), and a list of image paths.
"""

from typing import List, Dict, Callable, Tuple
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_image, denormalize_image


class XAIExplainersEvaluator:
    """Compute masked-image based accuracy for SHAP, Grad-CAM and LIME.

    Contract:
      - inputs: image paths list
      - outputs: dict of method -> accuracy (fraction of samples where top-1 label
        is unchanged after masking out low-importance regions)
    """

    def __init__(self, model: torch.nn.Module, preprocess_fn: Callable, device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.preprocess_fn = preprocess_fn
        self.device = device

    def _predict(self, image_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(image_tensor.unsqueeze(0).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def _mask_image_by_heatmap(self, image: np.ndarray, heatmap: np.ndarray, keep_fraction: float = 0.2) -> np.ndarray:
        """Mask image by keeping top `keep_fraction` of heatmap importance.

        image: HxWxC float in [0,1]
        heatmap: HxW float (0..1)
        """
        h, w = heatmap.shape
        flat = heatmap.flatten()
        kth = int((1.0 - keep_fraction) * flat.size)
        if kth <= 0:
            thresh = flat.min() - 1.0
        else:
            thresh = np.partition(flat, kth)[kth]

        mask = (heatmap >= thresh).astype(float)[..., None]
        masked = image * mask + (1 - mask) * 0.0  # replace low-importance with black
        return masked

    def evaluate_on_subset(
        self,
        image_paths: List[str],
        explainer_callables: Dict[str, Callable[[str], Dict]],
        keep_fraction: float = 0.2,
        max_samples: int = 100,
    ) -> Dict[str, float]:
        """Evaluate each explainer on a subset of images.

        explainer_callables: map name->callable(image_path) that returns a dict with keys:
          - 'heatmap': HxW float in [0,1]
        Returns map name->accuracy (top-1 unchanged fraction)
        """
        results = {name: 0 for name in explainer_callables.keys()}
        counts = {name: 0 for name in explainer_callables.keys()}

        image_paths = image_paths[:max_samples]

        for img_path in tqdm(image_paths, desc='Evaluating images'):
            # load and preprocess
            try:
                img_tensor = self.preprocess_fn(img_path)
            except Exception:
                continue

            # denormalized image for masking
            img_denorm = denormalize_image(img_tensor).cpu().numpy()
            img_denorm = np.transpose(img_denorm, (1, 2, 0))  # HWC

            # original prediction
            orig_probs = self._predict(img_tensor)
            orig_top = orig_probs.argmax()

            for name, explainer_fn in explainer_callables.items():
                try:
                    out = explainer_fn(img_path)
                except Exception:
                    continue

                heatmap = out.get('heatmap')
                if heatmap is None:
                    continue

                # ensure heatmap is HxW and same size as image
                # robustly handle heatmaps that have extra dims (C,H,W) or (1,H,W)
                hm = np.array(heatmap)
                # squeeze singleton dims
                hm = np.squeeze(hm)

                # If heatmap still has more than 2 dims, average across leading axes
                # so we end up with a 2D array (H, W). This handles shapes like
                # (1,1,H,W), (C,H,W), (num_classes, H, W), or other unexpected shapes.
                if hm.ndim > 2:
                    axes_to_mean = tuple(range(hm.ndim - 2))
                    hm = np.mean(np.abs(hm), axis=axes_to_mean)

                # now hm should be HxW or still mismatched; normalize to 0..1
                if hm.max() > 0:
                    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)
                else:
                    hm = np.zeros_like(hm)

                if hm.shape != (img_denorm.shape[0], img_denorm.shape[1]):
                    # resize via PIL
                    hm_img = Image.fromarray((hm * 255).astype(np.uint8))
                    hm_img = hm_img.resize((img_denorm.shape[1], img_denorm.shape[0]), resample=Image.BILINEAR)
                    heatmap = np.array(hm_img).astype(float) / 255.0
                else:
                    heatmap = hm

                masked = self._mask_image_by_heatmap(img_denorm, heatmap, keep_fraction=keep_fraction)

                # convert masked back to tensor normalized
                # masked is HWC [0,1]
                masked_t = torch.from_numpy(np.transpose(masked, (2, 0, 1))).float()
                # normalize
                # apply ImageNet normalization used by preprocessing
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                masked_norm = (masked_t - mean) / std

                probs_masked = self._predict(masked_norm)
                top_masked = probs_masked.argmax()

                counts[name] += 1
                if top_masked == orig_top:
                    results[name] += 1

        # compute fractions
        accuracies = {}
        for name in results.keys():
            if counts[name] == 0:
                accuracies[name] = float('nan')
            else:
                accuracies[name] = results[name] / counts[name]

        return accuracies


def plot_accuracies(accuracies: Dict[str, float], save_path: str = 'experiments/results/xai_accuracies.png'):
    names = list(accuracies.keys())
    vals = [accuracies[n] for n in names]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, vals, color=['#4C72B0', '#55A868', '#C44E52'])
    plt.ylim(0, 1)
    plt.ylabel('Top-1 preserved fraction')
    plt.title('XAI explainers: top-1 preservation on masked images')
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.2f}', ha='center')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
