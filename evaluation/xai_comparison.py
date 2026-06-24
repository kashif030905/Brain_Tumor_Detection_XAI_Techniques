"""
Compare XAI explainers quantitatively and visually.

Provides XAIComparisonEvaluator which computes fidelity, faithfulness,
stability, sparsity and runtime for a set of explainers and generates
matplotlib-only plots saved under a results folder.

This module relies on the project's existing explainer classes (SHAP, LIME, GradCAM)
and preprocessing utilities.
"""

from typing import List, Dict, Callable, Optional
import time
import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

from xai.shap_explainer import SHAPExplainer
from xai.gradcam_enhanced import EnhancedGradCAM
from xai.lime_explainer import LIMEExplainer
from utils.preprocessing import preprocess_image, denormalize_image

logger = logging.getLogger(__name__)


class XAIComparisonEvaluator:
    """Evaluate and compare XAI methods on image classification tasks.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    image_paths : list
        List of image file paths to evaluate on.
    methods : list
        List of method names to evaluate. Supported: ['lime','shap','gradcam']
    device : str
        'cpu' or 'cuda'
    subset_size : int
        Number of images to use from `image_paths`.
    keep_fraction : float
        Fraction of top pixels to keep when computing fidelity.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        image_paths: List[str],
        methods: List[str] = ['lime', 'shap', 'gradcam'],
        device: str = 'cpu',
        subset_size: int = 100,
        keep_fraction: float = 0.2,
        background_loader: Optional[object] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.methods = [m.lower() for m in methods]
        self.image_paths = image_paths[:subset_size]
        self.subset_size = min(subset_size, len(image_paths))
        self.keep_fraction = keep_fraction

        # initialize explainers
        self.explainers = {}
        if 'shap' in self.methods:
            # SHAP requires a background loader; prefer the provided one
            self.background_loader = background_loader
            self.explainers['shap'] = SHAPExplainer(background_loader=self.background_loader, num_samples=20, device=device)
        if 'gradcam' in self.methods:
            self.explainers['gradcam'] = EnhancedGradCAM(device=device)
        if 'lime' in self.methods:
            self.explainers['lime'] = LIMEExplainer(device=device, num_samples=150, num_features=50)

    def _call_explainer(self, name: str, img_path: str) -> Optional[np.ndarray]:
        """Return a normalized HxW heatmap in [0,1] or None on failure."""
        try:
            t = preprocess_image(img_path, augment=False)
            if name == 'shap':
                # SHAPExplainer.explain expects tensor and model; uses background set previously
                out = self.explainers['shap'].explain(image=t, model=self.model, target_class=None)
                hm = out.get('attributions')
                if hm is None:
                    return None
                if isinstance(hm, np.ndarray) and hm.ndim == 3:
                    hm2 = np.mean(np.abs(hm), axis=0)
                else:
                    hm2 = np.abs(hm)
            elif name == 'gradcam':
                # predict class
                with torch.no_grad():
                    logits = self.model(t.unsqueeze(0).to(self.device))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(probs.argmax())
                out = self.explainers['gradcam'].explain(image=t, model=self.model, predicted_class=pred, method='multiscale')
                hm2 = out.get('heatmap')
            elif name == 'lime':
                img_den = denormalize_image(t).cpu().numpy()
                img_den = np.transpose(img_den, (1, 2, 0))
                with torch.no_grad():
                    logits = self.model(t.unsqueeze(0).to(self.device))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(probs.argmax())
                out = self.explainers['lime'].explain(image=img_den, model=self.model, target_class=pred)
                hm2 = out.get('heatmap')
            else:
                return None

            if hm2 is None:
                return None

            hm2 = np.array(hm2)
            # reduce to 2D if necessary
            hm2 = np.squeeze(hm2)
            if hm2.ndim > 2:
                hm2 = np.mean(np.abs(hm2), axis=tuple(range(hm2.ndim - 2)))

            # normalize
            if hm2.max() > 0:
                hm2 = (hm2 - hm2.min()) / (hm2.max() - hm2.min() + 1e-12)
            else:
                hm2 = np.zeros_like(hm2)

            return hm2
        except Exception as e:
            logger.debug(f"Explainer {name} failed on {img_path}: {e}")
            return None

    def evaluate_all_methods(self) -> Dict[str, Dict[str, float]]:
        """Compute all requested metrics and return a nested dict with normalized scores."""
        # containers for raw metrics
        raw = {m: {'fidelity': [], 'faithfulness': [], 'stability': [], 'sparsity': [], 'runtime': []} for m in self.methods}

        # Precompute original predictions and confidences
        for img_path in self.image_paths:
            try:
                t = preprocess_image(img_path, augment=False)
            except Exception:
                continue

            with torch.no_grad():
                logits = self.model(t.unsqueeze(0).to(self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                orig_top = int(probs.argmax())
                orig_conf = float(probs[orig_top])

            for m in self.methods:
                start = time.perf_counter()
                hm = self._call_explainer(m, img_path)
                runtime = time.perf_counter() - start

                if hm is None:
                    # record NaNs to indicate missing
                    raw[m]['runtime'].append(runtime)
                    raw[m]['fidelity'].append(np.nan)
                    raw[m]['faithfulness'].append(np.nan)
                    raw[m]['stability'].append(np.nan)
                    raw[m]['sparsity'].append(np.nan)
                    continue

                # Ensure heatmap is 2D and resized to image dimensions
                hm = np.array(hm)
                hm = np.squeeze(hm)
                if hm.ndim > 2:
                    hm = np.mean(np.abs(hm), axis=tuple(range(hm.ndim - 2)))

                img_den = denormalize_image(t).cpu().numpy()
                img_den = np.transpose(img_den, (1, 2, 0))
                H, W = img_den.shape[0], img_den.shape[1]

                # Resize heatmap to image size if needed
                if hm.shape != (H, W):
                    try:
                        from PIL import Image
                        hm_img = Image.fromarray((np.clip(hm, 0, 1) * 255).astype(np.uint8))
                        hm_img = hm_img.resize((W, H), resample=Image.BILINEAR)
                        hm = np.array(hm_img).astype(float) / 255.0
                    except Exception:
                        # fallback: simple numpy resize by repeat/trim
                        hm = np.resize(hm, (H, W))

                # normalize heatmap to [0,1]
                if hm.max() > 0:
                    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)
                else:
                    hm = np.zeros((H, W), dtype=float)

                # fidelity: keep top fraction and check top-1 preserved
                k = int((1.0 - self.keep_fraction) * hm.size)
                flat = hm.flatten()
                thresh = np.partition(flat, k)[k] if k < flat.size else flat.min()
                mask = (hm >= thresh).astype(float)[..., None]

                # prepare masked image
                img_den = denormalize_image(t).cpu().numpy()
                img_den = np.transpose(img_den, (1, 2, 0))
                masked = img_den * mask + (1 - mask) * 0.0
                masked_t = torch.from_numpy(np.transpose(masked, (2, 0, 1))).float()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                masked_norm = (masked_t - mean) / std

                with torch.no_grad():
                    logits_mask = self.model(masked_norm.unsqueeze(0).to(self.device))
                    probs_mask = torch.softmax(logits_mask, dim=1).cpu().numpy()[0]
                    top_mask = int(probs_mask.argmax())
                    conf_mask = float(probs_mask[orig_top]) if orig_top < len(probs_mask) else 0.0

                fidelity = 1.0 if top_mask == orig_top else 0.0

                # faithfulness: iterative removal of top-k pixels and measure confidence drop
                steps = 5
                confidences = []
                hm_flat_idx = np.argsort(hm.flatten())[::-1]
                for s in range(1, steps + 1):
                    num_remove = int((s / steps) * hm.size)
                    mask2 = np.ones(hm.size, dtype=float)
                    if num_remove > 0:
                        mask2[hm_flat_idx[:num_remove]] = 0.0
                    mask2 = mask2.reshape(hm.shape)[..., None]
                    masked2 = img_den * mask2 + (1 - mask2) * 0.0
                    masked2_t = torch.from_numpy(np.transpose(masked2, (2, 0, 1))).float()
                    masked2_norm = (masked2_t - mean) / std
                    with torch.no_grad():
                        logits2 = self.model(masked2_norm.unsqueeze(0).to(self.device))
                        probs2 = torch.softmax(logits2, dim=1).cpu().numpy()[0]
                        confidences.append(float(probs2[orig_top]))

                # compute area under confidence curve normalized (higher drop -> higher faithfulness)
                conf_arr = np.array(confidences)
                # relative drop from orig_conf
                rel_drop = (orig_conf - conf_arr) / (orig_conf + 1e-12)
                faithfulness_score = np.mean(rel_drop)

                # stability: add small gaussian noise and compute cosine similarity between heatmaps
                noise_std = 0.01
                noisy_img = img_den + np.random.normal(0, noise_std, img_den.shape)
                # preprocess noisy image to tensor
                noisy_t = torch.from_numpy(np.transpose(np.clip(noisy_img, 0, 1), (2, 0, 1))).float()
                # call explainer on noisy input by writing a temporary wrapper that accepts tensor path? reuse _call_explainer by saving to temp file is expensive; instead, compute heatmap by calling explainer with in-memory tensor when supported
                try:
                    # Many explainers accept tensors; attempt direct calls
                    if m == 'shap':
                        out_noisy = self.explainers['shap'].explain(image=noisy_t, model=self.model, target_class=None)
                        hm_noisy = out_noisy.get('attributions')
                    elif m == 'gradcam':
                        with torch.no_grad():
                            logits_n = self.model(noisy_t.unsqueeze(0).to(self.device))
                            pred_n = int(torch.softmax(logits_n, dim=1).cpu().numpy()[0].argmax())
                        out_noisy = self.explainers['gradcam'].explain(image=noisy_t, model=self.model, predicted_class=pred_n, method='multiscale')
                        hm_noisy = out_noisy.get('heatmap')
                    elif m == 'lime':
                        # LIME expects numpy image HWC
                        noisy_img_np = np.transpose(noisy_t.numpy(), (1, 2, 0))
                        with torch.no_grad():
                            logits_n = self.model(t.unsqueeze(0).to(self.device))
                            pred_n = int(torch.softmax(logits_n, dim=1).cpu().numpy()[0].argmax())
                        out_noisy = self.explainers['lime'].explain(image=noisy_img_np, model=self.model, target_class=pred_n)
                        # LIME may return either a full HxW heatmap or per-superpixel weights + segments
                        hm_noisy = out_noisy.get('heatmap')
                        if hm_noisy is None:
                            # try to rasterize from weights + segments
                            weights = out_noisy.get('weights')
                            segments = out_noisy.get('segments')
                            if weights is not None and segments is not None:
                                hm_tmp = np.zeros_like(segments, dtype=float)
                                for sid, w in enumerate(weights):
                                    # focus on positive contributions
                                    hm_tmp[segments == sid] = max(w, 0.0)
                                # normalize
                                if hm_tmp.max() > 0:
                                    hm_tmp = (hm_tmp - hm_tmp.min()) / (hm_tmp.max() - hm_tmp.min() + 1e-12)
                                else:
                                    hm_tmp = np.zeros_like(hm_tmp, dtype=float)
                                hm_noisy = hm_tmp
                    else:
                        hm_noisy = None
                except Exception:
                    hm_noisy = None

                if hm_noisy is None:
                    stability = np.nan
                else:
                    hm_noisy = np.array(hm_noisy)
                    hm_noisy = np.squeeze(hm_noisy)
                    if hm_noisy.ndim > 2:
                        hm_noisy = np.mean(np.abs(hm_noisy), axis=tuple(range(hm_noisy.ndim - 2)))

                    # Resize noisy heatmap to (H, W) if needed
                    if hm_noisy.shape != (H, W):
                        try:
                            from PIL import Image
                            tmp = Image.fromarray((np.clip(hm_noisy, 0, 1) * 255).astype(np.uint8))
                            tmp = tmp.resize((W, H), resample=Image.BILINEAR)
                            hm_noisy = np.array(tmp).astype(float) / 255.0
                        except Exception:
                            hm_noisy = np.resize(hm_noisy, (H, W))

                    # normalize both
                    if hm_noisy.max() > 0:
                        hm_noisy = (hm_noisy - hm_noisy.min()) / (hm_noisy.max() - hm_noisy.min() + 1e-12)
                    else:
                        hm_noisy = np.zeros((H, W), dtype=float)

                    # cosine similarity
                    v1 = hm.flatten()
                    v2 = hm_noisy.flatten()
                    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                    stability = float(np.dot(v1, v2) / denom)

                # sparsity: fraction of pixels above 50th percentile -> lower fraction means more sparse
                thresh_sp = np.percentile(hm.flatten(), 50)
                frac = float((hm >= thresh_sp).sum()) / float(hm.size)
                sparsity = 1.0 - frac

                raw[m]['runtime'].append(runtime)
                raw[m]['fidelity'].append(fidelity)
                raw[m]['faithfulness'].append(faithfulness_score)
                raw[m]['stability'].append(stability)
                raw[m]['sparsity'].append(sparsity)

        # aggregate and normalize metrics across methods
        aggregated = {}
        metric_names = ['fidelity', 'faithfulness', 'stability', 'sparsity', 'runtime']

        # compute per-method means (ignore NaNs)
        means = {m: {} for m in self.methods}
        for m in self.methods:
            for metric in metric_names:
                vals = np.array(raw[m][metric], dtype=float)
                if vals.size == 0:
                    means[m][metric] = np.nan
                else:
                    means[m][metric] = float(np.nanmean(vals))

        # normalize each metric across methods to [0,1]
        normalized = {m: {} for m in self.methods}
        for metric in metric_names:
            vals = np.array([means[m][metric] for m in self.methods], dtype=float)
            # handle NaNs by ignoring
            valid = ~np.isnan(vals)
            if valid.sum() == 0:
                for m in self.methods:
                    normalized[m][metric] = float('nan')
                continue
            vmin = vals[valid].min()
            vmax = vals[valid].max()
            if vmax - vmin < 1e-12:
                # all equal -> set to 0.5 for valid entries
                for i, m in enumerate(self.methods):
                    normalized[m][metric] = 0.5 if valid[i] else float('nan')
            else:
                # for runtime, lower is better; invert after normalization
                for i, m in enumerate(self.methods):
                    if not valid[i]:
                        normalized[m][metric] = float('nan')
                        continue
                    val = vals[i]
                    norm = (val - vmin) / (vmax - vmin)
                    if metric == 'runtime':
                        norm = 1.0 - norm
                    normalized[m][metric] = float(norm)

        # Prepare final dict
        for m in self.methods:
            aggregated[m] = {
                'fidelity': normalized[m].get('fidelity', float('nan')),
                'faithfulness': normalized[m].get('faithfulness', float('nan')),
                'stability': normalized[m].get('stability', float('nan')),
                'sparsity': normalized[m].get('sparsity', float('nan')),
                'runtime': normalized[m].get('runtime', float('nan')),
            }

        return aggregated

    def generate_all_plots(self, results: Dict[str, Dict[str, float]], save_dir: str = 'results/xai_comparison') -> None:
        """Generate bar chart, radar chart, heatmap and line plot (optional) and save PNGs."""
        os.makedirs(save_dir, exist_ok=True)
        methods = list(results.keys())
        metrics = ['fidelity', 'faithfulness', 'stability', 'sparsity', 'runtime']

        # matrix methods x metrics
        mat = np.array([[results[m].get(metric, np.nan) for metric in metrics] for m in methods], dtype=float)

        # 1) Bar chart
        plt.figure(figsize=(10, 5))
        x = np.arange(len(metrics))
        width = 0.7 / len(methods)
        for i, m in enumerate(methods):
            plt.bar(x + i * width, mat[i], width=width, label=m)
        plt.xticks(x + width * (len(methods) - 1) / 2, metrics)
        plt.ylim(0, 1)
        plt.legend()
        plt.title('XAI methods comparison (bar chart)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bar_chart.png'), dpi=150)
        plt.close()

        # 2) Radar chart
        labels = metrics
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        for i, m in enumerate(methods):
            vals = mat[i].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, label=m)
            ax.fill(angles, vals, alpha=0.1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('XAI methods radar chart')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'radar_chart.png'), dpi=150)
        plt.close()

        # 3) Heatmap
        plt.figure(figsize=(6, 4))
        im = plt.imshow(mat, vmin=0, vmax=1, cmap='viridis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.yticks(range(len(methods)), methods)
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.title('XAI methods vs metrics (heatmap)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'heatmap.png'), dpi=150)
        plt.close()

        # 4) Line plot (performance vs subset_size) - optional placeholder
        # Save a small CSV for offline inspection
        import json
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved comparison plots to {save_dir}")
