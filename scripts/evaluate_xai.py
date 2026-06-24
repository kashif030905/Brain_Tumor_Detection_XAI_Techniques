"""CLI script to evaluate XAI explainers on a subset of test images.

Writes a bar plot to `experiments/results/xai_accuracies.png` and prints numeric results.

Usage:
    python scripts/evaluate_xai.py --config configs/config.yaml --model_path models/checkpoints/best_model.pth --subset_size 100
"""

import argparse
import yaml
import os
from pathlib import Path
import torch
import numpy as np

from evaluation.xai_explainers_eval import XAIExplainersEvaluator, plot_accuracies
from models.cnn_model import load_model
from xai.shap_explainer import SHAPExplainer
from xai.gradcam_enhanced import EnhancedGradCAM
from xai.lime_explainer import LIMEExplainer
from utils.preprocessing import preprocess_image
from utils.mri_dataset_loader import MRIDataLoader


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--subset_size', type=int, default=100)
    parser.add_argument('--keep_fraction', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    config = load_config(args.config)

    # load model
    model = load_model(args.model_path, num_classes=config['dataset']['num_classes'], device=args.device)

    # Prepare dataset list (test folder)
    dataset_dir = config['dataset']['path']
    test_folder = os.path.join(dataset_dir, config['dataset'].get('test_folder', 'Testing'))

    image_paths = []
    for class_name in config['dataset']['classes']:
        folder = os.path.join(test_folder, class_name)
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(folder, fn))

    image_paths = image_paths[:args.subset_size]

    # Prepare a background loader for SHAP
    data_loader = MRIDataLoader(dataset_dir=config['dataset']['path'], image_size=config['dataset']['image_size'])
    background_loader = data_loader.get_background_loader(
        num_samples=config['shap']['background_size'],
        batch_size=config['validation']['batch_size'],
        num_workers=config['validation']['num_workers'],
    )

    # Initialize explainers (wrappers that return {'heatmap': HxW})
    shap_explainer = SHAPExplainer(background_loader=background_loader, num_samples=config['shap']['background_size'], device=args.device)
    gradcam_explainer = EnhancedGradCAM(device=args.device)
    lime_explainer = LIMEExplainer(device=args.device, num_samples=150, num_features=50)

    def shap_wrapper(img_path):
        # preprocess
        t = preprocess_image(img_path, image_size=config['dataset']['image_size'])
        out = shap_explainer.explain(image=t, model=model, target_class=None)
        heat = out.get('attributions')
        # SHAP attributions might be CxHxW, reduce to HxW by sum(abs)
        if heat is None:
            return {'heatmap': None}
        if heat.ndim == 3:
            heatmap = np.sum(np.abs(heat), axis=0)
        else:
            heatmap = np.abs(heat)
        # normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return {'heatmap': heatmap}

    def gradcam_wrapper(img_path):
        t = preprocess_image(img_path, image_size=config['dataset']['image_size'])
        # get predicted class from model
        with torch.no_grad():
            logits = model(t.unsqueeze(0).to(args.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_cls = int(probs.argmax())

        preds = gradcam_explainer.explain(image=t, model=model, predicted_class=pred_cls, method='multiscale')
        hm = preds.get('heatmap')
        if hm is None:
            return {'heatmap': None}
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        return {'heatmap': hm}

    def lime_wrapper(img_path):
        # use denormalized np image for LIME explainer
        # preprocess -> denorm
        t = preprocess_image(img_path, image_size=config['dataset']['image_size'])
        # denormalize
        from utils.preprocessing import denormalize_image
        img_den = denormalize_image(t).cpu().numpy()
        img_den = np.transpose(img_den, (1, 2, 0))
        # determine predicted class
        with torch.no_grad():
            logits = model(t.unsqueeze(0).to(args.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_cls = int(probs.argmax())

        out = lime_explainer.explain(image=img_den, model=model, target_class=pred_cls)
        hm = out.get('heatmap')
        if hm is None:
            return {'heatmap': None}
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        return {'heatmap': hm}

    explainer_callables = {
        'SHAP': shap_wrapper,
        'GradCAM': gradcam_wrapper,
        'LIME': lime_wrapper,
    }

    evaluator = XAIExplainersEvaluator(model=model, preprocess_fn=lambda p: preprocess_image(p, image_size=config['dataset']['image_size']), device=args.device)

    accuracies = evaluator.evaluate_on_subset(
        image_paths=image_paths,
        explainer_callables=explainer_callables,
        keep_fraction=args.keep_fraction,
        max_samples=args.subset_size,
    )

    print('\nAccuracies:')
    for k, v in accuracies.items():
        print(f'  {k}: {v:.3f}')

    plot_accuracies(accuracies, save_path=os.path.join(config['paths']['results_dir'], 'xai_accuracies.png'))
    print(f"Saved plot to {os.path.join(config['paths']['results_dir'], 'xai_accuracies.png')}")

    # --- Run comprehensive XAI comparison evaluator (new module)
    try:
        from evaluation.xai_comparison import XAIComparisonEvaluator

        print('\nRunning full XAI comparison evaluator (this may take time)...')
        comp_eval = XAIComparisonEvaluator(
            model=model,
            image_paths=image_paths,
            methods=['lime', 'shap', 'gradcam'],
            device=args.device,
            subset_size=args.subset_size,
            keep_fraction=args.keep_fraction,
            background_loader=background_loader,
        )

        results = comp_eval.evaluate_all_methods()
        print('\nComparison results:')
        for k, v in results.items():
            print(f"{k}: {v}")

        comp_eval.generate_all_plots(results, save_dir=os.path.join(config['paths']['results_dir'], 'xai_comparison'))

    except Exception as e:
        print(f"Could not run XAIComparisonEvaluator: {e}")


if __name__ == '__main__':
    main()
