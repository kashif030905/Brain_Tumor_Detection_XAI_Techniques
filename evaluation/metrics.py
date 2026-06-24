"""
Evaluation metrics for multi-label classification.

Provides metrics for assessing model performance on chest X-ray prediction.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve, accuracy_score,
    f1_score, hamming_loss, confusion_matrix
)
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class MultiLabelMetrics:
    """Compute metrics for multi-label classification."""
    
    @staticmethod
    def compute_auroc(
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = 'macro',
    ) -> float:
        """
        Compute Area Under ROC curve.
        
        Args:
            predictions: Predicted probabilities (N, num_classes).
            labels: Ground truth labels (N, num_classes).
            average: Averaging method ('macro', 'micro', 'weighted').
            
        Returns:
            float: AUROC score.
        """
        try:
            auroc = roc_auc_score(labels, predictions, average=average)
            return auroc
        except Exception as e:
            logger.warning(f"Could not compute AUROC: {e}")
            return 0.0
    
    @staticmethod
    def compute_auprc(
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = 'macro',
    ) -> float:
        """
        Compute Area Under Precision-Recall curve.
        
        Args:
            predictions: Predicted probabilities (N, num_classes).
            labels: Ground truth labels (N, num_classes).
            average: Averaging method.
            
        Returns:
            float: AUPRC score.
        """
        try:
            # For each class, compute AUPRC
            auprc_scores = []
            
            for i in range(predictions.shape[1]):
                precision, recall, _ = precision_recall_curve(
                    labels[:, i], predictions[:, i]
                )
                auprc = auc(recall, precision)
                auprc_scores.append(auprc)
            
            if average == 'macro':
                return np.mean(auprc_scores)
            elif average == 'micro':
                # Flatten for micro average
                precision, recall, _ = precision_recall_curve(
                    labels.ravel(), predictions.ravel()
                )
                return auc(recall, precision)
            else:
                return np.mean(auprc_scores)
        
        except Exception as e:
            logger.warning(f"Could not compute AUPRC: {e}")
            return 0.0
    
    @staticmethod
    def compute_f1_score(
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
        average: str = 'macro',
    ) -> float:
        """
        Compute F1 score.
        
        Args:
            predictions: Predicted probabilities (N, num_classes).
            labels: Ground truth labels (N, num_classes).
            threshold: Threshold for binarizing predictions.
            average: Averaging method ('macro', 'micro', 'weighted').
            
        Returns:
            float: F1 score.
        """
        binary_predictions = (predictions >= threshold).astype(int)
        
        try:
            f1 = f1_score(labels, binary_predictions, average=average, zero_division=0)
            return f1
        except Exception as e:
            logger.warning(f"Could not compute F1 score: {e}")
            return 0.0
    
    @staticmethod
    def compute_accuracy(
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute accuracy (subset accuracy for multi-label).
        
        Args:
            predictions: Predicted probabilities (N, num_classes).
            labels: Ground truth labels (N, num_classes).
            threshold: Threshold for binarizing predictions.
            
        Returns:
            float: Accuracy score.
        """
        binary_predictions = (predictions >= threshold).astype(int)
        
        try:
            accuracy = accuracy_score(labels, binary_predictions)
            return accuracy
        except Exception as e:
            logger.warning(f"Could not compute accuracy: {e}")
            return 0.0
    
    @staticmethod
    def compute_hamming_loss(
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute Hamming loss (fraction of incorrect labels).
        
        Args:
            predictions: Predicted probabilities (N, num_classes).
            labels: Ground truth labels (N, num_classes).
            threshold: Threshold for binarizing predictions.
            
        Returns:
            float: Hamming loss.
        """
        binary_predictions = (predictions >= threshold).astype(int)
        
        try:
            loss = hamming_loss(labels, binary_predictions)
            return loss
        except Exception as e:
            logger.warning(f"Could not compute Hamming loss: {e}")
            return 0.0
    
    @staticmethod
    def compute_confusion_matrices(
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Compute confusion matrices for each class.
        
        Args:
            predictions: Predicted probabilities (N, num_classes).
            labels: Ground truth labels (N, num_classes).
            threshold: Threshold for binarizing predictions.
            
        Returns:
            np.ndarray: Array of confusion matrices (num_classes, 2, 2).
        """
        binary_predictions = (predictions >= threshold).astype(int)
        num_classes = predictions.shape[1]
        
        cms = []
        for i in range(num_classes):
            cm = confusion_matrix(labels[:, i], binary_predictions[:, i])
            cms.append(cm)
        
        return np.stack(cms, axis=0)
    
    @staticmethod
    def compute_per_class_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        disease_classes: List[str],
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each disease class individually.
        
        Args:
            predictions: Predicted probabilities (N, num_classes).
            labels: Ground truth labels (N, num_classes).
            disease_classes: List of disease class names.
            threshold: Threshold for binarizing predictions.
            
        Returns:
            dict: Metrics for each class.
        """
        binary_predictions = (predictions >= threshold).astype(int)
        
        per_class_metrics = {}
        
        for i, disease in enumerate(disease_classes):
            try:
                auroc = roc_auc_score(labels[:, i], predictions[:, i])
            except:
                auroc = 0.0
            
            try:
                precision, recall, _ = precision_recall_curve(labels[:, i], predictions[:, i])
                auprc = auc(recall, precision)
            except:
                auprc = 0.0
            
            f1 = f1_score(labels[:, i], binary_predictions[:, i], zero_division=0)
            cm = confusion_matrix(labels[:, i], binary_predictions[:, i])
            
            per_class_metrics[disease] = {
                'auroc': auroc,
                'auprc': auprc,
                'f1': f1,
                'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
                'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
                'sensitivity': cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
                'specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
            }
        
        return per_class_metrics


def get_prediction_confidence(predictions: np.ndarray) -> np.ndarray:
    """
    Get confidence of predictions (max probability per sample).
    
    Args:
        predictions: Predicted probabilities (N, num_classes).
        
    Returns:
        np.ndarray: Maximum confidence per sample (N,).
    """
    return predictions.max(axis=1)


def sensitivity_analysis(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    perturbation: float = 0.1,
    num_perturbations: int = 100,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Analyze model sensitivity to input perturbations.
    
    Args:
        model: PyTorch model.
        image: Input image tensor.
        target_class: Target class to analyze.
        perturbation: Magnitude of perturbation.
        num_perturbations: Number of perturbations to test.
        device: Device to use.
        
    Returns:
        dict: Sensitivity metrics.
    """
    model.eval()
    image = image.to(device)
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_logits = model(image.unsqueeze(0) if image.dim() == 3 else image)
        baseline_pred = torch.sigmoid(baseline_logits)[0, target_class].item()
    
    # Add perturbations
    differences = []
    
    for _ in range(num_perturbations):
        noise = torch.randn_like(image) * perturbation
        perturbed = torch.clamp(image + noise, 0, 1)
        
        with torch.no_grad():
            perturbed_logits = model(perturbed.unsqueeze(0) if perturbed.dim() == 3 else perturbed)
            perturbed_pred = torch.sigmoid(perturbed_logits)[0, target_class].item()
        
        differences.append(abs(baseline_pred - perturbed_pred))
    
    differences = np.array(differences)
    
    return {
        'mean_sensitivity': float(differences.mean()),
        'std_sensitivity': float(differences.std()),
        'max_sensitivity': float(differences.max()),
        'min_sensitivity': float(differences.min()),
        'baseline_prediction': baseline_pred,
    }
