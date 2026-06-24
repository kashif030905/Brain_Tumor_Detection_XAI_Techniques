"""
CNN model for multi-label chest X-ray classification.

Uses pretrained ResNet50 with modifications for 14-class multi-label classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class ResNet50ChestXray(nn.Module):
    """
    ResNet50 model adapted for multi-label chest X-ray classification.
    
    Uses pretrained ImageNet weights and modifies the final layer
    for 14 disease classes.
    """
    
    def __init__(self, num_classes: int = 14, dropout_rate: float = 0.5):
        """
        Initialize model.
        
        Args:
            num_classes: Number of output classes (diseases). Default: 14.
            dropout_rate: Dropout rate before final layer.
        """
        super(ResNet50ChestXray, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # Disable inplace operations in ReLU for SHAP compatibility
        for module in self.resnet.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        
        # Remove the original fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Add custom head for multi-label classification
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)
        
        logger.info(f"Initialized ResNet50 for {num_classes}-class multi-label classification")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
                         Note: No sigmoid applied here (use BCEWithLogitsLoss).
        """
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from ResNet backbone (before final FC layer).
        
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            
        Returns:
            torch.Tensor: Feature tensor of shape (B, 2048).
        """
        # Pass through all layers except fc
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sigmoid probabilities from model output.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            
        Returns:
            torch.Tensor: Probabilities of shape (B, num_classes) in range [0, 1].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


def get_model(
    num_classes: int = 14,
    dropout_rate: float = 0.5,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Create and return the model.
    
    Args:
        num_classes: Number of output classes.
        dropout_rate: Dropout rate.
        device: Device to place model on ('cuda' or 'cpu'). If None, auto-detect.
        
    Returns:
        nn.Module: Model instance.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ResNet50ChestXray(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    
    return model


def load_model(
    model_path: str,
    num_classes: int = 14,
    dropout_rate: float = 0.5,
    device: Optional[str] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate.
        device: Device to place model on. If None, auto-detect.
        strict: Whether to strictly enforce that all keys match.
        
    Returns:
        nn.Module: Loaded model.
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ResNet50ChestXray(num_classes=num_classes, dropout_rate=dropout_rate)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both direct state dict and checkpoint dict with 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    
    return model


def save_model(
    model: nn.Module,
    save_path: str,
    optimizer=None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save.
        save_path: Path to save checkpoint.
        optimizer: Optional optimizer to save.
        epoch: Optional epoch number.
        metrics: Optional metrics dictionary.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': 'ResNet50ChestXray',
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved model checkpoint to {save_path}")


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze ResNet backbone parameters (for transfer learning).
    
    Args:
        model: Model to freeze.
    """
    for param in model.resnet.parameters():
        param.requires_grad = False
    
    logger.info("Froze ResNet backbone parameters")


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze ResNet backbone parameters.
    
    Args:
        model: Model to unfreeze.
    """
    for param in model.resnet.parameters():
        param.requires_grad = True
    
    logger.info("Unfroze ResNet backbone parameters")


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary of model architecture.
    
    Args:
        model: Model to summarize.
        
    Returns:
        str: Model summary.
    """
    summary = f"""
Model: {model.__class__.__name__}
Number of classes: {model.num_classes}
Total parameters: {sum(p.numel() for p in model.parameters()):,}
Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
"""
    return summary
