"""
Training script for chest X-ray classification model.

Handles:
- Model training
- Validation
- Learning rate scheduling
- Model checkpointing
- Metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import logging
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import os
import json
from pathlib import Path
from datetime import datetime

from .cnn_model import ResNet50ChestXray, save_model
from utils.dataset_loader import ChestXrayDataLoader

logger = logging.getLogger(__name__)


class WarmupScheduler(LambdaLR):
    """Learning rate scheduler with linear warmup followed by cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr=0.001):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            base_lr: Base learning rate
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.cosine_epochs = total_epochs - warmup_epochs
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / self.cosine_epochs
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress))))
        
        super().__init__(optimizer, lr_lambda)


class Trainer:
    """Handles model training and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 2,
        num_epochs: int = 10,
        checkpoint_dir: str = 'models/checkpoints',
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train.
            device: Device to train on ('cuda' or 'cpu').
            learning_rate: Learning rate.
            weight_decay: Weight decay for optimizer.
            warmup_epochs: Number of warmup epochs.
            num_epochs: Total number of epochs to train.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss function for multi-class classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = WarmupScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            base_lr=learning_rate
        )
        
        # Metrics tracking
        self.train_history = {
            'loss': [],
            'epoch': [],
        }
        self.val_history = {
            'loss': [],
            'auroc': [],
            'epoch': [],
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info(f"Initialized Trainer on device: {device}")
    
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader.
            
        Returns:
            float: Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=True)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / num_batches})
        
        avg_loss = total_loss / num_batches
        logger.info(f"Training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation dataloader.
            
        Returns:
            tuple: (average_loss, auroc_score)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating', leave=True)
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store for AUROC calculation
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': total_loss / num_batches})
        
        avg_loss = total_loss / num_batches
        
        # Calculate AUROC
        import numpy as np
        from sklearn.metrics import roc_auc_score
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        try:
            auroc = roc_auc_score(all_labels, all_preds, average='macro')
        except Exception as e:
            logger.warning(f"Could not calculate AUROC: {e}")
            auroc = 0.0
        
        logger.info(f"Validation loss: {avg_loss:.4f}, AUROC: {auroc:.4f}")
        
        return avg_loss, auroc
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            num_epochs: Number of epochs to train. If None, use self.num_epochs.
            
        Returns:
            dict: Training history.
        """
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_history['loss'].append(train_loss)
            self.train_history['epoch'].append(epoch + 1)
            
            # Validate
            val_loss, auroc = self.validate(val_loader)
            self.val_history['loss'].append(val_loss)
            self.val_history['auroc'].append(auroc)
            self.val_history['epoch'].append(epoch + 1)
            
            # Step scheduler
            self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                
                best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                save_model(
                    self.model,
                    best_model_path,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metrics={
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'auroc': auroc,
                    }
                )
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'checkpoint_epoch_{epoch + 1}.pth'
                )
                save_model(
                    self.model,
                    checkpoint_path,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                )
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Training completed!")
        logger.info(f"Best model saved at epoch {self.best_epoch} with loss {self.best_val_loss:.4f}")
        logger.info(f"{'='*50}")
        
        return {
            'train': self.train_history,
            'val': self.val_history,
        }
    
    def get_history(self) -> Dict[str, dict]:
        """Get training history."""
        return {
            'train': self.train_history,
            'val': self.val_history,
        }
    
    def save_history(self, save_path: str) -> None:
        """Save training history to JSON."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.get_history(), f, indent=2)
        
        logger.info(f"Saved training history to {save_path}")


def train_model(
    config: dict,
    model: nn.Module,
    train_loader,
    val_loader,
    device: str = 'cuda',
) -> Trainer:
    """
    High-level function to train model.
    
    Args:
        config: Configuration dictionary.
        model: Model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        device: Device to train on.
        
    Returns:
        Trainer: Trainer instance with training history.
    """
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5),
        warmup_epochs=config.get('warmup_epochs', 2),
        num_epochs=config.get('num_epochs', 10),
        checkpoint_dir=config.get('models_dir', 'models/checkpoints'),
    )
    
    trainer.train(train_loader, val_loader)
    
    return trainer
