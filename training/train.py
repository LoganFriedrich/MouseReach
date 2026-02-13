"""
Training Script for MouseReach Deep Learning Pipeline.

Provides:
- Focal loss for class imbalance
- Session-level train/val split (no data leakage)
- Early stopping with patience
- Learning rate scheduling
- Comprehensive metric logging to CSV
- Checkpointing best models

Usage:
    python train.py --task boundary --epochs 100
    python train.py --task reach --epochs 100
    python train.py --task outcome --epochs 100

Environment: Y:\\2_Connectome\\envs\\mousereach
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)

from config import Config, get_config, TrainingConfig
from data_loader import MouseReachDataModule
from models import (
    create_boundary_detector, ReachDetector, OutcomeClassifier,
    ModelFactory, count_parameters
)


# ==============================================================================
# FOCAL LOSS
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where p_t is the probability of the correct class.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights tensor of shape (n_classes,)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C) or (N, T, C) for sequences
            targets: Labels of shape (N,) or (N, T) for sequences

        Returns:
            Focal loss
        """
        # Flatten if needed
        if inputs.dim() == 3:
            N, T, C = inputs.shape
            inputs = inputs.view(-1, C)
            targets = targets.view(-1)

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Get probability of correct class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weights
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==============================================================================
# METRICS TRACKER
# ==============================================================================

class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV file
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.csv_file = None
        self.csv_writer = None

        # Metrics history
        self.history: Dict[str, List[float]] = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': [],
            'epoch_time': []
        }

    def start_logging(self, header: Optional[List[str]] = None) -> None:
        """Initialize CSV logging."""
        if header is None:
            header = list(self.history.keys())

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(header)
        self.csv_file.flush()

    def log_epoch(self, metrics: Dict[str, float]) -> None:
        """Log metrics for one epoch."""
        # Update history
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        # Write to CSV
        if self.csv_writer:
            row = [metrics.get(key, '') for key in self.history.keys()]
            self.csv_writer.writerow(row)
            self.csv_file.flush()

    def close(self) -> None:
        """Close log file."""
        if self.csv_file:
            self.csv_file.close()

    def get_best_epoch(self, metric: str = 'val_f1', mode: str = 'max') -> int:
        """Get epoch with best metric value."""
        if metric not in self.history or not self.history[metric]:
            return 0

        values = self.history[metric]
        if mode == 'max':
            return int(np.argmax(values))
        else:
            return int(np.argmin(values))


# ==============================================================================
# EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' or 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'max':
            is_improvement = value > self.best_value + self.min_delta
        else:
            is_improvement = value < self.best_value - self.min_delta

        if is_improvement:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ==============================================================================
# TRAINER BASE CLASS
# ==============================================================================

class Trainer:
    """Base trainer class with common functionality."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        experiment_name: Optional[str] = None
    ):
        self.config = config or get_config().training
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"{model.__class__.__name__}_{timestamp}"
        self.experiment_name = experiment_name

        # Directories
        paths_config = get_config().paths
        self.log_dir = paths_config.logs_dir
        self.checkpoint_dir = paths_config.checkpoints_dir

        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Metrics tracker
        self.metrics = MetricsTracker(self.log_dir, experiment_name)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode=self.config.checkpoint_mode
        )

        # Best model state
        self.best_model_state = None
        self.best_metric = None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.checkpoint_mode,
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_gamma
            )
        else:
            return None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Override in subclass."""
        raise NotImplementedError

    def validate(self) -> Dict[str, float]:
        """Validate model. Override in subclass."""
        raise NotImplementedError

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Dictionary with training results
        """
        num_epochs = num_epochs or self.config.num_epochs

        print(f"\n{'=' * 60}")
        print(f"Training {self.experiment_name}")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"{'=' * 60}\n")

        # Start logging
        self.metrics.start_logging()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Epoch time
            epoch_time = time.time() - epoch_start

            # Current learning rate
            lr = self.optimizer.param_groups[0]['lr']

            # Combine metrics
            metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_acc': train_metrics.get('accuracy', 0),
                'val_acc': val_metrics.get('accuracy', 0),
                'val_precision': val_metrics.get('precision', 0),
                'val_recall': val_metrics.get('recall', 0),
                'val_f1': val_metrics.get('f1', 0),
                'learning_rate': lr,
                'epoch_time': epoch_time
            }

            # Log metrics
            self.metrics.log_epoch(metrics)

            # Print progress
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {metrics['train_loss']:.4f} | "
                f"Val Loss: {metrics['val_loss']:.4f} | "
                f"Val F1: {metrics['val_f1']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(metrics[self.config.checkpoint_metric])
                else:
                    self.scheduler.step()

            # Check for best model
            current_metric = metrics[self.config.checkpoint_metric]
            is_best = False

            if self.best_metric is None:
                is_best = True
            elif self.config.checkpoint_mode == 'max':
                is_best = current_metric > self.best_metric
            else:
                is_best = current_metric < self.best_metric

            if is_best:
                self.best_metric = current_metric
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, is_best=True)
                print(f"  -> New best {self.config.checkpoint_metric}: {current_metric:.4f}")

            # Early stopping
            if self.early_stopping(current_metric):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # Close logging
        self.metrics.close()

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        # Return results
        return {
            'experiment_name': self.experiment_name,
            'best_epoch': self.metrics.get_best_epoch(self.config.checkpoint_metric),
            'best_metric': self.best_metric,
            'history': self.metrics.history
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__
        }

        # Save latest
        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_metric = checkpoint.get('best_metric')


# ==============================================================================
# BOUNDARY TRAINER
# ==============================================================================

class BoundaryTrainer(Trainer):
    """Trainer for boundary detection task."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        config: Optional[TrainingConfig] = None,
        experiment_name: Optional[str] = None
    ):
        super().__init__(model, train_loader, val_loader, config, experiment_name)

        # Loss function
        if self.config.use_focal_loss:
            alpha = class_weights if class_weights is not None else None
            self.criterion = FocalLoss(
                alpha=alpha,
                gamma=self.config.focal_loss_gamma
            )
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        if class_weights is not None:
            self.criterion.alpha = class_weights.to(self.device) if hasattr(self.criterion, 'alpha') else None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            self.optimizer.step()

            total_loss += loss.item()

            # Collect predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )

        # ROC-AUC
        try:
            all_probs = np.array(all_probs)
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            roc_auc = 0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }


# ==============================================================================
# REACH TRAINER
# ==============================================================================

class ReachTrainer(Trainer):
    """Trainer for reach detection (sequence labeling) task."""

    def __init__(
        self,
        model: ReachDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        config: Optional[TrainingConfig] = None,
        experiment_name: Optional[str] = None
    ):
        super().__init__(model, train_loader, val_loader, config, experiment_name)
        self.class_weights = class_weights

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass (model returns loss when tags provided)
            self.optimizer.zero_grad()
            loss = self.model(inputs, tags=labels)

            # Backward pass
            loss.backward()

            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            self.optimizer.step()
            total_loss += loss.item()

            # Get predictions for accuracy
            with torch.no_grad():
                _, decoded = self.model(inputs)
                for pred_seq, label_seq in zip(decoded, labels):
                    all_preds.extend(pred_seq[:len(label_seq)])
                    all_labels.extend(label_seq.cpu().numpy()[:len(pred_seq)])

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Get loss
                loss = self.model(inputs, tags=labels)
                total_loss += loss.item()

                # Get predictions
                _, decoded = self.model(inputs)
                for pred_seq, label_seq in zip(decoded, labels):
                    all_preds.extend(pred_seq[:len(label_seq)])
                    all_labels.extend(label_seq.cpu().numpy()[:len(pred_seq)])

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        # Compute per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# ==============================================================================
# OUTCOME TRAINER
# ==============================================================================

class OutcomeTrainer(Trainer):
    """Trainer for outcome classification task."""

    def __init__(
        self,
        model: OutcomeClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        config: Optional[TrainingConfig] = None,
        experiment_name: Optional[str] = None
    ):
        super().__init__(model, train_loader, val_loader, config, experiment_name)

        # Loss function
        if self.config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=self.config.focal_loss_gamma
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None
            )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()

            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            self.optimizer.step()
            total_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def get_classification_report(self, loader: DataLoader) -> str:
        """Generate detailed classification report."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # Get class names from config
        config = get_config()
        class_names = config.outcome_classifier.outcome_classes

        return classification_report(
            all_labels, all_preds,
            target_names=class_names,
            zero_division=0
        )


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_boundary_detector(
    config: Optional[Config] = None,
    num_epochs: Optional[int] = None
) -> Dict[str, Any]:
    """Train boundary detection model."""
    config = config or get_config()

    print("\n" + "=" * 60)
    print("BOUNDARY DETECTION TRAINING")
    print("=" * 60)

    # Setup data
    dm = MouseReachDataModule(config)
    dm.setup(task='boundary')

    train_loader, val_loader, test_loader = dm.get_boundary_dataloaders()

    # Get input size from first batch
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    print(f"Input size: {input_size}")

    # Create model
    model = create_boundary_detector(input_size, config.boundary_detector)

    # Get class weights
    class_weights = train_loader.dataset.get_class_weights()
    print(f"Class weights: {class_weights}")

    # Create trainer
    trainer = BoundaryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config.training,
        experiment_name=f"boundary_{config.boundary_detector.model_type}"
    )

    # Train
    results = trainer.train(num_epochs)

    # Test evaluation
    print("\n" + "=" * 40)
    print("Test Set Evaluation")
    print("=" * 40)

    # Swap val_loader for test_loader
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")

    results['test_metrics'] = test_metrics
    return results


def train_reach_detector(
    config: Optional[Config] = None,
    num_epochs: Optional[int] = None
) -> Dict[str, Any]:
    """Train reach detection model."""
    config = config or get_config()

    print("\n" + "=" * 60)
    print("REACH DETECTION TRAINING")
    print("=" * 60)

    # Setup data
    dm = MouseReachDataModule(config)
    dm.setup(task='reach')

    train_loader, val_loader, test_loader = dm.get_reach_dataloaders()

    # Get input size
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    print(f"Input size: {input_size}")

    # Create model
    model = ReachDetector(input_size, config.reach_detector)

    # Get class weights
    class_weights = train_loader.dataset.get_class_weights()
    print(f"Class weights: {class_weights}")

    # Create trainer
    trainer = ReachTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config.training,
        experiment_name="reach_bilstm_crf"
    )

    # Train
    results = trainer.train(num_epochs)

    # Test evaluation
    print("\n" + "=" * 40)
    print("Test Set Evaluation")
    print("=" * 40)

    trainer.val_loader = test_loader
    test_metrics = trainer.validate()

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")

    results['test_metrics'] = test_metrics
    return results


def train_outcome_classifier(
    config: Optional[Config] = None,
    num_epochs: Optional[int] = None
) -> Dict[str, Any]:
    """Train outcome classification model."""
    config = config or get_config()

    print("\n" + "=" * 60)
    print("OUTCOME CLASSIFICATION TRAINING")
    print("=" * 60)

    # Setup data
    dm = MouseReachDataModule(config)
    dm.setup(task='outcome')

    train_loader, val_loader, test_loader = dm.get_outcome_dataloaders()

    # Get input size and class distribution
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    print(f"Input size: {input_size}")

    class_dist = train_loader.dataset.get_class_distribution()
    print(f"Class distribution: {class_dist}")

    # Create model
    model = OutcomeClassifier(input_size, config.outcome_classifier)

    # Get class weights
    class_weights = train_loader.dataset.get_class_weights()
    print(f"Class weights: {class_weights}")

    # Create trainer
    trainer = OutcomeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config.training,
        experiment_name="outcome_mlp"
    )

    # Train
    results = trainer.train(num_epochs)

    # Test evaluation
    print("\n" + "=" * 40)
    print("Test Set Evaluation")
    print("=" * 40)

    trainer.val_loader = test_loader
    test_metrics = trainer.validate()

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(trainer.get_classification_report(test_loader))

    results['test_metrics'] = test_metrics
    return results


# ==============================================================================
# CLI
# ==============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train MouseReach behavior analysis models'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['boundary', 'reach', 'outcome', 'all'],
        default='all',
        help='Task to train (default: all)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: from config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: from config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default: from config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu)'
    )

    args = parser.parse_args()

    # Load config
    config = get_config()

    # Override config with CLI arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.training.device = args.device

    # Set random seeds
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # Train
    results = {}

    if args.task in ['boundary', 'all']:
        try:
            results['boundary'] = train_boundary_detector(config, args.epochs)
        except Exception as e:
            print(f"Boundary training failed: {e}")
            import traceback
            traceback.print_exc()

    if args.task in ['reach', 'all']:
        try:
            results['reach'] = train_reach_detector(config, args.epochs)
        except Exception as e:
            print(f"Reach training failed: {e}")
            import traceback
            traceback.print_exc()

    if args.task in ['outcome', 'all']:
        try:
            results['outcome'] = train_outcome_classifier(config, args.epochs)
        except Exception as e:
            print(f"Outcome training failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    for task, result in results.items():
        if result:
            print(f"\n{task.upper()}:")
            print(f"  Best epoch: {result.get('best_epoch', 'N/A')}")
            print(f"  Best metric: {result.get('best_metric', 'N/A'):.4f}" if result.get('best_metric') else "  Best metric: N/A")
            if 'test_metrics' in result:
                print(f"  Test F1: {result['test_metrics'].get('f1', 'N/A'):.4f}" if isinstance(result['test_metrics'].get('f1'), (int, float)) else "  Test F1: N/A")


if __name__ == "__main__":
    main()
