"""
MouseReach Training Pipeline

Training infrastructure for MouseReach behavior analysis models.

Modules:
    config: Configuration classes for paths, hyperparameters, and features
    feature_extraction: Feature engineering from DLC pose data
    data_loader: PyTorch Dataset classes for training
    models: Neural network architectures (TCN, Transformer, BiLSTM-CRF, MLP)
    train: Training loops with focal loss, early stopping, and logging

Usage:
    # Train all models
    python -m training.train --task all --epochs 100

    # Train specific model
    python -m training.train --task outcome --epochs 50

    # Programmatic usage
    from training.config import get_config
    from training.data_loader import MouseReachDataModule
    from training.models import OutcomeClassifier
    from training.train import OutcomeTrainer

Environment: Y:\\2_Connectome\\envs\\mousereach
"""

__version__ = "1.0.0"
__author__ = "MouseReach Pipeline"

# Key exports
from .config import Config, get_config
from .feature_extraction import FeatureExtractor, ExtractedFeatures
from .data_loader import MouseReachDataModule, BoundaryDataset, ReachSequenceDataset, OutcomeDataset
from .models import (
    create_boundary_detector,
    BoundaryDetectorTCN,
    BoundaryDetectorTransformer,
    ReachDetector,
    OutcomeClassifier,
    ModelFactory
)
from .train import (
    FocalLoss,
    BoundaryTrainer,
    ReachTrainer,
    OutcomeTrainer,
    train_boundary_detector,
    train_reach_detector,
    train_outcome_classifier
)

__all__ = [
    # Config
    "Config",
    "get_config",

    # Feature Extraction
    "FeatureExtractor",
    "ExtractedFeatures",

    # Data Loading
    "MouseReachDataModule",
    "BoundaryDataset",
    "ReachSequenceDataset",
    "OutcomeDataset",

    # Models
    "create_boundary_detector",
    "BoundaryDetectorTCN",
    "BoundaryDetectorTransformer",
    "ReachDetector",
    "OutcomeClassifier",
    "ModelFactory",

    # Training
    "FocalLoss",
    "BoundaryTrainer",
    "ReachTrainer",
    "OutcomeTrainer",
    "train_boundary_detector",
    "train_reach_detector",
    "train_outcome_classifier",
]
