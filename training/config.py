"""
Configuration for MouseReach Deep Learning Pipeline.

This module contains all paths, hyperparameters, and feature settings
for training boundary detection, reach detection, and outcome classification models.

Environment: Y:\\2_Connectome\\envs\\mousereach
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

@dataclass
class PathConfig:
    """Paths for data, models, and outputs."""

    # Base directories
    base_dir: Path = Path(r"Y:\2_Connectome\Behavior\MouseReach")
    processing_dir: Path = field(default_factory=lambda: Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing"))
    training_dir: Path = field(default_factory=lambda: Path(r"Y:\2_Connectome\Behavior\MouseReach\training"))

    # Output directories
    models_dir: Path = field(default_factory=lambda: Path(r"Y:\2_Connectome\Behavior\MouseReach\training\models"))
    logs_dir: Path = field(default_factory=lambda: Path(r"Y:\2_Connectome\Behavior\MouseReach\training\logs"))
    checkpoints_dir: Path = field(default_factory=lambda: Path(r"Y:\2_Connectome\Behavior\MouseReach\training\checkpoints"))

    # Python environment
    python_exe: Path = Path(r"Y:\2_Connectome\envs\mousereach\python.exe")

    def __post_init__(self):
        """Create output directories if they don't exist."""
        for attr in ['models_dir', 'logs_dir', 'checkpoints_dir']:
            path = getattr(self, attr)
            path.mkdir(parents=True, exist_ok=True)

    @property
    def dlc_pattern(self) -> str:
        """Glob pattern for DLC h5 files."""
        return "*DLC_resnet50_MPSAOct27shuffle1_100000.h5"

    @property
    def unified_gt_pattern(self) -> str:
        """Glob pattern for unified ground truth files."""
        return "*_unified_ground_truth.json"

    @property
    def seg_gt_pattern(self) -> str:
        """Glob pattern for segmentation ground truth files."""
        return "*_seg_ground_truth.json"

    @property
    def outcome_gt_pattern(self) -> str:
        """Glob pattern for outcome ground truth files."""
        return "*_outcome_ground_truth.json"


# ==============================================================================
# DLC BODY PART CONFIGURATION
# ==============================================================================

@dataclass
class DLCConfig:
    """Configuration for DeepLabCut data structure."""

    # Scorer name (model identifier)
    scorer: str = "DLC_resnet50_MPSAOct27shuffle1_100000"

    # Body parts tracked (18 total)
    body_parts: List[str] = field(default_factory=lambda: [
        "Reference", "SATL", "SABL", "SABR", "SATR",
        "BOXL", "BOXR", "Pellet", "Pillar",
        "RightHand", "RHLeft", "RHOut", "RHRight",
        "Nose", "RightEar", "LeftEar", "LeftFoot", "TailBase"
    ])

    # Hand markers for ensemble averaging (critical due to low likelihood)
    hand_parts: List[str] = field(default_factory=lambda: [
        "RightHand", "RHLeft", "RHRight", "RHOut"
    ])

    # Task-critical body parts
    task_critical_parts: List[str] = field(default_factory=lambda: [
        "RightHand", "RHLeft", "RHRight", "RHOut",  # Hand
        "Pellet", "Pillar",                          # Task objects
        "Nose"                                       # Head reference
    ])

    # Static reference landmarks (high likelihood, can skip filtering)
    static_landmarks: List[str] = field(default_factory=lambda: [
        "Reference", "BOXL", "BOXR", "SATL", "SATR", "SABL", "SABR"
    ])

    # Coordinate columns
    coords: List[str] = field(default_factory=lambda: ["x", "y", "likelihood"])

    # Likelihood thresholds
    hand_likelihood_threshold: float = 0.02  # Very low due to poor tracking
    general_likelihood_threshold: float = 0.5

    # Frame rate (Hz)
    fps: int = 60  # Corrected based on ground truth files showing fps=60

    @property
    def n_body_parts(self) -> int:
        return len(self.body_parts)

    @property
    def n_hand_parts(self) -> int:
        return len(self.hand_parts)


# ==============================================================================
# FEATURE EXTRACTION CONFIGURATION
# ==============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Window sizes (in frames)
    boundary_window_size: int = 61  # +/- 30 frames around boundary
    reach_sequence_length: int = 120  # ~2 seconds at 60 Hz
    outcome_window_size: int = 180  # ~3 seconds for outcome features

    # Temporal smoothing
    smoothing_window: int = 5  # frames for rolling average

    # Features to extract
    use_raw_coords: bool = True
    use_velocity: bool = True
    use_acceleration: bool = True
    use_hand_pellet_distance: bool = True
    use_hand_pellet_angle: bool = True
    use_trajectory_features: bool = True
    use_likelihood_weighting: bool = True

    # Body parts for feature extraction (subset of all parts)
    feature_body_parts: List[str] = field(default_factory=lambda: [
        "RightHand", "Pellet", "Pillar", "Nose", "TailBase"
    ])

    # Normalization method: "minmax", "zscore", or "none"
    normalization: str = "zscore"

    # Data augmentation
    use_augmentation: bool = True
    time_shift_range: Tuple[int, int] = (-5, 5)  # frames
    noise_std: float = 0.01  # normalized coordinates

    @property
    def n_raw_features(self) -> int:
        """Number of raw coordinate features (x, y per body part)."""
        return len(self.feature_body_parts) * 2

    @property
    def n_velocity_features(self) -> int:
        """Number of velocity features (dx, dy, magnitude per body part)."""
        return len(self.feature_body_parts) * 3 if self.use_velocity else 0

    @property
    def n_acceleration_features(self) -> int:
        """Number of acceleration features."""
        return len(self.feature_body_parts) * 3 if self.use_acceleration else 0


# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================

@dataclass
class BoundaryDetectorConfig:
    """Hyperparameters for boundary detection model (TCN/Transformer)."""

    # Architecture
    model_type: str = "tcn"  # "tcn" or "transformer"

    # TCN parameters
    tcn_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 128, 64])
    tcn_kernel_size: int = 7
    tcn_dropout: float = 0.3

    # Transformer parameters
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 512
    transformer_dropout: float = 0.1

    # Output
    n_classes: int = 2  # boundary / non-boundary

    # Class weights (boundary is rare)
    class_weights: List[float] = field(default_factory=lambda: [1.0, 10.0])


@dataclass
class ReachDetectorConfig:
    """Hyperparameters for reach detection model (BiLSTM-CRF)."""

    # Architecture
    hidden_size: int = 256
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3

    # CRF tags: O (outside), B-reach (begin), I-reach (inside), E-reach (end)
    tags: List[str] = field(default_factory=lambda: ["O", "B-reach", "I-reach", "E-reach"])

    # Use CRF layer for sequence labeling
    use_crf: bool = True

    # Class weights
    class_weights: List[float] = field(default_factory=lambda: [1.0, 5.0, 2.0, 5.0])

    @property
    def n_tags(self) -> int:
        return len(self.tags)

    @property
    def tag_to_idx(self) -> Dict[str, int]:
        return {tag: idx for idx, tag in enumerate(self.tags)}

    @property
    def idx_to_tag(self) -> Dict[int, str]:
        return {idx: tag for idx, tag in enumerate(self.tags)}


@dataclass
class OutcomeClassifierConfig:
    """Hyperparameters for outcome classification model (MLP)."""

    # Architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.4
    use_batch_norm: bool = True

    # Output classes
    outcome_classes: List[str] = field(default_factory=lambda: [
        "retrieved",      # Successfully grabbed pellet
        "displaced_sa",   # Displaced within scoring area
        "displaced_outside",  # Displaced outside scoring area
        "untouched",      # No interaction
        "no_pellet"       # Pellet missing (rare)
    ])

    # Class weights for imbalanced data (adjust based on dataset)
    class_weights: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 1.0, 5.0])

    @property
    def n_classes(self) -> int:
        return len(self.outcome_classes)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {cls: idx for idx, cls in enumerate(self.outcome_classes)}

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {idx: cls for idx, cls in enumerate(self.outcome_classes)}


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Optimizer
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"

    # Learning rate scheduler
    scheduler: str = "cosine"  # "step", "cosine", "plateau"
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.5
    scheduler_patience: int = 10  # for plateau

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Loss function
    use_focal_loss: bool = True
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: Optional[List[float]] = None  # Use class weights if None

    # Gradient clipping
    gradient_clip_val: float = 1.0

    # Checkpointing
    save_top_k: int = 3
    checkpoint_metric: str = "val_f1"
    checkpoint_mode: str = "max"

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 4
    pin_memory: bool = True


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

@dataclass
class LoggingConfig:
    """Configuration for logging and metrics tracking."""

    # Log frequency
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0  # Check validation every epoch

    # Metrics to track
    metrics: List[str] = field(default_factory=lambda: [
        "loss", "accuracy", "precision", "recall", "f1",
        "confusion_matrix", "roc_auc"
    ])

    # CSV logging
    csv_log_file: str = "training_log.csv"

    # Tensorboard
    use_tensorboard: bool = True
    tensorboard_dir: str = "tensorboard_logs"


# ==============================================================================
# MASTER CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    dlc: DLCConfig = field(default_factory=DLCConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    boundary_detector: BoundaryDetectorConfig = field(default_factory=BoundaryDetectorConfig)
    reach_detector: ReachDetectorConfig = field(default_factory=ReachDetectorConfig)
    outcome_classifier: OutcomeClassifierConfig = field(default_factory=OutcomeClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def save(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "paths": {k: str(v) if isinstance(v, Path) else v
                     for k, v in self.paths.__dict__.items()},
            "dlc": self.dlc.__dict__,
            "features": self.features.__dict__,
            "boundary_detector": self.boundary_detector.__dict__,
            "reach_detector": self.reach_detector.__dict__,
            "outcome_classifier": self.outcome_classifier.__dict__,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "Config":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Convert path strings back to Path objects
        if "paths" in config_dict:
            for key in config_dict["paths"]:
                if key.endswith("_dir") or key.endswith("_exe"):
                    config_dict["paths"][key] = Path(config_dict["paths"][key])

        return cls(
            paths=PathConfig(**config_dict.get("paths", {})),
            dlc=DLCConfig(**config_dict.get("dlc", {})),
            features=FeatureConfig(**config_dict.get("features", {})),
            boundary_detector=BoundaryDetectorConfig(**config_dict.get("boundary_detector", {})),
            reach_detector=ReachDetectorConfig(**config_dict.get("reach_detector", {})),
            outcome_classifier=OutcomeClassifierConfig(**config_dict.get("outcome_classifier", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
        )


# ==============================================================================
# DEFAULT INSTANCE
# ==============================================================================

# Create default configuration instance
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the default configuration."""
    return DEFAULT_CONFIG


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(f"Processing directory: {config.paths.processing_dir}")
    print(f"DLC scorer: {config.dlc.scorer}")
    print(f"Number of body parts: {config.dlc.n_body_parts}")
    print(f"Hand parts: {config.dlc.hand_parts}")
    print(f"Boundary window size: {config.features.boundary_window_size}")
    print(f"Outcome classes: {config.outcome_classifier.outcome_classes}")

    # Save config
    config_path = config.paths.training_dir / "config.json"
    config.save(config_path)
    print(f"\nConfiguration saved to: {config_path}")
