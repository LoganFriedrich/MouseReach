"""
PyTorch Dataset Classes for MouseReach Deep Learning Pipeline.

Provides three dataset types:
1. BoundaryDataset: Windowed samples for boundary detection (+/- 30 frames)
2. ReachSequenceDataset: Sequence samples for reach detection (sequence labeling)
3. OutcomeDataset: Reach-aggregated samples for outcome classification

Handles session-level train/val split to prevent data leakage.

Environment: Y:\\2_Connectome\\envs\\mousereach
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Any
from dataclasses import dataclass, field
import random
from collections import defaultdict
import warnings

import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

from config import Config, get_config, PathConfig, FeatureConfig
from feature_extraction import FeatureExtractor, ExtractedFeatures, FeatureNormalizer


# ==============================================================================
# DATA DISCOVERY
# ==============================================================================

@dataclass
class SessionData:
    """Container for a single session's data."""

    video_name: str
    dlc_path: Path
    gt_path: Optional[Path]
    features: Optional[ExtractedFeatures] = None

    # Ground truth data
    boundaries: List[Dict] = field(default_factory=list)
    reaches: List[Dict] = field(default_factory=list)
    outcomes: List[Dict] = field(default_factory=list)

    # Metadata
    n_frames: int = 0
    fps: float = 60.0


class DataDiscovery:
    """
    Discover and match DLC files with ground truth files.
    """

    def __init__(self, config: Optional[PathConfig] = None):
        self.config = config or get_config().paths

    def discover_sessions(self) -> List[SessionData]:
        """
        Discover all sessions with DLC data and optional ground truth.

        Returns:
            List of SessionData objects
        """
        sessions = []
        processing_dir = self.config.processing_dir

        # Find all DLC files
        dlc_files = list(processing_dir.glob(self.config.dlc_pattern))

        for dlc_path in dlc_files:
            # Extract video name from DLC filename
            # Format: YYYYMMDD_CNTXXXX_PYDLC_resnet50_...
            video_name = dlc_path.stem.split("DLC_")[0]

            # Look for unified ground truth
            gt_path = processing_dir / f"{video_name}_unified_ground_truth.json"
            if not gt_path.exists():
                gt_path = None

            session = SessionData(
                video_name=video_name,
                dlc_path=dlc_path,
                gt_path=gt_path
            )

            # Load ground truth if available
            if gt_path:
                self._load_ground_truth(session)

            sessions.append(session)

        return sessions

    def _load_ground_truth(self, session: SessionData) -> None:
        """Load ground truth data into session."""
        if session.gt_path is None or not session.gt_path.exists():
            return

        with open(session.gt_path, 'r') as f:
            gt_data = json.load(f)

        # Extract boundaries from segmentation
        segmentation = gt_data.get("segmentation", {})
        session.boundaries = segmentation.get("boundaries", [])

        # Extract reaches
        reaches_data = gt_data.get("reaches", {})
        session.reaches = reaches_data.get("reaches", [])

        # Extract outcomes
        outcomes_data = gt_data.get("outcomes", {})
        session.outcomes = outcomes_data.get("segments", [])

    def get_sessions_with_ground_truth(self) -> List[SessionData]:
        """Get only sessions that have ground truth data."""
        sessions = self.discover_sessions()
        return [s for s in sessions if s.gt_path is not None]

    def get_sessions_with_determined_boundaries(self) -> List[SessionData]:
        """Get sessions with at least some determined boundaries."""
        sessions = self.get_sessions_with_ground_truth()
        return [s for s in sessions if any(b.get("determined", False) for b in s.boundaries)]

    def get_sessions_with_determined_reaches(self) -> List[SessionData]:
        """Get sessions with determined reach annotations."""
        sessions = self.get_sessions_with_ground_truth()
        return [s for s in sessions if any(
            r.get("start_determined", False) and r.get("end_determined", False)
            for r in s.reaches
        )]

    def get_sessions_with_determined_outcomes(self) -> List[SessionData]:
        """Get sessions with determined outcome annotations."""
        sessions = self.get_sessions_with_ground_truth()
        return [s for s in sessions if any(o.get("determined", False) for o in s.outcomes)]


# ==============================================================================
# SESSION-LEVEL DATA SPLITTING
# ==============================================================================

class SessionSplitter:
    """
    Split sessions into train/val/test sets at session level to prevent data leakage.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    def split(
        self,
        sessions: List[SessionData]
    ) -> Tuple[List[SessionData], List[SessionData], List[SessionData]]:
        """
        Split sessions into train/val/test sets.

        Args:
            sessions: List of SessionData objects

        Returns:
            Tuple of (train_sessions, val_sessions, test_sessions)
        """
        random.seed(self.seed)
        sessions = sessions.copy()
        random.shuffle(sessions)

        n = len(sessions)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_sessions = sessions[:n_train]
        val_sessions = sessions[n_train:n_train + n_val]
        test_sessions = sessions[n_train + n_val:]

        return train_sessions, val_sessions, test_sessions

    def stratified_split_by_outcome(
        self,
        sessions: List[SessionData]
    ) -> Tuple[List[SessionData], List[SessionData], List[SessionData]]:
        """
        Split sessions with stratification by outcome distribution.
        """
        # Group sessions by predominant outcome
        outcome_groups = defaultdict(list)

        for session in sessions:
            if session.outcomes:
                outcomes = [o.get("outcome", "unknown") for o in session.outcomes if o.get("determined", False)]
                if outcomes:
                    # Use most common outcome as group key
                    predominant = max(set(outcomes), key=outcomes.count)
                    outcome_groups[predominant].append(session)
                else:
                    outcome_groups["unknown"].append(session)
            else:
                outcome_groups["unknown"].append(session)

        # Split each group
        train_sessions, val_sessions, test_sessions = [], [], []

        random.seed(self.seed)
        for outcome, group in outcome_groups.items():
            random.shuffle(group)
            n = len(group)
            n_train = max(1, int(n * self.train_ratio))
            n_val = max(0, int(n * self.val_ratio))

            train_sessions.extend(group[:n_train])
            val_sessions.extend(group[n_train:n_train + n_val])
            test_sessions.extend(group[n_train + n_val:])

        return train_sessions, val_sessions, test_sessions


# ==============================================================================
# BOUNDARY DETECTION DATASET
# ==============================================================================

class BoundaryDataset(Dataset):
    """
    Dataset for boundary detection (binary classification at each frame).

    Creates windowed samples centered on each frame, with labels indicating
    whether the frame is a boundary (within tolerance) or not.
    """

    def __init__(
        self,
        sessions: List[SessionData],
        feature_extractor: FeatureExtractor,
        window_size: int = 61,
        boundary_tolerance: int = 3,
        negative_ratio: float = 1.0,
        normalizer: Optional[FeatureNormalizer] = None,
        augment: bool = False
    ):
        """
        Args:
            sessions: List of SessionData objects with ground truth
            feature_extractor: FeatureExtractor instance
            window_size: Size of feature window (should be odd)
            boundary_tolerance: Frames around boundary to also label as positive
            negative_ratio: Ratio of negative to positive samples (for balancing)
            normalizer: Optional FeatureNormalizer (fitted on training data)
            augment: Whether to apply data augmentation
        """
        self.sessions = sessions
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.boundary_tolerance = boundary_tolerance
        self.negative_ratio = negative_ratio
        self.normalizer = normalizer
        self.augment = augment

        self.half_window = window_size // 2

        # Build sample index
        self._build_samples()

    def _build_samples(self) -> None:
        """Build list of (session_idx, frame_idx, label) tuples."""
        self.samples = []
        self.features_cache: Dict[int, ExtractedFeatures] = {}

        for session_idx, session in enumerate(self.sessions):
            # Extract features for session
            features = self.feature_extractor.extract(session.dlc_path, session.video_name)
            self.features_cache[session_idx] = features
            session.n_frames = features.n_frames

            # Get determined boundary frames
            boundary_frames = set()
            for b in session.boundaries:
                if b.get("determined", False):
                    frame = b.get("frame", b.get("index", 0))
                    # Add tolerance around boundary
                    for offset in range(-self.boundary_tolerance, self.boundary_tolerance + 1):
                        boundary_frames.add(frame + offset)

            # Create positive samples (boundaries)
            positive_samples = []
            for frame in boundary_frames:
                if self.half_window <= frame < features.n_frames - self.half_window:
                    positive_samples.append((session_idx, frame, 1))

            # Create negative samples (non-boundaries)
            all_frames = set(range(self.half_window, features.n_frames - self.half_window))
            negative_frames = all_frames - boundary_frames

            # Sample negatives based on ratio
            n_negatives = int(len(positive_samples) * self.negative_ratio)
            if n_negatives > 0 and negative_frames:
                negative_samples = random.sample(
                    list(negative_frames),
                    min(n_negatives, len(negative_frames))
                )
                negative_samples = [(session_idx, frame, 0) for frame in negative_samples]
            else:
                negative_samples = []

            self.samples.extend(positive_samples)
            self.samples.extend(negative_samples)

        # Shuffle samples
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session_idx, frame_idx, label = self.samples[idx]

        # Get features
        features = self.features_cache[session_idx]
        window = self.feature_extractor.extract_window(features, frame_idx, self.window_size)

        # Normalize
        if self.normalizer:
            window = self.normalizer.transform(window)

        # Augment
        if self.augment:
            window = self._augment(window)

        # Convert to tensor
        x = torch.FloatTensor(window)
        y = torch.LongTensor([label])

        return x, y.squeeze()

    def _augment(self, window: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Add Gaussian noise
        noise = np.random.normal(0, 0.01, window.shape)
        window = window + noise

        # Random time shift (by rolling)
        shift = np.random.randint(-3, 4)
        if shift != 0:
            window = np.roll(window, shift, axis=0)

        return window

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced classes."""
        labels = [s[2] for s in self.samples]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        total = len(labels)

        weight_neg = total / (2 * n_neg) if n_neg > 0 else 1.0
        weight_pos = total / (2 * n_pos) if n_pos > 0 else 1.0

        return torch.FloatTensor([weight_neg, weight_pos])


# ==============================================================================
# REACH SEQUENCE DATASET
# ==============================================================================

class ReachSequenceDataset(Dataset):
    """
    Dataset for reach detection as sequence labeling.

    Each sample is a sequence of frames with BIO-style labels:
    - O: Outside any reach
    - B-reach: Beginning of reach
    - I-reach: Inside reach
    - E-reach: End of reach
    """

    def __init__(
        self,
        sessions: List[SessionData],
        feature_extractor: FeatureExtractor,
        sequence_length: int = 120,
        stride: int = 60,
        normalizer: Optional[FeatureNormalizer] = None,
        augment: bool = False
    ):
        """
        Args:
            sessions: List of SessionData with verified reach annotations
            feature_extractor: FeatureExtractor instance
            sequence_length: Length of each sequence (frames)
            stride: Stride between sequences
            normalizer: Optional FeatureNormalizer
            augment: Whether to augment data
        """
        self.sessions = sessions
        self.feature_extractor = feature_extractor
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalizer = normalizer
        self.augment = augment

        # Tag mapping
        self.tags = ["O", "B-reach", "I-reach", "E-reach"]
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx_to_tag = {idx: tag for idx, tag in enumerate(self.tags)}

        self._build_samples()

    def _build_samples(self) -> None:
        """Build sequence samples with labels."""
        self.samples = []
        self.features_cache: Dict[int, ExtractedFeatures] = {}

        for session_idx, session in enumerate(self.sessions):
            # Extract features
            features = self.feature_extractor.extract(session.dlc_path, session.video_name)
            self.features_cache[session_idx] = features
            n_frames = features.n_frames

            # Build frame-level labels
            frame_labels = np.zeros(n_frames, dtype=np.int64)  # All O by default

            for reach in session.reaches:
                # Only use determined reaches
                if not (reach.get("start_determined", False) and reach.get("end_determined", False)):
                    continue

                if reach.get("exclude_from_analysis", False):
                    continue

                start_frame = reach.get("start_frame", 0)
                end_frame = reach.get("end_frame", 0)

                if start_frame < 0 or end_frame >= n_frames or start_frame >= end_frame:
                    continue

                # Label frames
                frame_labels[start_frame] = self.tag_to_idx["B-reach"]
                frame_labels[end_frame] = self.tag_to_idx["E-reach"]
                for f in range(start_frame + 1, end_frame):
                    frame_labels[f] = self.tag_to_idx["I-reach"]

            # Create sequences
            for start in range(0, n_frames - self.sequence_length, self.stride):
                end = start + self.sequence_length
                self.samples.append((session_idx, start, end, frame_labels[start:end]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session_idx, start, end, labels = self.samples[idx]

        # Get feature sequence
        features = self.features_cache[session_idx]
        sequence = self.feature_extractor.extract_sequence(features, start, end)

        # Normalize
        if self.normalizer:
            sequence = self.normalizer.transform(sequence)

        # Augment
        if self.augment:
            sequence = self._augment(sequence)

        x = torch.FloatTensor(sequence)
        y = torch.LongTensor(labels)

        return x, y

    def _augment(self, sequence: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        noise = np.random.normal(0, 0.01, sequence.shape)
        return sequence + noise

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights."""
        all_labels = np.concatenate([s[3] for s in self.samples])
        counts = np.bincount(all_labels, minlength=len(self.tags))
        total = len(all_labels)

        weights = []
        for c in counts:
            if c > 0:
                weights.append(total / (len(self.tags) * c))
            else:
                weights.append(1.0)

        return torch.FloatTensor(weights)


# ==============================================================================
# OUTCOME CLASSIFICATION DATASET
# ==============================================================================

class OutcomeDataset(Dataset):
    """
    Dataset for outcome classification.

    Each sample is aggregated features from a segment (reach attempt),
    with the outcome label.
    """

    def __init__(
        self,
        sessions: List[SessionData],
        feature_extractor: FeatureExtractor,
        window_before: int = 60,
        window_after: int = 60,
        normalizer: Optional[FeatureNormalizer] = None,
        augment: bool = False
    ):
        """
        Args:
            sessions: List of SessionData with verified outcome annotations
            feature_extractor: FeatureExtractor instance
            window_before: Frames before interaction to include
            window_after: Frames after interaction to include
            normalizer: Optional FeatureNormalizer
            augment: Whether to augment data
        """
        self.sessions = sessions
        self.feature_extractor = feature_extractor
        self.window_before = window_before
        self.window_after = window_after
        self.normalizer = normalizer
        self.augment = augment

        # Outcome classes
        self.outcome_classes = [
            "retrieved", "displaced_sa", "displaced_outside", "untouched", "no_pellet"
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.outcome_classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.outcome_classes)}

        self._build_samples()

    def _build_samples(self) -> None:
        """Build outcome samples."""
        self.samples = []
        self.features_cache: Dict[int, ExtractedFeatures] = {}

        for session_idx, session in enumerate(self.sessions):
            # Extract features
            features = self.feature_extractor.extract(session.dlc_path, session.video_name)
            self.features_cache[session_idx] = features
            n_frames = features.n_frames

            for outcome in session.outcomes:
                if not outcome.get("determined", False):
                    continue

                outcome_label = outcome.get("outcome", "unknown")
                if outcome_label not in self.class_to_idx:
                    continue

                # Get interaction frame (or outcome known frame)
                interaction_frame = outcome.get("interaction_frame")
                if interaction_frame is None:
                    interaction_frame = outcome.get("outcome_known_frame")
                if interaction_frame is None:
                    continue

                # Check bounds
                start = interaction_frame - self.window_before
                end = interaction_frame + self.window_after

                if start < 0 or end >= n_frames:
                    continue

                label = self.class_to_idx[outcome_label]
                self.samples.append((session_idx, start, end, interaction_frame, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session_idx, start, end, interaction_frame, label = self.samples[idx]

        # Get feature sequence
        features = self.features_cache[session_idx]
        sequence = self.feature_extractor.extract_sequence(features, start, end)

        # Normalize
        if self.normalizer:
            sequence = self.normalizer.transform(sequence)

        # Aggregate features
        aggregated = self._aggregate_features(sequence, interaction_frame - start)

        # Augment
        if self.augment:
            aggregated = aggregated + np.random.normal(0, 0.01, aggregated.shape)

        x = torch.FloatTensor(aggregated)
        y = torch.LongTensor([label])

        return x, y.squeeze()

    def _aggregate_features(
        self,
        sequence: np.ndarray,
        interaction_idx: int
    ) -> np.ndarray:
        """
        Aggregate sequence features into a fixed-size vector.

        Computes statistics over before/after windows and at interaction point.
        """
        before = sequence[:interaction_idx]
        after = sequence[interaction_idx:]
        at_interaction = sequence[interaction_idx]

        # Compute statistics
        stats = []

        # Features at interaction point
        stats.append(at_interaction)

        # Before interaction statistics
        if len(before) > 0:
            stats.append(before.mean(axis=0))
            stats.append(before.std(axis=0))
            stats.append(before.max(axis=0))
            stats.append(before.min(axis=0))
        else:
            stats.extend([np.zeros_like(at_interaction)] * 4)

        # After interaction statistics
        if len(after) > 0:
            stats.append(after.mean(axis=0))
            stats.append(after.std(axis=0))
            stats.append(after.max(axis=0))
            stats.append(after.min(axis=0))
        else:
            stats.extend([np.zeros_like(at_interaction)] * 4)

        # Differences
        if len(before) > 0 and len(after) > 0:
            stats.append(after.mean(axis=0) - before.mean(axis=0))  # Change
        else:
            stats.append(np.zeros_like(at_interaction))

        return np.concatenate(stats)

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced classes."""
        labels = [s[4] for s in self.samples]
        counts = np.bincount(labels, minlength=len(self.outcome_classes))
        total = len(labels)

        weights = []
        for c in counts:
            if c > 0:
                weights.append(total / (len(self.outcome_classes) * c))
            else:
                weights.append(1.0)

        return torch.FloatTensor(weights)

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes."""
        labels = [s[4] for s in self.samples]
        counts = np.bincount(labels, minlength=len(self.outcome_classes))
        return {self.idx_to_class[i]: int(c) for i, c in enumerate(counts)}


# ==============================================================================
# DATA MODULE
# ==============================================================================

class MouseReachDataModule:
    """
    High-level data module for managing all datasets.

    Handles:
    - Data discovery
    - Session-level splitting
    - Feature extraction and caching
    - Normalization fitting
    - DataLoader creation
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

        self.discovery = DataDiscovery(self.config.paths)
        self.splitter = SessionSplitter(
            train_ratio=self.config.training.train_ratio,
            val_ratio=self.config.training.val_ratio,
            test_ratio=self.config.training.test_ratio,
            seed=self.config.training.seed
        )
        self.feature_extractor = FeatureExtractor(
            self.config.features,
            self.config.dlc
        )

        self.normalizer: Optional[FeatureNormalizer] = None
        self.train_sessions: List[SessionData] = []
        self.val_sessions: List[SessionData] = []
        self.test_sessions: List[SessionData] = []

    def setup(self, task: str = "all") -> None:
        """
        Set up data module for specified task.

        Args:
            task: "boundary", "reach", "outcome", or "all"
        """
        # Discover sessions
        if task == "boundary":
            sessions = self.discovery.get_sessions_with_determined_boundaries()
        elif task == "reach":
            sessions = self.discovery.get_sessions_with_determined_reaches()
        elif task == "outcome":
            sessions = self.discovery.get_sessions_with_determined_outcomes()
        else:
            sessions = self.discovery.get_sessions_with_ground_truth()

        if not sessions:
            raise ValueError(f"No sessions found for task: {task}")

        print(f"Found {len(sessions)} sessions for task '{task}'")

        # Split sessions
        self.train_sessions, self.val_sessions, self.test_sessions = self.splitter.split(sessions)

        print(f"  Train: {len(self.train_sessions)} sessions")
        print(f"  Val: {len(self.val_sessions)} sessions")
        print(f"  Test: {len(self.test_sessions)} sessions")

        # Fit normalizer on training data
        self._fit_normalizer()

    def _fit_normalizer(self) -> None:
        """Fit normalizer on training data."""
        print("Fitting normalizer on training data...")

        all_features = []
        for session in self.train_sessions:
            features = self.feature_extractor.extract(session.dlc_path, session.video_name)
            all_features.append(features.to_array())

        if all_features:
            combined = np.vstack(all_features)
            self.normalizer = FeatureNormalizer(method=self.config.features.normalization)
            self.normalizer.fit(combined)
            print(f"Normalizer fitted on {combined.shape[0]} frames")

    def get_boundary_dataloaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get DataLoaders for boundary detection task."""
        batch_size = batch_size or self.config.training.batch_size
        num_workers = num_workers or self.config.training.num_workers

        train_dataset = BoundaryDataset(
            self.train_sessions,
            self.feature_extractor,
            window_size=self.config.features.boundary_window_size,
            normalizer=self.normalizer,
            augment=self.config.features.use_augmentation
        )

        val_dataset = BoundaryDataset(
            self.val_sessions,
            self.feature_extractor,
            window_size=self.config.features.boundary_window_size,
            normalizer=self.normalizer,
            augment=False
        )

        test_dataset = BoundaryDataset(
            self.test_sessions,
            self.feature_extractor,
            window_size=self.config.features.boundary_window_size,
            normalizer=self.normalizer,
            augment=False
        )

        # Use weighted sampler for training
        class_weights = train_dataset.get_class_weights()
        sample_weights = [class_weights[s[2]] for s in train_dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.config.training.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader

    def get_reach_dataloaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get DataLoaders for reach detection task."""
        batch_size = batch_size or self.config.training.batch_size
        num_workers = num_workers or self.config.training.num_workers

        train_dataset = ReachSequenceDataset(
            self.train_sessions,
            self.feature_extractor,
            sequence_length=self.config.features.reach_sequence_length,
            normalizer=self.normalizer,
            augment=self.config.features.use_augmentation
        )

        val_dataset = ReachSequenceDataset(
            self.val_sessions,
            self.feature_extractor,
            sequence_length=self.config.features.reach_sequence_length,
            normalizer=self.normalizer,
            augment=False
        )

        test_dataset = ReachSequenceDataset(
            self.test_sessions,
            self.feature_extractor,
            sequence_length=self.config.features.reach_sequence_length,
            normalizer=self.normalizer,
            augment=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.training.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader

    def get_outcome_dataloaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get DataLoaders for outcome classification task."""
        batch_size = batch_size or self.config.training.batch_size
        num_workers = num_workers or self.config.training.num_workers

        train_dataset = OutcomeDataset(
            self.train_sessions,
            self.feature_extractor,
            window_before=self.config.features.outcome_window_size // 2,
            window_after=self.config.features.outcome_window_size // 2,
            normalizer=self.normalizer,
            augment=self.config.features.use_augmentation
        )

        val_dataset = OutcomeDataset(
            self.val_sessions,
            self.feature_extractor,
            window_before=self.config.features.outcome_window_size // 2,
            window_after=self.config.features.outcome_window_size // 2,
            normalizer=self.normalizer,
            augment=False
        )

        test_dataset = OutcomeDataset(
            self.test_sessions,
            self.feature_extractor,
            window_before=self.config.features.outcome_window_size // 2,
            window_after=self.config.features.outcome_window_size // 2,
            normalizer=self.normalizer,
            augment=False
        )

        # Use weighted sampler for training
        class_weights = train_dataset.get_class_weights()
        sample_weights = [class_weights[s[4]].item() for s in train_dataset.samples]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.config.training.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader


# ==============================================================================
# MAIN / TESTING
# ==============================================================================

if __name__ == "__main__":
    from config import get_config

    config = get_config()

    print("=" * 60)
    print("Testing Data Loading Pipeline")
    print("=" * 60)

    # Test data discovery
    print("\n1. Data Discovery")
    discovery = DataDiscovery()
    sessions = discovery.get_sessions_with_ground_truth()
    print(f"   Found {len(sessions)} sessions with ground truth")

    if sessions:
        print(f"\n   Sample session: {sessions[0].video_name}")
        print(f"   - DLC path: {sessions[0].dlc_path.name}")
        print(f"   - Boundaries: {len(sessions[0].boundaries)}")
        print(f"   - Reaches: {len(sessions[0].reaches)}")
        print(f"   - Outcomes: {len(sessions[0].outcomes)}")

    # Test data module
    print("\n2. Data Module Setup")
    dm = MouseReachDataModule()

    try:
        dm.setup(task="outcome")

        print("\n3. Outcome DataLoaders")
        train_loader, val_loader, test_loader = dm.get_outcome_dataloaders(batch_size=8)

        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

        # Test a batch
        for x, y in train_loader:
            print(f"\n   Sample batch:")
            print(f"   - Input shape: {x.shape}")
            print(f"   - Labels shape: {y.shape}")
            print(f"   - Label distribution: {torch.bincount(y)}")
            break

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Data loading test complete")
    print("=" * 60)
