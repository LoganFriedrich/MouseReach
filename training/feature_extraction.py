"""
Feature Extraction Module for MouseReach Deep Learning Pipeline.

Extracts kinematic and spatial features from DeepLabCut pose data:
- Raw coordinates (x, y) for key body parts
- Velocity (dx, dy, magnitude) via finite differences
- Acceleration (ddx, ddy, magnitude)
- Hand-to-pellet distance and angle
- Trajectory straightness, smoothness
- Likelihood-weighted averaging for hand position

Environment: Y:\\2_Connectome\\envs\\mousereach
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import warnings

from config import DLCConfig, FeatureConfig, get_config


# ==============================================================================
# DLC DATA LOADER
# ==============================================================================

class DLCDataLoader:
    """
    Load and preprocess DeepLabCut h5 files.

    Handles multi-level column indexing and provides convenient access methods.
    """

    def __init__(self, config: Optional[DLCConfig] = None):
        self.config = config or get_config().dlc
        self._df: Optional[pd.DataFrame] = None
        self._filepath: Optional[Path] = None

    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load DLC h5 file into DataFrame."""
        self._filepath = Path(filepath)
        self._df = pd.read_hdf(self._filepath)
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """Get loaded DataFrame."""
        if self._df is None:
            raise ValueError("No data loaded. Call load() first.")
        return self._df

    @property
    def n_frames(self) -> int:
        """Number of frames in loaded data."""
        return len(self.df)

    def get_bodypart(self, bodypart: str) -> pd.DataFrame:
        """
        Get x, y, likelihood for a single body part.

        Args:
            bodypart: Name of body part (e.g., 'RightHand', 'Pellet')

        Returns:
            DataFrame with columns ['x', 'y', 'likelihood']
        """
        return self.df.xs(bodypart, level=1, axis=1)

    def get_coords(self, bodypart: str) -> np.ndarray:
        """
        Get x, y coordinates for a body part as numpy array.

        Args:
            bodypart: Name of body part

        Returns:
            Array of shape (n_frames, 2) with [x, y] coordinates
        """
        bp_data = self.get_bodypart(bodypart)
        return bp_data[['x', 'y']].values

    def get_likelihood(self, bodypart: str) -> np.ndarray:
        """Get likelihood values for a body part."""
        return self.get_bodypart(bodypart)['likelihood'].values

    def get_all_coords(self, bodyparts: Optional[List[str]] = None) -> np.ndarray:
        """
        Get coordinates for multiple body parts.

        Args:
            bodyparts: List of body part names. If None, use all.

        Returns:
            Array of shape (n_frames, n_parts, 2)
        """
        if bodyparts is None:
            bodyparts = self.config.body_parts

        coords = np.stack([self.get_coords(bp) for bp in bodyparts], axis=1)
        return coords

    def get_all_likelihoods(self, bodyparts: Optional[List[str]] = None) -> np.ndarray:
        """
        Get likelihoods for multiple body parts.

        Args:
            bodyparts: List of body part names. If None, use all.

        Returns:
            Array of shape (n_frames, n_parts)
        """
        if bodyparts is None:
            bodyparts = self.config.body_parts

        likelihoods = np.stack([self.get_likelihood(bp) for bp in bodyparts], axis=1)
        return likelihoods


# ==============================================================================
# HAND POSITION ESTIMATION
# ==============================================================================

class HandPositionEstimator:
    """
    Estimate hand position using likelihood-weighted averaging of multiple markers.

    Critical for dealing with low-likelihood hand tracking in DLC data.
    """

    def __init__(self, config: Optional[DLCConfig] = None):
        self.config = config or get_config().dlc

    def estimate(
        self,
        loader: DLCDataLoader,
        method: str = "weighted_average"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate hand position from multiple hand markers.

        Args:
            loader: DLCDataLoader with loaded data
            method: "weighted_average", "max_likelihood", or "simple_average"

        Returns:
            Tuple of (positions, confidences) where:
                - positions: shape (n_frames, 2) with [x, y]
                - confidences: shape (n_frames,) with confidence scores
        """
        hand_parts = self.config.hand_parts

        # Get coordinates and likelihoods for all hand markers
        coords = np.stack([loader.get_coords(hp) for hp in hand_parts], axis=1)  # (n_frames, 4, 2)
        likelihoods = np.stack([loader.get_likelihood(hp) for hp in hand_parts], axis=1)  # (n_frames, 4)

        if method == "weighted_average":
            return self._weighted_average(coords, likelihoods)
        elif method == "max_likelihood":
            return self._max_likelihood(coords, likelihoods)
        elif method == "simple_average":
            return self._simple_average(coords, likelihoods)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _weighted_average(
        self,
        coords: np.ndarray,
        likelihoods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute likelihood-weighted average position."""
        # Normalize weights
        weights = likelihoods / (likelihoods.sum(axis=1, keepdims=True) + 1e-8)

        # Weighted average
        positions = (coords * weights[:, :, np.newaxis]).sum(axis=1)

        # Confidence is max likelihood among markers
        confidences = likelihoods.max(axis=1)

        return positions, confidences

    def _max_likelihood(
        self,
        coords: np.ndarray,
        likelihoods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select position from marker with highest likelihood."""
        best_idx = likelihoods.argmax(axis=1)
        n_frames = coords.shape[0]

        positions = coords[np.arange(n_frames), best_idx]
        confidences = likelihoods[np.arange(n_frames), best_idx]

        return positions, confidences

    def _simple_average(
        self,
        coords: np.ndarray,
        likelihoods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple average of all hand markers."""
        positions = coords.mean(axis=1)
        confidences = likelihoods.mean(axis=1)
        return positions, confidences


# ==============================================================================
# KINEMATIC FEATURES
# ==============================================================================

class KinematicFeatureExtractor:
    """
    Extract velocity and acceleration features from position data.
    """

    def __init__(self, fps: int = 60, smoothing_window: int = 5):
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.dt = 1.0 / fps

    def compute_velocity(
        self,
        positions: np.ndarray,
        smooth: bool = True
    ) -> np.ndarray:
        """
        Compute velocity using finite differences.

        Args:
            positions: Shape (n_frames, 2) with [x, y] coordinates
            smooth: Whether to apply smoothing

        Returns:
            Velocity array of shape (n_frames, 3) with [dx, dy, magnitude]
        """
        if smooth:
            positions = self._smooth(positions)

        # Finite difference (central difference for interior, forward/backward at edges)
        velocity = np.zeros_like(positions)
        velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * self.dt)
        velocity[0] = (positions[1] - positions[0]) / self.dt
        velocity[-1] = (positions[-1] - positions[-2]) / self.dt

        # Compute magnitude
        magnitude = np.linalg.norm(velocity, axis=1, keepdims=True)

        return np.hstack([velocity, magnitude])

    def compute_acceleration(
        self,
        positions: np.ndarray,
        smooth: bool = True
    ) -> np.ndarray:
        """
        Compute acceleration using second-order finite differences.

        Args:
            positions: Shape (n_frames, 2) with [x, y] coordinates
            smooth: Whether to apply smoothing

        Returns:
            Acceleration array of shape (n_frames, 3) with [ddx, ddy, magnitude]
        """
        if smooth:
            positions = self._smooth(positions)

        # Second derivative (central difference)
        accel = np.zeros_like(positions)
        accel[1:-1] = (positions[2:] - 2 * positions[1:-1] + positions[:-2]) / (self.dt ** 2)
        accel[0] = accel[1]
        accel[-1] = accel[-2]

        # Compute magnitude
        magnitude = np.linalg.norm(accel, axis=1, keepdims=True)

        return np.hstack([accel, magnitude])

    def compute_jerk(
        self,
        positions: np.ndarray,
        smooth: bool = True
    ) -> np.ndarray:
        """
        Compute jerk (third derivative) - useful for movement smoothness.

        Args:
            positions: Shape (n_frames, 2) with [x, y] coordinates
            smooth: Whether to apply smoothing

        Returns:
            Jerk array of shape (n_frames, 3) with [jx, jy, magnitude]
        """
        accel = self.compute_acceleration(positions, smooth)[:, :2]  # Get ddx, ddy only
        jerk = np.zeros_like(accel)
        jerk[1:-1] = (accel[2:] - accel[:-2]) / (2 * self.dt)
        jerk[0] = jerk[1]
        jerk[-1] = jerk[-2]

        magnitude = np.linalg.norm(jerk, axis=1, keepdims=True)
        return np.hstack([jerk, magnitude])

    def _smooth(self, data: np.ndarray) -> np.ndarray:
        """Apply rolling mean smoothing."""
        if self.smoothing_window <= 1:
            return data

        df = pd.DataFrame(data)
        smoothed = df.rolling(window=self.smoothing_window, center=True, min_periods=1).mean()
        return smoothed.values


# ==============================================================================
# SPATIAL FEATURES
# ==============================================================================

class SpatialFeatureExtractor:
    """
    Extract spatial relationship features between body parts and task objects.
    """

    @staticmethod
    def compute_distance(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance between two position arrays.

        Args:
            pos1, pos2: Shape (n_frames, 2)

        Returns:
            Distance array of shape (n_frames,)
        """
        return np.linalg.norm(pos1 - pos2, axis=1)

    @staticmethod
    def compute_angle(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """
        Compute angle from pos1 to pos2 in radians.

        Args:
            pos1, pos2: Shape (n_frames, 2)

        Returns:
            Angle array of shape (n_frames,) in radians [-pi, pi]
        """
        diff = pos2 - pos1
        return np.arctan2(diff[:, 1], diff[:, 0])

    @staticmethod
    def compute_relative_position(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """
        Compute relative position (pos2 - pos1).

        Args:
            pos1, pos2: Shape (n_frames, 2)

        Returns:
            Relative position array of shape (n_frames, 2)
        """
        return pos2 - pos1


# ==============================================================================
# TRAJECTORY FEATURES
# ==============================================================================

class TrajectoryFeatureExtractor:
    """
    Extract trajectory-level features like straightness, smoothness, curvature.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size

    def compute_straightness(
        self,
        positions: np.ndarray,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute trajectory straightness (direct path / actual path length).

        A value of 1.0 indicates perfectly straight movement.

        Args:
            positions: Shape (n_frames, 2)
            window_size: Window for computing local straightness

        Returns:
            Straightness values of shape (n_frames,)
        """
        ws = window_size or self.window_size
        n_frames = len(positions)
        straightness = np.ones(n_frames)

        for i in range(ws, n_frames - ws):
            start = positions[i - ws]
            end = positions[i + ws]

            # Direct distance
            direct_dist = np.linalg.norm(end - start)

            # Path length
            segment = positions[i - ws:i + ws + 1]
            path_length = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))

            if path_length > 0:
                straightness[i] = direct_dist / path_length

        # Fill edges
        straightness[:ws] = straightness[ws]
        straightness[-ws:] = straightness[-ws - 1]

        return straightness

    def compute_smoothness(
        self,
        positions: np.ndarray,
        fps: int = 60
    ) -> np.ndarray:
        """
        Compute movement smoothness using spectral arc length.

        Lower values indicate smoother movements.

        Args:
            positions: Shape (n_frames, 2)
            fps: Frame rate

        Returns:
            Smoothness values of shape (n_frames,)
        """
        # Compute velocity
        velocity = np.diff(positions, axis=0) * fps
        speed = np.linalg.norm(velocity, axis=1)

        # Rolling window smoothness (simplified version)
        ws = self.window_size
        n_frames = len(positions)
        smoothness = np.zeros(n_frames)

        for i in range(ws, n_frames - ws):
            window_speed = speed[max(0, i - ws):min(len(speed), i + ws)]

            if len(window_speed) > 1 and window_speed.max() > 0:
                # Normalize speed
                norm_speed = window_speed / window_speed.max()

                # Compute spectral arc length (simplified)
                speed_changes = np.abs(np.diff(norm_speed))
                smoothness[i] = -np.sum(speed_changes)  # More negative = less smooth

        return smoothness

    def compute_curvature(
        self,
        positions: np.ndarray,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute local curvature of trajectory.

        Args:
            positions: Shape (n_frames, 2)
            window_size: Window for curvature estimation

        Returns:
            Curvature values of shape (n_frames,)
        """
        ws = window_size or self.window_size // 2
        n_frames = len(positions)
        curvature = np.zeros(n_frames)

        # First and second derivatives
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        denominator = (dx ** 2 + dy ** 2) ** 1.5
        denominator = np.maximum(denominator, 1e-8)  # Avoid division by zero

        curvature = np.abs(dx * ddy - dy * ddx) / denominator

        return curvature


# ==============================================================================
# MAIN FEATURE EXTRACTOR
# ==============================================================================

@dataclass
class ExtractedFeatures:
    """Container for extracted features."""

    # Raw coordinates
    hand_position: np.ndarray  # (n_frames, 2)
    hand_confidence: np.ndarray  # (n_frames,)
    pellet_position: np.ndarray  # (n_frames, 2)
    nose_position: np.ndarray  # (n_frames, 2)

    # Kinematic features
    hand_velocity: np.ndarray  # (n_frames, 3) [dx, dy, magnitude]
    hand_acceleration: np.ndarray  # (n_frames, 3)

    # Spatial features
    hand_pellet_distance: np.ndarray  # (n_frames,)
    hand_pellet_angle: np.ndarray  # (n_frames,)
    nose_pellet_distance: np.ndarray  # (n_frames,)

    # Trajectory features
    straightness: np.ndarray  # (n_frames,)
    smoothness: np.ndarray  # (n_frames,)
    curvature: np.ndarray  # (n_frames,)

    # Metadata
    n_frames: int
    video_name: str

    def to_array(
        self,
        include_raw: bool = True,
        include_velocity: bool = True,
        include_acceleration: bool = True,
        include_spatial: bool = True,
        include_trajectory: bool = True
    ) -> np.ndarray:
        """
        Combine selected features into single array.

        Returns:
            Feature array of shape (n_frames, n_features)
        """
        features = []

        if include_raw:
            features.extend([
                self.hand_position,
                self.pellet_position,
                self.nose_position,
                self.hand_confidence.reshape(-1, 1)
            ])

        if include_velocity:
            features.append(self.hand_velocity)

        if include_acceleration:
            features.append(self.hand_acceleration)

        if include_spatial:
            features.extend([
                self.hand_pellet_distance.reshape(-1, 1),
                self.hand_pellet_angle.reshape(-1, 1),
                self.nose_pellet_distance.reshape(-1, 1)
            ])

        if include_trajectory:
            features.extend([
                self.straightness.reshape(-1, 1),
                self.smoothness.reshape(-1, 1),
                self.curvature.reshape(-1, 1)
            ])

        return np.hstack(features)

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        names = [
            "hand_x", "hand_y",
            "pellet_x", "pellet_y",
            "nose_x", "nose_y",
            "hand_confidence",
            "hand_vx", "hand_vy", "hand_speed",
            "hand_ax", "hand_ay", "hand_accel_mag",
            "hand_pellet_dist", "hand_pellet_angle",
            "nose_pellet_dist",
            "straightness", "smoothness", "curvature"
        ]
        return names


class FeatureExtractor:
    """
    Main feature extraction class combining all feature types.
    """

    def __init__(self, config: Optional[FeatureConfig] = None, dlc_config: Optional[DLCConfig] = None):
        self.config = config or get_config().features
        self.dlc_config = dlc_config or get_config().dlc

        self.dlc_loader = DLCDataLoader(self.dlc_config)
        self.hand_estimator = HandPositionEstimator(self.dlc_config)
        self.kinematic_extractor = KinematicFeatureExtractor(
            fps=self.dlc_config.fps,
            smoothing_window=self.config.smoothing_window
        )
        self.spatial_extractor = SpatialFeatureExtractor()
        self.trajectory_extractor = TrajectoryFeatureExtractor(
            window_size=self.config.smoothing_window * 2
        )

    def extract(
        self,
        dlc_filepath: Union[str, Path],
        video_name: Optional[str] = None
    ) -> ExtractedFeatures:
        """
        Extract all features from a DLC h5 file.

        Args:
            dlc_filepath: Path to DLC h5 file
            video_name: Optional video name for metadata

        Returns:
            ExtractedFeatures object containing all features
        """
        # Load DLC data
        self.dlc_loader.load(dlc_filepath)
        n_frames = self.dlc_loader.n_frames

        if video_name is None:
            video_name = Path(dlc_filepath).stem

        # Extract hand position with likelihood weighting
        hand_pos, hand_conf = self.hand_estimator.estimate(
            self.dlc_loader,
            method="weighted_average" if self.config.use_likelihood_weighting else "simple_average"
        )

        # Get other body part positions
        pellet_pos = self.dlc_loader.get_coords("Pellet")
        nose_pos = self.dlc_loader.get_coords("Nose")

        # Kinematic features
        hand_velocity = self.kinematic_extractor.compute_velocity(hand_pos)
        hand_acceleration = self.kinematic_extractor.compute_acceleration(hand_pos)

        # Spatial features
        hand_pellet_dist = self.spatial_extractor.compute_distance(hand_pos, pellet_pos)
        hand_pellet_angle = self.spatial_extractor.compute_angle(hand_pos, pellet_pos)
        nose_pellet_dist = self.spatial_extractor.compute_distance(nose_pos, pellet_pos)

        # Trajectory features
        straightness = self.trajectory_extractor.compute_straightness(hand_pos)
        smoothness = self.trajectory_extractor.compute_smoothness(hand_pos, fps=self.dlc_config.fps)
        curvature = self.trajectory_extractor.compute_curvature(hand_pos)

        return ExtractedFeatures(
            hand_position=hand_pos,
            hand_confidence=hand_conf,
            pellet_position=pellet_pos,
            nose_position=nose_pos,
            hand_velocity=hand_velocity,
            hand_acceleration=hand_acceleration,
            hand_pellet_distance=hand_pellet_dist,
            hand_pellet_angle=hand_pellet_angle,
            nose_pellet_distance=nose_pellet_dist,
            straightness=straightness,
            smoothness=smoothness,
            curvature=curvature,
            n_frames=n_frames,
            video_name=video_name
        )

    def extract_window(
        self,
        features: ExtractedFeatures,
        center_frame: int,
        window_size: int
    ) -> np.ndarray:
        """
        Extract features for a window centered on a frame.

        Args:
            features: ExtractedFeatures object
            center_frame: Center frame index
            window_size: Total window size (should be odd)

        Returns:
            Feature array of shape (window_size, n_features)
        """
        half_window = window_size // 2
        start = max(0, center_frame - half_window)
        end = min(features.n_frames, center_frame + half_window + 1)

        # Get feature array
        feature_array = features.to_array()

        # Extract window with padding if needed
        window = feature_array[start:end]

        # Pad if window is at edge
        if len(window) < window_size:
            pad_before = half_window - (center_frame - start)
            pad_after = window_size - len(window) - pad_before
            window = np.pad(
                window,
                ((max(0, pad_before), max(0, pad_after)), (0, 0)),
                mode='edge'
            )

        return window

    def extract_sequence(
        self,
        features: ExtractedFeatures,
        start_frame: int,
        end_frame: int
    ) -> np.ndarray:
        """
        Extract features for a frame sequence.

        Args:
            features: ExtractedFeatures object
            start_frame: Start frame index (inclusive)
            end_frame: End frame index (exclusive)

        Returns:
            Feature array of shape (sequence_length, n_features)
        """
        feature_array = features.to_array()
        return feature_array[start_frame:end_frame]


# ==============================================================================
# NORMALIZATION
# ==============================================================================

class FeatureNormalizer:
    """
    Normalize features using z-score or min-max normalization.
    """

    def __init__(self, method: str = "zscore"):
        self.method = method
        self.stats: Dict[str, np.ndarray] = {}

    def fit(self, features: np.ndarray) -> "FeatureNormalizer":
        """
        Compute normalization statistics from training data.

        Args:
            features: Shape (n_samples, n_features) or (n_samples, seq_len, n_features)
        """
        # Flatten if 3D
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])

        if self.method == "zscore":
            self.stats["mean"] = np.nanmean(features, axis=0)
            self.stats["std"] = np.nanstd(features, axis=0)
            self.stats["std"] = np.maximum(self.stats["std"], 1e-8)  # Avoid division by zero
        elif self.method == "minmax":
            self.stats["min"] = np.nanmin(features, axis=0)
            self.stats["max"] = np.nanmax(features, axis=0)
            self.stats["range"] = self.stats["max"] - self.stats["min"]
            self.stats["range"] = np.maximum(self.stats["range"], 1e-8)

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply normalization to features.

        Args:
            features: Shape (n_samples, n_features) or (n_samples, seq_len, n_features)

        Returns:
            Normalized features with same shape
        """
        original_shape = features.shape

        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])

        if self.method == "zscore":
            normalized = (features - self.stats["mean"]) / self.stats["std"]
        elif self.method == "minmax":
            normalized = (features - self.stats["min"]) / self.stats["range"]
        else:
            normalized = features

        # Handle NaN values
        normalized = np.nan_to_num(normalized, nan=0.0)

        return normalized.reshape(original_shape)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(features).transform(features)

    def save(self, filepath: Path) -> None:
        """Save normalization statistics."""
        np.savez(filepath, **self.stats, method=self.method)

    def load(self, filepath: Path) -> "FeatureNormalizer":
        """Load normalization statistics."""
        data = np.load(filepath, allow_pickle=True)
        self.method = str(data["method"])
        self.stats = {k: data[k] for k in data.files if k != "method"}
        return self


# ==============================================================================
# MAIN / TESTING
# ==============================================================================

if __name__ == "__main__":
    from config import get_config

    config = get_config()

    # Find a DLC file
    dlc_files = list(config.paths.processing_dir.glob(config.paths.dlc_pattern))

    if dlc_files:
        print(f"Found {len(dlc_files)} DLC files")
        print(f"\nTesting feature extraction on: {dlc_files[0].name}")

        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract(dlc_files[0])

        print(f"\nExtracted features:")
        print(f"  - Number of frames: {features.n_frames}")
        print(f"  - Hand position shape: {features.hand_position.shape}")
        print(f"  - Hand velocity shape: {features.hand_velocity.shape}")
        print(f"  - Hand-pellet distance range: [{features.hand_pellet_distance.min():.2f}, {features.hand_pellet_distance.max():.2f}]")

        # Convert to array
        feature_array = features.to_array()
        print(f"\nCombined feature array shape: {feature_array.shape}")
        print(f"Feature names: {features.feature_names}")

        # Test window extraction
        window = extractor.extract_window(features, center_frame=1000, window_size=61)
        print(f"\nWindow feature shape: {window.shape}")

        # Test normalization
        normalizer = FeatureNormalizer(method="zscore")
        normalized = normalizer.fit_transform(feature_array)
        print(f"\nNormalized features - mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")

    else:
        print("No DLC files found in processing directory")
