"""
boundary_polisher.py - ML-based boundary refinement for reach detection.

Takes coarse reach boundaries from the rule-based detector and applies
learned corrections using XGBoost models trained on ground truth data.

Two-stage conservative correction:
    Stage 1: Classifier predicts P(boundary needs correction)
    Stage 2: Regressor predicts the correction offset (in frames)
    Only applies correction when classifier confidence exceeds threshold.

This prevents the model from corrupting the ~80% of boundaries that
are already correct while fixing the ~20% that need adjustment.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Default model directory
# __file__ is .../MouseReach/src/mousereach/reach/core/boundary_polisher.py
# 5 parents up = .../MouseReach/  then down to training/models/
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent.parent.parent / "training" / "models"

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
HAND_THRESHOLD = 0.5


class BoundaryPolisher:
    """ML-based boundary correction for reach detection."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.loaded = False
        self.start_reg = None
        self.start_cls = None
        self.end_reg = None
        self.end_cls = None
        self.config = None
        self._load_models()

    def _load_models(self):
        """Load trained XGBoost models and config."""
        if not HAS_XGBOOST:
            return

        config_path = self.model_dir / "boundary_polisher_config.json"
        if not config_path.exists():
            return

        with open(config_path) as f:
            self.config = json.load(f)

        try:
            self.start_reg = xgb.XGBRegressor()
            self.start_reg.load_model(str(self.model_dir / "boundary_polisher_start_reg.json"))
            self.start_cls = xgb.XGBClassifier()
            self.start_cls.load_model(str(self.model_dir / "boundary_polisher_start_cls.json"))
            self.end_reg = xgb.XGBRegressor()
            self.end_reg.load_model(str(self.model_dir / "boundary_polisher_end_reg.json"))
            self.end_cls = xgb.XGBClassifier()
            self.end_cls.load_model(str(self.model_dir / "boundary_polisher_end_cls.json"))
            self.loaded = True
        except Exception:
            self.loaded = False

    def _extract_features(self, arrays, n_frames, boundary_frame, slit_x):
        """Extract feature vector for a single boundary."""
        window = self.config['window']
        n_features_per_frame = 13
        total_frames = 2 * window + 1
        features = np.full(n_features_per_frame * total_frames + 4, np.nan)

        prev_best_x = None
        for i, offset in enumerate(range(-window, window + 1)):
            frame = boundary_frame + offset
            base = i * n_features_per_frame

            if frame < 0 or frame >= n_frames:
                continue

            best_l = 0
            best_x = None
            visible_xs = []

            for pi, p in enumerate(RH_POINTS):
                x = arrays[f'{p}_x'][frame]
                l = arrays[f'{p}_l'][frame]
                features[base + pi] = (x - slit_x) if not np.isnan(x) else np.nan
                features[base + 4 + pi] = l
                if l >= HAND_THRESHOLD:
                    if not np.isnan(x):
                        visible_xs.append(x)
                    if l > best_l:
                        best_l = l
                        best_x = x

            nose_x = arrays['nose_x'][frame]
            nose_l = arrays['nose_l'][frame]
            features[base + 8] = (nose_x - slit_x) if not np.isnan(nose_x) else np.nan
            features[base + 9] = nose_l
            features[base + 10] = (np.mean(visible_xs) - slit_x) if visible_xs else np.nan
            features[base + 11] = len(visible_xs)

            if best_x is not None and prev_best_x is not None:
                features[base + 12] = best_x - prev_best_x
            else:
                features[base + 12] = np.nan

            if best_x is not None:
                prev_best_x = best_x

        return features

    def _precompute_arrays(self, df):
        """Pre-extract numpy arrays from DataFrame for fast feature extraction."""
        n = len(df)
        arrays = {}
        for p in RH_POINTS:
            arrays[f'{p}_x'] = df[f'{p}_x'].values if f'{p}_x' in df.columns else np.full(n, np.nan)
            arrays[f'{p}_l'] = df[f'{p}_likelihood'].values if f'{p}_likelihood' in df.columns else np.zeros(n)
        arrays['nose_x'] = df['Nose_x'].values if 'Nose_x' in df.columns else np.full(n, np.nan)
        arrays['nose_l'] = df['Nose_likelihood'].values if 'Nose_likelihood' in df.columns else np.zeros(n)
        return arrays, n

    def polish_reaches(self, reaches, df, slit_x):
        """Apply boundary corrections to a list of reaches.

        Args:
            reaches: List of Reach objects with start_frame and end_frame
            df: DLC DataFrame
            slit_x: Slit center x position

        Returns:
            List of Reach objects with corrected boundaries
        """
        if not self.loaded or not reaches:
            return reaches

        arrays, n_frames = self._precompute_arrays(df)
        s_params = self.config['start_params']
        e_params = self.config['end_params']
        window = self.config['window']
        max_correction = self.config.get('max_correction', 15)

        for reach in reaches:
            # Extract start features
            start_feats = self._extract_features(arrays, n_frames, reach.start_frame, slit_x)
            ctx_idx = 13 * (2 * window + 1)
            start_feats[ctx_idx] = reach.duration_frames
            start_feats[ctx_idx + 1] = reach.max_extent_pixels
            start_feats[ctx_idx + 2] = reach.confidence
            start_feats[ctx_idx + 3] = 0  # start boundary

            # Extract end features
            end_feats = self._extract_features(arrays, n_frames, reach.end_frame, slit_x)
            end_feats[ctx_idx] = reach.duration_frames
            end_feats[ctx_idx + 1] = reach.max_extent_pixels
            end_feats[ctx_idx + 2] = reach.confidence
            end_feats[ctx_idx + 3] = 1  # end boundary

            # Clean NaNs
            start_clean = np.nan_to_num(start_feats, nan=-999).reshape(1, -1)
            end_clean = np.nan_to_num(end_feats, nan=-999).reshape(1, -1)

            # Stage 1: Classify whether correction needed
            start_prob = self.start_cls.predict_proba(start_clean)[0, 1]
            end_prob = self.end_cls.predict_proba(end_clean)[0, 1]

            # Stage 2: Predict correction (only apply if classifier says so)
            start_correction = 0
            if start_prob >= s_params['cls_threshold']:
                pred = self.start_reg.predict(start_clean)[0]
                if abs(pred) >= s_params.get('reg_threshold', 0.5):
                    start_correction = int(np.clip(round(pred), -max_correction, max_correction))

            end_correction = 0
            if end_prob >= e_params['cls_threshold']:
                pred = self.end_reg.predict(end_clean)[0]
                if abs(pred) >= e_params.get('reg_threshold', 0.5):
                    end_correction = int(np.clip(round(pred), -max_correction, max_correction))

            # Apply corrections
            new_start = reach.start_frame + start_correction
            new_end = reach.end_frame + end_correction

            # Sanity checks
            if new_start < 0:
                new_start = 0
            if new_end >= n_frames:
                new_end = n_frames - 1
            if new_end <= new_start:
                # Don't apply if correction creates invalid range
                continue

            reach.start_frame = new_start
            reach.end_frame = new_end
            reach.duration_frames = new_end - new_start + 1

        return reaches
