"""
Corrective Watchdog Model for MouseReach Analysis.

The watchdog learns from human corrections to:
1. Flag likely algorithm errors BEFORE human review
2. Suggest corrections based on DLC features
3. Learn WHY the algorithm fails in specific cases

Key insight from failure analysis:
- Algorithm struggles most with "retrieved" outcomes (calls them displaced/untouched)
- Reach end frames are harder to identify than start frames
- Reach splitting (when to split vs merge) needs human judgment

Architecture: Two-stage watchdog
Stage 1: Binary classifier - "Does this need human review?"
Stage 2: If flagged, suggest the likely correct answer
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")


@dataclass
class WatchdogConfig:
    """Configuration for watchdog model."""
    # Feature extraction
    window_before: int = 30  # frames before event
    window_after: int = 30   # frames after event
    fps: int = 60

    # Model architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 50


# =============================================================================
# WATCHDOG TASK 1: OUTCOME CORRECTION
# =============================================================================

class OutcomeWatchdogDataset(Dataset):
    """
    Dataset for outcome correction watchdog.

    Training signal: cases where algo != GT (human corrected)
    Features: DLC data around interaction frame
    Labels:
        - Binary: was_corrected (True/False)
        - Multi-class: correct_outcome (if corrected)
    """

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load all algo-vs-GT comparison data."""
        for gt_file in DATA_DIR.glob("*_unified_ground_truth.json"):
            video = gt_file.stem.replace("_unified_ground_truth", "")

            # Load GT
            with open(gt_file) as f:
                gt = json.load(f)

            # Load algo outcomes
            algo_file = ALGO_DIR / f"{video}_pellet_outcomes.json"
            if not algo_file.exists():
                continue
            with open(algo_file) as f:
                algo = json.load(f)

            # Load DLC data
            dlc_pattern = f"{video}DLC*.h5"
            dlc_files = list(DATA_DIR.glob(dlc_pattern))
            if not dlc_files:
                continue
            dlc_df = pd.read_hdf(dlc_files[0])

            # Get segment boundaries for frame estimation
            boundaries = [b['frame'] for b in gt.get('segmentation', {}).get('boundaries', [])]

            # Match GT and algo outcomes
            gt_outcomes = {s['segment_num']: s for s in gt.get('outcomes', {}).get('segments', [])
                          if s.get('determined', False)}
            algo_outcomes = {s['segment_num']: s for s in algo.get('segments', [])}

            for seg_num in gt_outcomes:
                gt_seg = gt_outcomes[seg_num]
                algo_seg = algo_outcomes.get(seg_num)

                if algo_seg is None:
                    continue

                # Get frame for feature extraction
                # Priority: interaction_frame > midpoint of segment
                interaction_frame = gt_seg.get('interaction_frame') or algo_seg.get('interaction_frame')

                if interaction_frame is None:
                    # Use segment midpoint instead
                    if seg_num <= len(boundaries) and seg_num >= 1:
                        start = boundaries[seg_num - 1] if seg_num > 1 else 0
                        end = boundaries[seg_num] if seg_num < len(boundaries) else len(dlc_df) - 1
                        interaction_frame = (start + end) // 2
                    else:
                        continue

                # Extract DLC features around interaction
                features = self._extract_features(dlc_df, interaction_frame)
                if features is None:
                    continue

                # Labels
                algo_outcome = algo_seg['outcome']
                gt_outcome = gt_seg['outcome']
                was_corrected = algo_outcome != gt_outcome

                self.samples.append({
                    'video': video,
                    'segment': seg_num,
                    'features': features,
                    'algo_outcome': algo_outcome,
                    'gt_outcome': gt_outcome,
                    'was_corrected': was_corrected,
                    'interaction_frame': interaction_frame
                })

        print(f"Loaded {len(self.samples)} outcome samples")
        print(f"  Corrected: {sum(1 for s in self.samples if s['was_corrected'])}")
        print(f"  Not corrected: {sum(1 for s in self.samples if not s['was_corrected'])}")

    def _extract_features(self, dlc_df: pd.DataFrame, center_frame: int) -> Optional[np.ndarray]:
        """Extract DLC features around a frame."""
        start = center_frame - self.config.window_before
        end = center_frame + self.config.window_after

        if start < 0 or end >= len(dlc_df):
            return None

        # Get key body parts
        key_parts = ['RightHand', 'RHLeft', 'RHRight', 'RHOut', 'Pellet', 'Nose']

        features = []
        for part in key_parts:
            try:
                part_data = dlc_df.xs(part, level=1, axis=1)
                x = part_data['x'].values[start:end+1]
                y = part_data['y'].values[start:end+1]
                likelihood = part_data['likelihood'].values[start:end+1]
                features.extend([x, y, likelihood])
            except KeyError:
                # Part not found, skip
                continue

        if not features:
            return None

        return np.array(features).T  # Shape: (window_size, n_features)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'features': torch.tensor(sample['features'], dtype=torch.float32),
            'was_corrected': torch.tensor(sample['was_corrected'], dtype=torch.float32),
            'algo_outcome': sample['algo_outcome'],
            'gt_outcome': sample['gt_outcome']
        }


# =============================================================================
# WATCHDOG TASK 2: REACH END REFINEMENT
# =============================================================================

class ReachEndWatchdogDataset(Dataset):
    """
    Dataset for reach end frame correction.

    Training signal: cases where human moved the reach end frame
    Features: DLC data around the reach end
    Labels: correction magnitude and direction
    """

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load all reach end correction data."""
        for gt_file in DATA_DIR.glob("*_unified_ground_truth.json"):
            video = gt_file.stem.replace("_unified_ground_truth", "")

            with open(gt_file) as f:
                gt = json.load(f)

            # Load DLC
            dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
            if not dlc_files:
                continue
            dlc_df = pd.read_hdf(dlc_files[0])

            # NOTE: v2 GT files no longer track corrections (no original_end_frame or end_corrected fields).
            # This correction-based training requires v1 archive GT files.
            # TODO: Rework this section to train on other signals, or load v1 archive data.

            # Get reaches with end determinations
            # for reach in gt.get('reaches', {}).get('reaches', []):
            #     # Check if end was determined
            #     if not reach.get('end_determined', False):
            #         continue
            #
            #     end_frame = reach.get('end_frame')
            #     # v2 GT files no longer have original_end_frame or end_corrected
            #     # This training approach needs rethinking for v2 schema
            #     pass

        print(f"Loaded {len(self.samples)} reach end samples")
        if self.samples:
            corrections = [s['correction'] for s in self.samples]
            print(f"  Mean correction: {np.mean(corrections):.1f} frames")
            print(f"  Std correction: {np.std(corrections):.1f} frames")

    def _extract_features(self, dlc_df: pd.DataFrame, center_frame: int) -> Optional[np.ndarray]:
        """Extract features around reach end."""
        # Same as outcome extraction
        start = center_frame - self.config.window_before
        end = center_frame + self.config.window_after

        if start < 0 or end >= len(dlc_df):
            return None

        key_parts = ['RightHand', 'RHLeft', 'RHRight', 'RHOut', 'Pellet', 'Nose']

        features = []
        for part in key_parts:
            try:
                part_data = dlc_df.xs(part, level=1, axis=1)
                x = part_data['x'].values[start:end+1]
                y = part_data['y'].values[start:end+1]
                features.extend([x, y])
            except KeyError:
                continue

        if not features:
            return None

        return np.array(features).T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'features': torch.tensor(sample['features'], dtype=torch.float32),
            'correction': torch.tensor(sample['correction'], dtype=torch.float32)
        }


# =============================================================================
# WATCHDOG MODELS
# =============================================================================

class OutcomeWatchdog(nn.Module):
    """
    Two-head watchdog for outcome classification:
    Head 1: Binary - "Does this need correction?"
    Head 2: Multi-class - "What should the outcome be?"
    """

    def __init__(self, input_size: int, config: WatchdogConfig):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=True,
            batch_first=True
        )

        hidden_dim = config.hidden_size * 2  # bidirectional

        # Head 1: Binary classifier - needs correction?
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

        # Head 2: Outcome classifier
        self.outcome_classes = ['retrieved', 'displaced_sa', 'displaced_outside', 'untouched']
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, len(self.outcome_classes))
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        h_forward = h_n[-2]  # Last layer, forward
        h_backward = h_n[-1]  # Last layer, backward
        h = torch.cat([h_forward, h_backward], dim=1)

        # Two heads
        needs_correction = torch.sigmoid(self.correction_head(h))
        outcome_logits = self.outcome_head(h)

        return needs_correction, outcome_logits


class ReachEndWatchdog(nn.Module):
    """
    Regression model to predict reach end correction.

    Output: predicted correction in frames (+ = extend, - = shorten)
    """

    def __init__(self, input_size: int, config: WatchdogConfig):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=True,
            batch_first=True
        )

        hidden_dim = config.hidden_size * 2

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        correction = self.regressor(h)
        return correction


# =============================================================================
# USAGE ANALYSIS
# =============================================================================

def analyze_watchdog_opportunity():
    """Analyze what patterns the watchdog could learn."""
    print("\n" + "="*70)
    print("WATCHDOG OPPORTUNITY ANALYSIS")
    print("="*70)

    config = WatchdogConfig()

    # Load outcome data
    print("\n--- Outcome Correction Watchdog ---")
    outcome_ds = OutcomeWatchdogDataset(config)

    if outcome_ds.samples:
        # Analyze correction patterns
        corrected = [s for s in outcome_ds.samples if s['was_corrected']]

        print(f"\nCorrection patterns:")
        from collections import Counter
        patterns = Counter((s['algo_outcome'], s['gt_outcome']) for s in corrected)
        for (algo, gt), count in patterns.most_common():
            print(f"  {algo} -> {gt}: {count}")

        # Key insight: most corrections are algo calling retrieved wrong
        retrieved_errors = [s for s in corrected if s['gt_outcome'] == 'retrieved']
        print(f"\n'Retrieved' misclassifications: {len(retrieved_errors)}")
        print("  This is the PRIMARY target for watchdog learning")

    # Load reach end data
    print("\n--- Reach End Correction Watchdog ---")
    reach_ds = ReachEndWatchdogDataset(config)

    if reach_ds.samples:
        corrections = [s['correction'] for s in reach_ds.samples]
        print(f"\nCorrection statistics:")
        print(f"  Range: {min(corrections)} to {max(corrections)} frames")
        print(f"  Most corrections are small: {sum(1 for c in corrections if abs(c) <= 5)}/{len(corrections)}")


if __name__ == "__main__":
    analyze_watchdog_opportunity()
