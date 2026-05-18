"""
Reach detection v8 -- pattern-discovery based reach detector.

Replaces the rules-based v7.x.x detector. GBM-on-per-frame-features
architecture with boundary sample weighting (BSW) at b1 w=0.8.

Production model trained on the full 47-video GT corpus (20 exhaustive
videos used for training, per the calibration convention). Holdout
generalization test passed (precision 84.8%, recall 91.8% on 328
exhaustive holdout reaches; boundary deltas median=0).

Operates at the WHOLE-VIDEO level (no segment knowledge). Each detected
reach is a (start_frame, end_frame) pair; reach assignment to segments
and pellets is the responsibility of the separate `assignment` algo.

See `four_algo_decomposition.md` in cross-session memory and
`Improvement_Snapshots/reach_detection/v8.0.0_holdout_generalization_BSW_w0.8/`
for the holdout test result. The calibration corpus + CV-fold
definition is at `Improvement_Snapshots/_corpus/2026-04-30_restart_inventory/`.

Usage:
    from mousereach.reach.v8 import detect_reaches_v8
    from mousereach.reach.v8.features import load_dlc_h5
    dlc = load_dlc_h5("video_DLC.h5")
    reaches = detect_reaches_v8(dlc)
    # reaches is a list of (start_frame, end_frame) tuples
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib

VERSION = "8.0.1"

# Default production model path. Resolved relative to this file so the
# package ships with the artifact and a custom path can be passed at
# runtime if needed.
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "v8.0.0_bsw_w0.8.joblib"

# Production inference hyperparameters (must match the BSW w=0.8
# calibration to keep behaviour consistent with training).
DEFAULT_THRESHOLD = 0.5
DEFAULT_MERGE_GAP = 0
DEFAULT_MIN_SPAN = 3

_MODEL_CACHE = {}


def _load_model(model_path: Path):
    key = str(model_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    bundle = joblib.load(model_path)
    _MODEL_CACHE[key] = bundle
    return bundle


def detect_reaches_v8(
    dlc_df,
    model_path: Optional[Path] = None,
    threshold: float = DEFAULT_THRESHOLD,
    merge_gap: int = DEFAULT_MERGE_GAP,
    min_span: int = DEFAULT_MIN_SPAN,
) -> List[Tuple[int, int]]:
    """Run the v8 production reach detector on a DLC trajectory dataframe.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        DLC trajectories as loaded by `mousereach.reach.v8.features.load_dlc_h5`.
    model_path : Path, optional
        Path to the joblib-saved model bundle. Defaults to the bundled
        production model `v8.0.0_bsw_w0.8.joblib`.
    threshold : float
        Per-frame probability threshold for the binary reach call.
    merge_gap : int
        Consecutive sub-threshold frames allowed between two runs before
        they are kept separate.
    min_span : int
        Minimum length (in frames) for a run to be emitted as a reach.

    Returns
    -------
    list of (start_frame, end_frame) tuples, video-frame indices.
    """
    from .features import extract_features
    from .postprocess import probabilities_to_reaches

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    bundle = _load_model(Path(model_path))
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]

    feats = extract_features(dlc_df)
    X = feats[feat_cols].to_numpy(dtype="float32")
    proba = model.predict_proba(X)[:, 1]

    spans = probabilities_to_reaches(
        proba, threshold=threshold, merge_gap=merge_gap, min_span=min_span)
    return [(int(s.start_frame), int(s.end_frame)) for s in spans]
