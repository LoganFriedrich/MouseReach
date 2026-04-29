"""
reach_detector_v8.py - Visibility-run + direction-reversal reach detector (v7.1.0)

ALGORITHM SUMMARY
=================
Replaces the v6 state-machine (nose engagement gate, START_CONFIRM delay,
retraction detection, return-to-start detection, boundary polishing) with
a simpler two-phase architecture:

  Phase 1  Visibility runs. A visibility run is a contiguous span of frames
           where at least one hand keypoint has DLC likelihood >= 0.5. A
           run ends when ALL hand keypoints drop below 0.5 for
           DISAPPEAR_THRESHOLD consecutive frames. DISAPPEAR_THRESHOLD is
           1 here -- empirically, ~79% of rapid-fire GT reach boundaries
           are 1+ frame visibility gaps, so DT=1 captures them directly.

  Phase 2  Within each visibility run, segment by "shoot onsets". A
           shoot onset is a frame where mean hand position (across visible
           keypoints, smoothed over 3 frames) has frame-to-frame velocity
           (dx, dy) satisfying dx < -SHOOT_MAG and dy > +SHOOT_MAG.
           This detects the distinctive "down-left" initiation of a reach
           from the reset position. Each shoot onset (preceded by a frame
           where the condition was false) starts a new reach. A minimum
           inter-shoot gap prevents retrigger within a single reach's
           noise.

  Filter   Nose-engagement filter: each candidate reach's frame set must
           have at least NOSE_ENG_MIN fraction of frames where Nose is
           within 25 px of the slit center. Drops non-reach hand-visible
           events (mouse not oriented toward slit).

  Filter   Minimum duration: reaches shorter than MIN_REACH_DURATION
           frames are dropped.

BIOMECHANICAL MODEL
===================
Image coordinates: slit is a horizontal line across the frame; mouse body
(and BOXL/BOXR) are above the slit (smaller y); scoring area (SA) and
tray are below (larger y). A reach is a "down-left" shoot of the paw
from near the slit into the SA, followed by extension and retract back
toward the slit.

KEY PARAMETERS
==============
| Parameter             | Value    | Rationale                             |
|-----------------------|----------|---------------------------------------|
| DISAPPEAR_THRESHOLD   | 1 frame  | Catches 79% of rapid-fire boundaries  |
| SHOOT_MAG             | 0.5 px   | Per-frame velocity lower bound        |
| MIN_INTER_SHOOT_GAP   | 4 frames | Debounce within-reach oscillation     |
| NOSE_ENG_MIN          | 0.3      | Drops non-reach hand-visible events   |
| MIN_REACH_DURATION    | 2 frames | Drops 1-frame noise                   |
| NOSE_ENGAGEMENT_PX    | 25 px    | Distance from slit center             |
| HAND_LIKELIHOOD       | 0.5      | DLC threshold for paw visibility      |

PERFORMANCE (47 GT videos)
==========================
| Metric              | v6 baseline | v8 (this)  |
|---------------------|-------------|------------|
| GT coverage +-10f   | 94.6%       | 99.0%      |
| True misses (>30f)  | 40          | 17         |
| False positives     | 280         | 393        |
| False splits        | 282         | 268        |
| False merges        | 269         | 79         |
| Exact start match   | unknown     | 83.3%      |

Trade: +113 false positives, but all other metrics strictly better.
Nose-engagement filter keeps FPs from exploding (without it, the detector
produces ~2x GT in algo reaches).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .geometry import (
    compute_segment_geometry,
    get_boxr_reference,
    load_dlc,
    load_segments,
)
from .reach_detector import (
    Reach,
    SegmentReaches,
    VideoReaches,
)

VERSION = "7.2.0"
HAND_BODYPARTS = ('RightHand', 'RHLeft', 'RHOut', 'RHRight')

# v7.2.0: pose-alignment computation for per-reach feature emission.
# See feedback_data_driven_rule_design.md and the v4.0.0 outcome
# walkthrough's case-39 thread for context. Not gated on currently;
# emitted as a per-reach feature for downstream analysis.
POSE_LK_THRESHOLD = 0.7
POSE_CONTEXT_PRE = 60
POSE_CONTEXT_POST = 60
POSE_MIN_CO_CONFIDENT_FRAMES = 30

# Thresholds (calibrated on 47-video GT corpus; see module docstring)
DISAPPEAR_THRESHOLD = 1
SHOOT_MAG = 0.5
MIN_INTER_SHOOT_GAP = 4
NOSE_ENG_MIN = 0.3
MIN_REACH_DURATION = 2
NOSE_ENGAGEMENT_PX = 25.0
HAND_LIKELIHOOD = 0.5


def _visibility_runs(vis: np.ndarray, seg_start: int, seg_end: int,
                     disappear_threshold: int = DISAPPEAR_THRESHOLD
                     ) -> List[tuple[int, int]]:
    """Find contiguous spans of visibility within [seg_start, seg_end).

    A run ends after `disappear_threshold` consecutive invisible frames.
    """
    runs: List[tuple[int, int]] = []
    run_start = None
    gap = 0
    for f in range(seg_start, seg_end):
        if vis[f]:
            if run_start is None:
                run_start = f
            gap = 0
        else:
            if run_start is not None:
                gap += 1
                if gap >= disappear_threshold:
                    runs.append((run_start, f - gap))
                    run_start = None
                    gap = 0
    if run_start is not None:
        runs.append((run_start, seg_end - 1))
    return runs


def _detect_reaches_in_run(run_start: int, run_end: int,
                           shoot: np.ndarray,
                           nose_engaged: np.ndarray,
                           ) -> List[tuple[int, int]]:
    """Segment a visibility run into reaches by shoot onsets.

    First reach starts at run_start. Each subsequent shoot onset (f where
    shoot[f] is True and shoot[f-1] was False) starts a new reach,
    provided it is > 2 frames past the run start and >= MIN_INTER_SHOOT_GAP
    past the previous accepted start.
    """
    starts = [run_start]
    prev = False
    for f in range(run_start, run_end + 1):
        cur = bool(shoot[f])
        if cur and not prev and f > run_start + 2 and (f - starts[-1]) >= MIN_INTER_SHOOT_GAP:
            starts.append(f)
        prev = cur

    reaches: List[tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - 1 if i + 1 < len(starts) else run_end
        if e - s + 1 < MIN_REACH_DURATION:
            continue
        eng = nose_engaged[s:e + 1]
        if len(eng) == 0 or eng.sum() / len(eng) < NOSE_ENG_MIN:
            continue
        reaches.append((s, e))
    return reaches


def _hand_arrays(df: pd.DataFrame):
    """Extract hand x/y/likelihood arrays (4 kp x n_frames each)."""
    xs = np.stack([df[f'{bp}_x'].values for bp in HAND_BODYPARTS])
    ys = np.stack([df[f'{bp}_y'].values for bp in HAND_BODYPARTS])
    lks = np.stack([df[f'{bp}_likelihood'].values for bp in HAND_BODYPARTS])
    return xs, ys, lks


def _smoothed_hand_center(xs: np.ndarray, ys: np.ndarray, lks: np.ndarray,
                          smooth_win: int = 3):
    """Mean of visible keypoints per frame, smoothed over a rolling window."""
    vis = lks >= HAND_LIKELIHOOD
    hx = np.nanmean(np.where(vis, xs, np.nan), axis=0)
    hy = np.nanmean(np.where(vis, ys, np.nan), axis=0)
    hxs = pd.Series(hx).rolling(smooth_win, center=True, min_periods=1).mean().values
    hys = pd.Series(hy).rolling(smooth_win, center=True, min_periods=1).mean().values
    return hxs, hys, vis.any(axis=0)


def _nose_engaged(df: pd.DataFrame, slit_x: float, slit_y: float) -> np.ndarray:
    nose_x = df['Nose_x'].values
    nose_y = df['Nose_y'].values
    nose_lk = df['Nose_likelihood'].values
    dist = np.sqrt((nose_x - slit_x) ** 2 + (nose_y - slit_y) ** 2)
    return (nose_lk >= HAND_LIKELIHOOD) & (dist < NOSE_ENGAGEMENT_PX)


def _slit_center(df: pd.DataFrame) -> tuple[float, float]:
    sx = (df['BOXL_x'].median() + df['BOXR_x'].median()) / 2
    sy = (df['BOXL_y'].median() + df['BOXR_y'].median()) / 2
    return float(sx), float(sy)


def _pose_alignment_per_frame(df: pd.DataFrame, slit_x: float, slit_y: float
                              ) -> np.ndarray:
    """Per-frame cosine alignment between mouse facing vector and
    toward-slit vector. NaN where Nose or both Ears are below the
    co-confidence threshold."""
    nose_x = df['Nose_x'].values
    nose_y = df['Nose_y'].values
    nose_lk = df['Nose_likelihood'].values
    le_x = df['LeftEar_x'].values; le_y = df['LeftEar_y'].values
    le_lk = df['LeftEar_likelihood'].values
    re_x = df['RightEar_x'].values; re_y = df['RightEar_y'].values
    re_lk = df['RightEar_likelihood'].values

    nose_ok = nose_lk >= POSE_LK_THRESHOLD
    le_ok = le_lk >= POSE_LK_THRESHOLD
    re_ok = re_lk >= POSE_LK_THRESHOLD
    both_ok = le_ok & re_ok
    one_ok = le_ok | re_ok
    co_conf = nose_ok & one_ok

    ear_mx = np.where(both_ok, (le_x + re_x) / 2, np.where(le_ok, le_x, re_x))
    ear_my = np.where(both_ok, (le_y + re_y) / 2, np.where(le_ok, le_y, re_y))

    fx = nose_x - ear_mx
    fy = nose_y - ear_my
    sx = slit_x - ear_mx
    sy = slit_y - ear_my
    fmag = np.sqrt(fx ** 2 + fy ** 2)
    smag = np.sqrt(sx ** 2 + sy ** 2)
    valid = co_conf & (fmag > 1e-3) & (smag > 1e-3)
    align = np.where(valid, (fx * sx + fy * sy) / (fmag * smag + 1e-9), np.nan)
    return align


def _reach_pose_alignment(alignment: np.ndarray, start: int, end: int,
                          n_frames: int) -> Optional[float]:
    """Median pose alignment over [start - context, end + context] using
    only co-confident frames. Returns None if fewer than the minimum
    required co-confident frames."""
    a = max(0, start - POSE_CONTEXT_PRE)
    b = min(n_frames, end + POSE_CONTEXT_POST + 1)
    window = alignment[a:b]
    valid = window[~np.isnan(window)]
    if len(valid) < POSE_MIN_CO_CONFIDENT_FRAMES:
        return None
    return round(float(np.median(valid)), 4)


class ReachDetectorV8:
    """v7.1.0 reach detector. Same entry-point signature as v6's
    ReachDetector.detect(dlc_path, segments_path) -> VideoReaches."""

    def __init__(self):
        self._global_reach_id = 0

    def detect(self, dlc_path: Path, segments_path: Path) -> VideoReaches:
        self._global_reach_id = 0

        df = load_dlc(dlc_path)
        boundaries = load_segments(segments_path)
        video_name = Path(dlc_path).stem
        if 'DLC_' in video_name:
            video_name = video_name.split('DLC_')[0]
        boxr_x = get_boxr_reference(df)

        xs, ys, lks = _hand_arrays(df)
        hxs, hys, any_vis = _smoothed_hand_center(xs, ys, lks)
        slit_x, slit_y = _slit_center(df)
        nose_engaged = _nose_engaged(df, slit_x, slit_y)
        # v7.2.0: per-frame pose alignment for emitting per-reach feature
        pose_alignment = _pose_alignment_per_frame(df, slit_x, slit_y)
        n_total_frames = len(df)

        # Per-frame velocity + shoot condition
        dx = np.concatenate([[0.0], np.diff(hxs)])
        dy = np.concatenate([[0.0], np.diff(hys)])
        shoot = (dx < -SHOOT_MAG) & (dy > SHOOT_MAG) & ~np.isnan(dx) & ~np.isnan(dy)

        n_segments = len(boundaries)
        segment_results: List[SegmentReaches] = []
        total_reaches = 0
        all_durations: List[int] = []
        all_extents: List[float] = []

        for seg_idx in range(n_segments):
            seg_start = boundaries[seg_idx]
            seg_end = boundaries[seg_idx + 1] if seg_idx + 1 < len(boundaries) else len(df)
            segment_num = seg_idx + 1
            geom = compute_segment_geometry(df, seg_start, seg_end, segment_num)
            ruler_px = geom.ruler_pixels

            reaches: List[Reach] = []
            reach_num = 0
            for (rs, re) in _visibility_runs(any_vis, seg_start, seg_end):
                for (s, e) in _detect_reaches_in_run(rs, re, shoot, nose_engaged):
                    apex_frame, max_x = self._find_apex(s, e, xs, lks)
                    if apex_frame is None:
                        apex_frame = (s + e) // 2
                        max_x = boxr_x
                    reach_num += 1
                    self._global_reach_id += 1
                    extent_pixels = max_x - boxr_x
                    extent_ruler = extent_pixels / ruler_px if ruler_px > 0 else 0.0
                    reaches.append(Reach(
                        reach_id=self._global_reach_id,
                        reach_num=reach_num,
                        start_frame=int(s),
                        apex_frame=int(apex_frame),
                        end_frame=int(e),
                        duration_frames=int(e - s + 1),
                        max_extent_pixels=float(round(extent_pixels, 1)),
                        max_extent_ruler=float(round(extent_ruler, 3)),
                        confidence=None,
                        start_confidence=None,
                        end_confidence=None,
                        pose_alignment=_reach_pose_alignment(
                            pose_alignment, s, e, n_total_frames),
                    ))

            segment_results.append(SegmentReaches(
                segment_num=segment_num,
                start_frame=int(seg_start),
                end_frame=int(seg_end),
                ruler_pixels=round(float(ruler_px), 1),
                n_reaches=len(reaches),
                reaches=reaches,
                flagged_for_review=False,
                flag_reason=None,
            ))
            total_reaches += len(reaches)
            all_durations.extend([r.duration_frames for r in reaches])
            all_extents.extend([r.max_extent_ruler for r in reaches])

        summary = {
            'total_reaches': total_reaches,
            'n_segments': n_segments,
            'reaches_per_segment_mean': round(total_reaches / n_segments, 1) if n_segments else 0,
            'reaches_per_segment_std': float(round(np.std([s.n_reaches for s in segment_results]), 1)) if segment_results else 0.0,
            'mean_duration_frames': float(round(np.mean(all_durations), 1)) if all_durations else 0.0,
            'mean_extent_ruler': float(round(np.mean(all_extents), 3)) if all_extents else 0.0,
        }

        return VideoReaches(
            detector_version=VERSION,
            video_name=video_name,
            total_frames=len(df),
            boxr_x=round(boxr_x, 1),
            n_segments=n_segments,
            segments=segment_results,
            summary=summary,
            detected_at=datetime.now().isoformat(),
            validated=False,
            validated_by=None,
            validated_at=None,
            corrections_made=0,
            reaches_added=0,
            reaches_removed=0,
            segments_flagged=0,
        )

    @staticmethod
    def _find_apex(start: int, end: int, xs: np.ndarray, lks: np.ndarray):
        """Apex frame = frame with max hand_x across 4 keypoints, likelihood
        >= HAND_LIKELIHOOD. Returns (apex_frame, max_x) or (None, None)."""
        best_frame = None
        best_x = -np.inf
        for f in range(start, end + 1):
            for kp in range(lks.shape[0]):
                if lks[kp, f] >= HAND_LIKELIHOOD:
                    x = xs[kp, f]
                    if x > best_x:
                        best_x = x
                        best_frame = f
        if best_frame is None:
            return None, None
        return best_frame, float(best_x)

    @staticmethod
    def save_results(results: VideoReaches, output_path: Path,
                     validation_status: str = "needs_review") -> None:
        """Save results to JSON. Same format as v6 to preserve downstream
        consumers (review widget, outcome classifier, db sync)."""
        data = asdict(results)
        data["validation_status"] = validation_status
        data["validation_timestamp"] = datetime.now().isoformat()
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def detect_reaches_v8(dlc_path: str, segments_path: str) -> VideoReaches:
    """Convenience function mirroring the v6 detect_reaches signature."""
    detector = ReachDetectorV8()
    return detector.detect(Path(dlc_path), Path(segments_path))
