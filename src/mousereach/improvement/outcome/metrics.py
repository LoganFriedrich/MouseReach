"""
Outcome classification accuracy metrics for the MouseReach Improvement Process.

Matching rule
-------------
For each (video, segment_num) pair present in BOTH GT and algo output:
  - Compare outcome labels (retrieved, displaced_sa, displaced_outside,
    untouched, uncertain, unknown, no_pellet).
  - Compare interaction_frame (signed delta = algo - GT; only for segments
    where both sides have a non-null interaction_frame).
  - Derive GT causal reach from GT reaches + interaction_frame, then compare
    to algo's causal reach using +/-10f start-proximity matching.

Verdict per segment
-------------------
- ``label_correct_untouched``  -- both say untouched (no reach evaluation)
- ``label_and_reach_correct``  -- same outcome label AND causal reach matched
- ``label_correct_wrong_reach`` -- same outcome label BUT causal reach mismatch
- ``label_wrong``              -- outcome labels differ (both committed)
- ``abstained``                -- algo said uncertain/unknown, GT committed

Deliverables written to output_dir
------------------------------------
- ``outcome_per_segment.csv`` -- one row per paired segment.
- ``per_video.csv``           -- one row per video with aggregate stats.
- ``scalars.json``            -- nested summary with confusion matrix,
  interaction-frame histogram, causal-reach breakdown.

Usage::

    from mousereach.improvement.outcome.metrics import (
        compute_outcome_metrics,
        derive_gt_causal_reach,
    )

    scalars = compute_outcome_metrics(
        gt_dir=Path("Y:/.../Processing"),
        algo_dir=Path("Y:/.../Processing"),
        output_dir=Path("Y:/.../metrics"),
    )
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Outcome labels that indicate the pellet was touched (not untouched/uncertain)
TOUCHED_OUTCOMES = {"retrieved", "displaced_sa", "displaced_outside"}
# Outcome labels where algo abstained from committing
ABSTENTION_OUTCOMES = {"uncertain", "unknown"}
# All recognized committed outcomes (not abstention)
COMMITTED_OUTCOMES = {"retrieved", "displaced_sa", "displaced_outside", "untouched"}
# Non-evaluable outcomes -- visible on the confusion matrix but excluded
# from per-class precision/recall denominators. Tail-knockover and
# confirmed-unfixable video-quality cases land here. Both GT and algo can
# emit this label. See abnormal_exception_category_TODO.md memory entry.
NON_EVALUABLE_OUTCOMES = {"abnormal_exception"}

CAUSAL_REACH_WINDOW = 10  # +/- frames for start-proximity matching


# Flag-reason substrings that indicate the algo is explicitly punting on
# causal-reach attribution (no reach data / abnormal exception territory),
# NOT just routine "I classified this confidently but didn't attribute a
# specific reach." Routine retrievals via the Stage 0 grab path return
# cri=None + flagged=True with flag reasons like "Pellet grabbed by paw"
# -- those are still real retrievals and should NOT normalize to
# abnormal_exception. Only the explicit "no reach data" / abnormal-event
# patterns trigger the normalization.
_ABNORMAL_EXCEPTION_FLAG_PATTERNS = (
    "no reach data",       # the case-39 tail-knockover pattern
    "tail-knockover",      # if explicitly tagged
    "abnormal_exception",  # if the algo emits this in flag_reason
)


def _collapse_displaced_outside(outcome: str) -> str:
    """Collapse displaced_outside -> displaced_sa for both algo and GT
    sides.

    Per the user's directive (2026-04-29): the displaced_outside class
    is treated as a sub-case of displaced_sa for the purposes of
    classification accuracy. Whether the pellet ended on-tray vs
    off-tray is a kinematic detail; what matters for the outcome is
    that the pellet was DISPLACED (not retrieved, not untouched).
    """
    if outcome == "displaced_outside":
        return "displaced_sa"
    return outcome


def _normalize_algo_punted_outcome(algo_seg: dict) -> str:
    """Map algo-punted segments to ``abnormal_exception``.

    When the outcome detector emits a non-untouched outcome but
    EXPLICITLY indicates in flag_reason that no reach could be
    attributed (e.g., the case-39 tail-knockover pattern where flag
    reason contains "no reach data"), normalize to ``abnormal_exception``
    so it doesn't deflate per-class metrics.

    The narrower flag-reason check is critical: cri=None + flagged=True
    is also produced by routine Stage 0 grab retrievals
    (`pellet_outcome.py:745-763`), which are real retrievals, not
    abnormal cases. Discriminating via flag_reason content prevents
    those from being miscategorized.

    See memory: ``algo_punted_equivalence.md`` and
    ``abnormal_exception_category_TODO.md``.
    """
    raw = algo_seg.get("outcome", "unknown")
    if raw is None:
        return "unknown"
    # Already in a non-classification state -- leave alone.
    if raw in ("untouched", "unknown", "uncertain", "abnormal_exception"):
        return raw
    # Only normalize when the algo's flag_reason EXPLICITLY indicates
    # an abnormal-exception pattern. Other flagged cases (routine
    # retrievals via Stage 0 grab, etc.) are real classifications that
    # the metrics layer should evaluate as-is.
    if (
        algo_seg.get("causal_reach_id") is None
        and bool(algo_seg.get("flagged_for_review", False))
    ):
        flag_reason = (algo_seg.get("flag_reason") or "").lower()
        if any(p in flag_reason for p in _ABNORMAL_EXCEPTION_FLAG_PATTERNS):
            return "abnormal_exception"
    return raw


# ---------------------------------------------------------------------------
# Pure helper: derive GT causal reach
# ---------------------------------------------------------------------------

def derive_gt_causal_reach(
    gt_segment: dict,
    gt_reaches: list,
) -> Optional[int]:
    """Derive the GT causal reach_id for a touched segment.

    For each GT segment where outcome is in TOUCHED_OUTCOMES:
      1. Filter GT reaches to those with matching segment_num.
      2. Find the reach containing interaction_frame in [start_frame, end_frame].
      3. If none contains it, pick the closest by |start_frame - interaction_frame|.

    Parameters
    ----------
    gt_segment : dict
        GT segment record with keys: segment_num, outcome, interaction_frame.
    gt_reaches : list of dict
        GT reach records with keys: reach_id, segment_num, start_frame, end_frame.

    Returns
    -------
    int or None
        The reach_id of the derived causal reach, or None if no reaches match.
    """
    interaction_frame = gt_segment.get("interaction_frame")
    segment_num = gt_segment.get("segment_num")
    outcome = gt_segment.get("outcome", "")

    if outcome not in TOUCHED_OUTCOMES:
        return None
    if interaction_frame is None:
        return None

    # Filter to reaches in same segment
    seg_reaches = [
        r for r in gt_reaches
        if r.get("segment_num") == segment_num
    ]
    if not seg_reaches:
        return None

    # Pass 1: reach containing interaction_frame
    for r in seg_reaches:
        sf = r.get("start_frame")
        ef = r.get("end_frame")
        if sf is not None and ef is not None:
            if sf <= interaction_frame <= ef:
                return r.get("reach_id")

    # Pass 2: closest by |start_frame - interaction_frame|
    best_id = None
    best_dist = float("inf")
    for r in seg_reaches:
        sf = r.get("start_frame")
        if sf is not None:
            d = abs(sf - interaction_frame)
            if d < best_dist:
                best_dist = d
                best_id = r.get("reach_id")

    return best_id


def _get_reach_start(reach_id: int, reaches: list) -> Optional[int]:
    """Look up start_frame for a reach_id in a list of reach dicts."""
    for r in reaches:
        if r.get("reach_id") == reach_id:
            return r.get("start_frame")
    return None


# ---------------------------------------------------------------------------
# GT / algo loading
# ---------------------------------------------------------------------------

def _load_gt_outcomes(gt_path: Path) -> Tuple[list, list, bool]:
    """Load GT outcome segments and reaches from a unified GT file.

    Returns
    -------
    (gt_segments, gt_reaches, exhaustive)
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_segments = data.get("outcomes", {}).get("segments", [])
    gt_reaches = data.get("reaches", {}).get("reaches", [])
    exhaustive = data.get("reaches", {}).get("exhaustive", False)

    return gt_segments, gt_reaches, exhaustive


def _load_algo_outcomes(algo_path: Path) -> Tuple[list, str]:
    """Load algo outcome segments from a pellet_outcomes.json file.

    Returns
    -------
    (algo_segments, detector_version)
    """
    with open(algo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    algo_segments = data.get("segments", [])
    version = data.get("detector_version", "unknown")
    return algo_segments, version


def _load_algo_reaches(reaches_path: Path) -> list:
    """Load algo reaches from a _reaches.json file for causal reach lookup."""
    if not reaches_path.exists():
        return []
    with open(reaches_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_reaches = []
    for seg in data.get("segments", []):
        for r in seg.get("reaches", []):
            r_copy = dict(r)
            r_copy["segment_num"] = seg.get("segment_num")
            all_reaches.append(r_copy)
    return all_reaches


def _discover_video_ids(gt_dir: Path) -> List[str]:
    """Discover video IDs from unified GT files."""
    unified = list(gt_dir.glob("*_unified_ground_truth.json"))
    if unified:
        return sorted({f.stem.replace("_unified_ground_truth", "") for f in unified})
    return []


def _find_gt_file(gt_dir: Path, video_id: str) -> Optional[Path]:
    """Find GT file for a video ID."""
    p = gt_dir / f"{video_id}_unified_ground_truth.json"
    return p if p.exists() else None


def _find_algo_file(algo_dir: Path, video_id: str) -> Optional[Path]:
    """Find algo pellet_outcomes.json for a video ID."""
    p = algo_dir / f"{video_id}_pellet_outcomes.json"
    return p if p.exists() else None


def _find_algo_reaches_file(algo_dir: Path, video_id: str) -> Optional[Path]:
    """Find algo _reaches.json for a video ID (for causal reach lookup)."""
    p = algo_dir / f"{video_id}_reaches.json"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Causal reach matching
# ---------------------------------------------------------------------------

def _match_causal_reaches(
    algo_causal_start: Optional[int],
    gt_causal_start: Optional[int],
    window: int = CAUSAL_REACH_WINDOW,
) -> bool:
    """Return True if algo and GT causal reach starts are within +/- window.

    DEPRECATED: this comparison is unreliable because GT reach_ids and algo
    reach_ids are in different namespaces, so the start frames being
    compared often refer to different physical reaches. Use
    ``_algo_reach_contains_frame`` instead.
    """
    if algo_causal_start is None or gt_causal_start is None:
        return False
    return abs(algo_causal_start - gt_causal_start) <= window


def _algo_reach_contains_frame(
    algo_causal_id: Optional[int],
    algo_reaches: list,
    frame: Optional[int],
    slack: int = 2,
) -> bool:
    """Return True if the algo reach identified as causal has a window that
    contains the given frame (with +/- slack frames of tolerance).

    This is the correct way to evaluate causal-reach correctness when
    comparing across GT and algo namespaces. The algo's causal reach is
    "right" if its [start_frame, end_frame] window covers the moment GT
    identifies as the causal interaction.
    """
    if algo_causal_id is None or frame is None:
        return False
    for r in algo_reaches:
        if r.get("reach_id") == algo_causal_id:
            s = r.get("start_frame")
            e = r.get("end_frame")
            if s is None or e is None:
                return False
            return (s - slack) <= frame <= (e + slack)
    return False


# ---------------------------------------------------------------------------
# Per-segment verdict
# ---------------------------------------------------------------------------

def _compute_verdict(
    gt_outcome: str,
    algo_outcome: str,
    causal_reach_match: bool,
) -> str:
    """Compute the verdict for a single segment.

    Returns one of:
      - label_correct_untouched
      - label_and_reach_correct
      - label_correct_wrong_reach
      - label_wrong_reach_correct  (label is wrong but algo identified the
        same reach GT marked as causal -- the reach is investigable)
      - label_wrong_reach_wrong    (both label and reach attribution wrong)
      - abstained
    """
    # Abstention: algo said uncertain/unknown, GT committed
    if algo_outcome in ABSTENTION_OUTCOMES and gt_outcome not in ABSTENTION_OUTCOMES:
        return "abstained"

    # Both untouched
    if gt_outcome == "untouched" and algo_outcome == "untouched":
        return "label_correct_untouched"

    # Label match
    if algo_outcome == gt_outcome:
        if gt_outcome == "untouched":
            return "label_correct_untouched"
        elif causal_reach_match:
            return "label_and_reach_correct"
        else:
            return "label_correct_wrong_reach"

    # Label mismatch -- still useful to know whether the algo at least
    # identified the same physical reach GT considered causal.
    if causal_reach_match:
        return "label_wrong_reach_correct"
    return "label_wrong_reach_wrong"


# ---------------------------------------------------------------------------
# v4.0.0+: per-reach confusion (the canonical unit per the user's
# PER_REACH_IS_THE_UNIT memory). For each reach in the universe of
# (GT reaches union algo reaches), label by what each side says about
# that reach: causal-reach gets the segment outcome; non-causal reaches
# default to 'miss'; reaches that exist only on one side flow to/from
# 'absent' on the other side.
# ---------------------------------------------------------------------------

def _label_reach_by_side(reach, side_segments_by_seg, side_kind):
    """Label a single reach with its outcome per the given side (gt or algo).

    side_segments_by_seg : {segment_num: segment_dict_with_outcome_and_causal}
    side_kind : 'gt' or 'algo'

    GT side uses interaction_frame -> reach window containment to identify
    causal; algo side uses causal_reach_id == reach.reach_id.
    """
    seg_num = reach.get("segment_num") if side_kind == "gt" else reach.get("_seg_num")
    seg = side_segments_by_seg.get(seg_num, {})
    if side_kind == "algo":
        # Apply algo-punt normalization so abnormal_exception segments
        # are recognized at the per-reach level too.
        cls = _normalize_algo_punted_outcome(seg)
    else:
        cls = seg.get("outcome", "miss")
    # Collapse displaced_outside -> displaced_sa per user directive.
    cls = _collapse_displaced_outside(cls)
    # abnormal_exception segments propagate to all their reaches so the
    # case stays visible on the per-reach confusion matrix.
    if cls == "abnormal_exception":
        return "abnormal_exception"
    if cls in ("untouched", "uncertain", "unknown") or cls is None:
        return "miss"
    if side_kind == "gt":
        ifr = seg.get("interaction_frame")
        sf, ef = reach.get("start_frame"), reach.get("end_frame")
        if ifr is None or sf is None or ef is None:
            return "miss"
        if sf <= ifr <= ef:
            return cls if cls in ("retrieved", "displaced_sa", "displaced_outside") else "miss"
        return "miss"
    else:  # algo
        cid = seg.get("causal_reach_id")
        if cid is None:
            return "miss"
        if reach.get("reach_id") == cid:
            return cls if cls in ("retrieved", "displaced_sa", "displaced_outside") else "miss"
        return "miss"


def compute_per_reach_confusion(
    gt_dir: Path,
    algo_dir: Path,
    reaches_dir: Path,
    video_ids: List[str],
    match_window: int = 10,
) -> Dict[str, Any]:
    """Compute a per-reach confusion matrix across the universe of GT and
    algo reaches.

    For every reach in the union (matched pairs + GT-only + algo-only):
      - GT label: outcome attribution from GT side (causal -> segment
        outcome; non-causal -> 'miss'; not in GT -> 'absent')
      - algo label: outcome attribution from algo side (same rule)
    The (gt_label, algo_label) pair is incremented in the confusion matrix.

    Returns a dict with structure::
        {
          "n_reaches_universe": int,
          "confusion_matrix": {"gt__algo": count},
          "per_class": {class: {n_gt, n_algo, precision, recall, f1}},
        }
    """
    from mousereach.improvement.reach_detection.metrics import match_reaches, Reach

    confusion: Dict[str, int] = defaultdict(int)
    per_class_gt: Dict[str, int] = defaultdict(int)
    per_class_algo: Dict[str, int] = defaultdict(int)
    per_class_correct: Dict[str, int] = defaultdict(int)
    n_universe = 0

    for vid in video_ids:
        gt_file = _find_gt_file(gt_dir, vid)
        algo_file = _find_algo_file(algo_dir, vid)
        algo_reach_file = _find_algo_reaches_file(reaches_dir, vid)
        if gt_file is None or algo_file is None or algo_reach_file is None:
            continue

        gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
        gt_segments = {s["segment_num"]: s for s in gt_data.get("outcomes", {}).get("segments", [])}
        gt_reaches_raw = gt_data.get("reaches", {}).get("reaches", [])

        algo_data = json.loads(algo_file.read_text(encoding="utf-8"))
        algo_segs_by_num = {s["segment_num"]: s for s in algo_data.get("segments", [])}

        algo_reach_data = json.loads(algo_reach_file.read_text(encoding="utf-8"))
        algo_reach_list: List[Dict[str, Any]] = []
        for sg in algo_reach_data.get("segments", []):
            sn = sg["segment_num"]
            for r in sg.get("reaches", []):
                r["_seg_num"] = sn
                algo_reach_list.append(r)

        # Per-reach labels
        gt_labels = [_label_reach_by_side(r, gt_segments, "gt") for r in gt_reaches_raw]
        algo_labels = [_label_reach_by_side(r, algo_segs_by_num, "algo") for r in algo_reach_list]

        # Match using the reach-detection matcher (algo first, gt second per the API)
        gt_lite = [Reach(r["start_frame"], r["end_frame"], i)
                   for i, r in enumerate(gt_reaches_raw)
                   if r.get("start_frame") is not None]
        algo_lite = [Reach(r["start_frame"], r["end_frame"], i)
                     for i, r in enumerate(algo_reach_list)
                     if r.get("start_frame") is not None]
        results = match_reaches(algo_lite, gt_lite, window=match_window)
        for res in results:
            if res.status == "matched":
                gl = gt_labels[res.gt_reach_index]
                al = algo_labels[res.algo_reach_index]
            elif res.status == "fn":
                gl = gt_labels[res.gt_reach_index]
                al = "absent"
            elif res.status == "fp":
                gl = "absent"
                al = algo_labels[res.algo_reach_index]
            else:
                continue
            confusion[f"{gl}__{al}"] += 1
            per_class_gt[gl] += 1
            per_class_algo[al] += 1
            if gl == al:
                per_class_correct[gl] += 1
            n_universe += 1

    # Build per-class stats. abnormal_exception is excluded from
    # precision/recall denominators per design (visible on confusion
    # matrix, off the metric calculation).
    per_class_stats: Dict[str, Dict[str, Any]] = {}
    for cls in sorted(set(list(per_class_gt.keys()) + list(per_class_algo.keys()))):
        if cls in NON_EVALUABLE_OUTCOMES:
            continue
        n_gt = per_class_gt.get(cls, 0)
        n_algo = per_class_algo.get(cls, 0)
        n_correct = per_class_correct.get(cls, 0)
        precision = round(n_correct / n_algo, 4) if n_algo > 0 else 0.0
        recall = round(n_correct / n_gt, 4) if n_gt > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        per_class_stats[cls] = {
            "n_gt": n_gt,
            "n_algo": n_algo,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "n_reaches_universe": n_universe,
        "confusion_matrix": dict(confusion),
        "per_class": per_class_stats,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_outcome_metrics(
    gt_dir: Path,
    algo_dir: Path,
    output_dir: Path,
    video_ids: Optional[List[str]] = None,
    reaches_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute outcome classification accuracy metrics.

    Loads GT and algo outcomes for each video, pairs segments by segment_num,
    derives GT causal reach, matches causal reaches, computes verdicts, and
    writes three deliverables.

    Parameters
    ----------
    gt_dir : Path
        Directory containing ``*_unified_ground_truth.json`` files.
    algo_dir : Path
        Directory containing ``*_pellet_outcomes.json`` files.
    output_dir : Path
        Where to write deliverables.
    video_ids : list of str, optional
        Explicit video IDs. If None, auto-discovers from GT files.
    reaches_dir : Path, optional
        Directory containing ``*_reaches.json`` files used to look up the
        algo's causal-reach window. If None, defaults to ``algo_dir`` for
        backward compatibility (which is wrong when the outcome and reach
        outputs live in separate dirs -- pass the reach detector's output
        dir explicitly to get correct causal-reach matching).

    Returns
    -------
    dict
        The scalars dict (same structure written to scalars.json).
    """
    gt_dir = Path(gt_dir)
    algo_dir = Path(algo_dir)
    output_dir = Path(output_dir)
    if reaches_dir is None:
        reaches_dir = algo_dir
    else:
        reaches_dir = Path(reaches_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if video_ids is None:
        video_ids = _discover_video_ids(gt_dir)

    if not video_ids:
        logger.warning("No video IDs found in %s", gt_dir)
        return {}

    # Collect rows
    segment_rows: List[Dict[str, Any]] = []
    video_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    # Global accumulators
    all_verdicts: List[str] = []
    all_interaction_deltas: List[int] = []
    confusion: Dict[str, int] = defaultdict(int)
    per_class_gt: Dict[str, int] = defaultdict(int)
    per_class_algo: Dict[str, int] = defaultdict(int)
    per_class_correct: Dict[str, int] = defaultdict(int)
    causal_overall: Dict[str, int] = defaultdict(int)
    causal_per_class: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_exhaustive_videos = 0

    for vid in video_ids:
        gt_file = _find_gt_file(gt_dir, vid)
        algo_file = _find_algo_file(algo_dir, vid)

        if gt_file is None:
            logger.warning("No GT file for %s -- skipping", vid)
            skipped.append(vid)
            continue
        if algo_file is None:
            logger.warning("No algo file for %s -- skipping", vid)
            skipped.append(vid)
            continue

        gt_segments, gt_reaches, exhaustive = _load_gt_outcomes(gt_file)
        algo_segments, _version = _load_algo_outcomes(algo_file)

        if exhaustive:
            n_exhaustive_videos += 1

        # Load algo reaches for causal reach lookup. Reaches usually live in
        # a different dir than outcomes; reaches_dir was added explicitly.
        algo_reaches_file = _find_algo_reaches_file(reaches_dir, vid)
        algo_reaches = _load_algo_reaches(algo_reaches_file) if algo_reaches_file else []

        # Index segments by segment_num
        gt_by_seg = {s["segment_num"]: s for s in gt_segments}
        algo_by_seg = {s["segment_num"]: s for s in algo_segments}

        # Pair by segment_num (intersection)
        common_segs = sorted(set(gt_by_seg.keys()) & set(algo_by_seg.keys()))

        vid_verdicts: List[str] = []
        vid_interaction_deltas: List[int] = []

        for seg_num in common_segs:
            gt_seg = gt_by_seg[seg_num]
            algo_seg = algo_by_seg[seg_num]

            gt_outcome = gt_seg.get("outcome", "unknown")
            # Normalize algo-punted segments to abnormal_exception so they
            # are visible on the Sankey but excluded from precision/recall
            # denominators. Single source of truth for the normalization.
            algo_outcome = _normalize_algo_punted_outcome(algo_seg)
            # Collapse displaced_outside -> displaced_sa on both sides
            # (treated as one class per user directive).
            gt_outcome = _collapse_displaced_outside(gt_outcome)
            algo_outcome = _collapse_displaced_outside(algo_outcome)

            gt_interaction = gt_seg.get("interaction_frame")
            algo_interaction = algo_seg.get("interaction_frame")

            # Interaction frame delta
            interaction_delta = None
            if gt_interaction is not None and algo_interaction is not None:
                interaction_delta = algo_interaction - gt_interaction

            # Derive GT causal reach
            gt_causal_id = derive_gt_causal_reach(gt_seg, gt_reaches)
            gt_causal_start = _get_reach_start(gt_causal_id, gt_reaches) if gt_causal_id is not None else None

            # Algo causal reach
            algo_causal_id = algo_seg.get("causal_reach_id")
            algo_causal_frame = algo_seg.get("causal_reach_frame")
            # Try to get algo causal start from reaches file
            algo_causal_start = None
            if algo_causal_id is not None:
                algo_causal_start = _get_reach_start(algo_causal_id, algo_reaches)

            # Causal reach start delta
            causal_start_delta = None
            if algo_causal_start is not None and gt_causal_start is not None:
                causal_start_delta = algo_causal_start - gt_causal_start

            # Causal reach match: does the algo's causal reach window
            # contain GT's interaction_frame? (Reach IDs are in different
            # namespaces between GT and algo, so we use frame-containment
            # rather than reach-id matching.)
            causal_match = _algo_reach_contains_frame(
                algo_causal_id, algo_reaches, gt_interaction
            )

            # Verdict
            verdict = _compute_verdict(gt_outcome, algo_outcome, causal_match)

            # Record
            row = {
                "video_id": vid,
                "segment_num": seg_num,
                "gt_outcome": gt_outcome,
                "algo_outcome": algo_outcome,
                "outcome_label_match": gt_outcome == algo_outcome,
                "gt_interaction_frame": gt_interaction,
                "algo_interaction_frame": algo_interaction,
                "interaction_frame_delta": interaction_delta,
                "gt_causal_reach_id": gt_causal_id,
                "algo_causal_reach_id": algo_causal_id,
                "gt_causal_reach_start": gt_causal_start,
                "algo_causal_reach_start": algo_causal_start,
                "causal_reach_start_delta": causal_start_delta,
                "causal_reach_match": causal_match,
                "verdict": verdict,
                "gt_exhaustive": exhaustive,
            }
            segment_rows.append(row)

            vid_verdicts.append(verdict)
            all_verdicts.append(verdict)

            # Confusion matrix
            cm_key = f"{gt_outcome}__{algo_outcome}"
            confusion[cm_key] += 1

            # Per-class counts (only committed outcomes)
            if gt_outcome in COMMITTED_OUTCOMES:
                per_class_gt[gt_outcome] += 1
            if algo_outcome in COMMITTED_OUTCOMES:
                per_class_algo[algo_outcome] += 1
            if gt_outcome == algo_outcome and gt_outcome in COMMITTED_OUTCOMES:
                per_class_correct[gt_outcome] += 1

            # Interaction frame delta (only when both committed & non-null)
            if interaction_delta is not None and gt_outcome in TOUCHED_OUTCOMES and algo_outcome in TOUCHED_OUTCOMES:
                all_interaction_deltas.append(interaction_delta)
                vid_interaction_deltas.append(interaction_delta)

            # Causal reach breakdown (touched outcomes only)
            if gt_outcome in TOUCHED_OUTCOMES:
                causal_overall[verdict] += 1
                causal_per_class[gt_outcome][verdict] += 1

        # Per-video summary
        n_paired = len(common_segs)
        n_larc = sum(1 for v in vid_verdicts if v == "label_and_reach_correct")
        n_lcwr = sum(1 for v in vid_verdicts if v == "label_correct_wrong_reach")
        n_lw = sum(1 for v in vid_verdicts if v == "label_wrong")
        n_ab = sum(1 for v in vid_verdicts if v == "abstained")
        n_lcu = sum(1 for v in vid_verdicts if v == "label_correct_untouched")

        n_committed = n_paired - n_ab
        n_correct_strict = n_larc + n_lcwr + n_lcu
        strict_acc = round(n_correct_strict / n_paired, 4) if n_paired > 0 else None
        committed_acc = round(n_correct_strict / n_committed, 4) if n_committed > 0 else None
        abstention_rate = round(n_ab / n_paired, 4) if n_paired > 0 else None

        vid_abs_deltas = [abs(d) for d in vid_interaction_deltas]

        video_rows.append({
            "video_id": vid,
            "exhaustive": exhaustive,
            "n_segments_paired": n_paired,
            "n_label_and_reach_correct": n_larc,
            "n_label_correct_wrong_reach": n_lcwr,
            "n_label_wrong": n_lw,
            "n_abstained": n_ab,
            "n_label_correct_untouched": n_lcu,
            "strict_accuracy": strict_acc,
            "committed_accuracy": committed_acc,
            "abstention_rate": abstention_rate,
            "median_abs_interaction_delta": int(np.median(vid_abs_deltas)) if vid_abs_deltas else None,
            "mean_signed_interaction_delta": round(float(np.mean(vid_interaction_deltas)), 2) if vid_interaction_deltas else None,
        })

    # -----------------------------------------------------------------------
    # Build scalars
    # -----------------------------------------------------------------------

    n_total = len(all_verdicts)
    n_correct_strict = sum(1 for v in all_verdicts if v in ("label_and_reach_correct", "label_correct_wrong_reach", "label_correct_untouched"))
    n_abstained = sum(1 for v in all_verdicts if v == "abstained")
    n_committed = n_total - n_abstained

    # Per-class precision / recall / F1. abnormal_exception is excluded
    # from per-class denominators (visible on confusion matrix, off the
    # metric calculation).
    per_class_stats = {}
    for cls in sorted(set(list(per_class_gt.keys()) + list(per_class_algo.keys()))):
        if cls in NON_EVALUABLE_OUTCOMES:
            continue
        n_gt_cls = per_class_gt.get(cls, 0)
        n_algo_cls = per_class_algo.get(cls, 0)
        n_correct_cls = per_class_correct.get(cls, 0)
        precision = round(n_correct_cls / n_algo_cls, 4) if n_algo_cls > 0 else 0.0
        recall = round(n_correct_cls / n_gt_cls, 4) if n_gt_cls > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        per_class_stats[cls] = {
            "n_gt": n_gt_cls,
            "n_algo": n_algo_cls,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Interaction frame histogram (non-cumulative, per exact delta)
    interaction_histogram: Dict[str, int] = defaultdict(int)
    for d in all_interaction_deltas:
        interaction_histogram[str(d)] += 1
    interaction_histogram = dict(sorted(interaction_histogram.items(), key=lambda x: int(x[0])))

    abs_interaction_deltas = [abs(d) for d in all_interaction_deltas]

    # Causal reach breakdown
    causal_overall_dict = dict(causal_overall)
    causal_per_class_dict = {k: dict(v) for k, v in causal_per_class.items()}

    # v4.0.0+: per-reach is the canonical unit (per PER_REACH_IS_THE_UNIT
    # memory). Compute per-reach confusion and add it to scalars under
    # both `outcome_label_per_reach` (explicit) and as the default for
    # `outcome_label.confusion_matrix` so run_sankey() picks it up.
    per_reach = compute_per_reach_confusion(
        gt_dir=gt_dir,
        algo_dir=algo_dir,
        reaches_dir=reaches_dir,
        video_ids=video_ids,
    )

    scalars: Dict[str, Any] = {
        "n_videos": len(video_ids) - len(skipped),
        "n_videos_exhaustive": n_exhaustive_videos,
        "n_segments_paired": n_total,
        "n_reaches_universe": per_reach["n_reaches_universe"],
        "outcome_label": {
            # Per-reach by default (canonical unit). Per-segment mirror is
            # under outcome_label_per_segment.
            "strict_accuracy": round(n_correct_strict / n_total, 4) if n_total > 0 else None,
            "committed_accuracy": round(n_correct_strict / n_committed, 4) if n_committed > 0 else None,
            "abstention_rate": round(n_abstained / n_total, 4) if n_total > 0 else None,
            "per_class": per_reach["per_class"],
            "confusion_matrix": per_reach["confusion_matrix"],
        },
        "outcome_label_per_segment": {
            "strict_accuracy": round(n_correct_strict / n_total, 4) if n_total > 0 else None,
            "committed_accuracy": round(n_correct_strict / n_committed, 4) if n_committed > 0 else None,
            "abstention_rate": round(n_abstained / n_total, 4) if n_total > 0 else None,
            "per_class": per_class_stats,
            "confusion_matrix": dict(confusion),
        },
        "interaction_frame": {
            "n_eligible": len(all_interaction_deltas),
            "delta_histogram": interaction_histogram,
            "median_abs_delta": int(np.median(abs_interaction_deltas)) if abs_interaction_deltas else None,
            "mean_signed_delta": round(float(np.mean(all_interaction_deltas)), 2) if all_interaction_deltas else None,
        },
        "causal_reach": {
            "overall": causal_overall_dict,
            "per_class": causal_per_class_dict,
        },
    }

    # -----------------------------------------------------------------------
    # Write deliverables
    # -----------------------------------------------------------------------

    df_segments = pd.DataFrame(segment_rows)
    df_segments.to_csv(output_dir / "outcome_per_segment.csv", index=False)

    df_videos = pd.DataFrame(video_rows)
    df_videos.to_csv(output_dir / "per_video.csv", index=False)

    with open(output_dir / "scalars.json", "w", encoding="utf-8") as f:
        json.dump(scalars, f, indent=2, ensure_ascii=False)

    n_processed = len(video_ids) - len(skipped)
    logger.info(
        "Outcome metrics: %d videos processed, %d skipped",
        n_processed, len(skipped),
    )

    return scalars
