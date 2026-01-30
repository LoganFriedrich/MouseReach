"""
Metric computation functions for algorithm performance tracking.

These functions compare algorithm outputs to human-corrected results
and compute standard evaluation metrics.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class SegmentationMetrics:
    """Metrics for segmentation boundary detection."""
    n_algo_boundaries: int = 0
    n_human_boundaries: int = 0
    n_matched: int = 0
    n_missed: int = 0
    n_extra: int = 0

    # Timing errors (frames)
    mean_error_frames: float = 0.0
    max_error_frames: float = 0.0
    std_error_frames: float = 0.0

    # Accuracy
    accuracy: float = 0.0  # matched / human_boundaries

    # Per-boundary errors (for detailed analysis)
    boundary_errors: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReachMetrics:
    """Metrics for reach detection."""
    n_algo_reaches: int = 0
    n_human_reaches: int = 0
    n_matched: int = 0
    n_missed: int = 0  # False negatives (human added)
    n_extra: int = 0   # False positives (human deleted)
    n_corrected: int = 0  # Timing corrections

    # Classification metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Timing errors (frames)
    mean_start_error: float = 0.0
    mean_end_error: float = 0.0
    mean_apex_error: float = 0.0

    # Per-segment breakdown
    segment_metrics: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OutcomeMetrics:
    """Metrics for pellet outcome classification."""
    n_segments: int = 0
    n_correct: int = 0
    n_incorrect: int = 0

    # Overall accuracy
    accuracy: float = 0.0

    # Per-class metrics
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)

    # Confusion matrix
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Misclassifications for error analysis
    misclassifications: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_segmentation_metrics(
    algo_boundaries: List[int],
    human_boundaries: List[int],
    tolerance_frames: int = 5
) -> SegmentationMetrics:
    """
    Compare algorithm vs human-corrected segmentation boundaries.

    Args:
        algo_boundaries: Algorithm-detected boundary frames
        human_boundaries: Human-corrected boundary frames
        tolerance_frames: Max frames difference to count as match

    Returns:
        SegmentationMetrics with comparison results
    """
    metrics = SegmentationMetrics(
        n_algo_boundaries=len(algo_boundaries),
        n_human_boundaries=len(human_boundaries)
    )

    if not human_boundaries:
        return metrics

    # Match boundaries within tolerance
    algo_matched = [False] * len(algo_boundaries)
    human_matched = [False] * len(human_boundaries)
    errors = []

    for hi, h_frame in enumerate(human_boundaries):
        best_match = None
        best_error = float('inf')

        for ai, a_frame in enumerate(algo_boundaries):
            if algo_matched[ai]:
                continue
            error = abs(a_frame - h_frame)
            if error <= tolerance_frames and error < best_error:
                best_match = ai
                best_error = error

        if best_match is not None:
            algo_matched[best_match] = True
            human_matched[hi] = True
            errors.append(algo_boundaries[best_match] - h_frame)
            metrics.boundary_errors.append({
                'human_frame': h_frame,
                'algo_frame': algo_boundaries[best_match],
                'error': algo_boundaries[best_match] - h_frame
            })

    metrics.n_matched = sum(human_matched)
    metrics.n_missed = sum(not m for m in human_matched)
    metrics.n_extra = sum(not m for m in algo_matched)

    if errors:
        metrics.mean_error_frames = float(np.mean(np.abs(errors)))
        metrics.max_error_frames = float(np.max(np.abs(errors)))
        metrics.std_error_frames = float(np.std(errors))

    if metrics.n_human_boundaries > 0:
        metrics.accuracy = metrics.n_matched / metrics.n_human_boundaries

    return metrics


def compute_reach_metrics(
    algo_data: Dict,
    human_data: Dict,
    timing_tolerance: int = 10
) -> ReachMetrics:
    """
    Compare algorithm vs human-corrected reach detection.

    Uses the 'source' and 'human_corrected' fields to identify changes.

    Args:
        algo_data: Original algorithm output (reaches.json before editing)
        human_data: Human-corrected output (reaches.json after validation)
        timing_tolerance: Frames tolerance for matching reaches

    Returns:
        ReachMetrics with comparison results
    """
    metrics = ReachMetrics()

    algo_segments = algo_data.get('segments', [])
    human_segments = human_data.get('segments', [])

    total_algo = 0
    total_human = 0
    total_matched = 0
    total_missed = 0
    total_extra = 0
    total_corrected = 0

    start_errors = []
    end_errors = []
    apex_errors = []

    for seg_idx, human_seg in enumerate(human_segments):
        algo_seg = algo_segments[seg_idx] if seg_idx < len(algo_segments) else {'reaches': []}

        algo_reaches = algo_seg.get('reaches', [])
        human_reaches = human_seg.get('reaches', [])

        seg_algo_count = len(algo_reaches)
        seg_human_count = len(human_reaches)

        total_algo += seg_algo_count
        total_human += seg_human_count

        # Count human additions (missed by algorithm)
        for r in human_reaches:
            if r.get('source') == 'manual' or r.get('source') == 'human_added':
                total_missed += 1
            elif r.get('human_corrected', False):
                total_corrected += 1
                # Compute timing errors if original values stored
                if 'original_start_frame' in r:
                    start_errors.append(r['start_frame'] - r['original_start_frame'])
                if 'original_end_frame' in r:
                    end_errors.append(r['end_frame'] - r['original_end_frame'])

        # Count deletions (algorithm false positives)
        # Deletion count = algo_count - (human_count - additions)
        additions = sum(1 for r in human_reaches
                       if r.get('source') in ('manual', 'human_added'))
        algo_that_survived = seg_human_count - additions
        seg_deleted = seg_algo_count - algo_that_survived
        if seg_deleted > 0:
            total_extra += seg_deleted

        # Matched = algo reaches that weren't deleted or corrected
        seg_matched = algo_that_survived - sum(1 for r in human_reaches
                                                if r.get('human_corrected') and
                                                r.get('source') != 'manual')
        if seg_matched > 0:
            total_matched += seg_matched

        metrics.segment_metrics.append({
            'segment': seg_idx + 1,
            'algo_reaches': seg_algo_count,
            'human_reaches': seg_human_count,
            'added': additions,
            'deleted': max(0, seg_deleted),
            'corrected': sum(1 for r in human_reaches if r.get('human_corrected'))
        })

    metrics.n_algo_reaches = total_algo
    metrics.n_human_reaches = total_human
    metrics.n_matched = max(0, total_matched)
    metrics.n_missed = total_missed
    metrics.n_extra = total_extra
    metrics.n_corrected = total_corrected

    # Precision, recall, F1
    if total_algo > 0:
        # Precision = (algo reaches that were kept) / (total algo reaches)
        kept = total_algo - total_extra
        metrics.precision = kept / total_algo if total_algo > 0 else 0.0

    if total_human > 0:
        # Recall = (algo reaches that were kept) / (total human reaches)
        kept = total_human - total_missed
        metrics.recall = kept / total_human if total_human > 0 else 0.0

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

    # Timing errors
    if start_errors:
        metrics.mean_start_error = float(np.mean(np.abs(start_errors)))
    if end_errors:
        metrics.mean_end_error = float(np.mean(np.abs(end_errors)))
    if apex_errors:
        metrics.mean_apex_error = float(np.mean(np.abs(apex_errors)))

    return metrics


def compute_outcome_metrics(
    algo_data: Dict,
    human_data: Dict
) -> OutcomeMetrics:
    """
    Compare algorithm vs human-corrected outcome classification.

    Args:
        algo_data: Original algorithm output (pellet_outcomes.json before editing)
        human_data: Human-corrected output (pellet_outcomes.json after validation)

    Returns:
        OutcomeMetrics with comparison results
    """
    metrics = OutcomeMetrics()

    # Outcome classes
    CLASSES = ['retrieved', 'displaced_sa', 'displaced_outside', 'untouched', 'uncertain', 'no_pellet']

    # Initialize confusion matrix
    metrics.confusion_matrix = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}

    algo_segments = algo_data.get('segments', [])
    human_segments = human_data.get('segments', [])

    n_segments = len(human_segments)
    metrics.n_segments = n_segments

    correct = 0
    incorrect = 0

    # Per-class counts for precision/recall
    class_tp = {c: 0 for c in CLASSES}
    class_fp = {c: 0 for c in CLASSES}
    class_fn = {c: 0 for c in CLASSES}

    for seg_idx, human_seg in enumerate(human_segments):
        algo_seg = algo_segments[seg_idx] if seg_idx < len(algo_segments) else {}

        # Get outcomes (handle both 'outcome' and 'original_outcome' fields)
        human_outcome = human_seg.get('outcome', 'unknown')

        # If human_verified, check original_outcome for what algorithm predicted
        if human_seg.get('human_verified') and 'original_outcome' in human_seg:
            algo_outcome = human_seg['original_outcome']
        else:
            algo_outcome = algo_seg.get('outcome', 'unknown')

        # Normalize outcomes
        human_outcome = human_outcome.lower().replace(' ', '_')
        algo_outcome = algo_outcome.lower().replace(' ', '_')

        # Map variations
        outcome_map = {
            'displaced': 'displaced_sa',
            'retrieved (r)': 'retrieved',
            'r': 'retrieved',
            'd': 'displaced_sa',
            'u': 'untouched',
            'o': 'displaced_outside',
        }
        human_outcome = outcome_map.get(human_outcome, human_outcome)
        algo_outcome = outcome_map.get(algo_outcome, algo_outcome)

        if human_outcome not in CLASSES:
            human_outcome = 'uncertain'
        if algo_outcome not in CLASSES:
            algo_outcome = 'uncertain'

        # Update confusion matrix
        metrics.confusion_matrix[algo_outcome][human_outcome] += 1

        if algo_outcome == human_outcome:
            correct += 1
            class_tp[human_outcome] += 1
        else:
            incorrect += 1
            class_fp[algo_outcome] += 1
            class_fn[human_outcome] += 1
            metrics.misclassifications.append({
                'segment': seg_idx + 1,
                'algo': algo_outcome,
                'human': human_outcome
            })

    metrics.n_correct = correct
    metrics.n_incorrect = incorrect

    if n_segments > 0:
        metrics.accuracy = correct / n_segments

    # Per-class precision, recall, F1
    for c in CLASSES:
        tp = class_tp[c]
        fp = class_fp[c]
        fn = class_fn[c]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics.per_class_precision[c] = prec
        metrics.per_class_recall[c] = rec
        metrics.per_class_f1[c] = f1

    return metrics
