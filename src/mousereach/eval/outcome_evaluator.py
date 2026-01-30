"""
Outcome classification evaluation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .base import BaseEvaluator, EvalResult, ErrorCategory


# Valid outcome classes
OUTCOME_CLASSES = [
    "retrieved",
    "displaced_sa",
    "displaced_outside",
    "untouched",
    "uncertain",
    "no_pellet"
]


@dataclass
class OutcomeMatch:
    """Match between a GT outcome and an algorithm outcome for one segment."""
    segment_num: int
    gt_outcome: str
    algo_outcome: str
    matched: bool = False  # Same outcome class

    # Timing errors (for matched outcomes with interaction frames)
    interaction_frame_gt: Optional[int] = None
    interaction_frame_algo: Optional[int] = None
    interaction_frame_error: Optional[int] = None

    # Causal reach comparison
    causal_reach_gt: Optional[int] = None
    causal_reach_algo: Optional[int] = None
    causal_reach_matched: bool = False


@dataclass
class OutcomeEvalResult(EvalResult):
    """Outcome classification evaluation result for one video."""
    # Counts
    n_segments: int = 0

    # Per-class counts
    class_counts_gt: Dict[str, int] = field(default_factory=dict)
    class_counts_algo: Dict[str, int] = field(default_factory=dict)

    # Outcome matches
    matches: List[OutcomeMatch] = field(default_factory=list)

    # Overall metrics
    accuracy: float = 0.0  # Correct / Total
    precision_macro: float = 0.0  # Macro-averaged precision
    recall_macro: float = 0.0  # Macro-averaged recall
    f1_macro: float = 0.0  # Macro-averaged F1

    # Confusion matrix (as dict for JSON serialization)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Per-class metrics
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)

    # Timing accuracy (for segments where both have interaction frames)
    mean_interaction_frame_error: float = 0.0
    interaction_frame_errors: List[Dict] = field(default_factory=list)

    # Causal reach accuracy
    causal_reach_accuracy: float = 0.0

    # Error breakdowns
    misclassifications: List[Dict] = field(default_factory=list)


class OutcomeEvaluator(BaseEvaluator):
    """Evaluate outcome classification accuracy."""

    gt_pattern = "*_outcome_ground_truth.json"
    algo_pattern = "*_pellet_outcomes.json"
    step_name = "outcome detection"

    # Tolerance for interaction frame matching (frames)
    FRAME_TOLERANCE = 15

    def __init__(self, gt_dir: Path = None, algo_dir: Path = None, tolerance: int = None):
        super().__init__(gt_dir, algo_dir, tolerance or self.FRAME_TOLERANCE)
        self.frame_tolerance = int(self.tolerance)

    def _init_error_categories(self):
        """Initialize outcome detection error categories."""
        self.error_categories = {
            "retrieved_missed": ErrorCategory(
                "retrieved_missed",
                "GT=retrieved but algo classified differently (false negative for retrieved)"
            ),
            "retrieved_phantom": ErrorCategory(
                "retrieved_phantom",
                "Algo=retrieved but GT says different (false positive for retrieved)"
            ),
            "displaced_as_untouched": ErrorCategory(
                "displaced_as_untouched",
                "GT=displaced but algo=untouched (missed pellet interaction)"
            ),
            "untouched_as_displaced": ErrorCategory(
                "untouched_as_displaced",
                "GT=untouched but algo=displaced (phantom interaction)"
            ),
            "interaction_timing_errors": ErrorCategory(
                "interaction_timing_errors",
                "Outcome matched but interaction frame significantly off"
            ),
            "causal_reach_mismatch": ErrorCategory(
                "causal_reach_mismatch",
                "Outcome matched but different reach identified as causal"
            ),
            "low_confidence_errors": ErrorCategory(
                "low_confidence_errors",
                "Misclassification on low-confidence algo output"
            ),
        }

    def load_ground_truth(self, video_id: str) -> Optional[Dict]:
        """Load outcome ground truth."""
        # Try both singular and plural naming conventions
        gt_path = self.gt_dir / f"{video_id}_outcome_ground_truth.json"
        if not gt_path.exists():
            gt_path = self.gt_dir / f"{video_id}_outcomes_ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                return json.load(f)
        return None

    def load_algorithm_output(self, video_id: str) -> Optional[Dict]:
        """Load outcome algorithm output."""
        algo_path = self.algo_dir / f"{video_id}_pellet_outcomes.json"
        if algo_path.exists():
            with open(algo_path) as f:
                return json.load(f)
        return None

    def compare(self, video_id: str) -> OutcomeEvalResult:
        """Compare outcome detection algorithm vs ground truth."""
        result = OutcomeEvalResult(video_id=video_id)

        gt = self.load_ground_truth(video_id)
        algo = self.load_algorithm_output(video_id)

        if gt is None:
            result.success = False
            result.error_message = "Ground truth not found"
            return result

        if algo is None:
            result.success = False
            result.error_message = "Algorithm output not found"
            return result

        # Extract segments from both
        gt_segments = gt.get("segments", [])
        algo_segments = algo.get("segments", [])

        # Build lookup by segment number
        gt_by_seg = {s.get("segment_num", i+1): s for i, s in enumerate(gt_segments)}
        algo_by_seg = {s.get("segment_num", i+1): s for i, s in enumerate(algo_segments)}

        all_seg_nums = set(gt_by_seg.keys()) | set(algo_by_seg.keys())
        result.n_segments = len(all_seg_nums)

        # Initialize confusion matrix
        for gt_class in OUTCOME_CLASSES:
            result.confusion_matrix[gt_class] = {algo_class: 0 for algo_class in OUTCOME_CLASSES}

        # Initialize class counts
        for cls in OUTCOME_CLASSES:
            result.class_counts_gt[cls] = 0
            result.class_counts_algo[cls] = 0

        # Compare each segment
        for seg_num in sorted(all_seg_nums):
            gt_seg = gt_by_seg.get(seg_num)
            algo_seg = algo_by_seg.get(seg_num)

            if gt_seg is None or algo_seg is None:
                # Missing segment in one or the other
                continue

            gt_outcome = gt_seg.get("outcome", "uncertain")
            algo_outcome = algo_seg.get("outcome", "uncertain")

            # Update class counts
            if gt_outcome in OUTCOME_CLASSES:
                result.class_counts_gt[gt_outcome] += 1
            if algo_outcome in OUTCOME_CLASSES:
                result.class_counts_algo[algo_outcome] += 1

            # Update confusion matrix
            if gt_outcome in OUTCOME_CLASSES and algo_outcome in OUTCOME_CLASSES:
                result.confusion_matrix[gt_outcome][algo_outcome] += 1

            # Create match record
            match = OutcomeMatch(
                segment_num=seg_num,
                gt_outcome=gt_outcome,
                algo_outcome=algo_outcome,
                matched=(gt_outcome == algo_outcome),
                interaction_frame_gt=gt_seg.get("interaction_frame"),
                interaction_frame_algo=algo_seg.get("interaction_frame"),
                causal_reach_gt=gt_seg.get("causal_reach_id"),
                causal_reach_algo=algo_seg.get("causal_reach_id"),
            )

            # Interaction frame error
            if match.interaction_frame_gt is not None and match.interaction_frame_algo is not None:
                match.interaction_frame_error = match.interaction_frame_algo - match.interaction_frame_gt

            # Causal reach match
            if match.causal_reach_gt is not None and match.causal_reach_algo is not None:
                match.causal_reach_matched = (match.causal_reach_gt == match.causal_reach_algo)

            result.matches.append(match)

            # Track misclassifications
            if not match.matched:
                result.misclassifications.append({
                    "segment": seg_num,
                    "gt": gt_outcome,
                    "algo": algo_outcome,
                    "confidence": algo_seg.get("confidence", 0),
                })

        # Calculate metrics
        self._calculate_metrics(result)

        return result

    def _calculate_metrics(self, result: OutcomeEvalResult):
        """Calculate accuracy, precision, recall, F1 from matches."""
        if not result.matches:
            return

        # Overall accuracy
        correct = sum(1 for m in result.matches if m.matched)
        result.accuracy = correct / len(result.matches) if result.matches else 0

        # Per-class precision/recall/F1
        for cls in OUTCOME_CLASSES:
            # True positives: GT=cls AND algo=cls
            tp = result.confusion_matrix.get(cls, {}).get(cls, 0)

            # False positives: GT!=cls but algo=cls (sum column minus diagonal)
            fp = sum(
                result.confusion_matrix.get(gt_cls, {}).get(cls, 0)
                for gt_cls in OUTCOME_CLASSES if gt_cls != cls
            )

            # False negatives: GT=cls but algo!=cls (sum row minus diagonal)
            fn = sum(
                result.confusion_matrix.get(cls, {}).get(algo_cls, 0)
                for algo_cls in OUTCOME_CLASSES if algo_cls != cls
            )

            # Precision
            result.per_class_precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Recall
            result.per_class_recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0

            # F1
            p = result.per_class_precision[cls]
            r = result.per_class_recall[cls]
            result.per_class_f1[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # Macro-averaged metrics (only for classes with samples)
        classes_with_samples = [
            cls for cls in OUTCOME_CLASSES
            if result.class_counts_gt.get(cls, 0) > 0 or result.class_counts_algo.get(cls, 0) > 0
        ]

        if classes_with_samples:
            result.precision_macro = np.mean([
                result.per_class_precision[cls] for cls in classes_with_samples
            ])
            result.recall_macro = np.mean([
                result.per_class_recall[cls] for cls in classes_with_samples
            ])
            result.f1_macro = np.mean([
                result.per_class_f1[cls] for cls in classes_with_samples
            ])

        # Interaction frame accuracy
        frame_errors = [
            m for m in result.matches
            if m.interaction_frame_error is not None and m.matched
        ]
        if frame_errors:
            result.mean_interaction_frame_error = np.mean([
                abs(m.interaction_frame_error) for m in frame_errors
            ])
            result.interaction_frame_errors = [
                {"segment": m.segment_num, "error": m.interaction_frame_error}
                for m in frame_errors if abs(m.interaction_frame_error) > self.frame_tolerance
            ]

        # Causal reach accuracy
        reach_comparisons = [
            m for m in result.matches
            if m.causal_reach_gt is not None and m.causal_reach_algo is not None
        ]
        if reach_comparisons:
            result.causal_reach_accuracy = sum(
                1 for m in reach_comparisons if m.causal_reach_matched
            ) / len(reach_comparisons)

    def categorize_errors(self, result: OutcomeEvalResult):
        """Categorize outcome detection errors."""
        if not result.success:
            return

        for m in result.matches:
            # Retrieved missed (GT=retrieved, algo=something else)
            if m.gt_outcome == "retrieved" and m.algo_outcome != "retrieved":
                self.error_categories["retrieved_missed"].add_example(
                    result.video_id,
                    {"segment": m.segment_num, "algo_said": m.algo_outcome}
                )

            # Retrieved phantom (algo=retrieved but GT!=retrieved)
            if m.algo_outcome == "retrieved" and m.gt_outcome != "retrieved":
                self.error_categories["retrieved_phantom"].add_example(
                    result.video_id,
                    {"segment": m.segment_num, "gt_was": m.gt_outcome}
                )

            # Displaced as untouched
            if m.gt_outcome in ["displaced_sa", "displaced_outside"] and m.algo_outcome == "untouched":
                self.error_categories["displaced_as_untouched"].add_example(
                    result.video_id,
                    {"segment": m.segment_num, "gt_was": m.gt_outcome}
                )

            # Untouched as displaced
            if m.gt_outcome == "untouched" and m.algo_outcome in ["displaced_sa", "displaced_outside"]:
                self.error_categories["untouched_as_displaced"].add_example(
                    result.video_id,
                    {"segment": m.segment_num, "algo_said": m.algo_outcome}
                )

            # Interaction timing errors (matched but bad timing)
            if m.matched and m.interaction_frame_error is not None:
                if abs(m.interaction_frame_error) > self.frame_tolerance:
                    self.error_categories["interaction_timing_errors"].add_example(
                        result.video_id,
                        {"segment": m.segment_num, "error": m.interaction_frame_error}
                    )

            # Causal reach mismatch (matched outcome but different reach)
            if m.matched and not m.causal_reach_matched:
                if m.causal_reach_gt is not None and m.causal_reach_algo is not None:
                    self.error_categories["causal_reach_mismatch"].add_example(
                        result.video_id,
                        {"segment": m.segment_num, "gt_reach": m.causal_reach_gt,
                         "algo_reach": m.causal_reach_algo}
                    )

    def _format_overall_metrics(self) -> List[str]:
        """Format overall outcome detection metrics."""
        if not self.results:
            return []

        successful = [r for r in self.results if r.success and isinstance(r, OutcomeEvalResult)]
        if not successful:
            return []

        lines = [
            "Overall Metrics:",
            f"  Accuracy: {np.mean([r.accuracy for r in successful]):.1%}",
            f"  Macro F1: {np.mean([r.f1_macro for r in successful]):.2f}",
            "",
            "Per-Class Performance (averaged across videos):",
        ]

        # Aggregate per-class metrics
        for cls in OUTCOME_CLASSES:
            precisions = [r.per_class_precision.get(cls, 0) for r in successful
                         if r.class_counts_gt.get(cls, 0) > 0]
            recalls = [r.per_class_recall.get(cls, 0) for r in successful
                      if r.class_counts_gt.get(cls, 0) > 0]

            if precisions or recalls:
                avg_p = np.mean(precisions) if precisions else 0
                avg_r = np.mean(recalls) if recalls else 0
                lines.append(f"  {cls:20s} P={avg_p:.1%} R={avg_r:.1%}")

        # Interaction frame accuracy
        frame_errors = [r.mean_interaction_frame_error for r in successful
                       if r.mean_interaction_frame_error > 0]
        if frame_errors:
            lines.append("")
            lines.append(f"Interaction Frame MAE: {np.mean(frame_errors):.1f} frames")

        # Causal reach accuracy
        reach_accs = [r.causal_reach_accuracy for r in successful
                     if r.causal_reach_accuracy > 0]
        if reach_accs:
            lines.append(f"Causal Reach Accuracy: {np.mean(reach_accs):.1%}")

        return lines

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for outcome detection."""
        recommendations = []

        retrieved_missed = self.error_categories.get("retrieved_missed", ErrorCategory("", ""))
        retrieved_phantom = self.error_categories.get("retrieved_phantom", ErrorCategory("", ""))
        disp_as_untouched = self.error_categories.get("displaced_as_untouched", ErrorCategory("", ""))
        untouched_as_disp = self.error_categories.get("untouched_as_displaced", ErrorCategory("", ""))
        timing = self.error_categories.get("interaction_timing_errors", ErrorCategory("", ""))

        # Retrieved detection issues
        if retrieved_missed.count > 3:
            recommendations.append(
                f"{retrieved_missed.count} 'retrieved' outcomes missed. "
                "Check eating signature detection - may need lower thresholds or "
                "improved nose/hand proximity detection."
            )

        if retrieved_phantom.count > 3:
            recommendations.append(
                f"{retrieved_phantom.count} false 'retrieved' detections. "
                "Algorithm may be too aggressive - check pellet visibility thresholds."
            )

        # Displacement detection issues
        if disp_as_untouched.count > 3:
            recommendations.append(
                f"{disp_as_untouched.count} displacements classified as 'untouched'. "
                "Check displacement threshold (currently 0.25 ruler) - may need lowering. "
                "Also verify paw proximity detection isn't filtering real interactions."
            )

        if untouched_as_disp.count > 3:
            recommendations.append(
                f"{untouched_as_disp.count} 'untouched' classified as displaced. "
                "Algorithm detecting phantom displacements - check for tray wobble "
                "false positives or lower paw proximity requirement."
            )

        # Timing issues
        if timing.count > 5:
            recommendations.append(
                f"{timing.count} segments have interaction frame errors > {self.frame_tolerance} frames. "
                "Review displacement onset detection and pellet occlusion lookback logic."
            )

        if not recommendations:
            recommendations.append("Outcome detection is performing well.")

        return recommendations
