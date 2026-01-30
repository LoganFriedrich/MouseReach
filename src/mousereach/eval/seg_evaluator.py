"""
Segmentation boundary evaluation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .base import BaseEvaluator, EvalResult, ErrorCategory


@dataclass
class BoundaryMatch:
    """Match between a GT boundary and an algorithm boundary."""
    gt_frame: int
    algo_frame: Optional[int]  # None if missed
    error: int = 0  # Frame difference (algo - gt)
    matched: bool = False
    detection_method: str = ""  # "primary" or "fallback"
    confidence: float = 0.0


@dataclass
class SegEvalResult(EvalResult):
    """Segmentation evaluation result for one video."""
    # Counts
    n_gt_boundaries: int = 0
    n_algo_boundaries: int = 0

    # Matches
    matches: List[BoundaryMatch] = field(default_factory=list)

    # Metrics
    accuracy: float = 0.0  # % within tolerance
    mean_error: float = 0.0  # Mean absolute frame error
    max_error: int = 0  # Worst case
    std_error: float = 0.0  # Error standard deviation

    # Detection method stats
    n_primary: int = 0
    n_fallback: int = 0
    fallback_error_rate: float = 0.0  # % of fallback that were wrong

    # Lists for detailed analysis
    missed_boundaries: List[int] = field(default_factory=list)  # GT indices not found
    extra_boundaries: List[int] = field(default_factory=list)  # Algo frames not in GT
    early_detections: List[BoundaryMatch] = field(default_factory=list)  # Error < 0
    late_detections: List[BoundaryMatch] = field(default_factory=list)  # Error > 0
    low_confidence_errors: List[BoundaryMatch] = field(default_factory=list)


class SegmentationEvaluator(BaseEvaluator):
    """Evaluate segmentation boundary detection accuracy."""

    gt_pattern = "*_ground_truth.json"  # Also matches *_seg_ground_truth.json
    algo_pattern = "*_segments.json"
    step_name = "segmentation"

    # Tolerance: boundaries within this many frames are considered "correct"
    DEFAULT_TOLERANCE = 5  # ±5 frames

    def __init__(self, gt_dir: Path = None, algo_dir: Path = None, tolerance: int = None):
        super().__init__(gt_dir, algo_dir, tolerance or self.DEFAULT_TOLERANCE)
        self.tolerance = int(self.tolerance)

    def _init_error_categories(self):
        """Initialize segmentation error categories."""
        self.error_categories = {
            "missed_boundaries": ErrorCategory(
                "missed_boundaries",
                "GT boundary not detected by algorithm"
            ),
            "extra_boundaries": ErrorCategory(
                "extra_boundaries",
                "Algorithm detected a boundary not in GT"
            ),
            "early_detections": ErrorCategory(
                "early_detections",
                f"Algorithm boundary more than {self.tolerance} frames early"
            ),
            "late_detections": ErrorCategory(
                "late_detections",
                f"Algorithm boundary more than {self.tolerance} frames late"
            ),
            "fallback_failures": ErrorCategory(
                "fallback_failures",
                "Fallback detection method produced wrong result"
            ),
            "low_confidence_errors": ErrorCategory(
                "low_confidence_errors",
                "Low confidence (<0.7) detection was wrong"
            ),
        }

    def find_gt_files(self) -> List[Path]:
        """Find GT files, supporting both naming conventions."""
        if not self.gt_dir or not self.gt_dir.exists():
            return []

        # Try both patterns
        files = list(self.gt_dir.glob("*_seg_ground_truth.json"))
        files.extend(self.gt_dir.glob("*_ground_truth.json"))

        # Deduplicate (prefer _seg_ground_truth.json)
        seen = set()
        result = []
        for f in files:
            video_id = self.extract_video_id(f)
            if video_id not in seen:
                seen.add(video_id)
                result.append(f)

        return sorted(result)

    def load_ground_truth(self, video_id: str) -> Optional[Dict]:
        """Load segmentation ground truth."""
        # Try both naming conventions
        for pattern in [f"{video_id}_seg_ground_truth.json", f"{video_id}_ground_truth.json"]:
            gt_path = self.gt_dir / pattern
            if gt_path.exists():
                with open(gt_path) as f:
                    return json.load(f)
        return None

    def load_algorithm_output(self, video_id: str) -> Optional[Dict]:
        """Load segmentation algorithm output."""
        algo_path = self.algo_dir / f"{video_id}_segments.json"
        if algo_path.exists():
            with open(algo_path) as f:
                return json.load(f)
        return None

    def compare(self, video_id: str) -> SegEvalResult:
        """Compare segmentation algorithm vs ground truth."""
        result = SegEvalResult(video_id=video_id)

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

        # Extract boundaries
        gt_boundaries = gt.get("boundaries", [])
        algo_boundaries = algo.get("boundaries", [])

        result.n_gt_boundaries = len(gt_boundaries)
        result.n_algo_boundaries = len(algo_boundaries)

        # Get detection metadata from algo output
        detection = algo.get("detection", {})
        methods = detection.get("methods", ["primary"] * len(algo_boundaries))
        confidences = detection.get("confidences", [1.0] * len(algo_boundaries))

        # Match boundaries (greedy matching within tolerance)
        matches, missed, extra = self._match_boundaries(
            gt_boundaries, algo_boundaries, methods, confidences
        )

        result.matches = matches
        result.missed_boundaries = missed
        result.extra_boundaries = extra

        # Calculate metrics
        matched_errors = [m.error for m in matches if m.matched]
        if matched_errors:
            errors_abs = [abs(e) for e in matched_errors]
            result.mean_error = np.mean(errors_abs)
            result.max_error = max(errors_abs)
            result.std_error = np.std(errors_abs) if len(errors_abs) > 1 else 0.0

            # Accuracy: % within tolerance
            within_tolerance = sum(1 for e in errors_abs if e <= self.tolerance)
            result.accuracy = within_tolerance / len(gt_boundaries) if gt_boundaries else 1.0

        # Categorize errors
        for m in matches:
            if not m.matched:
                continue
            if m.error < -self.tolerance:
                result.early_detections.append(m)
            elif m.error > self.tolerance:
                result.late_detections.append(m)
            if m.confidence < 0.7 and abs(m.error) > self.tolerance:
                result.low_confidence_errors.append(m)

        # Detection method stats
        result.n_primary = sum(1 for m in matches if m.detection_method == "primary")
        result.n_fallback = sum(1 for m in matches if m.detection_method == "fallback")

        fallback_matches = [m for m in matches if m.detection_method == "fallback" and m.matched]
        if fallback_matches:
            fallback_wrong = sum(1 for m in fallback_matches if abs(m.error) > self.tolerance)
            result.fallback_error_rate = fallback_wrong / len(fallback_matches)

        return result

    def _match_boundaries(
        self,
        gt_boundaries: List[int],
        algo_boundaries: List[int],
        methods: List[str],
        confidences: List[float]
    ) -> Tuple[List[BoundaryMatch], List[int], List[int]]:
        """Match GT boundaries to algorithm boundaries.

        Uses greedy matching: for each GT boundary, find closest unmatched algo boundary.

        Returns:
            (matches, missed_gt_indices, extra_algo_frames)
        """
        matches = []
        used_algo = set()

        for gt_idx, gt_frame in enumerate(gt_boundaries):
            # Find closest unmatched algo boundary
            best_algo_idx = None
            best_error = float('inf')

            for algo_idx, algo_frame in enumerate(algo_boundaries):
                if algo_idx in used_algo:
                    continue
                error = algo_frame - gt_frame
                if abs(error) < abs(best_error):
                    best_error = error
                    best_algo_idx = algo_idx

            if best_algo_idx is not None:
                used_algo.add(best_algo_idx)
                matches.append(BoundaryMatch(
                    gt_frame=gt_frame,
                    algo_frame=algo_boundaries[best_algo_idx],
                    error=best_error,
                    matched=True,
                    detection_method=methods[best_algo_idx] if best_algo_idx < len(methods) else "unknown",
                    confidence=confidences[best_algo_idx] if best_algo_idx < len(confidences) else 1.0
                ))
            else:
                # No match found - missed boundary
                matches.append(BoundaryMatch(
                    gt_frame=gt_frame,
                    algo_frame=None,
                    error=0,
                    matched=False
                ))

        # Find missed GT boundaries (indices)
        missed = [i for i, m in enumerate(matches) if not m.matched]

        # Find extra algo boundaries (frames not matched to any GT)
        extra = [algo_boundaries[i] for i in range(len(algo_boundaries)) if i not in used_algo]

        return matches, missed, extra

    def categorize_errors(self, result: SegEvalResult):
        """Categorize errors from result into error categories."""
        if not result.success:
            return

        # Missed boundaries
        for idx in result.missed_boundaries:
            if idx < len(result.matches):
                self.error_categories["missed_boundaries"].add_example(
                    result.video_id,
                    {"boundary_index": idx, "gt_frame": result.matches[idx].gt_frame}
                )

        # Extra boundaries
        for frame in result.extra_boundaries:
            self.error_categories["extra_boundaries"].add_example(
                result.video_id,
                {"frame": frame}
            )

        # Early detections
        for m in result.early_detections:
            self.error_categories["early_detections"].add_example(
                result.video_id,
                {"gt_frame": m.gt_frame, "algo_frame": m.algo_frame, "error": m.error,
                 "confidence": m.confidence}
            )

        # Late detections
        for m in result.late_detections:
            self.error_categories["late_detections"].add_example(
                result.video_id,
                {"gt_frame": m.gt_frame, "algo_frame": m.algo_frame, "error": m.error,
                 "confidence": m.confidence}
            )

        # Fallback failures
        for m in result.matches:
            if m.detection_method == "fallback" and abs(m.error) > self.tolerance:
                self.error_categories["fallback_failures"].add_example(
                    result.video_id,
                    {"gt_frame": m.gt_frame, "algo_frame": m.algo_frame, "error": m.error}
                )

        # Low confidence errors
        for m in result.low_confidence_errors:
            self.error_categories["low_confidence_errors"].add_example(
                result.video_id,
                {"gt_frame": m.gt_frame, "algo_frame": m.algo_frame, "error": m.error,
                 "confidence": m.confidence}
            )

    def _format_overall_metrics(self) -> List[str]:
        """Format overall segmentation metrics."""
        if not self.results:
            return []

        successful = [r for r in self.results if r.success and isinstance(r, SegEvalResult)]
        if not successful:
            return []

        # Aggregate metrics
        all_accuracies = [r.accuracy for r in successful]
        all_mean_errors = [r.mean_error for r in successful]
        all_max_errors = [r.max_error for r in successful]

        lines = [
            f"Overall Metrics (±{self.tolerance} frame tolerance):",
            f"  Mean accuracy: {np.mean(all_accuracies):.1%}",
            f"  Mean frame error: {np.mean(all_mean_errors):.1f} frames",
            f"  Max frame error: {max(all_max_errors)} frames",
        ]

        # Detection method stats
        total_primary = sum(r.n_primary for r in successful)
        total_fallback = sum(r.n_fallback for r in successful)
        if total_fallback > 0:
            fallback_pct = total_fallback / (total_primary + total_fallback)
            avg_fallback_error = np.mean([r.fallback_error_rate for r in successful if r.n_fallback > 0])
            lines.append(f"  Fallback method used: {fallback_pct:.1%} ({total_fallback} boundaries)")
            lines.append(f"  Fallback error rate: {avg_fallback_error:.1%}")

        return lines

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []

        # Check for common patterns
        late = self.error_categories.get("late_detections", ErrorCategory("", ""))
        early = self.error_categories.get("early_detections", ErrorCategory("", ""))
        fallback = self.error_categories.get("fallback_failures", ErrorCategory("", ""))
        low_conf = self.error_categories.get("low_confidence_errors", ErrorCategory("", ""))
        missed = self.error_categories.get("missed_boundaries", ErrorCategory("", ""))

        # Timing bias
        if late.count > early.count * 2:
            recommendations.append(
                "Algorithm tends to detect boundaries LATE. Consider adjusting the crossing "
                "detection threshold to trigger earlier."
            )
        elif early.count > late.count * 2:
            recommendations.append(
                "Algorithm tends to detect boundaries EARLY. Consider requiring more "
                "confidence before triggering crossing detection."
            )

        # Fallback issues
        if fallback.count > 5:
            recommendations.append(
                f"Fallback detection method failed {fallback.count} times. Consider "
                "lowering the primary detection threshold or improving fallback logic."
            )

        # Low confidence correlation
        if low_conf.count > 5:
            # Check if low confidence predicts errors
            recommendations.append(
                f"Low confidence (<0.7) predictions are often wrong ({low_conf.count} cases). "
                "Consider flagging these for manual review or adjusting confidence calculation."
            )

        # Missed boundaries
        if missed.count > 0:
            recommendations.append(
                f"{missed.count} boundaries were completely missed. Check for DLC tracking "
                "dropouts or unusual mouse behavior at these frames."
            )

        if not recommendations:
            recommendations.append("Algorithm is performing well. No major issues detected.")

        return recommendations
