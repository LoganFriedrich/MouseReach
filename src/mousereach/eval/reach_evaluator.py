"""
Reach detection evaluation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .base import BaseEvaluator, EvalResult, ErrorCategory


@dataclass
class ReachMatch:
    """Match between a GT reach and an algorithm reach."""
    segment_num: int
    gt_reach_id: int
    algo_reach_id: Optional[int]
    matched: bool = False

    # Timing errors (if matched)
    start_error: int = 0
    apex_error: int = 0
    end_error: int = 0

    # Extent error
    extent_gt: float = 0.0
    extent_algo: float = 0.0
    extent_error: float = 0.0  # Relative error


@dataclass
class ReachEvalResult(EvalResult):
    """Reach detection evaluation result for one video."""
    # Counts
    n_segments: int = 0
    n_gt_reaches: int = 0
    n_algo_reaches: int = 0

    # Per-segment counts
    segment_count_errors: List[Dict] = field(default_factory=list)  # [{seg, gt, algo, diff}]

    # Reach matches
    matches: List[ReachMatch] = field(default_factory=list)

    # Metrics
    precision: float = 0.0  # TP / (TP + FP)
    recall: float = 0.0  # TP / (TP + FN)
    f1: float = 0.0

    # Timing accuracy (for matched reaches)
    mean_start_error: float = 0.0
    mean_apex_error: float = 0.0
    mean_end_error: float = 0.0

    # Extent accuracy
    extent_correlation: float = 0.0
    mean_extent_error: float = 0.0

    # Error lists
    false_positives: List[Dict] = field(default_factory=list)
    false_negatives: List[Dict] = field(default_factory=list)
    merged_reaches: List[Dict] = field(default_factory=list)  # 2 GT -> 1 algo
    split_reaches: List[Dict] = field(default_factory=list)  # 1 GT -> 2 algo


class ReachEvaluator(BaseEvaluator):
    """Evaluate reach detection accuracy."""

    gt_pattern = "*_reach_ground_truth.json"
    algo_pattern = "*_reaches.json"
    step_name = "reach detection"

    # Tolerance for reach matching (frames)
    FRAME_TOLERANCE = 10  # Reaches within 10 frames are potential matches
    IOU_THRESHOLD = 0.5  # IoU threshold for considering a match

    def __init__(self, gt_dir: Path = None, algo_dir: Path = None, tolerance: int = None):
        super().__init__(gt_dir, algo_dir, tolerance or self.FRAME_TOLERANCE)
        self.frame_tolerance = int(self.tolerance)

    def _init_error_categories(self):
        """Initialize reach detection error categories."""
        self.error_categories = {
            "missed_reaches": ErrorCategory(
                "missed_reaches",
                "GT reach not detected by algorithm (false negative)"
            ),
            "phantom_reaches": ErrorCategory(
                "phantom_reaches",
                "Algorithm detected reach not in GT (false positive)"
            ),
            "merged_reaches": ErrorCategory(
                "merged_reaches",
                "Two GT reaches matched to single algo reach"
            ),
            "split_reaches": ErrorCategory(
                "split_reaches",
                "Single GT reach matched to multiple algo reaches"
            ),
            "timing_errors": ErrorCategory(
                "timing_errors",
                "Matched but start/apex/end timing significantly off"
            ),
            "extent_underestimate": ErrorCategory(
                "extent_underestimate",
                "Algorithm extent < GT extent (missed full reach)"
            ),
            "extent_overestimate": ErrorCategory(
                "extent_overestimate",
                "Algorithm extent > GT extent"
            ),
        }

    def load_ground_truth(self, video_id: str) -> Optional[Dict]:
        """Load reach ground truth."""
        gt_path = self.gt_dir / f"{video_id}_reach_ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                return json.load(f)
        return None

    def load_algorithm_output(self, video_id: str) -> Optional[Dict]:
        """Load reach algorithm output."""
        algo_path = self.algo_dir / f"{video_id}_reaches.json"
        if algo_path.exists():
            with open(algo_path) as f:
                return json.load(f)
        return None

    def compare(self, video_id: str) -> ReachEvalResult:
        """Compare reach detection algorithm vs ground truth."""
        result = ReachEvalResult(video_id=video_id)

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

        gt_segments = gt.get("segments", [])
        algo_segments = algo.get("segments", [])

        result.n_segments = max(len(gt_segments), len(algo_segments))

        # Process each segment
        all_gt_reaches = []
        all_algo_reaches = []

        for gt_seg in gt_segments:
            seg_num = gt_seg.get("segment_num", 0)
            gt_reaches = gt_seg.get("reaches", [])
            for r in gt_reaches:
                r["_segment_num"] = seg_num
            all_gt_reaches.extend(gt_reaches)

            # Find corresponding algo segment
            algo_seg = next((s for s in algo_segments if s.get("segment_num") == seg_num), None)
            algo_reaches = algo_seg.get("reaches", []) if algo_seg else []
            for r in algo_reaches:
                r["_segment_num"] = seg_num
            all_algo_reaches.extend(algo_reaches)

            # Record count differences
            gt_count = len(gt_reaches)
            algo_count = len(algo_reaches)
            if gt_count != algo_count:
                result.segment_count_errors.append({
                    "segment": seg_num,
                    "gt_count": gt_count,
                    "algo_count": algo_count,
                    "diff": algo_count - gt_count
                })

        result.n_gt_reaches = len(all_gt_reaches)
        result.n_algo_reaches = len(all_algo_reaches)

        # Match reaches using IoU
        matches, fp, fn = self._match_reaches(all_gt_reaches, all_algo_reaches)
        result.matches = matches
        result.false_positives = fp
        result.false_negatives = fn

        # Calculate precision/recall
        tp = sum(1 for m in matches if m.matched)
        if result.n_algo_reaches > 0:
            result.precision = tp / result.n_algo_reaches
        if result.n_gt_reaches > 0:
            result.recall = tp / result.n_gt_reaches
        if result.precision + result.recall > 0:
            result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)

        # Timing accuracy
        matched = [m for m in matches if m.matched]
        if matched:
            result.mean_start_error = np.mean([abs(m.start_error) for m in matched])
            result.mean_apex_error = np.mean([abs(m.apex_error) for m in matched])
            result.mean_end_error = np.mean([abs(m.end_error) for m in matched])

            # Extent accuracy
            gt_extents = [m.extent_gt for m in matched if m.extent_gt > 0]
            algo_extents = [m.extent_algo for m in matched if m.extent_gt > 0]
            if gt_extents and algo_extents:
                result.extent_correlation = np.corrcoef(gt_extents, algo_extents)[0, 1]
                result.mean_extent_error = np.mean([abs(m.extent_error) for m in matched])

        return result

    def _match_reaches(
        self,
        gt_reaches: List[Dict],
        algo_reaches: List[Dict]
    ) -> Tuple[List[ReachMatch], List[Dict], List[Dict]]:
        """Match GT reaches to algorithm reaches using IoU.

        Returns:
            (matches, false_positives, false_negatives)
        """
        matches = []
        used_algo = set()

        for gt_r in gt_reaches:
            gt_start = gt_r.get("start_frame", 0)
            gt_end = gt_r.get("end_frame", 0)
            gt_apex = gt_r.get("apex_frame", (gt_start + gt_end) // 2)
            gt_extent = gt_r.get("max_extent_ruler", 0)
            gt_seg = gt_r.get("_segment_num", 0)
            gt_id = gt_r.get("reach_id", 0)

            best_match = None
            best_iou = 0

            for algo_idx, algo_r in enumerate(algo_reaches):
                if algo_idx in used_algo:
                    continue
                if algo_r.get("_segment_num") != gt_seg:
                    continue

                algo_start = algo_r.get("start_frame", 0)
                algo_end = algo_r.get("end_frame", 0)

                # Calculate IoU
                intersection_start = max(gt_start, algo_start)
                intersection_end = min(gt_end, algo_end)
                intersection = max(0, intersection_end - intersection_start)

                union_start = min(gt_start, algo_start)
                union_end = max(gt_end, algo_end)
                union = union_end - union_start

                iou = intersection / union if union > 0 else 0

                if iou > best_iou and iou >= self.IOU_THRESHOLD:
                    best_iou = iou
                    best_match = (algo_idx, algo_r)

            if best_match:
                algo_idx, algo_r = best_match
                used_algo.add(algo_idx)

                algo_apex = algo_r.get("apex_frame", (algo_r["start_frame"] + algo_r["end_frame"]) // 2)
                algo_extent = algo_r.get("max_extent_ruler", 0)

                matches.append(ReachMatch(
                    segment_num=gt_seg,
                    gt_reach_id=gt_id,
                    algo_reach_id=algo_r.get("reach_id"),
                    matched=True,
                    start_error=algo_r.get("start_frame", 0) - gt_start,
                    apex_error=algo_apex - gt_apex,
                    end_error=algo_r.get("end_frame", 0) - gt_end,
                    extent_gt=gt_extent,
                    extent_algo=algo_extent,
                    extent_error=(algo_extent - gt_extent) / gt_extent if gt_extent > 0 else 0
                ))
            else:
                # No match - false negative
                matches.append(ReachMatch(
                    segment_num=gt_seg,
                    gt_reach_id=gt_id,
                    algo_reach_id=None,
                    matched=False
                ))

        # False positives: algo reaches not matched
        false_positives = [
            {"segment": algo_reaches[i].get("_segment_num"),
             "reach_id": algo_reaches[i].get("reach_id"),
             "start": algo_reaches[i].get("start_frame"),
             "end": algo_reaches[i].get("end_frame")}
            for i in range(len(algo_reaches)) if i not in used_algo
        ]

        # False negatives: unmatched GT reaches
        false_negatives = [
            {"segment": m.segment_num, "reach_id": m.gt_reach_id}
            for m in matches if not m.matched
        ]

        return matches, false_positives, false_negatives

    def categorize_errors(self, result: ReachEvalResult):
        """Categorize reach detection errors."""
        if not result.success:
            return

        # Missed reaches (false negatives)
        for fn in result.false_negatives:
            self.error_categories["missed_reaches"].add_example(
                result.video_id, fn
            )

        # Phantom reaches (false positives)
        for fp in result.false_positives:
            self.error_categories["phantom_reaches"].add_example(
                result.video_id, fp
            )

        # Timing and extent errors
        for m in result.matches:
            if not m.matched:
                continue

            # Significant timing errors
            if abs(m.start_error) > 10 or abs(m.end_error) > 10:
                self.error_categories["timing_errors"].add_example(
                    result.video_id,
                    {"segment": m.segment_num, "reach_id": m.gt_reach_id,
                     "start_error": m.start_error, "end_error": m.end_error}
                )

            # Extent errors
            if m.extent_error < -0.2:  # 20% underestimate
                self.error_categories["extent_underestimate"].add_example(
                    result.video_id,
                    {"segment": m.segment_num, "reach_id": m.gt_reach_id,
                     "gt_extent": m.extent_gt, "algo_extent": m.extent_algo}
                )
            elif m.extent_error > 0.2:  # 20% overestimate
                self.error_categories["extent_overestimate"].add_example(
                    result.video_id,
                    {"segment": m.segment_num, "reach_id": m.gt_reach_id,
                     "gt_extent": m.extent_gt, "algo_extent": m.extent_algo}
                )

    def _format_overall_metrics(self) -> List[str]:
        """Format overall reach detection metrics."""
        if not self.results:
            return []

        successful = [r for r in self.results if r.success and isinstance(r, ReachEvalResult)]
        if not successful:
            return []

        lines = [
            "Overall Metrics:",
            f"  Precision: {np.mean([r.precision for r in successful]):.1%}",
            f"  Recall: {np.mean([r.recall for r in successful]):.1%}",
            f"  F1 Score: {np.mean([r.f1 for r in successful]):.2f}",
            "",
            "Timing Accuracy:",
            f"  Mean start error: {np.mean([r.mean_start_error for r in successful]):.1f} frames",
            f"  Mean apex error: {np.mean([r.mean_apex_error for r in successful]):.1f} frames",
            f"  Mean end error: {np.mean([r.mean_end_error for r in successful]):.1f} frames",
        ]

        # Extent correlation
        corrs = [r.extent_correlation for r in successful if not np.isnan(r.extent_correlation)]
        if corrs:
            lines.append(f"  Extent correlation: {np.mean(corrs):.2f}")

        return lines

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for reach detection."""
        recommendations = []

        missed = self.error_categories.get("missed_reaches", ErrorCategory("", ""))
        phantom = self.error_categories.get("phantom_reaches", ErrorCategory("", ""))
        timing = self.error_categories.get("timing_errors", ErrorCategory("", ""))
        underest = self.error_categories.get("extent_underestimate", ErrorCategory("", ""))

        if missed.count > phantom.count * 2:
            recommendations.append(
                f"Algorithm is missing {missed.count} reaches (high false negative rate). "
                "Consider lowering the detection threshold or extent requirement."
            )
        elif phantom.count > missed.count * 2:
            recommendations.append(
                f"Algorithm has {phantom.count} phantom detections (high false positive rate). "
                "Consider increasing the detection threshold."
            )

        if timing.count > 5:
            recommendations.append(
                f"{timing.count} reaches have significant timing errors. "
                "Check the start/end detection logic for edge cases."
            )

        if underest.count > 5:
            recommendations.append(
                f"{underest.count} reaches have underestimated extent. "
                "The algorithm may be ending reaches too early."
            )

        if not recommendations:
            recommendations.append("Reach detection is performing well.")

        return recommendations
