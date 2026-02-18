"""
Aggregate Evaluation - Combines all evaluators and filters for human-verified data.

Provides:
- Unified evaluation across all GT-able features
- Human-verification filtering
- Version-based comparison
- Historical trend analysis
- Timing accuracy breakdowns (exact, +/-1fr, +/-2fr, etc.)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base import ErrorCategory
from .seg_evaluator import SegmentationEvaluator, SegEvalResult
from .reach_evaluator import ReachEvaluator, ReachEvalResult
from .outcome_evaluator import OutcomeEvaluator, OutcomeEvalResult


@dataclass
class TimingBreakdown:
    """Breakdown of timing accuracy at different tolerances."""
    exact: float = 0.0  # % within 0 frames
    within_1: float = 0.0  # % within 1 frame
    within_2: float = 0.0  # % within 2 frames
    within_5: float = 0.0  # % within 5 frames
    n_samples: int = 0

    @classmethod
    def from_errors(cls, errors: List[int]) -> "TimingBreakdown":
        """Compute breakdown from a list of timing errors."""
        if not errors:
            return cls()

        abs_errors = [abs(e) for e in errors]
        n = len(abs_errors)

        return cls(
            exact=sum(1 for e in abs_errors if e == 0) / n,
            within_1=sum(1 for e in abs_errors if e <= 1) / n,
            within_2=sum(1 for e in abs_errors if e <= 2) / n,
            within_5=sum(1 for e in abs_errors if e <= 5) / n,
            n_samples=n
        )


@dataclass
class FeatureMetrics:
    """Metrics for a single GT-able feature."""
    feature_name: str
    n_gt_files: int = 0
    n_human_verified: int = 0  # GT items that show human intervention
    n_items: int = 0  # Total items evaluated

    # Detection metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Timing accuracy (if applicable)
    start_timing: Optional[TimingBreakdown] = None
    end_timing: Optional[TimingBreakdown] = None

    # Error breakdown
    error_categories: Dict[str, int] = field(default_factory=dict)

    # Version info
    algorithm_version: str = ""
    evaluated_at: str = ""


@dataclass
class AggregateResult:
    """Aggregated evaluation results across all features."""
    # Per-feature results
    segments: Optional[FeatureMetrics] = None
    reaches: Optional[FeatureMetrics] = None
    outcomes: Optional[FeatureMetrics] = None

    # Summary
    total_gt_files: int = 0
    total_human_verified: int = 0

    # Metadata
    evaluated_at: str = ""
    algorithm_versions: Dict[str, str] = field(default_factory=dict)

    # Historical data
    previous_results: List[Dict] = field(default_factory=list)


class HumanVerificationDetector:
    """
    Detects whether GT data has genuine human intervention.

    Uses multiple signals:
    - `original_outcome` vs current outcome differences
    - `source: 'human_added'` markers
    - `human_corrected: true` flags
    - Timestamp presence in correction fields
    """

    @staticmethod
    def is_human_verified_outcome(segment: Dict) -> bool:
        """Check if an outcome segment has human verification."""
        # Check for original_outcome diff
        if "original_outcome" in segment:
            if segment.get("outcome") != segment.get("original_outcome"):
                return True

        # Check human_corrected flag
        if segment.get("human_corrected"):
            return True

        # Check correction timestamp
        if segment.get("correction_timestamp"):
            return True

        return False

    @staticmethod
    def is_human_verified_reach(reach: Dict) -> bool:
        """Check if a reach has human verification."""
        # Check source field
        if reach.get("source") == "human_added":
            return True

        # Check for timing corrections
        if reach.get("timing_corrected"):
            return True

        # Check manual annotation marker
        if reach.get("manual") or reach.get("manually_annotated"):
            return True

        return False

    @staticmethod
    def count_human_verified_in_file(gt_path: Path) -> Tuple[int, int]:
        """
        Count total items and human-verified items in a GT file.

        Returns:
            (total_items, human_verified_count)
        """
        try:
            with open(gt_path) as f:
                data = json.load(f)
        except:
            return (0, 0)

        total = 0
        verified = 0

        # Handle different GT file types
        if "_reach_ground_truth" in gt_path.name:
            for seg in data.get("segments", []):
                for reach in seg.get("reaches", []):
                    total += 1
                    if HumanVerificationDetector.is_human_verified_reach(reach):
                        verified += 1

        elif "_outcome_ground_truth" in gt_path.name:
            for seg in data.get("segments", []):
                total += 1
                if HumanVerificationDetector.is_human_verified_outcome(seg):
                    verified += 1

        elif "_seg_ground_truth" in gt_path.name:
            # Segment boundaries - count boundary corrections
            boundaries = data.get("boundaries", [])
            total = len(boundaries)
            # Check for correction markers
            for b in boundaries:
                if isinstance(b, dict) and b.get("human_corrected"):
                    verified += 1

        return (total, verified)


class AggregateEvaluator:
    """
    Evaluates algorithm performance across all GT-able features.

    Features:
    - Segment boundaries
    - Reaches (detection + timing)
    - Outcomes (classification + interaction timing)

    Supports:
    - Human-verification filtering
    - Version comparison
    - Historical trend tracking
    """

    def __init__(self, processing_root: Path = None):
        """
        Initialize aggregate evaluator.

        Args:
            processing_root: Root directory containing Processing folders
        """
        if processing_root is None:
            from mousereach.config import PROCESSING_ROOT
            processing_root = PROCESSING_ROOT

        self.processing_root = Path(processing_root)
        self.verification_detector = HumanVerificationDetector()

        # Results cache
        self._last_result: Optional[AggregateResult] = None

    def find_all_gt_files(self) -> Dict[str, List[Path]]:
        """
        Find all GT files organized by type.

        Returns:
            Dict with keys: 'reach', 'outcome', 'segment'
        """
        gt_files = {
            'reach': [],
            'outcome': [],
            'segment': []
        }

        # Search in Processing folders
        for processing_dir in self.processing_root.glob("*/Processing"):
            # Reach GT
            gt_files['reach'].extend(processing_dir.glob("*_reach_ground_truth.json"))

            # Outcome GT
            gt_files['outcome'].extend(processing_dir.glob("*_outcome_ground_truth.json"))
            gt_files['outcome'].extend(processing_dir.glob("*_outcomes_ground_truth.json"))

            # Segment GT
            gt_files['segment'].extend(processing_dir.glob("*_seg_ground_truth.json"))

        return gt_files

    def evaluate_all(
        self,
        human_verified_only: bool = True,
        progress_callback=None
    ) -> AggregateResult:
        """
        Evaluate all GT files and aggregate results.

        Args:
            human_verified_only: If True, only count human-verified items
            progress_callback: Optional callback(current, total, message)

        Returns:
            AggregateResult with metrics for all features
        """
        result = AggregateResult(evaluated_at=datetime.now().isoformat())

        gt_files = self.find_all_gt_files()

        # Count GT files
        result.total_gt_files = sum(len(files) for files in gt_files.values())

        # Evaluate reaches
        if gt_files['reach']:
            result.reaches = self._evaluate_reaches(
                gt_files['reach'],
                human_verified_only,
                progress_callback
            )

        # Evaluate outcomes
        if gt_files['outcome']:
            result.outcomes = self._evaluate_outcomes(
                gt_files['outcome'],
                human_verified_only,
                progress_callback
            )

        # Evaluate segments
        if gt_files['segment']:
            result.segments = self._evaluate_segments(
                gt_files['segment'],
                human_verified_only,
                progress_callback
            )

        # Count total human-verified
        result.total_human_verified = sum([
            result.reaches.n_human_verified if result.reaches else 0,
            result.outcomes.n_human_verified if result.outcomes else 0,
            result.segments.n_human_verified if result.segments else 0,
        ])

        self._last_result = result
        return result

    def _evaluate_reaches(
        self,
        gt_files: List[Path],
        human_verified_only: bool,
        progress_callback
    ) -> FeatureMetrics:
        """Evaluate reach detection across all GT files."""
        metrics = FeatureMetrics(
            feature_name="reaches",
            n_gt_files=len(gt_files),
            evaluated_at=datetime.now().isoformat()
        )

        all_start_errors = []
        all_end_errors = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        error_counts = {}

        for i, gt_path in enumerate(gt_files):
            if progress_callback:
                progress_callback(i + 1, len(gt_files), f"Evaluating {gt_path.stem}...")

            # Count human verification
            total_items, verified = self.verification_detector.count_human_verified_in_file(gt_path)
            metrics.n_items += total_items
            metrics.n_human_verified += verified

            # Run reach evaluator
            gt_dir = gt_path.parent
            video_id = gt_path.stem.replace("_reach_ground_truth", "")

            evaluator = ReachEvaluator(gt_dir=gt_dir, algo_dir=gt_dir)
            result = evaluator.compare(video_id)

            if not result.success:
                continue

            # Accumulate metrics
            tp = sum(1 for m in result.matches if m.matched)
            total_tp += tp
            total_fp += len(result.false_positives)
            total_fn += len(result.false_negatives)

            # Collect timing errors for matched reaches
            for m in result.matches:
                if m.matched:
                    all_start_errors.append(m.start_error)
                    all_end_errors.append(m.end_error)

            # Categorize errors
            evaluator.categorize_errors(result)
            for cat_name, cat in evaluator.error_categories.items():
                error_counts[cat_name] = error_counts.get(cat_name, 0) + cat.count

        # Calculate aggregate precision/recall/F1
        if total_tp + total_fp > 0:
            metrics.precision = total_tp / (total_tp + total_fp)
        if total_tp + total_fn > 0:
            metrics.recall = total_tp / (total_tp + total_fn)
        if metrics.precision + metrics.recall > 0:
            metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

        # Timing breakdowns
        if all_start_errors:
            metrics.start_timing = TimingBreakdown.from_errors(all_start_errors)
        if all_end_errors:
            metrics.end_timing = TimingBreakdown.from_errors(all_end_errors)

        metrics.error_categories = error_counts

        return metrics

    def _evaluate_outcomes(
        self,
        gt_files: List[Path],
        human_verified_only: bool,
        progress_callback
    ) -> FeatureMetrics:
        """Evaluate outcome classification across all GT files."""
        metrics = FeatureMetrics(
            feature_name="outcomes",
            n_gt_files=len(gt_files),
            evaluated_at=datetime.now().isoformat()
        )

        accuracies = []
        error_counts = {}

        for i, gt_path in enumerate(gt_files):
            if progress_callback:
                progress_callback(i + 1, len(gt_files), f"Evaluating {gt_path.stem}...")

            # Count human verification
            total_items, verified = self.verification_detector.count_human_verified_in_file(gt_path)
            metrics.n_items += total_items
            metrics.n_human_verified += verified

            # Run outcome evaluator
            gt_dir = gt_path.parent
            video_id = gt_path.stem.replace("_outcome_ground_truth", "").replace("_outcomes_ground_truth", "")

            evaluator = OutcomeEvaluator(gt_dir=gt_dir, algo_dir=gt_dir)
            result = evaluator.compare(video_id)

            if not result.success:
                continue

            accuracies.append(result.accuracy)

            # Categorize errors
            evaluator.categorize_errors(result)
            for cat_name, cat in evaluator.error_categories.items():
                error_counts[cat_name] = error_counts.get(cat_name, 0) + cat.count

        if accuracies:
            metrics.precision = np.mean(accuracies)  # For outcomes, accuracy ≈ precision
            metrics.recall = np.mean(accuracies)
            metrics.f1 = np.mean(accuracies)

        metrics.error_categories = error_counts

        return metrics

    def _evaluate_segments(
        self,
        gt_files: List[Path],
        human_verified_only: bool,
        progress_callback
    ) -> FeatureMetrics:
        """Evaluate segmentation boundaries across all GT files."""
        metrics = FeatureMetrics(
            feature_name="segments",
            n_gt_files=len(gt_files),
            evaluated_at=datetime.now().isoformat()
        )

        accuracies = []
        error_counts = {}

        for i, gt_path in enumerate(gt_files):
            if progress_callback:
                progress_callback(i + 1, len(gt_files), f"Evaluating {gt_path.stem}...")

            # Count human verification
            total_items, verified = self.verification_detector.count_human_verified_in_file(gt_path)
            metrics.n_items += total_items
            metrics.n_human_verified += verified

            # Run segment evaluator
            gt_dir = gt_path.parent
            video_id = gt_path.stem.replace("_seg_ground_truth", "")

            evaluator = SegmentationEvaluator(gt_dir=gt_dir, algo_dir=gt_dir)
            result = evaluator.compare(video_id)

            if not result.success:
                continue

            if hasattr(result, 'accuracy'):
                accuracies.append(result.accuracy)

            # Categorize errors
            evaluator.categorize_errors(result)
            for cat_name, cat in evaluator.error_categories.items():
                error_counts[cat_name] = error_counts.get(cat_name, 0) + cat.count

        if accuracies:
            metrics.precision = np.mean(accuracies)
            metrics.recall = np.mean(accuracies)
            metrics.f1 = np.mean(accuracies)

        metrics.error_categories = error_counts

        return metrics

    def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance between two algorithm versions.

        Args:
            version_a: First version string
            version_b: Second version string (usually newer)

        Returns:
            Dict with delta values for each metric
        """
        # Load historical results
        from .history import load_historical_results

        results_a = load_historical_results(version_a)
        results_b = load_historical_results(version_b)

        comparison = {}

        for feature in ['reaches', 'outcomes', 'segments']:
            if feature in results_a and feature in results_b:
                a = results_a[feature]
                b = results_b[feature]

                comparison[feature] = {
                    'recall_delta': b.get('recall', 0) - a.get('recall', 0),
                    'precision_delta': b.get('precision', 0) - a.get('precision', 0),
                    'f1_delta': b.get('f1', 0) - a.get('f1', 0),
                }

        return comparison

    def get_trend_data(self, feature: str, metric: str, n_points: int = 20) -> List[Dict]:
        """
        Get historical trend data for visualization.

        Args:
            feature: 'reaches', 'outcomes', or 'segments'
            metric: 'recall', 'precision', 'f1', etc.
            n_points: Maximum data points to return

        Returns:
            List of {date, version, value} dicts
        """
        from mousereach.performance import PerformanceLogger

        logger = PerformanceLogger()

        # Map feature to log type
        log_map = {
            'reaches': 'reach',
            'outcomes': 'outcome',
            'segments': 'segmentation'
        }

        log_type = log_map.get(feature)
        if not log_type:
            return []

        entries = logger.get_entries(log_type)

        # Group by date and version
        points = []
        for entry in entries:
            metrics = entry.get('metrics', {})
            if metric in metrics:
                points.append({
                    'date': entry.get('logged_at', '')[:10],
                    'version': entry.get('algorithm_version', 'unknown'),
                    'value': metrics[metric]
                })

        # Sort by date and take last n_points
        points.sort(key=lambda x: x['date'])
        return points[-n_points:]

    def generate_report(self, format: str = 'text') -> str:
        """
        Generate a human-readable report.

        Args:
            format: 'text' or 'markdown'

        Returns:
            Formatted report string
        """
        if not self._last_result:
            self.evaluate_all()

        result = self._last_result
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("ALGORITHM PERFORMANCE REPORT")
        lines.append(f"Generated: {result.evaluated_at[:19]}")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append(f"Total GT files: {result.total_gt_files}")
        lines.append(f"Human-verified items: {result.total_human_verified}")
        lines.append("")

        # Per-feature reports
        for feature_name, metrics in [
            ("SEGMENT BOUNDARIES", result.segments),
            ("REACHES", result.reaches),
            ("OUTCOMES", result.outcomes)
        ]:
            lines.append(f"=== {feature_name} ===")

            if metrics is None or metrics.n_gt_files == 0:
                lines.append("  No GT data available")
                lines.append("")
                continue

            lines.append(f"  GT files: {metrics.n_gt_files}")
            lines.append(f"  Human-verified: {metrics.n_human_verified} items")
            lines.append("")
            lines.append(f"  Detection:")
            lines.append(f"    Precision: {metrics.precision:.1%}")
            lines.append(f"    Recall: {metrics.recall:.1%}")
            lines.append(f"    F1: {metrics.f1:.2f}")

            # Timing breakdown
            if metrics.start_timing and metrics.start_timing.n_samples > 0:
                lines.append("")
                lines.append(f"  Start Timing ({metrics.start_timing.n_samples} samples):")
                lines.append(f"    Exact: {metrics.start_timing.exact:.0%}")
                lines.append(f"    +/-1 frame: {metrics.start_timing.within_1:.0%}")
                lines.append(f"    +/-2 frames: {metrics.start_timing.within_2:.0%}")

            if metrics.end_timing and metrics.end_timing.n_samples > 0:
                lines.append("")
                lines.append(f"  End Timing ({metrics.end_timing.n_samples} samples):")
                lines.append(f"    Exact: {metrics.end_timing.exact:.0%}")
                lines.append(f"    +/-1 frame: {metrics.end_timing.within_1:.0%}")
                lines.append(f"    +/-2 frames: {metrics.end_timing.within_2:.0%}")

            # Error categories
            if metrics.error_categories:
                lines.append("")
                lines.append("  Error breakdown:")
                for cat, count in sorted(metrics.error_categories.items(), key=lambda x: -x[1]):
                    if count > 0:
                        lines.append(f"    {cat}: {count}")

            lines.append("")

        return "\n".join(lines)


# Convenience functions
def evaluate_all_gt(human_verified_only: bool = True) -> AggregateResult:
    """Convenience function to evaluate all GT files."""
    evaluator = AggregateEvaluator()
    return evaluator.evaluate_all(human_verified_only=human_verified_only)


def get_performance_summary() -> Dict[str, Any]:
    """Get a quick summary of current algorithm performance."""
    evaluator = AggregateEvaluator()
    result = evaluator.evaluate_all()

    return {
        'total_gt_files': result.total_gt_files,
        'total_human_verified': result.total_human_verified,
        'reaches': {
            'recall': result.reaches.recall if result.reaches else 0,
            'precision': result.reaches.precision if result.reaches else 0,
            'f1': result.reaches.f1 if result.reaches else 0,
        } if result.reaches else None,
        'outcomes': {
            'accuracy': result.outcomes.f1 if result.outcomes else 0,
        } if result.outcomes else None,
        'segments': {
            'accuracy': result.segments.f1 if result.segments else 0,
        } if result.segments else None,
    }
