"""
Base classes for algorithm evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime


@dataclass
class ErrorCategory:
    """A category of errors with examples."""
    name: str
    description: str
    count: int = 0
    examples: List[Dict] = field(default_factory=list)

    def add_example(self, video_id: str, details: Dict):
        """Add an example to this error category."""
        self.count += 1
        if len(self.examples) < 10:  # Keep at most 10 examples
            self.examples.append({
                "video_id": video_id,
                **details
            })


@dataclass
class EvalResult:
    """Base class for evaluation results."""
    video_id: str
    success: bool = True
    error_message: str = ""

    # Timing
    evaluated_at: str = ""

    def __post_init__(self):
        if not self.evaluated_at:
            self.evaluated_at = datetime.now().isoformat()


class BaseEvaluator(ABC):
    """
    Base class for algorithm evaluators.

    Subclasses implement specific evaluation logic for:
    - Segmentation boundary detection
    - Reach detection
    - Pellet outcome classification
    """

    # Override in subclasses
    gt_pattern: str = "*_ground_truth.json"
    algo_pattern: str = "*.json"
    step_name: str = "unknown"

    def __init__(
        self,
        gt_dir: Path = None,
        algo_dir: Path = None,
        tolerance: float = 0.0
    ):
        """
        Initialize evaluator.

        Args:
            gt_dir: Directory containing ground truth files
            algo_dir: Directory containing algorithm output files (defaults to gt_dir)
            tolerance: Tolerance for comparisons (meaning varies by evaluator)
        """
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.algo_dir = Path(algo_dir) if algo_dir else self.gt_dir
        self.tolerance = tolerance

        self.results: List[EvalResult] = []
        self.error_categories: Dict[str, ErrorCategory] = {}

        self._init_error_categories()

    @abstractmethod
    def _init_error_categories(self):
        """Initialize error categories specific to this evaluator."""
        pass

    @abstractmethod
    def load_ground_truth(self, video_id: str) -> Optional[Dict]:
        """Load ground truth file for a video.

        Args:
            video_id: Video identifier

        Returns:
            Ground truth data dict, or None if not found
        """
        pass

    @abstractmethod
    def load_algorithm_output(self, video_id: str) -> Optional[Dict]:
        """Load algorithm output file for a video.

        Args:
            video_id: Video identifier

        Returns:
            Algorithm output data dict, or None if not found
        """
        pass

    @abstractmethod
    def compare(self, video_id: str) -> EvalResult:
        """Compare algorithm output vs ground truth for one video.

        Args:
            video_id: Video identifier

        Returns:
            Evaluation result with metrics and error details
        """
        pass

    @abstractmethod
    def categorize_errors(self, result: EvalResult):
        """Categorize errors from a result into error categories.

        Args:
            result: Evaluation result to categorize
        """
        pass

    def find_gt_files(self) -> List[Path]:
        """Find all ground truth files in the GT directory."""
        if not self.gt_dir or not self.gt_dir.exists():
            return []
        return sorted(self.gt_dir.glob(self.gt_pattern))

    def extract_video_id(self, path: Path) -> str:
        """Extract video ID from a file path.

        Override if your naming convention differs.
        """
        # Default: remove common suffixes
        # NOTE: Order matters - longer suffixes must come first
        name = path.stem
        for suffix in ["_outcomes_ground_truth", "_outcome_ground_truth",
                       "_reach_ground_truth", "_seg_ground_truth",
                       "_ground_truth", "_segments", "_reaches", "_pellet_outcomes"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break  # Only remove one suffix
        return name

    def evaluate_all(self, progress_callback: Callable[[int, int, str], None] = None) -> List[EvalResult]:
        """Evaluate all videos with ground truth files.

        Args:
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of evaluation results
        """
        self.results = []
        self.error_categories = {}
        self._init_error_categories()

        gt_files = self.find_gt_files()
        total = len(gt_files)

        if total == 0:
            print(f"No ground truth files found matching {self.gt_pattern}")
            return self.results

        for i, gt_file in enumerate(gt_files):
            video_id = self.extract_video_id(gt_file)

            if progress_callback:
                progress_callback(i + 1, total, f"Evaluating {video_id}...")

            try:
                result = self.compare(video_id)
                self.results.append(result)
                self.categorize_errors(result)
            except Exception as e:
                self.results.append(EvalResult(
                    video_id=video_id,
                    success=False,
                    error_message=str(e)
                ))

        return self.results

    def get_summary(self) -> Dict:
        """Get summary statistics from all results."""
        if not self.results:
            return {"n_videos": 0}

        successful = [r for r in self.results if r.success]

        return {
            "n_videos": len(self.results),
            "n_successful": len(successful),
            "n_failed": len(self.results) - len(successful),
            "error_categories": {
                name: cat.count
                for name, cat in self.error_categories.items()
            }
        }

    def generate_report(self, format: str = "text", detailed: bool = True) -> str:
        """Generate a human-readable report.

        Args:
            format: Output format ("text", "markdown")
            detailed: Include detailed error examples

        Returns:
            Formatted report string
        """
        lines = []
        summary = self.get_summary()

        # Header
        header = f"=== {self.step_name.upper()} EVALUATION ==="
        lines.append(header)
        lines.append("=" * len(header))
        lines.append("")

        # Summary
        lines.append(f"Videos evaluated: {summary['n_videos']}")
        lines.append(f"Successful: {summary['n_successful']}")
        if summary['n_failed'] > 0:
            lines.append(f"Failed: {summary['n_failed']}")
        lines.append("")

        # Overall metrics (subclass-specific)
        metrics_lines = self._format_overall_metrics()
        if metrics_lines:
            lines.extend(metrics_lines)
            lines.append("")

        # Error categories
        if any(cat.count > 0 for cat in self.error_categories.values()):
            lines.append("Error Categories:")
            for name, cat in sorted(self.error_categories.items(), key=lambda x: -x[1].count):
                if cat.count == 0:
                    continue
                lines.append(f"  {name}: {cat.count}")
                if detailed and cat.examples:
                    for ex in cat.examples[:3]:  # Show up to 3 examples
                        ex_str = self._format_error_example(ex)
                        lines.append(f"    - {ex_str}")
            lines.append("")

        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            lines.append("RECOMMENDATIONS:")
            for rec in recommendations:
                lines.append(f"  • {rec}")
            lines.append("")

        return "\n".join(lines)

    @abstractmethod
    def _format_overall_metrics(self) -> List[str]:
        """Format overall metrics for report.

        Returns:
            List of formatted lines
        """
        pass

    def _format_error_example(self, example: Dict) -> str:
        """Format an error example for the report.

        Override for custom formatting.
        """
        video_id = example.get("video_id", "?")
        details = {k: v for k, v in example.items() if k != "video_id"}
        return f"{video_id}: {details}"

    @abstractmethod
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns.

        Returns:
            List of recommendation strings
        """
        pass
