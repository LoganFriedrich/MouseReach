"""
Scientific report generation for algorithm performance.

Generates publication-ready reports with:
- Algorithm descriptions
- Performance metrics tables
- Error analysis
- Methods section text
"""

from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from .logger import PerformanceLogger


class ScientificReportGenerator:
    """Generate publication-ready performance reports."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize with optional custom log directory."""
        self.logger = PerformanceLogger(log_dir)

    def generate(self, format: str = "markdown") -> str:
        """
        Generate report in specified format.

        Args:
            format: "markdown", "methods", or "table"

        Returns:
            Formatted report string
        """
        if format == "methods":
            return self._format_methods_section()
        elif format == "table":
            return self._format_performance_table()
        else:
            return self._format_markdown_report()

    def _format_markdown_report(self) -> str:
        """Generate full markdown report."""
        lines = []

        lines.append("# MouseReach Algorithm Performance Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Summary table
        lines.append("## Performance Summary")
        lines.append("")
        lines.append(self._format_performance_table())
        lines.append("")

        # Detailed sections for each algorithm
        for algo, key in [("Segmentation", "segmentation"),
                          ("Reach Detection", "reach"),
                          ("Outcome Classification", "outcome")]:
            section = self._format_algorithm_section(key)
            if section:
                lines.append(f"## {algo}")
                lines.append("")
                lines.append(section)
                lines.append("")

        # Methods section
        lines.append("## Methods Section (for publication)")
        lines.append("")
        lines.append(self._format_methods_section())
        lines.append("")

        return "\n".join(lines)

    def _format_performance_table(self) -> str:
        """Generate markdown performance table."""
        lines = []

        lines.append("| Algorithm | Version | Videos | Accuracy | F1 | Precision | Recall |")
        lines.append("|-----------|---------|--------|----------|-----|-----------|--------|")

        summary = self.logger.get_summary()

        # Segmentation
        seg = summary.get("segmentation", {})
        if seg.get("n_videos", 0) > 0:
            lines.append(
                f"| Segmentation | {seg.get('algorithm_version', '-')} | "
                f"{seg.get('n_videos', 0)} | "
                f"{seg.get('mean_accuracy', 0):.1%} | - | - | - |"
            )

        # Reach Detection
        reach = summary.get("reach_detection", {})
        if reach.get("n_videos", 0) > 0:
            lines.append(
                f"| Reach Detection | {reach.get('algorithm_version', '-')} | "
                f"{reach.get('n_videos', 0)} | - | "
                f"{reach.get('mean_f1', 0):.2f} | "
                f"{reach.get('mean_precision', 0):.2f} | "
                f"{reach.get('mean_recall', 0):.2f} |"
            )

        # Outcome Classification
        outcome = summary.get("outcome_classification", {})
        if outcome.get("n_videos", 0) > 0:
            lines.append(
                f"| Outcome Classification | {outcome.get('algorithm_version', '-')} | "
                f"{outcome.get('n_videos', 0)} | "
                f"{outcome.get('mean_accuracy', 0):.1%} | - | - | - |"
            )

        return "\n".join(lines)

    def _format_algorithm_section(self, algo: str) -> str:
        """Format detailed section for one algorithm."""
        summary_key = {
            "segmentation": "segmentation",
            "reach": "reach_detection",
            "outcome": "outcome_classification"
        }.get(algo, algo)

        summary = self.logger.get_summary(algo).get(summary_key, {})

        if not summary or summary.get("n_videos", 0) == 0:
            return ""

        lines = []

        if algo == "segmentation":
            lines.append(f"**Algorithm version:** {summary.get('algorithm_version', 'unknown')}")
            lines.append(f"**Videos validated:** {summary.get('n_videos', 0)}")
            lines.append(f"**Mean boundary accuracy:** {summary.get('mean_accuracy', 0):.1%}")
            lines.append(f"**Mean timing error:** {summary.get('mean_error_frames', 0):.1f} frames")
            lines.append("")
            lines.append("**Error summary:**")
            lines.append(f"- Missed boundaries: {summary.get('total_missed', 0)}")
            lines.append(f"- Extra boundaries: {summary.get('total_extra', 0)}")

        elif algo == "reach":
            lines.append(f"**Algorithm version:** {summary.get('algorithm_version', 'unknown')}")
            lines.append(f"**Videos validated:** {summary.get('n_videos', 0)}")
            lines.append(f"**Mean F1 score:** {summary.get('mean_f1', 0):.2f} (+/- {summary.get('std_f1', 0):.2f})")
            lines.append(f"**Mean precision:** {summary.get('mean_precision', 0):.2f}")
            lines.append(f"**Mean recall:** {summary.get('mean_recall', 0):.2f}")
            lines.append("")
            lines.append("**Error summary:**")
            lines.append(f"- Missed reaches (false negatives): {summary.get('total_missed', 0)}")
            lines.append(f"- Extra reaches (false positives): {summary.get('total_extra', 0)}")
            lines.append(f"- Timing corrections: {summary.get('total_corrected', 0)}")

        elif algo == "outcome":
            lines.append(f"**Algorithm version:** {summary.get('algorithm_version', 'unknown')}")
            lines.append(f"**Videos validated:** {summary.get('n_videos', 0)}")
            lines.append(f"**Mean accuracy:** {summary.get('mean_accuracy', 0):.1%}")
            lines.append(f"**Total correct:** {summary.get('total_correct', 0)}")
            lines.append(f"**Total incorrect:** {summary.get('total_incorrect', 0)}")

            # Confusion matrix if available
            cm = summary.get('confusion_matrix', {})
            if cm:
                lines.append("")
                lines.append("**Confusion matrix:** (algo prediction vs human label)")
                # Simplified - just show the diagonal totals
                for cls in ['retrieved', 'displaced_sa', 'untouched']:
                    if cls in cm:
                        total_pred = sum(cm[cls].values())
                        correct = cm[cls].get(cls, 0)
                        if total_pred > 0:
                            lines.append(f"- {cls}: {correct}/{total_pred} correct")

        return "\n".join(lines)

    def _format_methods_section(self) -> str:
        """Generate methods section suitable for publication."""
        summary = self.logger.get_summary()

        lines = []
        lines.append("### Automated Behavioral Analysis")
        lines.append("")

        # Segmentation
        seg = summary.get("segmentation", {})
        if seg.get("n_videos", 0) > 0:
            lines.append(
                f"Trial boundaries were automatically detected using the SABL-centered crossing "
                f"algorithm (v{seg.get('algorithm_version', '?')}), which identifies when scoring area "
                f"anchor points cross the box center during pellet presentation. Algorithm performance "
                f"was validated against N={seg.get('n_videos', 0)} manually annotated videos, achieving "
                f"{seg.get('mean_accuracy', 0):.1%} accuracy with a mean timing error of "
                f"{seg.get('mean_error_frames', 0):.1f} frames."
            )
            lines.append("")

        # Reach Detection
        reach = summary.get("reach_detection", {})
        if reach.get("n_videos", 0) > 0:
            lines.append(
                f"Individual reaching attempts were identified using a rule-based detection algorithm "
                f"(v{reach.get('algorithm_version', '?')}) that tracks hand point visibility while the nose "
                f"is engaged at the slit opening. Reach boundaries (start, apex, and end frames) were "
                f"automatically extracted from DeepLabCut pose estimation data. Performance was validated "
                f"against N={reach.get('n_videos', 0)} manually annotated videos, achieving "
                f"F1={reach.get('mean_f1', 0):.2f} (precision={reach.get('mean_precision', 0):.2f}, "
                f"recall={reach.get('mean_recall', 0):.2f})."
            )
            lines.append("")

        # Outcome Classification
        outcome = summary.get("outcome_classification", {})
        if outcome.get("n_videos", 0) > 0:
            lines.append(
                f"Pellet outcomes were classified into categories (retrieved, displaced within scoring area, "
                f"displaced outside, untouched) using geometric tracking of pellet position relative to the "
                f"pillar (v{outcome.get('algorithm_version', '?')}). Classification accuracy was validated "
                f"against N={outcome.get('n_videos', 0)} manually reviewed videos, achieving "
                f"{outcome.get('mean_accuracy', 0):.1%} accuracy."
            )
            lines.append("")

        if not any([seg.get("n_videos", 0), reach.get("n_videos", 0), outcome.get("n_videos", 0)]):
            lines.append("*No validation data available yet. Validate some videos to generate methods text.*")

        return "\n".join(lines)
