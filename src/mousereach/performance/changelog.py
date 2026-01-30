"""
Algorithm Changelog - Tracks what changed and whether it helped.

Provides:
- Change logging with before/after metrics
- Version comparison
- Historical trend data for graphs
- Impact assessment
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class MetricDelta:
    """Before/after comparison for a metric."""
    metric_name: str
    before: float
    after: float
    delta: float = 0.0
    delta_percent: float = 0.0
    improved: bool = False

    def __post_init__(self):
        self.delta = self.after - self.before
        if self.before != 0:
            self.delta_percent = (self.delta / self.before) * 100
        self.improved = self.delta > 0  # Assumes higher is better

    def format_delta(self) -> str:
        """Format delta as human-readable string."""
        sign = "+" if self.delta >= 0 else ""
        if abs(self.delta) < 0.01:
            return f"{sign}{self.delta_percent:.1f}%"
        return f"{sign}{self.delta:.3f} ({sign}{self.delta_percent:.1f}%)"


@dataclass
class ChangeEntry:
    """A single change log entry."""
    # Identity
    date: str
    version: str
    change_id: str = ""

    # What changed
    change_type: str = ""  # "algorithm", "threshold", "bugfix", "feature"
    component: str = ""  # "reach_detector", "outcome_classifier", etc.
    change_summary: str = ""  # Short description
    change_detail: str = ""  # Full explanation

    # Why
    reason: str = ""
    based_on: str = ""  # "GT analysis showed...", "User feedback", etc.

    # Impact (metrics before/after)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    deltas: Dict[str, Dict] = field(default_factory=dict)

    # Summary
    overall_impact: str = ""  # "positive", "negative", "mixed", "neutral"
    gt_basis: str = ""  # "2 reach GT files, 730 human-verified reaches"

    # Notes
    notes: str = ""
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.change_id:
            self.change_id = f"{self.version}_{self.date.replace('-', '')}_{id(self) % 10000:04d}"

    def compute_deltas(self):
        """Compute metric deltas from before/after values."""
        self.deltas = {}
        for metric in set(self.metrics_before.keys()) | set(self.metrics_after.keys()):
            before = self.metrics_before.get(metric, 0)
            after = self.metrics_after.get(metric, 0)
            delta = MetricDelta(metric_name=metric, before=before, after=after)
            self.deltas[metric] = {
                'before': before,
                'after': after,
                'delta': delta.delta,
                'delta_percent': delta.delta_percent,
                'improved': delta.improved,
            }

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'date': self.date,
            'version': self.version,
            'change_id': self.change_id,
            'change_type': self.change_type,
            'component': self.component,
            'change_summary': self.change_summary,
            'change_detail': self.change_detail,
            'reason': self.reason,
            'based_on': self.based_on,
            'metrics_before': self.metrics_before,
            'metrics_after': self.metrics_after,
            'deltas': self.deltas,
            'overall_impact': self.overall_impact,
            'gt_basis': self.gt_basis,
            'notes': self.notes,
            'warnings': self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChangeEntry":
        """Create from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ChangeLog:
    """
    Manages algorithm change history.

    Tracks:
    - What changed (code, thresholds, etc.)
    - Why it changed (GT analysis, user feedback)
    - Impact (before/after metrics)
    """

    VERSION = "1.0.0"

    def __init__(self, log_dir: Path = None):
        """
        Initialize changelog.

        Args:
            log_dir: Directory for changelog file
        """
        if log_dir is None:
            from mousereach.config import PROCESSING_ROOT
            log_dir = PROCESSING_ROOT / "performance_logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.changelog_path = self.log_dir / "changelog.json"

        self._entries: List[ChangeEntry] = []
        self._load()

    def _load(self):
        """Load existing changelog."""
        if self.changelog_path.exists():
            try:
                with open(self.changelog_path) as f:
                    data = json.load(f)
                self._entries = [
                    ChangeEntry.from_dict(e) for e in data.get('changes', [])
                ]
            except Exception:
                self._entries = []

    def _save(self):
        """Save changelog to disk."""
        data = {
            'version': self.VERSION,
            'last_updated': datetime.now().isoformat(),
            'changes': [e.to_dict() for e in self._entries]
        }
        with open(self.changelog_path, 'w') as f:
            json.dump(data, f, indent=2)

    def log_change(
        self,
        version: str,
        change_summary: str,
        change_type: str = "algorithm",
        component: str = "",
        reason: str = "",
        metrics_before: Dict[str, float] = None,
        metrics_after: Dict[str, float] = None,
        gt_basis: str = "",
        notes: str = "",
        change_detail: str = ""
    ) -> ChangeEntry:
        """
        Log a new change.

        Args:
            version: Algorithm version (e.g., "v3.4.0")
            change_summary: Short description of change
            change_type: Type of change (algorithm, threshold, bugfix, feature)
            component: Component affected (reach_detector, outcome_classifier, etc.)
            reason: Why the change was made
            metrics_before: Metrics before the change
            metrics_after: Metrics after the change
            gt_basis: GT data used for evaluation
            notes: Additional notes
            change_detail: Detailed explanation

        Returns:
            The created ChangeEntry
        """
        entry = ChangeEntry(
            date=datetime.now().strftime("%Y-%m-%d"),
            version=version,
            change_type=change_type,
            component=component,
            change_summary=change_summary,
            change_detail=change_detail,
            reason=reason,
            metrics_before=metrics_before or {},
            metrics_after=metrics_after or {},
            gt_basis=gt_basis,
            notes=notes,
        )

        # Compute deltas
        entry.compute_deltas()

        # Determine overall impact
        entry.overall_impact = self._assess_impact(entry.deltas)

        # Add warnings if needed
        if entry.overall_impact == "negative":
            entry.warnings.append("This change had negative impact on metrics")

        self._entries.append(entry)
        self._save()

        return entry

    def _assess_impact(self, deltas: Dict[str, Dict]) -> str:
        """Assess overall impact from metric deltas."""
        if not deltas:
            return "neutral"

        # Priority metrics (higher weight)
        priority_metrics = ['recall', 'f1', 'accuracy']

        positive_count = 0
        negative_count = 0
        significant_positive = False
        significant_negative = False

        for metric, delta_info in deltas.items():
            delta = delta_info.get('delta', 0)
            delta_pct = abs(delta_info.get('delta_percent', 0))

            if delta > 0:
                positive_count += 1
                if delta_pct > 5 and metric in priority_metrics:
                    significant_positive = True
            elif delta < 0:
                negative_count += 1
                if delta_pct > 5 and metric in priority_metrics:
                    significant_negative = True

        if significant_positive and not significant_negative:
            return "positive"
        elif significant_negative and not significant_positive:
            return "negative"
        elif positive_count > negative_count * 2:
            return "positive"
        elif negative_count > positive_count * 2:
            return "negative"
        elif positive_count > 0 and negative_count > 0:
            return "mixed"
        else:
            return "neutral"

    def get_entries(
        self,
        since: str = None,
        component: str = None,
        change_type: str = None,
        limit: int = None
    ) -> List[ChangeEntry]:
        """
        Get changelog entries with optional filters.

        Args:
            since: Filter entries after this date (YYYY-MM-DD)
            component: Filter by component
            change_type: Filter by change type
            limit: Maximum entries to return

        Returns:
            List of matching entries (newest first)
        """
        entries = self._entries.copy()

        if since:
            entries = [e for e in entries if e.date >= since]

        if component:
            entries = [e for e in entries if e.component == component]

        if change_type:
            entries = [e for e in entries if e.change_type == change_type]

        # Sort by date descending
        entries.sort(key=lambda e: e.date, reverse=True)

        if limit:
            entries = entries[:limit]

        return entries

    def get_version_comparison(self, version_a: str, version_b: str) -> Dict:
        """
        Compare metrics between two versions.

        Args:
            version_a: Earlier version
            version_b: Later version

        Returns:
            Dict with metric comparisons
        """
        entries_a = [e for e in self._entries if e.version == version_a]
        entries_b = [e for e in self._entries if e.version == version_b]

        if not entries_a or not entries_b:
            return {}

        # Use most recent entry for each version
        entry_a = max(entries_a, key=lambda e: e.date)
        entry_b = max(entries_b, key=lambda e: e.date)

        # Compare metrics_after values
        comparison = {}
        all_metrics = set(entry_a.metrics_after.keys()) | set(entry_b.metrics_after.keys())

        for metric in all_metrics:
            val_a = entry_a.metrics_after.get(metric, 0)
            val_b = entry_b.metrics_after.get(metric, 0)
            delta = MetricDelta(metric_name=metric, before=val_a, after=val_b)
            comparison[metric] = {
                version_a: val_a,
                version_b: val_b,
                'delta': delta.delta,
                'delta_percent': delta.delta_percent,
                'improved': delta.improved,
            }

        return comparison

    def get_trend_data(
        self,
        metric: str,
        n_points: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get historical trend data for a metric (for graphing).

        Args:
            metric: Metric name (recall, precision, f1, etc.)
            n_points: Maximum data points

        Returns:
            List of {date, version, value} dicts
        """
        points = []

        for entry in self._entries:
            value = entry.metrics_after.get(metric)
            if value is not None:
                points.append({
                    'date': entry.date,
                    'version': entry.version,
                    'value': value,
                    'change_summary': entry.change_summary,
                })

        # Sort by date
        points.sort(key=lambda p: p['date'])

        return points[-n_points:]

    def get_all_versions(self) -> List[str]:
        """Get list of all logged versions."""
        versions = sorted(set(e.version for e in self._entries))
        return versions

    def generate_report(self, n_entries: int = 10) -> str:
        """
        Generate human-readable changelog report.

        Args:
            n_entries: Number of recent entries to include
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ALGORITHM CHANGELOG")
        lines.append("=" * 60)
        lines.append("")

        entries = self.get_entries(limit=n_entries)

        if not entries:
            lines.append("No changes logged yet.")
            return "\n".join(lines)

        for entry in entries:
            # Header
            impact_icon = {
                'positive': '+',
                'negative': '-',
                'mixed': '~',
                'neutral': '='
            }.get(entry.overall_impact, '?')

            lines.append(f"[{entry.date}] {entry.version} - {entry.change_summary} [{impact_icon}]")

            # Details
            if entry.change_detail:
                lines.append(f"  Changed: {entry.change_detail}")

            if entry.reason:
                lines.append(f"  Reason: {entry.reason}")

            # Metrics
            if entry.deltas:
                lines.append("  Impact:")
                for metric, delta_info in entry.deltas.items():
                    before = delta_info['before']
                    after = delta_info['after']
                    delta_pct = delta_info['delta_percent']
                    sign = "+" if delta_pct >= 0 else ""

                    # Format based on metric type
                    if metric in ['recall', 'precision', 'accuracy']:
                        lines.append(f"    {metric}: {before:.1%} -> {after:.1%} ({sign}{delta_pct:.1f}%)")
                    else:
                        lines.append(f"    {metric}: {before:.3f} -> {after:.3f} ({sign}{delta_pct:.1f}%)")

            if entry.gt_basis:
                lines.append(f"  Based on: {entry.gt_basis}")

            if entry.notes:
                lines.append(f"  Note: {entry.notes}")

            if entry.warnings:
                for warning in entry.warnings:
                    lines.append(f"  WARNING: {warning}")

            lines.append("")

        return "\n".join(lines)


# Pre-populate with known historical changes
def initialize_known_changes(changelog: ChangeLog):
    """Add known historical changes to the changelog."""

    # v3.4.0 - The big recall fix
    if not any(e.version == "v3.4.0" for e in changelog._entries):
        changelog.log_change(
            version="v3.4.0",
            change_summary="Removed negative extent filter",
            change_type="bugfix",
            component="reach_detector",
            reason="GT analysis showed filter was dropping valid reaches",
            change_detail=(
                "Removed the filter at line 518 that rejected reaches with "
                "max_extent_pixels < 0. Analysis showed human-marked reaches "
                "include attempts where hand doesn't fully cross slit."
            ),
            metrics_before={
                'recall': 0.155,
                'precision': 0.962,
                'f1': 0.27,
            },
            metrics_after={
                'recall': 0.990,
                'precision': 0.955,
                'f1': 0.97,
            },
            gt_basis="2 reach GT files, 730 human-verified reaches",
            notes="Major improvement. Watch for false positive increase.",
        )


# Convenience function
def get_changelog() -> ChangeLog:
    """Get the singleton changelog instance."""
    changelog = ChangeLog()
    initialize_known_changes(changelog)
    return changelog
