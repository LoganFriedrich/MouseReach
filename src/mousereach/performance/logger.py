"""
Performance Logger - Automatically logs algorithm vs human comparison metrics.

Logs are stored as JSON files in PROCESSING_ROOT/performance_logs/
Each algorithm has its own log file with append-only entries.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import asdict
import numpy as np

from .metrics import (
    compute_segmentation_metrics,
    compute_reach_metrics,
    compute_outcome_metrics,
    SegmentationMetrics,
    ReachMetrics,
    OutcomeMetrics,
)


class PerformanceLogger:
    """
    Logs algorithm performance metrics during validation.

    Automatically called by review widgets when saving validated results.
    Can also be used manually for batch evaluation.
    """

    LOG_VERSION = "1.0.0"

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files. Defaults to PROCESSING_ROOT/performance_logs/
        """
        if log_dir is None:
            from mousereach.config import PROCESSING_ROOT
            log_dir = PROCESSING_ROOT / "performance_logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log file paths
        self.seg_log = self.log_dir / "segmentation_performance.json"
        self.reach_log = self.log_dir / "reach_detection_performance.json"
        self.outcome_log = self.log_dir / "outcome_performance.json"

    def _load_log(self, log_path: Path) -> Dict:
        """Load existing log or create new one."""
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # New log structure
        return {
            "log_version": self.LOG_VERSION,
            "algorithm": log_path.stem.replace("_performance", ""),
            "entries": [],
            "aggregate": {}
        }

    def _save_log(self, log_path: Path, data: Dict):
        """Save log file."""
        with open(log_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _get_username(self) -> str:
        """Get current username."""
        try:
            return os.getlogin()
        except Exception:
            return os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))

    def log_segmentation(
        self,
        video_id: str,
        algo_boundaries: List[int],
        human_boundaries: List[int],
        algo_version: str = None,
        validator: str = None,
        source: str = "validation"
    ) -> SegmentationMetrics:
        """
        Log segmentation boundary comparison.

        Args:
            video_id: Video identifier
            algo_boundaries: Algorithm-detected boundaries (original)
            human_boundaries: Human-corrected boundaries (final)
            algo_version: Algorithm version string
            validator: Username of validator
            source: "validation" or "batch_eval"

        Returns:
            Computed metrics
        """
        metrics = compute_segmentation_metrics(algo_boundaries, human_boundaries)

        entry = {
            "video_id": video_id,
            "logged_at": datetime.now().isoformat(),
            "log_source": source,
            "validator": validator or self._get_username(),
            "algorithm_version": algo_version,
            "algo_count": metrics.n_algo_boundaries,
            "human_count": metrics.n_human_boundaries,
            "metrics": {
                "accuracy": metrics.accuracy,
                "n_matched": metrics.n_matched,
                "n_missed": metrics.n_missed,
                "n_extra": metrics.n_extra,
                "mean_error_frames": metrics.mean_error_frames,
                "max_error_frames": metrics.max_error_frames,
                "std_error_frames": metrics.std_error_frames,
            },
            "boundary_errors": metrics.boundary_errors[:20]  # Keep top 20
        }

        # Append to log
        log_data = self._load_log(self.seg_log)
        log_data["entries"].append(entry)
        log_data["algorithm_version"] = algo_version
        log_data["aggregate"] = self._compute_seg_aggregate(log_data["entries"])
        self._save_log(self.seg_log, log_data)

        return metrics

    def log_reach_detection(
        self,
        video_id: str,
        algo_data: Dict,
        human_data: Dict,
        algo_version: str = None,
        validator: str = None,
        source: str = "validation"
    ) -> ReachMetrics:
        """
        Log reach detection comparison.

        Args:
            video_id: Video identifier
            algo_data: Original algorithm output (before human edits)
            human_data: Human-corrected output (after validation)
            algo_version: Algorithm version string
            validator: Username of validator
            source: "validation" or "batch_eval"

        Returns:
            Computed metrics
        """
        metrics = compute_reach_metrics(algo_data, human_data)

        if algo_version is None:
            algo_version = algo_data.get('detector_version', human_data.get('detector_version'))

        entry = {
            "video_id": video_id,
            "logged_at": datetime.now().isoformat(),
            "log_source": source,
            "validator": validator or self._get_username(),
            "algorithm_version": algo_version,
            "algo_count": metrics.n_algo_reaches,
            "human_count": metrics.n_human_reaches,
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "n_matched": metrics.n_matched,
                "n_missed": metrics.n_missed,
                "n_extra": metrics.n_extra,
                "n_corrected": metrics.n_corrected,
                "mean_start_error": metrics.mean_start_error,
                "mean_end_error": metrics.mean_end_error,
            },
            "segment_metrics": metrics.segment_metrics
        }

        # Append to log
        log_data = self._load_log(self.reach_log)
        log_data["entries"].append(entry)
        log_data["algorithm_version"] = algo_version
        log_data["aggregate"] = self._compute_reach_aggregate(log_data["entries"])
        self._save_log(self.reach_log, log_data)

        return metrics

    def log_outcome(
        self,
        video_id: str,
        algo_data: Dict,
        human_data: Dict,
        algo_version: str = None,
        validator: str = None,
        source: str = "validation"
    ) -> OutcomeMetrics:
        """
        Log outcome classification comparison.

        Args:
            video_id: Video identifier
            algo_data: Original algorithm output (before human edits)
            human_data: Human-corrected output (after validation)
            algo_version: Algorithm version string
            validator: Username of validator
            source: "validation" or "batch_eval"

        Returns:
            Computed metrics
        """
        metrics = compute_outcome_metrics(algo_data, human_data)

        if algo_version is None:
            algo_version = algo_data.get('detector_version', human_data.get('detector_version'))

        entry = {
            "video_id": video_id,
            "logged_at": datetime.now().isoformat(),
            "log_source": source,
            "validator": validator or self._get_username(),
            "algorithm_version": algo_version,
            "n_segments": metrics.n_segments,
            "metrics": {
                "accuracy": metrics.accuracy,
                "n_correct": metrics.n_correct,
                "n_incorrect": metrics.n_incorrect,
                "per_class_precision": metrics.per_class_precision,
                "per_class_recall": metrics.per_class_recall,
                "per_class_f1": metrics.per_class_f1,
            },
            "confusion_matrix": metrics.confusion_matrix,
            "misclassifications": metrics.misclassifications[:20]  # Keep top 20
        }

        # Append to log
        log_data = self._load_log(self.outcome_log)
        log_data["entries"].append(entry)
        log_data["algorithm_version"] = algo_version
        log_data["aggregate"] = self._compute_outcome_aggregate(log_data["entries"])
        self._save_log(self.outcome_log, log_data)

        return metrics

    def _compute_seg_aggregate(self, entries: List[Dict]) -> Dict:
        """Compute aggregate statistics for segmentation."""
        if not entries:
            return {}

        accuracies = [e["metrics"]["accuracy"] for e in entries]
        mean_errors = [e["metrics"]["mean_error_frames"] for e in entries if e["metrics"]["mean_error_frames"] > 0]

        return {
            "n_videos": len(entries),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_error_frames": float(np.mean(mean_errors)) if mean_errors else 0.0,
            "total_missed": sum(e["metrics"]["n_missed"] for e in entries),
            "total_extra": sum(e["metrics"]["n_extra"] for e in entries),
            "last_updated": datetime.now().isoformat()
        }

    def _compute_reach_aggregate(self, entries: List[Dict]) -> Dict:
        """Compute aggregate statistics for reach detection."""
        if not entries:
            return {}

        precisions = [e["metrics"]["precision"] for e in entries]
        recalls = [e["metrics"]["recall"] for e in entries]
        f1s = [e["metrics"]["f1"] for e in entries]

        return {
            "n_videos": len(entries),
            "mean_precision": float(np.mean(precisions)),
            "mean_recall": float(np.mean(recalls)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "total_missed": sum(e["metrics"]["n_missed"] for e in entries),
            "total_extra": sum(e["metrics"]["n_extra"] for e in entries),
            "total_corrected": sum(e["metrics"]["n_corrected"] for e in entries),
            "last_updated": datetime.now().isoformat()
        }

    def _compute_outcome_aggregate(self, entries: List[Dict]) -> Dict:
        """Compute aggregate statistics for outcome classification."""
        if not entries:
            return {}

        accuracies = [e["metrics"]["accuracy"] for e in entries]

        # Aggregate confusion matrix
        CLASSES = ['retrieved', 'displaced_sa', 'displaced_outside', 'untouched', 'uncertain', 'no_pellet']
        total_cm = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
        for e in entries:
            cm = e.get("confusion_matrix", {})
            for c1 in cm:
                for c2 in cm[c1]:
                    if c1 in total_cm and c2 in total_cm[c1]:
                        total_cm[c1][c2] += cm[c1][c2]

        return {
            "n_videos": len(entries),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "total_correct": sum(e["metrics"]["n_correct"] for e in entries),
            "total_incorrect": sum(e["metrics"]["n_incorrect"] for e in entries),
            "confusion_matrix": total_cm,
            "last_updated": datetime.now().isoformat()
        }

    def get_summary(self, algorithm: str = None) -> Dict:
        """
        Get performance summary for one or all algorithms.

        Args:
            algorithm: "segmentation", "reach", "outcome", or None for all

        Returns:
            Summary dict with aggregate statistics
        """
        summary = {}

        if algorithm is None or algorithm == "segmentation":
            log_data = self._load_log(self.seg_log)
            summary["segmentation"] = {
                "algorithm_version": log_data.get("algorithm_version"),
                **log_data.get("aggregate", {})
            }

        if algorithm is None or algorithm == "reach":
            log_data = self._load_log(self.reach_log)
            summary["reach_detection"] = {
                "algorithm_version": log_data.get("algorithm_version"),
                **log_data.get("aggregate", {})
            }

        if algorithm is None or algorithm == "outcome":
            log_data = self._load_log(self.outcome_log)
            summary["outcome_classification"] = {
                "algorithm_version": log_data.get("algorithm_version"),
                **log_data.get("aggregate", {})
            }

        return summary

    def get_entries(self, algorithm: str, since: str = None) -> List[Dict]:
        """
        Get log entries for an algorithm.

        Args:
            algorithm: "segmentation", "reach", or "outcome"
            since: ISO date string to filter entries after

        Returns:
            List of log entries
        """
        log_map = {
            "segmentation": self.seg_log,
            "reach": self.reach_log,
            "outcome": self.outcome_log,
        }

        log_path = log_map.get(algorithm)
        if not log_path:
            return []

        log_data = self._load_log(log_path)
        entries = log_data.get("entries", [])

        if since:
            entries = [e for e in entries if e.get("logged_at", "") >= since]

        return entries
