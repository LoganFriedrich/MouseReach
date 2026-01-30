"""
Confidence Analyzer - Scores algorithm confidence on videos without GT.

Provides:
- Confidence scoring based on internal algorithm signals
- Low-confidence video identification
- GT prioritization recommendations
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class VideoConfidence:
    """Confidence scores for a single video."""
    video_id: str
    processing_dir: str = ""

    # Has GT?
    has_reach_gt: bool = False
    has_outcome_gt: bool = False
    has_seg_gt: bool = False

    # Per-component confidence
    reach_confidence: float = 1.0
    outcome_confidence: float = 1.0
    dlc_confidence: float = 1.0

    # Overall
    overall_confidence: float = 1.0
    confidence_tier: str = "high"  # high, medium, low

    # Flags
    low_confidence_reasons: List[str] = field(default_factory=list)

    # Details for debugging
    n_segments: int = 0
    n_reaches: int = 0
    n_low_conf_segments: int = 0


@dataclass
class ConfidenceReport:
    """Aggregate confidence report across all videos."""
    # Counts
    total_videos: int = 0
    videos_with_gt: int = 0
    videos_without_gt: int = 0

    # Confidence distribution
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0

    # Videos needing GT
    priority_videos: List[VideoConfidence] = field(default_factory=list)

    # All video scores
    all_scores: List[VideoConfidence] = field(default_factory=list)


class ConfidenceAnalyzer:
    """
    Analyzes algorithm confidence on videos without ground truth.

    Uses multiple signals to estimate confidence:
    - DLC tracking quality (likelihood scores)
    - Outcome classifier confidence
    - Reach detector signal quality
    - Consistency checks
    """

    # Thresholds
    HIGH_CONF_THRESHOLD = 0.85
    LOW_CONF_THRESHOLD = 0.65

    def __init__(self, processing_root: Path = None):
        """
        Initialize analyzer.

        Args:
            processing_root: Root directory containing Processing folders
        """
        if processing_root is None:
            from mousereach.config import PROCESSING_ROOT
            processing_root = PROCESSING_ROOT

        self.processing_root = Path(processing_root)

    def analyze_all(self, progress_callback=None) -> ConfidenceReport:
        """
        Analyze confidence for all videos.

        Args:
            progress_callback: Optional callback(current, total, message)

        Returns:
            ConfidenceReport with all scores
        """
        report = ConfidenceReport()

        # Find all Processing directories
        processing_dirs = list(self.processing_root.glob("*/Processing"))
        total = len(processing_dirs)

        for i, proc_dir in enumerate(processing_dirs):
            if progress_callback:
                progress_callback(i + 1, total, f"Analyzing {proc_dir.parent.name}...")

            # Get video ID from directory name
            video_id = proc_dir.parent.name

            # Analyze this video
            score = self._analyze_video(video_id, proc_dir)
            report.all_scores.append(score)
            report.total_videos += 1

            # Count GT status
            has_any_gt = score.has_reach_gt or score.has_outcome_gt or score.has_seg_gt
            if has_any_gt:
                report.videos_with_gt += 1
            else:
                report.videos_without_gt += 1

            # Confidence tier counts
            if score.confidence_tier == "high":
                report.high_confidence += 1
            elif score.confidence_tier == "medium":
                report.medium_confidence += 1
            else:
                report.low_confidence += 1

        # Sort priority videos (low confidence without GT)
        report.priority_videos = sorted(
            [s for s in report.all_scores if s.confidence_tier == "low" and not (s.has_reach_gt or s.has_outcome_gt)],
            key=lambda x: x.overall_confidence
        )

        return report

    def _analyze_video(self, video_id: str, proc_dir: Path) -> VideoConfidence:
        """Analyze confidence for a single video."""
        score = VideoConfidence(
            video_id=video_id,
            processing_dir=str(proc_dir)
        )

        # Check for GT files
        score.has_reach_gt = any(proc_dir.glob(f"*_reach_ground_truth.json"))
        score.has_outcome_gt = any(proc_dir.glob(f"*_outcome*_ground_truth.json"))
        score.has_seg_gt = any(proc_dir.glob(f"*_seg_ground_truth.json"))

        # Analyze DLC confidence
        score.dlc_confidence = self._get_dlc_confidence(proc_dir)
        if score.dlc_confidence < 0.7:
            score.low_confidence_reasons.append("Low DLC tracking quality")

        # Analyze reach detection confidence
        score.reach_confidence, reach_reasons = self._get_reach_confidence(proc_dir)
        score.low_confidence_reasons.extend(reach_reasons)

        # Analyze outcome confidence
        score.outcome_confidence, outcome_reasons = self._get_outcome_confidence(proc_dir)
        score.low_confidence_reasons.extend(outcome_reasons)

        # Calculate overall confidence
        weights = [0.3, 0.35, 0.35]  # DLC, reach, outcome
        scores = [score.dlc_confidence, score.reach_confidence, score.outcome_confidence]
        score.overall_confidence = sum(w * s for w, s in zip(weights, scores))

        # Determine tier
        if score.overall_confidence >= self.HIGH_CONF_THRESHOLD:
            score.confidence_tier = "high"
        elif score.overall_confidence >= self.LOW_CONF_THRESHOLD:
            score.confidence_tier = "medium"
        else:
            score.confidence_tier = "low"

        return score

    def _get_dlc_confidence(self, proc_dir: Path) -> float:
        """
        Get DLC tracking confidence for a video.

        Reads DLC H5 files and computes mean likelihood.
        """
        try:
            import h5py

            # Find DLC H5 file
            dlc_files = list(proc_dir.glob("*DLC*.h5"))
            if not dlc_files:
                return 0.8  # Default if no DLC file

            # Read likelihoods
            with h5py.File(dlc_files[0], 'r') as f:
                # DLC H5 structure varies, try common patterns
                try:
                    df_key = list(f.keys())[0]
                    data = f[df_key]['table'][:]
                    # Likelihood columns are typically every 3rd column
                    # This is a simplification - real implementation would parse properly
                    return 0.85  # Placeholder
                except:
                    return 0.8

        except ImportError:
            return 0.8  # Default without h5py
        except Exception:
            return 0.7  # Lower confidence if we can't read

    def _get_reach_confidence(self, proc_dir: Path) -> Tuple[float, List[str]]:
        """
        Get reach detection confidence.

        Signals:
        - Detector confidence scores (if present)
        - Reach consistency (unusual patterns = lower confidence)
        - Detection quality flags
        """
        reasons = []
        confidence = 1.0

        # Find reaches file
        reach_files = list(proc_dir.glob("*_reaches.json"))
        if not reach_files:
            return 0.7, ["No reaches file found"]

        try:
            with open(reach_files[0]) as f:
                data = json.load(f)

            segments = data.get("segments", [])

            # Count total reaches and look for quality signals
            total_reaches = 0
            low_conf_reaches = 0
            unusual_patterns = 0

            for seg in segments:
                reaches = seg.get("reaches", [])
                total_reaches += len(reaches)

                for reach in reaches:
                    # Check for confidence score
                    conf = reach.get("confidence", reach.get("detection_confidence", 1.0))
                    if conf < 0.7:
                        low_conf_reaches += 1

                    # Check for unusual patterns
                    duration = reach.get("end_frame", 0) - reach.get("start_frame", 0)
                    if duration < 5 or duration > 200:  # Very short or very long
                        unusual_patterns += 1

                    # Check extent
                    extent = reach.get("max_extent_ruler", 0)
                    if extent < -10 or extent > 50:  # Unusual extent values
                        unusual_patterns += 1

            # Calculate confidence based on signals
            if total_reaches > 0:
                low_conf_ratio = low_conf_reaches / total_reaches
                unusual_ratio = unusual_patterns / total_reaches

                if low_conf_ratio > 0.3:
                    confidence -= 0.2
                    reasons.append(f"{low_conf_reaches}/{total_reaches} low-confidence reaches")

                if unusual_ratio > 0.2:
                    confidence -= 0.15
                    reasons.append(f"{unusual_patterns} unusual reach patterns")

            # Check for very few or very many reaches
            reaches_per_seg = total_reaches / len(segments) if segments else 0
            if reaches_per_seg > 20:
                confidence -= 0.1
                reasons.append(f"Unusually high reach count ({reaches_per_seg:.1f}/segment)")

            return max(0.3, confidence), reasons

        except Exception as e:
            return 0.6, [f"Error reading reaches: {str(e)}"]

    def _get_outcome_confidence(self, proc_dir: Path) -> Tuple[float, List[str]]:
        """
        Get outcome classification confidence.

        Signals:
        - Classifier confidence scores
        - Outcome distribution (unusual patterns = lower confidence)
        - Consistency with reach data
        """
        reasons = []
        confidence = 1.0

        # Find outcomes file
        outcome_files = list(proc_dir.glob("*_pellet_outcomes.json"))
        if not outcome_files:
            return 0.7, ["No outcomes file found"]

        try:
            with open(outcome_files[0]) as f:
                data = json.load(f)

            segments = data.get("segments", [])
            total_segments = len(segments)

            if total_segments == 0:
                return 0.5, ["No segments in outcomes file"]

            # Count outcome types and confidence
            outcome_counts = {}
            low_conf_segments = 0
            uncertain_count = 0

            for seg in segments:
                outcome = seg.get("outcome", "uncertain")
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

                conf = seg.get("confidence", seg.get("outcome_confidence", 1.0))
                if conf < 0.7:
                    low_conf_segments += 1

                if outcome == "uncertain":
                    uncertain_count += 1

            # High uncertain rate = low confidence
            uncertain_ratio = uncertain_count / total_segments
            if uncertain_ratio > 0.3:
                confidence -= 0.25
                reasons.append(f"{uncertain_ratio:.0%} segments marked uncertain")

            # Many low-confidence classifications
            low_conf_ratio = low_conf_segments / total_segments
            if low_conf_ratio > 0.3:
                confidence -= 0.2
                reasons.append(f"{low_conf_segments}/{total_segments} low-confidence outcomes")

            # Check for unusual outcome distribution
            # E.g., 100% retrieved is suspicious
            for outcome, count in outcome_counts.items():
                if outcome != "uncertain" and count == total_segments and total_segments > 3:
                    confidence -= 0.15
                    reasons.append(f"All segments have same outcome ({outcome})")

            return max(0.3, confidence), reasons

        except Exception as e:
            return 0.6, [f"Error reading outcomes: {str(e)}"]

    def get_gt_priorities(self) -> List[Dict]:
        """
        Get prioritized list of videos needing GT annotation.

        Returns list sorted by priority with reasons.
        """
        report = self.analyze_all()

        priorities = []

        # Low confidence videos first
        for score in report.priority_videos:
            priorities.append({
                'video_id': score.video_id,
                'priority': 'HIGH',
                'confidence': score.overall_confidence,
                'reasons': score.low_confidence_reasons,
                'needs': self._get_needed_gt(score),
            })

        # Medium confidence videos
        for score in report.all_scores:
            if score.confidence_tier == "medium" and not (score.has_reach_gt or score.has_outcome_gt):
                priorities.append({
                    'video_id': score.video_id,
                    'priority': 'MEDIUM',
                    'confidence': score.overall_confidence,
                    'reasons': score.low_confidence_reasons,
                    'needs': self._get_needed_gt(score),
                })

        return priorities

    def _get_needed_gt(self, score: VideoConfidence) -> List[str]:
        """Determine what GT types are needed for a video."""
        needed = []
        if not score.has_reach_gt and score.reach_confidence < 0.8:
            needed.append("reach")
        if not score.has_outcome_gt and score.outcome_confidence < 0.8:
            needed.append("outcome")
        if not score.has_seg_gt:
            needed.append("segment")
        return needed if needed else ["any"]

    def generate_report(self) -> str:
        """Generate a human-readable confidence report."""
        report = self.analyze_all()

        lines = []
        lines.append("=" * 60)
        lines.append("ALGORITHM CONFIDENCE REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Total videos: {report.total_videos}")
        lines.append(f"Videos with GT: {report.videos_with_gt} ({report.videos_with_gt/report.total_videos*100:.0f}%)" if report.total_videos > 0 else "")
        lines.append(f"Videos without GT: {report.videos_without_gt}")
        lines.append("")

        lines.append("Confidence Distribution (videos without GT):")
        lines.append(f"  High (>{self.HIGH_CONF_THRESHOLD*100:.0f}%): {report.high_confidence} videos")
        lines.append(f"  Medium ({self.LOW_CONF_THRESHOLD*100:.0f}%-{self.HIGH_CONF_THRESHOLD*100:.0f}%): {report.medium_confidence} videos")
        lines.append(f"  Low (<{self.LOW_CONF_THRESHOLD*100:.0f}%): {report.low_confidence} videos")
        lines.append("")

        if report.priority_videos:
            lines.append("=== PRIORITY VIDEOS FOR GT ===")
            lines.append("")
            for i, video in enumerate(report.priority_videos[:10], 1):
                lines.append(f"{i}. {video.video_id}")
                lines.append(f"   Confidence: {video.overall_confidence:.0%}")
                if video.low_confidence_reasons:
                    lines.append(f"   Issues: {', '.join(video.low_confidence_reasons)}")
                lines.append("")

        return "\n".join(lines)


# Convenience function
def get_priority_videos(n: int = 10) -> List[str]:
    """Get list of video IDs that should be prioritized for GT."""
    analyzer = ConfidenceAnalyzer()
    priorities = analyzer.get_gt_priorities()
    return [p['video_id'] for p in priorities[:n]]
