"""
Temporal analysis - trends over time within and across sessions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from .data_loader import SessionData, DataLoader


@dataclass
class WithinSessionTrends:
    """Trends within a single session (early vs. late segments)."""

    video_name: str

    # Early segments (first 25%)
    early_success_rate: float
    early_attention: float
    early_reach_velocity: float

    # Late segments (last 25%)
    late_success_rate: float
    late_attention: float
    late_reach_velocity: float

    # Fatigue indicators
    performance_decline: float  # late_success - early_success (negative = worse)
    attention_decline: float
    velocity_decline: float

    # Segment-by-segment slopes (linear regression)
    success_rate_slope: Optional[float] = None
    attention_slope: Optional[float] = None
    velocity_slope: Optional[float] = None


@dataclass
class AcrossSessionTrends:
    """Trends across multiple sessions for one mouse."""

    mouse_id: str
    n_sessions: int
    date_range_days: int

    # Learning indicators
    initial_success_rate: float  # First session
    final_success_rate: float    # Last session
    learning_gain: float  # final - initial

    # Skill refinement
    initial_reach_velocity: float
    final_reach_velocity: float
    velocity_change: float

    # Consistency
    success_rate_variability: float  # Std dev across sessions
    attention_variability: float

    # Trends (linear regression across sessions)
    success_rate_slope: Optional[float] = None
    velocity_slope: Optional[float] = None
    attention_slope: Optional[float] = None


class TemporalAnalyzer:
    """Analyze temporal trends in reaching behavior."""

    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader

    def analyze_within_session(self, session: SessionData) -> WithinSessionTrends:
        """
        Analyze trends within a single session (fatigue, attention decline).

        Args:
            session: SessionData object

        Returns:
            WithinSessionTrends object
        """
        # Get segment-level data from features
        segments = session.features.get('segments', [])

        if len(segments) == 0:
            # Return empty trends
            return WithinSessionTrends(
                video_name=session.metadata.video_name,
                early_success_rate=0, early_attention=0, early_reach_velocity=0,
                late_success_rate=0, late_attention=0, late_reach_velocity=0,
                performance_decline=0, attention_decline=0, velocity_decline=0
            )

        n_segments = len(segments)

        # Split into early and late quarters
        early_cutoff = max(1, n_segments // 4)
        late_start = n_segments - early_cutoff

        early_segs = segments[:early_cutoff]
        late_segs = segments[late_start:]

        # Compute early metrics
        early_success_rate = self._compute_success_rate(early_segs)
        early_attention = self._mean_value(early_segs, 'attention_score')
        early_velocity = self._mean_reach_velocity(early_segs)

        # Compute late metrics
        late_success_rate = self._compute_success_rate(late_segs)
        late_attention = self._mean_value(late_segs, 'attention_score')
        late_velocity = self._mean_reach_velocity(late_segs)

        # Fatigue indicators (negative = decline)
        performance_decline = late_success_rate - early_success_rate
        attention_decline = late_attention - early_attention
        velocity_decline = late_velocity - early_velocity

        # Compute slopes across all segments
        success_slope = self._compute_trend_slope(segments, self._get_segment_success)
        attention_slope = self._compute_trend_slope(segments, lambda s: s.get('attention_score'))
        velocity_slope = self._compute_trend_slope(segments, self._get_segment_velocity)

        return WithinSessionTrends(
            video_name=session.metadata.video_name,
            early_success_rate=early_success_rate,
            early_attention=early_attention,
            early_reach_velocity=early_velocity,
            late_success_rate=late_success_rate,
            late_attention=late_attention,
            late_reach_velocity=late_velocity,
            performance_decline=performance_decline,
            attention_decline=attention_decline,
            velocity_decline=velocity_decline,
            success_rate_slope=success_slope,
            attention_slope=attention_slope,
            velocity_slope=velocity_slope
        )

    def analyze_across_sessions(self, mouse_id: str) -> Optional[AcrossSessionTrends]:
        """
        Analyze trends across sessions for one mouse (learning).

        Args:
            mouse_id: Mouse identifier

        Returns:
            AcrossSessionTrends object or None if insufficient data
        """
        sessions = self.loader.get_mouse_sessions(mouse_id)

        if len(sessions) < 2:
            return None

        # Sessions already sorted by date in DataLoader
        first = sessions[0]
        last = sessions[-1]

        # Date range
        date_range = (last.metadata.date - first.metadata.date).days

        # Learning indicators
        initial_success = first.success_rate
        final_success = last.success_rate
        learning_gain = final_success - initial_success

        # Skill refinement
        initial_velocity = first.mean_peak_velocity
        final_velocity = last.mean_peak_velocity
        velocity_change = final_velocity - initial_velocity

        # Variability
        success_rates = [s.success_rate for s in sessions]
        attentions = [s.mean_attention_score for s in sessions if s.mean_attention_score > 0]

        success_variability = float(np.std(success_rates)) if len(success_rates) > 1 else 0
        attention_variability = float(np.std(attentions)) if len(attentions) > 1 else 0

        # Trends (linear regression)
        success_slope = self._compute_session_slope(sessions, lambda s: s.success_rate)
        velocity_slope = self._compute_session_slope(sessions, lambda s: s.mean_peak_velocity)
        attention_slope = self._compute_session_slope(sessions, lambda s: s.mean_attention_score)

        return AcrossSessionTrends(
            mouse_id=mouse_id,
            n_sessions=len(sessions),
            date_range_days=date_range,
            initial_success_rate=initial_success,
            final_success_rate=final_success,
            learning_gain=learning_gain,
            initial_reach_velocity=initial_velocity,
            final_reach_velocity=final_velocity,
            velocity_change=velocity_change,
            success_rate_variability=success_variability,
            attention_variability=attention_variability,
            success_rate_slope=success_slope,
            velocity_slope=velocity_slope,
            attention_slope=attention_slope
        )

    def _compute_success_rate(self, segments: List[Dict]) -> float:
        """Compute success rate for a list of segments."""
        retrieved = sum(1 for s in segments if s.get('outcome') == 'retrieved')
        displaced = sum(1 for s in segments if 'displaced' in s.get('outcome', ''))
        untouched = sum(1 for s in segments if s.get('outcome') == 'untouched')

        total = retrieved + displaced + untouched
        return retrieved / total if total > 0 else 0

    def _mean_value(self, segments: List[Dict], key: str) -> float:
        """Compute mean of a field across segments."""
        values = [s.get(key, 0) for s in segments if s.get(key) is not None]
        return float(np.mean(values)) if values else 0

    def _mean_reach_velocity(self, segments: List[Dict]) -> float:
        """Compute mean reach velocity across segments."""
        velocities = []
        for seg in segments:
            for reach in seg.get('reaches', []):
                vel = reach.get('peak_velocity_px_per_frame')
                if vel is not None and vel > 0:
                    velocities.append(vel)
        return float(np.mean(velocities)) if velocities else 0

    def _get_segment_success(self, segment: Dict) -> Optional[float]:
        """Get success (1.0 or 0.0) for a segment."""
        outcome = segment.get('outcome')
        if outcome == 'retrieved':
            return 1.0
        elif outcome in ['displaced_sa', 'displaced_outside', 'untouched']:
            return 0.0
        return None

    def _get_segment_velocity(self, segment: Dict) -> Optional[float]:
        """Get mean velocity for a segment."""
        velocities = []
        for reach in segment.get('reaches', []):
            vel = reach.get('peak_velocity_px_per_frame')
            if vel is not None and vel > 0:
                velocities.append(vel)
        return float(np.mean(velocities)) if velocities else None

    def _compute_trend_slope(self, segments: List[Dict], value_func) -> Optional[float]:
        """Compute linear regression slope across segments."""
        values = []
        indices = []

        for i, seg in enumerate(segments):
            val = value_func(seg)
            if val is not None:
                values.append(val)
                indices.append(i)

        if len(values) < 2:
            return None

        # Simple linear regression
        x = np.array(indices)
        y = np.array(values)

        if len(x) < 2 or np.std(x) == 0:
            return None

        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def _compute_session_slope(self, sessions: List[SessionData], value_func) -> Optional[float]:
        """Compute linear regression slope across sessions."""
        values = [value_func(s) for s in sessions]
        values = [v for v in values if v is not None and v > 0]

        if len(values) < 2:
            return None

        x = np.arange(len(values))
        y = np.array(values)

        if len(x) < 2 or np.std(x) == 0:
            return None

        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def export_within_session_trends(self, output_path: Path):
        """Export within-session trends for all sessions to CSV."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'video_name', 'early_success_rate', 'late_success_rate', 'performance_decline',
                'early_attention', 'late_attention', 'attention_decline',
                'early_velocity', 'late_velocity', 'velocity_decline',
                'success_slope', 'attention_slope', 'velocity_slope'
            ])

            # Analyze each session
            for session in self.loader.sessions:
                trends = self.analyze_within_session(session)

                writer.writerow([
                    trends.video_name,
                    f"{trends.early_success_rate:.3f}",
                    f"{trends.late_success_rate:.3f}",
                    f"{trends.performance_decline:.3f}",
                    f"{trends.early_attention:.1f}",
                    f"{trends.late_attention:.1f}",
                    f"{trends.attention_decline:.1f}",
                    f"{trends.early_velocity:.2f}",
                    f"{trends.late_velocity:.2f}",
                    f"{trends.velocity_decline:.2f}",
                    f"{trends.success_rate_slope:.4f}" if trends.success_rate_slope else "",
                    f"{trends.attention_slope:.4f}" if trends.attention_slope else "",
                    f"{trends.velocity_slope:.4f}" if trends.velocity_slope else ""
                ])

        print(f"Exported within-session trends to {output_path}")

    def export_across_session_trends(self, output_path: Path):
        """Export across-session trends for all mice to CSV."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'mouse_id', 'n_sessions', 'date_range_days',
                'initial_success_rate', 'final_success_rate', 'learning_gain',
                'initial_velocity', 'final_velocity', 'velocity_change',
                'success_variability', 'attention_variability',
                'success_slope', 'velocity_slope', 'attention_slope'
            ])

            # Analyze each mouse
            for mouse_id in self.loader.get_mouse_ids():
                trends = self.analyze_across_sessions(mouse_id)

                if trends is None:
                    continue

                writer.writerow([
                    trends.mouse_id,
                    trends.n_sessions,
                    trends.date_range_days,
                    f"{trends.initial_success_rate:.3f}",
                    f"{trends.final_success_rate:.3f}",
                    f"{trends.learning_gain:.3f}",
                    f"{trends.initial_reach_velocity:.2f}",
                    f"{trends.final_reach_velocity:.2f}",
                    f"{trends.velocity_change:.2f}",
                    f"{trends.success_rate_variability:.3f}",
                    f"{trends.attention_variability:.1f}",
                    f"{trends.success_rate_slope:.4f}" if trends.success_rate_slope else "",
                    f"{trends.velocity_slope:.4f}" if trends.velocity_slope else "",
                    f"{trends.attention_slope:.4f}" if trends.attention_slope else ""
                ])

        print(f"Exported across-session trends to {output_path}")
