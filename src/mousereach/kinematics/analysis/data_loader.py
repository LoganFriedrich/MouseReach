"""
Data loader for multi-level analysis.

Parses video filenames, loads feature data, and organizes by hierarchy:
- Mouse → Sessions → Segments → Reaches
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict


@dataclass
class VideoMetadata:
    """Metadata extracted from video filename."""

    video_name: str
    date: datetime
    mouse_id: str
    phase: str  # P1, P2, P3, P4, etc.

    # Derived fields
    session_id: str  # mouse_id + date + phase
    day_of_week: str
    time_of_day: Optional[str] = None  # If available from other metadata

    @classmethod
    def from_filename(cls, filename: str) -> 'VideoMetadata':
        """Parse metadata from filename like '20250624_CNT0115_P2'."""
        # Pattern: YYYYMMDD_CNTxxxx_Py
        match = re.match(r'(\d{8})_(CNT\d+)_P(\d+)', filename)

        if not match:
            raise ValueError(f"Cannot parse filename: {filename}")

        date_str, mouse_id, phase_num = match.groups()
        date = datetime.strptime(date_str, '%Y%m%d')
        phase = f'P{phase_num}'

        session_id = f"{mouse_id}_{date_str}_{phase}"
        day_of_week = date.strftime('%A')

        return cls(
            video_name=filename,
            date=date,
            mouse_id=mouse_id,
            phase=phase,
            session_id=session_id,
            day_of_week=day_of_week
        )


@dataclass
class SessionData:
    """Data for a single video session."""

    metadata: VideoMetadata
    features: Dict
    outcomes: Optional[Dict] = None
    reaches: Optional[Dict] = None

    # Computed summary stats
    n_segments: int = 0
    n_reaches: int = 0
    n_causal_reaches: int = 0

    success_rate: float = 0.0  # retrieved / (retrieved + displaced + untouched)
    retrieval_rate: float = 0.0  # retrieved / total

    mean_reach_extent: float = 0.0
    mean_reach_duration: float = 0.0
    mean_peak_velocity: float = 0.0
    mean_attention_score: float = 0.0  # % of frames attending to tray (from old ASPA)

    def __post_init__(self):
        """Compute summary statistics."""
        if self.features:
            summary = self.features.get('summary', {})

            self.n_segments = self.features.get('n_segments', 0)
            self.n_reaches = summary.get('total_reaches', 0)
            self.n_causal_reaches = summary.get('causal_reaches', 0)

            # Outcome counts
            outcome_counts = summary.get('outcome_counts', {})
            retrieved = outcome_counts.get('retrieved', 0)
            displaced = outcome_counts.get('displaced_sa', 0) + outcome_counts.get('displaced_outside', 0)
            untouched = outcome_counts.get('untouched', 0)

            total_attempts = retrieved + displaced + untouched
            if total_attempts > 0:
                self.success_rate = retrieved / total_attempts
                self.retrieval_rate = retrieved / self.n_segments

            # Feature means
            self.mean_reach_extent = summary.get('mean_extent_mm', 0.0) or 0.0
            self.mean_reach_duration = summary.get('mean_duration_frames', 0.0) or 0.0
            self.mean_peak_velocity = summary.get('mean_peak_velocity', 0.0) or 0.0
            self.mean_attention_score = summary.get('mean_attention_score', 0.0) or 0.0


class DataLoader:
    """Load and organize feature data across multiple levels."""

    def __init__(self, base_dir: Path):
        """
        Initialize data loader.

        Args:
            base_dir: Directory containing *_features.json files
        """
        self.base_dir = Path(base_dir)
        self.sessions: List[SessionData] = []

        # Hierarchical organization
        self.by_mouse: Dict[str, List[SessionData]] = defaultdict(list)
        self.by_phase: Dict[str, List[SessionData]] = defaultdict(list)
        self.by_date: Dict[str, List[SessionData]] = defaultdict(list)

    def load_all(self):
        """Load all feature files in directory."""
        feature_files = sorted(self.base_dir.glob('*_features.json'))

        for fpath in feature_files:
            video_name = fpath.stem.replace('_features', '')

            try:
                metadata = VideoMetadata.from_filename(video_name)

                # Load features
                with open(fpath) as f:
                    features = json.load(f)

                # Try to load outcomes
                outcomes_path = self.base_dir / f"{video_name}_pellet_outcomes.json"
                outcomes = None
                if outcomes_path.exists():
                    with open(outcomes_path) as f:
                        outcomes = json.load(f)

                # Try to load reaches
                reaches_path = self.base_dir / f"{video_name}_reaches.json"
                reaches = None
                if reaches_path.exists():
                    with open(reaches_path) as f:
                        reaches = json.load(f)

                # Create session data
                session = SessionData(
                    metadata=metadata,
                    features=features,
                    outcomes=outcomes,
                    reaches=reaches
                )

                self.sessions.append(session)

                # Organize hierarchically
                self.by_mouse[metadata.mouse_id].append(session)
                self.by_phase[metadata.phase].append(session)
                date_key = metadata.date.strftime('%Y-%m-%d')
                self.by_date[date_key].append(session)

            except Exception as e:
                print(f"Warning: Could not load {video_name}: {e}")
                continue

        # Sort sessions within each group
        for mouse_id in self.by_mouse:
            self.by_mouse[mouse_id].sort(key=lambda s: s.metadata.date)

        for phase in self.by_phase:
            self.by_phase[phase].sort(key=lambda s: s.metadata.date)

        print(f"Loaded {len(self.sessions)} sessions")
        print(f"  {len(self.by_mouse)} unique mice")
        print(f"  {len(self.by_phase)} phases")
        print(f"  {len(self.by_date)} unique dates")

    def get_mouse_sessions(self, mouse_id: str) -> List[SessionData]:
        """Get all sessions for a specific mouse, sorted by date."""
        return self.by_mouse.get(mouse_id, [])

    def get_mouse_ids(self) -> List[str]:
        """Get list of all mouse IDs."""
        return sorted(self.by_mouse.keys())

    def get_cohort(self, mouse_ids: List[str]) -> List[SessionData]:
        """Get all sessions for a cohort of mice."""
        sessions = []
        for mouse_id in mouse_ids:
            sessions.extend(self.by_mouse.get(mouse_id, []))
        return sorted(sessions, key=lambda s: s.metadata.date)

    def get_phase_sessions(self, phase: str) -> List[SessionData]:
        """Get all sessions for a specific phase (e.g., 'P2')."""
        return self.by_phase.get(phase, [])

    def get_date_range(self, start_date: datetime, end_date: datetime) -> List[SessionData]:
        """Get all sessions within a date range."""
        return [s for s in self.sessions if start_date <= s.metadata.date <= end_date]

    def filter_sessions(self, **criteria) -> List[SessionData]:
        """
        Filter sessions by arbitrary criteria.

        Examples:
            filter_sessions(mouse_id='CNT0115')
            filter_sessions(phase='P2')
            filter_sessions(day_of_week='Monday')
        """
        filtered = self.sessions

        for key, value in criteria.items():
            if key == 'mouse_id':
                filtered = [s for s in filtered if s.metadata.mouse_id == value]
            elif key == 'phase':
                filtered = [s for s in filtered if s.metadata.phase == value]
            elif key == 'day_of_week':
                filtered = [s for s in filtered if s.metadata.day_of_week == value]
            elif key == 'date':
                filtered = [s for s in filtered if s.metadata.date == value]
            # Add more filters as needed

        return filtered

    def export_summary_table(self, output_path: Path):
        """Export summary table of all sessions to CSV."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'video_name', 'date', 'mouse_id', 'phase', 'day_of_week',
                'n_segments', 'n_reaches', 'n_causal_reaches',
                'success_rate', 'retrieval_rate',
                'mean_extent_mm', 'mean_duration_frames', 'mean_velocity',
                'mean_attention_score'
            ])

            # Data rows
            for session in self.sessions:
                m = session.metadata
                writer.writerow([
                    m.video_name,
                    m.date.strftime('%Y-%m-%d'),
                    m.mouse_id,
                    m.phase,
                    m.day_of_week,
                    session.n_segments,
                    session.n_reaches,
                    session.n_causal_reaches,
                    f"{session.success_rate:.3f}",
                    f"{session.retrieval_rate:.3f}",
                    f"{session.mean_reach_extent:.2f}",
                    f"{session.mean_reach_duration:.1f}",
                    f"{session.mean_peak_velocity:.2f}",
                    f"{session.mean_attention_score:.1f}"
                ])

        print(f"Exported summary table to {output_path}")
