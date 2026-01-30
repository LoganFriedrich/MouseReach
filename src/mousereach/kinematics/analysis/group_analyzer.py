"""
Group-level analysis - cohorts, phases, day-of-week effects.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path
from collections import defaultdict
from .data_loader import SessionData, DataLoader


class GroupAnalyzer:
    """Analyze data across groups, phases, and calendar effects."""

    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader

    def analyze_by_day_of_week(self) -> pd.DataFrame:
        """
        Analyze performance by day of week.

        Returns:
            DataFrame with day-of-week statistics
        """
        day_stats = defaultdict(list)

        for session in self.loader.sessions:
            day = session.metadata.day_of_week

            day_stats[day].append({
                'success_rate': session.success_rate,
                'retrieval_rate': session.retrieval_rate,
                'attention_score': session.mean_attention_score,
                'reach_velocity': session.mean_peak_velocity,
                'reach_extent': session.mean_reach_extent,
                'n_reaches': session.n_reaches
            })

        # Compute statistics for each day
        results = []
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for day in day_order:
            if day not in day_stats:
                continue

            sessions = day_stats[day]
            n_sessions = len(sessions)

            results.append({
                'day_of_week': day,
                'n_sessions': n_sessions,
                'mean_success_rate': np.mean([s['success_rate'] for s in sessions]),
                'std_success_rate': np.std([s['success_rate'] for s in sessions]),
                'mean_attention': np.mean([s['attention_score'] for s in sessions if s['attention_score'] > 0]),
                'mean_velocity': np.mean([s['reach_velocity'] for s in sessions if s['reach_velocity'] > 0]),
                'mean_extent': np.mean([s['reach_extent'] for s in sessions if s['reach_extent'] > 0]),
                'mean_n_reaches': np.mean([s['n_reaches'] for s in sessions])
            })

        return pd.DataFrame(results)

    def analyze_by_phase(self) -> pd.DataFrame:
        """
        Analyze performance by experimental phase (P1, P2, P3, etc.).

        Returns:
            DataFrame with phase statistics
        """
        phase_stats = {}

        for phase in sorted(self.loader.by_phase.keys()):
            sessions = self.loader.by_phase[phase]

            if not sessions:
                continue

            phase_stats[phase] = {
                'phase': phase,
                'n_sessions': len(sessions),
                'n_mice': len(set(s.metadata.mouse_id for s in sessions)),
                'mean_success_rate': np.mean([s.success_rate for s in sessions]),
                'std_success_rate': np.std([s.success_rate for s in sessions]),
                'mean_retrieval_rate': np.mean([s.retrieval_rate for s in sessions]),
                'mean_attention': np.mean([s.mean_attention_score for s in sessions if s.mean_attention_score > 0]),
                'mean_velocity': np.mean([s.mean_peak_velocity for s in sessions if s.mean_peak_velocity > 0]),
                'mean_extent': np.mean([s.mean_reach_extent for s in sessions if s.mean_reach_extent > 0]),
                'mean_duration': np.mean([s.mean_reach_duration for s in sessions if s.mean_reach_duration > 0]),
                'mean_n_reaches': np.mean([s.n_reaches for s in sessions]),
                'mean_n_causal': np.mean([s.n_causal_reaches for s in sessions])
            }

        return pd.DataFrame(list(phase_stats.values()))

    def analyze_cohort(self, mouse_ids: List[str], cohort_name: str) -> Dict:
        """
        Analyze a cohort of mice.

        Args:
            mouse_ids: List of mouse IDs in cohort
            cohort_name: Name for this cohort

        Returns:
            Dictionary of cohort statistics
        """
        sessions = self.loader.get_cohort(mouse_ids)

        if not sessions:
            return {}

        return {
            'cohort_name': cohort_name,
            'n_mice': len(mouse_ids),
            'n_sessions': len(sessions),
            'date_range': f"{sessions[0].metadata.date.strftime('%Y-%m-%d')} to {sessions[-1].metadata.date.strftime('%Y-%m-%d')}",
            'mean_success_rate': np.mean([s.success_rate for s in sessions]),
            'std_success_rate': np.std([s.success_rate for s in sessions]),
            'mean_retrieval_rate': np.mean([s.retrieval_rate for s in sessions]),
            'mean_attention': np.mean([s.mean_attention_score for s in sessions if s.mean_attention_score > 0]),
            'mean_velocity': np.mean([s.mean_peak_velocity for s in sessions if s.mean_peak_velocity > 0]),
            'mean_extent': np.mean([s.mean_reach_extent for s in sessions if s.mean_reach_extent > 0]),
            'sessions_per_mouse': len(sessions) / len(mouse_ids) if mouse_ids else 0
        }

    def compare_cohorts(self, cohort_definitions: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compare multiple cohorts.

        Args:
            cohort_definitions: Dict mapping cohort names to lists of mouse IDs
                Example: {'Control': ['CNT0101', 'CNT0102'], 'Treatment': ['CNT0201', 'CNT0202']}

        Returns:
            DataFrame comparing cohorts
        """
        results = []

        for cohort_name, mouse_ids in cohort_definitions.items():
            stats = self.analyze_cohort(mouse_ids, cohort_name)
            if stats:
                results.append(stats)

        return pd.DataFrame(results)

    def analyze_phase_progression(self, mouse_id: str) -> pd.DataFrame:
        """
        Analyze how one mouse progresses through phases.

        Args:
            mouse_id: Mouse identifier

        Returns:
            DataFrame with phase-by-phase progression
        """
        sessions = self.loader.get_mouse_sessions(mouse_id)

        if not sessions:
            return pd.DataFrame()

        # Group by phase
        phase_groups = defaultdict(list)
        for session in sessions:
            phase_groups[session.metadata.phase].append(session)

        results = []
        for phase in sorted(phase_groups.keys()):
            phase_sessions = phase_groups[phase]

            results.append({
                'mouse_id': mouse_id,
                'phase': phase,
                'n_sessions': len(phase_sessions),
                'mean_success_rate': np.mean([s.success_rate for s in phase_sessions]),
                'mean_attention': np.mean([s.mean_attention_score for s in phase_sessions if s.mean_attention_score > 0]),
                'mean_velocity': np.mean([s.mean_peak_velocity for s in phase_sessions if s.mean_peak_velocity > 0]),
                'mean_n_reaches': np.mean([s.n_reaches for s in phase_sessions])
            })

        return pd.DataFrame(results)

    def find_mice_by_pattern(self, pattern: str) -> List[str]:
        """
        Find mouse IDs matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., 'CNT01*', 'CNT02*')

        Returns:
            List of matching mouse IDs
        """
        from fnmatch import fnmatch

        all_mice = self.loader.get_mouse_ids()
        matching = [m for m in all_mice if fnmatch(m, pattern)]

        return sorted(matching)

    def export_day_of_week_analysis(self, output_path: Path):
        """Export day-of-week analysis to CSV."""
        df = self.analyze_by_day_of_week()
        df.to_csv(output_path, index=False, float_format='%.3f')
        print(f"Exported day-of-week analysis to {output_path}")

    def export_phase_analysis(self, output_path: Path):
        """Export phase analysis to CSV."""
        df = self.analyze_by_phase()
        df.to_csv(output_path, index=False, float_format='%.3f')
        print(f"Exported phase analysis to {output_path}")

    def export_cohort_comparison(self, cohort_definitions: Dict[str, List[str]], output_path: Path):
        """Export cohort comparison to CSV."""
        df = self.compare_cohorts(cohort_definitions)
        df.to_csv(output_path, index=False, float_format='%.3f')
        print(f"Exported cohort comparison to {output_path}")
