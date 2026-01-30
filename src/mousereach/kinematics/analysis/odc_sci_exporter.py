"""
ODC-SCI export - format data for Open Data Commons for Spinal Cord Injury standard.

Exports analysis findings into manual scoring file tabs/formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from .data_loader import DataLoader, SessionData


class ODC_SCI_Exporter:
    """Export MouseReach findings in ODC-SCI compatible format."""

    def __init__(self, data_loader: DataLoader):
        """
        Initialize exporter.

        Args:
            data_loader: DataLoader with session data
        """
        self.loader = data_loader

    def export_behavioral_summary(self, output_path: Path, metadata: Optional[Dict] = None):
        """
        Export behavioral summary in ODC-SCI format.

        Creates Excel file with standard tabs:
        - Session Summary
        - Reach Kinematics
        - Outcome Statistics
        - Temporal Trends

        Args:
            output_path: Output Excel file path
            metadata: Optional metadata dict (study, PI, etc.)
        """
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

        # Tab 1: Session Summary
        session_df = self._create_session_summary_tab()
        session_df.to_excel(writer, sheet_name='Session Summary', index=False)

        # Tab 2: Reach Kinematics
        kinematics_df = self._create_kinematics_tab()
        kinematics_df.to_excel(writer, sheet_name='Reach Kinematics', index=False)

        # Tab 3: Outcome Statistics
        outcomes_df = self._create_outcomes_tab()
        outcomes_df.to_excel(writer, sheet_name='Outcome Statistics', index=False)

        # Tab 4: Temporal Trends
        trends_df = self._create_trends_tab()
        trends_df.to_excel(writer, sheet_name='Temporal Trends', index=False)

        # Tab 5: Metadata
        if metadata:
            metadata_df = self._create_metadata_tab(metadata)
            metadata_df.to_excel(writer, sheet_name='Study Metadata', index=False)

        writer.close()
        print(f"Exported ODC-SCI formatted data to {output_path}")

    def _create_session_summary_tab(self) -> pd.DataFrame:
        """Create session summary tab in ODC-SCI format."""
        rows = []

        for session in self.loader.sessions:
            m = session.metadata

            rows.append({
                'Date': m.date.strftime('%Y-%m-%d'),
                'Day of Week': m.day_of_week,
                'Subject ID': m.mouse_id,
                'Session': m.phase,
                'Video Name': m.video_name,

                # Performance metrics
                'Success Rate (%)': session.success_rate * 100,
                'Retrieval Rate (%)': session.retrieval_rate * 100,
                'Total Segments': session.n_segments,
                'Total Reaches': session.n_reaches,
                'Causal Reaches': session.n_causal_reaches,

                # Behavioral engagement
                'Attention Score (%)': session.mean_attention_score,

                # Reach kinematics
                'Mean Reach Extent (mm)': session.mean_reach_extent,
                'Mean Reach Duration (frames)': session.mean_reach_duration,
                'Mean Peak Velocity (px/frame)': session.mean_peak_velocity,

                # Data quality
                'Analysis Date': datetime.now().strftime('%Y-%m-%d'),
                'Pipeline Version': 'MouseReach-v2.1.0'
            })

        return pd.DataFrame(rows)

    def _create_kinematics_tab(self) -> pd.DataFrame:
        """Create reach kinematics tab with detailed metrics."""
        rows = []

        for session in self.loader.sessions:
            # Extract reach-level data from features
            features = session.features
            segments = features.get('segments', [])

            for segment in segments:
                for reach in segment.get('reaches', []):
                    rows.append({
                        'Subject ID': session.metadata.mouse_id,
                        'Session': session.metadata.phase,
                        'Date': session.metadata.date.strftime('%Y-%m-%d'),
                        'Segment Number': segment['segment_num'],
                        'Reach ID': reach['reach_id'],
                        'Is Causal': reach.get('causal_reach', False),
                        'Outcome': reach.get('outcome', 'N/A'),

                        # Temporal
                        'Duration (frames)': reach.get('duration_frames'),
                        'Start Frame': reach.get('start_frame'),
                        'Apex Frame': reach.get('apex_frame'),
                        'End Frame': reach.get('end_frame'),

                        # Spatial
                        'Max Extent (mm)': reach.get('max_extent_mm'),
                        'Max Extent (ruler)': reach.get('max_extent_ruler'),

                        # Velocity
                        'Peak Velocity (px/frame)': reach.get('peak_velocity_px_per_frame'),
                        'Velocity at Apex (mm/s)': reach.get('velocity_at_apex_mm_per_sec'),
                        'Mean Velocity (px/frame)': reach.get('mean_velocity_px_per_frame'),

                        # Trajectory
                        'Straightness': reach.get('trajectory_straightness'),
                        'Smoothness': reach.get('trajectory_smoothness'),

                        # Orientation
                        'Hand Angle at Apex (deg)': reach.get('hand_angle_at_apex_deg'),
                        'Hand Rotation Total (deg)': reach.get('hand_rotation_total_deg'),

                        # Body/posture
                        'Head Width at Apex (mm)': reach.get('head_width_at_apex_mm'),
                        'Nose to Slit at Apex (mm)': reach.get('nose_to_slit_at_apex_mm'),
                        'Head Angle at Apex (deg)': reach.get('head_angle_at_apex_deg'),
                        'Head Angle Change (deg)': reach.get('head_angle_change_deg'),

                        # Confidence
                        'Mean DLC Confidence': reach.get('mean_likelihood'),
                        'Frames Low Confidence': reach.get('frames_low_confidence'),

                        # Quality/Review flags
                        'Exclude From Analysis': reach.get('exclude_from_analysis', False),
                        'Exclude Reason': reach.get('exclude_reason', ''),
                        'Human Corrected': reach.get('human_corrected', False),
                        'Source': reach.get('source', 'algorithm'),
                        'Review Note': reach.get('review_note', ''),
                        'Segment Flagged': segment.get('flagged_for_review', False),
                        'Segment Flag Reason': segment.get('flag_reason', '')
                    })

        return pd.DataFrame(rows)

    def _create_outcomes_tab(self) -> pd.DataFrame:
        """Create outcome statistics tab."""
        rows = []

        for session in self.loader.sessions:
            features = session.features
            summary = features.get('summary', {})
            outcome_counts = summary.get('outcome_counts', {})

            rows.append({
                'Subject ID': session.metadata.mouse_id,
                'Session': session.metadata.phase,
                'Date': session.metadata.date.strftime('%Y-%m-%d'),

                # Outcome counts
                'Retrieved': outcome_counts.get('retrieved', 0),
                'Displaced (SA)': outcome_counts.get('displaced_sa', 0),
                'Displaced (Outside)': outcome_counts.get('displaced_outside', 0),
                'Untouched': outcome_counts.get('untouched', 0),
                'Uncertain': outcome_counts.get('uncertain', 0),
                'No Pellet': outcome_counts.get('no_pellet', 0),

                # Rates
                'Success Rate (%)': session.success_rate * 100,
                'Retrieval Rate (%)': session.retrieval_rate * 100,

                # Reach efficiency
                'Total Reaches': session.n_reaches,
                'Causal Reaches': session.n_causal_reaches,
                'Reach Efficiency (%)': (session.n_causal_reaches / session.n_reaches * 100) if session.n_reaches > 0 else 0
            })

        return pd.DataFrame(rows)

    def _create_trends_tab(self) -> pd.DataFrame:
        """Create temporal trends tab (learning/fatigue)."""
        rows = []

        # Group by mouse
        for mouse_id in self.loader.get_mouse_ids():
            sessions = self.loader.get_mouse_sessions(mouse_id)

            if len(sessions) < 2:
                continue

            # Compute trends
            session_numbers = list(range(1, len(sessions) + 1))
            success_rates = [s.success_rate for s in sessions]
            attention_scores = [s.mean_attention_score for s in sessions]
            velocities = [s.mean_peak_velocity for s in sessions if s.mean_peak_velocity > 0]

            # Linear regression slopes
            success_slope = np.polyfit(session_numbers, success_rates, 1)[0] if len(success_rates) > 1 else 0
            attention_slope = np.polyfit(session_numbers, attention_scores, 1)[0] if len(attention_scores) > 1 else 0
            velocity_slope = np.polyfit(range(len(velocities)), velocities, 1)[0] if len(velocities) > 1 else 0

            rows.append({
                'Subject ID': mouse_id,
                'Number of Sessions': len(sessions),
                'Date Range': f"{sessions[0].metadata.date.strftime('%Y-%m-%d')} to {sessions[-1].metadata.date.strftime('%Y-%m-%d')}",
                'Days Spanned': (sessions[-1].metadata.date - sessions[0].metadata.date).days,

                # Initial vs final
                'Initial Success Rate (%)': sessions[0].success_rate * 100,
                'Final Success Rate (%)': sessions[-1].success_rate * 100,
                'Learning Gain (%)': (sessions[-1].success_rate - sessions[0].success_rate) * 100,

                # Trends
                'Success Rate Slope (per session)': success_slope,
                'Attention Slope (per session)': attention_slope,
                'Velocity Slope (per session)': velocity_slope,

                # Variability
                'Success Rate Std Dev': np.std(success_rates),
                'Attention Std Dev': np.std(attention_scores)
            })

        return pd.DataFrame(rows)

    def _create_metadata_tab(self, metadata: Dict) -> pd.DataFrame:
        """Create metadata tab with study information."""
        rows = [
            {'Field': 'Study Name', 'Value': metadata.get('study_name', 'N/A')},
            {'Field': 'Principal Investigator', 'Value': metadata.get('pi', 'N/A')},
            {'Field': 'Institution', 'Value': metadata.get('institution', 'N/A')},
            {'Field': 'Analysis Date', 'Value': datetime.now().strftime('%Y-%m-%d')},
            {'Field': 'Pipeline', 'Value': 'MouseReach (Automated Single Pellet Analysis v2)'},
            {'Field': 'Pipeline Version', 'Value': 'v2.1.0'},
            {'Field': 'DLC Model', 'Value': metadata.get('dlc_model', 'User-trained model')},
            {'Field': 'Total Subjects', 'Value': len(self.loader.get_mouse_ids())},
            {'Field': 'Total Sessions', 'Value': len(self.loader.sessions)},
            {'Field': 'Date Range', 'Value': f"{self.loader.sessions[0].metadata.date.strftime('%Y-%m-%d')} to {self.loader.sessions[-1].metadata.date.strftime('%Y-%m-%d')}"},
            {'Field': 'ODC-SCI Compliance', 'Value': 'Version 1.0'},
            {'Field': 'Data Repository', 'Value': metadata.get('repository', 'N/A')},
            {'Field': 'DOI', 'Value': metadata.get('doi', 'N/A')}
        ]

        return pd.DataFrame(rows)

    def export_per_mouse_summary(self, mouse_id: str, output_path: Path):
        """
        Export detailed summary for a single mouse (ODC-SCI format).

        Args:
            mouse_id: Mouse identifier
            output_path: Output Excel file path
        """
        sessions = self.loader.get_mouse_sessions(mouse_id)

        if not sessions:
            print(f"No sessions found for {mouse_id}")
            return

        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

        # Session timeline
        timeline_data = []
        for i, session in enumerate(sessions, 1):
            timeline_data.append({
                'Session Number': i,
                'Date': session.metadata.date.strftime('%Y-%m-%d'),
                'Day of Week': session.metadata.day_of_week,
                'Phase': session.metadata.phase,
                'Success Rate (%)': session.success_rate * 100,
                'Attention (%)': session.mean_attention_score,
                'Mean Velocity': session.mean_peak_velocity,
                'Total Reaches': session.n_reaches
            })

        timeline_df = pd.DataFrame(timeline_data)
        timeline_df.to_excel(writer, sheet_name='Session Timeline', index=False)

        # Detailed kinematics (all reaches)
        kinematics_rows = []
        for session in sessions:
            for segment in session.features.get('segments', []):
                for reach in segment.get('reaches', []):
                    kinematics_rows.append({
                        'Date': session.metadata.date.strftime('%Y-%m-%d'),
                        'Session': session.metadata.phase,
                        'Segment': segment['segment_num'],
                        'Reach ID': reach['reach_id'],
                        'Is Causal': reach.get('causal_reach', False),
                        'Extent (mm)': reach.get('max_extent_mm'),
                        'Velocity (mm/s)': reach.get('velocity_at_apex_mm_per_sec'),
                        'Duration (frames)': reach.get('duration_frames'),
                        'Straightness': reach.get('trajectory_straightness'),
                        'Smoothness': reach.get('trajectory_smoothness')
                    })

        if kinematics_rows:
            kinematics_df = pd.DataFrame(kinematics_rows)
            kinematics_df.to_excel(writer, sheet_name='All Reaches', index=False)

        writer.close()
        print(f"Exported {mouse_id} summary to {output_path}")
