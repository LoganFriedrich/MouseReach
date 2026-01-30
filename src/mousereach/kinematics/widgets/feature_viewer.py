"""
Interactive viewer for extracted grasp kinematics features.

Allows browsing and inspecting feature extraction results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class FeatureViewer:
    """Interactive viewer for grasp kinematics features."""

    def __init__(self, features_path: Path):
        """
        Initialize viewer.

        Args:
            features_path: Path to *_grasp_features.json file
        """
        self.features_path = Path(features_path)
        self.features = None
        self.current_segment = 0
        self.current_reach = 0

        # Load features
        with open(features_path) as f:
            self.features = json.load(f)

    def show_summary(self):
        """Display video-level summary."""
        print("=" * 80)
        print(f"VIDEO: {self.features['video_name']}")
        print("=" * 80)
        print(f"Extractor version: {self.features['extractor_version']}")
        print(f"Total frames: {self.features['total_frames']:,}")
        print(f"Number of segments: {self.features['n_segments']}")
        print(f"Extracted: {self.features['extracted_at']}")

        # Summary statistics
        summary = self.features.get('summary', {})
        if summary:
            print("\nSUMMARY STATISTICS:")
            print(f"  Total reaches: {summary.get('total_reaches', 'N/A')}")
            print(f"  Causal reaches: {summary.get('causal_reaches', 'N/A')}")

            if 'outcome_counts' in summary:
                print("\n  Outcome distribution:")
                for outcome, count in summary['outcome_counts'].items():
                    print(f"    {outcome}: {count}")

            if 'mean_reach_extent_mm' in summary and summary['mean_reach_extent_mm']:
                print(f"\n  Mean reach extent: {summary['mean_reach_extent_mm']:.2f} mm")
                print(f"  Mean peak velocity: {summary.get('mean_peak_velocity', 0):.2f} px/frame")
                print(f"  Mean straightness: {summary.get('mean_straightness', 0):.3f}")

        print("\nNavigation:")
        print("  [n]ext segment  [p]revious segment  [#] jump to segment")
        print("  [r]each view    [s]egment view")
        print("  [e]xport to CSV [q]uit")
        print("=" * 80)

    def show_segment(self, seg_num: int):
        """Display detailed segment information."""
        if seg_num < 0 or seg_num >= self.features['n_segments']:
            print(f"Invalid segment number: {seg_num}")
            return

        seg = self.features['segments'][seg_num]
        self.current_segment = seg_num

        print("\n" + "=" * 80)
        print(f"SEGMENT {seg['segment_num']} (Index: {seg_num})")
        print("=" * 80)

        # Basic info
        print(f"Frames: {seg['start_frame']} to {seg['end_frame']} ({seg['end_frame'] - seg['start_frame'] + 1} frames)")
        print(f"Outcome: {seg['outcome']} (confidence: {seg['outcome_confidence']:.2f})")
        if seg['outcome_flagged']:
            print(f"  *** FLAGGED FOR REVIEW ***")
        print(f"Reaches: {seg['n_reaches']} total")
        if seg['causal_reach_id']:
            print(f"  Causal reach ID: {seg['causal_reach_id']}")

        # Temporal context
        print("\nTEMPORAL CONTEXT:")
        print(f"  Duration: {seg.get('segment_duration_sec', 0):.2f} sec")
        if seg.get('time_to_first_reach_sec') is not None:
            print(f"  Time to first reach: {seg['time_to_first_reach_sec']:.2f} sec")
        if seg.get('mean_inter_reach_interval_sec') is not None:
            print(f"  Inter-reach interval: {seg['mean_inter_reach_interval_sec']:.2f} sec")

        # Behavioral engagement
        print("\nBEHAVIORAL ENGAGEMENT:")
        if seg.get('attention_score') is not None:
            print(f"  Attention score: {seg['attention_score']:.1f}%")
            if seg['attention_score'] > 70:
                print(f"    -> Highly engaged")
            elif seg['attention_score'] > 40:
                print(f"    -> Moderately engaged")
            else:
                print(f"    -> Low engagement / distracted")

        # Pellet positioning
        print("\nPELLET POSITIONING (before first reach):")
        if seg.get('pellet_position_idealness') is not None:
            idealness = seg['pellet_position_idealness']
            print(f"  Idealness: {idealness:.3f}")
            if idealness > 0.8:
                print(f"    -> Excellent positioning")
            elif idealness > 0.5:
                print(f"    -> Acceptable positioning")
            else:
                print(f"    -> Poor positioning (handicapped)")
            print(f"  Lateral offset: {seg['pellet_lateral_offset_mm']:.2f} mm")
            print(f"  Depth offset: {seg['pellet_depth_offset_mm']:.2f} mm")
        else:
            print(f"  N/A (no pellet data)")

        # Body posture
        print("\nBODY POSTURE (segment averages):")
        if seg.get('mean_head_width_mm') is not None:
            print(f"  Head width: {seg['mean_head_width_mm']:.2f} mm")
        if seg.get('mean_nose_to_slit_mm') is not None:
            print(f"  Nose-to-slit distance: {seg['mean_nose_to_slit_mm']:.2f} mm")
        if seg.get('mean_head_angle_deg') is not None:
            print(f"  Head angle: {seg['mean_head_angle_deg']:.1f} degrees")

        # Data quality
        print("\nDATA QUALITY:")
        if seg.get('mean_tracking_quality') is not None:
            quality = seg['mean_tracking_quality']
            print(f"  Mean tracking quality: {quality:.3f}")
            if quality > 0.85:
                print(f"    -> Excellent tracking")
            elif quality > 0.70:
                print(f"    -> Good tracking")
            else:
                print(f"    -> Poor tracking - may be unreliable")
        print(f"  Dropout frames: {seg.get('tracking_dropout_frames', 0)}")

        # Reach summary
        if seg['n_reaches'] > 0:
            print(f"\nREACHES ({seg['n_reaches']} total):")
            for i, reach in enumerate(seg['reaches']):
                causal = " [CAUSAL]" if reach.get('causal_reach') else ""
                first = " [FIRST]" if reach.get('is_first_reach') else ""
                last = " [LAST]" if reach.get('is_last_reach') else ""
                extent = reach.get('max_extent_mm', 0)
                vel = reach.get('peak_velocity_px_per_frame', 0)
                print(f"  {i+1}. Reach ID {reach['reach_id']}: {extent:.2f}mm extent, {vel:.1f} px/f velocity{causal}{first}{last}")

        print("\nCommands: [n]ext [p]rev [r<#>] view reach [s]egment list [q]uit")

    def show_reach(self, reach_idx: int):
        """Display detailed reach information."""
        seg = self.features['segments'][self.current_segment]

        if reach_idx < 0 or reach_idx >= len(seg['reaches']):
            print(f"Invalid reach index: {reach_idx}")
            return

        reach = seg['reaches'][reach_idx]
        self.current_reach = reach_idx

        print("\n" + "=" * 80)
        print(f"REACH {reach_idx + 1} of {len(seg['reaches'])} (Segment {seg['segment_num']}, Reach ID {reach['reach_id']})")
        print("=" * 80)

        # Context flags
        print("CONTEXT:")
        print(f"  Position in sequence: {reach_idx + 1} of {reach.get('n_reaches_in_segment', len(seg['reaches']))}")
        print(f"  First reach: {reach.get('is_first_reach', False)}")
        print(f"  Last reach: {reach.get('is_last_reach', False)}")
        print(f"  Causal reach: {reach.get('causal_reach', False)}")
        if reach.get('outcome'):
            print(f"  Outcome: {reach['outcome']}")

        # Temporal
        print("\nTEMPORAL:")
        print(f"  Start frame: {reach['start_frame']}")
        print(f"  Apex frame: {reach.get('apex_frame', 'N/A')}")
        print(f"  End frame: {reach['end_frame']}")
        print(f"  Duration: {reach.get('duration_frames', 0)} frames ({reach.get('duration_frames', 0) / 30:.2f} sec)")

        # Spatial extent
        print("\nSPATIAL EXTENT:")
        if reach.get('max_extent_mm') is not None:
            print(f"  Max extent: {reach['max_extent_mm']:.2f} mm")
        if reach.get('apex_distance_to_pellet_mm') is not None:
            dist = reach['apex_distance_to_pellet_mm']
            print(f"  Distance to pellet at apex: {dist:.2f} mm")
            if dist < 3:
                print(f"    -> Reached pellet")
            elif dist < 10:
                print(f"    -> Close to pellet")
            else:
                print(f"    -> Missed pellet")

        # Velocity
        print("\nVELOCITY:")
        if reach.get('peak_velocity_px_per_frame') is not None:
            print(f"  Peak velocity: {reach['peak_velocity_px_per_frame']:.2f} px/frame")
        if reach.get('velocity_at_apex_mm_per_sec') is not None:
            print(f"  Velocity at apex: {reach['velocity_at_apex_mm_per_sec']:.2f} mm/sec")
        if reach.get('mean_velocity_px_per_frame') is not None:
            print(f"  Mean velocity: {reach['mean_velocity_px_per_frame']:.2f} px/frame")

        # Trajectory quality
        print("\nTRAJECTORY QUALITY:")
        if reach.get('trajectory_straightness') is not None:
            straightness = reach['trajectory_straightness']
            print(f"  Straightness: {straightness:.3f}")
            if straightness > 0.8:
                print(f"    -> Very direct path")
            elif straightness > 0.5:
                print(f"    -> Moderately curved")
            else:
                print(f"    -> Highly curved/corrective")
        if reach.get('trajectory_smoothness') is not None:
            print(f"  Smoothness: {reach['trajectory_smoothness']:.3f}")
        if reach.get('lateral_deviation_mm') is not None:
            print(f"  Lateral deviation: {reach['lateral_deviation_mm']:.2f} mm")

        # Hand orientation
        print("\nHAND ORIENTATION:")
        if reach.get('hand_angle_at_apex_deg') is not None:
            print(f"  Angle at apex: {reach['hand_angle_at_apex_deg']:.1f} degrees")
        if reach.get('hand_rotation_total_deg') is not None:
            print(f"  Total rotation: {reach['hand_rotation_total_deg']:.1f} degrees")

        # Body at apex
        print("\nBODY AT APEX:")
        if reach.get('head_width_at_apex_mm') is not None:
            print(f"  Head width: {reach['head_width_at_apex_mm']:.2f} mm")
        if reach.get('nose_to_slit_at_apex_mm') is not None:
            print(f"  Nose-to-slit distance: {reach['nose_to_slit_at_apex_mm']:.2f} mm")
        if reach.get('head_angle_at_apex_deg') is not None:
            print(f"  Head angle: {reach['head_angle_at_apex_deg']:.1f} degrees")
        if reach.get('head_angle_change_deg') is not None:
            print(f"  Head rotation during reach: {reach['head_angle_change_deg']:.1f} degrees")

        # Quality
        print("\nQUALITY:")
        if reach.get('mean_likelihood') is not None:
            print(f"  Mean DLC confidence: {reach['mean_likelihood']:.3f}")
        if reach.get('frames_low_confidence') is not None:
            print(f"  Low confidence frames: {reach['frames_low_confidence']}")
        if reach.get('flagged_for_review'):
            print(f"  *** FLAGGED: {reach.get('flag_reason', 'Unknown reason')} ***")

        print("\nCommands: [n]ext reach [p]rev reach [b]ack to segment [q]uit")

    def export_to_csv(self, output_dir: Path):
        """Export features to CSV files for analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        video_name = self.features['video_name']

        # Export segment-level features
        seg_rows = []
        for seg in self.features['segments']:
            row = {
                'video_name': video_name,
                'segment_num': seg['segment_num'],
                'start_frame': seg['start_frame'],
                'end_frame': seg['end_frame'],
                'outcome': seg['outcome'],
                'outcome_confidence': seg['outcome_confidence'],
                'outcome_flagged': seg['outcome_flagged'],
                'n_reaches': seg['n_reaches'],
                'causal_reach_id': seg.get('causal_reach_id'),
                'attention_score': seg.get('attention_score'),
                'segment_duration_sec': seg.get('segment_duration_sec'),
                'time_to_first_reach_sec': seg.get('time_to_first_reach_sec'),
                'mean_inter_reach_interval_sec': seg.get('mean_inter_reach_interval_sec'),
                'pellet_position_idealness': seg.get('pellet_position_idealness'),
                'pellet_lateral_offset_mm': seg.get('pellet_lateral_offset_mm'),
                'pellet_depth_offset_mm': seg.get('pellet_depth_offset_mm'),
                'mean_head_width_mm': seg.get('mean_head_width_mm'),
                'mean_nose_to_slit_mm': seg.get('mean_nose_to_slit_mm'),
                'mean_head_angle_deg': seg.get('mean_head_angle_deg'),
                'head_angle_variance': seg.get('head_angle_variance'),
                'nose_position_variance': seg.get('nose_position_variance'),
                'mean_tracking_quality': seg.get('mean_tracking_quality'),
                'tracking_dropout_frames': seg.get('tracking_dropout_frames'),
            }
            seg_rows.append(row)

        seg_df = pd.DataFrame(seg_rows)
        seg_path = output_dir / f"{video_name}_segments.csv"
        seg_df.to_csv(seg_path, index=False)
        print(f"Exported segment features to: {seg_path}")

        # Export reach-level features
        reach_rows = []
        for seg in self.features['segments']:
            for reach in seg.get('reaches', []):
                row = {
                    'video_name': video_name,
                    'segment_num': seg['segment_num'],
                    'reach_id': reach['reach_id'],
                    'reach_num': reach.get('reach_num'),
                    'is_first_reach': reach.get('is_first_reach'),
                    'is_last_reach': reach.get('is_last_reach'),
                    'n_reaches_in_segment': reach.get('n_reaches_in_segment'),
                    'causal_reach': reach.get('causal_reach'),
                    'outcome': reach.get('outcome'),
                    'start_frame': reach['start_frame'],
                    'apex_frame': reach.get('apex_frame'),
                    'end_frame': reach['end_frame'],
                    'duration_frames': reach.get('duration_frames'),
                    'max_extent_mm': reach.get('max_extent_mm'),
                    'apex_distance_to_pellet_mm': reach.get('apex_distance_to_pellet_mm'),
                    'lateral_deviation_mm': reach.get('lateral_deviation_mm'),
                    'peak_velocity_px_per_frame': reach.get('peak_velocity_px_per_frame'),
                    'velocity_at_apex_mm_per_sec': reach.get('velocity_at_apex_mm_per_sec'),
                    'mean_velocity_px_per_frame': reach.get('mean_velocity_px_per_frame'),
                    'trajectory_straightness': reach.get('trajectory_straightness'),
                    'trajectory_smoothness': reach.get('trajectory_smoothness'),
                    'hand_angle_at_apex_deg': reach.get('hand_angle_at_apex_deg'),
                    'hand_rotation_total_deg': reach.get('hand_rotation_total_deg'),
                    'head_width_at_apex_mm': reach.get('head_width_at_apex_mm'),
                    'nose_to_slit_at_apex_mm': reach.get('nose_to_slit_at_apex_mm'),
                    'head_angle_at_apex_deg': reach.get('head_angle_at_apex_deg'),
                    'head_angle_change_deg': reach.get('head_angle_change_deg'),
                    'mean_likelihood': reach.get('mean_likelihood'),
                    'frames_low_confidence': reach.get('frames_low_confidence'),
                }
                reach_rows.append(row)

        if reach_rows:
            reach_df = pd.DataFrame(reach_rows)
            reach_path = output_dir / f"{video_name}_reaches.csv"
            reach_df.to_csv(reach_path, index=False)
            print(f"Exported reach features to: {reach_path}")

        print(f"\nExport complete! Files in: {output_dir}")

    def run(self):
        """Run interactive viewer."""
        self.show_summary()

        while True:
            try:
                cmd = input("\n> ").strip().lower()

                if cmd == 'q':
                    print("Exiting viewer.")
                    break

                elif cmd == 'n':
                    self.show_segment(self.current_segment + 1)

                elif cmd == 'p':
                    self.show_segment(self.current_segment - 1)

                elif cmd.isdigit():
                    self.show_segment(int(cmd) - 1)

                elif cmd == 's':
                    self.show_summary()

                elif cmd.startswith('r'):
                    if len(cmd) > 1 and cmd[1:].isdigit():
                        reach_idx = int(cmd[1:]) - 1
                        self.show_reach(reach_idx)
                    else:
                        print("Reach commands: r1, r2, etc. to view specific reach")

                elif cmd == 'b':
                    self.show_segment(self.current_segment)

                elif cmd == 'e':
                    output_dir = self.features_path.parent / "csv_exports"
                    self.export_to_csv(output_dir)

                elif cmd == 'help' or cmd == 'h':
                    print("\nCommands:")
                    print("  n - next segment")
                    print("  p - previous segment")
                    print("  # - jump to segment number")
                    print("  r# - view reach (e.g., r1, r2)")
                    print("  s - show summary")
                    print("  b - back to current segment")
                    print("  e - export to CSV")
                    print("  q - quit")

                else:
                    print(f"Unknown command: {cmd}. Type 'help' for commands.")

            except KeyboardInterrupt:
                print("\nUse 'q' to quit.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """CLI entry point for feature viewer."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m mousereach_grasp_kinematics.widgets.feature_viewer <features.json>")
        print("\nOr:")
        print("  mousereach-grasp-view <features.json>")
        sys.exit(1)

    features_path = Path(sys.argv[1])

    if not features_path.exists():
        print(f"Error: File not found: {features_path}")
        sys.exit(1)

    viewer = FeatureViewer(features_path)
    viewer.run()


if __name__ == '__main__':
    main()
