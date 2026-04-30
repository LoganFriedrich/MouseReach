"""
Feature extraction for reaches linked to pellet outcomes.

Extracts kinematic and behavioral features from DLC tracking data
for reaches (Step 4) that have been linked to pellet outcomes (Step 3).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd


@dataclass
class ReachFeatures:
    """Features extracted for a single reach."""

    # Identifiers
    reach_id: int
    reach_num: int  # Within segment (1-indexed)
    segment_num: int

    # Outcome linkage (from Step 3)
    outcome: Optional[str] = None
    causal_reach: bool = False  # Was this the reach that caused the outcome?
    interaction_frame: Optional[int] = None
    distance_to_interaction: Optional[int] = None  # Frames between apex and interaction

    # Contextual flags
    is_first_reach: bool = False  # First reach in segment
    is_last_reach: bool = False  # Last reach in segment
    n_reaches_in_segment: int = 0  # Total reaches in this segment

    # Temporal features (from Step 4 - reach detection)
    start_frame: int = 0
    apex_frame: Optional[int] = None
    end_frame: int = 0
    duration_frames: int = 0

    # Extent features (from Step 4 - reach detection)
    # Kept for backward compatibility; superseded by per-paw extension_past_nose in `extended`.
    max_extent_pixels: Optional[float] = None
    max_extent_ruler: Optional[float] = None  # Normalized to 9mm ruler
    max_extent_mm: Optional[float] = None  # Physical units

    # Velocity features (computed from DLC)
    # Kept for backward compatibility; superseded by per-paw mean/peak/apex_speed in `extended`.
    velocity_at_apex_px_per_frame: Optional[float] = None
    velocity_at_apex_mm_per_sec: Optional[float] = None  # Assuming 30 fps
    peak_velocity_px_per_frame: Optional[float] = None
    mean_velocity_px_per_frame: Optional[float] = None

    # Trajectory features
    # Kept for backward compatibility; superseded by per-paw path_directness / motion_smoothness.
    trajectory_straightness: Optional[float] = None
    trajectory_smoothness: Optional[float] = None

    # Hand orientation features (kept)
    hand_angle_at_apex_deg: Optional[float] = None
    hand_rotation_total_deg: Optional[float] = None

    # Grasp aperture features (deprecated; superseded by paw_width_proxy_* in `extended`)
    grasp_aperture_max_mm: Optional[float] = None
    grasp_aperture_at_contact_mm: Optional[float] = None

    # Body/posture during reach
    head_width_at_apex_mm: Optional[float] = None
    nose_to_slit_at_apex_mm: Optional[float] = None
    head_angle_at_apex_deg: Optional[float] = None
    head_angle_change_deg: Optional[float] = None

    # Spatial context (deprecated; lateral_deviation now per-paw in `extended`)
    apex_distance_to_pellet_mm: Optional[float] = None
    lateral_deviation_mm: Optional[float] = None

    # Confidence/quality metrics
    # Kept; per-paw visibility profile in `extended` carries the richer signal.
    mean_likelihood: Optional[float] = None
    frames_low_confidence: int = 0
    tracking_quality_score: Optional[float] = None

    # Flags
    flagged_for_review: bool = False
    flag_reason: Optional[str] = None

    # Extended kinematic feature set (flat dict; keys are stable column names).
    # Populated by FeatureExtractor for v2 outputs. See REACH_KINEMATIC_DATA_DICTIONARY.md.
    extended: Optional[Dict[str, float]] = None


@dataclass
class SegmentFeatures:
    """Features for all reaches within a segment."""
    segment_num: int
    start_frame: int
    end_frame: int
    ruler_pixels: float

    outcome: str
    outcome_confidence: float
    outcome_flagged: bool

    n_reaches: int
    causal_reach_id: Optional[int] = None
    reaches: List[ReachFeatures] = None

    # Behavioral engagement
    attention_score: Optional[float] = None  # % of frames attending to tray

    # Body size/orientation (segment-level averages)
    mean_head_width_mm: Optional[float] = None  # LeftEar-RightEar (size proxy)
    mean_nose_to_slit_mm: Optional[float] = None  # Distance to tray opening
    mean_nose_height: Optional[float] = None  # Nose Y position (posture)
    mean_head_angle_deg: Optional[float] = None  # Head orientation

    # Stability/variability
    head_angle_variance: Optional[float] = None  # Head orientation stability
    nose_position_variance: Optional[float] = None  # Postural stability

    # Temporal context
    segment_duration_sec: Optional[float] = None  # Total segment time
    time_to_first_reach_sec: Optional[float] = None  # Latency to engage
    time_to_outcome_sec: Optional[float] = None  # When outcome occurred
    mean_inter_reach_interval_sec: Optional[float] = None  # Pacing of attempts

    # Pellet positioning context (before first reach or after tray movements)
    pellet_position_idealness: Optional[float] = None  # 0-1: How well-positioned pellet is relative to slit
    pellet_lateral_offset_mm: Optional[float] = None  # Lateral distance from ideal center position
    pellet_depth_offset_mm: Optional[float] = None  # Depth difference from ideal reachable position

    # Data quality context
    mean_tracking_quality: Optional[float] = None  # Overall DLC confidence
    tracking_dropout_frames: int = 0  # Frames with poor/no tracking

    def __post_init__(self):
        if self.reaches is None:
            self.reaches = []


@dataclass
class VideoFeatures:
    """Complete feature extraction results for a video."""
    video_name: str
    extractor_version: str
    total_frames: int
    n_segments: int

    segments: List[SegmentFeatures]

    summary: Dict = None
    extracted_at: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        return data


class FeatureExtractor:
    """Extract features from reaches linked to outcomes."""

    VERSION = "2.0.0"  # 2.0: extended kinematic feature set (nose-relative,
                       # slit-anchored, per-paw trajectories, paw shape proxies,
                       # per-paw visibility profile, tray contact). See
                       # REACH_KINEMATIC_DATA_DICTIONARY.md.
    FRAMERATE = 30.0  # fps (assumed - not authoritative; see data dictionary)
    RULER_MM = 9.0  # Physical size of ruler (SABL-SABR distance)

    def __init__(self):
        pass

    def extract(
        self,
        dlc_path: Path,
        reaches_path: Path,
        outcomes_path: Path
    ) -> VideoFeatures:
        """
        Extract features for all reaches in a video.

        Args:
            dlc_path: Path to DLC .h5 file
            reaches_path: Path to *_reaches.json from Step 4
            outcomes_path: Path to *_pellet_outcomes.json from Step 3

        Returns:
            VideoFeatures object with all extracted features
        """
        # Load DLC data
        df = self._load_dlc(dlc_path)

        # Load reaches
        with open(reaches_path) as f:
            reaches_data = json.load(f)

        # Load outcomes
        with open(outcomes_path) as f:
            outcomes_data = json.load(f)

        video_name = reaches_data['video_name']
        total_frames = reaches_data['total_frames']
        n_segments = reaches_data['n_segments']

        # Extract features for each segment
        segment_features = []

        for seg_data, outcome_data in zip(reaches_data['segments'], outcomes_data['segments']):
            seg_num = seg_data['segment_num']

            # Compute behavioral/postural features for this segment
            attention_score = self._compute_attention_score(
                df,
                seg_data['start_frame'],
                seg_data['end_frame']
            )

            body_features = self._compute_body_features_segment(
                df,
                seg_data['start_frame'],
                seg_data['end_frame'],
                seg_data['ruler_pixels']
            )

            # Compute temporal context
            temporal_context = self._compute_temporal_context(
                seg_data,
                seg_data['reaches']
            )

            # Compute pellet positioning (before first reach)
            first_reach_start = seg_data['reaches'][0]['start_frame'] if seg_data['reaches'] else None
            pellet_positioning = self._compute_pellet_positioning(
                df,
                seg_data['start_frame'],
                first_reach_start,
                seg_data['ruler_pixels']
            )

            # Compute quality metrics
            quality_metrics = self._compute_quality_metrics(
                df,
                seg_data['start_frame'],
                seg_data['end_frame']
            )

            # Create segment features container
            seg_features = SegmentFeatures(
                segment_num=seg_num,
                start_frame=seg_data['start_frame'],
                end_frame=seg_data['end_frame'],
                ruler_pixels=seg_data['ruler_pixels'],
                outcome=outcome_data['outcome'],
                outcome_confidence=outcome_data['confidence'],
                outcome_flagged=outcome_data['flagged_for_review'],
                n_reaches=seg_data['n_reaches'],
                causal_reach_id=outcome_data.get('causal_reach_id'),
                reaches=[],
                attention_score=attention_score,
                **body_features,  # Unpack body feature dict
                **temporal_context,  # Unpack temporal context
                **pellet_positioning,  # Unpack pellet positioning
                **quality_metrics  # Unpack quality metrics
            )

            # Extract features for each reach in this segment
            n_reaches = len(seg_data['reaches'])
            for i, reach in enumerate(seg_data['reaches']):
                reach_features = self._extract_reach_features(
                    reach,
                    df,
                    seg_data['ruler_pixels'],
                    outcome_data
                )

                # Set contextual flags
                reach_features.is_first_reach = (i == 0)
                reach_features.is_last_reach = (i == n_reaches - 1)
                reach_features.n_reaches_in_segment = n_reaches

                seg_features.reaches.append(reach_features)

            segment_features.append(seg_features)

        # Compute summary statistics
        summary = self._compute_summary(segment_features)

        # Get timestamp
        from datetime import datetime
        extracted_at = datetime.now().isoformat()

        return VideoFeatures(
            video_name=video_name,
            extractor_version=self.VERSION,
            total_frames=total_frames,
            n_segments=n_segments,
            segments=segment_features,
            summary=summary,
            extracted_at=extracted_at
        )

    def _extract_reach_features(
        self,
        reach: Dict,
        df: pd.DataFrame,
        ruler_pixels: float,
        outcome_data: Dict
    ) -> ReachFeatures:
        """Extract all features for a single reach."""

        reach_id = reach['reach_id']
        start_frame = reach['start_frame']
        apex_frame = reach.get('apex_frame')
        end_frame = reach['end_frame']

        # Initialize features
        features = ReachFeatures(
            reach_id=reach_id,
            reach_num=reach['reach_num'],
            segment_num=0,  # Will be set by caller
            start_frame=start_frame,
            apex_frame=apex_frame,
            end_frame=end_frame,
            duration_frames=reach['duration_frames'],
            max_extent_pixels=reach.get('max_extent_pixels'),
            max_extent_ruler=reach.get('max_extent_ruler')
        )

        # Check if this is the causal reach
        causal_reach_id = outcome_data.get('causal_reach_id')
        if causal_reach_id is not None and causal_reach_id == reach_id:
            features.causal_reach = True
            features.outcome = outcome_data['outcome']
            features.interaction_frame = outcome_data.get('interaction_frame')

            # Get reach features from outcome data
            reach_features_dict = outcome_data.get('reach_features', {})
            if reach_features_dict:
                features.distance_to_interaction = reach_features_dict.get('distance_to_interaction')

        # Convert extent to mm
        if features.max_extent_ruler is not None:
            features.max_extent_mm = features.max_extent_ruler * self.RULER_MM

        # Extract reach trajectory for this reach
        reach_df = df.iloc[start_frame:end_frame+1]

        # Compute velocity features
        velocity_features = self._compute_velocity_features(reach_df, apex_frame - start_frame if apex_frame else None)
        features.velocity_at_apex_px_per_frame = velocity_features['velocity_at_apex']
        features.peak_velocity_px_per_frame = velocity_features['peak_velocity']
        features.mean_velocity_px_per_frame = velocity_features['mean_velocity']

        # Convert velocity to mm/s
        if features.velocity_at_apex_px_per_frame is not None:
            px_to_mm = self.RULER_MM / ruler_pixels
            features.velocity_at_apex_mm_per_sec = features.velocity_at_apex_px_per_frame * px_to_mm * self.FRAMERATE

        # Compute trajectory features
        traj_features = self._compute_trajectory_features(reach_df)
        features.trajectory_straightness = traj_features['straightness']
        features.trajectory_smoothness = traj_features['smoothness']

        # Compute hand orientation features
        orient_features = self._compute_orientation_features(reach_df, apex_frame - start_frame if apex_frame else None)
        features.hand_angle_at_apex_deg = orient_features['angle_at_apex']
        features.hand_rotation_total_deg = orient_features['total_rotation']

        # Compute confidence metrics
        conf_features = self._compute_confidence_features(reach_df)
        features.mean_likelihood = conf_features['mean_likelihood']
        features.frames_low_confidence = conf_features['frames_low_confidence']

        # Compute body/posture features at apex
        if apex_frame is not None:
            body_features = self._compute_body_features_reach(df, reach_df, apex_frame, start_frame, ruler_pixels)
            features.head_width_at_apex_mm = body_features['head_width_mm']
            features.nose_to_slit_at_apex_mm = body_features['nose_to_slit_mm']
            features.head_angle_at_apex_deg = body_features['head_angle_deg']
            features.head_angle_change_deg = body_features['head_angle_change']

        # Extended (v2) kinematic feature set populated as a flat dict on
        # `features.extended`. See REACH_KINEMATIC_DATA_DICTIONARY.md.
        try:
            features.extended = self._extended_features(
                df,
                start_frame=start_frame,
                apex_frame=apex_frame,
                end_frame=end_frame,
                interaction_frame=features.interaction_frame,
                ruler_pixels=ruler_pixels,
            )
        except Exception as exc:
            features.extended = {'_extended_features_error': str(exc)}

        return features

    def _compute_velocity_features(self, reach_df: pd.DataFrame, apex_idx: Optional[int]) -> Dict:
        """Compute velocity features from RightHand trajectory."""

        rh_x = reach_df['RightHand_x'].values
        rh_y = reach_df['RightHand_y'].values

        # Compute frame-to-frame velocity
        dx = np.diff(rh_x)
        dy = np.diff(rh_y)
        velocity = np.sqrt(dx**2 + dy**2)

        # Handle NaNs
        velocity = velocity[~np.isnan(velocity)]

        if len(velocity) == 0:
            return {
                'velocity_at_apex': None,
                'peak_velocity': None,
                'mean_velocity': None
            }

        velocity_at_apex = None
        if apex_idx is not None and 0 <= apex_idx < len(velocity):
            velocity_at_apex = float(velocity[apex_idx])

        return {
            'velocity_at_apex': velocity_at_apex,
            'peak_velocity': float(velocity.max()),
            'mean_velocity': float(velocity.mean())
        }

    def _compute_trajectory_features(self, reach_df: pd.DataFrame) -> Dict:
        """Compute trajectory straightness and smoothness."""

        rh_x = reach_df['RightHand_x'].values
        rh_y = reach_df['RightHand_y'].values

        # Remove NaNs
        valid = ~(np.isnan(rh_x) | np.isnan(rh_y))
        rh_x = rh_x[valid]
        rh_y = rh_y[valid]

        if len(rh_x) < 2:
            return {'straightness': None, 'smoothness': None}

        # Straightness: ratio of straight-line distance to path length
        straight_dist = np.sqrt((rh_x[-1] - rh_x[0])**2 + (rh_y[-1] - rh_y[0])**2)

        dx = np.diff(rh_x)
        dy = np.diff(rh_y)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))

        straightness = straight_dist / path_length if path_length > 0 else None

        # Smoothness: inverse of normalized jerk
        if len(rh_x) < 3:
            smoothness = None
        else:
            # Jerk = third derivative of position
            vel_x = np.diff(rh_x)
            vel_y = np.diff(rh_y)
            acc_x = np.diff(vel_x)
            acc_y = np.diff(vel_y)
            jerk_x = np.diff(acc_x)
            jerk_y = np.diff(acc_y)
            jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2)

            # Normalized jerk (lower is smoother)
            if len(jerk_mag) > 0:
                normalized_jerk = np.mean(jerk_mag)
                smoothness = 1.0 / (1.0 + normalized_jerk) if normalized_jerk > 0 else 1.0
            else:
                smoothness = None

        return {
            'straightness': float(straightness) if straightness is not None else None,
            'smoothness': float(smoothness) if smoothness is not None else None
        }

    def _compute_orientation_features(self, reach_df: pd.DataFrame, apex_idx: Optional[int]) -> Dict:
        """Compute hand orientation from RH keypoints."""

        # Use RHLeft and RHRight to compute hand angle
        rhl_x = reach_df['RHLeft_x'].values
        rhl_y = reach_df['RHLeft_y'].values
        rhr_x = reach_df['RHRight_x'].values
        rhr_y = reach_df['RHRight_y'].values

        # Compute angle for each frame
        angles = []
        for i in range(len(rhl_x)):
            if not (np.isnan(rhl_x[i]) or np.isnan(rhr_x[i])):
                dx = rhr_x[i] - rhl_x[i]
                dy = rhr_y[i] - rhl_y[i]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                angles.append(angle)

        if len(angles) == 0:
            return {'angle_at_apex': None, 'total_rotation': None}

        angles = np.array(angles)

        angle_at_apex = None
        if apex_idx is not None and 0 <= apex_idx < len(angles):
            angle_at_apex = float(angles[apex_idx])

        # Total rotation: sum of absolute angle changes
        angle_diffs = np.diff(angles)
        # Handle wraparound at ±180°
        angle_diffs = np.where(angle_diffs > 180, angle_diffs - 360, angle_diffs)
        angle_diffs = np.where(angle_diffs < -180, angle_diffs + 360, angle_diffs)
        total_rotation = float(np.sum(np.abs(angle_diffs)))

        return {
            'angle_at_apex': angle_at_apex,
            'total_rotation': total_rotation
        }

    def _compute_confidence_features(self, reach_df: pd.DataFrame) -> Dict:
        """Compute DLC confidence metrics for the reach."""

        rh_likelihood = reach_df['RightHand_likelihood'].values

        # Remove NaNs
        valid_likelihood = rh_likelihood[~np.isnan(rh_likelihood)]

        if len(valid_likelihood) == 0:
            return {'mean_likelihood': None, 'frames_low_confidence': 0}

        mean_likelihood = float(valid_likelihood.mean())
        frames_low_confidence = int(np.sum(valid_likelihood < 0.5))

        return {
            'mean_likelihood': mean_likelihood,
            'frames_low_confidence': frames_low_confidence
        }

    def _compute_body_features_reach(
        self,
        df: pd.DataFrame,
        reach_df: pd.DataFrame,
        apex_frame: int,
        start_frame: int,
        ruler_pixels: float
    ) -> Dict:
        """
        Compute body/posture features at reach apex.

        Args:
            df: Full DLC dataframe
            reach_df: Reach-only dataframe
            apex_frame: Absolute apex frame index
            start_frame: Reach start frame (absolute)
            ruler_pixels: Pixels per ruler unit

        Returns:
            Dictionary of body features at apex
        """
        mm_per_pixel = self.RULER_MM / ruler_pixels if ruler_pixels > 0 else 0
        features = {}

        apex_row = df.iloc[apex_frame]
        start_row = df.iloc[start_frame]

        # Head width at apex
        if 'LeftEar_x' in df.columns and 'RightEar_x' in df.columns:
            if apex_row['LeftEar_likelihood'] > 0.7 and apex_row['RightEar_likelihood'] > 0.7:
                head_width_px = np.sqrt(
                    (apex_row['RightEar_x'] - apex_row['LeftEar_x'])**2 +
                    (apex_row['RightEar_y'] - apex_row['LeftEar_y'])**2
                )
                features['head_width_mm'] = float(head_width_px * mm_per_pixel)
            else:
                features['head_width_mm'] = None
        else:
            features['head_width_mm'] = None

        # Nose to slit distance at apex
        if 'Nose_x' in df.columns and 'BOXR_x' in df.columns:
            if apex_row['Nose_likelihood'] > 0.7 and apex_row['BOXR_likelihood'] > 0.7:
                distance_px = np.sqrt(
                    (apex_row['Nose_x'] - apex_row['BOXR_x'])**2 +
                    (apex_row['Nose_y'] - apex_row['BOXR_y'])**2
                )
                features['nose_to_slit_mm'] = float(distance_px * mm_per_pixel)
            else:
                features['nose_to_slit_mm'] = None
        else:
            features['nose_to_slit_mm'] = None

        # Head angle at apex
        if 'LeftEar_x' in df.columns and 'RightEar_x' in df.columns:
            if apex_row['LeftEar_likelihood'] > 0.7 and apex_row['RightEar_likelihood'] > 0.7:
                dx = apex_row['RightEar_x'] - apex_row['LeftEar_x']
                dy = apex_row['RightEar_y'] - apex_row['LeftEar_y']
                angle_apex = np.arctan2(dy, dx) * 180 / np.pi
                features['head_angle_deg'] = float(angle_apex)

                # Head angle change from start to apex
                if start_row['LeftEar_likelihood'] > 0.7 and start_row['RightEar_likelihood'] > 0.7:
                    dx_start = start_row['RightEar_x'] - start_row['LeftEar_x']
                    dy_start = start_row['RightEar_y'] - start_row['LeftEar_y']
                    angle_start = np.arctan2(dy_start, dx_start) * 180 / np.pi
                    angle_change = angle_apex - angle_start
                    # Handle wraparound
                    if angle_change > 180:
                        angle_change -= 360
                    elif angle_change < -180:
                        angle_change += 360
                    features['head_angle_change'] = float(angle_change)
                else:
                    features['head_angle_change'] = None
            else:
                features['head_angle_deg'] = None
                features['head_angle_change'] = None
        else:
            features['head_angle_deg'] = None
            features['head_angle_change'] = None

        return features

    def _compute_body_features_segment(
        self,
        df: pd.DataFrame,
        start_frame: int,
        end_frame: int,
        ruler_pixels: float
    ) -> Dict:
        """
        Compute body/posture features for a segment.

        Args:
            df: Full DLC dataframe
            start_frame: Start frame of segment
            end_frame: End frame of segment
            ruler_pixels: Pixels per ruler unit for this segment

        Returns:
            Dictionary of body feature values
        """
        seg_df = df.iloc[start_frame:end_frame + 1]
        mm_per_pixel = self.RULER_MM / ruler_pixels if ruler_pixels > 0 else 0

        features = {}

        # Head width (LeftEar to RightEar distance) - proxy for mouse size
        if 'LeftEar_x' in seg_df.columns and 'RightEar_x' in seg_df.columns:
            # Only use high-confidence frames
            good_frames = (seg_df['LeftEar_likelihood'] > 0.7) & (seg_df['RightEar_likelihood'] > 0.7)
            if good_frames.any():
                head_widths = np.sqrt(
                    (seg_df.loc[good_frames, 'RightEar_x'] - seg_df.loc[good_frames, 'LeftEar_x'])**2 +
                    (seg_df.loc[good_frames, 'RightEar_y'] - seg_df.loc[good_frames, 'LeftEar_y'])**2
                ) * mm_per_pixel
                features['mean_head_width_mm'] = float(head_widths.mean()) if len(head_widths) > 0 else None
            else:
                features['mean_head_width_mm'] = None
        else:
            features['mean_head_width_mm'] = None

        # Distance from nose to slit (BOXR) - how close to tray
        if 'Nose_x' in seg_df.columns and 'BOXR_x' in seg_df.columns:
            good_frames = (seg_df['Nose_likelihood'] > 0.7) & (seg_df['BOXR_likelihood'] > 0.7)
            if good_frames.any():
                distances = np.sqrt(
                    (seg_df.loc[good_frames, 'Nose_x'] - seg_df.loc[good_frames, 'BOXR_x'])**2 +
                    (seg_df.loc[good_frames, 'Nose_y'] - seg_df.loc[good_frames, 'BOXR_y'])**2
                ) * mm_per_pixel
                features['mean_nose_to_slit_mm'] = float(distances.mean()) if len(distances) > 0 else None
            else:
                features['mean_nose_to_slit_mm'] = None
        else:
            features['mean_nose_to_slit_mm'] = None

        # Nose height (Y position) - postural measure
        if 'Nose_y' in seg_df.columns:
            good_frames = seg_df['Nose_likelihood'] > 0.7
            if good_frames.any():
                features['mean_nose_height'] = float(seg_df.loc[good_frames, 'Nose_y'].mean())
            else:
                features['mean_nose_height'] = None
        else:
            features['mean_nose_height'] = None

        # Head angle (ear-ear line relative to horizontal)
        if 'LeftEar_x' in seg_df.columns and 'RightEar_x' in seg_df.columns:
            good_frames = (seg_df['LeftEar_likelihood'] > 0.7) & (seg_df['RightEar_likelihood'] > 0.7)
            if good_frames.any():
                dx = seg_df.loc[good_frames, 'RightEar_x'] - seg_df.loc[good_frames, 'LeftEar_x']
                dy = seg_df.loc[good_frames, 'RightEar_y'] - seg_df.loc[good_frames, 'LeftEar_y']
                angles = np.arctan2(dy, dx) * 180 / np.pi
                features['mean_head_angle_deg'] = float(angles.mean()) if len(angles) > 0 else None
                features['head_angle_variance'] = float(angles.var()) if len(angles) > 1 else None
            else:
                features['mean_head_angle_deg'] = None
                features['head_angle_variance'] = None
        else:
            features['mean_head_angle_deg'] = None
            features['head_angle_variance'] = None

        # Nose position variance (postural stability)
        if 'Nose_x' in seg_df.columns and 'Nose_y' in seg_df.columns:
            good_frames = seg_df['Nose_likelihood'] > 0.7
            if good_frames.any() and good_frames.sum() > 1:
                nose_x_var = seg_df.loc[good_frames, 'Nose_x'].var()
                nose_y_var = seg_df.loc[good_frames, 'Nose_y'].var()
                features['nose_position_variance'] = float(nose_x_var + nose_y_var)
            else:
                features['nose_position_variance'] = None
        else:
            features['nose_position_variance'] = None

        return features

    def _compute_attention_score(
        self,
        df: pd.DataFrame,
        start_frame: int,
        end_frame: int
    ) -> Optional[float]:
        """
        Compute attention score for a segment.

        Attention score = percentage of frames where mouse is attending to tray.
        From old ASPA: frames where Nose is visible and positioned at the tray.

        Uses BOXR (box right edge) as reference point since it's stationary.

        Args:
            df: Full DLC dataframe
            start_frame: Start frame of segment
            end_frame: End frame of segment

        Returns:
            Attention score as percentage (0-100), or None if cannot compute
        """
        seg_df = df.iloc[start_frame:end_frame + 1]

        # Get reference Y position from BOXR (stationary box point)
        if 'BOXR_y' not in seg_df.columns:
            return None

        ref_y = seg_df['BOXR_y'].median()
        if pd.isna(ref_y):
            return None

        # Count frames where nose is attending (high confidence + near tray)
        # Criteria from old ASPA:
        # - Nose_likelihood > 0.9 (high confidence)
        # - Nose_y > ref_y - 80 (nose positioned below reference - at the tray)
        attending_frames = seg_df[
            (seg_df['Nose_likelihood'] > 0.9) &
            (seg_df['Nose_y'] > ref_y - 80)
        ].shape[0]

        total_frames = len(seg_df)

        if total_frames == 0:
            return None

        attention_score = (attending_frames / total_frames) * 100.0
        return float(attention_score)

    def _compute_temporal_context(
        self,
        seg_data: Dict,
        reaches: List[Dict]
    ) -> Dict:
        """
        Compute temporal context features for a segment.

        Args:
            seg_data: Segment data with start/end frames
            reaches: List of reach dicts

        Returns:
            Dict with temporal context features
        """
        features = {}

        start_frame = seg_data['start_frame']
        end_frame = seg_data['end_frame']

        # Segment duration
        segment_frames = end_frame - start_frame + 1
        features['segment_duration_sec'] = float(segment_frames / self.FRAMERATE)

        # Time to first reach
        if reaches:
            first_reach_frame = reaches[0]['start_frame']
            time_to_first = (first_reach_frame - start_frame) / self.FRAMERATE
            features['time_to_first_reach_sec'] = float(time_to_first)
        else:
            features['time_to_first_reach_sec'] = None

        # Time to outcome (assume outcome happens at segment end)
        features['time_to_outcome_sec'] = features['segment_duration_sec']

        # Inter-reach intervals
        if len(reaches) >= 2:
            intervals = []
            for i in range(1, len(reaches)):
                prev_end = reaches[i-1]['end_frame']
                curr_start = reaches[i]['start_frame']
                interval_frames = curr_start - prev_end
                intervals.append(interval_frames / self.FRAMERATE)

            features['mean_inter_reach_interval_sec'] = float(np.mean(intervals))
        else:
            features['mean_inter_reach_interval_sec'] = None

        return features

    def _compute_pellet_positioning(
        self,
        df: pd.DataFrame,
        start_frame: int,
        first_reach_start: Optional[int],
        ruler_pixels: float
    ) -> Dict:
        """
        Compute pellet positioning metrics before first interaction.

        Measures how ideally positioned the pellet is relative to the slit opening
        before the mouse first reaches. Accounts for tray movements.

        Args:
            df: Full DLC dataframe
            start_frame: Segment start frame
            first_reach_start: Frame of first reach, or None if no reaches
            ruler_pixels: Ruler size in pixels

        Returns:
            Dict with pellet positioning features
        """
        features = {}
        mm_per_pixel = self.RULER_MM / ruler_pixels if ruler_pixels > 0 else 0

        # Use frames before first reach, or first 30 frames if no reaches
        if first_reach_start is not None:
            end_assessment = first_reach_start
        else:
            end_assessment = min(start_frame + 30, len(df) - 1)

        assessment_df = df.iloc[start_frame:end_assessment + 1]

        # Need pellet position and box opening (BOXR)
        if 'Pellet_x' not in assessment_df.columns or 'BOXR_x' not in assessment_df.columns:
            features['pellet_position_idealness'] = None
            features['pellet_lateral_offset_mm'] = None
            features['pellet_depth_offset_mm'] = None
            return features

        # Filter for good pellet tracking
        good_pellet = assessment_df['Pellet_likelihood'] > 0.7
        if not good_pellet.any():
            features['pellet_position_idealness'] = None
            features['pellet_lateral_offset_mm'] = None
            features['pellet_depth_offset_mm'] = None
            return features

        # Get pellet position (median over assessment period)
        pellet_x = assessment_df.loc[good_pellet, 'Pellet_x'].median()
        pellet_y = assessment_df.loc[good_pellet, 'Pellet_y'].median()

        # Get slit opening position (BOXR = right edge of box/slit)
        boxr_x = assessment_df['BOXR_x'].median()
        boxr_y = assessment_df['BOXR_y'].median()

        if pd.isna(pellet_x) or pd.isna(pellet_y) or pd.isna(boxr_x) or pd.isna(boxr_y):
            features['pellet_position_idealness'] = None
            features['pellet_lateral_offset_mm'] = None
            features['pellet_depth_offset_mm'] = None
            return features

        # Lateral offset (Y direction in image - perpendicular to reach direction)
        lateral_offset_px = abs(pellet_y - boxr_y)
        features['pellet_lateral_offset_mm'] = float(lateral_offset_px * mm_per_pixel)

        # Depth offset (X direction - along reach direction)
        # Ideal pellet is slightly past the slit (reachable but not too far)
        # Typical ideal range: 20-40 pixels past BOXR
        depth_offset_px = pellet_x - boxr_x
        ideal_depth_px = 30  # Middle of ideal range
        depth_deviation_px = abs(depth_offset_px - ideal_depth_px)
        features['pellet_depth_offset_mm'] = float(depth_deviation_px * mm_per_pixel)

        # Idealness score (0-1): Combined metric
        # Penalize lateral and depth deviations
        # Perfect position: lateral=0, depth=30px
        lateral_penalty = min(lateral_offset_px / 50, 1.0)  # Full penalty at 50px off
        depth_penalty = min(depth_deviation_px / 40, 1.0)   # Full penalty at 40px off ideal

        idealness = 1.0 - (0.6 * lateral_penalty + 0.4 * depth_penalty)
        features['pellet_position_idealness'] = float(max(0.0, idealness))

        return features

    def _compute_quality_metrics(
        self,
        df: pd.DataFrame,
        start_frame: int,
        end_frame: int
    ) -> Dict:
        """
        Compute data quality metrics for a segment.

        Args:
            df: Full DLC dataframe
            start_frame: Segment start frame
            end_frame: Segment end frame

        Returns:
            Dict with quality metrics
        """
        features = {}
        seg_df = df.iloc[start_frame:end_frame + 1]

        # Get all likelihood columns
        likelihood_cols = [col for col in seg_df.columns if col.endswith('_likelihood')]

        if not likelihood_cols:
            features['mean_tracking_quality'] = None
            features['tracking_dropout_frames'] = 0
            return features

        # Mean tracking quality across all bodyparts
        all_likelihoods = seg_df[likelihood_cols].values.flatten()
        features['mean_tracking_quality'] = float(np.mean(all_likelihoods))

        # Count frames where ANY tracked bodypart has low confidence
        frame_min_likelihood = seg_df[likelihood_cols].min(axis=1)
        dropout_frames = (frame_min_likelihood < 0.5).sum()
        features['tracking_dropout_frames'] = int(dropout_frames)

        return features

    # ------------------------------------------------------------------
    # Extended (v2) per-reach kinematic features.
    # See docs/REACH_KINEMATIC_DATA_DICTIONARY.md for the full spec.
    # ------------------------------------------------------------------

    PAW_POINTS = ('RightHand', 'RHLeft', 'RHRight', 'RHOut')
    PAW_COL_PREFIX = {
        'RightHand': 'righthand',
        'RHLeft': 'rhleft',
        'RHRight': 'rhright',
        'RHOut': 'rhout',
    }

    def _build_augmented_nose_relative_trajectory(
        self,
        df: pd.DataFrame,
        paw_point: str,
        start_frame: int,
        end_frame: int,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]:
        """
        Construct a nose-subtracted paw trajectory bracketed by synthetic slit anchors.

        Synthetic pre-anchor: midpoint of (Nose, BoxL) at start_frame, then nose-subtracted.
            (= (BoxL - Nose) / 2 in nose-relative coords)
        Synthetic post-anchor: midpoint of (BoxR, Nose) at end_frame, then nose-subtracted.
            (= (BoxR - Nose) / 2 in nose-relative coords)

        Returns:
            (x, y) nose-relative augmented trajectory arrays (length = real_frames + 2),
            (pre_x, pre_y) synthetic pre-anchor in nose-relative coords,
            (post_x, post_y) synthetic post-anchor in nose-relative coords.
        """
        real_df = df.iloc[start_frame:end_frame + 1]
        paw_x = real_df[f'{paw_point}_x'].values - real_df['Nose_x'].values
        paw_y = real_df[f'{paw_point}_y'].values - real_df['Nose_y'].values

        s_row = df.iloc[start_frame]
        pre_x = (s_row['BOXL_x'] - s_row['Nose_x']) / 2.0
        pre_y = (s_row['BOXL_y'] - s_row['Nose_y']) / 2.0

        e_row = df.iloc[end_frame]
        post_x = (e_row['BOXR_x'] - e_row['Nose_x']) / 2.0
        post_y = (e_row['BOXR_y'] - e_row['Nose_y']) / 2.0

        x = np.concatenate([[pre_x], paw_x, [post_x]])
        y = np.concatenate([[pre_y], paw_y, [post_y]])
        return x, y, (float(pre_x), float(pre_y)), (float(post_x), float(post_y))

    def _paw_trajectory_features(
        self,
        df: pd.DataFrame,
        paw_point: str,
        start_frame: int,
        end_frame: int,
        ruler_pixels: float,
    ) -> Dict[str, Optional[float]]:
        """
        Compute the per-paw-point trajectory feature group on the augmented,
        nose-relative trajectory. Image y increases downward, so reach extension
        out through the slit produces y > 0 in nose-relative coords.
        """
        prefix = self.PAW_COL_PREFIX[paw_point]
        mm_per_px = self.RULER_MM / ruler_pixels if ruler_pixels and ruler_pixels > 0 else 0.0

        x, y, (pre_x, pre_y), (post_x, post_y) = self._build_augmented_nose_relative_trajectory(
            df, paw_point, start_frame, end_frame
        )

        out: Dict[str, Optional[float]] = {}

        # Per-point apex: real frame within reach with maximum nose-relative y.
        real_y = y[1:-1]
        if real_y.size > 0 and not np.all(np.isnan(real_y)):
            apex_idx_real = int(np.nanargmax(real_y))
            apex_frame = int(start_frame + apex_idx_real)
            ext_px = float(real_y[apex_idx_real])
        else:
            apex_idx_real = None
            apex_frame = None
            ext_px = None

        out[f'{prefix}_apex_frame'] = apex_frame
        out[f'{prefix}_extension_past_nose_px'] = ext_px
        out[f'{prefix}_extension_past_nose_mm'] = ext_px * mm_per_px if ext_px is not None else None

        # Frame-to-frame segment lengths on the augmented trajectory.
        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.sqrt(dx ** 2 + dy ** 2)
        path_px = float(np.nansum(seg_len)) if seg_len.size else 0.0
        out[f'{prefix}_total_path_px'] = path_px if seg_len.size else None
        out[f'{prefix}_total_path_mm'] = path_px * mm_per_px if seg_len.size else None

        # Lateral spread = max(x) - min(x) on the augmented trajectory.
        if x.size > 1 and not np.all(np.isnan(x)):
            spread_px = float(np.nanmax(x) - np.nanmin(x))
        else:
            spread_px = None
        out[f'{prefix}_lateral_spread_px'] = spread_px
        out[f'{prefix}_lateral_spread_mm'] = spread_px * mm_per_px if spread_px is not None else None

        # Swept area: shoelace polygon on valid points.
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() >= 3:
            xv = x[valid]
            yv = y[valid]
            area_px = 0.5 * abs(
                np.dot(xv, np.roll(yv, 1)) - np.dot(yv, np.roll(xv, 1))
            )
            out[f'{prefix}_swept_area_px2'] = float(area_px)
            out[f'{prefix}_swept_area_mm2'] = float(area_px * (mm_per_px ** 2))
        else:
            out[f'{prefix}_swept_area_px2'] = None
            out[f'{prefix}_swept_area_mm2'] = None

        # Path directness = straight-line(pre, post) / total path length.
        straight_px = float(np.sqrt((post_x - pre_x) ** 2 + (post_y - pre_y) ** 2))
        if path_px > 0:
            out[f'{prefix}_path_directness'] = float(straight_px / path_px)
        else:
            out[f'{prefix}_path_directness'] = None

        # Motion smoothness = 1 / (1 + mean |jerk|), where jerk is the third diff.
        if x.size >= 4:
            jx = np.diff(np.diff(np.diff(x)))
            jy = np.diff(np.diff(np.diff(y)))
            jmag = np.sqrt(jx ** 2 + jy ** 2)
            mean_jerk = float(np.nanmean(jmag)) if jmag.size and not np.all(np.isnan(jmag)) else None
            if mean_jerk is not None:
                out[f'{prefix}_motion_smoothness'] = 1.0 / (1.0 + mean_jerk)
            else:
                out[f'{prefix}_motion_smoothness'] = None
        else:
            out[f'{prefix}_motion_smoothness'] = None

        # Lateral deviation: max perpendicular distance from any trajectory point
        # to the synthetic-pre to synthetic-post line.
        if straight_px > 0 and valid.any():
            ax_, ay_ = post_x - pre_x, post_y - pre_y
            cross = (x - pre_x) * ay_ - (y - pre_y) * ax_
            perp = np.abs(cross) / straight_px
            max_perp = float(np.nanmax(perp)) if not np.all(np.isnan(perp)) else None
        else:
            max_perp = None
        out[f'{prefix}_lateral_deviation_px'] = max_perp
        out[f'{prefix}_lateral_deviation_mm'] = max_perp * mm_per_px if max_perp is not None else None

        # Speeds. Mean speed = path_length / number_of_inter_frame_segments.
        if seg_len.size > 0 and not np.all(np.isnan(seg_len)):
            mean_speed = float(np.nansum(seg_len) / seg_len.size)
            peak_speed = float(np.nanmax(seg_len))
        else:
            mean_speed = None
            peak_speed = None
        out[f'{prefix}_mean_speed_px_per_frame'] = mean_speed
        out[f'{prefix}_mean_speed_mm_per_frame'] = mean_speed * mm_per_px if mean_speed is not None else None
        out[f'{prefix}_peak_speed_px_per_frame'] = peak_speed
        out[f'{prefix}_peak_speed_mm_per_frame'] = peak_speed * mm_per_px if peak_speed is not None else None

        # Apex speed: segment length straddling the per-point apex frame
        # (uses the segment immediately after the apex frame in the augmented array).
        apex_speed = None
        if apex_idx_real is not None:
            idx_aug = apex_idx_real + 1  # +1 to skip the synthetic pre-anchor
            if idx_aug < seg_len.size:
                v = seg_len[idx_aug]
                if not np.isnan(v):
                    apex_speed = float(v)
            elif idx_aug - 1 >= 0:
                v = seg_len[idx_aug - 1]
                if not np.isnan(v):
                    apex_speed = float(v)
        out[f'{prefix}_apex_speed_px_per_frame'] = apex_speed
        out[f'{prefix}_apex_speed_mm_per_frame'] = apex_speed * mm_per_px if apex_speed is not None else None

        return out

    def _cross_paw_timing(
        self,
        per_point_apex_frames: Dict[str, Optional[int]],
        df: pd.DataFrame,
        start_frame: int,
        end_frame: int,
    ) -> Dict[str, Optional[float]]:
        """Cross-paw apex timing and velocity-profile coordination."""
        out: Dict[str, Optional[float]] = {}

        valid = {p: f for p, f in per_point_apex_frames.items() if f is not None}
        if len(valid) >= 2:
            frames = list(valid.values())
            out['paw_apex_lead_frames'] = int(max(frames) - min(frames))
            leading_pt = min(valid, key=valid.get)
            out['paw_leading_point'] = self.PAW_COL_PREFIX[leading_pt]
        else:
            out['paw_apex_lead_frames'] = None
            out['paw_leading_point'] = None

        # Velocity-correlation: mean pairwise Pearson r of frame-to-frame speed
        # series across the four paw points (real frames only, no synthetic anchors).
        speed_series = []
        real_df = df.iloc[start_frame:end_frame + 1]
        nose_x = real_df['Nose_x'].values
        nose_y = real_df['Nose_y'].values
        for p in self.PAW_POINTS:
            px = real_df[f'{p}_x'].values - nose_x
            py = real_df[f'{p}_y'].values - nose_y
            if px.size < 2:
                continue
            sp = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)
            speed_series.append(sp)

        if len(speed_series) >= 2 and all(s.size >= 2 for s in speed_series):
            import itertools
            rs = []
            for a, b in itertools.combinations(speed_series, 2):
                m = ~(np.isnan(a) | np.isnan(b))
                if m.sum() >= 2:
                    sa = a[m]
                    sb = b[m]
                    if sa.std() > 0 and sb.std() > 0:
                        rs.append(float(np.corrcoef(sa, sb)[0, 1]))
            out['paw_velocity_correlation'] = float(np.mean(rs)) if rs else None
        else:
            out['paw_velocity_correlation'] = None

        return out

    def _paw_shape_features(
        self,
        df: pd.DataFrame,
        start_frame: int,
        apex_frame: Optional[int],
        end_frame: int,
        interaction_frame: Optional[int],
        ruler_pixels: float,
    ) -> Dict[str, Optional[float]]:
        """
        Inter-paw geometry: paw_width_proxy (RHLeft<->RHRight),
        paw_outline_area (4-point polygon, ordered by angle around centroid),
        paw_spread_max (max of all 6 pairwise distances).

        Each measure summarised at start/apex/end/contact and as max/min/mean/range
        over the reach. Operates on real frames only (no synthetic anchors).
        """
        prefix_map = {
            'paw_width_proxy': ('px', 'mm', 1),
            'paw_outline_area': ('px2', 'mm2', 2),
            'paw_spread_max': ('px', 'mm', 1),
        }
        out: Dict[str, Optional[float]] = {}

        seg = df.iloc[start_frame:end_frame + 1]
        n = len(seg)
        if n == 0:
            for measure, (raw_u, mm_u, _) in prefix_map.items():
                for sfx in ('at_start', 'at_apex', 'at_end', 'at_contact', 'max', 'min', 'mean', 'range'):
                    out[f'{measure}_{sfx}_{raw_u}'] = None
                    out[f'{measure}_{sfx}_{mm_u}'] = None
            return out

        mm_per_px = self.RULER_MM / ruler_pixels if ruler_pixels and ruler_pixels > 0 else 0.0

        # Per-frame width proxy: distance(RHLeft, RHRight)
        width = np.sqrt(
            (seg['RHLeft_x'].values - seg['RHRight_x'].values) ** 2
            + (seg['RHLeft_y'].values - seg['RHRight_y'].values) ** 2
        )

        # Per-frame paw spread max (max pairwise distance among the 4 points).
        import itertools
        pts = [(seg[f'{p}_x'].values, seg[f'{p}_y'].values) for p in self.PAW_POINTS]
        pair_dists = []
        for (ax, ay), (bx, by) in itertools.combinations(pts, 2):
            pair_dists.append(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))
        spread = np.nanmax(np.stack(pair_dists, axis=0), axis=0)

        # Per-frame outline area: shoelace polygon over the 4 paw points,
        # ordered by angle around their per-frame centroid.
        outline = np.full(n, np.nan)
        all_x = np.stack([p[0] for p in pts], axis=0)  # (4, n)
        all_y = np.stack([p[1] for p in pts], axis=0)
        for i in range(n):
            xi = all_x[:, i]
            yi = all_y[:, i]
            if np.any(np.isnan(xi)) or np.any(np.isnan(yi)):
                continue
            cx = xi.mean()
            cy = yi.mean()
            ang = np.arctan2(yi - cy, xi - cx)
            order = np.argsort(ang)
            xo = xi[order]
            yo = yi[order]
            outline[i] = 0.5 * abs(
                np.dot(xo, np.roll(yo, 1)) - np.dot(yo, np.roll(xo, 1))
            )

        def _frame_idx_relative(absolute_frame: Optional[int]) -> Optional[int]:
            if absolute_frame is None:
                return None
            idx = absolute_frame - start_frame
            if 0 <= idx < n:
                return idx
            return None

        anchors = {
            'at_start': 0,
            'at_apex': _frame_idx_relative(apex_frame),
            'at_end': n - 1,
            'at_contact': _frame_idx_relative(interaction_frame),
        }

        def _summarize(measure: str, arr: np.ndarray, raw_u: str, mm_u: str, exp: int):
            mm_factor = mm_per_px ** exp
            for sfx, idx in anchors.items():
                if idx is None or idx >= arr.size:
                    out[f'{measure}_{sfx}_{raw_u}'] = None
                    out[f'{measure}_{sfx}_{mm_u}'] = None
                else:
                    v = arr[idx]
                    if np.isnan(v):
                        out[f'{measure}_{sfx}_{raw_u}'] = None
                        out[f'{measure}_{sfx}_{mm_u}'] = None
                    else:
                        out[f'{measure}_{sfx}_{raw_u}'] = float(v)
                        out[f'{measure}_{sfx}_{mm_u}'] = float(v * mm_factor)
            if arr.size and not np.all(np.isnan(arr)):
                vmax = float(np.nanmax(arr))
                vmin = float(np.nanmin(arr))
                vmean = float(np.nanmean(arr))
                out[f'{measure}_max_{raw_u}'] = vmax
                out[f'{measure}_max_{mm_u}'] = vmax * mm_factor
                out[f'{measure}_min_{raw_u}'] = vmin
                out[f'{measure}_min_{mm_u}'] = vmin * mm_factor
                out[f'{measure}_mean_{raw_u}'] = vmean
                out[f'{measure}_mean_{mm_u}'] = vmean * mm_factor
                out[f'{measure}_range_{raw_u}'] = vmax - vmin
                out[f'{measure}_range_{mm_u}'] = (vmax - vmin) * mm_factor
            else:
                for stat in ('max', 'min', 'mean', 'range'):
                    out[f'{measure}_{stat}_{raw_u}'] = None
                    out[f'{measure}_{stat}_{mm_u}'] = None

        _summarize('paw_width_proxy', width, 'px', 'mm', 1)
        _summarize('paw_outline_area', outline, 'px2', 'mm2', 2)
        _summarize('paw_spread_max', spread, 'px', 'mm', 1)

        return out

    def _paw_visibility_features(
        self,
        df: pd.DataFrame,
        start_frame: int,
        apex_frame: Optional[int],
        end_frame: int,
        interaction_frame: Optional[int],
    ) -> Dict[str, Optional[float]]:
        """
        Per-paw-point likelihood profile across the reach. Likelihood is
        dimensionless (0-1); reported as visibility because dropouts often
        carry behavioural signal (rotation, grasp closure, occlusion).
        """
        seg = df.iloc[start_frame:end_frame + 1]
        n = len(seg)
        out: Dict[str, Optional[float]] = {}

        def _frame_idx_relative(absolute_frame: Optional[int]) -> Optional[int]:
            if absolute_frame is None:
                return None
            idx = absolute_frame - start_frame
            if 0 <= idx < n:
                return idx
            return None

        anchors = {
            'at_start': 0 if n > 0 else None,
            'at_apex': _frame_idx_relative(apex_frame),
            'at_end': (n - 1) if n > 0 else None,
            'at_contact': _frame_idx_relative(interaction_frame),
        }

        for paw in self.PAW_POINTS:
            prefix = self.PAW_COL_PREFIX[paw]
            lk = seg[f'{paw}_likelihood'].values

            for sfx, idx in anchors.items():
                if idx is None or idx >= lk.size or np.isnan(lk[idx]):
                    out[f'{prefix}_visibility_{sfx}'] = None
                else:
                    out[f'{prefix}_visibility_{sfx}'] = float(lk[idx])

            if lk.size and not np.all(np.isnan(lk)):
                out[f'{prefix}_visibility_max'] = float(np.nanmax(lk))
                out[f'{prefix}_visibility_min'] = float(np.nanmin(lk))
                out[f'{prefix}_visibility_mean'] = float(np.nanmean(lk))
                out[f'{prefix}_visibility_range'] = float(np.nanmax(lk) - np.nanmin(lk))
            else:
                out[f'{prefix}_visibility_max'] = None
                out[f'{prefix}_visibility_min'] = None
                out[f'{prefix}_visibility_mean'] = None
                out[f'{prefix}_visibility_range'] = None

        # Aggregate: frames where the minimum likelihood across all 4 paw points < 0.5.
        if n > 0:
            paw_lk = np.stack([seg[f'{p}_likelihood'].values for p in self.PAW_POINTS], axis=0)
            min_per_frame = np.nanmin(paw_lk, axis=0)
            out['frames_any_paw_low_confidence'] = int(np.nansum(min_per_frame < 0.5))
        else:
            out['frames_any_paw_low_confidence'] = 0

        return out

    def _tray_contact_duration(
        self,
        df: pd.DataFrame,
        start_frame: int,
        end_frame: int,
    ) -> Optional[int]:
        """
        Count of frames within the reach window where pellet/tray motion is
        detected while the wrist (RightHand) is not confidently tracked.

        Mirrors ASPA's tray_motion: pellet x rolling-std > 2 px AND wrist
        likelihood < 0.5 AND pillar likelihood < 0.5.
        """
        seg = df.iloc[start_frame:end_frame + 1]
        if len(seg) < 4:
            return 0

        try:
            std_x = seg['Pellet_x'].rolling(4).std().values
            motion = (
                (std_x > 2)
                & (seg['Pillar_likelihood'].values < 0.5)
                & (seg['RightHand_likelihood'].values < 0.5)
            )
            return int(np.nansum(motion))
        except KeyError:
            return None

    def _extended_features(
        self,
        df: pd.DataFrame,
        start_frame: int,
        apex_frame: Optional[int],
        end_frame: int,
        interaction_frame: Optional[int],
        ruler_pixels: float,
    ) -> Dict[str, Optional[float]]:
        """Run all extended-feature groups and return a flat dict."""
        ext: Dict[str, Optional[float]] = {}

        # Per-paw-point trajectory features (4 points x ~14 columns each)
        per_point_apex: Dict[str, Optional[int]] = {}
        for p in self.PAW_POINTS:
            point_features = self._paw_trajectory_features(
                df, p, start_frame, end_frame, ruler_pixels
            )
            ext.update(point_features)
            per_point_apex[p] = point_features[f'{self.PAW_COL_PREFIX[p]}_apex_frame']

        # Cross-paw timing and coordination
        ext.update(self._cross_paw_timing(per_point_apex, df, start_frame, end_frame))

        # Inter-paw shape proxies
        ext.update(self._paw_shape_features(
            df, start_frame, apex_frame, end_frame, interaction_frame, ruler_pixels
        ))

        # Per-paw visibility profile
        ext.update(self._paw_visibility_features(
            df, start_frame, apex_frame, end_frame, interaction_frame
        ))

        # Tray contact
        ext['tray_contact_duration_frames'] = self._tray_contact_duration(
            df, start_frame, end_frame
        )

        return ext

    def _compute_summary(self, segment_features: List[SegmentFeatures]) -> Dict:
        """Compute summary statistics across all reaches."""

        all_reaches = []
        causal_reaches = []

        for seg in segment_features:
            for reach in seg.reaches:
                all_reaches.append(reach)
                if reach.causal_reach:
                    causal_reaches.append(reach)

        if len(all_reaches) == 0:
            return {}

        # Collect features
        extents = [r.max_extent_mm for r in all_reaches if r.max_extent_mm is not None]
        durations = [r.duration_frames for r in all_reaches]
        velocities = [r.peak_velocity_px_per_frame for r in all_reaches if r.peak_velocity_px_per_frame is not None]

        # Collect attention scores
        attention_scores = [seg.attention_score for seg in segment_features if seg.attention_score is not None]

        summary = {
            'total_reaches': len(all_reaches),
            'causal_reaches': len(causal_reaches),
            'mean_extent_mm': float(np.mean(extents)) if extents else None,
            'mean_duration_frames': float(np.mean(durations)) if durations else None,
            'mean_peak_velocity': float(np.mean(velocities)) if velocities else None,
            'mean_attention_score': float(np.mean(attention_scores)) if attention_scores else None,
        }

        # Outcome breakdown
        outcome_counts = {}
        for seg in segment_features:
            outcome = seg.outcome
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        summary['outcome_counts'] = outcome_counts

        return summary

    def _load_dlc(self, dlc_path: Path) -> pd.DataFrame:
        """Load DLC data and flatten column names."""
        # Try HDF5 first (superior format - faster, smaller), fall back to CSV
        try:
            df = pd.read_hdf(dlc_path)
            df.columns = ['_'.join(col[1:]) for col in df.columns]
            return df
        except Exception as hdf5_error:
            # Fall back to CSV if HDF5 fails (e.g., tables not installed)
            csv_path = dlc_path.with_suffix('.csv')
            if csv_path.exists():
                df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
                df.columns = ['_'.join(col[1:]) for col in df.columns]
                return df
            else:
                raise RuntimeError(
                    f"Could not load DLC data from HDF5 ({dlc_path.name}) or CSV ({csv_path.name}). "
                    f"HDF5 error: {hdf5_error}"
                )
