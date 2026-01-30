"""
pellet_outcome.py - Pellet outcome classification algorithm

ALGORITHM SUMMARY
=================
Classifies the outcome of each pellet presentation (trial) into categories:
retrieved, displaced (within or outside scoring area), or untouched. Uses
geometric tracking of pellet position relative to the pillar throughout each
trial segment.

SCIENTIFIC DESCRIPTION
======================
After each pellet is presented, the mouse may: (1) successfully retrieve and
eat the pellet, (2) displace the pellet without eating it, or (3) leave it
untouched. This algorithm tracks the pellet's position in ruler units relative
to the calculated pillar position (geometric center derived from SABL/SABR
anchor points). Classification is based on pellet visibility changes and
displacement magnitude between segment start and end.

INPUT REQUIREMENTS
==================
- DeepLabCut tracking file (.h5) with 18 bodyparts
- Segmentation boundaries file (_segments.json)
- Reach detection results (_reaches.json) for causal reach attribution
- Required bodyparts:
  - Target tracking: Pellet, Pillar
  - Calibration: SABL, SABR (for ruler-unit conversion and pillar geometry)
  - Scoring area bounds: SATL, SATR, SABL, SABR

DETECTION RULES
===============
1. Retrieved (R)
   - Pellet visible at segment start, disappears before segment end
   - Last visible position is near pillar (within ON_PILLAR_THRESHOLD)
   - Indicates successful grasp and consumption

2. Displaced in Scoring Area (D)
   - Pellet remains visible throughout segment
   - Final position moved > DISPLACED_THRESHOLD from pillar
   - Final position still within scoring area bounds

3. Displaced Outside (O)
   - Pellet remains visible but moves outside scoring area bounds
   - OR pellet disappears at a location away from pillar

4. Untouched (U)
   - Pellet position unchanged from segment start to end
   - Movement < DISPLACED_THRESHOLD throughout
   - Indicates no successful reach interaction

5. Uncertain
   - Ambiguous tracking (low likelihood, conflicting signals)
   - Pellet tracking quality too poor for confident classification

KEY PARAMETERS
==============
| Parameter | Value | Unit | Rationale |
|-----------|-------|------|-----------|
| ON_PILLAR_THRESHOLD | 0.20 | ruler | ~1.8mm, defines "on pillar" region |
| DISPLACED_THRESHOLD | 0.25 | ruler | ~2.25mm, minimum motion for "displaced" |
| PILLAR_PERP_DISTANCE | 0.944 | ruler | Geometric pillar-to-SA distance |
| CONFIDENCE_THRESHOLD | 0.6 | DLC | Minimum likelihood for pellet tracking |
| SA_BOUNDARY_MARGIN | 0.1 | ruler | Tolerance for "inside SA" determination |

OUTPUT FORMAT
=============
JSON file (*_pellet_outcomes.json) containing:
- Per-segment outcome classification
- Confidence score for each classification
- Interaction frame (first pellet touch)
- Causal reach ID (which reach caused displacement/retrieval)
- Human verification flags and correction tracking
- validation_status: "auto_approved", "needs_review", or "validated"

VALIDATION HISTORY
==================
- v1.0: Initial geometric classification
- v2.0: Added pillar geometry calculation, improved thresholds
- v2.3: Refined SA boundary detection, added causal reach attribution

KNOWN LIMITATIONS
=================
- Pellet tracking quality degrades when occluded by paw
- "Retrieved" requires pellet to disappear (may miss if eaten out of frame)
- Cannot distinguish pellet knocked by paw vs. pellet rolled naturally
- Multiple reaches in same segment may have ambiguous causal attribution

REFERENCES
==========
- Pillar geometry: Derived from Pillar_Tray.stl measurements
- Scoring area: SABL-SABR-SATL-SATR quadrilateral
- Ground truth: outcomes/review_widget.py for manual classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import json
from datetime import datetime

from .geometry import (
    compute_segment_geometry, SegmentGeometry,
    load_dlc, load_segments
)
from mousereach.config import Thresholds


VERSION = "2.4.4"  # v2.4.4: Better handling of pellet start/end position for retrieval vs displacement


@dataclass
class PelletOutcome:
    """Outcome for one pellet (one segment)"""
    segment_num: int
    
    # Outcome classification
    # retrieved | displaced_sa | displaced_outside | untouched | uncertain | no_pellet
    outcome: str

    # Timing information (auto-detected, can be corrected by annotator)
    interaction_frame: Optional[int] = None      # First pellet touch: eating frame for retrieved, causal reach for displaced, None if untouched
    outcome_known_frame: Optional[int] = None    # First moment outcome determinable (set manually by annotator)
    
    # Algorithm-computed pellet state (for reference)
    pellet_visible_start: bool = True
    distance_from_pillar_start: Optional[float] = None  # Distance from geometric pillar position (ruler units)
    pellet_visible_end: bool = True
    distance_from_pillar_end: Optional[float] = None  # Distance from geometric pillar position (ruler units)
    
    # Causal reach
    causal_reach_id: Optional[int] = None
    causal_reach_frame: Optional[int] = None
    
    # Confidence and review tracking
    confidence: float = 0.0
    human_verified: bool = False
    original_outcome: Optional[str] = None  # What algo said, if changed
    flagged_for_review: bool = False
    flag_reason: Optional[str] = None


@dataclass
class VideoOutcomes:
    """Complete pellet outcome results for one video"""
    detector_version: str
    video_name: str
    total_frames: int
    n_segments: int
    
    segments: List[PelletOutcome]
    summary: Dict
    detected_at: str
    
    # Validation tracking
    validated: bool = False
    validated_by: Optional[str] = None
    validated_at: Optional[str] = None
    corrections_made: int = 0
    segments_flagged: int = 0


class PelletOutcomeDetector:
    """Detects pellet outcomes using per-segment geometry."""
    
    PELLET_LIKELIHOOD_THRESHOLD = 0.5
    SA_LIKELIHOOD_THRESHOLD = 0.8
    RH_LIKELIHOOD_THRESHOLD = 0.5
    
    # Thresholds in ruler units (1 ruler = SABL-SABR = 9mm)
    ON_PILLAR_THRESHOLD = 0.20  # ~1.8mm - pellet considered on pillar
    DISPLACED_THRESHOLD = 0.25  # ~2.25mm - pellet definitely off pillar (adjusted from 0.40 based on ground truth analysis)
    
    # Geometry: pillar is 0.944 ruler from SABL-SABR midpoint (55° isosceles triangle)
    PILLAR_PERP_DISTANCE = 0.944
    
    def __init__(self):
        pass
    
    def compute_expected_pillar(self, df: pd.DataFrame, seg_start: int, seg_end: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute expected pillar position from SABL/SABR geometry.
        Returns (pillar_x, pillar_y, ruler) or (None, None, None) if can't compute.
        """
        seg_df = df.iloc[seg_start:seg_end]
        
        # Get stable SABL/SABR positions
        good_sa = seg_df[(seg_df['SABL_likelihood'] > self.SA_LIKELIHOOD_THRESHOLD) & 
                         (seg_df['SABR_likelihood'] > self.SA_LIKELIHOOD_THRESHOLD)]
        
        if len(good_sa) < 10:
            return None, None, None
        
        sabl_x = good_sa['SABL_x'].median()
        sabl_y = good_sa['SABL_y'].median()
        sabr_x = good_sa['SABR_x'].median()
        sabr_y = good_sa['SABR_y'].median()
        
        # Ruler = SABL-SABR distance
        ruler = np.sqrt((sabr_x - sabl_x)**2 + (sabr_y - sabl_y)**2)
        if ruler < 1:
            return None, None, None
        
        # Midpoint
        mid_x = (sabl_x + sabr_x) / 2
        mid_y = (sabl_y + sabr_y) / 2
        
        # Perpendicular direction (toward SAT/pillar, which is lower Y in image coords)
        dx = sabr_x - sabl_x
        dy = sabr_y - sabl_y
        perp_x = dy / ruler
        perp_y = -dx / ruler
        
        # Pillar position
        perp_dist = self.PILLAR_PERP_DISTANCE * ruler
        pillar_x = mid_x + perp_x * perp_dist
        pillar_y = mid_y + perp_y * perp_dist
        
        return pillar_x, pillar_y, ruler
    
    def get_pellet_trajectory(
        self,
        df: pd.DataFrame,
        seg_start: int,
        seg_end: int,
        ruler: float
    ) -> Dict[str, Any]:
        """
        Track pellet position relative to SA reference points FRAME BY FRAME.
        
        Key insight: The tray moves during segments, so we must compute pellet
        position relative to SABL/SABR at each frame, not use a fixed baseline.
        """
        seg_df = df.iloc[seg_start:seg_end]
        
        # Need frames where both pellet AND SA points are visible
        good_frames = seg_df[
            (seg_df['Pellet_likelihood'] > self.PELLET_LIKELIHOOD_THRESHOLD) &
            (seg_df['SABL_likelihood'] > self.SA_LIKELIHOOD_THRESHOLD) &
            (seg_df['SABR_likelihood'] > self.SA_LIKELIHOOD_THRESHOLD)
        ]
        
        if len(good_frames) < 20:
            return {
                'visible': len(good_frames) > 0,
                'visibility_pct': len(good_frames) / len(seg_df) if len(seg_df) > 0 else 0,
                'start_distance_from_pillar': None,
                'end_distance_from_pillar': None,
                'max_distance_from_pillar': None,
                'wobble_pattern_detected': False,
                'pellet_frames': good_frames.index.tolist() if len(good_frames) > 0 else [],
                'distance_array': np.array([]),  # Empty array for insufficient data
                'pellet_confidence': np.array([])  # Empty array for insufficient data
            }
        
        # Compute pellet position relative to SA midpoint for each frame
        # This handles tray movement automatically
        pellet_x = good_frames['Pellet_x'].values
        pellet_y = good_frames['Pellet_y'].values
        pellet_likelihood = good_frames['Pellet_likelihood'].values  # Extract likelihood array
        sabl_x = good_frames['SABL_x'].values
        sabl_y = good_frames['SABL_y'].values
        sabr_x = good_frames['SABR_x'].values
        sabr_y = good_frames['SABR_y'].values
        
        # SA midpoint per frame
        mid_x = (sabl_x + sabr_x) / 2
        mid_y = (sabl_y + sabr_y) / 2
        
        # Ruler per frame (for normalization)
        frame_rulers = np.sqrt((sabr_x - sabl_x)**2 + (sabr_y - sabl_y)**2)
        frame_rulers = np.clip(frame_rulers, 1, None)  # Avoid div by zero
        
        # Pellet position relative to SA midpoint (in ruler units)
        rel_x = (pellet_x - mid_x) / frame_rulers
        rel_y = (pellet_y - mid_y) / frame_rulers
        
        # Geometric pillar position (relative to SA midpoint)
        # Pillar is at SA midpoint horizontally, 0.944 ruler units "above" (negative Y in image coords)
        # This is the 55° isosceles triangle apex position
        pillar_rel_x_expected = 0.0
        pillar_rel_y_expected = -0.944

        # Distance from pellet to expected pillar position (frame-by-frame)
        pellet_to_pillar_dist = np.sqrt(
            (rel_x - pillar_rel_x_expected)**2 +
            (rel_y - pillar_rel_y_expected)**2
        )

        # Start: Is pellet on pillar initially? (first 10 frames median)
        start_distance = np.median(pellet_to_pillar_dist[:10])

        # End: Is pellet still on pillar? (last 10 frames median)
        end_distance = np.median(pellet_to_pillar_dist[-10:])

        # Max: Did pellet ever leave pillar during segment?
        max_distance = float(pellet_to_pillar_dist.max())
        
        # Also track visibility relative to total segment
        all_pellet = seg_df[seg_df['Pellet_likelihood'] > self.PELLET_LIKELIHOOD_THRESHOLD]
        visibility_pct = len(all_pellet) / len(seg_df)
        
        # Pattern detection: Did pellet wobble then return?
        wobble_pattern = (max_distance > 0.4 and end_distance < 0.20)

        return {
            'visible': True,
            'visibility_pct': visibility_pct,
            'start_distance_from_pillar': float(start_distance),
            'end_distance_from_pillar': float(end_distance),
            'max_distance_from_pillar': float(max_distance),
            'wobble_pattern_detected': wobble_pattern,
            'pellet_frames': good_frames.index.tolist(),
            'distance_array': pellet_to_pillar_dist,  # Full frame-by-frame distance for detailed analysis
            'pellet_confidence': pellet_likelihood  # Full confidence array
        }
    
    def detect_pellet_movement_direction(
        self,
        df: pd.DataFrame,
        pellet_frames: List[int],
        distance_array: np.ndarray,
        seg_start: int,
        seg_end: int
    ) -> Tuple[str, float]:
        """
        Detect direction of pellet movement before disappearance.

        v2.4: Added based on ground truth analysis showing retrievals
        are often missed when pellet moves toward box (not SA) before disappearing.

        Returns:
            (direction, confidence) where direction is:
            - 'toward_box': pellet moved right/up (toward box/mouse)
            - 'toward_sa': pellet moved left/down (into scoring area)
            - 'stationary': pellet didn't move significantly
            - 'unknown': insufficient data
        """
        if len(pellet_frames) < 20 or len(distance_array) < 20:
            return 'unknown', 0.0

        # Get box reference (slit position) for direction determination
        seg_df = df.iloc[seg_start:seg_end]
        boxr_x = seg_df['BOXR_x'].median() if 'BOXR_x' in seg_df.columns else None

        if boxr_x is None:
            return 'unknown', 0.0

        # Look at last 30 frames of pellet visibility
        last_n = min(30, len(pellet_frames))
        last_frames = pellet_frames[-last_n:]

        # Get pellet positions for these frames
        pellet_x_vals = []
        pellet_y_vals = []

        for frame_idx in last_frames:
            if frame_idx < len(df):
                row = df.iloc[frame_idx]
                if row.get('Pellet_likelihood', 0) > self.PELLET_LIKELIHOOD_THRESHOLD:
                    pellet_x_vals.append(row.get('Pellet_x', np.nan))
                    pellet_y_vals.append(row.get('Pellet_y', np.nan))

        if len(pellet_x_vals) < 10:
            return 'unknown', 0.0

        # Calculate movement direction (first half to second half)
        half = len(pellet_x_vals) // 2
        first_half_x = np.nanmean(pellet_x_vals[:half])
        second_half_x = np.nanmean(pellet_x_vals[half:])
        first_half_y = np.nanmean(pellet_y_vals[:half])
        second_half_y = np.nanmean(pellet_y_vals[half:])

        dx = second_half_x - first_half_x
        dy = second_half_y - first_half_y

        # Positive dx = moving right (toward box)
        # Negative dx = moving left (into SA)
        movement_magnitude = np.sqrt(dx**2 + dy**2)

        if movement_magnitude < Thresholds.STATIONARY_MOVEMENT_THRESHOLD:
            return 'stationary', 0.9

        # Direction relative to box
        if dx > Thresholds.TOWARD_BOX_THRESHOLD:  # Moving right toward box
            return 'toward_box', min(0.9, movement_magnitude / 20.0)
        elif dx < -Thresholds.TOWARD_SA_THRESHOLD:  # Moving left into SA
            return 'toward_sa', min(0.9, movement_magnitude / 20.0)
        else:
            return 'stationary', 0.7

    def detect_pellet_grab(
        self,
        df: pd.DataFrame,
        seg_start: int,
        seg_end: int
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """
        v2.4.4: Detect pellet being grabbed by paw (retrieval pattern).

        Pattern: pellet visible -> visibility drops while paw is near -> pellet stays invisible

        This catches quick retrievals that don't show as "sustained displacement"
        because the pellet disappears immediately when grabbed.

        Returns (grab_detected, grab_frame, distance_from_pillar_at_grab).
        """
        seg_df = df.iloc[seg_start:seg_end]
        paw_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

        # Look for sudden pellet visibility drops
        pellet_conf = seg_df['Pellet_likelihood'].values

        for i in range(Thresholds.OUTCOME_SKIP_START_FRAMES, len(pellet_conf) - Thresholds.OUTCOME_SKIP_END_FRAMES):
            # Check for visibility drop: high before, low after
            before_avg = pellet_conf[max(0, i-10):i].mean()
            at_frame = pellet_conf[i]
            after_avg = pellet_conf[i+5:i+30].mean()

            if before_avg > 0.8 and at_frame < 0.3 and after_avg < 0.3:
                # Sudden visibility drop found - check if paw was near
                frame_idx = seg_start + i
                row = df.iloc[frame_idx]

                # Get last known pellet position (from just before drop)
                pellet_row = df.iloc[frame_idx - 3]
                if pellet_row.get('Pellet_likelihood', 0) < 0.5:
                    continue

                pellet_x = pellet_row.get('Pellet_x', np.nan)
                pellet_y = pellet_row.get('Pellet_y', np.nan)

                if np.isnan([pellet_x, pellet_y]).any():
                    continue

                # Check if any paw point is near pellet position at drop time
                min_dist = float('inf')
                for paw in paw_points:
                    paw_conf = row.get(f'{paw}_likelihood', 0)
                    if paw_conf < 0.5:
                        continue
                    paw_x = row.get(f'{paw}_x', np.nan)
                    paw_y = row.get(f'{paw}_y', np.nan)
                    if np.isnan([paw_x, paw_y]).any():
                        continue

                    dist = np.sqrt((paw_x - pellet_x)**2 + (paw_y - pellet_y)**2)
                    min_dist = min(min_dist, dist)

                # If paw was near pellet when it disappeared -> grab detected
                if min_dist < Thresholds.PAW_PROXIMITY_THRESHOLD:
                    # v2.4.4: Calculate distance from pillar at grab time
                    pillar_x, pillar_y, ruler_scale = self.compute_expected_pillar(df, seg_start, seg_end)
                    grab_dist_from_pillar = None
                    if pillar_x is not None and ruler_scale is not None and ruler_scale > 0:
                        pixel_dist = np.sqrt((pellet_x - pillar_x)**2 + (pellet_y - pillar_y)**2)
                        grab_dist_from_pillar = pixel_dist / ruler_scale
                    return True, frame_idx, grab_dist_from_pillar

        return False, None, None

    def detect_eating_signature(
        self,
        df: pd.DataFrame,
        seg_start: int,
        seg_end: int
    ) -> Tuple[bool, Optional[int]]:
        """
        Enhanced eating detection using multiple signals.

        Eating behavior:
        - Nose retreats from slit (higher X)
        - Hand near nose/mouth
        - Sustained behavior (30+ frames)
        - Pellet disappears during this period (strong signal)

        Returns (eating_detected, eating_start_frame).
        """
        seg_df = df.iloc[seg_start:seg_end]

        # Get box reference (slit position)
        boxr_x = seg_df['BOXR_x'].median() if 'BOXR_x' in seg_df.columns else None
        if boxr_x is None:
            return False, None

        eating_frames = []

        for frame_idx in range(seg_start + 30, seg_end):  # Skip first 30 frames
            row = df.iloc[frame_idx]

            # Nose position and retreat
            nose_conf = row.get('Nose_likelihood', 0)
            if nose_conf < 0.5:
                continue

            nose_x = row['Nose_x']
            nose_retreated = nose_x > boxr_x + 30  # Pulled back from slit

            if not nose_retreated:
                continue

            # Check if any RH point is near nose (eating motion)
            rh_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
            hand_near_mouth = False

            for rh in rh_points:
                rh_conf = row.get(f'{rh}_likelihood', 0)
                if rh_conf < 0.5:
                    continue

                rh_x = row[f'{rh}_x']
                rh_y = row[f'{rh}_y']
                nose_y = row['Nose_y']

                # RH close to nose
                rh_nose_dist = np.sqrt((rh_x - nose_x)**2 + (rh_y - nose_y)**2)
                if rh_nose_dist < Thresholds.EATING_DISTANCE_THRESHOLD:
                    hand_near_mouth = True
                    break

            if hand_near_mouth:
                eating_frames.append(frame_idx)

        # Need sustained eating behavior (at least 30 frames)
        if len(eating_frames) >= 30:
            # Additional check: Did pellet visibility drop during eating period?
            # This is a strong signal of actual consumption
            eating_start = eating_frames[0]
            eating_end = eating_frames[-1]

            eating_period = seg_df.iloc[eating_start - seg_start:eating_end - seg_start]
            pellet_visible_during_eating = (eating_period['Pellet_likelihood'] > 0.5).mean()

            # If pellet disappeared during eating, we're very confident
            # (This boosts confidence in the calling function, but doesn't change detection result)

            return True, eating_frames[0]

        return False, None
    
    def check_paw_proximity(
        self,
        df: pd.DataFrame,
        pellet_frames: List[int],
        before_frame: int,
        lookback_frames: int = 30,
        proximity_threshold_pixels: float = 50.0
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if any paw keypoint was near pellet before displacement.

        Returns:
            (paw_was_near, frame_of_closest_approach)
        """
        # Paw keypoints to check
        paw_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

        # Look at frames before displacement
        lookback_start = max(0, before_frame - lookback_frames)
        lookback_end = before_frame

        min_distance = float('inf')
        closest_frame = None

        for frame_idx in range(lookback_start, lookback_end):
            if frame_idx >= len(df):
                continue

            row = df.iloc[frame_idx]

            # Get pellet position
            pellet_conf = row.get('Pellet_likelihood', 0)
            if pellet_conf < self.PELLET_LIKELIHOOD_THRESHOLD:
                continue

            pellet_x = row.get('Pellet_x', np.nan)
            pellet_y = row.get('Pellet_y', np.nan)

            if np.isnan([pellet_x, pellet_y]).any():
                continue

            # Check all paw points
            for paw in paw_points:
                paw_conf = row.get(f'{paw}_likelihood', 0)
                if paw_conf < self.RH_LIKELIHOOD_THRESHOLD:
                    continue

                paw_x = row.get(f'{paw}_x', np.nan)
                paw_y = row.get(f'{paw}_y', np.nan)

                if np.isnan([paw_x, paw_y]).any():
                    continue

                # Distance from paw to pellet
                dist = np.sqrt((paw_x - pellet_x)**2 + (paw_y - pellet_y)**2)

                if dist < min_distance:
                    min_distance = dist
                    closest_frame = frame_idx

        # Was paw close enough to cause displacement?
        paw_was_near = min_distance < proximity_threshold_pixels

        return paw_was_near, closest_frame

    def detect_sustained_displacement(
        self,
        distance_array: np.ndarray,
        pellet_conf: np.ndarray,
        threshold: float = 0.30,
        min_duration: int = 10
    ) -> Tuple[bool, int, float, int]:
        """
        Detect if pellet had sustained displacement (not just momentary jitter).

        Returns:
            (is_displaced, displacement_sustained_idx, max_dist_during_displacement, displacement_onset_idx)

        displacement_sustained_idx: First frame where displacement was sustained for 10+ frames
        displacement_onset_idx: Estimated touch frame based on pellet occlusion
        """
        # Find regions where distance > threshold
        displaced_frames = distance_array > threshold

        if not displaced_frames.any():
            return False, -1, 0.0, -1

        # Find consecutive runs of displaced frames
        # Add padding to detect boundaries
        padded = np.pad(displaced_frames, (1, 1), constant_values=False)
        diff = np.diff(padded.astype(int))

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Check each run for sustained displacement
        for start, end in zip(starts, ends):
            duration = end - start
            if duration >= min_duration:
                # Found sustained displacement
                max_dist = distance_array[start:end].max()

                # Find interaction frame using pellet occlusion
                # Look back from sustained displacement to find visibility drop
                lookback_start = max(0, start - 30)

                # Default: 1 frame before sustained displacement (empirically optimal)
                # Sustained displacement requires 10+ frames, so touch happened ~1 frame earlier
                onset_idx = max(0, start - 1)

                if lookback_start < start:
                    # Look for pellet likelihood drop before displacement
                    for i in range(start - 1, lookback_start, -1):
                        if i < len(pellet_conf) - 1:
                            # Check if likelihood dropped significantly
                            if pellet_conf[i] < 0.5 and pellet_conf[i-1] >= 0.5:
                                # Found where pellet was occluded - use frame BEFORE drop
                                onset_idx = i - 1
                                break

                return True, int(start), float(max_dist), int(onset_idx)

        return False, -1, 0.0, -1

    def detect_segment_outcome(
        self,
        df: pd.DataFrame,
        seg_start: int,
        seg_end: int,
        segment_num: int,
        segment_reaches: Optional[Dict[str, Any]] = None
    ) -> PelletOutcome:
        """
        Multi-stage progressive validation approach to determine pellet outcome.

        STAGE 1: Initial hypothesis based on visibility and position
        STAGE 2: Feature validation - does it have expected characteristics?
        STAGE 3: Temporal consistency - does behavior match hypothesis over time?
        STAGE 4: Final confidence scoring and uncertainty flagging
        """

        # Get ruler from SABL/SABR
        _, _, ruler = self.compute_expected_pillar(df, seg_start, seg_end)

        if ruler is None:
            return PelletOutcome(
                segment_num=segment_num,
                outcome='uncertain',
                confidence=0.0,
                flagged_for_review=True,
                flag_reason="Could not compute geometry from SABL/SABR"
            )

        # Get pellet trajectory data
        traj = self.get_pellet_trajectory(df, seg_start, seg_end, ruler)

        # Detect eating behavior
        eating_detected, eating_frame = self.detect_eating_signature(df, seg_start, seg_end)

        # v2.4.4: Detect pellet grab (paw near pellet when visibility drops)
        # Now also returns distance from pillar at grab time
        grab_detected, grab_frame, grab_dist = self.detect_pellet_grab(df, seg_start, seg_end)

        # Extract arrays for detailed analysis
        distance_array = traj['distance_array']
        pellet_conf = traj['pellet_confidence']

        # Initialize output variables
        confidence = 1.0
        flagged = False
        flag_reason = None
        outcome = 'uncertain'
        displacement_start_frame = None  # Track frame where displacement began

        # ============================================================================
        # STAGE 0: EARLY RETRIEVAL DETECTION (v2.4.1)
        # ============================================================================
        # If pellet grab detected (paw near when pellet disappeared), it's likely a retrieval
        # But v2.4.4: Check position at grab time to distinguish retrieval from displacement
        if grab_detected:
            start_dist = traj.get('start_distance_from_pillar') or 0
            end_dist = traj.get('end_distance_from_pillar') or 0

            # v2.4.4: If pellet started far from pillar (>0.4), it's displaced_outside not retrieved
            if start_dist and start_dist > 0.4:
                return PelletOutcome(
                    segment_num=segment_num,
                    outcome='displaced_outside',
                    interaction_frame=grab_frame,
                    outcome_known_frame=None,
                    pellet_visible_start=traj['visible'],
                    distance_from_pillar_start=start_dist,
                    pellet_visible_end=False,
                    distance_from_pillar_end=end_dist,
                    causal_reach_id=None,
                    causal_reach_frame=None,
                    confidence=0.75,
                    human_verified=False,
                    original_outcome=None,
                    flagged_for_review=True,
                    flag_reason=f"Pellet started off pillar ({start_dist:.2f}) and disappeared - likely displaced outside"
                )

            else:
                # Normal retrieval - pellet was on/near pillar when grabbed
                return PelletOutcome(
                    segment_num=segment_num,
                    outcome='retrieved',
                    interaction_frame=grab_frame,
                    outcome_known_frame=None,
                    pellet_visible_start=traj['visible'],
                    distance_from_pillar_start=start_dist,
                    pellet_visible_end=False,
                    distance_from_pillar_end=end_dist,
                    causal_reach_id=None,
                    causal_reach_frame=None,
                    confidence=0.85,
                    human_verified=False,
                    original_outcome=None,
                    flagged_for_review=True,
                    flag_reason="Pellet grabbed by paw (visibility dropped with paw near) - likely retrieved"
                )

        # ============================================================================
        # STAGE 1: INITIAL HYPOTHESIS
        # ============================================================================

        # Hypothesis 1: Pellet disappeared → RETRIEVED
        if not traj['visible'] or traj['visibility_pct'] < 0.10:
            hypothesis = 'retrieved'
            initial_confidence = 0.85 if eating_detected else 0.70

            # v2.4.4: Check if pellet started far from pillar - if so, classify as displaced_outside
            start_dist = traj.get('start_distance_from_pillar') or 0
            if start_dist and start_dist > 0.40:
                # Pellet started off pillar and disappeared - displaced outside, not retrieved
                outcome = 'displaced_outside'
                confidence = 0.75
                flagged = True
                flag_reason = f"Pellet started off pillar ({start_dist:.2f}) and disappeared - likely displaced outside"
            # v2.4: Check pellet movement direction before disappearance
            # This helps distinguish retrieval (toward box) from displacement (into SA)
            else:
                movement_direction, direction_conf = self.detect_pellet_movement_direction(
                    df, traj['pellet_frames'], distance_array, seg_start, seg_end
                )

                # STAGE 2: Validate retrieved features
                # Feature: Pellet should disappear during/after interaction
                # Feature: If eating detected, confidence is higher
                # Feature: If pellet moved toward box, more likely retrieval
                if eating_detected:
                    outcome = 'retrieved'
                    confidence = 0.85
                    # STAGE 3: Temporal check - did pellet disappear during eating window?
                    # (Already validated by eating_detected)
                elif movement_direction == 'toward_box':
                    # v2.4: Pellet moved toward box before disappearing - likely retrieval
                    outcome = 'retrieved'
                    confidence = 0.80
                    flagged = True
                    flag_reason = "Pellet moved toward box before disappearing - likely retrieved"
                elif movement_direction == 'toward_sa':
                    # v2.4: Pellet moved into SA before disappearing - could be displacement
                    # Check end distance to decide
                    end_dist = traj.get('end_distance_from_pillar', 0.0)
                    if end_dist and end_dist > 0.30:
                        outcome = 'displaced_sa'
                        confidence = 0.70
                        flagged = True
                        flag_reason = "Pellet moved into SA before disappearing - check if displaced"
                    else:
                        outcome = 'retrieved'
                        confidence = 0.70
                        flagged = True
                        flag_reason = "Pellet disappeared - assumed retrieved"
                else:
                    outcome = 'retrieved'
                    confidence = 0.70
                    flagged = True
                    flag_reason = "Pellet disappeared - assumed retrieved"
                    # STAGE 4: Flag for review because no eating signature

        # Hypothesis 2: Insufficient data → UNCERTAIN
        elif traj['end_distance_from_pillar'] is None:
            outcome = 'uncertain'
            confidence = 0.40
            flagged = True
            flag_reason = "Insufficient pellet tracking data"

        # Hypothesis 3: Pellet started off pillar → OPERATOR ERROR
        elif traj['start_distance_from_pillar'] > 0.30:
            outcome = 'untouched'
            confidence = 0.60
            flagged = True
            flag_reason = f"Pellet started off pillar (distance: {traj['start_distance_from_pillar']:.2f})"

        # Hypothesis 4: Check for SUSTAINED DISPLACEMENT during segment
        else:
            # STAGE 2: Look for sustained displacement pattern
            # This is key: pellet may have been displaced then returned to pillar
            is_displaced, disp_start_idx, max_disp_dist, disp_onset_idx = self.detect_sustained_displacement(
                distance_array,
                pellet_conf,
                threshold=0.30,
                min_duration=10
            )

            if is_displaced:
                # Pellet WAS displaced at some point (even if it returned)
                hypothesis = 'displaced_sa'

                # STAGE 2.5: PAW PROXIMITY CHECK
                # If pellet displaced but no paw was near, likely false positive (tray wobble)
                # Also use paw proximity to find exact interaction frame (more accurate)
                pellet_frames = traj['pellet_frames']
                paw_was_near = False
                paw_touch_frame = None

                if len(pellet_frames) > 0 and disp_start_idx < len(pellet_frames):
                    # Get actual frame index from displacement index
                    disp_frame = pellet_frames[disp_start_idx]
                    paw_was_near, paw_touch_frame = self.check_paw_proximity(
                        df, pellet_frames, disp_frame,
                        lookback_frames=30,
                        proximity_threshold_pixels=50.0
                    )

                    if not paw_was_near:
                        # Displacement detected but no paw was near!
                        # v2.4.3: Check end distance - if pellet ended far, it's still displaced
                        end_distance = traj['end_distance_from_pillar']
                        if end_distance > 0.35:
                            # Pellet ended far from pillar - definitely displaced
                            # Don't require paw proximity if result is obvious
                            outcome = 'displaced_sa'
                            confidence = 0.75
                            flagged = True
                            flag_reason = f"Pellet displaced (max: {max_disp_dist:.2f}) to far position (end: {end_distance:.2f}) - no paw detected but result is clear"
                            # Keep is_displaced = True to continue processing
                        else:
                            # Pellet ended near pillar despite displacement - tray wobble
                            outcome = 'untouched'
                            confidence = 0.80
                            flagged = True
                            flag_reason = f"Displacement detected (max: {max_disp_dist:.2f}) but no paw proximity and pellet near pillar (end: {end_distance:.2f}) - likely tray wobble"
                            # Override is_displaced to skip further displacement logic
                            is_displaced = False

                if is_displaced:
                    # STAGE 3: Temporal validation - check if pellet settled after displacement
                    # Did pellet disappear after displacement? → retrieved
                    after_displacement = pellet_conf[disp_start_idx + 10:]
                    if len(after_displacement) > 0 and (after_displacement > 0.5).mean() < 0.50:
                        outcome = 'retrieved'
                        confidence = 0.85
                    else:
                        # Pellet displaced and stayed visible → displaced_sa
                        # v2.4.1: But check if pellet ultimately disappeared (retrieval)
                        end_distance = traj['end_distance_from_pillar']

                        # Check overall pellet visibility trend
                        # Use segment-level visibility, not just tracked frames
                        seg_pellet_conf = df.iloc[seg_start:seg_end]['Pellet_likelihood'].values
                        late_segment_vis = seg_pellet_conf[-50:] if len(seg_pellet_conf) > 50 else seg_pellet_conf
                        pellet_visible_at_end = (late_segment_vis > 0.5).mean() > 0.30

                        if end_distance < 0.12 and not pellet_visible_at_end and max_disp_dist > 0.8:
                            # Pellet returned very near pillar AND disappeared after large displacement → likely retrieved
                            # v2.4.1: Stricter threshold to reduce false positives
                            outcome = 'retrieved'
                            confidence = 0.80
                            flagged = True
                            flag_reason = f"Pellet displaced (max: {max_disp_dist:.2f}), returned to pillar, then disappeared - likely retrieved"
                        else:
                            outcome = 'displaced_sa'
                            confidence = 0.85

                            # STAGE 4: Additional validation - did pellet return near pillar?
                            if end_distance < 0.15:
                                # Pellet ended up back on/near pillar after being displaced
                                # This is unusual but possible (bounced back)
                                flagged = True
                                flag_reason = f"Pellet displaced (max: {max_disp_dist:.2f}) but returned near pillar (end: {end_distance:.2f})"

                    # Record displacement start frame for interaction timing
                    # Use onset detection (lookback from sustained displacement)
                    pellet_frames = traj['pellet_frames']
                    if disp_onset_idx >= 0 and disp_onset_idx < len(pellet_frames):
                        # Use the onset frame (first frame where pellet started moving)
                        displacement_start_frame = pellet_frames[disp_onset_idx]
                    elif len(pellet_frames) > 0 and disp_start_idx < len(pellet_frames):
                        # Fallback: use sustained threshold crossing frame
                        displacement_start_frame = pellet_frames[disp_start_idx]

            else:
                # No sustained displacement detected
                # Check end position to determine untouched vs displaced
                end_distance = traj['end_distance_from_pillar']

                if end_distance < 0.20:
                    # Pellet stayed on/near pillar throughout
                    hypothesis = 'untouched'

                    # STAGE 3: Temporal consistency check
                    # Did pellet ever spike far from pillar momentarily?
                    max_distance = traj['max_distance_from_pillar']

                    if max_distance < 0.35:
                        # Pellet truly never left pillar
                        outcome = 'untouched'
                        confidence = 0.95
                    else:
                        # Pellet had momentary spikes (jitter or brief tray motion)
                        # But no sustained displacement
                        # v2.4: Check if paw was near during spike - if so, likely real interaction
                        pellet_frames = traj['pellet_frames']
                        spike_paw_near = False
                        spike_touch_frame = None

                        if max_distance > 0.55 and len(pellet_frames) > 0:
                            # Find frame of max distance (spike)
                            spike_idx = int(np.argmax(distance_array))
                            if spike_idx < len(pellet_frames):
                                spike_frame = pellet_frames[spike_idx]
                                spike_paw_near, spike_touch_frame = self.check_paw_proximity(
                                    df, pellet_frames, spike_frame,
                                    lookback_frames=15,  # Tighter window for spike analysis
                                    proximity_threshold_pixels=40.0  # Stricter threshold
                                )

                        if spike_paw_near and max_distance > 0.55:
                            # v2.4.2: Check if this is tray wobble (pellet moves WITH SA reference)
                            # If pellet relative movement is LESS than SA movement, it's tray wobble
                            is_tray_wobble = False
                            seg_pellet_x = df.iloc[seg_start:seg_end]['Pellet_x'].values
                            seg_pellet_l = df.iloc[seg_start:seg_end]['Pellet_likelihood'].values
                            seg_sabl_x = df.iloc[seg_start:seg_end]['SABL_x'].values
                            seg_sabr_x = df.iloc[seg_start:seg_end]['SABR_x'].values
                            seg_sa_mid_x = (seg_sabl_x + seg_sabr_x) / 2
                            good_track = seg_pellet_l > 0.5
                            if good_track.sum() > 20:
                                pellet_rel_x = seg_pellet_x - seg_sa_mid_x
                                pellet_rel_std = np.std(pellet_rel_x[good_track])
                                sa_std = np.std(seg_sa_mid_x[good_track])
                                if sa_std > 0 and pellet_rel_std < sa_std * 0.6:
                                    is_tray_wobble = True

                            if is_tray_wobble:
                                # Pellet moved WITH tray - not a real interaction
                                outcome = 'untouched'
                                confidence = 0.85
                                flagged = True
                                flag_reason = f"Spike (max: {max_distance:.2f}) detected but pellet moved with tray - likely wobble"
                            else:
                                # Paw was near during spike - this is a real interaction
                                # v2.4.1: Check if pellet subsequently disappeared (retrieved/eaten)
                                # vs just returned to pillar (displaced_sa)
                                pellet_disappeared_after_spike = False
                                if spike_idx < len(pellet_conf) - 20:
                                    # Check pellet visibility in last 30% of segment after spike
                                    remaining_segment = len(pellet_conf) - spike_idx
                                    check_start = spike_idx + int(remaining_segment * 0.5)
                                    late_visibility = pellet_conf[check_start:]
                                    if len(late_visibility) > 0:
                                        late_visible_pct = (late_visibility > 0.5).mean()
                                        pellet_disappeared_after_spike = late_visible_pct < 0.30

                                if pellet_disappeared_after_spike:
                                    # Pellet grabbed then eaten -> retrieved
                                    outcome = 'retrieved'
                                    confidence = 0.80
                                    flagged = True
                                    flag_reason = f"Spike (max: {max_distance:.2f}) with paw, pellet disappeared after - likely retrieved"
                                    displacement_start_frame = spike_touch_frame
                                elif end_distance < 0.15:
                                    # v2.4.3: Pellet returned to pillar (<0.15) - classify as untouched
                                    # Even with paw contact spike, if pellet is back on pillar it wasn't displaced
                                    outcome = 'untouched'
                                    confidence = 0.85
                                    flagged = True
                                    flag_reason = f"Spike (max: {max_distance:.2f}) with paw but pellet returned to pillar (end: {end_distance:.2f})"
                                else:
                                    # Pellet grabbed then stayed away from pillar -> displaced_sa
                                    outcome = 'displaced_sa'
                                    confidence = 0.75
                                    flagged = True
                                    flag_reason = f"Large spike (max: {max_distance:.2f}) with paw proximity - pellet stayed displaced (end: {end_distance:.2f})"
                                    displacement_start_frame = spike_touch_frame
                        else:
                            outcome = 'untouched'
                            confidence = 0.85

                            # STAGE 4: Flag high spikes for review
                            if max_distance > 0.55:
                                flagged = True
                                flag_reason = f"Momentary spike (max: {max_distance:.2f}) but no sustained displacement"

                elif end_distance > 0.25:
                    # Pellet ended far from pillar
                    # But we didn't detect sustained displacement pattern earlier
                    # This could mean displacement happened very late in segment

                    # STAGE 3: Check if eating occurred
                    if eating_detected:
                        outcome = 'retrieved'
                        confidence = 0.90
                    else:
                        outcome = 'displaced_sa'
                        confidence = 0.80

                        # STAGE 4: Lower confidence because pattern is ambiguous
                        flagged = True
                        flag_reason = f"Pellet ended far from pillar ({end_distance:.2f}) but no sustained displacement pattern detected"

                else:
                    # Ambiguous zone: 0.20 - 0.25
                    outcome = 'uncertain'
                    confidence = 0.50
                    flagged = True
                    flag_reason = f"Ambiguous distance: start={traj['start_distance_from_pillar']:.2f}, end={end_distance:.2f}, max={traj['max_distance_from_pillar']:.2f}"

        # ============================================================================
        # FINAL: Determine interaction frame and causal reach
        # ============================================================================

        # STEP 1: Get initial interaction frame estimate from displacement/eating detection
        initial_interaction_frame = None
        if eating_detected:
            initial_interaction_frame = eating_frame
        elif displacement_start_frame is not None:
            initial_interaction_frame = displacement_start_frame

        # STEP 2: Find causal reach using initial interaction frame estimate
        causal_id, causal_frame, reach_features = self.find_causal_reach(
            segment_reaches, outcome, initial_interaction_frame
        )

        # STEP 3: Use reach apex as final interaction frame if available (most accurate)
        if causal_frame is not None:
            interaction_frame = causal_frame
        else:
            interaction_frame = initial_interaction_frame

            # Flag if using displacement detection without reach confirmation
            if displacement_start_frame is not None and not flagged:
                pellet_frames = traj.get('pellet_frames', [])
                if len(pellet_frames) > 0 and disp_onset_idx < len(pellet_frames):
                    expected_default = max(0, disp_start_idx - 1)
                    if disp_onset_idx == expected_default:
                        flagged = True
                        flag_reason = "Interaction frame from displacement detection (no reach data) - verify timing"

        return PelletOutcome(
            segment_num=segment_num,
            outcome=outcome,
            interaction_frame=interaction_frame,
            outcome_known_frame=None,
            pellet_visible_start=traj['visible'],
            distance_from_pillar_start=traj.get('start_distance_from_pillar'),
            pellet_visible_end=traj['visible'] and traj['visibility_pct'] > 0.5,
            distance_from_pillar_end=traj.get('end_distance_from_pillar'),
            causal_reach_id=causal_id,
            causal_reach_frame=causal_frame,
            confidence=round(confidence, 2),
            human_verified=False,
            original_outcome=None,
            flagged_for_review=flagged,
            flag_reason=flag_reason
        )
    
    def find_causal_reach(
        self,
        segment_reaches: Optional[Dict[str, Any]],
        outcome: str,
        interaction_frame: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[int], Optional[Dict[str, Any]]]:
        """
        Identify which reach caused the outcome.

        Strategy: Find the reach whose apex is CLOSEST BEFORE the interaction frame.
        The causal reach must happen before the pellet displacement/eating.

        Returns:
            (reach_id, apex_frame, reach_features)
        """
        if outcome in ['untouched', 'no_pellet', 'uncertain']:
            return None, None, None

        if not segment_reaches or not segment_reaches.get('reaches'):
            return None, None, None

        reaches = segment_reaches['reaches']

        if len(reaches) == 0:
            return None, None, None

        # Strategy: Find reach that contains or is closest before interaction_frame
        if interaction_frame is not None:
            # First pass: Check if interaction falls WITHIN any reach window
            for reach in reaches:
                start = reach.get('start_frame')
                end = reach.get('end_frame')
                apex = reach.get('apex_frame')

                if start is not None and end is not None:
                    if start <= interaction_frame <= end:
                        # Interaction happened during this reach - this is the causal reach
                        features = {
                            'max_extent_ruler': reach.get('max_extent_ruler'),
                            'duration_frames': reach.get('duration_frames'),
                            'start_frame': start,
                            'end_frame': end,
                            'apex_frame': apex,
                            'distance_to_interaction': 0  # Interaction within reach window
                        }
                        return reach.get('reach_id'), apex, features

            # Second pass: Find reach with apex closest BEFORE interaction
            closest_reach = None
            min_distance = float('inf')

            for reach in reaches:
                apex = reach.get('apex_frame')
                if apex is not None and apex <= interaction_frame:
                    distance = interaction_frame - apex  # Distance backward in time
                    if distance < min_distance:
                        min_distance = distance
                        closest_reach = reach

            if closest_reach:
                # Extract features for this reach
                features = {
                    'max_extent_ruler': closest_reach.get('max_extent_ruler'),
                    'duration_frames': closest_reach.get('duration_frames'),
                    'start_frame': closest_reach.get('start_frame'),
                    'end_frame': closest_reach.get('end_frame'),
                    'apex_frame': closest_reach.get('apex_frame'),
                    'distance_to_interaction': min_distance
                }
                return closest_reach.get('reach_id'), closest_reach.get('apex_frame'), features

        # Fallback: use last reach (original heuristic)
        last_reach = reaches[-1]
        features = {
            'max_extent_ruler': last_reach.get('max_extent_ruler'),
            'duration_frames': last_reach.get('duration_frames'),
            'start_frame': last_reach.get('start_frame'),
            'end_frame': last_reach.get('end_frame'),
            'apex_frame': last_reach.get('apex_frame'),
            'distance_to_interaction': None
        }
        return last_reach.get('reach_id'), last_reach.get('apex_frame'), features
    
    def detect(
        self,
        dlc_path: Path,
        segments_path: Path,
        reaches_path: Optional[Path] = None
    ) -> VideoOutcomes:
        """Main entry point: detect all pellet outcomes in a video."""
        df = load_dlc(dlc_path)
        boundaries = load_segments(segments_path)
        
        reaches_data = None
        if reaches_path and reaches_path.exists():
            with open(reaches_path) as f:
                reaches_data = json.load(f)
        
        video_name = Path(dlc_path).stem
        if 'DLC_' in video_name:
            video_name = video_name.split('DLC_')[0]
        
        n_segments = len(boundaries) - 1
        outcomes = []
        
        for seg_idx in range(n_segments):
            seg_start = boundaries[seg_idx]
            seg_end = boundaries[seg_idx + 1]
            segment_num = seg_idx + 1
            
            # Get reaches for this segment
            segment_reaches = None
            if reaches_data:
                for seg in reaches_data.get('segments', []):
                    if seg['segment_num'] == segment_num:
                        segment_reaches = seg
                        break
            
            outcome = self.detect_segment_outcome(df, seg_start, seg_end, segment_num, segment_reaches)
            outcomes.append(outcome)
        
        summary = {
            'total_segments': n_segments,
            'retrieved': sum(1 for o in outcomes if o.outcome == 'retrieved'),
            'displaced_sa': sum(1 for o in outcomes if o.outcome == 'displaced_sa'),
            'displaced_outside': sum(1 for o in outcomes if o.outcome == 'displaced_outside'),
            'untouched': sum(1 for o in outcomes if o.outcome == 'untouched'),
            'no_pellet': sum(1 for o in outcomes if o.outcome == 'no_pellet'),
            'uncertain': sum(1 for o in outcomes if o.outcome == 'uncertain'),
            'flagged': sum(1 for o in outcomes if o.flagged_for_review),
            'mean_confidence': round(np.mean([o.confidence for o in outcomes]), 2)
        }
        
        return VideoOutcomes(
            detector_version=VERSION,
            video_name=video_name,
            total_frames=len(df),
            n_segments=n_segments,
            segments=outcomes,
            summary=summary,
            detected_at=datetime.now().isoformat(),
            validated=False,
            validated_by=None,
            validated_at=None,
            corrections_made=0,
            segments_flagged=sum(1 for o in outcomes if o.flagged_for_review)
        )
    
    @staticmethod
    def save_results(results: VideoOutcomes, output_path: Path, validation_status: str = "needs_review") -> None:
        """Save results to JSON with validation_status.

        Args:
            results: Detection results
            output_path: Output file path
            validation_status: Initial status - "needs_review" (default) or "auto_approved"
        """
        from datetime import datetime

        data = asdict(results)

        # Add validation_status for new architecture (v2.3+)
        data["validation_status"] = validation_status
        data["validation_timestamp"] = datetime.now().isoformat()

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Update pipeline index with new file
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_file_created(output_path, metadata={
                "outcome_count": results.n_segments,
                "outcomes_version": results.detector_version,
                "outcome_validation": validation_status,
                "outcome_breakdown": {
                    "retrieved": results.summary.retrieved if results.summary else 0,
                    "displaced_sa": results.summary.displaced_sa if results.summary else 0,
                    "displaced_outside": results.summary.displaced_outside if results.summary else 0,
                    "untouched": results.summary.untouched if results.summary else 0,
                },
            })
            index.save()
        except Exception:
            pass  # Don't fail outcome detection if index update fails

        # Sync to central database
        try:
            from mousereach.sync.database import sync_file_to_database
            sync_file_to_database(output_path)
        except Exception:
            pass  # Don't fail outcome detection if database sync fails

    @staticmethod
    def load_results(path: Path) -> VideoOutcomes:
        """Load results from JSON"""
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct segments with optional field handling
        segments = []
        for s in data['segments']:
            outcome = PelletOutcome(
                segment_num=s['segment_num'],
                outcome=s['outcome'],
                interaction_frame=s.get('interaction_frame'),
                outcome_known_frame=s.get('outcome_known_frame'),
                pellet_visible_start=s.get('pellet_visible_start', True),
                distance_from_pillar_start=s.get('distance_from_pillar_start'),
                pellet_visible_end=s.get('pellet_visible_end', True),
                distance_from_pillar_end=s.get('distance_from_pillar_end'),
                causal_reach_id=s.get('causal_reach_id'),
                causal_reach_frame=s.get('causal_reach_frame'),
                confidence=s.get('confidence', 0.0),
                human_verified=s.get('human_verified', False),
                original_outcome=s.get('original_outcome'),
                flagged_for_review=s.get('flagged_for_review', False),
                flag_reason=s.get('flag_reason')
            )
            segments.append(outcome)
        
        return VideoOutcomes(
            detector_version=data['detector_version'],
            video_name=data['video_name'],
            total_frames=data['total_frames'],
            n_segments=data['n_segments'],
            segments=segments,
            summary=data['summary'],
            detected_at=data['detected_at'],
            validated=data.get('validated', False),
            validated_by=data.get('validated_by'),
            validated_at=data.get('validated_at'),
            corrections_made=data.get('corrections_made', 0),
            segments_flagged=data.get('segments_flagged', 0)
        )


def detect_pellet_outcomes(
    dlc_path: str,
    segments_path: str,
    reaches_path: Optional[str] = None
) -> VideoOutcomes:
    """Convenience function for pellet outcome detection."""
    detector = PelletOutcomeDetector()
    return detector.detect(
        Path(dlc_path),
        Path(segments_path),
        Path(reaches_path) if reaches_path else None
    )
