"""
reach_detector.py - Core reach detection algorithm

ALGORITHM SUMMARY
=================
Detects individual mouse reaching attempts within pellet presentation segments
using DeepLabCut pose estimation data. Identifies reach start, apex, and end
frames based on hand visibility while the nose is engaged at the slit.

SCIENTIFIC DESCRIPTION
======================
This algorithm identifies discrete reaching events during the single-pellet
reaching task. A "reach" is defined as the period when the mouse extends its
paw through the slit toward the pellet. Detection relies on two key criteria:
(1) the mouse must be engaged with the slit opening (nose within threshold
distance), and (2) at least one hand tracking point must be visible with
sufficient confidence. The algorithm tracks four hand points (RightHand,
RHLeft, RHOut, RHRight) and triggers a reach when any becomes visible during
nose engagement.

INPUT REQUIREMENTS
==================
- DeepLabCut tracking file (.h5) with 18 bodyparts
- Segmentation boundaries file (_segments.json)
- Required bodyparts for detection:
  - Hand tracking: RightHand, RHLeft, RHOut, RHRight
  - Engagement detection: Nose
  - Slit reference: BOXL, BOXR (for slit center calculation)
  - Calibration: SABL, SABR (for ruler-unit conversion)

DETECTION RULES
===============
1. Nose Engagement Check
   - Calculate slit center from BOXL and BOXR midpoint
   - Nose must be within NOSE_ENGAGEMENT_THRESHOLD pixels of slit center
   - This ensures the mouse is actually attempting to reach

2. Reach Start Detection
   - First frame where ANY hand point has likelihood >= HAND_LIKELIHOOD_THRESHOLD
   - Must occur while nose is engaged at slit

3. Reach End Detection (first of these conditions)
   - Hand disappears: All hand points below threshold for DISAPPEAR_THRESHOLD consecutive frames
   - Hand retracts: Hand x-position moves significantly leftward from slit

4. Minimum Duration Filter
   - Reaches shorter than MIN_REACH_DURATION frames are filtered as noise

5. Apex Detection
   - Frame with maximum hand extension (rightmost x-position during reach)

KEY PARAMETERS
==============
| Parameter | Value | Unit | Rationale |
|-----------|-------|------|-----------|
| HAND_LIKELIHOOD_THRESHOLD | 0.5 | confidence | Matches display threshold in review widget |
| NOSE_ENGAGEMENT_THRESHOLD | 25 | pixels | Derived from ground truth analysis |
| MIN_REACH_DURATION | 4 | frames | Filters tracking noise (v3.5: increased from 2) |
| MIN_EXTENT_THRESHOLD | -15 | pixels | Filters non-reach hand visibility (v3.5) |
| DISAPPEAR_THRESHOLD | 2 | frames | Handles brief tracking dropouts |
| GAP_TOLERANCE | 2 | frames | Merges reaches separated by brief gaps |

OUTPUT FORMAT
=============
JSON file (*_reaches.json) containing:
- Per-segment reach list with timing (start_frame, apex_frame, end_frame)
- Reach extent in pixels and ruler units
- Human correction tracking (source, human_corrected, original_start/end)
- Flagging for review with reason

VALIDATION HISTORY
==================
- v3.0.0: Initial data-driven rules from 12-video ground truth analysis
- v3.1.0: Refined thresholds, matched display threshold for consistency

KNOWN LIMITATIONS
=================
- May merge reaches if mouse re-extends without fully retracting
- Tracking dropout during reach causes premature end detection
- Low DLC quality (likelihood < threshold) causes missed reaches
- Ambiguous when mouse "paws" at slit without full extension

REFERENCES
==========
- Rule derivation: src/mousereach/reach/analysis/DISCOVERED_RULES.md
- Evaluation scripts: src/mousereach/reach/analysis/
- Ground truth videos: Processing/ folder with _reach_ground_truth.json files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from .geometry import compute_segment_geometry, get_boxr_reference, load_dlc, load_segments
from .boundary_refiner import split_reach_boundaries
from .boundary_polisher import BoundaryPolisher


VERSION = "5.3.0"  # v5.3: Retrained polisher (window=20, max_correction=30, 300 trees)


@dataclass
class Reach:
    """Single reach event"""
    reach_id: int              # Global ID within video
    reach_num: int             # Number within segment
    start_frame: int           # Paw first visible in SA
    apex_frame: int            # Maximum extension (algorithm-computed)
    end_frame: int             # Paw disappears from SA
    duration_frames: int
    max_extent_pixels: float   # Algorithm-computed feature
    max_extent_ruler: float    # Algorithm-computed feature
    
    # Human annotation tracking
    source: str = "algorithm"  # "algorithm" or "human_added"
    human_corrected: bool = False
    original_start: Optional[int] = None  # What algo said, if corrected
    original_end: Optional[int] = None    # What algo said, if corrected

    # Review notes (for flagging discrepancies, etc.)
    review_note: Optional[str] = None

    # Analysis exclusion flag - mark reaches that shouldn't be used in analysis
    exclude_from_analysis: bool = False
    exclude_reason: Optional[str] = None  # e.g., "tracking lost", "ambiguous", "incomplete"

    # Boundary confidence (v3.3+) - how clear were the start/end boundaries?
    confidence: Optional[float] = None           # 0.0-1.0, min of start/end
    start_confidence: Optional[float] = None     # How clear was start boundary (likelihood jump)
    end_confidence: Optional[float] = None       # How clear was end boundary (likelihood drop)


@dataclass
class SegmentReaches:
    """All reaches within one pellet segment"""
    segment_num: int
    start_frame: int
    end_frame: int
    ruler_pixels: float
    n_reaches: int
    reaches: List[Reach]
    
    # Review flagging
    flagged_for_review: bool = False
    flag_reason: Optional[str] = None  # Why human flagged this segment


@dataclass
class VideoReaches:
    """Complete reach detection results for one video"""
    detector_version: str
    video_name: str
    total_frames: int
    boxr_x: float
    n_segments: int
    segments: List[SegmentReaches]
    summary: Dict
    detected_at: str
    
    # Validation tracking
    validated: bool = False
    validated_by: Optional[str] = None
    validated_at: Optional[str] = None
    corrections_made: int = 0
    reaches_added: int = 0
    reaches_removed: int = 0
    segments_flagged: int = 0


class ReachDetector:
    """
    Data-driven reach detector (v3.0).

    Rules derived from ground truth analysis:
    1. Nose Engagement: Nose must be within 20px of slit center (BOXL-BOXR midpoint)
    2. Reach Start: First frame ANY hand point likelihood >= 0.3 while nose engaged
    3. Reach End: Either hand disappears OR hand retracts left significantly

    See src/mousereach/reach/analysis/DISCOVERED_RULES.md for full derivation.
    """

    # Thresholds derived from GT analysis (v3.1 refinements)
    # NOTE: Display threshold in review_widget.py is 0.5 - humans only SEE points >= 0.5
    # So algorithm must use 0.5 to match what humans annotate against
    HAND_LIKELIHOOD_THRESHOLD = 0.5  # Match display threshold in review widget
    CONFIDENT_START_THRESHOLD = 0.5  # Same as HAND_LIKELIHOOD_THRESHOLD
    NOSE_ENGAGEMENT_THRESHOLD = 25   # pixels from slit center
    # v5.0: Require hand visible for START_CONFIRM consecutive frames before
    # starting a reach. 83 of 179 early-start errors were single/few-frame
    # noise where hand likelihood briefly crossed 0.5 then dropped. Genuine
    # reach starts have the hand visible continuously.
    START_CONFIRM = 2                # consecutive visible frames to confirm start
    MIN_REACH_DURATION = 4           # frames (increased from 2 based on GT analysis - 42% of FPs were ≤3 frames)
    LOOKAHEAD_FRAMES = 3             # frames to check for sustained hand disappearance
    DISAPPEAR_THRESHOLD = 3          # v4.0: consecutive invisible frames before ending reach (was 2)
    GAP_TOLERANCE = 0                # v4.0: no post-processing merge - handled by tolerance in state machine

    # v3.1: End-on-drop detection (not just disappearance)
    HIGH_CONFIDENCE = 0.70           # "Confident" tracking level
    DROP_TO_THRESHOLD = 0.50         # If drops from HIGH to this, consider reach ended
    VALLEY_THRESHOLD = 0.50          # Don't merge if gap has confidence below this

    # Splitting parameters (data-driven from GT analysis)
    # Splits long reaches at DLC confidence dips. Helps existence (+3 matches) at cost
    # of imprecise split points (~5 frame early-end errors). Net positive.
    SPLIT_THRESHOLD_FRAMES = 25      # 95th percentile of GT reach duration
    CONFIDENCE_HIGH = 0.5            # "High" confidence before drop
    CONFIDENCE_LOW = 0.35            # "Low" confidence in gap

    # v3.5: Negative extent filter (data-driven from GT analysis 2026-01)
    # 44% of FPs had extent < -10px (hand visible but behind slit, not reaching)
    # Small negative (-2 to -10px) = valid attempt reaches (keep)
    # Large negative (< -15px) = non-reach hand visibility (filter)
    # Note: Attempted -10 threshold in v3.6 but recall dropped 10%, reverted
    MIN_EXTENT_THRESHOLD = -15.0     # pixels - filter reaches below this

    # v4.2: Restored to 5px. 15px caused 99% of early-end errors by splitting
    # single reaches during normal hand oscillation near the slit. The tolerance-based
    # disappearance (DISAPPEAR_THRESHOLD=3) handles 89% of real splits; this check
    # is only a safety net for the rare case where hand retracts without disappearing.
    HAND_RETURN_THRESHOLD = 5.0      # pixels - hand returned to starting position

    # v5.0: Look-ahead confirmation for retraction/return-to-start
    # 98.8% of 683 early-end cases had hand visible at EVERY frame between algo_end
    # and gt_end. Root cause: retraction/return checks fire on single-frame DLC
    # bodypart switches (the 4 hand points swap which is "best", causing apparent
    # 10-25px position jumps). Fix: require retraction to be sustained over
    # RETRACTION_CONFIRM consecutive additional frames before ending the reach.
    RETRACTION_CONFIRM = 2           # frames of sustained retraction needed to confirm end
    BP_SWITCH_GRACE = 3              # v5.1: skip retraction checks for N frames after BP switch

    RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

    def __init__(self, model_dir=None):
        self.global_reach_id = 0
        # v5.2: Initialize ML boundary polisher (gracefully handles missing models)
        try:
            self._polisher = BoundaryPolisher(model_dir=model_dir)
        except Exception:
            self._polisher = None

    def _get_slit_center(self, df: pd.DataFrame, seg_start: int, seg_end: int) -> Tuple[float, float]:
        """Get stable slit center from segment median of BOXL and BOXR."""
        segment_df = df.iloc[seg_start:seg_end]
        boxl_x = segment_df['BOXL_x'].median()
        boxl_y = segment_df['BOXL_y'].median()
        boxr_x = segment_df['BOXR_x'].median()
        boxr_y = segment_df['BOXR_y'].median()

        center_x = (boxl_x + boxr_x) / 2
        center_y = (boxl_y + boxr_y) / 2

        return center_x, center_y

    def _is_nose_engaged(self, row, slit_x: float, slit_y: float) -> bool:
        """Check if nose is close to slit (engaged position)."""
        nose_x = row.get('Nose_x', np.nan)
        nose_y = row.get('Nose_y', np.nan)
        nose_l = row.get('Nose_likelihood', 0)

        if nose_l < 0.3 or np.isnan(nose_x):
            return False  # Can't determine, assume not engaged

        distance = np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2)
        return distance < self.NOSE_ENGAGEMENT_THRESHOLD

    def _any_hand_visible(self, row) -> bool:
        """Check if ANY hand point is visible (likelihood >= 0.3)."""
        for p in self.RH_POINTS:
            if row.get(f'{p}_likelihood', 0) >= self.HAND_LIKELIHOOD_THRESHOLD:
                return True
        return False

    def _hand_confident(self, row) -> bool:
        """Check if ANY hand point has confident tracking (likelihood >= 0.5)."""
        for p in self.RH_POINTS:
            if row.get(f'{p}_likelihood', 0) >= self.CONFIDENT_START_THRESHOLD:
                return True
        return False

    def _should_end_on_drop(self, prev_likelihood: float, curr_likelihood: float) -> bool:
        """
        v3.1: Check if reach should end due to confidence drop.

        End reach when confidence drops from HIGH (>=0.7) to below 0.5,
        even if hand is still technically visible (>0.3).
        """
        if prev_likelihood >= self.HIGH_CONFIDENCE and curr_likelihood < self.DROP_TO_THRESHOLD:
            return True
        return False

    def _get_best_hand_position(self, row) -> Tuple[Optional[float], Optional[float], bool]:
        """Get position of most confident hand point."""
        best_x = None
        best_y = None
        best_l = 0

        for p in self.RH_POINTS:
            likelihood = row.get(f'{p}_likelihood', 0)
            if likelihood >= self.HAND_LIKELIHOOD_THRESHOLD and likelihood > best_l:
                best_x = row.get(f'{p}_x', np.nan)
                best_y = row.get(f'{p}_y', np.nan)
                best_l = likelihood

        return best_x, best_y, best_l > 0

    def _get_mean_hand_x(self, row) -> Optional[float]:
        """
        v5.1: Get mean X position of ALL visible hand points.

        Using the mean of all visible points (instead of only the best single
        point) dampens position jumps from DLC bodypart switching. When one of
        the 4 hand points suddenly becomes "best" and its X differs by 10-20px,
        the mean of 2-3 visible points barely changes — preventing false
        retraction triggers from the artifact persisting for several frames
        after the decision tree caught the initial switch.
        """
        xs = []
        for p in self.RH_POINTS:
            if row.get(f'{p}_likelihood', 0) >= self.HAND_LIKELIHOOD_THRESHOLD:
                x = row.get(f'{p}_x', np.nan)
                if not np.isnan(x):
                    xs.append(x)
        if xs:
            return np.mean(xs)
        return None

    def _hand_will_disappear(self, df: pd.DataFrame, frame: int) -> bool:
        """
        Check if hand will disappear (sustained) in the next few frames.

        Requires DISAPPEAR_THRESHOLD consecutive frames with no hand visible
        to avoid false triggers from momentary tracking dropouts.
        """
        consecutive_invisible = 0

        for offset in range(1, self.LOOKAHEAD_FRAMES + 1):
            future_frame = frame + offset
            if future_frame >= len(df):
                consecutive_invisible += 1
                continue

            if not self._any_hand_visible(df.iloc[future_frame]):
                consecutive_invisible += 1
            else:
                consecutive_invisible = 0  # Reset on visible frame

            if consecutive_invisible >= self.DISAPPEAR_THRESHOLD:
                return True

        return False

    def _detect_hand_retraction(
        self,
        df: pd.DataFrame,
        frame: int,
        reach_max_x: float,
        slit_x: float
    ) -> bool:
        """
        v3.2: Detect if hand has started retracting (moving away from pellet).

        Ground truth analysis showed humans end reaches earlier than algorithm,
        typically when hand starts moving back toward slit even if still visible.

        Returns True if hand is retracting significantly.
        """
        # Get current hand position
        hand_x, _, valid = self._get_best_hand_position(df.iloc[frame])
        if not valid or hand_x is None:
            return False

        # Check if hand has retracted significantly from max extension
        # Only consider retraction if hand was extended past slit
        extension = reach_max_x - slit_x
        if extension < 5:  # Didn't extend much
            return False

        # Current position relative to max
        retraction = reach_max_x - hand_x

        # v5.0: Raised from 40% to 50% - combined with look-ahead confirmation,
        # the higher threshold prevents false retraction triggers from hand oscillation
        # during reaching (hand wobbles 40-50% during extension but isn't truly retracting).
        retraction_threshold = extension * 0.50

        return retraction > retraction_threshold and retraction > 5

    def _get_best_hand_bp(self, row) -> Optional[str]:
        """Get the name of the most confident hand bodypart."""
        best_bp = None
        best_l = 0
        for p in self.RH_POINTS:
            l = row.get(f'{p}_likelihood', 0)
            if l >= self.HAND_LIKELIHOOD_THRESHOLD and l > best_l:
                best_l = l
                best_bp = p
        return best_bp

    def _get_hand_x_spread(self, row) -> float:
        """Get the X-position spread among visible hand points."""
        xs = []
        for p in self.RH_POINTS:
            if row.get(f'{p}_likelihood', 0) >= self.HAND_LIKELIHOOD_THRESHOLD:
                x = row.get(f'{p}_x', np.nan)
                if not np.isnan(x):
                    xs.append(x)
        if len(xs) >= 2:
            return max(xs) - min(xs)
        return 0.0

    def _check_multi_point_retraction(self, df: pd.DataFrame, frame: int) -> Tuple[int, bool]:
        """
        Check if ALL visible hand points agree on retraction direction.
        Returns (n_visible, all_agree_retract).
        """
        if frame < 1 or frame >= len(df):
            return 0, True

        prev_row = df.iloc[frame - 1]
        curr_row = df.iloc[frame]
        n_visible = 0
        all_retracted = True

        for p in self.RH_POINTS:
            prev_l = prev_row.get(f'{p}_likelihood', 0)
            curr_l = curr_row.get(f'{p}_likelihood', 0)
            if prev_l >= self.HAND_LIKELIHOOD_THRESHOLD and curr_l >= self.HAND_LIKELIHOOD_THRESHOLD:
                n_visible += 1
                prev_x = prev_row.get(f'{p}_x', np.nan)
                curr_x = curr_row.get(f'{p}_x', np.nan)
                if not np.isnan(prev_x) and not np.isnan(curr_x):
                    if curr_x >= prev_x:  # This point didn't retract
                        all_retracted = False

        return n_visible, all_retracted

    def _evaluate_end_candidate(
        self,
        df: pd.DataFrame,
        frame: int,
        reach_max_x: float,
        slit_x: float
    ) -> Tuple[bool, bool]:
        """
        v5.0: Multi-signal decision tree for reach end candidates.

        Called when retraction or return-to-start fires at frame X.
        Evaluates three independent signals to distinguish real retraction
        from DLC bodypart switching artifacts:

        1. Bodypart identity switch: Did the "best" hand point change?
           49% of early-end errors are caused by DLC switching which of the
           4 hand points (RightHand, RHLeft, RHOut, RHRight) has highest
           confidence. The position jumps 10-20px but the hand didn't move.

        2. Multi-point disagreement: Do all visible points agree the hand
           retracted? If some points moved left but others didn't, the
           "retraction" is noise from a single point, not real movement.

        3. Look-ahead confirmation: Does retraction sustain over the next
           RETRACTION_CONFIRM frames? Transient artifacts resolve in 1-2
           frames.

        Returns (should_end, bp_switched):
          - should_end: True to END the reach, False to CONTINUE
          - bp_switched: True if a bodypart switch was detected at this frame
        """
        prev_frame = frame - 1
        bp_switched = False

        # ─── Node 1: Bodypart identity switch ───
        if 0 <= prev_frame < len(df):
            prev_bp = self._get_best_hand_bp(df.iloc[prev_frame])
            curr_bp = self._get_best_hand_bp(df.iloc[frame])

            if (prev_bp is not None and curr_bp is not None
                    and prev_bp != curr_bp):
                bp_switched = True
                # Best bodypart changed — check if this is a tracking artifact
                x_spread = self._get_hand_x_spread(df.iloc[frame])
                if x_spread > 10:
                    # Large spread (>10px) + BP switch = artifact
                    # The "retraction" is just a different point on the hand
                    # becoming most confident, not the hand actually moving
                    return False, bp_switched  # CONTINUE

        # ─── Node 2: Multi-point disagreement ───
        n_visible, all_retracted = self._check_multi_point_retraction(df, frame)
        if n_visible >= 2 and not all_retracted:
            # Multiple hand points visible, but NOT all moved leftward.
            # If the hand truly retracted, all points would agree.
            # Disagreement means a single point jumped (DLC noise).
            return False, bp_switched  # CONTINUE

        # ─── Node 3: Look-ahead confirmation ───
        # If we reach here, the retraction looks real (same BP, or all points
        # agree). Verify it's sustained over RETRACTION_CONFIRM frames.
        for offset in range(1, self.RETRACTION_CONFIRM + 1):
            future_frame = frame + offset
            if future_frame >= len(df):
                return True, bp_switched  # End of data — confirm

            future_row = df.iloc[future_frame]
            if not self._any_hand_visible(future_row):
                continue  # Invisible frame doesn't contradict retraction

            future_x, _, valid = self._get_best_hand_position(future_row)
            if not valid or future_x is None:
                continue

            extension = reach_max_x - slit_x
            if extension < 5:
                continue

            retraction = reach_max_x - future_x
            retraction_threshold = extension * 0.50

            hand_offset = future_x - slit_x
            returned = hand_offset < self.HAND_RETURN_THRESHOLD

            if not (retraction > retraction_threshold and retraction > 5) and not returned:
                return False, bp_switched  # Hand re-extended — cancel

        return True, bp_switched  # All checks passed — END the reach

    def _hand_returned_to_start(self, hand_x: float, slit_x: float, reach_max_x: float) -> bool:
        """
        Check if hand has returned close to slit center after being extended.

        This indicates a new reach may be starting. But we're conservative:
        only trigger if hand clearly returned to starting position AND
        was previously extended significantly.
        """
        if hand_x is None or np.isnan(hand_x):
            return False

        # Hand position relative to slit
        hand_offset = hand_x - slit_x

        # Was reach extended significantly (>5 pixels from slit)?
        was_extended = reach_max_x and (reach_max_x - slit_x) > 5

        # Has hand returned close to slit center?
        returned_to_start = hand_offset < self.HAND_RETURN_THRESHOLD

        return was_extended and returned_to_start

    def detect_reaches_in_segment(
        self,
        df: pd.DataFrame,
        seg_start: int,
        seg_end: int,
        boxr_x: float,
        ruler_pixels: float
    ) -> List[Reach]:
        """
        Detect reaches using state machine approach.

        States:
        - IDLE: Not engaged, no reach
        - ENGAGED: Nose near slit, waiting for hand
        - REACHING: Hand visible, tracking reach
        """
        reaches = []

        # Get stable slit center for this segment
        slit_x, slit_y = self._get_slit_center(df, seg_start, seg_end)

        # State tracking
        in_reach = False
        reach_start = None
        reach_data = []  # [(frame, x, y), ...]
        reach_max_x = 0  # Track max extension for return detection
        consecutive_invisible = 0  # v4.0: tolerance-based disappearance tracking
        consecutive_start_visible = 0  # v5.0: consecutive visible frames for start confirmation
        pending_start_frame = None     # v5.0: first frame of candidate start
        bp_switch_grace = 0  # v5.1: remaining frames to skip retraction after BP switch

        reach_num = 0

        for frame in range(seg_start, min(seg_end, len(df))):
            row = df.iloc[frame]
            nose_engaged = self._is_nose_engaged(row, slit_x, slit_y)
            hand_visible = self._any_hand_visible(row)
            hand_x, hand_y, _ = self._get_best_hand_position(row)

            if not in_reach:
                # Not currently in a reach - check for reach start
                # v5.0: Require START_CONFIRM consecutive frames with hand visible
                # and nose engaged before committing to start. Filters single-frame
                # DLC noise where likelihood briefly crosses 0.5.
                if nose_engaged and hand_visible:
                    consecutive_start_visible += 1
                    if pending_start_frame is None:
                        pending_start_frame = frame
                    if consecutive_start_visible >= self.START_CONFIRM:
                        # REACH START confirmed: hand visible for enough frames
                        in_reach = True
                        reach_start = pending_start_frame  # Start at first visible frame
                        # Build reach_data from the confirmation window
                        reach_data = []
                        reach_max_x = 0
                        for f in range(pending_start_frame, frame + 1):
                            hx, hy, _ = self._get_best_hand_position(df.iloc[f])
                            reach_data.append((f, hx, hy))
                            if hx and hx > reach_max_x:
                                reach_max_x = hx
                        consecutive_invisible = 0
                        consecutive_start_visible = 0
                        pending_start_frame = None
                else:
                    consecutive_start_visible = 0
                    pending_start_frame = None

            else:
                # Currently in a reach - check for reach end
                end_reach = False
                end_frame = frame - 1  # Default: end at previous frame

                if not hand_visible:
                    # v4.0: Tolerance-based disappearance detection
                    # Don't end immediately on single invisible frame (DLC flicker).
                    # Only end after DISAPPEAR_THRESHOLD consecutive invisible frames.
                    # This replaces the old immediate-end + merge-postprocessing approach
                    # which caused cascading merges across DLC flicker cycles.
                    consecutive_invisible += 1
                    if consecutive_invisible >= self.DISAPPEAR_THRESHOLD:
                        # Hand has been gone long enough - this is a real disappearance
                        end_reach = True
                        # End at the last frame where hand was actually visible
                        if reach_data:
                            end_frame = reach_data[-1][0]
                        else:
                            end_frame = frame - consecutive_invisible
                    # else: stay in reach, don't add invisible frame to data

                else:
                    # Hand IS visible - reset invisible counter
                    consecutive_invisible = 0

                    # Check end conditions (only when hand is visible)
                    # v5.1: Decrement grace period counter
                    if bp_switch_grace > 0:
                        bp_switch_grace -= 1

                    # v4.0: Check for hand retraction
                    # v5.0: Added look-ahead confirmation via decision tree
                    # v5.1: Skip retraction/return checks during BP switch grace period
                    if bp_switch_grace == 0 and self._detect_hand_retraction(df, frame, reach_max_x, slit_x):
                        should_end, bp_switched = self._evaluate_end_candidate(df, frame, reach_max_x, slit_x)
                        if should_end:
                            end_reach = True
                            end_frame = frame - 1  # End at last frame before retraction
                        elif bp_switched:
                            # v5.1: BP switch detected — enter grace period AND
                            # correct reach_max_x. The grace period skips retraction
                            # checks for BP_SWITCH_GRACE frames while DLC stabilizes.
                            # The max correction prevents the inflated max from
                            # causing false retraction triggers after the grace period.
                            bp_switch_grace = self.BP_SWITCH_GRACE
                            if hand_x is not None and hand_x < reach_max_x:
                                reach_max_x = hand_x

                    # v4.0: Check if hand returned to starting position
                    # v5.0: Also requires decision tree approval
                    elif bp_switch_grace == 0 and self._hand_returned_to_start(hand_x, slit_x, reach_max_x):
                        should_end, bp_switched = self._evaluate_end_candidate(df, frame, reach_max_x, slit_x)
                        if should_end:
                            end_reach = True
                            end_frame = frame - 1
                        elif bp_switched:
                            bp_switch_grace = self.BP_SWITCH_GRACE
                            if hand_x is not None and hand_x < reach_max_x:
                                reach_max_x = hand_x

                    if not end_reach:
                        # Continue tracking reach
                        reach_data.append((frame, hand_x, hand_y))
                        if hand_x and hand_x > reach_max_x:
                            reach_max_x = hand_x

                if end_reach and reach_data:
                    # Finalize the reach
                    actual_start = reach_start
                    actual_end = end_frame
                    duration = actual_end - actual_start + 1

                    if duration >= self.MIN_REACH_DURATION:
                        # Find apex (max X extension)
                        max_x = 0
                        apex_frame = actual_start
                        for f, x, y in reach_data:
                            if x is not None and x > max_x:
                                max_x = x
                                apex_frame = f

                        reach_num += 1
                        self.global_reach_id += 1
                        extent_pixels = max_x - boxr_x if max_x else 0
                        extent_ruler = extent_pixels / ruler_pixels if ruler_pixels > 0 else 0

                        # Compute boundary confidence (v3.3)
                        conf = self._compute_reach_confidence(df, actual_start, actual_end)

                        reaches.append(Reach(
                            reach_id=self.global_reach_id,
                            reach_num=reach_num,
                            start_frame=actual_start,
                            apex_frame=apex_frame,
                            end_frame=actual_end,
                            duration_frames=duration,
                            max_extent_pixels=round(extent_pixels, 1),
                            max_extent_ruler=round(extent_ruler, 3),
                            confidence=conf['confidence'],
                            start_confidence=conf['start_confidence'],
                            end_confidence=conf['end_confidence']
                        ))

                    # Reset state
                    in_reach = False
                    reach_start = None
                    reach_data = []
                    reach_max_x = 0
                    consecutive_invisible = 0
                    consecutive_start_visible = 0
                    pending_start_frame = None
                    bp_switch_grace = 0

                    # Check if this frame starts a new reach (hand still visible)
                    # v5.0: Still requires START_CONFIRM, so just seed the counter
                    if hand_visible and nose_engaged:
                        consecutive_start_visible = 1
                        pending_start_frame = frame
                        if self.START_CONFIRM <= 1:
                            in_reach = True
                            reach_start = frame
                            reach_data = [(frame, hand_x, hand_y)]
                            reach_max_x = hand_x if hand_x else 0
                            consecutive_start_visible = 0
                            pending_start_frame = None

        # Handle reach that extends to end of segment
        if in_reach and reach_data:
            actual_start = reach_start
            actual_end = reach_data[-1][0]
            duration = actual_end - actual_start + 1

            if duration >= self.MIN_REACH_DURATION:
                max_x = 0
                apex_frame = actual_start
                for f, x, y in reach_data:
                    if x is not None and x > max_x:
                        max_x = x
                        apex_frame = f

                reach_num += 1
                self.global_reach_id += 1
                extent_pixels = max_x - boxr_x if max_x else 0
                extent_ruler = extent_pixels / ruler_pixels if ruler_pixels > 0 else 0

                # Compute boundary confidence (v3.3)
                conf = self._compute_reach_confidence(df, actual_start, actual_end)

                reaches.append(Reach(
                    reach_id=self.global_reach_id,
                    reach_num=reach_num,
                    start_frame=actual_start,
                    apex_frame=apex_frame,
                    end_frame=actual_end,
                    duration_frames=duration,
                    max_extent_pixels=round(extent_pixels, 1),
                    max_extent_ruler=round(extent_ruler, 3),
                    confidence=conf['confidence'],
                    start_confidence=conf['start_confidence'],
                    end_confidence=conf['end_confidence']
                ))

        # v3.5: Refined negative extent filter (data-driven from GT analysis 2026-01)
        # - Small negative extent (-2 to -15px) = valid attempt reaches (KEEP)
        # - Large negative extent (< -15px) = non-reach hand visibility (FILTER)
        # 44% of false positives had extent below -10px - these are typically
        # hand visibility during grooming/positioning, not actual reach attempts
        before_filter = len(reaches)
        reaches = [r for r in reaches if r.max_extent_pixels >= self.MIN_EXTENT_THRESHOLD]
        if len(reaches) < before_filter:
            filtered_count = before_filter - len(reaches)
            # Silently filter - these are non-reach hand movements

        # Post-processing: merge reaches separated by small gaps (tracking dropout)
        if len(reaches) > 1 and self.GAP_TOLERANCE > 0:
            reaches = self._merge_close_reaches(reaches, boxr_x, ruler_pixels)

        # Post-processing: split long reaches using multi-signal approach (v5.0)
        # Uses hand position + velocity + confidence to place split boundaries
        # precisely, replacing the old confidence-only splitter that was ~5 frames
        # early on 643/2608 GT reaches (24.7% of all errors).
        split_reaches = []
        for reach in reaches:
            sub_boundaries = split_reach_boundaries(
                reach.start_frame, reach.end_frame, df, slit_x,
                self.RH_POINTS, self.SPLIT_THRESHOLD_FRAMES,
                self.CONFIDENCE_HIGH, self.CONFIDENCE_LOW,
                self.MIN_REACH_DURATION,
            )
            if len(sub_boundaries) == 1 and sub_boundaries[0] == (reach.start_frame, reach.end_frame):
                # No split needed - keep original reach
                split_reaches.append(reach)
            else:
                # Create sub-reaches from split boundaries
                for sub_start, sub_end in sub_boundaries:
                    sub_duration = sub_end - sub_start + 1
                    if sub_duration < self.MIN_REACH_DURATION:
                        continue

                    # Find apex (max X extension)
                    max_x = 0
                    apex_frame = sub_start
                    for f in range(sub_start, min(sub_end + 1, len(df))):
                        hand_x, _, _ = self._get_best_hand_position(df.iloc[f])
                        if hand_x is not None and hand_x > max_x:
                            max_x = hand_x
                            apex_frame = f

                    extent_pixels = max_x - boxr_x if max_x else 0
                    extent_ruler = extent_pixels / ruler_pixels if ruler_pixels > 0 else 0

                    # Filter by minimum extent
                    if extent_pixels < self.MIN_EXTENT_THRESHOLD:
                        continue

                    conf = self._compute_reach_confidence(df, sub_start, sub_end)

                    self.global_reach_id += 1
                    split_reaches.append(Reach(
                        reach_id=self.global_reach_id,
                        reach_num=0,  # Renumbered below
                        start_frame=sub_start,
                        apex_frame=apex_frame,
                        end_frame=sub_end,
                        duration_frames=sub_duration,
                        max_extent_pixels=round(extent_pixels, 1),
                        max_extent_ruler=round(extent_ruler, 3),
                        confidence=conf['confidence'],
                        start_confidence=conf['start_confidence'],
                        end_confidence=conf['end_confidence'],
                    ))

        # v5.2: Apply ML boundary polishing
        # Uses trained XGBoost models to correct boundary placement.
        # Conservative: only corrects boundaries where classifier is confident.
        if self._polisher is not None and self._polisher.loaded:
            split_reaches = self._polisher.polish_reaches(split_reaches, df, slit_x)

        # Renumber reaches sequentially within segment
        for i, reach in enumerate(split_reaches):
            reach.reach_num = i + 1

        return split_reaches

    def _get_best_hand_likelihood(self, row) -> float:
        """Get the highest hand point likelihood."""
        best_l = 0
        for p in self.RH_POINTS:
            l = row.get(f'{p}_likelihood', 0)
            if l > best_l:
                best_l = l
        return best_l

    def _compute_reach_confidence(self, df: pd.DataFrame, start_frame: int, end_frame: int) -> Dict[str, float]:
        """
        Compute confidence based on boundary clarity (v3.3).

        Confidence measures how certain we are about the start and end frame boundaries.
        - start_confidence: likelihood jump at reach start (hand appears)
        - end_confidence: likelihood drop at reach end (hand disappears)
        - confidence: min(start, end) - both boundaries must be clear

        Returns:
            Dict with 'confidence', 'start_confidence', 'end_confidence' (0.0-1.0)
        """
        def get_best_likelihood(frame_idx: int) -> float:
            if frame_idx < 0 or frame_idx >= len(df):
                return 0.0
            row = df.iloc[frame_idx]
            return max(row.get(f'{p}_likelihood', 0) for p in self.RH_POINTS)

        # Start boundary: likelihood jump (hand appears)
        # Higher jump = clearer boundary = higher confidence
        before_start = get_best_likelihood(start_frame - 1)
        at_start = get_best_likelihood(start_frame)
        start_conf = min(1.0, max(0.0, (at_start - before_start) + 0.5))

        # End boundary: likelihood drop (hand disappears)
        # Higher drop = clearer boundary = higher confidence
        at_end = get_best_likelihood(end_frame)
        after_end = get_best_likelihood(end_frame + 1)
        end_conf = min(1.0, max(0.0, (at_end - after_end) + 0.5))

        return {
            'confidence': round(min(start_conf, end_conf), 3),
            'start_confidence': round(start_conf, 3),
            'end_confidence': round(end_conf, 3)
        }

    # NOTE: _find_split_points and _split_long_reach were removed in v5.0.
    # Split logic now lives in boundary_refiner.split_reach_boundaries()
    # which uses position + velocity + confidence (not just confidence).
    # Old code archived at: Archive/code_snapshots/reach_detector_v4.2_*.py

    def _merge_close_reaches(
        self,
        reaches: List[Reach],
        boxr_x: float,
        ruler_pixels: float
    ) -> List[Reach]:
        """Merge reaches separated by small gaps (tracking dropouts)."""
        if not reaches:
            return reaches

        merged = []
        current = reaches[0]

        for i in range(1, len(reaches)):
            next_reach = reaches[i]
            gap = next_reach.start_frame - current.end_frame

            if gap <= self.GAP_TOLERANCE:
                # Merge: extend current reach to include next
                new_end = next_reach.end_frame
                new_duration = new_end - current.start_frame + 1

                # Recalculate max extent
                new_max_x = max(
                    current.max_extent_pixels + boxr_x,
                    next_reach.max_extent_pixels + boxr_x
                )
                new_extent_pixels = new_max_x - boxr_x
                new_extent_ruler = new_extent_pixels / ruler_pixels if ruler_pixels > 0 else 0

                # Determine apex
                if next_reach.max_extent_pixels > current.max_extent_pixels:
                    new_apex = next_reach.apex_frame
                else:
                    new_apex = current.apex_frame

                current = Reach(
                    reach_id=current.reach_id,
                    reach_num=current.reach_num,
                    start_frame=current.start_frame,
                    apex_frame=new_apex,
                    end_frame=new_end,
                    duration_frames=new_duration,
                    max_extent_pixels=round(new_extent_pixels, 1),
                    max_extent_ruler=round(new_extent_ruler, 3)
                )
            else:
                # Gap too large - finalize current and start new
                merged.append(current)
                current = next_reach

        merged.append(current)

        # Renumber reaches
        for i, reach in enumerate(merged):
            reach.reach_num = i + 1

        return merged
    
    def detect(self, dlc_path: Path, segments_path: Path) -> VideoReaches:
        """Main entry point: detect all reaches in a video."""
        self.global_reach_id = 0
        
        df = load_dlc(dlc_path)
        boundaries = load_segments(segments_path)
        
        video_name = Path(dlc_path).stem
        if 'DLC_' in video_name:
            video_name = video_name.split('DLC_')[0]
        
        boxr_x = get_boxr_reference(df)

        # Each boundary marks the START of a segment
        # n_segments = len(boundaries), not len(boundaries) - 1
        # Last segment goes from boundaries[-1] to end of video
        n_segments = len(boundaries)

        segment_results = []
        total_reaches = 0
        all_durations = []
        all_extents = []

        for seg_idx in range(n_segments):
            seg_start = boundaries[seg_idx]
            # End at next boundary, or end of video for last segment
            if seg_idx + 1 < len(boundaries):
                seg_end = boundaries[seg_idx + 1]
            else:
                seg_end = len(df)
            segment_num = seg_idx + 1
            
            geom = compute_segment_geometry(df, seg_start, seg_end, segment_num)
            reaches = self.detect_reaches_in_segment(
                df, seg_start, seg_end, boxr_x, geom.ruler_pixels
            )
            
            segment_results.append(SegmentReaches(
                segment_num=segment_num,
                start_frame=seg_start,
                end_frame=seg_end,
                ruler_pixels=round(geom.ruler_pixels, 1),
                n_reaches=len(reaches),
                reaches=reaches,
                flagged_for_review=False,
                flag_reason=None
            ))
            
            total_reaches += len(reaches)
            all_durations.extend([r.duration_frames for r in reaches])
            all_extents.extend([r.max_extent_ruler for r in reaches])
        
        summary = {
            'total_reaches': total_reaches,
            'n_segments': n_segments,
            'reaches_per_segment_mean': round(total_reaches / n_segments, 1) if n_segments else 0,
            'reaches_per_segment_std': round(np.std([s.n_reaches for s in segment_results]), 1),
            'mean_duration_frames': round(np.mean(all_durations), 1) if all_durations else 0,
            'mean_extent_ruler': round(np.mean(all_extents), 3) if all_extents else 0,
        }
        
        return VideoReaches(
            detector_version=VERSION,
            video_name=video_name,
            total_frames=len(df),
            boxr_x=round(boxr_x, 1),
            n_segments=n_segments,
            segments=segment_results,
            summary=summary,
            detected_at=datetime.now().isoformat(),
            validated=False,
            validated_by=None,
            validated_at=None,
            corrections_made=0,
            reaches_added=0,
            reaches_removed=0,
            segments_flagged=0
        )
    
    @staticmethod
    def save_results(results: VideoReaches, output_path: Path, validation_status: str = "needs_review") -> None:
        """Save results to JSON with validation_status.

        Args:
            results: Detection results
            output_path: Output file path
            validation_status: Initial status - "needs_review" (default) or "auto_approved"
        """
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
                "reach_count": results.summary.total_reaches if results.summary else 0,
                "reach_version": results.detector_version,
                "reach_validation": validation_status,
            })
            index.save()
        except Exception:
            pass  # Don't fail reach detection if index update fails

        # Sync to central database
        try:
            from mousereach.sync.database import sync_file_to_database
            sync_file_to_database(output_path)
        except Exception:
            pass  # Don't fail reach detection if database sync fails

    @staticmethod
    def load_results(path: Path) -> VideoReaches:
        """Load results from JSON"""
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct dataclasses, handling optional fields
        segments = []
        for seg_data in data['segments']:
            reaches = []
            for r in seg_data['reaches']:
                # Handle optional fields with defaults for older files
                reach = Reach(
                    reach_id=r['reach_id'],
                    reach_num=r['reach_num'],
                    start_frame=r['start_frame'],
                    apex_frame=r['apex_frame'],
                    end_frame=r['end_frame'],
                    duration_frames=r['duration_frames'],
                    max_extent_pixels=r['max_extent_pixels'],
                    max_extent_ruler=r['max_extent_ruler'],
                    source=r.get('source', 'algorithm'),
                    human_corrected=r.get('human_corrected', False),
                    original_start=r.get('original_start'),
                    original_end=r.get('original_end'),
                    review_note=r.get('review_note'),
                    exclude_from_analysis=r.get('exclude_from_analysis', False),
                    exclude_reason=r.get('exclude_reason')
                )
                reaches.append(reach)
            
            segment = SegmentReaches(
                segment_num=seg_data['segment_num'],
                start_frame=seg_data['start_frame'],
                end_frame=seg_data['end_frame'],
                ruler_pixels=seg_data['ruler_pixels'],
                n_reaches=seg_data['n_reaches'],
                reaches=reaches,
                flagged_for_review=seg_data.get('flagged_for_review', False),
                flag_reason=seg_data.get('flag_reason')
            )
            segments.append(segment)
        
        return VideoReaches(
            detector_version=data['detector_version'],
            video_name=data['video_name'],
            total_frames=data['total_frames'],
            boxr_x=data['boxr_x'],
            n_segments=data['n_segments'],
            segments=segments,
            summary=data['summary'],
            detected_at=data['detected_at'],
            validated=data.get('validated', False),
            validated_by=data.get('validated_by'),
            validated_at=data.get('validated_at'),
            corrections_made=data.get('corrections_made', 0),
            reaches_added=data.get('reaches_added', 0),
            reaches_removed=data.get('reaches_removed', 0),
            segments_flagged=data.get('segments_flagged', 0)
        )


def detect_reaches(dlc_path: str, segments_path: str) -> VideoReaches:
    """Convenience function for reach detection."""
    detector = ReachDetector()
    return detector.detect(Path(dlc_path), Path(segments_path))
