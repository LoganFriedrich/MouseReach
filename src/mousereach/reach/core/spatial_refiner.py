"""
spatial_refiner.py - Multi-pass spatial refinement for reach boundaries

Stage 4 in the reach detection pipeline. Operates on reaches AFTER the
boundary polisher (XGBoost). Each pass applies one specific spatial/physical
constraint that the confidence-threshold-based detection may have missed.

Pass order:
  1. AbsorbedReachSplitter  - Split merged reaches using position valleys
  2. LateEndTrimmer         - Trim ends where hand returned past slit
  3. ShortFalsePositiveFilter - Remove short "reaches" behind slit
  4. EarlyStartCorrector    - Trim starts with hand behind slit

Design principles:
  - Each pass is independent and testable
  - Passes use SPATIAL constraints (where is hand relative to slit?)
    not confidence thresholds (does DLC believe hand exists?)
  - Conservative: better to skip than to break a correct reach
  - Auditable: every action recorded with evidence
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .boundary_refiner import compute_frame_signals, POSITION_TRUST_THRESHOLD, FrameSignal
from .reach_detector import Reach


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_REACH_DURATION = 4  # Never create or trim a reach below this duration

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

# Threshold for trusting individual hand point position when computing
# mean hand x. Higher than POSITION_TRUST_THRESHOLD because mean-hand
# checks need reliable positions, not speculative low-confidence ones.
HAND_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

@dataclass
class RefinementAction:
    """Record of a single refinement decision for audit trail.

    Every modification (split, trim, removal, correction) produces one of
    these so the user can inspect exactly what changed and why.
    """
    pass_name: str            # Which pass produced this action
    reach_id: int             # Original reach_id acted upon
    action: str               # "split", "trim_end", "trim_start", "remove"
    original_start: int       # Start frame before action
    original_end: int         # End frame before action
    new_start: Optional[int] = None   # Start frame after (None if removed)
    new_end: Optional[int] = None     # End frame after (None if removed)
    evidence: str = ""        # Human-readable explanation of why
    sub_reach_count: Optional[int] = None  # For splits: how many pieces


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RefinementPass(ABC):
    """Common interface for all spatial refinement passes.

    Each pass receives reaches and the DLC DataFrame plus geometry, and
    returns a (possibly modified) list of reaches plus any actions taken.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this pass (used in audit trail)."""
        ...

    @abstractmethod
    def run(
        self,
        reaches: List[Reach],
        df: pd.DataFrame,
        slit_x: float,
        slit_y: float,
        boxr_x: float,
        ruler_pixels: float,
        hand_points: List[str],
    ) -> Tuple[List[Reach], List[RefinementAction]]:
        """Apply this pass to the reach list.

        Args:
            reaches: Current list of Reach objects (sorted by start_frame).
            df: Full DLC DataFrame for the video.
            slit_x: Slit center x (midpoint of BOXL and BOXR).
            slit_y: Slit center y.
            boxr_x: Right edge of slit opening (physical boundary).
            ruler_pixels: SABL-SABR distance in pixels for unit conversion.
            hand_points: List of hand bodypart column prefixes.

        Returns:
            (updated_reaches, actions) -- reaches sorted by start_frame.
        """
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpatialRefinerConfig:
    """Centralized thresholds for all spatial refinement passes.

    Calibrated from GT position distributions (calibrate_thresholds.py):
      - Hand during GT reaches: mean +2.4px from slit_x (P25=-2.3, P75=+6.9)
      - Hand outside reaches: mean -0.2px from slit_x
      - First 2 frames: mean -3.6px from slit_x
      - Last 2 frames: mean +4.0px from slit_x
      - boxr_x is ~9.4px right of slit_x (half-slit width)
      - Hand is almost NEVER past boxr_x even during valid reaches (12.7%)
    All offsets are from slit_x (NOT boxr_x).
    """

    # --- Pass 1: AbsorbedReachSplitter ---
    min_duration_to_split: int = 15       # Skip reaches shorter than this
    valley_threshold: float = 2.0         # Offset (from slit_x) below which = "valley"
    extension_threshold: float = 5.0      # Offset above which = "extended" (P75=+6.9)
    return_to_slit_px: float = 2.0        # Valley must drop to within this of slit_x
    re_extension_px: float = 5.0          # Must re-extend past slit by this much
    nose_proximity_threshold: float = 25.0  # Nose must be within this of slit (px)
    smoothing_window: int = 3             # Moving average window for offset smoothing

    # --- Pass 2: LateEndTrimmer ---
    end_confidence_skip: float = 0.8      # Skip if end_confidence above this
    past_slit_threshold: float = 0.0      # Offset threshold: hand must be AT/BEHIND slit_x
    min_trim_frames: int = 3              # Only trim if trimming >= this many frames
    sustain_frames: int = 4               # Hand must stay behind slit for this many frames

    # --- Pass 3: ShortFalsePositiveFilter ---
    # NOTE: Uses slit_x (not boxr_x). During GT reaches hand averages +2.4px
    # from slit_x; only 62.9% of frames are past slit_x at all.
    # For short GT reaches, mean past-slit ratio (slit_x) is ~63%.
    # We use a very low threshold to only catch the clearest non-reaches.
    max_duration_to_check: int = 10       # Only check reaches this short
    min_past_slit_ratio: float = 0.10     # Remove only if <10% of frames past slit_x

    # --- Pass 4: EarlyStartCorrector ---
    # First 2 frames of GT reaches average -3.6px from slit_x (P95=+1.1).
    # Only correct when hand is VERY far behind slit (well below P5=-8.6).
    check_frames: int = 3                 # Check this many frames at reach start
    min_behind_count: int = 3             # All checked frames must be behind threshold
    behind_slit_threshold: float = -10.0  # Hand must be this far behind slit_x to correct


# ---------------------------------------------------------------------------
# Pass 1: AbsorbedReachSplitter
# ---------------------------------------------------------------------------

class AbsorbedReachSplitter(RefinementPass):
    """Split long reaches that contain 2+ actual reaches merged together.

    Detection relies on position valleys: the hand extends past the slit,
    retracts back near the slit, then extends again. Each extension-retraction
    cycle is a separate reach. A nose engagement check at the re-extension
    confirms reaching intent (vs. grooming).

    Uses a 3-frame moving average on hand offset to dampen single-frame
    DLC position noise.
    """

    def __init__(self, config: SpatialRefinerConfig):
        self._cfg = config

    @property
    def name(self) -> str:
        return "AbsorbedReachSplitter"

    def run(
        self,
        reaches: List[Reach],
        df: pd.DataFrame,
        slit_x: float,
        slit_y: float,
        boxr_x: float,
        ruler_pixels: float,
        hand_points: List[str],
    ) -> Tuple[List[Reach], List[RefinementAction]]:
        output: List[Reach] = []
        actions: List[RefinementAction] = []
        next_id = _next_reach_id(reaches)

        for reach in reaches:
            if reach.duration_frames <= self._cfg.min_duration_to_split:
                output.append(reach)
                continue

            split_points = self._find_split_points(
                reach, df, slit_x, slit_y, hand_points
            )

            if not split_points:
                output.append(reach)
                continue

            # Build sub-reaches from split points
            boundaries = self._build_sub_boundaries(
                reach.start_frame, reach.end_frame, split_points
            )

            sub_reaches: List[Reach] = []
            for sub_start, sub_end in boundaries:
                duration = sub_end - sub_start + 1
                if duration < MIN_REACH_DURATION:
                    continue
                apex_frame, extent_px = _compute_apex_and_extent(
                    df, sub_start, sub_end, boxr_x, hand_points
                )
                extent_ruler = extent_px / ruler_pixels if ruler_pixels > 0 else 0.0
                sub_reaches.append(Reach(
                    reach_id=next_id,
                    reach_num=0,
                    start_frame=sub_start,
                    apex_frame=apex_frame,
                    end_frame=sub_end,
                    duration_frames=duration,
                    max_extent_pixels=round(extent_px, 1),
                    max_extent_ruler=round(extent_ruler, 3),
                    confidence=None,
                    start_confidence=None,
                    end_confidence=None,
                    source="algorithm",
                ))
                next_id += 1

            if len(sub_reaches) >= 2:
                output.extend(sub_reaches)
                actions.append(RefinementAction(
                    pass_name=self.name,
                    reach_id=reach.reach_id,
                    action="split",
                    original_start=reach.start_frame,
                    original_end=reach.end_frame,
                    new_start=sub_reaches[0].start_frame,
                    new_end=sub_reaches[-1].end_frame,
                    evidence=(
                        f"Found {len(split_points)} position valley(s) in "
                        f"{reach.duration_frames}-frame reach; split into "
                        f"{len(sub_reaches)} sub-reaches"
                    ),
                    sub_reach_count=len(sub_reaches),
                ))
            else:
                # Splitting didn't produce 2+ valid sub-reaches; keep original
                output.append(reach)

        output.sort(key=lambda r: r.start_frame)
        return output, actions

    # -- internals --

    def _find_split_points(
        self,
        reach: Reach,
        df: pd.DataFrame,
        slit_x: float,
        slit_y: float,
        hand_points: List[str],
    ) -> List[int]:
        """Return frame numbers where the reach should be split."""
        signals = compute_frame_signals(
            df, reach.start_frame, reach.end_frame, slit_x, hand_points
        )
        if len(signals) < self._cfg.smoothing_window:
            return []

        # Extract raw offsets (None -> NaN for smoothing)
        raw_offsets = np.array(
            [s.offset if s.offset is not None else np.nan for s in signals],
            dtype=float,
        )

        # 3-frame moving average (ignoring NaN)
        smoothed = _moving_average(raw_offsets, self._cfg.smoothing_window)

        # Find valleys: extended -> valley -> re-extended
        cfg = self._cfg
        split_frames: List[int] = []
        was_extended = False
        valley_start_idx: Optional[int] = None
        valley_min_offset = np.inf
        valley_min_idx: Optional[int] = None

        for i, offset in enumerate(smoothed):
            if np.isnan(offset):
                continue

            if not was_extended:
                if offset > cfg.extension_threshold:
                    was_extended = True
                continue

            # We were extended; check if we dropped into a valley
            if offset <= cfg.valley_threshold:
                if valley_start_idx is None:
                    valley_start_idx = i
                if offset < valley_min_offset:
                    valley_min_offset = offset
                    valley_min_idx = i
            elif offset > cfg.extension_threshold and valley_start_idx is not None:
                # Rose back above extension threshold -- valley complete
                if (
                    valley_min_offset <= cfg.return_to_slit_px
                    and offset > cfg.re_extension_px
                    and valley_min_idx is not None
                ):
                    # Nose engagement check at re-extension frame
                    re_ext_frame = signals[i].frame
                    if self._is_nose_near_slit(
                        df, re_ext_frame, slit_x, slit_y,
                        cfg.nose_proximity_threshold
                    ):
                        split_frames.append(signals[valley_min_idx].frame)

                # Reset valley tracking; we are extended again
                valley_start_idx = None
                valley_min_offset = np.inf
                valley_min_idx = None
                # was_extended stays True

        return split_frames

    @staticmethod
    def _is_nose_near_slit(
        df: pd.DataFrame,
        frame: int,
        slit_x: float,
        slit_y: float,
        threshold: float = 25.0,
    ) -> bool:
        """Check whether the nose is close to the slit at a given frame.

        If nose position is unavailable (low likelihood or missing column),
        returns True so we don't block the split -- better to split
        optimistically than to silently swallow a valid split.
        """
        if frame < 0 or frame >= len(df):
            return True
        row = df.iloc[frame]
        nose_x = row.get('Nose_x', np.nan)
        nose_y = row.get('Nose_y', np.nan)
        nose_l = row.get('Nose_likelihood', 0)
        if nose_l < 0.3 or np.isnan(nose_x):
            return True  # Can't determine, don't block the split
        return float(np.sqrt((nose_x - slit_x)**2 + (nose_y - slit_y)**2)) < threshold

    @staticmethod
    def _build_sub_boundaries(
        start: int, end: int, split_frames: List[int]
    ) -> List[Tuple[int, int]]:
        """Convert split frame list into (start, end) boundary pairs.

        Split frames become the end of the preceding sub-reach. The next
        sub-reach starts at split_frame + 1.
        """
        boundaries: List[Tuple[int, int]] = []
        current_start = start
        for sf in sorted(split_frames):
            if sf <= current_start:
                continue
            boundaries.append((current_start, sf))
            current_start = sf + 1
        if current_start <= end:
            boundaries.append((current_start, end))
        return boundaries


# ---------------------------------------------------------------------------
# Pass 2: LateEndTrimmer
# ---------------------------------------------------------------------------

class LateEndTrimmer(RefinementPass):
    """Trim reach ends that extend beyond where the hand returned past the slit.

    After the boundary polisher, some reaches still include frames at the
    end where the hand has already retracted behind the slit. This pass
    scans backward from the end to find the last frame the hand was past
    the slit opening, and trims the end if the hand demonstrably stayed
    behind the slit for several consecutive frames afterward.
    """

    def __init__(self, config: SpatialRefinerConfig):
        self._cfg = config

    @property
    def name(self) -> str:
        return "LateEndTrimmer"

    def run(
        self,
        reaches: List[Reach],
        df: pd.DataFrame,
        slit_x: float,
        slit_y: float,
        boxr_x: float,
        ruler_pixels: float,
        hand_points: List[str],
    ) -> Tuple[List[Reach], List[RefinementAction]]:
        output: List[Reach] = []
        actions: List[RefinementAction] = []

        for reach in reaches:
            # Skip if end boundary is already high-confidence
            if (
                reach.end_confidence is not None
                and reach.end_confidence > self._cfg.end_confidence_skip
            ):
                output.append(reach)
                continue

            signals = compute_frame_signals(
                df, reach.start_frame, reach.end_frame, slit_x, hand_points
            )
            if len(signals) < 2:
                output.append(reach)
                continue

            new_end = self._find_trim_point(
                signals, reach.apex_frame, reach.end_frame
            )

            if new_end is None:
                output.append(reach)
                continue

            frames_trimmed = reach.end_frame - new_end
            if frames_trimmed < self._cfg.min_trim_frames:
                output.append(reach)
                continue

            # Verify hand stayed behind slit after trim point
            if not self._verify_sustain(df, new_end, reach.end_frame, slit_x, hand_points):
                output.append(reach)
                continue

            new_duration = new_end - reach.start_frame + 1
            if new_duration < MIN_REACH_DURATION:
                output.append(reach)
                continue

            # Recompute apex if it fell beyond new end
            apex = reach.apex_frame
            if apex > new_end:
                apex, _ = _compute_apex_and_extent(
                    df, reach.start_frame, new_end, boxr_x, hand_points
                )

            extent_px = _max_hand_x_in_range(
                df, reach.start_frame, new_end, hand_points
            ) - boxr_x
            extent_ruler = extent_px / ruler_pixels if ruler_pixels > 0 else 0.0

            actions.append(RefinementAction(
                pass_name=self.name,
                reach_id=reach.reach_id,
                action="trim_end",
                original_start=reach.start_frame,
                original_end=reach.end_frame,
                new_start=reach.start_frame,
                new_end=new_end,
                evidence=(
                    f"Hand offset dropped below {self._cfg.past_slit_threshold}px "
                    f"at frame {new_end}; trimmed {frames_trimmed} trailing frames"
                ),
            ))

            output.append(Reach(
                reach_id=reach.reach_id,
                reach_num=reach.reach_num,
                start_frame=reach.start_frame,
                apex_frame=apex,
                end_frame=new_end,
                duration_frames=new_duration,
                max_extent_pixels=round(extent_px, 1),
                max_extent_ruler=round(extent_ruler, 3),
                confidence=reach.confidence,
                start_confidence=reach.start_confidence,
                end_confidence=reach.end_confidence,
                source=reach.source,
                human_corrected=reach.human_corrected,
                original_start=reach.original_start,
                original_end=reach.original_end,
                review_note=reach.review_note,
                exclude_from_analysis=reach.exclude_from_analysis,
                exclude_reason=reach.exclude_reason,
            ))

        output.sort(key=lambda r: r.start_frame)
        return output, actions

    # -- internals --

    def _find_trim_point(
        self,
        signals: List[FrameSignal],
        apex_frame: int,
        end_frame: int,
    ) -> Optional[int]:
        """Scan backward from end toward apex to find last past-slit frame.

        Returns the frame number to trim to, or None if no trim warranted.
        """
        last_past_slit_frame: Optional[int] = None

        for s in reversed(signals):
            # Don't trim before the apex
            if s.frame <= apex_frame:
                break
            if s.offset is not None and s.offset > self._cfg.past_slit_threshold:
                last_past_slit_frame = s.frame
                break

        if last_past_slit_frame is None:
            return None
        if last_past_slit_frame >= end_frame:
            return None  # Already at end, no trimming needed
        return last_past_slit_frame

    def _verify_sustain(
        self,
        df: pd.DataFrame,
        trim_frame: int,
        original_end: int,
        slit_x: float,
        hand_points: List[str],
    ) -> bool:
        """Verify the hand stayed at/behind slit for sustain_frames after trim point.

        This prevents trimming on a single-frame dip where the hand
        immediately re-extends.
        """
        sustain_needed = self._cfg.sustain_frames
        consecutive_behind = 0

        for f in range(trim_frame + 1, min(original_end + 1, len(df))):
            row = df.iloc[f]
            best_x = _best_hand_x(row, hand_points)
            if best_x is None:
                # Hand not visible -- counts as "not past slit"
                consecutive_behind += 1
            elif best_x - slit_x <= self._cfg.past_slit_threshold:
                consecutive_behind += 1
            else:
                consecutive_behind = 0  # Reset: hand re-extended

            if consecutive_behind >= sustain_needed:
                return True

        # If we ran out of frames, accept if we accumulated enough
        return consecutive_behind >= sustain_needed


# ---------------------------------------------------------------------------
# Pass 3: ShortFalsePositiveFilter
# ---------------------------------------------------------------------------

class ShortFalsePositiveFilter(RefinementPass):
    """Remove short "reaches" (4-10 frames) where the hand never crosses the slit.

    Very short reaches are most likely to be tracking artifacts. If the
    mean hand position (averaged across all visible hand points) stays
    behind the slit opening for most of the reach, it is not a real reach.
    """

    def __init__(self, config: SpatialRefinerConfig):
        self._cfg = config

    @property
    def name(self) -> str:
        return "ShortFalsePositiveFilter"

    def run(
        self,
        reaches: List[Reach],
        df: pd.DataFrame,
        slit_x: float,
        slit_y: float,
        boxr_x: float,
        ruler_pixels: float,
        hand_points: List[str],
    ) -> Tuple[List[Reach], List[RefinementAction]]:
        output: List[Reach] = []
        actions: List[RefinementAction] = []

        for reach in reaches:
            if reach.duration_frames > self._cfg.max_duration_to_check:
                output.append(reach)
                continue

            # Use slit_x (center), NOT boxr_x. During GT reaches the hand
            # averages only +2.4px from slit_x but -7.5px from boxr_x.
            frames_past, frames_with_position = self._count_past_slit(
                df, reach.start_frame, reach.end_frame, slit_x
            )

            if frames_with_position == 0:
                # No position data at all -- keep conservatively
                output.append(reach)
                continue

            past_slit_ratio = frames_past / frames_with_position

            if past_slit_ratio < self._cfg.min_past_slit_ratio:
                actions.append(RefinementAction(
                    pass_name=self.name,
                    reach_id=reach.reach_id,
                    action="remove",
                    original_start=reach.start_frame,
                    original_end=reach.end_frame,
                    evidence=(
                        f"Short reach ({reach.duration_frames} frames): "
                        f"only {frames_past}/{frames_with_position} frames "
                        f"({past_slit_ratio:.0%}) had mean hand past slit "
                        f"(threshold: {self._cfg.min_past_slit_ratio:.0%})"
                    ),
                ))
                # Do NOT append to output -- reach is removed
            else:
                output.append(reach)

        output.sort(key=lambda r: r.start_frame)
        return output, actions

    # -- internals --

    def _count_past_slit(
        self,
        df: pd.DataFrame,
        start: int,
        end: int,
        reference_x: float,
    ) -> Tuple[int, int]:
        """Count frames where mean hand x is past the reference point.

        Uses slit_x (center of slit), not boxr_x.
        Returns (frames_past_reference, total_frames_with_position).
        """
        frames_past = 0
        frames_with_pos = 0

        for f in range(start, min(end + 1, len(df))):
            mean_x = _get_mean_hand_x(df.iloc[f])
            if mean_x is None:
                continue
            frames_with_pos += 1
            if mean_x > reference_x:
                frames_past += 1

        return frames_past, frames_with_pos


# ---------------------------------------------------------------------------
# Pass 4: EarlyStartCorrector
# ---------------------------------------------------------------------------

class EarlyStartCorrector(RefinementPass):
    """Push start frame forward when early frames have the hand behind the slit.

    Pre-reach postural adjustments sometimes trigger the state machine
    before the hand actually crosses the slit. If the first few frames
    consistently show the hand behind the slit opening, the start frame
    is advanced to where the hand first crosses.
    """

    def __init__(self, config: SpatialRefinerConfig):
        self._cfg = config

    @property
    def name(self) -> str:
        return "EarlyStartCorrector"

    def run(
        self,
        reaches: List[Reach],
        df: pd.DataFrame,
        slit_x: float,
        slit_y: float,
        boxr_x: float,
        ruler_pixels: float,
        hand_points: List[str],
    ) -> Tuple[List[Reach], List[RefinementAction]]:
        output: List[Reach] = []
        actions: List[RefinementAction] = []

        for reach in reaches:
            # Use slit_x with behind_slit_threshold, not boxr_x.
            # First 2 frames of GT reaches average -3.6px from slit_x;
            # only correct when hand is VERY far behind (< -10px from slit_x).
            correction = self._find_correction(
                df, reach.start_frame, reach.end_frame, slit_x
            )

            if correction is None:
                output.append(reach)
                continue

            new_start = correction
            new_duration = reach.end_frame - new_start + 1
            if new_duration < MIN_REACH_DURATION:
                output.append(reach)
                continue

            # Recompute apex if it fell before new start (shouldn't happen
            # normally, but be safe)
            apex = reach.apex_frame
            if apex < new_start:
                apex, _ = _compute_apex_and_extent(
                    df, new_start, reach.end_frame, boxr_x, hand_points
                )

            extent_px = _max_hand_x_in_range(
                df, new_start, reach.end_frame, hand_points
            ) - boxr_x
            extent_ruler = extent_px / ruler_pixels if ruler_pixels > 0 else 0.0

            frames_trimmed = new_start - reach.start_frame

            actions.append(RefinementAction(
                pass_name=self.name,
                reach_id=reach.reach_id,
                action="trim_start",
                original_start=reach.start_frame,
                original_end=reach.end_frame,
                new_start=new_start,
                new_end=reach.end_frame,
                evidence=(
                    f"First {frames_trimmed} frame(s) had mean hand behind "
                    f"slit (boxr_x={boxr_x:.1f}); advanced start to frame "
                    f"{new_start}"
                ),
            ))

            output.append(Reach(
                reach_id=reach.reach_id,
                reach_num=reach.reach_num,
                start_frame=new_start,
                apex_frame=apex,
                end_frame=reach.end_frame,
                duration_frames=new_duration,
                max_extent_pixels=round(extent_px, 1),
                max_extent_ruler=round(extent_ruler, 3),
                confidence=reach.confidence,
                start_confidence=reach.start_confidence,
                end_confidence=reach.end_confidence,
                source=reach.source,
                human_corrected=reach.human_corrected,
                original_start=reach.original_start,
                original_end=reach.original_end,
                review_note=reach.review_note,
                exclude_from_analysis=reach.exclude_from_analysis,
                exclude_reason=reach.exclude_reason,
            ))

        output.sort(key=lambda r: r.start_frame)
        return output, actions

    # -- internals --

    def _find_correction(
        self,
        df: pd.DataFrame,
        start: int,
        end: int,
        slit_x: float,
    ) -> Optional[int]:
        """Check first N frames; if all are very far behind slit, return corrected start.

        Only corrects when the hand is well behind slit center (< behind_slit_threshold
        from slit_x). GT data shows first 2 frames of reaches average -3.6px from
        slit_x, so we only correct at -10px or worse.

        Returns the first frame where the hand moves past the threshold, or None
        if no correction is warranted.
        """
        threshold_x = slit_x + self._cfg.behind_slit_threshold  # e.g., slit_x - 10

        check_end = min(start + self._cfg.check_frames, end + 1)
        behind_count = 0

        for f in range(start, min(check_end, len(df))):
            mean_x = _get_mean_hand_x(df.iloc[f])
            if mean_x is not None and mean_x <= threshold_x:
                behind_count += 1

        if behind_count < self._cfg.min_behind_count:
            return None

        # Find the first frame where the hand moves past the threshold
        for f in range(start, min(end + 1, len(df))):
            mean_x = _get_mean_hand_x(df.iloc[f])
            if mean_x is not None and mean_x > threshold_x:
                return f

        return None  # Hand never crossed threshold -- don't alter


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SpatialRefiner:
    """Run all spatial refinement passes sequentially on a reach list.

    Usage::

        refiner = SpatialRefiner()  # or SpatialRefiner(config=custom_config)

        # Simple: just get refined reaches
        refined = refiner.refine(reaches, df, slit_x, slit_y, boxr_x, ruler_pixels)

        # With audit trail
        refined, actions = refiner.refine_with_audit(
            reaches, df, slit_x, slit_y, boxr_x, ruler_pixels
        )
    """

    def __init__(
        self,
        config: Optional[SpatialRefinerConfig] = None,
        hand_points: Optional[List[str]] = None,
    ):
        self.config = config or SpatialRefinerConfig()
        self.hand_points = hand_points or list(RH_POINTS)
        # All passes re-calibrated from GT position distributions
        # (calibrate_thresholds.py). Key fix: all passes now use slit_x
        # (center of slit) instead of boxr_x (right edge, ~9.4px further right).
        self._passes: List[RefinementPass] = [
            AbsorbedReachSplitter(self.config),
            LateEndTrimmer(self.config),
            ShortFalsePositiveFilter(self.config),
            EarlyStartCorrector(self.config),
        ]

    def refine(
        self,
        reaches: List[Reach],
        df: pd.DataFrame,
        slit_x: float,
        slit_y: float,
        boxr_x: float,
        ruler_pixels: float,
    ) -> List[Reach]:
        """Apply all passes and return refined reaches (no audit trail)."""
        refined, _ = self.refine_with_audit(
            reaches, df, slit_x, slit_y, boxr_x, ruler_pixels
        )
        return refined

    def refine_with_audit(
        self,
        reaches: List[Reach],
        df: pd.DataFrame,
        slit_x: float,
        slit_y: float,
        boxr_x: float,
        ruler_pixels: float,
    ) -> Tuple[List[Reach], List[RefinementAction]]:
        """Apply all passes and return (refined_reaches, all_actions).

        Passes run in order. Each pass receives the output of the previous
        pass, so later passes operate on already-refined data.
        """
        all_actions: List[RefinementAction] = []
        current = sorted(reaches, key=lambda r: r.start_frame)

        for rpass in self._passes:
            current, actions = rpass.run(
                current, df, slit_x, slit_y, boxr_x,
                ruler_pixels, self.hand_points,
            )
            all_actions.extend(actions)

        return current, all_actions


# ---------------------------------------------------------------------------
# Shared helpers (module-private)
# ---------------------------------------------------------------------------

def _next_reach_id(reaches: List[Reach]) -> int:
    """Return a starting reach_id that avoids collisions.

    Uses max existing id + 1000 to leave plenty of room. The IDs will be
    renumbered by reach_detector.py after the refiner runs.
    """
    if not reaches:
        return 1000
    return max(r.reach_id for r in reaches) + 1000


def _get_mean_hand_x(row) -> Optional[float]:
    """Get mean X position of all visible hand points.

    Uses HAND_THRESHOLD (0.5) for individual point reliability. Returns
    None if no hand points are visible above threshold.
    """
    xs: List[float] = []
    for p in RH_POINTS:
        likelihood = row.get(f'{p}_likelihood', 0)
        if likelihood >= HAND_THRESHOLD:
            x = row.get(f'{p}_x', np.nan)
            if not np.isnan(x):
                xs.append(float(x))
    return float(np.mean(xs)) if xs else None


def _best_hand_x(row, hand_points: List[str]) -> Optional[float]:
    """Get X position of the most confident hand point.

    Uses POSITION_TRUST_THRESHOLD for maximum coverage (same as
    compute_frame_signals).
    """
    best_x: Optional[float] = None
    best_l = 0.0
    for p in hand_points:
        l = row.get(f'{p}_likelihood', 0)
        if l > best_l and l >= POSITION_TRUST_THRESHOLD:
            x = row.get(f'{p}_x', np.nan)
            if not np.isnan(x):
                best_x = float(x)
                best_l = l
    return best_x


def _compute_apex_and_extent(
    df: pd.DataFrame,
    start: int,
    end: int,
    boxr_x: float,
    hand_points: List[str],
) -> Tuple[int, float]:
    """Find the apex frame (max hand x) and extent (max_x - boxr_x).

    Returns (apex_frame, extent_pixels).
    """
    max_x = 0.0
    apex_frame = start
    for f in range(start, min(end + 1, len(df))):
        row = df.iloc[f]
        # Use best single point (highest confidence) for apex, matching
        # the existing reach_detector logic
        for p in hand_points:
            l = row.get(f'{p}_likelihood', 0)
            if l >= POSITION_TRUST_THRESHOLD:
                x = row.get(f'{p}_x', np.nan)
                if not np.isnan(x) and x > max_x:
                    max_x = float(x)
                    apex_frame = f
    extent = max_x - boxr_x if max_x > 0 else 0.0
    return apex_frame, extent


def _max_hand_x_in_range(
    df: pd.DataFrame,
    start: int,
    end: int,
    hand_points: List[str],
) -> float:
    """Return the maximum hand x position in [start, end]."""
    max_x = 0.0
    for f in range(start, min(end + 1, len(df))):
        row = df.iloc[f]
        for p in hand_points:
            l = row.get(f'{p}_likelihood', 0)
            if l >= POSITION_TRUST_THRESHOLD:
                x = row.get(f'{p}_x', np.nan)
                if not np.isnan(x) and x > max_x:
                    max_x = float(x)
    return max_x


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a centered moving average, ignoring NaN values.

    At the edges where the full window is not available, uses whatever
    values are available (shrinking window). Returns an array of the same
    length as the input; positions where all window elements are NaN
    remain NaN.
    """
    n = len(values)
    result = np.full(n, np.nan)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            result[i] = float(np.mean(valid))

    return result
