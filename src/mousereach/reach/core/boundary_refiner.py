"""
boundary_refiner.py - Multi-signal reach boundary refinement

Replaces the confidence-only splitter with a multi-pass approach that uses
hand position, velocity, AND confidence to place split boundaries precisely.

The key insight: humans place reach boundaries where the hand is most retracted
(closest to slit), not where DLC confidence first drops. The old splitter backed
up to "last high-confidence frame before dip" which was ~5 frames too early in
643/2608 GT reaches (24.7% of all errors).

Architecture:
    State Machine (existing, unchanged)
      → Raw reaches (good existence, imprecise boundaries)
        → Phase 1: Multi-Signal Split (this module)
          → Phase 2: Boundary Refinement (future)
            → Phase 3: Reach Validation (future)
              → Final reaches
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd


# Minimum likelihood for trusting position data. Below this, DLC position
# estimates are essentially random. We use a lower threshold than the main
# hand visibility threshold (0.5) so we get position data during confidence
# dips, which is exactly where we need it most.
POSITION_TRUST_THRESHOLD = 0.15


@dataclass
class FrameSignal:
    """Per-frame tracking signals for boundary analysis."""
    frame: int
    hand_x: Optional[float]
    hand_y: Optional[float]
    likelihood: float
    offset: Optional[float]     # hand_x - slit_x (positive = past slit)
    velocity_x: Optional[float]  # frame-to-frame hand_x delta


@dataclass
class SplitCandidate:
    """A candidate region for splitting a long reach."""
    drop_idx: int         # Index in signals where confidence first dropped
    rise_idx: int         # Index in signals where confidence rose back
    drop_frame: int       # Frame number where confidence first dropped
    rise_frame: int       # Frame number where confidence rose back
    min_conf: float       # Minimum confidence in the region
    min_x_idx: int        # Index of minimum hand_x in region
    min_x_frame: int      # Frame with minimum hand_x
    min_hand_x: Optional[float]  # Minimum hand_x value in region
    pre_max_x: float      # Max hand_x before the drop
    has_velocity_reversal: bool
    score: float = 0.0


def compute_frame_signals(
    df: pd.DataFrame,
    start: int,
    end: int,
    slit_x: float,
    hand_points: List[str]
) -> List[FrameSignal]:
    """Compute position, velocity, and confidence for each frame.

    Uses a lower confidence threshold (POSITION_TRUST_THRESHOLD) for position
    data than for hand visibility detection, so we get position estimates
    during DLC confidence dips.
    """
    signals = []

    for f in range(start, min(end + 1, len(df))):
        row = df.iloc[f]

        # Get best hand position - use any point above POSITION_TRUST_THRESHOLD
        # for position data (even low confidence gives usable position estimates)
        best_x, best_y, best_l = None, None, 0.0
        for p in hand_points:
            l = row.get(f'{p}_likelihood', 0)
            if l > best_l:
                best_l = l
                if l >= POSITION_TRUST_THRESHOLD:
                    best_x = row.get(f'{p}_x', None)
                    best_y = row.get(f'{p}_y', None)

        signals.append(FrameSignal(
            frame=f,
            hand_x=best_x,
            hand_y=best_y,
            likelihood=best_l,
            offset=(best_x - slit_x) if best_x is not None else None,
            velocity_x=None,
        ))

    # Compute velocity as frame-to-frame delta of hand_x
    for i in range(1, len(signals)):
        if signals[i].hand_x is not None and signals[i - 1].hand_x is not None:
            signals[i].velocity_x = signals[i].hand_x - signals[i - 1].hand_x

    return signals


def _find_confidence_dips(
    signals: List[FrameSignal],
    conf_high: float,
    conf_low: float
) -> List[SplitCandidate]:
    """Find confidence dip regions (high → low → high transitions).

    This is similar to the old _find_split_points but returns richer
    information about each dip region for multi-signal scoring.
    """
    candidates = []
    in_drop = False
    drop_idx = None

    for i in range(1, len(signals)):
        prev_l = signals[i - 1].likelihood
        curr_l = signals[i].likelihood

        if not in_drop:
            if prev_l >= conf_high and curr_l < conf_low:
                in_drop = True
                drop_idx = i - 1  # Last high-confidence index
        else:
            if curr_l >= conf_high and prev_l < conf_low:
                # Complete dip found: high → low → high
                rise_idx = i

                # Compute features of this candidate region
                region = signals[drop_idx:rise_idx + 1]

                # Min confidence in region
                min_conf = min(s.likelihood for s in region)

                # Running max hand_x before the drop
                pre_max_x = 0.0
                for s in signals[:drop_idx + 1]:
                    if s.hand_x is not None and s.hand_x > pre_max_x:
                        pre_max_x = s.hand_x

                # Min hand_x in region (hand closest to slit)
                min_x_idx = drop_idx
                min_hand_x = None
                for j in range(drop_idx, min(rise_idx + 1, len(signals))):
                    s = signals[j]
                    if s.hand_x is not None:
                        if min_hand_x is None or s.hand_x < min_hand_x:
                            min_hand_x = s.hand_x
                            min_x_idx = j

                # Check for velocity reversal in region
                # (velocity goes negative then positive = retract then re-extend)
                has_vel_rev = False
                saw_negative = False
                for j in range(drop_idx, min(rise_idx + 1, len(signals))):
                    v = signals[j].velocity_x
                    if v is not None:
                        if v < -0.5:  # Retracting (moving toward slit)
                            saw_negative = True
                        elif v > 0.5 and saw_negative:  # Re-extending after retraction
                            has_vel_rev = True
                            break

                candidates.append(SplitCandidate(
                    drop_idx=drop_idx,
                    rise_idx=rise_idx,
                    drop_frame=signals[drop_idx].frame,
                    rise_frame=signals[rise_idx].frame,
                    min_conf=min_conf,
                    min_x_idx=min_x_idx,
                    min_x_frame=signals[min_x_idx].frame,
                    min_hand_x=min_hand_x,
                    pre_max_x=pre_max_x,
                    has_velocity_reversal=has_vel_rev,
                ))

                in_drop = False
            elif curr_l >= conf_high:
                # Confidence rose but not from low - end drop detection
                in_drop = False

    return candidates


def _score_candidate(candidate: SplitCandidate, slit_x: float) -> float:
    """Score a split candidate based on multiple signals.

    Returns 0.0-1.0. Only split if score > 0.5 (at least 2 signals agree).

    Signal weights:
        - Position return (0.4): strongest signal - hand moved back toward slit
        - Confidence dip (0.3): DLC lost tracking in the gap
        - Velocity reversal (0.3): hand changed direction
    """
    score = 0.0

    # Signal 1: Confidence dip depth (max 0.3)
    # conf_high is 0.5, so a drop to 0.2 = 0.3 drop = max score
    conf_drop = max(0, 0.5 - candidate.min_conf)
    score += min(conf_drop / 0.3, 1.0) * 0.3

    # Signal 2: Position return toward slit (max 0.4)
    if candidate.min_hand_x is not None and candidate.pre_max_x > 0:
        extension = candidate.pre_max_x - slit_x
        if extension > 1:
            retraction = candidate.pre_max_x - candidate.min_hand_x
            position_return_pct = retraction / extension
            score += min(position_return_pct / 0.3, 1.0) * 0.4

    # Signal 3: Velocity reversal (max 0.3)
    if candidate.has_velocity_reversal:
        score += 0.3

    return score


def _find_precise_boundary(
    candidate: SplitCandidate,
    signals: List[FrameSignal]
) -> Tuple[int, int]:
    """Find the precise frame where first sub-reach ends and second begins.

    Strategy priority:
    1. Hand position minimum (closest to slit) = where hand actually retracted
    2. Last frame with positive velocity before reversal
    3. Confidence dip center (fallback)

    Returns (end_frame_of_first, start_frame_of_second).
    """
    region = signals[candidate.drop_idx:candidate.rise_idx + 1]

    # Strategy 1: Hand position minimum
    valid_positions = [(i, s.frame, s.hand_x) for i, s in enumerate(region)
                       if s.hand_x is not None]

    if valid_positions:
        _, min_frame, min_x = min(valid_positions, key=lambda p: p[2])

        # Verify it's a meaningful retraction (not just noise)
        if candidate.pre_max_x > 0 and min_x < candidate.pre_max_x - 3:
            # End first reach at the position minimum
            # Start second reach at the rise point (where confidence recovers)
            return min_frame, signals[candidate.rise_idx].frame

    # Strategy 2: Last frame with positive velocity before reversal
    for s in reversed(region):
        if s.velocity_x is not None and s.velocity_x > 0.5:
            return s.frame, signals[candidate.rise_idx].frame

    # Strategy 3: Confidence dip center
    center_idx = (candidate.drop_idx + candidate.rise_idx) // 2
    return signals[center_idx].frame, signals[candidate.rise_idx].frame


def _find_position_returns(
    signals: List[FrameSignal],
    slit_x: float,
    min_extension: float = 10.0,
    return_fraction: float = 0.50,
    min_re_extend: float = 10.0,
) -> List[SplitCandidate]:
    """Find position return patterns (extend → retract → re-extend).

    v5.1: Detects merged reaches that don't have confidence dips.
    The hand stays visible but clearly returns toward the slit and then
    extends again — indicating two separate reaches.

    A position return is detected when:
    1. Hand extends past slit by at least min_extension pixels
    2. Hand retracts by at least return_fraction of the extension
    3. Hand then re-extends by at least min_re_extend pixels

    The split point is at the position minimum (hand closest to slit).
    """
    candidates = []

    # Find running max hand_x (tracks peak extension)
    running_max = slit_x
    running_max_idx = 0

    i = 0
    while i < len(signals):
        s = signals[i]
        if s.hand_x is None:
            i += 1
            continue

        # Update running max
        if s.hand_x > running_max:
            running_max = s.hand_x
            running_max_idx = i

        # Check if hand has retracted significantly from running max
        extension = running_max - slit_x
        if extension < min_extension:
            i += 1
            continue

        retraction = running_max - s.hand_x
        if retraction < extension * return_fraction:
            i += 1
            continue

        # Hand has retracted — find the minimum position
        min_x = s.hand_x
        min_x_idx = i

        j = i + 1
        while j < len(signals):
            sj = signals[j]
            if sj.hand_x is None:
                j += 1
                continue
            if sj.hand_x < min_x:
                min_x = sj.hand_x
                min_x_idx = j
            elif sj.hand_x > min_x + min_re_extend:
                # Hand re-extended — this is a valid position return
                break
            j += 1
        else:
            # Didn't re-extend sufficiently — not a valid split
            i += 1
            continue

        # Valid position return found: peak at running_max_idx, trough at min_x_idx
        # Check for velocity reversal in the region
        has_vel_rev = False
        saw_negative = False
        for k in range(running_max_idx, min(j + 1, len(signals))):
            v = signals[k].velocity_x
            if v is not None:
                if v < -0.5:
                    saw_negative = True
                elif v > 0.5 and saw_negative:
                    has_vel_rev = True
                    break

        candidates.append(SplitCandidate(
            drop_idx=running_max_idx,
            rise_idx=j,
            drop_frame=signals[running_max_idx].frame,
            rise_frame=signals[j].frame,
            min_conf=min(signals[k].likelihood for k in range(running_max_idx, min(j + 1, len(signals)))),
            min_x_idx=min_x_idx,
            min_x_frame=signals[min_x_idx].frame,
            min_hand_x=min_x,
            pre_max_x=running_max,
            has_velocity_reversal=has_vel_rev,
            score=0.0,
        ))

        # Reset: start tracking from the re-extension point
        running_max = signals[j].hand_x if signals[j].hand_x else slit_x
        running_max_idx = j
        i = j + 1

    return candidates


def split_reach_boundaries(
    start: int,
    end: int,
    df: pd.DataFrame,
    slit_x: float,
    hand_points: List[str],
    split_threshold: int = 25,
    conf_high: float = 0.5,
    conf_low: float = 0.35,
    min_duration: int = 4,
    min_split_score: float = 0.5,
) -> List[Tuple[int, int]]:
    """Split a long reach into sub-reaches using multiple signals.

    This is the main entry point for Phase 1 of the boundary refinement pipeline.
    Returns list of (start_frame, end_frame) tuples.

    Args:
        start: Reach start frame
        end: Reach end frame
        df: DLC DataFrame
        slit_x: Slit center x-position
        hand_points: List of hand bodypart names
        split_threshold: Only split reaches longer than this (frames)
        conf_high: "High" confidence level
        conf_low: "Low" confidence level (dip detection)
        min_duration: Minimum sub-reach duration
        min_split_score: Minimum score to accept a split candidate

    Returns:
        List of (start_frame, end_frame) tuples. Returns [(start, end)]
        unchanged if no split needed.
    """
    duration = end - start + 1
    if duration <= split_threshold:
        return [(start, end)]

    # Step 1: Compute per-frame signals
    signals = compute_frame_signals(df, start, end, slit_x, hand_points)
    if len(signals) < 2:
        return [(start, end)]

    # Step 2: Find candidate split regions from BOTH confidence dips AND position returns
    conf_candidates = _find_confidence_dips(signals, conf_high, conf_low)
    pos_candidates = _find_position_returns(signals, slit_x)

    # Score confidence-dip candidates
    for c in conf_candidates:
        c.score = _score_candidate(c, slit_x)

    # Score position-return candidates (they already have strong position signal)
    for c in pos_candidates:
        c.score = _score_candidate(c, slit_x)

    # Merge candidates, deduplicate overlapping regions
    all_candidates = conf_candidates + pos_candidates
    if not all_candidates:
        return [(start, end)]

    # Step 3: Filter to candidates that pass the threshold
    good_candidates = [c for c in all_candidates if c.score >= min_split_score]
    if not good_candidates:
        return [(start, end)]

    # Deduplicate: if two candidates overlap (split frames within 5 of each other),
    # keep the one with the higher score
    good_candidates.sort(key=lambda c: c.min_x_frame)
    deduped = [good_candidates[0]]
    for c in good_candidates[1:]:
        if abs(c.min_x_frame - deduped[-1].min_x_frame) < 5:
            # Overlap — keep higher score
            if c.score > deduped[-1].score:
                deduped[-1] = c
        else:
            deduped.append(c)
    good_candidates = deduped

    # Sort by frame order
    good_candidates.sort(key=lambda c: c.drop_frame)

    # Step 5: Place precise boundaries
    split_boundaries = []
    for c in good_candidates:
        end_first, start_second = _find_precise_boundary(c, signals)
        split_boundaries.append((end_first, start_second))

    # Step 6: Build sub-reach list from boundaries
    sub_reaches = []
    current_start = start

    for end_first, start_second in split_boundaries:
        # Validate: end must be after current_start
        if end_first <= current_start:
            continue

        sub_duration = end_first - current_start + 1
        if sub_duration >= min_duration:
            sub_reaches.append((current_start, end_first))

        current_start = start_second

    # Last sub-reach: from last split to original end
    last_duration = end - current_start + 1
    if last_duration >= min_duration:
        sub_reaches.append((current_start, end))

    # If splitting produced nothing valid, return original
    if not sub_reaches:
        return [(start, end)]

    return sub_reaches
