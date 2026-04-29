"""
Signal 27: tray-motion gate for segment boundaries.

A real ASPA (Automatic Single Pellet Apparatus) cycle produces:
  1. SA corner peak-to-peak excursion -- the tray fixed points
     (SABL/SABR/SATL/SATR) sweep through a wide range of x positions
     (>= 30 px) within a 30-frame motion window. Operator alignment
     adjustments only produce 5-15 px shifts.
  2. Pillar likelihood drop -- a fresh pellet appears on the pillar
     after the cycle, occluding the pillar tip. Pillar likelihood drops
     sharply (e.g., 1.00 -> 0.03-0.13). Operator adjustments do NOT
     cause pillar lk to drop because no new pellet appears.

If a candidate boundary fails BOTH conditions, the segmenter inserted
it in error (operator adjustment, tray-stick artifact, or other
spurious source). The boundary is rejected and replaced with a
median-cadence projection from a valid neighbor.

This is the canonical physical test for whether a segment boundary is
real -- see ``tray_motion_segment_boundary_test.md`` memory entry and
case 36 (``20251009_CNT0307_P4 segment 18``) of the v4.0.0_dev outcome
walkthrough where the gate would have rejected a spurious boundary at
frame 30995.

Initial implementation (v2.2.x): the simple two-test gate (excursion +
pillar lk drop). Iterate-and-evaluate principle applies -- additional
sub-tests (direction reversal, slit crossing, pellet decoupling) are
deferred until the simple gate's eval reports motivate them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Sequence, Tuple


# Default thresholds. Calibrated values land here once eval iterations
# settle on stable numbers; placeholders below come from the memory
# entry's user-stated bounds.
DEFAULT_EXCURSION_THRESHOLD = 30.0   # px peak-to-peak across SA corner
DEFAULT_PILLAR_LK_DROP_THRESHOLD = 0.3  # median pre - median post
DEFAULT_WINDOW = 50                   # frames each side of boundary
DEFAULT_SA_BODYPARTS = ("SABL", "SABR", "SATL", "SATR")


def validate_tray_motion(
    df: pd.DataFrame,
    boundary_frame: int,
    window: int = DEFAULT_WINDOW,
    excursion_threshold: float = DEFAULT_EXCURSION_THRESHOLD,
    pillar_lk_drop_threshold: float = DEFAULT_PILLAR_LK_DROP_THRESHOLD,
    sa_bodyparts: Sequence[str] = DEFAULT_SA_BODYPARTS,
) -> Tuple[bool, List[str]]:
    """Validate a candidate boundary against the tray-motion test.

    Returns
    -------
    is_valid : bool
        True if the boundary corresponds to a real ASPA cycle.
    reasons : list of str
        Human-readable reasons each sub-test failed (empty if valid).
    """
    n = len(df)
    if not (0 <= boundary_frame < n):
        return False, [f"boundary_out_of_range (frame={boundary_frame}, n={n})"]

    pre_start = max(0, boundary_frame - window)
    pre_end = boundary_frame
    post_start = boundary_frame
    post_end = min(n, boundary_frame + window + 1)

    reasons: List[str] = []

    # Test 1: SA corner peak-to-peak excursion in [F-window, F+window].
    # Real ASPA cycles sweep at least 30 px in at least one corner's x.
    excursion_pass = False
    max_excursion = 0.0
    best_bp = None
    for bp in sa_bodyparts:
        col = f"{bp}_x"
        if col not in df.columns:
            continue
        seg = df[col].values[pre_start:post_end]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue
        excursion = float(np.max(seg) - np.min(seg))
        if excursion > max_excursion:
            max_excursion = excursion
            best_bp = bp
        if excursion >= excursion_threshold:
            excursion_pass = True
            break
    if not excursion_pass:
        reasons.append(
            f"insufficient_tray_excursion "
            f"(max={max_excursion:.1f}px on {best_bp}, threshold={excursion_threshold:.1f})"
        )

    # Test 2: pillar likelihood drops across boundary (new pellet now
    # occludes pillar tip). Operator adjustments don't introduce a new
    # pellet so pillar lk stays high.
    pillar_lk_pass = True  # default pass if pillar column unavailable
    if "Pillar_likelihood" in df.columns:
        pre = df["Pillar_likelihood"].values[pre_start:pre_end]
        post = df["Pillar_likelihood"].values[post_start:post_end]
        pre = pre[~np.isnan(pre)]
        post = post[~np.isnan(post)]
        if len(pre) > 0 and len(post) > 0:
            pre_med = float(np.median(pre))
            post_med = float(np.median(post))
            drop = pre_med - post_med
            if drop < pillar_lk_drop_threshold:
                pillar_lk_pass = False
                reasons.append(
                    f"pillar_lk_no_drop "
                    f"(pre={pre_med:.2f}, post={post_med:.2f}, drop={drop:.2f}, "
                    f"threshold={pillar_lk_drop_threshold:.2f})"
                )
        # else: not enough data either side; keep default pass

    is_valid = excursion_pass and pillar_lk_pass
    return is_valid, reasons


def replace_invalid_boundaries(
    boundaries: Sequence[int],
    validity_flags: Sequence[bool],
    total_frames: int,
    expected_interval: float,
) -> List[int]:
    """Replace invalid boundaries with median-cadence projections.

    For each invalid boundary at index i, project from the nearest valid
    neighbor using the median cadence over remaining valid boundaries.
    If no valid anchor is available, fall back to ``expected_interval``.

    The returned list always has the same length as ``boundaries`` and is
    clamped to ``[0, total_frames - 1]``.
    """
    boundaries = list(boundaries)
    flags = list(validity_flags)
    n = len(boundaries)
    if n == 0 or all(flags):
        return boundaries

    valid_boundaries = [b for b, v in zip(boundaries, flags) if v]
    if len(valid_boundaries) >= 2:
        median_interval = float(np.median(np.diff(valid_boundaries)))
    else:
        median_interval = float(expected_interval)

    new_boundaries = list(boundaries)
    for i, valid in enumerate(flags):
        if valid:
            continue
        # Try projecting from the previous valid boundary, then the next.
        prev_valid_idx = next((j for j in range(i - 1, -1, -1) if flags[j]), None)
        next_valid_idx = next((j for j in range(i + 1, n) if flags[j]), None)
        if prev_valid_idx is not None:
            steps = i - prev_valid_idx
            new_boundaries[i] = int(new_boundaries[prev_valid_idx] + steps * median_interval)
        elif next_valid_idx is not None:
            steps = next_valid_idx - i
            new_boundaries[i] = int(new_boundaries[next_valid_idx] - steps * median_interval)
        # else: no anchor anywhere -- leave the original frame.

    new_boundaries = [max(0, min(total_frames - 1, b)) for b in new_boundaries]
    return sorted(new_boundaries)


def apply_tray_motion_gate(
    df: pd.DataFrame,
    boundaries: Sequence[int],
    total_frames: int,
    expected_interval: float,
    window: int = DEFAULT_WINDOW,
    excursion_threshold: float = DEFAULT_EXCURSION_THRESHOLD,
    pillar_lk_drop_threshold: float = DEFAULT_PILLAR_LK_DROP_THRESHOLD,
    sa_bodyparts: Sequence[str] = DEFAULT_SA_BODYPARTS,
) -> Tuple[List[int], List[Tuple[int, int, List[str]]]]:
    """Validate every boundary and substitute invalid ones with projections.

    Returns
    -------
    filtered_boundaries : list of int
        Length equals input ``boundaries``. Invalid entries replaced with
        median-cadence projections from valid neighbors.
    rejections : list of (index, original_frame, reasons)
        One entry per rejected boundary.
    """
    flags: List[bool] = []
    rejections: List[Tuple[int, int, List[str]]] = []
    for idx, b in enumerate(boundaries):
        is_valid, reasons = validate_tray_motion(
            df, b,
            window=window,
            excursion_threshold=excursion_threshold,
            pillar_lk_drop_threshold=pillar_lk_drop_threshold,
            sa_bodyparts=sa_bodyparts,
        )
        flags.append(is_valid)
        if not is_valid:
            rejections.append((idx, b, reasons))

    filtered = replace_invalid_boundaries(
        boundaries, flags, total_frames, expected_interval
    )
    return filtered, rejections
