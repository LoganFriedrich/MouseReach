"""
Pellet transition moment detector.

Stage X of the residuals-targeted approach (per user 2026-05-03):
scan a segment for moments where SOMETHING happened to the pellet,
without yet trying to attribute the cause to a specific reach. Outputs
candidate transition frames + the signal that flagged them.

Each candidate is a (frame, signal_type, strength) triple. Stage Y
(empirical learning + reach triangulation) consumes these candidates
and converges on the causal reach + outcome class.

Signals:
    1. Pellet on-pillar -> off-pillar:
       Last frame where pellet was confidently on-pillar with high lk
       and paw not past slit, before it transitions away.
    2. Pillar bodypart lk rise:
       Pillar_lk transitions from sustained-low to sustained-high
       (= pellet uncovered the pillar).
    3. Pellet visibility drop:
       Pellet_lk transitions from sustained-high to sustained-low
       (= pellet detection lost).
    4. Pellet position jump:
       Pellet position changes by >> tracking-error between
       consecutive confident frames (= sudden displacement).

The detector is deliberately permissive -- it surfaces candidates;
selecting the right one is downstream work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PELLET_LK_HIGH = 0.95
PELLET_LK_LOW = 0.5
PILLAR_LK_LOW = 0.3
PILLAR_LK_HIGH = 0.5
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0
MIN_SUSTAINED_RUN = 5
POSITION_JUMP_PX = 25.0  # frame-to-frame pellet position jump above
                         # this = potential displacement event


@dataclass
class TransitionCandidate:
    """Candidate transition moment within a segment."""
    frame: int           # absolute frame index in video
    signal_type: str     # 'pellet_left_pillar' | 'pillar_lk_rise' |
                         # 'pellet_vanished' | 'pellet_jumped'
    strength: float      # 0..1, signal confidence
    detail: dict         # signal-specific debug info


def _sustained_runs(arr, min_run):
    """Yield (start_idx, end_idx_inclusive) for True-runs of >= min_run."""
    n = len(arr)
    in_run = False
    rs = -1
    for i in range(n):
        if arr[i]:
            if not in_run:
                rs = i
                in_run = True
        else:
            if in_run:
                if i - rs >= min_run:
                    yield (rs, i - 1)
                in_run = False
    if in_run and n - rs >= min_run:
        yield (rs, n - 1)


def detect_transition_moments(
    seg: SegmentInput,
    transition_zone_half: int = TRANSITION_ZONE_HALF,
    pellet_lk_high: float = PELLET_LK_HIGH,
    pellet_lk_low: float = PELLET_LK_LOW,
    pillar_lk_low: float = PILLAR_LK_LOW,
    pillar_lk_high: float = PILLAR_LK_HIGH,
    paw_lk_threshold: float = PAW_LK_THR,
    on_pillar_radii: float = ON_PILLAR_RADII,
    min_sustained_run: int = MIN_SUSTAINED_RUN,
    position_jump_px: float = POSITION_JUMP_PX,
) -> List[TransitionCandidate]:
    """Return all candidate transition moments in the segment's clean
    zone. Each candidate has a frame (absolute), signal type, and
    strength.
    """
    clean_end = seg.seg_end - transition_zone_half
    if clean_end <= seg.seg_start:
        return []

    sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
    n = len(sub_raw)
    if n == 0:
        return []

    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
    pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
    pillar_r = geom["pillar_r"].to_numpy(dtype=float)
    slit_y_line = pillar_cy + pillar_r

    pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
    pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
    dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2
                          + (pellet_y - pillar_cy) ** 2)
                  / np.maximum(pillar_r, 1e-6))

    pillar_lk_raw = sub_raw["Pillar_likelihood"].to_numpy(dtype=float)

    paw_past_y = np.zeros(n, dtype=bool)
    for bp in PAW_BODYPARTS:
        py = sub[f"{bp}_y"].to_numpy(dtype=float)
        pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
        paw_past_y |= (py <= slit_y_line) & (pl >= paw_lk_threshold)

    candidates: List[TransitionCandidate] = []

    # --- Signal 1: pellet last-confident-on-pillar -> not-on-pillar
    on_pillar = (
        (pellet_lk >= pellet_lk_high)
        & (dist_radii <= on_pillar_radii)
        & (~paw_past_y)
    )
    last_on_pillar_local = -1
    for i in range(n - 1, -1, -1):
        if on_pillar[i]:
            last_on_pillar_local = i
            break
    if last_on_pillar_local >= 0 and last_on_pillar_local < n - 1:
        # The transition frame is the one IMMEDIATELY after the last
        # confident on-pillar frame.
        candidates.append(TransitionCandidate(
            frame=int(seg.seg_start + last_on_pillar_local + 1),
            signal_type='pellet_left_pillar',
            strength=1.0,
            detail={'last_on_pillar_local': last_on_pillar_local},
        ))

    # --- Signal 2: Pillar lk transition via sliding-min-max rolling
    # median (per legacy v4.0.0-step-5 detect_pillar_lk_transition).
    # Catches transitions anywhere in segment using a rolling 100-frame
    # median; fires only when low_med < 0.3 AND high_med > 0.7 AND
    # rise > 0.30 AND low precedes high in time.
    if n >= 200:
        import pandas as pd
        roll = (pd.Series(pillar_lk_raw).rolling(window=100, center=True,
                                                 min_periods=50).median().values)
        valid = ~np.isnan(roll)
        if valid.any():
            idx_low = int(np.nanargmin(roll))
            idx_high = int(np.nanargmax(roll))
            low_med = float(roll[idx_low])
            high_med = float(roll[idx_high])
            rise = high_med - low_med
            fired = (
                (rise > 0.30)
                and (idx_low < idx_high)
                and (low_med < 0.3)
                and (high_med > 0.7)
            )
            if fired:
                # Find first crossing of midpoint after idx_low.
                midpoint = (low_med + high_med) / 2.0
                trans_idx = None
                for i in range(idx_low, len(roll)):
                    if not np.isnan(roll[i]) and roll[i] >= midpoint:
                        trans_idx = i
                        break
                if trans_idx is not None:
                    candidates.append(TransitionCandidate(
                        frame=int(seg.seg_start + trans_idx),
                        signal_type='pillar_lk_rise',
                        strength=min(1.0, rise / 0.5),
                        detail={
                            'low_med': low_med, 'high_med': high_med,
                            'rise': rise, 'idx_low_local': idx_low,
                            'idx_high_local': idx_high,
                        },
                    ))

    # --- Signal 3: Pellet visibility drop (sustained-high to
    # sustained-low) -- requires sustained-high to have happened first.
    pellet_high_lk = (pellet_lk >= pellet_lk_high) & (~paw_past_y)
    pellet_low_lk = (pellet_lk < pellet_lk_low) & (~paw_past_y)
    high_lk_runs = list(_sustained_runs(pellet_high_lk, min_sustained_run))
    if high_lk_runs:
        last_high_end = high_lk_runs[-1][1]
        # First sustained-low after this
        low_lk_runs = list(_sustained_runs(pellet_low_lk, min_sustained_run))
        for ls, le in low_lk_runs:
            if ls > last_high_end:
                candidates.append(TransitionCandidate(
                    frame=int(seg.seg_start + ls),
                    signal_type='pellet_vanished',
                    strength=min(1.0, (le - ls + 1) / 100.0),
                    detail={'last_high_end_local': last_high_end,
                            'low_run_start_local': ls},
                ))
                break

    # --- Signal 4: Pellet position jump between consecutive confident
    # frames -- big position change indicates sudden displacement.
    confident_idxs = [i for i in range(n) if pellet_high_lk[i]]
    for k in range(1, len(confident_idxs)):
        i_prev = confident_idxs[k - 1]
        i_cur = confident_idxs[k]
        if i_cur - i_prev > 30:
            continue  # gap too large; not a frame-to-frame jump
        d = float(np.sqrt((pellet_x[i_cur] - pellet_x[i_prev]) ** 2
                          + (pellet_y[i_cur] - pellet_y[i_prev]) ** 2))
        if d >= position_jump_px:
            candidates.append(TransitionCandidate(
                frame=int(seg.seg_start + i_cur),
                signal_type='pellet_jumped',
                strength=min(1.0, d / 100.0),
                detail={'prev_local': i_prev, 'cur_local': i_cur, 'dist_px': d},
            ))

    return candidates
