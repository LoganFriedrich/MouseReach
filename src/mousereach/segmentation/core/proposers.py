"""
Multi-proposer boundary candidate generators (v2.2.0).

Each proposer independently produces boundary candidates from a single
apparatus-tied signal. Proposers fire when the tray advances. The
combiner (see consensus.py) merges candidates across proposers.

Frame-source proposers (aligned with GT, mean offset < 0.2 f):
  - SABL: primary signal source
  - SATL: secondary signal source

Confirmer-only proposers (systematic ~7 f early bias relative to GT;
used only to vote for consensus, never to set the final frame):
  - SABR
  - SATR

Disabled:
  - Pellet (likelihood/position too noisy; see FINAL_DESIGN.md)
  - Pillar (unreliable per user; frequently occluded by pellet)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .segmenter_robust import (
    load_dlc,
    get_clean_signal,
    compute_velocity,
    assess_reference_quality,
)


@dataclass
class Candidate:
    """A boundary candidate from a single proposer."""
    frame: int
    proposer: str       # 'SABL', 'SABR', 'SATL', 'SATR', 'Pellet'
    velocity: float     # peak velocity in detection window
    position: float     # relative position at candidate frame
    confidence: float   # proposer-local confidence [0,1]
    notes: List[str] = field(default_factory=list)


def _group_events(candidate_frames: np.ndarray, min_gap: int) -> List[List[int]]:
    """Group consecutive candidate frames into events separated by min_gap."""
    if len(candidate_frames) == 0:
        return []
    events = []
    event_frames = [candidate_frames[0]]
    for i in range(1, len(candidate_frames)):
        if candidate_frames[i] - candidate_frames[i - 1] > min_gap:
            events.append(event_frames)
            event_frames = []
        event_frames.append(candidate_frames[i])
    events.append(event_frames)
    return events


def _pick_best_frame(event_frames: List[int], rel: np.ndarray,
                     vel: np.ndarray, center_target: float) -> Tuple[int, float]:
    """Pick the best frame from an event based on position and velocity."""
    best_frame = None
    best_score = -np.inf
    for f in event_frames:
        pos_score = -abs(rel[f] - center_target)
        vel_score = vel[f]
        score = pos_score + vel_score * 2
        if score > best_score:
            best_score = score
            best_frame = f
    pos_confidence = max(0, 1 - abs(rel[best_frame] - center_target) / 10)
    vel_confidence = min(1, vel[best_frame] / 2.5)
    confidence = (pos_confidence + vel_confidence) / 2
    return best_frame, confidence


def sa_proposer(df: pd.DataFrame, bodypart: str, box_center: float,
                center_range: Tuple[float, float] = (-5, 10),
                vel_threshold: float = 0.8,
                min_gap: int = 25,
                smooth_window: int = 30,
                center_target: float = 2.5,
                endpoint_vel_threshold: float = 1.4) -> List[Candidate]:
    """Generic SA corner proposer.

    Two passes:
      1. Centered crossings: position in `center_range` AND velocity
         above `vel_threshold`. Catches internal boundaries reliably.
      2. Endpoint rescue: velocity-only peaks above
         `endpoint_vel_threshold` before the first / after the last
         primary candidate. Catches B1/B21 where position has drifted
         past the centered window.
    """
    sig = get_clean_signal(df, bodypart, 'x')
    if sig is None:
        return []

    rel = sig - box_center
    vel = compute_velocity(sig, smooth_window=smooth_window)

    # Pass 1: centered crossings
    centered = (rel > center_range[0]) & (rel < center_range[1])
    moving = vel > vel_threshold
    candidate_frames = np.where(centered & moving)[0]
    events = _group_events(candidate_frames, min_gap)

    candidates: List[Candidate] = []
    for event_frames in events:
        best_frame, confidence = _pick_best_frame(
            event_frames, rel, vel, center_target)
        candidates.append(Candidate(
            frame=best_frame,
            proposer=bodypart,
            velocity=float(vel[best_frame]),
            position=float(rel[best_frame]),
            confidence=float(confidence),
            notes=[f"primary, pos={rel[best_frame]:.1f}, vel={vel[best_frame]:.2f}"],
        ))

    # Pass 2: endpoint rescue
    if len(candidates) >= 15:
        cand_frames = sorted([c.frame for c in candidates])
        first_cand = cand_frames[0]
        last_cand = cand_frames[-1]

        if len(cand_frames) >= 2:
            intervals = np.diff(cand_frames)
            valid_intervals = intervals[(intervals > 500) & (intervals < 5000)]
            cadence = float(np.median(valid_intervals)) if len(valid_intervals) > 0 else 1839.0
        else:
            cadence = 1839.0

        # Missing B1?
        if first_cand > cadence * 0.7:
            b1_search_end = first_cand - min_gap
            if b1_search_end > 0:
                search_vel = vel[:b1_search_end]
                high_vel_frames = np.where(search_vel > endpoint_vel_threshold)[0]
                if len(high_vel_frames) > 0:
                    ep_events = _group_events(high_vel_frames, min_gap)
                    for event_frames in ep_events:
                        best_f = event_frames[np.argmax(vel[event_frames])]
                        ep_conf = min(1, vel[best_f] / 2.5) * 0.85
                        if all(abs(best_f - c.frame) > min_gap for c in candidates):
                            candidates.append(Candidate(
                                frame=int(best_f),
                                proposer=bodypart,
                                velocity=float(vel[best_f]),
                                position=float(rel[best_f]),
                                confidence=float(ep_conf),
                                notes=[f"endpoint_rescue_B1, vel={vel[best_f]:.2f}"],
                            ))

        # Missing B21?
        b21_search_start = last_cand + min_gap
        if b21_search_start < len(vel):
            search_vel = vel[b21_search_start:]
            high_vel_frames = np.where(search_vel > endpoint_vel_threshold)[0]
            if len(high_vel_frames) > 0:
                high_vel_frames = high_vel_frames + b21_search_start
                ep_events = _group_events(high_vel_frames, min_gap)
                for event_frames in ep_events:
                    best_f = event_frames[np.argmax(vel[event_frames])]
                    ep_conf = min(1, vel[best_f] / 2.5) * 0.85
                    if all(abs(best_f - c.frame) > min_gap for c in candidates):
                        candidates.append(Candidate(
                            frame=int(best_f),
                            proposer=bodypart,
                            velocity=float(vel[best_f]),
                            position=float(rel[best_f]),
                            confidence=float(ep_conf),
                            notes=[f"endpoint_rescue_B21, vel={vel[best_f]:.2f}"],
                        ))

    return candidates


def get_all_sa_candidates(df: pd.DataFrame, box_center: float,
                          vel_threshold: float = 0.8,
                          min_gap: int = 25,
                          center_range: Tuple[float, float] = (-5, 10),
                          center_target: float = 2.5) -> List[Candidate]:
    """Run all four SA corner proposers and return combined candidates."""
    all_candidates: List[Candidate] = []
    for bp in ['SABL', 'SABR', 'SATL', 'SATR']:
        candidates = sa_proposer(
            df, bp, box_center,
            center_range=center_range,
            vel_threshold=vel_threshold,
            min_gap=min_gap,
            center_target=center_target,
        )
        all_candidates.extend(candidates)
    return all_candidates


def pellet_swap_proposer(df: pd.DataFrame,
                         like_drop_threshold: float = 0.2,
                         pos_shift_threshold: float = 8.0,
                         time_window: int = 30,
                         min_gap: int = 25) -> List[Candidate]:
    """Pellet-swap proposer (currently disabled by default).

    Detects when the pellet likelihood drops or position shifts within a
    short window. Intended as a weak confirmer, but in practice the
    pellet signal is noisy enough to add 30-40 spurious candidates per
    video. Disabled in production (MultiProposerConfig.pellet_enabled =
    False). Kept here for future tuning work.
    """
    like_col = 'Pellet_likelihood'
    px_col = 'Pellet_x'
    py_col = 'Pellet_y'
    if like_col not in df.columns or px_col not in df.columns:
        return []

    like = df[like_col].values
    px = df[px_col].values
    py = df[py_col].values
    n = len(like)

    pos_shift = np.zeros(n)
    for i in range(time_window, n):
        dx = np.median(px[i:min(n, i + time_window)]) - np.median(px[max(0, i - time_window):i])
        dy = np.median(py[i:min(n, i + time_window)]) - np.median(py[max(0, i - time_window):i])
        pos_shift[i] = np.sqrt(dx ** 2 + dy ** 2)

    like_smooth = pd.Series(like).rolling(window=5, center=True, min_periods=1).mean().values
    like_drop = np.zeros(n)
    for i in range(time_window, n):
        before = np.mean(like_smooth[max(0, i - time_window):i])
        after = np.mean(like_smooth[i:min(n, i + time_window)])
        like_drop[i] = before - after

    candidates_raw = np.where(
        (pos_shift > pos_shift_threshold) | (like_drop > like_drop_threshold)
    )[0]
    if len(candidates_raw) == 0:
        return []

    events = []
    event_frames = [candidates_raw[0]]
    for i in range(1, len(candidates_raw)):
        if candidates_raw[i] - candidates_raw[i - 1] > min_gap:
            events.append(event_frames)
            event_frames = []
        event_frames.append(candidates_raw[i])
    events.append(event_frames)

    candidates: List[Candidate] = []
    for event_frames in events:
        if not event_frames:
            continue
        best_frame = max(event_frames, key=lambda f: pos_shift[f] + like_drop[f] * 10)
        shift_conf = min(1, pos_shift[best_frame] / 20.0)
        drop_conf = min(1, max(0, like_drop[best_frame]) / 0.5)
        confidence = max(shift_conf, drop_conf) * 0.6
        candidates.append(Candidate(
            frame=int(best_frame),
            proposer='Pellet',
            velocity=float(pos_shift[best_frame]),
            position=float(like_drop[best_frame]),
            confidence=float(confidence),
            notes=[f"shift={pos_shift[best_frame]:.1f}, drop={like_drop[best_frame]:.3f}"],
        ))

    return candidates
