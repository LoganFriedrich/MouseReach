"""
Consensus merging for the multi-proposer segmenter (v2.2.0).

Clusters candidates from multiple proposers by frame proximity, picks a
consensus frame (prefer SABL/SATL -- they align with GT; SABR/SATR have
a systematic ~7 f early bias), scores by agreement, and selects the
final 21 boundaries.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .proposers import Candidate


@dataclass
class MergedCandidate:
    """A boundary candidate after consensus merging."""
    frame: int
    n_proposers: int
    proposers: List[str]
    consensus_score: float
    individual_candidates: List[Candidate]
    notes: List[str] = field(default_factory=list)


def cluster_candidates(candidates: List[Candidate],
                       merge_window: int = 30) -> List[List[Candidate]]:
    """Single-linkage clustering by frame proximity within merge_window."""
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda c: c.frame)
    clusters: List[List[Candidate]] = []
    current_cluster = [sorted_cands[0]]
    for i in range(1, len(sorted_cands)):
        if sorted_cands[i].frame - current_cluster[-1].frame <= merge_window:
            current_cluster.append(sorted_cands[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_cands[i]]
    clusters.append(current_cluster)
    return clusters


def merge_cluster(cluster: List[Candidate],
                  frame_proposers: Tuple[str, ...] = ('SABL', 'SATL')) -> MergedCandidate:
    """Merge a cluster of candidates into one MergedCandidate.

    Frame is drawn from SABL or SATL (whichever has higher confidence
    for this boundary). SABR and SATR have a systematic ~7 f early bias
    vs GT; they vote for consensus but never set the frame.
    """
    unique_proposers = list({c.proposer for c in cluster})
    n_unique = len(unique_proposers)

    frame_cands = [c for c in cluster if c.proposer in frame_proposers]
    if frame_cands:
        best_cand = max(frame_cands, key=lambda c: c.confidence)
    else:
        best_cand = max(cluster, key=lambda c: c.confidence)
    frame = best_cand.frame

    sum_conf = sum(c.confidence for c in cluster)
    avg_conf = sum_conf / len(cluster)
    if n_unique >= 4:
        consensus_score = min(1.0, avg_conf * 1.5)
    elif n_unique >= 3:
        consensus_score = min(1.0, avg_conf * 1.3)
    elif n_unique >= 2:
        consensus_score = min(1.0, avg_conf * 1.1)
    else:
        consensus_score = avg_conf * 0.9

    return MergedCandidate(
        frame=int(frame),
        n_proposers=n_unique,
        proposers=unique_proposers,
        consensus_score=float(consensus_score),
        individual_candidates=cluster,
        notes=[f"n_prop={n_unique}, avg_conf={avg_conf:.3f}"],
    )


def build_consensus(candidates: List[Candidate],
                    merge_window: int = 30) -> List[MergedCandidate]:
    """Cluster, merge, and return MergedCandidates sorted by frame."""
    clusters = cluster_candidates(candidates, merge_window=merge_window)
    merged = [merge_cluster(c) for c in clusters]
    merged.sort(key=lambda m: m.frame)
    return merged


def select_boundaries(merged: List[MergedCandidate],
                      total_frames: int,
                      n_expected: int = 21,
                      expected_interval: float = 1839.0,
                      ) -> Tuple[List[int], List[str], List[MergedCandidate]]:
    """Select exactly n_expected boundaries from merged candidates.

    Mirrors v2.1.3's fit_grid_to_candidates but uses consensus_score
    instead of single-proposer confidence when ranking drops.
    """
    anomalies: List[str] = []

    if len(merged) == 0:
        interval = total_frames / (n_expected + 1)
        anomalies.append("No candidates found - using evenly spaced fallback")
        return [int((i + 1) * interval) for i in range(n_expected)], anomalies, []

    frames = [m.frame for m in merged]

    if len(frames) >= 2:
        intervals = np.diff(frames)
        valid = intervals[(intervals > expected_interval * 0.4) &
                          (intervals < expected_interval * 1.6)]
        actual_interval = float(np.median(valid)) if len(valid) > 0 else expected_interval
    else:
        actual_interval = expected_interval

    # Pre-filter: remove candidates outside the densest plausible window
    if len(merged) > n_expected + 2:
        strong = [m for m in merged if m.n_proposers >= 3]
        if len(strong) >= 10:
            strong_frames = sorted([m.frame for m in strong])
            strong_intervals = np.diff(strong_frames)
            cadence_like = strong_intervals[
                (strong_intervals > expected_interval * 0.3) &
                (strong_intervals < expected_interval * 1.8)
            ]
            est_cadence = float(np.median(cadence_like)) if len(cadence_like) > 0 else expected_interval
            expected_span = (n_expected - 1) * est_cadence
            search_span = expected_span * 1.3
            best_window_start = strong_frames[0]
            best_count = 0
            for sf in strong_frames:
                count = sum(1 for f in strong_frames if sf <= f <= sf + search_span)
                if count > best_count:
                    best_count = count
                    best_window_start = sf
            window_lo = best_window_start - est_cadence * 1.5
            window_hi = best_window_start + search_span + est_cadence * 1.5
            pre_count = len(merged)
            merged = [m for m in merged if window_lo <= m.frame <= window_hi]
            n_prefiltered = pre_count - len(merged)
            if n_prefiltered > 0:
                frames = [m.frame for m in merged]
                anomalies.append(
                    f"Pre-filtered {n_prefiltered} candidates outside plausible window"
                )

    # Too many: drop lowest (n_proposers, consensus_score)
    if len(merged) > n_expected:
        excess = len(merged) - n_expected
        sorted_indices = sorted(
            range(len(merged)),
            key=lambda i: (merged[i].n_proposers, merged[i].consensus_score),
        )
        to_drop = set(sorted_indices[:excess])
        merged = [m for i, m in enumerate(merged) if i not in to_drop]
        frames = [m.frame for m in merged]
        anomalies.append(f"Dropped {excess} lowest-quality candidates")

    if len(merged) == n_expected:
        boundaries = frames[:]
        boundaries, anomalies = _validate_and_correct(
            boundaries, total_frames, anomalies, expected_interval)
        return boundaries, anomalies, merged

    if n_expected - 4 <= len(merged) < n_expected:
        selected = list(merged)
        frames = [m.frame for m in selected]
        missing = n_expected - len(frames)
        gap_to_end = total_frames - frames[-1]
        internal_gaps = np.diff(frames)
        max_internal = float(np.max(internal_gaps)) if len(internal_gaps) > 0 else 0

        if gap_to_end > max_internal * 0.8 and missing <= 2:
            for _ in range(missing):
                projected = int(frames[-1] + actual_interval)
                projected = min(projected, total_frames - 1)
                frames.append(projected)
                anomalies.append(f"Projected endpoint at frame {projected}")
        elif frames[0] > actual_interval * 0.8 and missing <= 2:
            for _ in range(missing):
                projected = max(0, int(frames[0] - actual_interval))
                frames.insert(0, projected)
                anomalies.append(f"Projected start at frame {projected}")
        else:
            gap_indices = np.argsort(internal_gaps)[::-1][:missing]
            for gap_idx in sorted(gap_indices, reverse=True):
                gap_start = frames[gap_idx]
                gap_end = frames[gap_idx + 1]
                interp_frame = (gap_start + gap_end) // 2
                frames.insert(gap_idx + 1, interp_frame)
                anomalies.append(f"Interpolated boundary at frame {interp_frame}")
            frames = frames[:n_expected]

        frames, anomalies = _validate_and_correct(
            frames, total_frames, anomalies, expected_interval)
        return frames, anomalies, selected

    # Far from expected: build grid from first candidate + median interval
    estimated_b1 = frames[0]
    if estimated_b1 > actual_interval:
        estimated_b1 = max(
            50, int(estimated_b1 - actual_interval * round(estimated_b1 / actual_interval)))
        anomalies.append(f"Estimated B1 at frame {estimated_b1}")

    boundaries = []
    for i in range(n_expected):
        expected_frame = estimated_b1 + i * actual_interval
        expected_frame = max(0, min(total_frames - 1, int(expected_frame)))
        nearby = [f for f in frames if abs(f - expected_frame) < actual_interval * 0.2]
        if nearby:
            boundaries.append(min(nearby, key=lambda f: abs(f - expected_frame)))
        else:
            boundaries.append(expected_frame)
    return boundaries, anomalies, []


def _validate_and_correct(frames: List[int], total_frames: int,
                          anomalies: List[str],
                          expected_interval: float) -> Tuple[List[int], List[str]]:
    """Phantom removal + endpoint projection (same logic as v2.1.3)."""
    if len(frames) < 3:
        return frames, anomalies

    frames = list(frames)

    # Phantom removal
    while True:
        intervals = np.diff(frames)
        if len(intervals) < 4:
            break
        median_interval = float(np.median(intervals))
        phantom_idx = None
        for i in range(1, len(intervals) - 2):
            a = float(intervals[i])
            b = float(intervals[i + 1])
            before = float(intervals[i - 1])
            after = float(intervals[i + 2])
            if (a < median_interval * 0.95
                    and b < median_interval * 0.95
                    and abs((a + b) - median_interval) < median_interval * 0.15
                    and abs(before - median_interval) < median_interval * 0.15
                    and abs(after - median_interval) < median_interval * 0.15):
                phantom_idx = i + 1
                break
        if phantom_idx is None:
            break
        dropped = frames.pop(phantom_idx)
        anomalies.append(f"Removed phantom boundary at frame {dropped}")

    # Endpoint projection
    if len(frames) < 2:
        return frames, anomalies
    intervals = np.diff(frames)
    median_interval = float(np.median(intervals))
    if median_interval < 500 or median_interval > 5000:
        median_interval = expected_interval

    guard = 0
    while len(frames) < 21 and guard < 4:
        guard += 1
        gap_to_start = frames[0]
        gap_to_end = total_frames - frames[-1]
        start_suspicious = gap_to_start > median_interval * 0.9
        end_suspicious = gap_to_end > median_interval * 0.9
        if not (start_suspicious or end_suspicious):
            break
        if start_suspicious and gap_to_start >= gap_to_end:
            projected = max(0, int(frames[0] - median_interval))
            frames.insert(0, projected)
            anomalies.append(f"Projected missing B1 at frame {projected}")
        elif end_suspicious:
            projected = min(total_frames - 1, int(frames[-1] + median_interval))
            frames.append(projected)
            anomalies.append(f"Projected missing B21 at frame {projected}")
        else:
            break

    return frames, anomalies
