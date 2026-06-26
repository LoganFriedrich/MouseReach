"""
Multi-proposer segmenter (v2.2.0).

Replaces the single-SABL candidate producer with four SA corner proposers
merged by consensus. SABL + SATL are frame sources (aligned with GT).
SABR + SATR are confirmers (systematic ~7 f early bias vs GT).

Architecture:
  Phase 1:   reference & SA coverage quality (unchanged from v2.1.3).
  Phase 2:   four SA corner proposers run independently.
  Phase 2.5: hybrid SABL-primary / consensus mode:
               - If SABL produces exactly 21 candidates, use those as the
                 backbone and use merged consensus to refine each frame
                 (which usually keeps the SABL frame).
               - Else run full consensus selection.
  Phase 3:   select 21 boundaries with phantom removal + endpoint projection.
  Phase 4:   anomaly detection & emit (unchanged from v2.1.3).

The production segmenter_robust.py is used as a library (load_dlc,
get_clean_signal, compute_velocity, assess_reference_quality,
assess_sa_quality, detect_anomalies, save_segmentation,
SegmentationDiagnostics). It is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .segmenter_robust import (
    SegmentationDiagnostics,
    assess_reference_quality,
    assess_sa_quality,
    compute_velocity,
    detect_anomalies,
    get_clean_signal,
    load_dlc,
)
from .proposers import (
    Candidate,
    pellet_swap_proposer,
    sa_proposer,
)
from .consensus import (
    MergedCandidate,
    build_consensus,
    select_boundaries,
)
from .tray_motion import apply_tray_motion_gate

SEGMENTER_VERSION = "2.2.2"
SEGMENTER_ALGORITHM = "multi_proposer_sabl_primary_v1+tray_motion_gate+pellet_window_gate"


@dataclass
class MultiProposerConfig:
    """All hyperparameters for the multi-proposer segmenter.

    Defaults are LOO-validated on the 47-video corpus and stable across
    180 sweep combinations -- see
    Y:\\2_Connectome\\Behavior\\MouseReach_Improvement\\validation_runs\\DLC_2026_03_27\\exploration\\multi_proposer\\FINAL_DESIGN.md
    """
    # SA proposer params (shared across all 4 corners)
    sa_vel_threshold: float = 0.8
    sa_min_gap: int = 25
    sa_center_range: Tuple[float, float] = (-5, 10)
    sa_center_target: float = 2.5
    sa_smooth_window: int = 30
    sa_endpoint_vel_threshold: float = 1.4

    # Pellet proposer (disabled: too noisy in production)
    pellet_enabled: bool = False
    pellet_like_drop_threshold: float = 0.2
    pellet_pos_shift_threshold: float = 8.0
    pellet_time_window: int = 30
    pellet_min_gap: int = 25

    # Consensus merging
    merge_window: int = 30

    # Grid fitting
    expected_interval: float = 1839.0
    n_expected: int = 21

    # v2.2.1: tray-motion gate (Signal 27).
    # Validate each boundary against ASPA-cycle physical signature.
    # Reject and substitute boundaries that fail.
    tray_motion_gate_enabled: bool = True
    tray_motion_window: int = 50
    tray_motion_excursion_threshold: float = 30.0
    tray_motion_pillar_lk_drop_threshold: float = 0.3

    # v2.2.2: pellet-active-window gate (Model-4.0 first-cycle damping).
    # A stronger DLC model (e.g. Model 4.0) tracks the SA corners more stably
    # but emits spurious low-velocity boundary candidates in the pre-trial
    # setup and post-session shutdown -- periods where NO pellet is ever on the
    # pillar. Those phantoms push a proposer above n_expected, knocking it out
    # of the reliable SABL-primary path into the fragile consensus/grid-fit,
    # which phase-shifts the whole segment numbering (canonical: CNT0312_P2,
    # an 8376-frame B1 miss). Gate a proposer's candidates to the pellet-active
    # window. CONSERVATIVE by design:
    #   - acts only on a proposer with > n_expected candidates (a real surplus),
    #     never an already-clean count;
    #   - reverts if gating would drop below n_expected (not clean removal);
    #   - excludes only the pre/post dead zones -- a no-pellet MIDDLE segment is
    #     always kept.
    # Pellet *presence* is binary/robust even though pellet *position* is noisy;
    # it is used here ONLY as a dead-zone exclusion, not a per-segment "must
    # have a pellet" test (a real cycle can rarely present no pellet). The one
    # residual ambiguity -- a no-pellet FIRST/LAST segment -- falls back to the
    # existing count-check / triage.
    pellet_window_gate_enabled: bool = True
    pellet_window_lk_threshold: float = 0.5
    pellet_window_smooth: int = 60
    pellet_window_active_frac: float = 0.3
    pellet_window_margin: int = 200


def _pellet_active_window(df: pd.DataFrame, lk_threshold: float,
                          smooth: int, active_frac: float
                          ) -> Optional[Tuple[int, int]]:
    """First and last frame of sustained pellet presence (the active window).

    Returns ``None`` if there is no pellet column or no pellet is ever present
    (in which case the gate is a no-op).
    """
    if "Pellet_likelihood" not in df.columns:
        return None
    present = (df["Pellet_likelihood"].values > lk_threshold).astype(float)
    run = np.convolve(present, np.ones(smooth) / smooth, mode="same")
    active = np.where(run > active_frac)[0]
    if len(active) == 0:
        return None
    return int(active[0]), int(active[-1])


def _gate_candidates_to_window(cands: List[Candidate],
                               window: Optional[Tuple[int, int]],
                               margin: int, n_expected: int) -> List[Candidate]:
    """Drop dead-zone phantom candidates outside the pellet-active window.

    Acts only when there are MORE than ``n_expected`` candidates (a real
    phantom surplus), and reverts to the originals if gating would leave fewer
    than ``n_expected`` (i.e. it was not clean phantom removal). So it can
    recover a phantom-padded proposer back toward ``n_expected`` but can never
    strip a real boundary from an already-clean count.
    """
    if window is None or len(cands) <= n_expected:
        return cands
    lo, hi = window
    gated = [c for c in cands if (lo - margin) <= c.frame <= (hi + margin)]
    if len(gated) < n_expected:
        return cands
    return gated


def segment_video_multi(dlc_path: Path,
                        fps: float = 60.0,
                        config: Optional[MultiProposerConfig] = None
                        ) -> Tuple[List[int], SegmentationDiagnostics]:
    """Multi-proposer segmentation.

    Returns the same ``(boundaries, SegmentationDiagnostics)`` signature as
    ``segment_video_robust`` so downstream consumers (batch, save_segmentation)
    do not change.
    """
    if config is None:
        config = MultiProposerConfig()

    dlc_path = Path(dlc_path)
    df = load_dlc(dlc_path)
    total_frames = len(df)

    box_center, boxl_std, boxr_std, ref_quality = assess_reference_quality(df)
    sa_coverage = assess_sa_quality(df)
    anomalies: List[str] = []

    # Reference bailout -- evenly spaced fallback, same behavior as v2.1.3.
    if box_center is None or ref_quality == 'bad':
        anomalies.append(f"Bad reference quality: {ref_quality}")
        interval = total_frames / (config.n_expected + 1)
        boundaries = [int((i + 1) * interval) for i in range(config.n_expected)]
        diag = _diagnostics(
            dlc_path, total_frames, fps,
            box_center or 0.0, boxl_std, boxr_std, ref_quality,
            sa_coverage,
            n_primary=0, n_fallback=0,
            boundaries=boundaries,
            boundary_methods=['fallback'] * config.n_expected,
            boundary_confidences=[0.0] * config.n_expected,
            anomalies=anomalies,
        )
        return boundaries, diag

    # Phase 2: run all proposers independently
    per_proposer: Dict[str, List[Candidate]] = {}
    all_candidates: List[Candidate] = []
    for bp in ['SABL', 'SABR', 'SATL', 'SATR']:
        cands = sa_proposer(
            df, bp, box_center,
            center_range=config.sa_center_range,
            vel_threshold=config.sa_vel_threshold,
            min_gap=config.sa_min_gap,
            smooth_window=config.sa_smooth_window,
            center_target=config.sa_center_target,
            endpoint_vel_threshold=config.sa_endpoint_vel_threshold,
        )
        per_proposer[bp] = cands
        all_candidates.extend(cands)

    # Phase 2.2: pellet-active-window gate (v2.2.2). Trim pre-trial/post-session
    # phantom candidates so a phantom surplus does not knock a proposer out of
    # the SABL-primary path. Excess-only + revert-if-undershoot (see config).
    if config.pellet_window_gate_enabled:
        window = _pellet_active_window(
            df, config.pellet_window_lk_threshold,
            config.pellet_window_smooth, config.pellet_window_active_frac)
        if window is not None:
            all_candidates = []
            for bp in ['SABL', 'SABR', 'SATL', 'SATR']:
                original = per_proposer.get(bp, [])
                gated = _gate_candidates_to_window(
                    original, window, config.pellet_window_margin,
                    config.n_expected)
                if len(gated) < len(original):
                    anomalies.append(
                        f"pellet_window_gate: {bp} dropped "
                        f"{len(original) - len(gated)} dead-zone candidate(s)")
                per_proposer[bp] = gated
                all_candidates.extend(gated)

    n_primary = len(per_proposer.get('SABL', []))
    n_sa_total = len(all_candidates)

    if config.pellet_enabled:
        pellet_cands = pellet_swap_proposer(
            df,
            like_drop_threshold=config.pellet_like_drop_threshold,
            pos_shift_threshold=config.pellet_pos_shift_threshold,
            time_window=config.pellet_time_window,
            min_gap=config.pellet_min_gap,
        )
        all_candidates.extend(pellet_cands)

    # Phase 2.5: hybrid SABL-primary / consensus
    sabl_cands = per_proposer.get('SABL', [])
    use_sabl_primary = (len(sabl_cands) == config.n_expected)

    merged = build_consensus(all_candidates, merge_window=config.merge_window)

    if use_sabl_primary:
        sabl_frames = sorted([c.frame for c in sabl_cands])
        boundaries: List[int] = []
        for sf in sabl_frames:
            closest = min(merged, key=lambda m: abs(m.frame - sf))
            if abs(closest.frame - sf) < config.merge_window:
                boundaries.append(closest.frame)
            else:
                boundaries.append(sf)
        anomalies.append(f"SABL-primary mode: {len(sabl_cands)} SABL candidates")
        selected = [
            min(merged, key=lambda m: abs(m.frame - sf)) if merged else None
            for sf in sabl_frames
        ]
    else:
        boundaries, grid_anomalies, selected = select_boundaries(
            merged, total_frames,
            n_expected=config.n_expected,
            expected_interval=config.expected_interval,
        )
        anomalies.extend(grid_anomalies)
        anomalies.append(
            f"Consensus mode: SABL={len(sabl_cands)} != {config.n_expected}"
        )

    # Safety net: guarantee exactly n_expected
    if len(boundaries) > config.n_expected:
        boundaries = sorted(boundaries)[:config.n_expected]
    elif len(boundaries) < config.n_expected:
        if len(boundaries) >= 2:
            med_int = float(np.median(np.diff(boundaries)))
        else:
            med_int = config.expected_interval
        while len(boundaries) < config.n_expected:
            boundaries.append(min(total_frames - 1,
                                  int(boundaries[-1] + med_int)))
    boundaries = sorted(boundaries)

    # Phase 3.5: tray-motion gate (Signal 27).
    # Reject boundaries that don't show a real ASPA cycle signature
    # (SA corner excursion + pillar lk drop). Replace each rejection
    # with a median-cadence projection from valid neighbors. See
    # tray_motion.py and the tray_motion_segment_boundary_test memory
    # entry.
    if config.tray_motion_gate_enabled:
        boundaries, tray_rejections = apply_tray_motion_gate(
            df, boundaries, total_frames, config.expected_interval,
            window=config.tray_motion_window,
            excursion_threshold=config.tray_motion_excursion_threshold,
            pillar_lk_drop_threshold=config.tray_motion_pillar_lk_drop_threshold,
        )
        for idx, original, reasons in tray_rejections:
            anomalies.append(
                f"tray_motion_gate_rejected b{idx}@{original}: {','.join(reasons)}"
            )

    # Phase 4: anomaly detection
    boundary_anomalies = detect_anomalies(boundaries, fps)
    anomalies.extend(boundary_anomalies)

    # Per-boundary method / confidence from the merged cluster we picked
    boundary_methods: List[str] = []
    boundary_confidences: List[float] = []
    for b in boundaries:
        if merged:
            closest = min(merged, key=lambda m: abs(m.frame - b))
            if abs(closest.frame - b) <= config.merge_window:
                method = '+'.join(sorted(closest.proposers))
                conf = closest.consensus_score
            else:
                method = 'interpolated'
                conf = 0.0
        else:
            method = 'fallback'
            conf = 0.0
        boundary_methods.append(method)
        boundary_confidences.append(float(conf))

    diag = _diagnostics(
        dlc_path, total_frames, fps,
        box_center, boxl_std, boxr_std, ref_quality,
        sa_coverage,
        n_primary=n_primary, n_fallback=max(0, n_sa_total - n_primary),
        boundaries=boundaries,
        boundary_methods=boundary_methods,
        boundary_confidences=boundary_confidences,
        anomalies=anomalies,
    )
    return boundaries, diag


def _diagnostics(dlc_path: Path, total_frames: int, fps: float,
                 box_center: float, boxl_std: float, boxr_std: float,
                 ref_quality: str,
                 sa_coverage: Dict[str, float],
                 n_primary: int, n_fallback: int,
                 boundaries: List[int],
                 boundary_methods: List[str],
                 boundary_confidences: List[float],
                 anomalies: List[str]) -> SegmentationDiagnostics:
    """Build a SegmentationDiagnostics with interval stats computed."""
    if len(boundaries) >= 2:
        intervals = np.diff(boundaries)
        interval_mean = float(np.mean(intervals))
        interval_std = float(np.std(intervals))
        interval_cv = interval_std / interval_mean if interval_mean > 0 else 1.0
    else:
        interval_mean = interval_std = 0.0
        interval_cv = 1.0

    return SegmentationDiagnostics(
        video_name=dlc_path.stem,
        total_frames=total_frames,
        fps=fps,
        box_center=float(box_center),
        boxl_std=float(boxl_std),
        boxr_std=float(boxr_std),
        reference_quality=ref_quality,
        sabl_coverage=float(sa_coverage.get('SABL', 0.0)),
        sabr_coverage=float(sa_coverage.get('SABR', 0.0)),
        satl_coverage=float(sa_coverage.get('SATL', 0.0)),
        satr_coverage=float(sa_coverage.get('SATR', 0.0)),
        n_primary_candidates=int(n_primary),
        n_secondary_candidates=0,
        n_fallback_candidates=int(n_fallback),
        boundaries=list(boundaries),
        boundary_methods=list(boundary_methods),
        boundary_confidences=list(boundary_confidences),
        anomalies=list(anomalies),
        interval_mean=float(interval_mean),
        interval_std=float(interval_std),
        interval_cv=float(interval_cv),
    )
