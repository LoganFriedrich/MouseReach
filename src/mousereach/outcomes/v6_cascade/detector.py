"""
Production entry point for the v6 cascade outcome detector.

Takes (dlc_df, segments, reaches) and returns the standard
pellet_outcomes dict that the rest of the pipeline expects.

This is a thin wrapper around the 30 cascade stages (0-29 + 98 + 99).
The stages themselves are deterministic rule-based classifiers --
no model artifact required.

Usage::

    from mousereach.outcomes.v6_cascade import detect_outcomes_v6_cascade

    result = detect_outcomes_v6_cascade(
        dlc_df=dlc,
        segments=[(100, 500), (501, 900), ...],
        reaches=[(150, 200), (520, 580), ...],
        video_id="20250626_CNT0102_P4",
    )

Returns a dict with the standard pellet_outcomes JSON shape::

    {
        "video_id": str,
        "detector": "v6_cascade",
        "detector_version": "6.0.0",
        "segments": [
            {
                "segment_num": int,
                "outcome": str,
                "outcome_known_frame": int | None,
                "interaction_frame": int | None,
                "stage": str,
                "flagged_for_review": bool,
            },
            ...
        ],
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import VERSION
from .stage_base import SegmentInput, StageDecision

# ---------------------------------------------------------------------------
# Stage imports (same canonical list as the validated holdout runner)
# ---------------------------------------------------------------------------
from .stage_0_short_segment_triage import Stage0ShortSegmentTriage
from .stage_1_pellet_position_never_changed import Stage1PelletPositionNeverChanged
from .stage_2_pellet_stable_untouched import Stage2PelletStableUntouched
from .stage_3_paw_never_in_pellet_area import Stage3PawNeverInPelletArea
from .stage_4_pellet_returns_to_pillar import Stage4PelletReturnsToPillar
from .stage_5_pellet_off_pillar_throughout import Stage5PelletOffPillarThroughout
from .stage_6_pellet_predominantly_on_pillar import Stage6PelletPredominantlyOnPillar
from .stage_6b_never_entered_sa import Stage6bNeverEnteredSA
from .stage_7_pellet_settled_off_pillar_late import Stage7PelletSettledOffPillarLate
from .stage_8_pellet_displaced_to_sa import Stage8PelletDisplacedToSA
from .stage_9_pellet_vanished_after_reach import Stage9PelletVanishedAfterReach
from .stage_10_pillar_revealed_after_reach import Stage10PillarRevealedAfterReach
from .stage_11_single_reach_clean_displacement import Stage11SingleReachCleanDisplacement
from .stage_12_retrieved_pellet_above_slit import Stage12RetrievedPelletAboveSlit
from .stage_13_retrieved_via_pillar_lk_transition import Stage13RetrievedViaPillarLkTransition
from .stage_14_single_reach_moderate_displacement_evidence import Stage14SingleReachModerateDisplacementEvidence
from .stage_15_multi_reach_retrieved_via_above_slit_split import Stage15MultiReachRetrievedViaAboveSlitSplit
from .stage_16_displaced_via_max_displacement_reach import Stage16DisplacedViaMaxDisplacement
from .stage_17_displaced_via_dominant_max_displacement import Stage17DisplacedViaDominantMaxDisplacement
from .stage_18_displaced_via_first_significant_displacement import Stage18DisplacedViaFirstSignificantDisplacement
from .stage_19_retrieved_via_pillar_lk_first_reach import Stage19RetrievedViaPillarLkFirstReach
from .stage_20_per_bout_classifier_displaced import Stage20PerBoutClassifierDisplaced
from .stage_21_causal_reach_via_immediate_on_off_transition import Stage21CausalReachViaImmediateOnOffTransition
from .stage_22_retry_with_stabilized_dlc import Stage22RetryWithStabilizedDlc
from .stage_23_retrieved_with_pillar_tip_noise import Stage23RetrievedWithPillarTipNoise
from .stage_24_transition_triangulation import Stage24TransitionTriangulation
from .stage_25_retry_with_strict_pellet_confidence import Stage25RetryWithStrictPelletConfidence
from .stage_26_retrieved_via_unique_vanish_reach import Stage26RetrievedViaUniqueVanishReach
from .stage_27_displaced_sa_via_unique_high_displacement_reach import Stage27DisplacedSaViaUniqueHighDisplacement
from .stage_28_retrieved_via_pillar_visibility_transition import Stage28RetrievedViaPillarVisibilityTransition
from .stage_29_displaced_sa_via_pillar_disambiguated_multi_displacement import Stage29DisplacedSaViaPillarDisambiguatedMultiDisplacement
from .stage_98_lost_in_shadow_triage import Stage98LostInShadowTriage
from .stage_99_residual_triage import Stage99ResidualTriage
from .guards import (
    DISPLACED_VANISH_GUARD_CLASSES,
    PAW_LK_OVERRIDES,
    wrap_vanish_guard,
    wrap_sa_presence_guard,
)


def _build_production_stages(
    video_dir: Optional[Path] = None,
) -> List[tuple]:
    """Build the canonical stage list for production.

    Parameters
    ----------
    video_dir : Path, optional
        Directory containing .avi/.mp4 video files, needed by Stage 98
        (lost-in-shadow triage) for CV-based dark-SA detection. If None,
        Stage 98 will skip its CV check and defer to Stage 99.

    Returns
    -------
    list of (label, stage_instance) tuples
    """
    raw_stages = [
        ("stage_0_short_segment_triage", Stage0ShortSegmentTriage()),
        ("stage_1_position_never_changed", Stage1PelletPositionNeverChanged()),
        ("stage_2_stable_on_pillar", Stage2PelletStableUntouched(commit_frac=0.95, commit_distance_radii=1.5)),
        ("stage_3_paw_never_in_pellet_area", Stage3PawNeverInPelletArea()),
        ("stage_4_pellet_returns_to_pillar", Stage4PelletReturnsToPillar()),
        ("stage_5_pellet_off_pillar_throughout", Stage5PelletOffPillarThroughout()),
        ("stage_6_predominantly_on_pillar", Stage6PelletPredominantlyOnPillar()),
        ("stage_6b_never_entered_sa", Stage6bNeverEnteredSA()),
        ("stage_7_settled_off_pillar_late", Stage7PelletSettledOffPillarLate()),
        ("stage_8_pellet_displaced_to_sa", Stage8PelletDisplacedToSA()),
        ("stage_9_pellet_vanished_after_reach", Stage9PelletVanishedAfterReach()),
        ("stage_10_pillar_revealed_after_reach", Stage10PillarRevealedAfterReach()),
        ("stage_11_single_reach_clean_displacement", Stage11SingleReachCleanDisplacement()),
        ("stage_12_retrieved_pellet_above_slit", Stage12RetrievedPelletAboveSlit()),
        ("stage_13_retrieved_via_pillar_lk_transition", Stage13RetrievedViaPillarLkTransition()),
        ("stage_14_single_reach_moderate_displacement", Stage14SingleReachModerateDisplacementEvidence()),
        ("stage_15_multi_reach_retrieved_above_slit", Stage15MultiReachRetrievedViaAboveSlitSplit()),
        ("stage_16_displaced_via_max_displacement", Stage16DisplacedViaMaxDisplacement()),
        ("stage_17_displaced_via_dominant_max_displacement", Stage17DisplacedViaDominantMaxDisplacement()),
        ("stage_18_displaced_via_first_significant_displacement", Stage18DisplacedViaFirstSignificantDisplacement()),
        ("stage_19_retrieved_via_pillar_lk_first_reach", Stage19RetrievedViaPillarLkFirstReach()),
        ("stage_20_per_bout_classifier_displaced", Stage20PerBoutClassifierDisplaced()),
        ("stage_21_causal_reach_via_on_off_transition", Stage21CausalReachViaImmediateOnOffTransition()),
        ("stage_22_retry_with_stabilized_dlc", Stage22RetryWithStabilizedDlc()),
        ("stage_23_retrieved_with_pillar_tip_noise", Stage23RetrievedWithPillarTipNoise()),
        ("stage_24_transition_triangulation", Stage24TransitionTriangulation()),
        ("stage_25_retry_with_strict_pellet_confidence", Stage25RetryWithStrictPelletConfidence()),
        ("stage_26_retrieved_via_unique_vanish_reach", Stage26RetrievedViaUniqueVanishReach()),
        ("stage_27_displaced_sa_via_unique_high_displacement", Stage27DisplacedSaViaUniqueHighDisplacement()),
        ("stage_28_retrieved_via_pillar_visibility_transition", Stage28RetrievedViaPillarVisibilityTransition()),
        ("stage_29_displaced_sa_pillar_disambiguated_multi_disp", Stage29DisplacedSaViaPillarDisambiguatedMultiDisplacement()),
        ("stage_98_lost_in_shadow_triage", Stage98LostInShadowTriage(video_dir=video_dir)),
        ("stage_99_residual_triage", Stage99ResidualTriage()),
    ]

    # 4.0 recalibration: apply guards to all stage instances.
    # Order: paw_lk overrides -> vanish guard (ALL stages; only fires
    # on displaced_sa commits internally) -> SA-presence guard (ALL
    # stages; only fires on displaced_sa commits internally).
    out = []
    for lab, s in raw_stages:
        cn = type(s).__name__
        if cn in PAW_LK_OVERRIDES:
            s.paw_lk_threshold = PAW_LK_OVERRIDES[cn]
        wrap_vanish_guard(s)
        wrap_sa_presence_guard(s)
        out.append((lab, s))
    return out


def _find_reaches_in_segment(
    seg_start: int,
    seg_end: int,
    reaches: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Return reaches whose start falls within [seg_start, seg_end]."""
    out = []
    for r_start, r_end in reaches:
        # A reach belongs to this segment if its start frame is inside
        # the segment window. End frame may extend slightly past seg_end
        # (reach in progress when tray cycled), which is fine -- the
        # stage code handles clipping internally.
        if seg_start <= r_start <= seg_end:
            out.append((r_start, r_end))
    return out


def detect_outcomes_v6_cascade(
    dlc_df: pd.DataFrame,
    segments: List[Tuple[int, int]],
    reaches: List[Tuple[int, int]],
    *,
    video_id: str = "",
    video_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the v6 cascade outcome detector end-to-end.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        Full-video DLC tracking data (multi-level columns).
    segments : list of (start_frame, end_frame) tuples
        Segment boundaries. Both endpoints inclusive.
    reaches : list of (start_frame, end_frame) tuples
        Detected reach windows. Both endpoints inclusive.
    video_id : str, optional
        Identifier for this video (used in output JSON and diagnostics).
    video_dir : Path, optional
        Directory with source video files for Stage 98 CV checks.
        If None, Stage 98 skips CV and defers to Stage 99.

    Returns
    -------
    dict
        Standard pellet_outcomes JSON shape. Caller writes to disk.
    """
    stages = _build_production_stages(video_dir=video_dir)

    # Build SegmentInput for each segment
    seg_inputs = []
    for idx, (seg_start, seg_end) in enumerate(segments):
        seg_num = idx + 1  # 1-indexed
        seg_reaches = _find_reaches_in_segment(seg_start, seg_end, reaches)
        seg_inputs.append(SegmentInput(
            video_id=video_id,
            segment_num=seg_num,
            seg_start=seg_start,
            seg_end=seg_end,
            dlc_df=dlc_df,
            reach_windows=seg_reaches,
        ))

    # Thread each segment through cascade: first stage to commit wins
    output_segments = []
    for seg_input in seg_inputs:
        decision = None  # type: Optional[StageDecision]
        committing_stage = "residual (auto-triage)"

        for label, stage in stages:
            dec = stage.decide(seg_input)
            if dec.decision == "commit":
                decision = dec
                committing_stage = label
                break
            elif dec.decision == "triage":
                decision = dec
                committing_stage = label
                break
            # else "continue" -> next stage

        # Build output record
        if decision is not None and decision.decision == "commit":
            output_segments.append({
                "segment_num": seg_input.segment_num,
                "outcome": decision.committed_class,
                "outcome_known_frame": decision.whens.get("outcome_known_frame"),
                "interaction_frame": decision.whens.get("interaction_frame"),
                "stage": committing_stage,
                "flagged_for_review": False,
            })
        else:
            # Triage (explicit from a stage, or residual if somehow
            # all stages returned "continue" without Stage 99 catching it)
            reason = decision.reason if decision is not None else "fell_through_all_stages"
            output_segments.append({
                "segment_num": seg_input.segment_num,
                "outcome": "triaged",
                "outcome_known_frame": None,
                "interaction_frame": None,
                "stage": committing_stage,
                "flagged_for_review": True,
                "flag_reason": reason,
            })

    return {
        "video_id": video_id,
        "detector": "v6_cascade",
        "detector_version": VERSION,
        "segments": output_segments,
    }
