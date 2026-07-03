"""
Shared guard functions for the v6 cascade outcome detector.

These guards wrap stage decide() methods to reject commits that violate
physical invariants detectable from pellet tracking signals. They are
applied in _build_production_stages() after stage construction.

Guard application order (per the validated 4.0 deploy config):
1. paw_lk overrides (set attribute on specific stages)
2. vanish guard (on displaced stages in DISPLACED_VANISH_GUARD_CLASSES)
3. SA-presence guard (on ALL stages)

Validated 2026-07-02 against train + gen corpora.
"""
from __future__ import annotations

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import StageDecision


def lrun(mask):
    """Longest run of True values in a boolean-like array."""
    best = cur = 0
    for x in mask:
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def pellet_displaced_or_vanished(dlc, seg_start, seg_end,
                                 sa_run_thr=15, vanish_run_thr=60):
    """Shared 'the pellet was touched' guard for untouched-committing stages.

    Returns True if the pellet was sustained in the SA (>3 radii for >=
    sa_run_thr frames) OR vanished (lk<0.5 for >= vanish_run_thr frames).
    Either means displaced/retrieved -- do NOT commit untouched.
    """
    ce = seg_end - 5
    sub_raw = dlc.iloc[seg_start:ce + 1]
    n = len(sub_raw)
    if n < 5:
        return False
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    g = compute_pillar_geometry_series(sub)
    cx = g["pillar_cx"].to_numpy(float)
    cy = g["pillar_cy"].to_numpy(float)
    r = g["pillar_r"].to_numpy(float)
    px = sub["Pellet_x"].to_numpy(float)
    py = sub["Pellet_y"].to_numpy(float)
    plk = sub_raw["Pellet_likelihood"].to_numpy(float)
    prad = np.sqrt((px - cx) ** 2 + (py - cy) ** 2) / np.maximum(r, 1e-6)
    return (lrun((plk >= 0.7) & (prad > 3.0)) >= sa_run_thr
            or lrun(plk < 0.5) >= vanish_run_thr)


def pellet_vanishes(dlc, seg_start, seg_end, vanish_run_thr=60):
    """Displaced-stage guard: True if the pellet disappears for a sustained
    stretch (lk<0.5 for >= vanish_run_thr frames in the clean zone).

    A displaced pellet stays visible in the SA; a retrieved pellet is
    carried off and vanishes. So a displaced-committing stage that sees a
    sustained vanish is looking at a RETRIEVED segment -- do not commit
    displaced.
    """
    ce = seg_end - 5
    plk = dlc.iloc[seg_start:ce + 1]["Pellet_likelihood"].to_numpy(float)
    return lrun(plk < 0.5) >= vanish_run_thr


def pellet_sustained_in_sa(dlc, seg_start, seg_end):
    """Longest run of confident pellet (lk>=0.7) held beyond 3 pillar-radii
    (sustained in/through the scoring area), over the clean zone.

    By definition a DISPLACED pellet ends up SUSTAINED in the SA --
    corpus true-displaced p10=378 frames. A retrieved or near-pillar
    pellet is not (residual retr->disp separable ones all <= 68).
    Gates displaced commits.
    """
    ce = seg_end - 5
    sub_raw = dlc.iloc[seg_start:ce + 1]
    n = len(sub_raw)
    if n < 10:
        return 0
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    g = compute_pillar_geometry_series(sub)
    cx = g["pillar_cx"].to_numpy(float)
    cy = g["pillar_cy"].to_numpy(float)
    r = g["pillar_r"].to_numpy(float)
    px = sub["Pellet_x"].to_numpy(float)
    py = sub["Pellet_y"].to_numpy(float)
    plk = sub_raw["Pellet_likelihood"].to_numpy(float)
    rad = np.sqrt((px - cx) ** 2 + (py - cy) ** 2) / np.maximum(r, 1e-6)
    return lrun((plk >= 0.7) & (rad > 3.0))


# ---------------------------------------------------------------------------
# Stage classes whose displaced_sa commits get the vanish guard.
# Keyed by CLASS NAME (type(s).__name__), not cascade build label.
# ---------------------------------------------------------------------------
DISPLACED_VANISH_GUARD_CLASSES = {
    "Stage16DisplacedViaMaxDisplacement",
    "Stage17DisplacedViaDominantMaxDisplacement",
    "Stage27DisplacedSaViaUniqueHighDisplacement",
    "Stage29DisplacedSaViaPillarDisambiguatedMultiDisplacement",
}

# Displaced stages with 4.0 paw-inflation requiring paw_lk override.
PAW_LK_OVERRIDES = {
    "Stage16DisplacedViaMaxDisplacement": 0.9,
    "Stage17DisplacedViaDominantMaxDisplacement": 0.9,
}

# Sustained-SA-presence gate threshold (frames).
SUSTAINED_SA_GATE_T = 30


def wrap_vanish_guard(stage):
    """Wrap a stage's decide() to reject displaced_sa commits when the
    pellet has a sustained vanish (retrieved signature)."""
    orig = stage.decide

    def decide(seg):
        dec = orig(seg)
        if (dec.decision == "commit" and dec.committed_class == "displaced_sa"
                and pellet_vanishes(seg.dlc_df, seg.seg_start, seg.seg_end)):
            return StageDecision(
                decision="continue",
                reason="pellet_vanished_retrieved_not_displaced_guard")
        return dec

    stage.decide = decide
    return stage


def wrap_sa_presence_guard(stage):
    """Guard ANY displaced_sa commit: if the pellet was never SUSTAINED
    in the SA (offrun < SUSTAINED_SA_GATE_T), the 'displaced' call is
    not supported -- defer so a retrieved stage can catch or triage."""
    orig = stage.decide

    def decide(seg):
        dec = orig(seg)
        if (dec.decision == "commit" and dec.committed_class == "displaced_sa"
                and pellet_sustained_in_sa(seg.dlc_df, seg.seg_start,
                                           seg.seg_end) < SUSTAINED_SA_GATE_T):
            return StageDecision(
                decision="continue",
                reason=f"pellet_not_sustained_in_SA_gate(offrun<{SUSTAINED_SA_GATE_T})")
        return dec

    stage.decide = decide
    return stage
