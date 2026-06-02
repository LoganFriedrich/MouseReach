"""
Outcome cascade Fix C: Stage 21 Edit 2 (Stage-9-aligned retrieved corroboration).

Layered on top of accepted v6.0.3 (= v6.0.2 + Fix B), which is the
cumulative best on the 20-video 2026-05-11 generalization corpus
(382/400 = 95.50%).

Per A-side proposal `Improvement_Snapshots/outcome/
v6_cascade_dev_stage_21_tightening_proposal/RESULTS.md`, "Edit 2":

  Before Stage 21 commits "retrieved" (the "not stable_in_sa" branch),
  add a Stage-9-aligned anti-displaced check: count sustained off-pillar
  pellet observations (>=3 consecutive frames of `pellet_lk >= 0.7 AND
  dist_radii > 1.5 AND not paw_past_y`) in the segment from the causal
  reach onward. If > 5 sustained off-pillar frames, defer to next stage.

Targets on the 20-video holdout (current v6.0.3 state):
  - 4 Stage 21 errors total (4 of 9 confusion errors)
  - 2 are displaced_outside (DLC-limited; Edit 2 unlikely to help)
  - 2 are displaced_sa -> retrieved:
      CNT0216_P1 s1   -- displaced_sa GT, algo retrieved
      CNT0214_P1 s2   -- displaced_sa GT, algo retrieved
    These are the expected recoveries.

================================================================
Pre-experiment checklist
================================================================

1. Cumulative-stacking check (verified 2026-06-02 via probe + manifests).
   Production reach detector: v8.0.4. Outcome cascade cumulative best:
   v6.0.3 (= v6.0.2 + Fix B), merged to master 2026-06-02 (commit e72926d).
   v6.0.3 numbers on 20-video corpus: 382 correct + 9 triaged + 9 errors.
   Master at fe15587 -> 5487397 (reach-detection investigations merged
   today). Outcome state unchanged since e72926d.

   Stacked improvements applied in this runner:
     - Stage 16 vanish veto (v6.0.1)
     - Stage 27 vanish veto (v6.0.1)
     - Stage 30 retrieved fallback (v6.0.1) -- conservative
     - Stage 31 retrieved rescue with displacement (v6.0.3 Fix B)
     - Stage 32 zero-reach untouched fallback (v6.0.2 Fix A)
     - NEW: Stage 21 with retrieved corroboration (this experiment)

   Comparison baseline: v6.0.3 snapshot at
   `Improvement_Snapshots/outcome/v6.0.3_fix_b_retrieved_rescue_2026-06-02/`
   (382/400 correct, 9 triaged, 9 errors).

2. Existing-code-modification check: NO. Subclass-based inline override
   per `feedback_file_editing_rules`. Module code under
   `src/mousereach/outcomes/v6_cascade/` is unchanged.

3. Unverified hypotheses
   - H1: Edit 2 will defer CNT0216_P1 s1 (currently algo=retrieved,
     GT=displaced_sa). Expectation: post-causal sustained off-pillar
     observations > 5 will fire the new gate.
   - H2: Edit 2 will defer CNT0214_P1 s2 (currently algo=retrieved,
     GT=displaced_sa). Same expectation.
   - H3: The 32 currently-correct Stage 21 commits on holdout will not
     all defer. Some might defer to a different stage that re-commits
     correctly (still OK), but if many currently-correct commits get
     deferred to Stage 99 triage, that's a regression.
   - H4: Edit 2 will NOT fire on the 2 displaced_outside cases (CNT0102_P4
     s10, CNT0209_P2 s17). Those are DLC-limited; the pellet vanishes
     from camera, so sustained off-pillar count may be low.

4. FN-direction-reporting
   Lead-line in RESULTS.md: "Retrieved class correct went from
   <baseline> to <new>: <delta>." Two deltas: vs cumulative best (v6.0.3)
   and vs pure baseline (v6.0.0 = 377/400).

5. Framework-not-adhoc
   Snapshot dir: Improvement_Snapshots/outcome/
                 v6.0.4_fix_c_stage_21_corroboration_2026-06-02/
   Canonical scoring via mousereach.improvement.outcome.compute_outcome_metrics.

6. Branch + tag
   - Pre-experiment tag: pre-stage-21-edit-2-2026-06-02
   - Feature branch: feature/outcome-stage-21-edit-2

7. Decision rule
   - ACCEPT: retrieved class correct does NOT decrease AND
             displaced_sa class correct rises by >= 1 AND
             no class regresses by > 1.
     (Edit 2 is structurally a tightening, so the expected sign of
      change is: displaced_sa correct rises (deferred Stage 21 retrieveds
      get re-committed correctly by Stage 26 or later) OR triage rises
      (deferred but no later stage commits). Retrieved correct should be
      essentially unchanged.)
   - REJECT: retrieved correct drops by > 0, OR displaced_sa correct
             stays flat or decreases, OR any class regresses by > 1.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.reach.v8 import detect_reaches_v8
from mousereach.reach.v8.features import load_dlc_h5
from mousereach.outcomes.v6_cascade.detector import _build_production_stages
from mousereach.outcomes.v6_cascade.stage_base import (
    SegmentInput, Stage, StageDecision,
)
from mousereach.outcomes.v6_cascade.stage_16_displaced_via_max_displacement_reach import (
    Stage16DisplacedViaMaxDisplacement,
)
from mousereach.outcomes.v6_cascade.stage_21_causal_reach_via_immediate_on_off_transition import (
    Stage21CausalReachViaImmediateOnOffTransition,
)
from mousereach.outcomes.v6_cascade.stage_27_displaced_sa_via_unique_high_displacement_reach import (
    Stage27DisplacedSaViaUniqueHighDisplacement,
)
from mousereach.improvement.outcome.metrics import compute_outcome_metrics


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
VANISH_VETO_FRAMES = 60


# ---------------------------------------------------------------------------
# v6.0.1 - 6.0.3 stacked stages (replicated inline)
# ---------------------------------------------------------------------------

def _max_consecutive_low_lk(dlc_df, start, end, thr=0.5):
    if end <= start:
        return 0
    lk = dlc_df["Pellet_likelihood"].iloc[start:end].to_numpy(dtype=float)
    below = lk < thr
    max_run = 0
    cur = 0
    for b in below:
        if b:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


class Stage16WithVanishVeto(Stage16DisplacedViaMaxDisplacement):
    name = "stage_16_with_vanish_veto"

    def decide(self, seg):
        d = super().decide(seg)
        if d.decision != "commit":
            return d
        ci = d.features.get("causal_idx")
        if ci is None:
            return d
        r = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        if ci >= len(r):
            return d
        cre = r[ci][1]
        mr = _max_consecutive_low_lk(seg.dlc_df, cre + 1, seg.seg_end + 1)
        f = dict(d.features)
        f["post_causal_max_vanish_run"] = int(mr)
        if mr >= VANISH_VETO_FRAMES:
            f["vanish_veto_fired"] = True
            return StageDecision(
                decision="continue",
                reason=f"global_post_causal_vanish_veto ({mr}f)",
                features=f,
            )
        return StageDecision(
            decision=d.decision,
            committed_class=d.committed_class,
            whens=d.whens,
            reason=d.reason,
            features=f,
        )


class Stage27WithVanishVeto(Stage27DisplacedSaViaUniqueHighDisplacement):
    name = "stage_27_with_vanish_veto"

    def decide(self, seg):
        d = super().decide(seg)
        if d.decision != "commit":
            return d
        ci = d.features.get("causal_reach_idx")
        if ci is None:
            return d
        r = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        if ci >= len(r):
            return d
        cre = r[ci][1]
        mr = _max_consecutive_low_lk(seg.dlc_df, cre + 1, seg.seg_end + 1)
        f = dict(d.features)
        f["post_causal_max_vanish_run"] = int(mr)
        if mr >= VANISH_VETO_FRAMES:
            f["vanish_veto_fired"] = True
            return StageDecision(
                decision="continue",
                reason=f"global_post_causal_vanish_veto ({mr}f)",
                features=f,
            )
        return StageDecision(
            decision=d.decision,
            committed_class=d.committed_class,
            whens=d.whens,
            reason=d.reason,
            features=f,
        )


class Stage30FallbackRetrievedViaSustainedVanish(Stage):
    name = "stage_30_retrieved_via_sustained_vanish_fallback"
    target_class = "retrieved"

    def decide(self, seg):
        r = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        feats = {"n_reaches": len(r)}
        if not r:
            return StageDecision(decision="continue", reason="no_reaches", features=feats)
        last_rs, last_re = r[-1]
        if seg.seg_end + 1 - (last_re + 1) < 100:
            return StageDecision(decision="continue", reason="window_too_short", features=feats)
        mr = _max_consecutive_low_lk(seg.dlc_df, last_re + 1, seg.seg_end + 1)
        feats["max_vanish_run"] = int(mr)
        if mr < 100:
            return StageDecision(decision="continue", reason=f"vanish_short ({mr})", features=feats)
        first_rs = r[0][0]
        pre_p = (
            seg.dlc_df["Pillar_likelihood"].iloc[seg.seg_start:first_rs].to_numpy(dtype=float)
            if first_rs > seg.seg_start
            else np.array([], dtype=float)
        )
        post_p = seg.dlc_df["Pillar_likelihood"].iloc[last_re + 1:seg.seg_end + 1].to_numpy(dtype=float)
        if not len(pre_p) or not len(post_p):
            return StageDecision(decision="continue", reason="empty_pillar_windows", features=feats)
        pm = float(np.mean(pre_p))
        qm = float(np.mean(post_p))
        feats["pillar_lk_pre_mean"] = round(pm, 3)
        feats["pillar_lk_post_mean"] = round(qm, 3)
        if pm >= 0.5:
            return StageDecision(decision="continue", reason="pillar_pre_high", features=feats)
        if qm <= 0.85:
            return StageDecision(decision="continue", reason="pillar_post_low", features=feats)
        max_disp = 0.0
        for rs, re in r:
            pws = max(seg.seg_start, rs - 30)
            pwe = min(seg.seg_end + 1, re + 1 + 30)
            pre_lk = seg.dlc_df["Pellet_likelihood"].iloc[pws:rs].to_numpy(dtype=float)
            post_lk = seg.dlc_df["Pellet_likelihood"].iloc[re + 1:pwe].to_numpy(dtype=float)
            pc = pre_lk >= 0.7
            qc = post_lk >= 0.7
            if not pc.any() or not qc.any():
                continue
            px = seg.dlc_df["Pellet_x"].iloc[pws:rs].to_numpy(dtype=float)
            py = seg.dlc_df["Pellet_y"].iloc[pws:rs].to_numpy(dtype=float)
            qx = seg.dlc_df["Pellet_x"].iloc[re + 1:pwe].to_numpy(dtype=float)
            qy = seg.dlc_df["Pellet_y"].iloc[re + 1:pwe].to_numpy(dtype=float)
            d = float(np.sqrt(
                (float(np.median(qx[qc])) - float(np.median(px[pc]))) ** 2
                + (float(np.median(qy[qc])) - float(np.median(py[pc]))) ** 2
            ))
            if d > max_disp:
                max_disp = d
        feats["max_competing_disp_px"] = round(max_disp, 2)
        if max_disp >= 12.0:
            return StageDecision(
                decision="continue",
                reason=f"competing_disp ({max_disp:.1f}px)",
                features=feats,
            )
        return StageDecision(
            decision="commit",
            committed_class="retrieved",
            whens={
                "interaction_frame": int(last_rs + (last_re - last_rs + 1) // 2),
                "outcome_known_frame": int(last_re + 5),
            },
            reason=f"sustained_vanish_fallback ({mr}f)",
            features=feats,
        )


STAGE31_MIN_VANISH = 100
STAGE31_PILLAR_PRE_MAX = 0.5
STAGE31_PILLAR_POST_MIN = 0.85


class Stage31RetrievedRescueWithDisplacement(Stage):
    name = "stage_31_retrieved_rescue_with_displacement"
    target_class = "retrieved"

    def decide(self, seg: SegmentInput) -> StageDecision:
        r = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        feats = {"n_reaches": len(r)}
        if not r:
            return StageDecision(decision="continue", reason="no_reaches", features=feats)
        last_rs, last_re = r[-1]
        scan_lo = last_re + 1
        scan_hi = seg.seg_end + 1
        if scan_hi - scan_lo < STAGE31_MIN_VANISH:
            return StageDecision(
                decision="continue",
                reason=f"window_too_short ({scan_hi - scan_lo})",
                features=feats,
            )
        mr = _max_consecutive_low_lk(seg.dlc_df, scan_lo, scan_hi, thr=0.5)
        feats["max_vanish_run"] = int(mr)
        if mr < STAGE31_MIN_VANISH:
            return StageDecision(
                decision="continue",
                reason=f"insufficient_vanish ({mr} < {STAGE31_MIN_VANISH})",
                features=feats,
            )
        first_rs = r[0][0]
        pre_p = (
            seg.dlc_df["Pillar_likelihood"].iloc[seg.seg_start:first_rs].to_numpy(dtype=float)
            if first_rs > seg.seg_start
            else np.array([], dtype=float)
        )
        post_p = seg.dlc_df["Pillar_likelihood"].iloc[last_re + 1:seg.seg_end + 1].to_numpy(dtype=float)
        if not len(pre_p) or not len(post_p):
            return StageDecision(decision="continue", reason="empty_pillar_windows", features=feats)
        pm = float(np.mean(pre_p))
        qm = float(np.mean(post_p))
        feats["pillar_lk_pre_mean"] = round(pm, 3)
        feats["pillar_lk_post_mean"] = round(qm, 3)
        if pm >= STAGE31_PILLAR_PRE_MAX:
            return StageDecision(
                decision="continue",
                reason=f"pillar_pre_high ({pm:.2f} >= {STAGE31_PILLAR_PRE_MAX})",
                features=feats,
            )
        if qm <= STAGE31_PILLAR_POST_MIN:
            return StageDecision(
                decision="continue",
                reason=f"pillar_post_low ({qm:.2f} <= {STAGE31_PILLAR_POST_MIN})",
                features=feats,
            )
        bout_length = last_re - last_rs + 1
        return StageDecision(
            decision="commit",
            committed_class="retrieved",
            whens={
                "interaction_frame": int(last_rs + bout_length // 2),
                "outcome_known_frame": int(last_re + 5),
            },
            reason=(
                f"retrieved_rescue_with_displacement_allowed "
                f"(vanish={mr}f, pillar {pm:.2f}->{qm:.2f})"
            ),
            features=feats,
        )


class Stage32ZeroReachUntouchedFallback(Stage):
    name = "stage_32_zero_reach_untouched_fallback"
    target_class = "untouched"

    def decide(self, seg):
        TZ = 5
        feats = {"n_reaches": len(seg.reach_windows)}
        if seg.reach_windows:
            return StageDecision(decision="continue", reason="reaches_present", features=feats)
        clean_end = seg.seg_end - TZ
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue", reason="too_short", features=feats)
        sub = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub)
        if n < 60:
            return StageDecision(decision="continue", reason="too_short_stability", features=feats)
        plk = sub["Pellet_likelihood"].to_numpy(dtype=float)
        vf = float((plk >= 0.7).mean())
        feats["pellet_vis_frac"] = round(vf, 3)
        if vf < 0.8:
            return StageDecision(decision="continue", reason="pellet_obscured", features=feats)
        px = sub["Pellet_x"].to_numpy(dtype=float)
        py = sub["Pellet_y"].to_numpy(dtype=float)
        early = (np.arange(n) < 30) & (plk >= 0.7)
        late = (np.arange(n) >= n - 30) & (plk >= 0.7)
        if early.sum() < 5 or late.sum() < 5:
            return StageDecision(decision="continue", reason="early_late_too_thin", features=feats)
        drift = float(np.sqrt(
            (float(np.median(px[late])) - float(np.median(px[early]))) ** 2
            + (float(np.median(py[late])) - float(np.median(py[early]))) ** 2
        ))
        feats["pellet_pos_drift_px"] = round(drift, 2)
        if drift > 10:
            return StageDecision(
                decision="continue",
                reason=f"drift ({drift:.1f}px)",
                features=feats,
            )
        return StageDecision(
            decision="commit",
            committed_class="untouched",
            whens={"outcome_known_frame": int(clean_end), "interaction_frame": None},
            reason=f"zero_reaches + stable_pellet (vis={vf:.2f}, drift={drift:.1f}px)",
            features=feats,
        )


# ---------------------------------------------------------------------------
# Fix C: Stage 21 with Stage-9-aligned retrieved corroboration
# ---------------------------------------------------------------------------

# Thresholds from Stage 9. Stage 21's retrieved branch must now satisfy
# the Stage 9 anti-displaced gate: post-causal sustained off-pillar count
# (>=3 consecutive frames of pellet_lk>=0.7 AND dist_radii>1.5 AND
# not paw_past_y) must be <= MAX_POST_OFF_FOR_RETRIEVED (5).
EDIT2_PELLET_LK_THR = 0.7
EDIT2_OFF_PILLAR_RADII = 1.5
EDIT2_PAW_LK_THR = 0.5
EDIT2_MIN_SUSTAINED_RUN = 3
EDIT2_MAX_POST_OFF_FOR_RETRIEVED = 5


def _sustained_mask(arr: np.ndarray, min_run: int) -> np.ndarray:
    """Boolean mask: True at frames inside a True-run of length >= min_run."""
    out = np.zeros_like(arr, dtype=bool)
    run = 0
    for i in range(len(arr)):
        if arr[i]:
            run += 1
        else:
            if run >= min_run:
                out[i - run:i] = True
            run = 0
    if run >= min_run:
        out[len(arr) - run:] = True
    return out


class Stage21WithRetrievedCorroboration(Stage21CausalReachViaImmediateOnOffTransition):
    """Stage 21 with a Stage-9-aligned anti-displaced check before
    committing 'retrieved'. The displaced_sa branch is unchanged.

    The check: in the segment from causal reach onward, count sustained
    off-pillar pellet observations (>= 3 consecutive frames of pellet_lk
    >= 0.7, dist_radii > 1.5, and not paw_past_y). If > 5 sustained
    off-pillar frames, defer (segment looks displaced, not retrieved).
    """

    name = "stage_21_with_retrieved_corroboration"

    def decide(self, seg: SegmentInput) -> StageDecision:
        parent = super().decide(seg)
        if parent.decision != "commit":
            return parent
        if parent.committed_class != "retrieved":
            return parent  # displaced_sa branch unchanged

        # Identify causal reach end (segment-relative) from parent features.
        ci = parent.features.get("causal_idx")
        if ci is None:
            return parent

        # Replicate Stage 21's reach_windows_local construction.
        clean_end = seg.seg_end - self.transition_zone_half
        if clean_end <= seg.seg_start:
            return parent
        n = clean_end - seg.seg_start + 1

        reach_windows_local = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le >= ls:
                reach_windows_local.append((ls, le))
        reach_windows_local.sort()
        if ci >= len(reach_windows_local):
            return parent
        _, be = reach_windows_local[ci]  # causal reach end (segment-relative)

        # Compute pellet + pillar + paw features over the segment (same
        # pipeline as Stage 9).
        sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        if len(sub_raw) == 0:
            return parent
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        geom = compute_pillar_geometry_series(sub)
        pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
        pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
        pillar_r = geom["pillar_r"].to_numpy(dtype=float)
        slit_y_line = pillar_cy + pillar_r

        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        dist_radii = (
            np.sqrt((pellet_x - pillar_cx) ** 2 + (pellet_y - pillar_cy) ** 2)
            / np.maximum(pillar_r, 1e-6)
        )

        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            py_bp = sub[f"{bp}_y"].to_numpy(dtype=float)
            pl_bp = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (py_bp <= slit_y_line) & (pl_bp >= EDIT2_PAW_LK_THR)

        # Anti-displaced raw mask: confident pellet, off-pillar, paw not past slit.
        off_pillar_anywhere_raw = (
            (pellet_lk >= EDIT2_PELLET_LK_THR)
            & (dist_radii > EDIT2_OFF_PILLAR_RADII)
            & (~paw_past_y)
        )
        off_pillar_anywhere = _sustained_mask(
            off_pillar_anywhere_raw, EDIT2_MIN_SUSTAINED_RUN
        )

        # Count sustained off-pillar frames AFTER the causal reach end.
        post_causal_sustained_off = int(off_pillar_anywhere[be + 1:].sum())

        f = dict(parent.features)
        f["edit2_post_causal_sustained_off"] = post_causal_sustained_off

        if post_causal_sustained_off > EDIT2_MAX_POST_OFF_FOR_RETRIEVED:
            f["edit2_anti_displaced_fired"] = True
            return StageDecision(
                decision="continue",
                reason=(
                    f"retrieved_commit_anti_displaced_veto "
                    f"({post_causal_sustained_off}f sustained off-pillar "
                    f"post-causal > {EDIT2_MAX_POST_OFF_FOR_RETRIEVED}; "
                    f"looks displaced not retrieved)"
                ),
                features=f,
            )

        return StageDecision(
            decision=parent.decision,
            committed_class=parent.committed_class,
            whens=parent.whens,
            reason=parent.reason,
            features=f,
        )


# ---------------------------------------------------------------------------
# Cascade build
# ---------------------------------------------------------------------------

def build_stages_with_fix_c(video_dir=None):
    base = _build_production_stages(video_dir=video_dir)
    modified = []
    for label, stage in base:
        if isinstance(stage, Stage16DisplacedViaMaxDisplacement):
            modified.append(("stage_16_with_vanish_veto", Stage16WithVanishVeto()))
        elif isinstance(stage, Stage27DisplacedSaViaUniqueHighDisplacement):
            modified.append(("stage_27_with_vanish_veto", Stage27WithVanishVeto()))
        elif isinstance(stage, Stage21CausalReachViaImmediateOnOffTransition):
            modified.append((
                "stage_21_with_retrieved_corroboration",
                Stage21WithRetrievedCorroboration(),
            ))
        elif "stage_99_residual_triage" in label:
            modified.append((
                "stage_30_retrieved_via_sustained_vanish_fallback",
                Stage30FallbackRetrievedViaSustainedVanish(),
            ))
            modified.append((
                "stage_31_retrieved_rescue_with_displacement",
                Stage31RetrievedRescueWithDisplacement(),
            ))
            modified.append((
                "stage_32_zero_reach_untouched_fallback",
                Stage32ZeroReachUntouchedFallback(),
            ))
            modified.append((label, stage))
        else:
            modified.append((label, stage))
    return modified


def run_cascade_on_segments(seg_inputs, stages):
    out = {}
    for si in seg_inputs:
        decision = None
        committing = "residual (auto-triage)"
        for label, stage in stages:
            dec = stage.decide(si)
            if dec.decision in ("commit", "triage"):
                decision = dec
                committing = label
                break
        if decision is not None and decision.decision == "commit":
            rec = {
                "segment_num": si.segment_num,
                "outcome": decision.committed_class,
                "outcome_known_frame": decision.whens.get("outcome_known_frame"),
                "interaction_frame": decision.whens.get("interaction_frame"),
                "stage": committing,
                "flagged_for_review": False,
            }
        else:
            reason = decision.reason if decision is not None else "fell_through"
            rec = {
                "segment_num": si.segment_num,
                "outcome": "triaged",
                "outcome_known_frame": None,
                "interaction_frame": None,
                "stage": committing,
                "flagged_for_review": True,
                "flag_reason": reason,
            }
        out.setdefault(si.video_id, []).append(rec)
    return out


# ---------------------------------------------------------------------------
# Paths + main
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11"
)
DLC_DIR = CORPUS_DIR / "algo_outputs_current"
GT_DIR = CORPUS_DIR / "gt"
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\Improvement_Snapshots\outcome"
    r"\v6.0.4_fix_c_stage_21_corroboration_2026-06-02"
)


def find_dlc(vid):
    m = sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))
    if not m:
        raise FileNotFoundError(f"No DLC for {vid}")
    return m[0]


def load_gt_segments(vid):
    gt = json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    bs = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    if len(bs) < 2:
        return []
    return [(bs[i], bs[i + 1] - 1) for i in range(len(bs) - 1)]


def save_reaches_segmented(vid, reaches, segments, out_path):
    segs_data = []
    gid = 0
    for si, (s, e) in enumerate(segments):
        sn = si + 1
        rs_list = []
        rn = 0
        for r0, r1 in reaches:
            if s <= r0 <= e:
                rn += 1
                gid += 1
                rs_list.append({
                    "reach_id": gid,
                    "reach_num": rn,
                    "start_frame": int(r0),
                    "end_frame": int(r1),
                    "duration_frames": int(r1 - r0 + 1),
                })
        segs_data.append({
            "segment_num": sn,
            "start_frame": int(s),
            "end_frame": int(e),
            "n_reaches": len(rs_list),
            "reaches": rs_list,
        })
    out_path.write_text(
        json.dumps({
            "video_id": vid,
            "detector": "v8.0.4_production",
            "n_segments": len(segments),
            "segments": segs_data,
        }, indent=2),
        encoding="utf-8",
    )


def main():
    t0 = time.time()
    print("Fix C experiment: Stage 21 retrieved corroboration (Edit 2)")
    print(f"Snapshot dir: {SNAPSHOT_DIR}")
    algo_dir = SNAPSHOT_DIR / "algo_outputs"
    metrics_dir = SNAPSHOT_DIR / "metrics"
    algo_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    video_ids = sorted({
        p.stem.replace("_unified_ground_truth", "")
        for p in GT_DIR.glob("*_unified_ground_truth.json")
    })
    print(f"Found {len(video_ids)} videos")
    stages = build_stages_with_fix_c(video_dir=None)
    for i, vid in enumerate(video_ids, 1):
        t_vid = time.time()
        print(f"[{i}/{len(video_ids)}] {vid}", flush=True)
        dlc = load_dlc_h5(find_dlc(vid))
        segments = load_gt_segments(vid)
        reaches = detect_reaches_v8(dlc)
        print(f"   detected {len(reaches)} reaches", flush=True)
        save_reaches_segmented(vid, reaches, segments, algo_dir / f"{vid}_reaches.json")
        seg_inputs = []
        for si_idx, (s, e) in enumerate(segments):
            sn = si_idx + 1
            seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
            seg_inputs.append(SegmentInput(
                video_id=vid,
                segment_num=sn,
                seg_start=s,
                seg_end=e,
                dlc_df=dlc,
                reach_windows=seg_r,
            ))
        outs = run_cascade_on_segments(seg_inputs, stages)
        if vid in outs:
            (algo_dir / f"{vid}_pellet_outcomes.json").write_text(
                json.dumps({
                    "video_id": vid,
                    "detector": "v6_cascade_fix_c",
                    "detector_version": "6.0.4_fix_c",
                    "reach_detector_version": "v8.0.4_production",
                    "segments": outs[vid],
                }, indent=2),
                encoding="utf-8",
            )
        print(f"   ({time.time() - t_vid:.1f}s)", flush=True)
    print("\nScoring with compute_outcome_metrics...", flush=True)
    compute_outcome_metrics(
        gt_dir=GT_DIR,
        algo_dir=algo_dir,
        output_dir=metrics_dir,
        video_ids=video_ids,
        reaches_dir=algo_dir,
    )
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Snapshot complete at: {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
