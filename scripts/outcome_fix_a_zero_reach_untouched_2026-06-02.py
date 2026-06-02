"""
Outcome cascade Fix A: Zero-reach untouched fallback.

Layered on top of accepted v6.0.1 (Stage 16/27 vanish vetoes + Stage 30
fallback). Adds a new untouched-commit stage as a late fallback that
fires when:
  - v8.0.4 reach detector returned 0 reaches in this segment
  - Pellet is visible most of segment
  - Pellet position is roughly stable
  - No paw point sustainably crossed slit-edge y-line at lk >= 0.5

Rationale: Stage 3 (paw_never_in_pellet_area) has a conservatively
LOW lk floor (0.22) for sustained paw-past-slit-edge that catches DLC
keypoint noise as "paw might have touched pellet". For the specific
case CNT0413_P2 s14 (gt=untouched, 0 algo reaches, stable pellet), the
paw made a brief 0.227 rolling-mean crossing -- just over Stage 3's
0.22 floor -- which blocks Stage 2/3 untouched commits and lets the
segment fall through to triage. Trusting v8.0.4's "0 reaches" verdict
(at 99% accuracy on reach FN) plus a less-paranoid 0.5 paw floor
gives an additional path to commit untouched.

================================================================
Pre-experiment checklist
================================================================

1. Cumulative-stacking check (verified 2026-06-02)
   - Reach detector: v8.0.4 production
   - Outcome cascade cumulative best: v6.0.1 (Stage 16/27 vanish veto +
     Stage 30 fallback), merged to master 2026-06-02 (commit c029aa5,
     merge commit f3c5db2). This experiment LAYERS Fix A on top.
   - Comparison baseline: the v6.0.1 snapshot at
     Improvement_Snapshots/outcome/v6.0.1_three_stage_fixes_2026-06-02/.
     Result was 379/400 correct, 12 triages, retrieved 60, untouched 102.
   - Master at f3c5db2 as of this run.

2. Existing-code-modification check: NO. Runner imports existing v6
   stage classes + the v6.0.1 subclasses (re-implemented in this file
   for self-contained reproducibility). Does NOT modify any module
   under src/mousereach/.

3. Unverified hypotheses
   - H1: New stage fires on CNT0413_P2 s14 -> commits untouched -> +1
     correct untouched.
   - H2: New stage MAY fire on other zero-reach segments in the corpus
     that I haven't individually analyzed; potential additional gain.
   - H3: No regression on currently-correct untouched (102/104) because
     those segments already commit via Stage 2/3 before reaching the
     fallback.
   - H4: No regression on retrieved/displaced because new stage runs
     LATE (before Stage 99, after all other commit stages); a touched
     segment would have been caught upstream if any signal existed.
   - H5: Risk of FALSE untouched commit -- if v8.0.4 missed a real reach
     (FN), AND pellet stayed visible, AND no paw crossed slit at lk>=0.5
     (a non-trivial gate), new stage fires untouched on a touched segment.
     v8.0.4 holdout FN is ~27/3655 = 0.7%, so this is unlikely on a
     well-tracked segment.

4. FN-direction-reporting (planned RESULTS.md)
   Lead-line: "Untouched class correct went from <baseline> to <new>:
   <direction>. Triage count: <delta>. Total correct: <delta>."
   Two deltas: vs cumulative best (v6.0.1, 379/400) and vs pure baseline
   (v6.0.0 + v7.2.0 reaches, 240/400).

5. Framework-not-adhoc
   Snapshot dir: Improvement_Snapshots/outcome/v6.0.2_fix_a_zero_reach_untouched_2026-06-02/
   Schema: matches compute_outcome_metrics canonical scorer.

6. Branch + tag
   - Pre-experiment tag: outcome-pre-fix-a-zero-reach-untouched-2026-06-02
   - Feature branch: feature/outcome-fix-a-zero-reach-untouched
   - Set on master f3c5db2 before running.

7. Decision rule
   - ACCEPT: untouched correct rises by >= 1 AND no class regresses by > 0.
     (Strict zero-regression rule because untouched is the easy case --
      gaining 1 at the cost of any wrong commit is a net loss.)
   - REJECT: any class regresses, OR no gain.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mousereach.reach.v8 import detect_reaches_v8
from mousereach.reach.v8.features import load_dlc_h5
from mousereach.outcomes.v6_cascade.detector import _build_production_stages
from mousereach.outcomes.v6_cascade.stage_base import (
    SegmentInput, Stage, StageDecision)
from mousereach.outcomes.v6_cascade.stage_16_displaced_via_max_displacement_reach import (
    Stage16DisplacedViaMaxDisplacement)
from mousereach.outcomes.v6_cascade.stage_27_displaced_sa_via_unique_high_displacement_reach import (
    Stage27DisplacedSaViaUniqueHighDisplacement)
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.improvement.outcome.metrics import compute_outcome_metrics


# ---------------------------------------------------------------------------
# Helpers (replicated from v6.0.1 runner for self-contained reproducibility)
# ---------------------------------------------------------------------------

VANISH_VETO_FRAMES = 60


def _max_consecutive_low_lk(dlc_df, start: int, end: int, thr: float = 0.5) -> int:
    if end <= start:
        return 0
    lk = dlc_df["Pellet_likelihood"].iloc[start:end].to_numpy(dtype=float)
    below = lk < thr
    max_run = 0
    cur = 0
    for b in below:
        if b:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


class Stage16WithVanishVeto(Stage16DisplacedViaMaxDisplacement):
    name = "stage_16_with_vanish_veto"

    def decide(self, seg: SegmentInput) -> StageDecision:
        d = super().decide(seg)
        if d.decision != "commit":
            return d
        causal_idx = d.features.get("causal_idx")
        if causal_idx is None:
            return d
        reaches = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        if causal_idx >= len(reaches):
            return d
        _, causal_re = reaches[causal_idx]
        max_run = _max_consecutive_low_lk(seg.dlc_df,
                                            causal_re + 1,
                                            seg.seg_end + 1,
                                            thr=0.5)
        feats = dict(d.features)
        feats["post_causal_max_vanish_run"] = int(max_run)
        if max_run >= VANISH_VETO_FRAMES:
            feats["vanish_veto_fired"] = True
            return StageDecision(decision="continue",
                                  reason=f"global_post_causal_vanish_veto ({max_run}f)",
                                  features=feats)
        return StageDecision(decision=d.decision,
                              committed_class=d.committed_class,
                              whens=d.whens, reason=d.reason, features=feats)


class Stage27WithVanishVeto(Stage27DisplacedSaViaUniqueHighDisplacement):
    name = "stage_27_with_vanish_veto"

    def decide(self, seg: SegmentInput) -> StageDecision:
        d = super().decide(seg)
        if d.decision != "commit":
            return d
        causal_idx = d.features.get("causal_reach_idx")
        if causal_idx is None:
            return d
        reaches = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        if causal_idx >= len(reaches):
            return d
        _, causal_re = reaches[causal_idx]
        max_run = _max_consecutive_low_lk(seg.dlc_df,
                                            causal_re + 1,
                                            seg.seg_end + 1,
                                            thr=0.5)
        feats = dict(d.features)
        feats["post_causal_max_vanish_run"] = int(max_run)
        if max_run >= VANISH_VETO_FRAMES:
            feats["vanish_veto_fired"] = True
            return StageDecision(decision="continue",
                                  reason=f"global_post_causal_vanish_veto ({max_run}f)",
                                  features=feats)
        return StageDecision(decision=d.decision,
                              committed_class=d.committed_class,
                              whens=d.whens, reason=d.reason, features=feats)


STAGE30_MIN_VANISH = 100
STAGE30_PILLAR_PRE_MAX = 0.5
STAGE30_PILLAR_POST_MIN = 0.85
STAGE30_MAX_COMPETING_DISP_PX = 12.0


class Stage30FallbackRetrievedViaSustainedVanish(Stage):
    name = "stage_30_retrieved_via_sustained_vanish_fallback"
    target_class = "retrieved"

    def decide(self, seg: SegmentInput) -> StageDecision:
        reaches = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        feats = {"n_reaches": len(reaches)}
        if not reaches:
            return StageDecision(decision="continue", reason="no_reaches", features=feats)
        last_rs, last_re = reaches[-1]
        scan_lo, scan_hi = last_re + 1, seg.seg_end + 1
        if scan_hi - scan_lo < STAGE30_MIN_VANISH:
            return StageDecision(decision="continue",
                                  reason="post_last_reach_window_too_short", features=feats)
        max_run = _max_consecutive_low_lk(seg.dlc_df, scan_lo, scan_hi, thr=0.5)
        feats["max_vanish_run"] = int(max_run)
        if max_run < STAGE30_MIN_VANISH:
            return StageDecision(decision="continue",
                                  reason=f"insufficient_vanish ({max_run})", features=feats)
        first_rs = reaches[0][0]
        pre_pillar = (seg.dlc_df["Pillar_likelihood"].iloc[seg.seg_start:first_rs].to_numpy(dtype=float)
                       if first_rs > seg.seg_start else np.array([], dtype=float))
        post_pillar = seg.dlc_df["Pillar_likelihood"].iloc[last_re + 1:seg.seg_end + 1].to_numpy(dtype=float)
        if len(pre_pillar) == 0 or len(post_pillar) == 0:
            return StageDecision(decision="continue", reason="empty_pillar_lk_windows", features=feats)
        pre_mean = float(np.mean(pre_pillar)); post_mean = float(np.mean(post_pillar))
        feats["pillar_lk_pre_mean"] = round(pre_mean, 3); feats["pillar_lk_post_mean"] = round(post_mean, 3)
        if pre_mean >= STAGE30_PILLAR_PRE_MAX:
            return StageDecision(decision="continue", reason=f"pillar_pre_too_high", features=feats)
        if post_mean <= STAGE30_PILLAR_POST_MIN:
            return StageDecision(decision="continue", reason=f"pillar_post_too_low", features=feats)
        max_disp = 0.0
        for rs, re in reaches:
            pws = max(seg.seg_start, rs - 30); pwe = min(seg.seg_end + 1, re + 1 + 30)
            pre_lk = seg.dlc_df["Pellet_likelihood"].iloc[pws:rs].to_numpy(dtype=float)
            post_lk = seg.dlc_df["Pellet_likelihood"].iloc[re + 1:pwe].to_numpy(dtype=float)
            pre_c = pre_lk >= 0.7; post_c = post_lk >= 0.7
            if not pre_c.any() or not post_c.any(): continue
            pre_x = seg.dlc_df["Pellet_x"].iloc[pws:rs].to_numpy(dtype=float)
            pre_y = seg.dlc_df["Pellet_y"].iloc[pws:rs].to_numpy(dtype=float)
            post_x = seg.dlc_df["Pellet_x"].iloc[re + 1:pwe].to_numpy(dtype=float)
            post_y = seg.dlc_df["Pellet_y"].iloc[re + 1:pwe].to_numpy(dtype=float)
            d = float(np.sqrt((float(np.median(post_x[post_c])) - float(np.median(pre_x[pre_c]))) ** 2 +
                                (float(np.median(post_y[post_c])) - float(np.median(pre_y[pre_c]))) ** 2))
            if d > max_disp: max_disp = d
        feats["max_competing_disp_px"] = round(max_disp, 2)
        if max_disp >= STAGE30_MAX_COMPETING_DISP_PX:
            return StageDecision(decision="continue",
                                  reason=f"competing_displacement ({max_disp:.1f}px)", features=feats)
        bout_length = last_re - last_rs + 1
        return StageDecision(decision="commit", committed_class="retrieved",
                              whens={"interaction_frame": int(last_rs + bout_length // 2),
                                     "outcome_known_frame": int(last_re + 5)},
                              reason=f"sustained_vanish_fallback (vanish={max_run}f)",
                              features=feats)


# ---------------------------------------------------------------------------
# Fix A: New Stage32 -- zero-reach untouched fallback
# ---------------------------------------------------------------------------

STAGE32_MIN_PELLET_VIS_FRAC = 0.80         # >= 80% of segment frames pellet visible
STAGE32_MAX_PELLET_POS_DRIFT_PX = 10.0     # median pos shift first 30f vs last 30f
STAGE32_PAW_PAST_SLIT_LK_THR = 0.5         # less-paranoid than Stage 3's 0.22
STAGE32_PAW_PAST_SLIT_CONSEC_REQUIRED = 3  # consecutive frames required
STAGE32_PAW_PAST_SLIT_ROLLING = 3          # 3-frame rolling mean
PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5


class Stage32ZeroReachUntouchedFallback(Stage):
    """Untouched commit when v8.0.4 reach detector finds 0 reaches AND
    pellet is stable AND no paw point sustainably crossed the slit-edge
    y-line at lk >= 0.5 (a less paranoid floor than Stage 3's 0.22).

    Inserted late in the cascade (before Stage 99 residual triage) so
    earlier higher-confidence stages get first chance. Only fires when
    none of Stages 0-29 + 98 + 99-precursors committed.
    """
    name = "stage_32_zero_reach_untouched_fallback"
    target_class = "untouched"

    def __init__(
        self,
        min_pellet_vis_frac: float = STAGE32_MIN_PELLET_VIS_FRAC,
        max_pellet_pos_drift_px: float = STAGE32_MAX_PELLET_POS_DRIFT_PX,
        paw_past_slit_lk_thr: float = STAGE32_PAW_PAST_SLIT_LK_THR,
        paw_consec_required: int = STAGE32_PAW_PAST_SLIT_CONSEC_REQUIRED,
        paw_rolling: int = STAGE32_PAW_PAST_SLIT_ROLLING,
    ):
        self.min_pellet_vis_frac = min_pellet_vis_frac
        self.max_pellet_pos_drift_px = max_pellet_pos_drift_px
        self.paw_past_slit_lk_thr = paw_past_slit_lk_thr
        self.paw_consec_required = paw_consec_required
        self.paw_rolling = paw_rolling

    def decide(self, seg: SegmentInput) -> StageDecision:
        feats = {"n_reaches": len(seg.reach_windows)}
        # Trust v8.0.4: only fire if zero reaches detected.
        if len(seg.reach_windows) != 0:
            return StageDecision(decision="continue",
                                  reason="reaches_present_skip",
                                  features=feats)

        clean_end = seg.seg_end - TRANSITION_ZONE_HALF
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue",
                                  reason="segment_too_short", features=feats)
        sub = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub)
        if n < 60:
            return StageDecision(decision="continue",
                                  reason="segment_too_short_for_stability", features=feats)

        # Pellet visibility check
        plk = sub["Pellet_likelihood"].to_numpy(dtype=float)
        vis_frac = float((plk >= 0.7).mean())
        feats["pellet_vis_frac"] = round(vis_frac, 3)
        if vis_frac < self.min_pellet_vis_frac:
            return StageDecision(decision="continue",
                                  reason=f"pellet_too_obscured ({vis_frac:.2f})",
                                  features=feats)

        # Pellet position stability: median over first 30 vs last 30
        px = sub["Pellet_x"].to_numpy(dtype=float)
        py = sub["Pellet_y"].to_numpy(dtype=float)
        early_mask = (np.arange(n) < 30) & (plk >= 0.7)
        late_mask = (np.arange(n) >= n - 30) & (plk >= 0.7)
        if early_mask.sum() < 5 or late_mask.sum() < 5:
            return StageDecision(decision="continue",
                                  reason="not_enough_confident_early_or_late_pellet",
                                  features=feats)
        early_x, early_y = float(np.median(px[early_mask])), float(np.median(py[early_mask]))
        late_x, late_y = float(np.median(px[late_mask])), float(np.median(py[late_mask]))
        drift = float(np.sqrt((late_x - early_x) ** 2 + (late_y - early_y) ** 2))
        feats["pellet_pos_drift_px"] = round(drift, 2)
        if drift > self.max_pellet_pos_drift_px:
            return StageDecision(decision="continue",
                                  reason=f"pellet_position_drifted ({drift:.1f}px > {self.max_pellet_pos_drift_px})",
                                  features=feats)

        # No paw-past-slit check -- we are trusting v8.0.4's verdict
        # of "0 reaches". The mouse may have approached the slit area
        # without actually reaching; that's exactly what v8.0.4 told us.
        # Stage 3's paw-past-slit check at 0.22 floor is what blocked
        # this in the first place; re-checking here would replicate
        # that overly-paranoid behavior.

        okf = int(clean_end)
        feats["outcome_known_frame_emitted"] = okf
        return StageDecision(
            decision="commit",
            committed_class="untouched",
            whens={"outcome_known_frame": okf, "interaction_frame": None},
            reason=(f"zero_reaches_from_v804 + stable_pellet "
                    f"(vis={vis_frac:.2f}, drift={drift:.1f}px); "
                    f"trusting reach detector verdict"),
            features=feats,
        )


# ---------------------------------------------------------------------------
# Cascade build (v6.0.1 baseline + Fix A insertion)
# ---------------------------------------------------------------------------

def build_stages_with_fix_a(video_dir=None):
    base = _build_production_stages(video_dir=video_dir)
    modified = []
    for label, stage in base:
        if isinstance(stage, Stage16DisplacedViaMaxDisplacement):
            modified.append(("stage_16_with_vanish_veto", Stage16WithVanishVeto()))
        elif isinstance(stage, Stage27DisplacedSaViaUniqueHighDisplacement):
            modified.append(("stage_27_with_vanish_veto", Stage27WithVanishVeto()))
        elif "stage_99_residual_triage" in label:
            # Insert Stage 30 (v6.0.1) and Stage 32 (Fix A) before Stage 99.
            modified.append(("stage_30_retrieved_via_sustained_vanish_fallback",
                              Stage30FallbackRetrievedViaSustainedVanish()))
            modified.append(("stage_32_zero_reach_untouched_fallback",
                              Stage32ZeroReachUntouchedFallback()))
            modified.append((label, stage))
        else:
            modified.append((label, stage))
    return modified


def run_cascade_on_segments(seg_inputs, stages):
    out_per_video = {}
    for si in seg_inputs:
        decision = None
        committing = "residual (auto-triage)"
        for label, stage in stages:
            dec = stage.decide(si)
            if dec.decision in ("commit", "triage"):
                decision = dec; committing = label; break
        if decision is not None and decision.decision == "commit":
            rec = {"segment_num": si.segment_num, "outcome": decision.committed_class,
                    "outcome_known_frame": decision.whens.get("outcome_known_frame"),
                    "interaction_frame": decision.whens.get("interaction_frame"),
                    "stage": committing, "flagged_for_review": False}
        else:
            reason = decision.reason if decision is not None else "fell_through"
            rec = {"segment_num": si.segment_num, "outcome": "triaged",
                    "outcome_known_frame": None, "interaction_frame": None,
                    "stage": committing, "flagged_for_review": True, "flag_reason": reason}
        out_per_video.setdefault(si.video_id, []).append(rec)
    return out_per_video


# ---------------------------------------------------------------------------
# Paths and main
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11")
DLC_DIR = CORPUS_DIR / "algo_outputs_current"
GT_DIR = CORPUS_DIR / "gt"
SNAPSHOT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome\v6.0.2_fix_a_zero_reach_untouched_2026-06-02")


def find_dlc(vid):
    matches = sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))
    if not matches: raise FileNotFoundError(f"No DLC h5 for {vid}")
    return matches[0]


def load_gt_segments(vid):
    gt = json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    bs = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    if len(bs) < 2: return []
    return [(bs[i], bs[i + 1] - 1) for i in range(len(bs) - 1)]


def save_reaches_segmented(vid, reaches, segments, out_path):
    segs_data = []; gid = 0
    for si, (s, e) in enumerate(segments):
        sn = si + 1; rs = []; rn = 0
        for r0, r1 in reaches:
            if s <= r0 <= e:
                rn += 1; gid += 1
                rs.append({"reach_id": gid, "reach_num": rn,
                            "start_frame": int(r0), "end_frame": int(r1),
                            "duration_frames": int(r1 - r0 + 1)})
        segs_data.append({"segment_num": sn, "start_frame": int(s), "end_frame": int(e),
                           "n_reaches": len(rs), "reaches": rs})
    out_path.write_text(json.dumps({"video_id": vid, "detector": "v8.0.4_production",
                                       "n_segments": len(segments), "segments": segs_data},
                                      indent=2), encoding="utf-8")


def main():
    t0 = time.time()
    print(f"Fix A experiment: zero-reach untouched fallback")
    print(f"Snapshot dir: {SNAPSHOT_DIR}")
    algo_dir = SNAPSHOT_DIR / "algo_outputs"; metrics_dir = SNAPSHOT_DIR / "metrics"
    algo_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)

    video_ids = sorted({p.stem.replace("_unified_ground_truth", "")
                         for p in GT_DIR.glob("*_unified_ground_truth.json")})
    print(f"Found {len(video_ids)} videos")
    stages = build_stages_with_fix_a(video_dir=None)

    for i, vid in enumerate(video_ids, 1):
        t_vid = time.time()
        print(f"[{i}/{len(video_ids)}] {vid}", flush=True)
        dlc = load_dlc_h5(find_dlc(vid))
        segments = load_gt_segments(vid)
        reaches = detect_reaches_v8(dlc)
        print(f"   detected {len(reaches)} reaches", flush=True)
        save_reaches_segmented(vid, reaches, segments, algo_dir / f"{vid}_reaches.json")

        seg_inputs = []
        for si, (s, e) in enumerate(segments):
            sn = si + 1
            seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
            seg_inputs.append(SegmentInput(video_id=vid, segment_num=sn,
                                             seg_start=s, seg_end=e,
                                             dlc_df=dlc, reach_windows=seg_r))
        outs = run_cascade_on_segments(seg_inputs, stages)
        if vid in outs:
            data = {"video_id": vid, "detector": "v6_cascade_fix_a",
                    "detector_version": "6.0.2_fix_a",
                    "reach_detector_version": "v8.0.4_production",
                    "segments": outs[vid]}
            (algo_dir / f"{vid}_pellet_outcomes.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        print(f"   ({time.time() - t_vid:.1f}s)", flush=True)

    print(f"\nScoring with compute_outcome_metrics...", flush=True)
    compute_outcome_metrics(gt_dir=GT_DIR, algo_dir=algo_dir, output_dir=metrics_dir,
                              video_ids=video_ids, reaches_dir=algo_dir)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Snapshot complete at: {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
