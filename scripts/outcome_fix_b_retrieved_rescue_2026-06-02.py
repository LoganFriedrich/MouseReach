"""
Outcome cascade Fix B: Stage 31 retrieved rescue (allows competing displacement).

Layered on top of accepted v6.0.2 (= v6.0.1 + Fix A).

Stage 31 mirrors Stage 30's trigger pattern but drops the
"no competing displacement" gate. Targets cases where the segment shows
clear retrieved DLC signal (sustained post-last-reach vanish + pillar
revealed) AND ALSO has displacement evidence that blocked Stage 30.

Background: v6.0.1's Stage 27 vanish veto correctly DEFERRED on two
known cases (CNT0316_P3 s13 with 355f vanish; CNT0414_P4 s17 with 100f
vanish), but no downstream stage committed retrieved -- Stage 30 was
blocked by the competing displacement signal. Both segments fell
through to Stage 99 residual triage.

================================================================
Pre-experiment checklist
================================================================

1. Cumulative-stacking check (verified 2026-06-02)
   - Reach detector: v8.0.4 production
   - Outcome cascade cumulative best: v6.0.2 (= v6.0.1 + Fix A),
     merged to master 2026-06-02 (merge commit 06a619c).
   - This experiment LAYERS Fix B on top.
   - Comparison baseline: v6.0.2 snapshot. 380/400 correct (+1 vs v6.0.1
     from Stage 32 zero-reach untouched).
   - Master at 06a619c as of this run.

2. Existing-code-modification check: NO. Runner imports existing v6
   stages + re-defines the v6.0.1 + Fix A subclasses inline. No edits
   to src/mousereach/.

3. Unverified hypotheses
   - H1: Stage 31 fires on CNT0316_P3 s13 (vanish=355f, disp=27px)
     -> commits retrieved -> +1 retrieved correct (recovered from triage).
   - H2: Stage 31 fires on CNT0414_P4 s17 (vanish=100f, disp moderate)
     -> commits retrieved -> +1 retrieved correct.
   - H3: Stage 31 may also fire on CNT0402_P4 s17 (vanish=132f, disp=25px)
     -> commits retrieved -> +1 retrieved correct.
   - H4: Stage 31 may CREATE wrong retrieved commits on displaced_outside
     cases (DLC signal is identical to retrieved). Estimated risk: 0-2 events.
   - H5: Stage 31 will NOT fire on untouched or displaced_sa segments
     without sustained vanish (gate requires >=100f vanish post-last-reach).

4. FN-direction-reporting (planned RESULTS.md)
   Lead-line: "Retrieved correct went from <baseline> to <new>: <delta>.
   Triage count: <delta>. Total correct: <delta>."
   Two deltas: vs cumulative best (v6.0.2, 380/400) and vs pure baseline
   (v6.0.0 + v7.2.0 reaches, 240/400).

5. Framework-not-adhoc
   Snapshot dir: Improvement_Snapshots/outcome/v6.0.3_fix_b_retrieved_rescue_2026-06-02/

6. Branch + tag
   - Pre-experiment tag: outcome-pre-fix-b-retrieved-rescue-2026-06-02
   - Feature branch: feature/outcome-fix-b-retrieved-rescue

7. Decision rule
   - ACCEPT: retrieved correct rises by >= 2 AND no class regresses by > 1.
     (Allowing slightly larger regression tolerance than Fix A because
      Fix B is by design more aggressive on retrieved commits.)
   - REJECT: retrieved correct does not rise, OR any class regresses by > 1.
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
from mousereach.improvement.outcome.metrics import compute_outcome_metrics


# ---------------------------------------------------------------------------
# Helpers + v6.0.1 + Fix A stages (replicated inline for reproducibility)
# ---------------------------------------------------------------------------

VANISH_VETO_FRAMES = 60


def _max_consecutive_low_lk(dlc_df, start, end, thr=0.5):
    if end <= start: return 0
    lk = dlc_df["Pellet_likelihood"].iloc[start:end].to_numpy(dtype=float)
    below = lk < thr
    max_run = 0; cur = 0
    for b in below:
        if b: cur += 1; max_run = max(max_run, cur)
        else: cur = 0
    return max_run


class Stage16WithVanishVeto(Stage16DisplacedViaMaxDisplacement):
    name = "stage_16_with_vanish_veto"
    def decide(self, seg):
        d = super().decide(seg)
        if d.decision != "commit": return d
        ci = d.features.get("causal_idx")
        if ci is None: return d
        r = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        if ci >= len(r): return d
        cre = r[ci][1]
        mr = _max_consecutive_low_lk(seg.dlc_df, cre + 1, seg.seg_end + 1)
        f = dict(d.features); f["post_causal_max_vanish_run"] = int(mr)
        if mr >= VANISH_VETO_FRAMES:
            f["vanish_veto_fired"] = True
            return StageDecision(decision="continue",
                                  reason=f"global_post_causal_vanish_veto ({mr}f)",
                                  features=f)
        return StageDecision(decision=d.decision, committed_class=d.committed_class,
                              whens=d.whens, reason=d.reason, features=f)


class Stage27WithVanishVeto(Stage27DisplacedSaViaUniqueHighDisplacement):
    name = "stage_27_with_vanish_veto"
    def decide(self, seg):
        d = super().decide(seg)
        if d.decision != "commit": return d
        ci = d.features.get("causal_reach_idx")
        if ci is None: return d
        r = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        if ci >= len(r): return d
        cre = r[ci][1]
        mr = _max_consecutive_low_lk(seg.dlc_df, cre + 1, seg.seg_end + 1)
        f = dict(d.features); f["post_causal_max_vanish_run"] = int(mr)
        if mr >= VANISH_VETO_FRAMES:
            f["vanish_veto_fired"] = True
            return StageDecision(decision="continue",
                                  reason=f"global_post_causal_vanish_veto ({mr}f)",
                                  features=f)
        return StageDecision(decision=d.decision, committed_class=d.committed_class,
                              whens=d.whens, reason=d.reason, features=f)


# Stage 30 (v6.0.1): retrieved fallback, requires no competing displacement
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
        pre_p = (seg.dlc_df["Pillar_likelihood"].iloc[seg.seg_start:first_rs].to_numpy(dtype=float)
                  if first_rs > seg.seg_start else np.array([], dtype=float))
        post_p = seg.dlc_df["Pillar_likelihood"].iloc[last_re + 1:seg.seg_end + 1].to_numpy(dtype=float)
        if not len(pre_p) or not len(post_p):
            return StageDecision(decision="continue", reason="empty_pillar_windows", features=feats)
        pm = float(np.mean(pre_p)); qm = float(np.mean(post_p))
        feats["pillar_lk_pre_mean"] = round(pm, 3); feats["pillar_lk_post_mean"] = round(qm, 3)
        if pm >= 0.5: return StageDecision(decision="continue", reason="pillar_pre_high", features=feats)
        if qm <= 0.85: return StageDecision(decision="continue", reason="pillar_post_low", features=feats)
        # Competing displacement gate (this is what makes Stage 30 conservative)
        max_disp = 0.0
        for rs, re in r:
            pws = max(seg.seg_start, rs - 30); pwe = min(seg.seg_end + 1, re + 1 + 30)
            pre_lk = seg.dlc_df["Pellet_likelihood"].iloc[pws:rs].to_numpy(dtype=float)
            post_lk = seg.dlc_df["Pellet_likelihood"].iloc[re + 1:pwe].to_numpy(dtype=float)
            pc = pre_lk >= 0.7; qc = post_lk >= 0.7
            if not pc.any() or not qc.any(): continue
            px = seg.dlc_df["Pellet_x"].iloc[pws:rs].to_numpy(dtype=float)
            py = seg.dlc_df["Pellet_y"].iloc[pws:rs].to_numpy(dtype=float)
            qx = seg.dlc_df["Pellet_x"].iloc[re + 1:pwe].to_numpy(dtype=float)
            qy = seg.dlc_df["Pellet_y"].iloc[re + 1:pwe].to_numpy(dtype=float)
            d = float(np.sqrt((float(np.median(qx[qc])) - float(np.median(px[pc]))) ** 2 +
                                (float(np.median(qy[qc])) - float(np.median(py[pc]))) ** 2))
            if d > max_disp: max_disp = d
        feats["max_competing_disp_px"] = round(max_disp, 2)
        if max_disp >= 12.0:
            return StageDecision(decision="continue", reason=f"competing_disp ({max_disp:.1f}px)", features=feats)
        return StageDecision(decision="commit", committed_class="retrieved",
                              whens={"interaction_frame": int(last_rs + (last_re - last_rs + 1) // 2),
                                     "outcome_known_frame": int(last_re + 5)},
                              reason=f"sustained_vanish_fallback ({mr}f)", features=feats)


# Stage 32 (Fix A): zero-reach untouched fallback
class Stage32ZeroReachUntouchedFallback(Stage):
    name = "stage_32_zero_reach_untouched_fallback"
    target_class = "untouched"
    def decide(self, seg):
        TZ = 5
        feats = {"n_reaches": len(seg.reach_windows)}
        if seg.reach_windows: return StageDecision(decision="continue", reason="reaches_present", features=feats)
        clean_end = seg.seg_end - TZ
        if clean_end <= seg.seg_start: return StageDecision(decision="continue", reason="too_short", features=feats)
        sub = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]; n = len(sub)
        if n < 60: return StageDecision(decision="continue", reason="too_short_stability", features=feats)
        plk = sub["Pellet_likelihood"].to_numpy(dtype=float)
        vf = float((plk >= 0.7).mean()); feats["pellet_vis_frac"] = round(vf, 3)
        if vf < 0.8: return StageDecision(decision="continue", reason="pellet_obscured", features=feats)
        px = sub["Pellet_x"].to_numpy(dtype=float); py = sub["Pellet_y"].to_numpy(dtype=float)
        early = (np.arange(n) < 30) & (plk >= 0.7)
        late = (np.arange(n) >= n - 30) & (plk >= 0.7)
        if early.sum() < 5 or late.sum() < 5:
            return StageDecision(decision="continue", reason="early_late_too_thin", features=feats)
        drift = float(np.sqrt((float(np.median(px[late])) - float(np.median(px[early]))) ** 2 +
                                 (float(np.median(py[late])) - float(np.median(py[early]))) ** 2))
        feats["pellet_pos_drift_px"] = round(drift, 2)
        if drift > 10:
            return StageDecision(decision="continue", reason=f"drift ({drift:.1f}px)", features=feats)
        return StageDecision(decision="commit", committed_class="untouched",
                              whens={"outcome_known_frame": int(clean_end), "interaction_frame": None},
                              reason=f"zero_reaches + stable_pellet (vis={vf:.2f}, drift={drift:.1f}px)",
                              features=feats)


# ---------------------------------------------------------------------------
# Fix B: Stage 31 -- retrieved rescue with displacement-allowed
# ---------------------------------------------------------------------------

STAGE31_MIN_VANISH = 100
STAGE31_PILLAR_PRE_MAX = 0.5
STAGE31_PILLAR_POST_MIN = 0.85
# No competing-displacement gate -- the difference from Stage 30.


class Stage31RetrievedRescueWithDisplacement(Stage):
    """Like Stage 30 but allows competing displacement signal. Catches
    retrieved cases where post-causal vanish is strong but pellet also
    moved before being retrieved (or where DLC ambiguity creates a fake
    displacement signal in a real retrieval segment).
    """
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
            return StageDecision(decision="continue",
                                  reason=f"window_too_short ({scan_hi - scan_lo})",
                                  features=feats)
        mr = _max_consecutive_low_lk(seg.dlc_df, scan_lo, scan_hi, thr=0.5)
        feats["max_vanish_run"] = int(mr)
        if mr < STAGE31_MIN_VANISH:
            return StageDecision(decision="continue",
                                  reason=f"insufficient_vanish ({mr} < {STAGE31_MIN_VANISH})",
                                  features=feats)
        first_rs = r[0][0]
        pre_p = (seg.dlc_df["Pillar_likelihood"].iloc[seg.seg_start:first_rs].to_numpy(dtype=float)
                  if first_rs > seg.seg_start else np.array([], dtype=float))
        post_p = seg.dlc_df["Pillar_likelihood"].iloc[last_re + 1:seg.seg_end + 1].to_numpy(dtype=float)
        if not len(pre_p) or not len(post_p):
            return StageDecision(decision="continue",
                                  reason="empty_pillar_windows", features=feats)
        pm = float(np.mean(pre_p)); qm = float(np.mean(post_p))
        feats["pillar_lk_pre_mean"] = round(pm, 3)
        feats["pillar_lk_post_mean"] = round(qm, 3)
        if pm >= STAGE31_PILLAR_PRE_MAX:
            return StageDecision(decision="continue",
                                  reason=f"pillar_pre_high ({pm:.2f} >= {STAGE31_PILLAR_PRE_MAX})",
                                  features=feats)
        if qm <= STAGE31_PILLAR_POST_MIN:
            return StageDecision(decision="continue",
                                  reason=f"pillar_post_low ({qm:.2f} <= {STAGE31_PILLAR_POST_MIN})",
                                  features=feats)
        # NO competing-displacement gate -- this is the difference from Stage 30.
        bout_length = last_re - last_rs + 1
        return StageDecision(
            decision="commit",
            committed_class="retrieved",
            whens={"interaction_frame": int(last_rs + bout_length // 2),
                   "outcome_known_frame": int(last_re + 5)},
            reason=(f"retrieved_rescue_with_displacement_allowed "
                    f"(vanish={mr}f, pillar {pm:.2f}->{qm:.2f})"),
            features=feats,
        )


# ---------------------------------------------------------------------------
# Cascade build
# ---------------------------------------------------------------------------

def build_stages_with_fix_b(video_dir=None):
    base = _build_production_stages(video_dir=video_dir)
    modified = []
    for label, stage in base:
        if isinstance(stage, Stage16DisplacedViaMaxDisplacement):
            modified.append(("stage_16_with_vanish_veto", Stage16WithVanishVeto()))
        elif isinstance(stage, Stage27DisplacedSaViaUniqueHighDisplacement):
            modified.append(("stage_27_with_vanish_veto", Stage27WithVanishVeto()))
        elif "stage_99_residual_triage" in label:
            # Order: Stage 30 (conservative retrieved) -> Stage 31 (aggressive retrieved)
            # -> Stage 32 (zero-reach untouched) -> Stage 99 (residual)
            modified.append(("stage_30_retrieved_via_sustained_vanish_fallback",
                              Stage30FallbackRetrievedViaSustainedVanish()))
            modified.append(("stage_31_retrieved_rescue_with_displacement",
                              Stage31RetrievedRescueWithDisplacement()))
            modified.append(("stage_32_zero_reach_untouched_fallback",
                              Stage32ZeroReachUntouchedFallback()))
            modified.append((label, stage))
        else:
            modified.append((label, stage))
    return modified


def run_cascade_on_segments(seg_inputs, stages):
    out = {}
    for si in seg_inputs:
        decision = None; committing = "residual (auto-triage)"
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
        out.setdefault(si.video_id, []).append(rec)
    return out


# ---------------------------------------------------------------------------
# Paths + main
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11")
DLC_DIR = CORPUS_DIR / "algo_outputs_current"
GT_DIR = CORPUS_DIR / "gt"
SNAPSHOT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome\v6.0.3_fix_b_retrieved_rescue_2026-06-02")


def find_dlc(vid):
    m = sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))
    if not m: raise FileNotFoundError(f"No DLC for {vid}")
    return m[0]


def load_gt_segments(vid):
    gt = json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    bs = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    if len(bs) < 2: return []
    return [(bs[i], bs[i + 1] - 1) for i in range(len(bs) - 1)]


def save_reaches_segmented(vid, reaches, segments, out_path):
    segs_data = []; gid = 0
    for si, (s, e) in enumerate(segments):
        sn = si + 1; rs_list = []; rn = 0
        for r0, r1 in reaches:
            if s <= r0 <= e:
                rn += 1; gid += 1
                rs_list.append({"reach_id": gid, "reach_num": rn,
                                 "start_frame": int(r0), "end_frame": int(r1),
                                 "duration_frames": int(r1 - r0 + 1)})
        segs_data.append({"segment_num": sn, "start_frame": int(s), "end_frame": int(e),
                          "n_reaches": len(rs_list), "reaches": rs_list})
    out_path.write_text(json.dumps({"video_id": vid, "detector": "v8.0.4_production",
                                       "n_segments": len(segments), "segments": segs_data},
                                      indent=2), encoding="utf-8")


def main():
    t0 = time.time()
    print(f"Fix B experiment: Stage 31 retrieved rescue (displacement-allowed)")
    print(f"Snapshot dir: {SNAPSHOT_DIR}")
    algo_dir = SNAPSHOT_DIR / "algo_outputs"
    metrics_dir = SNAPSHOT_DIR / "metrics"
    algo_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    video_ids = sorted({p.stem.replace("_unified_ground_truth", "")
                         for p in GT_DIR.glob("*_unified_ground_truth.json")})
    print(f"Found {len(video_ids)} videos")
    stages = build_stages_with_fix_b(video_dir=None)
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
            seg_inputs.append(SegmentInput(video_id=vid, segment_num=sn,
                                             seg_start=s, seg_end=e,
                                             dlc_df=dlc, reach_windows=seg_r))
        outs = run_cascade_on_segments(seg_inputs, stages)
        if vid in outs:
            (algo_dir / f"{vid}_pellet_outcomes.json").write_text(
                json.dumps({"video_id": vid, "detector": "v6_cascade_fix_b",
                            "detector_version": "6.0.3_fix_b",
                            "reach_detector_version": "v8.0.4_production",
                            "segments": outs[vid]}, indent=2), encoding="utf-8")
        print(f"   ({time.time() - t_vid:.1f}s)", flush=True)
    print(f"\nScoring with compute_outcome_metrics...", flush=True)
    compute_outcome_metrics(gt_dir=GT_DIR, algo_dir=algo_dir, output_dir=metrics_dir,
                              video_ids=video_ids, reaches_dir=algo_dir)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Snapshot complete at: {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
