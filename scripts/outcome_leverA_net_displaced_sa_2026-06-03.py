"""
EXPERIMENT -- Lever A: net-displacement displaced_sa stage (causal-agnostic).

Adds one new committing stage that catches the displaced_sa "causal-pick
gap" diagnosed 2026-06-03: multi-reach segments where the pellet is clearly
displaced and resting off-pillar in the SA, but no existing displaced_sa
stage can isolate a single causal reach (two+ reaches each nudge the pellet
a similar amount), so the segment falls through to Stage 99 triage.

Mechanism (from the stage-by-stage trace, outcome_leverA_trace_2026-06-03):
  Stage 8 (canonical displaced_sa) does NOT fail on the class signal -- on
  0105_P1 s7 it confirmed "pellet settled off-pillar in SA" and only failed
  the pre-causal-bout on-pillar PRECISION gate. The other 3 fail the
  on-pillar-before-FIRST-reach precondition or causal-bout-before-rest gate.
  All four are causal-ATTRIBUTION failures, not class-signal failures.

New stage = Stage 8's class gates, causal gates relaxed:
  KEEP (class-defining, unchanged thresholds from Stage 8):
    - co-detection defer
    - pellet settles off-pillar (>1.0 radii), >=40 accumulated frames
    - median rest position inside the SA quadrilateral
    - >=40 frames clustered near that median (resting, not bouncing)
    - LATE-segment observability >=250 off-pillar frames in last 25%
      (this is the retrieved discriminator -- retrieved pellets vanish)
    - PELLET_LK 0.95 (filters DLC label-switch)
  REPLACE the three causal-precision gates with ONE causal-agnostic
  net on->off check:
    - require >=PRE_ON_PILLAR_SUSTAINED sustained on-pillar frames at
      ANY point before the rest period begins (confirms a real on->off
      transition happened within this segment -- it was displaced HERE,
      not entering already-displaced), WITHOUT requiring it be right
      before the first reach or a specific causal bout.
  interaction_frame = best guess: walk back from rest-start to the most
  recent paw-past-slit bout (Stage 8's IFR method); if none, midpoint of
  the segment's largest reach. OKF = rest-start + 6 (Stage 8 convention).

Placement: inserted right before Stage 30 (the retrieved-rescue fallbacks).
It requires sustained off-pillar PRESENCE; the retrieved rescues require
sustained VANISH -- mutually exclusive signals, so it cannot steal a real
retrieved. It runs after all existing displaced_sa stages, so it only sees
their fall-throughs.

================================================================
PRE-EXPERIMENT CHECKLIST
================================================================
1. Cumulative-stacking check (verified 2026-06-03)
   - Reach detector: v8.0.4 production.
   - Outcome cascade cumulative best: v6.0.3 (Fix B = v6.0.1 vetoes +
     Fix A Stage 32 + Fix B Stage 31), via build_stages_with_fix_b().
   - This experiment LAYERS one new stage on top of that build. Baselines:
     * model-3.1 corpus: 383/400 (snapshot v6.0.3_eval_model31_corpus_2026-06-03,
       GT corrected 2026-06-03: s16->displaced_sa, s12->retrieved).
     * generalization corpus: 382/400 (snapshot v6.0.3_fix_b_retrieved_rescue_2026-06-02).

2. Existing-code-modification check: NO. New Stage class defined inline;
   reuses stage_8 helpers/constants by import. No edits under src/.

3. Unverified hypotheses
   - H1: new stage commits displaced_sa on >=1 of the 4 Lever-A segments
     (0105_P1 s7, 0215_P4 s11, 0215_P4 s13, 0404_P4 s16). s7 near-certain
     (Stage 8 already confirmed its class signal); others UNVERIFIED until
     run (Stage 8 failed them before reaching the core gates).
   - H2: no regression on generalization corpus (no real retrieved/
     untouched converted to displaced_sa). UNVERIFIED -- the whole point
     of dual-corpus scoring.
   - H3: approximate interaction_frame may miss the trust IFR tolerance on
     some commits; strict per-segment CLASS accuracy is unaffected (it is
     class-based). IFR deltas reported for awareness.

4. Reporting: per-class confusion + strict accuracy, BOTH corpora, deltas
   vs the two baselines above. Lead with class deltas + any regression.

5. Framework: snapshot dir per corpus under Improvement_Snapshots/outcome/.
   Canonical compute_outcome_metrics scorer. No hand-rolled figures.

6. Branch + tag: deferred to ship. The experiment is entirely a new
   untracked runner; nothing in tracked code changes, so there is nothing
   to revert if rejected (delete the script). On ACCEPT, create
   outcome-pre-leverA-net-displaced-sa-<date> tag + feature branch and
   integrate. (Per [[feedback_ask_before_every_push]] no push regardless.)

7. Decision rule (dual-corpus, [[feedback-outcome-experiments-dual-corpus]]):
   - ACCEPT iff: model-3.1 total correct rises (catches Lever-A) AND
     generalization total correct >= 382 (NO regression) AND no class
     recall regresses by >1 on EITHER corpus.
   - REJECT if generalization total correct < 382, OR any class regresses
     materially on either corpus.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_outcome_fix_b", SCRIPTS_DIR / "outcome_fix_b_retrieved_rescue_2026-06-02.py")
fixb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixb)

detect_reaches_v8 = fixb.detect_reaches_v8
load_dlc_h5 = fixb.load_dlc_h5
compute_outcome_metrics = fixb.compute_outcome_metrics
SegmentInput = fixb.SegmentInput
Stage = fixb.Stage
StageDecision = fixb.StageDecision
build_stages_with_fix_b = fixb.build_stages_with_fix_b
run_cascade_on_segments = fixb.run_cascade_on_segments
save_reaches_segmented = fixb.save_reaches_segmented

# Reuse Stage 8's geometry helpers + constants verbatim (no src edits).
from mousereach.outcomes.v6_cascade import stage_8_pellet_displaced_to_sa as s8
from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series


PRE_ON_PILLAR_SUSTAINED = 5   # net on->off: >=N sustained on-pillar frames
                              # somewhere before the rest period begins.


class StageNetDisplacedSAResting(Stage):
    """displaced_sa via net rest-in-SA, causal-pick-agnostic.

    Stage 8's class gates, with the causal-precision gates replaced by a
    single net on->off-transition requirement. Catches multi-reach
    segments where the pellet is clearly displaced and resting in the SA
    but no single causal reach can be isolated.
    """
    name = "stage_33_net_displaced_sa_resting"
    target_class = "displaced_sa"

    def decide(self, seg: SegmentInput) -> StageDecision:
        tz = s8.TRANSITION_ZONE_HALF
        clean_end = seg.seg_end - tz
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue", reason="segment_too_short")
        sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub_raw)
        if n == 0:
            return StageDecision(decision="continue", reason="empty_segment")

        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        geom = compute_pillar_geometry_series(sub)
        pillar_cx = geom["pillar_cx"].to_numpy()
        pillar_cy = geom["pillar_cy"].to_numpy()
        pillar_r = geom["pillar_r"].to_numpy()
        slit_y_line = pillar_cy + pillar_r

        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        pellet_dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2 + (pellet_y - pillar_cy) ** 2)
                             / np.maximum(pillar_r, 1e-6))

        # ----- Co-detection defer (Stage 8 verbatim).
        if s8._detect_codetection_triage(
                pellet_lk, sub_raw["Pillar_likelihood"].to_numpy(dtype=float),
                pellet_x, pellet_y,
                sub_raw["Pillar_x"].to_numpy(dtype=float),
                sub_raw["Pillar_y"].to_numpy(dtype=float),
                pillar_r, s8.CODETECTION_PELLET_LK_THR, s8.CODETECTION_PILLAR_LK_THR,
                s8.CODETECTION_DISTANCE_RADII, s8.CODETECTION_SUSTAINED_FRAMES):
            return StageDecision(decision="continue", reason="codetection_defer",
                                 features={"codetection_observed": True})

        paw_past_y = np.zeros(n, dtype=bool)
        for bp in s8.PAW_BODYPARTS:
            paw_y = sub[f"{bp}_y"].to_numpy(dtype=float)
            paw_lk = sub[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (paw_y <= slit_y_line) & (paw_lk >= s8.PAW_LK_THR)
        bouts = s8._find_paw_past_y_line_bouts(paw_past_y)
        if not bouts:
            return StageDecision(decision="continue", reason="no_paw_past_y_line_bouts")

        feats = {"n_clean_zone_frames": int(n), "n_bouts": int(len(bouts))}

        # ----- CLASS GATE 1: pellet settled off-pillar, >=REST frames.
        off_elig = ((pellet_lk >= s8.PELLET_LK_THR)
                    & (pellet_dist_radii > s8.PELLET_OFF_PILLAR_RADII)
                    & (~paw_past_y))
        off_count = int(off_elig.sum())
        feats["off_pillar_frame_count"] = off_count
        if off_count < s8.REST_FRAMES_TOTAL:
            return StageDecision(decision="continue",
                                 reason=f"off_pillar_count_low ({off_count})", features=feats)

        median_x = float(np.median(pellet_x[off_elig]))
        median_y = float(np.median(pellet_y[off_elig]))

        # ----- CLASS GATE 2: median inside SA quadrilateral.
        sa_top_y = float(np.median((sub["SATL_y"].to_numpy() + sub["SATR_y"].to_numpy())[off_elig] / 2.0))
        sa_bot_y = float(np.median((sub["SABL_y"].to_numpy() + sub["SABR_y"].to_numpy())[off_elig] / 2.0))
        sa_left_x = float(np.median(np.minimum(sub["SABL_x"].to_numpy(), sub["SATL_x"].to_numpy())[off_elig]))
        sa_right_x = float(np.median(np.maximum(sub["SABR_x"].to_numpy(), sub["SATR_x"].to_numpy())[off_elig]))
        if not (sa_top_y <= median_y <= sa_bot_y and sa_left_x <= median_x <= sa_right_x):
            return StageDecision(decision="continue", reason="median_outside_sa_quad", features=feats)

        # ----- CLASS GATE 3: clustered near median (resting).
        dev_radii = np.sqrt((pellet_x - median_x) ** 2 + (pellet_y - median_y) ** 2) / np.maximum(pillar_r, 1e-6)
        near_elig = off_elig & (dev_radii <= s8.NEAR_MEDIAN_TOLERANCE_RADII)
        near_count = int(near_elig.sum())
        feats["near_median_count"] = near_count
        if near_count < s8.REST_FRAMES_TOTAL:
            return StageDecision(decision="continue", reason=f"not_resting ({near_count})", features=feats)

        # ----- CLASS GATE 4: late-segment observability (retrieved filter).
        late_start = int(n * (1 - s8.LATE_SEGMENT_FRACTION))
        late_elig = off_elig.copy(); late_elig[:late_start] = False
        late_count = int(late_elig.sum())
        feats["late_observable_count"] = late_count
        if late_count < s8.LATE_SEGMENT_OBSERVABLE_MIN_FRAMES:
            return StageDecision(decision="continue",
                                 reason=f"late_observability_low ({late_count}) -> retrieved-like",
                                 features=feats)

        # rest start = first eligible frame near the off-pillar median.
        rest_start = s8._find_first_near_median(off_elig, pellet_x, pellet_y,
                                                median_x, median_y, pillar_r,
                                                s8.NEAR_MEDIAN_TOLERANCE_RADII)
        if rest_start < 0:
            return StageDecision(decision="continue", reason="no_rest_start_frame", features=feats)
        feats["rest_start_idx"] = int(rest_start)

        # ----- NET on->off transition (REPLACES the causal-precision gates):
        # require sustained on-pillar BEFORE the rest period starts, at ANY
        # point -- not tied to a specific causal bout.
        on_elig = ((pellet_lk >= s8.PELLET_LK_THR)
                   & (pellet_dist_radii <= s8.ON_PILLAR_RADII)
                   & (~paw_past_y))
        run = 0; net_ok = False
        for i in range(rest_start):
            if on_elig[i]:
                run += 1
                if run >= PRE_ON_PILLAR_SUSTAINED:
                    net_ok = True; break
            else:
                run = 0
        feats["net_on_to_off_satisfied"] = bool(net_ok)
        if not net_ok:
            return StageDecision(
                decision="continue",
                reason=(f"no_sustained_on_pillar_before_rest "
                        f"(need {PRE_ON_PILLAR_SUSTAINED}+ on-pillar frames before rest "
                        f"start {rest_start} -- not a within-segment on->off displacement)"),
                features=feats)

        # ----- IFR best guess: most recent bout ending before rest start;
        # fallback to midpoint of the segment's largest reach.
        causal_bout = -1
        for bidx in range(len(bouts) - 1, -1, -1):
            if bouts[bidx][1] < rest_start:
                causal_bout = bidx; break
        if causal_bout >= 0:
            bs, be = bouts[causal_bout]
            interaction_idx = int(bs + round(s8.IFR_POSITION_IN_BOUT * (be - bs + 1)))
            interaction_idx = max(bs, min(be, interaction_idx))
        else:
            # no bout before rest -> use largest GT/algo reach midpoint.
            r_local = [(max(0, rs - seg.seg_start), min(n - 1, re - seg.seg_start))
                       for rs, re in seg.reach_windows]
            r_local = [(a, b) for a, b in r_local if b >= a]
            if r_local:
                a, b = max(r_local, key=lambda ab: ab[1] - ab[0])
                interaction_idx = (a + b) // 2
            else:
                interaction_idx = rest_start
        okf_idx = min(int(rest_start) + 6, n - 1)

        feats.update({"off_pillar_median_dist_radii": float(np.median(pellet_dist_radii[off_elig])),
                      "causal_bout_idx": int(causal_bout)})
        return StageDecision(
            decision="commit", committed_class="displaced_sa",
            whens={"outcome_known_frame": int(seg.seg_start + okf_idx),
                   "interaction_frame": int(seg.seg_start + interaction_idx)},
            reason=(f"net_displaced_sa_resting (off={off_count}, near={near_count}, "
                    f"late={late_count}, rest_start={rest_start}, causal_bout={causal_bout})"),
            features=feats)


def build_stages_with_leverA(video_dir=None):
    """Fix B build + the new net-displaced_sa stage, inserted right before
    the Stage 30 retrieved-rescue fallbacks."""
    base = build_stages_with_fix_b(video_dir=video_dir)
    out = []
    inserted = False
    for label, stage in base:
        if (not inserted) and label == "stage_30_retrieved_via_sustained_vanish_fallback":
            out.append(("stage_33_net_displaced_sa_resting", StageNetDisplacedSAResting()))
            inserted = True
        out.append((label, stage))
    if not inserted:  # defensive: append before stage_99 if 30 not found
        out2 = []
        for label, stage in out:
            if "stage_99_residual_triage" in label and not inserted:
                out2.append(("stage_33_net_displaced_sa_resting", StageNetDisplacedSAResting()))
                inserted = True
            out2.append((label, stage))
        out = out2
    return out


# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------
M31 = dict(
    name="model31",
    dlc=Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
             r"\DLC_2026_03_27\Processing\updated dlc model 3.1"),
    gt=Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
            r"\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt"),
    ids=["20250624_CNT0107_P3", "20250627_CNT0105_P1", "20250630_CNT0104_P3",
         "20250701_CNT0111_P1", "20250710_CNT0215_P4", "20250812_CNT0301_P3",
         "20250813_CNT0314_P4", "20250820_CNT0103_P3", "20250821_CNT0110_P4",
         "20250909_CNT0209_P4", "20251009_CNT0310_P2", "20251010_CNT0308_P2",
         "20251022_CNT0413_P4", "20251028_CNT0404_P4", "20251030_CNT0403_P1",
         "20251031_CNT0407_P1", "20250626_CNT0102_P4", "20250708_CNT0210_P3",
         "20250811_CNT0303_P4", "20251024_CNT0402_P4"],
    baseline=383,
)
GEN = dict(
    name="generalization",
    dlc=Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
             r"\generalization_test_2026-05-11\algo_outputs_current"),
    gt=Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
            r"\generalization_test_2026-05-11\gt"),
    ids=None,  # auto-discover all 20
    baseline=382,
)

SNAP_ROOT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
                 r"\Improvement_Snapshots\outcome\v6.0.4_leverA_net_displaced_sa_2026-06-03")


def load_gt_segments(gt_dir, vid):
    gt = json.loads((gt_dir / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    bs = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    return [(bs[i], bs[i + 1] - 1) for i in range(len(bs) - 1)] if len(bs) >= 2 else []


def run_corpus(cfg, stages):
    name = cfg["name"]
    algo_dir = SNAP_ROOT / name / "algo_outputs"
    metrics_dir = SNAP_ROOT / name / "metrics"
    algo_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ids = cfg["ids"]
    if ids is None:
        ids = sorted(p.stem.replace("_unified_ground_truth", "")
                     for p in cfg["gt"].glob("*_unified_ground_truth.json"))
    print(f"\n=== corpus {name}: {len(ids)} videos ===", flush=True)
    for i, vid in enumerate(ids, 1):
        h5 = sorted(cfg["dlc"].glob(f"{vid}DLC_*.h5"))
        if not h5:
            print(f"  [skip] {vid} no DLC"); continue
        dlc = load_dlc_h5(h5[0])
        segments = load_gt_segments(cfg["gt"], vid)
        reaches = detect_reaches_v8(dlc)
        save_reaches_segmented(vid, reaches, segments, algo_dir / f"{vid}_reaches.json")
        seg_inputs = []
        for si, (s, e) in enumerate(segments):
            seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
            seg_inputs.append(SegmentInput(video_id=vid, segment_num=si + 1,
                                           seg_start=s, seg_end=e, dlc_df=dlc, reach_windows=seg_r))
        outs = run_cascade_on_segments(seg_inputs, stages)
        if vid in outs:
            (algo_dir / f"{vid}_pellet_outcomes.json").write_text(
                json.dumps({"video_id": vid, "detector": "v6_cascade_leverA",
                            "detector_version": "6.0.4_leverA",
                            "segments": outs[vid]}, indent=2), encoding="utf-8")
    scalars = compute_outcome_metrics(gt_dir=cfg["gt"], algo_dir=algo_dir,
                                      output_dir=metrics_dir, video_ids=ids, reaches_dir=algo_dir)
    ps = scalars["outcome_label_per_segment"]
    n = scalars["n_segments_paired"]
    correct = round(ps["strict_accuracy"] * n)
    print(f"  {name}: {correct}/{n} = {ps['strict_accuracy']:.4f}  (baseline {cfg['baseline']}/{n})")
    print(f"  delta vs baseline: {correct - cfg['baseline']:+d}")
    print(f"  confusion: " + ", ".join(f"{k}={v}" for k, v in
          sorted(ps["confusion_matrix"].items(), key=lambda x: -x[1])))
    return correct, n, ps


def main():
    t0 = time.time()
    print("EXPERIMENT: Lever A net-displaced_sa stage (v6.0.4 candidate)")
    stages = build_stages_with_leverA(video_dir=None)
    print("stage chain (committing + new):")
    for label, _ in stages:
        mark = "  <== NEW" if label == "stage_33_net_displaced_sa_resting" else ""
        if "stage_3" in label or "stage_99" in label or "net_displaced" in label:
            print(f"   {label}{mark}")
    for cfg in (M31, GEN):
        run_corpus(cfg, stages)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Snapshot: {SNAP_ROOT}")


if __name__ == "__main__":
    main()
