"""
EXPERIMENT -- Idea 3: lower the displaced_sa pellet-lk gate + pillar-reveal veto.

`20250626_CNT0102_P4 s16` (GT displaced_sa) has the pellet tracked off-pillar in
the SA for 421 frames, but at likelihood 0.7-0.95. Stage 8 and stage_33 require
lk >= 0.95 (raised originally to filter DLC label-switch), so they count only 3
off-pillar frames and decline -> triage.

Change (applied to stage_33, the Lever A net-displaced_sa stage, only):
  (a) lower its pellet-lk gate 0.95 -> 0.85 to admit moderate-confidence
      displaced pellets (transient swap of s8.PELLET_LK_THR during stage_33's
      decide; restored after; Stage 8 unaffected -- it captured 0.95 at build).
  (b) add a PILLAR-REVEAL veto: only commit displaced_sa if the pillar became
      revealed (mean Pillar lk after the last reach >= 0.5), i.e. the pellet
      actually left the pillar. This REPLACES the label-switch protection the
      0.95 gate provided: a label-switch / phantom-in-SA while the real pellet
      stays ON the pillar leaves the pillar occluded (post lk low) -> vetoed.
      This also fixes the Lever A overfire `20250625_CNT0102_P4 s12` (GT
      untouched, phantom in SA, pillar post lk 0.10 -> vetoed).

Thresholds from physics/geometry, NOT GT-fit: 0.85 = moderate confidence
(below the 0.95 label-switch filter, above the ~0.7 noise floor); pillar-reveal
0.5 = pillar more-likely-visible-than-not after the pellet leaves. Dual-corpus
gate is the check.

================================================================
PRE-EXPERIMENT CHECKLIST (applied in writing, 2026-06-03)
================================================================
1. Cumulative-stacking (git log; master @ 68df918): cumulative best = v6.0.4
   (Lever A merged; Lever B + Idea 1 REJECT-only; no reverts). Stacking on
   build_stages_with_leverA. Baselines: model-3.1 389/400, generalization 385/400.

2. Existing module code modified: NO. Wraps stage_33 (defined in the Lever A
   runner, my code); transient s8.PELLET_LK_THR swap restored in finally +
   pillar-reveal post-check. No edits under src/. (Stage 8 stays at 0.95.)

3. Unverified hypotheses:
   - H1: stage_33 at lk=0.85 catches `20250626_CNT0102_P4 s16` (+1 model-3.1);
     its pillar reveals (0.28 -> 0.99) so it passes the veto. UNVERIFIED.
   - H2: pillar-reveal veto blocks `20250625_CNT0102_P4 s12` (untouched phantom,
     pillar post 0.10) -> removes that Lever A confident-wrong (raw-neutral).
   - H3 (risk): lowering lk to 0.85 re-admits DLC label-switch on OTHER segments
     -> may convert correct untouched/retrieved into displaced_sa. The
     pillar-reveal veto is meant to contain this; dual-corpus gate is the check.
   - H4 (risk): pillar-reveal veto may block a REAL displaced catch whose pillar
     reveal is weak -> could cost an existing stage_33 catch. UNVERIFIED.

4. Reporting: dual-corpus, per-class confusion + per-segment diff; lead with
   recovery + any regression on BOTH corpora.

5. Framework: snapshot dir below + canonical compute_outcome_metrics.

6. Branch + tag created BEFORE this runner:
   tag outcome-pre-idea3-lowlk-pillarreveal-2026-06-03 @ 68df918;
   branch feature/outcome-idea3-lowlk-displaced.

7. Decision rule (dual-corpus): ACCEPT iff model-3.1 total correct rises above
   389 AND generalization total >= 385 (no-regression gate) AND no class recall
   regresses on either corpus. Otherwise REJECT. ASCII-only; report remaining
   error; do not call results "good".
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_leverA", SCRIPTS_DIR / "outcome_leverA_net_displaced_sa_2026-06-03.py")
lva = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lva)

detect_reaches_v8 = lva.detect_reaches_v8
load_dlc_h5 = lva.load_dlc_h5
compute_outcome_metrics = lva.compute_outcome_metrics
SegmentInput = lva.SegmentInput
Stage = lva.Stage
StageDecision = lva.StageDecision
build_stages_with_leverA = lva.build_stages_with_leverA
run_cascade_on_segments = lva.run_cascade_on_segments
save_reaches_segmented = lva.save_reaches_segmented
load_gt_segments = lva.load_gt_segments
M31 = lva.M31
GEN = lva.GEN

from mousereach.outcomes.v6_cascade import stage_8_pellet_displaced_to_sa as s8
from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts

LOW_LK = 0.85
PILLAR_REVEAL_MIN = 0.5
SNAP_ROOT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
                 r"\Improvement_Snapshots\outcome"
                 r"\v6.0.4_idea3_lowlk_pillarreveal_2026-06-03")


def pillar_revealed(seg):
    """True if the pillar is revealed after the last reach (mean Pillar lk
    post-last-reach >= PILLAR_REVEAL_MIN) -- i.e. the pellet actually left."""
    s, e = seg.seg_start, seg.seg_end
    rl = sorted((r0, r1) for r0, r1 in seg.reach_windows if s <= r0 <= e)
    if not rl:
        return False
    last_re = rl[-1][1]
    post = seg.dlc_df["Pillar_likelihood"].iloc[last_re + 1:e + 1].to_numpy(float)
    if len(post) == 0:
        return False
    return float(np.mean(post)) >= PILLAR_REVEAL_MIN


class Stage33LowLkPillarReveal(Stage):
    """stage_33 with a lowered pellet-lk gate + a pillar-reveal veto."""
    def __init__(self, inner, low_lk=LOW_LK):
        self.inner = inner
        self.name = inner.name
        self.target_class = "displaced_sa"
        self.low_lk = low_lk

    def decide(self, seg):
        old = s8.PELLET_LK_THR
        s8.PELLET_LK_THR = self.low_lk
        try:
            d = self.inner.decide(seg)
        finally:
            s8.PELLET_LK_THR = old
        if d.decision == "commit" and d.committed_class == "displaced_sa":
            if not pillar_revealed(seg):
                f = dict(d.features or {}); f["pillar_reveal_veto"] = True
                return StageDecision(decision="continue",
                                     reason="pillar_reveal_veto (pellet did not leave the pillar -> phantom/label-switch, defer)",
                                     features=f)
        return d


def build_stages_with_idea3():
    out = []
    for label, stage in build_stages_with_leverA(video_dir=None):
        if label == "stage_33_net_displaced_sa_resting":
            out.append((label, Stage33LowLkPillarReveal(stage)))
        else:
            out.append((label, stage))
    return out


def run_corpus(cfg, stages):
    name = cfg["name"]
    ids = cfg["ids"] or sorted(p.stem.replace("_unified_ground_truth", "")
                               for p in cfg["gt"].glob("*_unified_ground_truth.json"))
    algo_dir = SNAP_ROOT / name / "algo_outputs"
    metrics_dir = SNAP_ROOT / name / "metrics"
    algo_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== corpus {name}: {len(ids)} videos ===", flush=True)
    for vid in ids:
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
                json.dumps({"video_id": vid, "detector": "v6_cascade_idea3", "segments": outs[vid]},
                           indent=2), encoding="utf-8")
    scalars = compute_outcome_metrics(gt_dir=cfg["gt"], algo_dir=algo_dir, output_dir=metrics_dir,
                                      video_ids=ids, reaches_dir=algo_dir)
    ps = scalars["outcome_label_per_segment"]
    n = scalars["n_segments_paired"]
    correct = round(ps["strict_accuracy"] * n)
    base = {"model31": 389, "generalization": 385}[name]
    print(f"  {name}: {correct}/{n}  (v6.0.4 baseline {base}, delta {correct-base:+d})")
    print(f"  confusion: " + ", ".join(f"{k}={v}" for k, v in
          sorted(ps["confusion_matrix"].items(), key=lambda x: -x[1])))
    return correct


def main():
    t0 = time.time()
    print("EXPERIMENT: Idea 3 -- stage_33 low-lk (0.85) + pillar-reveal veto (on v6.0.4)")
    stages = build_stages_with_idea3()
    for cfg in (M31, GEN):
        run_corpus(cfg, stages)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Snapshot: {SNAP_ROOT}")


if __name__ == "__main__":
    main()
