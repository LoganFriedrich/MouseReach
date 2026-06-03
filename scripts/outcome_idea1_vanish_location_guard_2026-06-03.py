"""
EXPERIMENT -- Idea 1: vanish-LOCATION guard on retrieved commits.

The biggest remaining error cluster (8 segments) is "pellet displaced then
vanished -> cascade wrongly commits retrieved" (GT displaced_sa/_outside).
The 2026-06-03 pre-vanish diagnostic (outcome_prevanish_direction_diag) tested
whether direction-toward-mouse separates these from real retrievals: it does
NOT (real retrievals show ~0 toward-mouse motion -- pellet is lifted faster
than DLC sees). BUT pre-vanish DISTANCE-FROM-PILLAR does separate a subset:
  - real retrievals (n=14): pre-vanish pellet dist_r ALL <= 0.95 radii
    (pellet vanishes from right next to the pillar -- lifted straight off)
  - displaced-vanish group A: 3 of 8 vanish from clearly off-pillar
    (2.48r, 4.76r, 10.4r); the other 5 vanish near-pillar (<=1.65r, DLC-limited)

Guard: before a stage commits `retrieved` on a sustained vanish, check the
pellet's last confident position before the vanish onset. If it is clearly
off-pillar (>= OFF_PILLAR_RADII = 2.0 radii = a full pillar-diameter off
center, geometrically "not on the pillar"; well above every real retrieval's
0.95r), the pellet was displaced out there, not lifted from the pillar ->
commit `displaced_sa` instead (reusing the stage's causal-reach when-frames).

Threshold 2.0r is chosen from APPARATUS GEOMETRY (pillar radius is the natural
scale; 2r = clearly off the pillar), NOT fit to GT -- the dual-corpus gate is
the check that it generalizes and breaks no real retrieval.

================================================================
PRE-EXPERIMENT CHECKLIST (applied in writing, 2026-06-03)
================================================================
1. Cumulative-stacking (verified via git log; master @ 4db0237): cumulative
   best = v6.0.4 (Lever A net-displaced_sa MERGED; Lever B was REJECT-only;
   no reverts of Fix A/B/Lever A). Stacking on build_stages_with_leverA.
   Baselines to beat: model-3.1 389/400, generalization 385/400.

2. Existing module code modified: NO. A wrapper around each retrieved-target
   stage post-inspects its decision and overrides retrieved->displaced_sa when
   the pre-vanish pellet is far off-pillar. Geometry inlined in this runner.
   No edits under src/.

3. Unverified hypotheses:
   - FALSIFIED prior hypothesis (documented): "retrieved pellet moves toward
     the mouse before vanishing" -- real retrievals show dy_toward_mouse ~ 0.
   - H1: guard converts 20251024_CNT0402_P4 s5 + 20250821_CNT0110_P4 s8
     (model-3.1, +2) and 20251008_CNT0303_P2 s6 (generalization, +1) from
     wrong-retrieved to displaced_sa. UNVERIFIED until run.
   - H2 (risk): a pellet displaced far off-pillar and THEN retrieved from there
     would be mislabeled displaced_sa. Rare; the dual-corpus gate checks that
     no real retrieval has a far-off-pillar pre-vanish position. UNVERIFIED.
   - The other 5 group-A cases vanish near-pillar (<=1.65r) and are NOT
     addressable by this guard (DLC-limited).

4. Reporting: dual-corpus, per-class confusion + per-segment diff; lead with
   recovery + any regression on BOTH corpora.

5. Framework: snapshot dir below + canonical compute_outcome_metrics.

6. Branch + tag created BEFORE this runner:
   tag outcome-pre-idea1-vanish-location-2026-06-03 @ 4db0237;
   branch feature/outcome-idea1-vanish-location-guard.

7. Decision rule (dual-corpus): ACCEPT iff model-3.1 total correct rises above
   389 AND generalization total >= 385 (no-regression gate) AND no class recall
   regresses on either corpus (specifically: retrieved recall must not drop --
   the guard must not flip a real retrieval to displaced_sa). Otherwise REJECT.
   ASCII-only; report remaining error; do not call results "good".
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

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series

OFF_PILLAR_RADII = 2.0
LK = 0.7
SNAP_ROOT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
                 r"\Improvement_Snapshots\outcome"
                 r"\v6.0.4_idea1_vanish_location_guard_2026-06-03")


def prevanish_dist_r(seg):
    """Median pellet distance-from-pillar (radii) over the last <=10 confident
    frames before the longest sustained (>=30f) post-first-reach vanish. None
    if no sustained vanish or too few pre-vanish confident frames."""
    s, e = seg.seg_start, seg.seg_end
    sub_raw = seg.dlc_df.iloc[s:e + 1]
    n = len(sub_raw)
    if n == 0:
        return None
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pcx = geom["pillar_cx"].to_numpy(float); pcy = geom["pillar_cy"].to_numpy(float)
    pr = geom["pillar_r"].to_numpy(float)
    px = sub["Pellet_x"].to_numpy(float); py = sub["Pellet_y"].to_numpy(float)
    plk = sub_raw["Pellet_likelihood"].to_numpy(float)
    conf = plk >= LK
    dist_r = np.sqrt((px - pcx) ** 2 + (py - pcy) ** 2) / np.maximum(pr, 1e-6)
    seg_reaches = sorted((r0 - s, r1 - s) for r0, r1 in seg.reach_windows if s <= r0 <= e)
    first_reach = seg_reaches[0][0] if seg_reaches else 0
    low = (~conf).copy(); low[:first_reach] = False
    runs = []; st = -1
    for i in range(n):
        if low[i] and st < 0:
            st = i
        elif not low[i] and st >= 0:
            if i - st >= 30:
                runs.append((st, i - 1))
            st = -1
    if st >= 0 and n - st >= 30:
        runs.append((st, n - 1))
    if not runs:
        return None
    onset = max(runs, key=lambda ab: ab[1] - ab[0])[0]
    pre = [i for i in range(onset) if conf[i]][-10:]
    if len(pre) < 3:
        return None
    return float(np.median(dist_r[pre]))


class VanishLocationGuard(Stage):
    """Wraps a retrieved-target stage; if it commits retrieved on a sustained
    vanish whose pre-vanish pellet was clearly off-pillar, override to
    displaced_sa (reuse the stage's when-frames)."""
    def __init__(self, inner, off_pillar_radii=OFF_PILLAR_RADII):
        self.inner = inner
        self.name = inner.name
        self.target_class = getattr(inner, "target_class", None)
        self.off_radii = off_pillar_radii

    def decide(self, seg):
        d = self.inner.decide(seg)
        if d.decision != "commit" or d.committed_class != "retrieved":
            return d
        dist = prevanish_dist_r(seg)
        if dist is not None and dist >= self.off_radii:
            f = dict(d.features or {}); f["vanish_location_guard_dist_r"] = round(dist, 2)
            return StageDecision(
                decision="commit", committed_class="displaced_sa",
                whens=d.whens,
                reason=(f"vanish_location_guard: pellet vanished from {dist:.1f}r off-pillar "
                        f">= {self.off_radii} -> displaced_sa not retrieved (orig stage "
                        f"{self.inner.name} said retrieved)"),
                features=f)
        return d


def build_stages_with_idea1():
    out = []
    for label, stage in build_stages_with_leverA(video_dir=None):
        if getattr(stage, "target_class", None) == "retrieved":
            out.append((label, VanishLocationGuard(stage)))
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
                json.dumps({"video_id": vid, "detector": "v6_cascade_idea1", "segments": outs[vid]},
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
    return correct, n


def main():
    t0 = time.time()
    print("EXPERIMENT: Idea 1 -- vanish-location guard on retrieved commits (on v6.0.4)")
    print(f"OFF_PILLAR_RADII = {OFF_PILLAR_RADII}")
    stages = build_stages_with_idea1()
    for cfg in (M31, GEN):
        run_corpus(cfg, stages)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Snapshot: {SNAP_ROOT}")


if __name__ == "__main__":
    main()
