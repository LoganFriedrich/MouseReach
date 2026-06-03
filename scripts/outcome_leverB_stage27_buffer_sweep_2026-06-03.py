"""
EXPERIMENT -- Lever B: Stage 27 end-edge-buffer sweep (characterization).

Stage 27 (displaced_sa via unique high-displacement reach) TRIAGES when the
causal reach lands within START_EDGE_BUFFER_FRAMES (30) of segment start or
END_EDGE_BUFFER_FRAMES (60) of segment end, because boundary noise (ASPA
reload / next-pellet) makes near-edge commits unreliable. The 2026-06-03
diagnostic trace showed `20251031_CNT0407_P1 s19` (GT=displaced_sa) deferred
by exactly this end-edge guard (causal reach ends 20f before segment end).

This sweep relaxes ONLY the end buffer over {60, 40, 20, 10} and scores BOTH
corpora at each value, to measure the recovery-vs-regression tradeoff. It is a
CHARACTERIZATION, not a tune-to-ship: a relaxed value is only acceptable if it
recovers model-3.1 cases AND holds the generalization set (the no-regression
gate, per feedback-outcome-experiments-dual-corpus). The generalization gate is
what keeps a small buffer from being eval-overfitting.

================================================================
PRE-EXPERIMENT CHECKLIST (applied in writing, 2026-06-03)
================================================================
1. Cumulative-stacking (verified via git log, master @ 5e9b5c7):
   cumulative best = v6.0.4 (Lever A net-displaced_sa MERGED today;
   Fix A/B intact; no reverts). Stacking on build_stages_with_leverA.
   Baselines to beat: model-3.1 389/400, generalization 385/400.

2. Existing module code modified: NO. Stage 27's edge buffers are
   module-level constants read as globals inside decide(). This runner
   does NOT edit src/: it TRANSIENTLY swaps s27.END/START_EDGE_BUFFER_FRAMES
   for the duration of each Stage 27 decide() call and restores them in a
   finally block. Canonical Stage 27 logic runs unchanged; source files on
   disk are untouched; module state is restored after every call.

3. Unverified hypotheses:
   - H1: END buffer <= 20 recovers 20251031_CNT0407_P1 s19 IF its P4
     post-displacement off-pillar check passes (only ~15-20f of post-reach
     window remain). UNVERIFIED.
   - H2: relaxing the end buffer likely creates NEW wrong commits on the
     generalization corpus where a near-end reach is followed by ASPA-reload
     / next-pellet noise. This is the regression the gate exists to catch.
     UNVERIFIED -- the point of the sweep.
   - 20250630_CNT0104_P3 s17 is NOT recoverable here: Stage 27 declines it
     at P1 (found 0 high-disp reaches), not at the edge buffer. Confirmed by
     trace. The buffer sweep cannot touch it.

4. Reporting: dual-corpus per-buffer table; lead with recovery + any
   regression on BOTH corpora. (FN-direction rule is reach-detection-specific;
   outcome leads with per-class confusion deltas.)

5. Framework: snapshot dir below + canonical compute_outcome_metrics. No
   hand-rolled metrics.

6. Branch + tag created BEFORE this runner:
   tag outcome-pre-leverB-stage27-buffer-2026-06-03 @ 5e9b5c7;
   branch feature/outcome-leverB-stage27-buffer.

7. Decision rule (dual-corpus):
   - ACCEPT iff some END buffer value raises model-3.1 total correct above
     389 AND holds generalization total >= 385 AND no class recall regresses
     on either corpus.
   - REJECT (park as boundary-evidence ceiling) if no value nets positive on
     model-3.1 without dropping generalization below 385.
   - ASCII-only output; do not call any result "good"; report remaining error.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

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
build_stages_with_leverA = lva.build_stages_with_leverA
run_cascade_on_segments = lva.run_cascade_on_segments
save_reaches_segmented = lva.save_reaches_segmented
load_gt_segments = lva.load_gt_segments
M31 = lva.M31
GEN = lva.GEN

from mousereach.outcomes.v6_cascade import (
    stage_27_displaced_sa_via_unique_high_displacement_reach as s27)

BUFFERS = [60, 40, 20, 10]          # END_EDGE_BUFFER values to sweep (60 = baseline)
START_BUFFER = 30                    # held at baseline (isolate the end lever)
SNAP_ROOT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
                 r"\Improvement_Snapshots\outcome"
                 r"\v6.0.4_leverB_stage27_buffer_sweep_2026-06-03")


class Stage27BufferOverride(Stage):
    """Runs the inner Stage 27 (with vanish veto) but transiently swaps the
    module-level edge-buffer constants for the call, restoring after."""
    def __init__(self, inner, end_buffer, start_buffer):
        self.inner = inner
        self.name = inner.name
        self.target_class = getattr(inner, "target_class", None)
        self.end_buffer = end_buffer
        self.start_buffer = start_buffer

    def decide(self, seg):
        oe, ostart = s27.END_EDGE_BUFFER_FRAMES, s27.START_EDGE_BUFFER_FRAMES
        s27.END_EDGE_BUFFER_FRAMES = self.end_buffer
        s27.START_EDGE_BUFFER_FRAMES = self.start_buffer
        try:
            return self.inner.decide(seg)
        finally:
            s27.END_EDGE_BUFFER_FRAMES = oe
            s27.START_EDGE_BUFFER_FRAMES = ostart


def build_stages_with_leverB(end_buffer, start_buffer=START_BUFFER):
    out = []
    for label, stage in build_stages_with_leverA(video_dir=None):
        if label == "stage_27_with_vanish_veto":
            out.append((label, Stage27BufferOverride(stage, end_buffer, start_buffer)))
        else:
            out.append((label, stage))
    return out


def run_corpus(cfg):
    name = cfg["name"]
    ids = cfg["ids"] or sorted(
        p.stem.replace("_unified_ground_truth", "")
        for p in cfg["gt"].glob("*_unified_ground_truth.json"))
    reaches_dir = SNAP_ROOT / name / "reaches"
    reaches_dir.mkdir(parents=True, exist_ok=True)

    # Cache DLC + reaches + seg_inputs once per video (buffer-independent).
    cache = {}
    print(f"\n=== corpus {name}: {len(ids)} videos (caching reaches) ===", flush=True)
    for vid in ids:
        h5 = sorted(cfg["dlc"].glob(f"{vid}DLC_*.h5"))
        if not h5:
            print(f"  [skip] {vid} no DLC"); continue
        dlc = load_dlc_h5(h5[0])
        segments = load_gt_segments(cfg["gt"], vid)
        reaches = detect_reaches_v8(dlc)
        save_reaches_segmented(vid, reaches, segments, reaches_dir / f"{vid}_reaches.json")
        seg_inputs = []
        for si, (s, e) in enumerate(segments):
            seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
            seg_inputs.append(SegmentInput(video_id=vid, segment_num=si + 1,
                                           seg_start=s, seg_end=e, dlc_df=dlc, reach_windows=seg_r))
        cache[vid] = seg_inputs

    results = {}
    for buf in BUFFERS:
        stages = build_stages_with_leverB(buf)
        algo_dir = SNAP_ROOT / name / f"end_{buf}" / "algo_outputs"
        metrics_dir = SNAP_ROOT / name / f"end_{buf}" / "metrics"
        algo_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for vid, seg_inputs in cache.items():
            outs = run_cascade_on_segments(seg_inputs, stages)
            if vid in outs:
                (algo_dir / f"{vid}_pellet_outcomes.json").write_text(
                    json.dumps({"video_id": vid, "detector": "v6_cascade_leverB",
                                "end_edge_buffer": buf, "segments": outs[vid]}, indent=2),
                    encoding="utf-8")
        scalars = compute_outcome_metrics(gt_dir=cfg["gt"], algo_dir=algo_dir,
                                          output_dir=metrics_dir, video_ids=list(cache.keys()),
                                          reaches_dir=reaches_dir)
        ps = scalars["outcome_label_per_segment"]
        n = scalars["n_segments_paired"]
        correct = round(ps["strict_accuracy"] * n)
        results[buf] = (correct, n, ps["confusion_matrix"])
    return results


def main():
    t0 = time.time()
    print("EXPERIMENT: Lever B -- Stage 27 end-edge-buffer sweep (on v6.0.4)")
    print(f"BUFFERS (END_EDGE) = {BUFFERS}; START_EDGE held at {START_BUFFER}")
    all_res = {}
    for cfg in (M31, GEN):
        all_res[cfg["name"]] = (run_corpus(cfg), cfg["baseline"])
    print("\n" + "=" * 64)
    print("SWEEP SUMMARY (per-segment strict accuracy, class-based)")
    print("=" * 64)
    for name, (res, base) in all_res.items():
        print(f"\n{name}  (v6.0.4 baseline = {base})")
        print(f"  {'END_buf':>8} {'correct':>8} {'delta_vs_v604':>14}")
        for buf in BUFFERS:
            c, n, _ = res[buf]
            print(f"  {buf:>8} {c:>8}/{n} {c - base:>+12}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Snapshot: {SNAP_ROOT}")


if __name__ == "__main__":
    main()
