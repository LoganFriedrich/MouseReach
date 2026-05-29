"""v8.0.x experiment: GBM learning_rate sweep (Phase 3 of hyperparam retune).

Anchored at Phase 2 winner (max_iter=400, max_depth=8). Sweep
learning_rate in {0.05, 0.03, 0.07}. The lr=0.05 cell IS the Phase 2
winner -- serves as within-Phase-3 reference.

Phase 2 winner result (max_iter=400, max_depth=8, lr=0.05):
  - Cal: TP=2327, FP=55, FN=76 (+2/+1/-2 vs Phase 1 winner)
  - Hol: TP=3673, FP=57, FN=77 (+3/-3/-3 vs Phase 1 winner)
  - Cardinal Rule preserved both axes both corpora
  - Topology: cal MERGED -1, FRAG 0; hol MERGED +1, FRAG -2 (small mixed)

This is the conditional Phase 3 -- triggered because Phase 2 found a
winner (depth=8) that beat Phase 1 winner (depth=6).

Pre-experiment checklist (same as Phase 1/2, just different sweep dim):

1. Cumulative-stacking: same as Phase 2 (BSW b=1/w=0.8, mg=0, full
   v8.0.4 postprocess, asymmetric matcher, matcher-aware topology,
   model 3.1, NEW corpus). Comparison baselines: Phase 2 winner cell
   for within-sweep; v8.0.4 production for overall.

2. Existing-code-modification: NO. Reuses Phase 1's train_gbm.

3. Unverified hypotheses:
   - Lower lr=0.03 with iter=400 might generalize better (smaller per-
     tree contribution, smoother ensemble)
   - Higher lr=0.07 might cause overfit at iter=400
   - lr=0.05 at iter=400+depth=8 is already at/near the optimum

4. FN-direction-reporting: TWO deltas per cell (vs v8.0.4 production
   AND vs Phase 2 winner cell). Topology paired with legacy.

5. Framework: outputs to
   Improvement_Snapshots/reach_detection/v8.0.4_dev_gbm_learning_rate_sweep/

6. Branch + tag: continuing on feature/v8-gbm-hyperparam.

7. Decision rule (per cell, applied to Phase 2 winner = lr=0.05):
   ACCEPT if (vs Phase 2 winner):
     - Cardinal Rule both axes both corpora
     - FN strictly decreases on at least one corpus, non-increasing on other
     - TP non-decreasing both corpora
     - MERGED, FRAGMENTED non-increasing both corpora
   For OVERALL ship decision: same criteria vs v8.0.4 production.
   Phase 3 result = Phase 4 STOP point per Logan's instructions.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from restart_phase_b_gbm_max_iter_sweep import (
    CORPUS_DIR, HOLDOUT_DLC_DIR, GEN_GT_DIR, CAL_GT_DIR, DLC_SUFFIX,
    RANDOM_STATE,
    BOUNDARY_BUFFER, BOUNDARY_WEIGHT,
    THRESHOLD, MERGE_GAP, MIN_SPAN,
    TRIM_LEADING_THRESHOLD, TRIM_LEADING_SUSTAIN_N,
    TRIM_TRAILING_THRESHOLD, TRIM_TRAILING_SUSTAIN_N,
    APEX_PROMINENCE, APEX_DEPTH_MIN, APEX_PEAK2_REL_MAX, APEX_MIN_DISTANCE,
    SPAN_TOL_FRAC, SPAN_TOL_MIN,
    PARQUET_LK_COLS,
    compute_boundary_weights, train_gbm, proba_to_algos,
    overlap_exists, greedy_match, classify_matcher_aware,
    load_live_gt, cal_per_video_aux,
)
from mousereach.reach.v8.features import (
    feature_columns, extract_features, load_dlc_h5,
)
from mousereach.reach.v8.postprocess import (
    compute_paw_mean_lk, compute_hand_to_boxl_norm_pos,
)

# Phase 2 winner anchors
PHASE2_WINNER_MAX_ITER = 400
PHASE2_WINNER_MAX_DEPTH = 8
# Phase 3 sweep: learning_rate. Reordered to put production lr (0.05) first
# (which IS the Phase 2 winner) as the within-sweep reference cell.
LEARNING_RATE_CONFIGS = [0.05, 0.03, 0.07]

OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.4_dev_gbm_learning_rate_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)


def evaluate_cell_lr(learning_rate, df, feat_cols, train_pool_ids, cal_aux,
                      holdout_data):
    print(f"\n=== Cell lr={learning_rate} "
          f"(max_iter={PHASE2_WINNER_MAX_ITER}, depth={PHASE2_WINNER_MAX_DEPTH}) ===",
          flush=True)
    t_start = time.time()

    cal_totals = {"tp": 0, "fp": 0, "fn": 0}
    cal_topo = defaultdict(int)
    cal_start_deltas = []
    cal_span_deltas = []
    cal_per_video = {}
    for fold_idx, val_vid in enumerate(train_pool_ids):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        train_mask = df["video_id"].isin(train_ids) & df["exhaustive"]
        train_df = df.loc[train_mask]
        val_df = df.loc[df["video_id"] == val_vid].sort_values("frame")
        t0 = time.time()
        clf = train_gbm(train_df, feat_cols,
                        max_iter=PHASE2_WINNER_MAX_ITER,
                        learning_rate=learning_rate,
                        max_depth=PHASE2_WINNER_MAX_DEPTH)
        Xv = val_df[feat_cols].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(Xv)[:, 1]
        max_frame = int(val_df["frame"].max())
        proba_arr = np.zeros(max_frame + 1, dtype=np.float32)
        proba_arr[val_df["frame"].to_numpy()] = proba
        paw_lk = cal_aux[val_vid]["paw_lk"]
        norm_pos = cal_aux[val_vid]["norm_pos"]
        L = min(len(proba_arr), len(paw_lk), len(norm_pos))
        algos = proba_to_algos(proba_arr[:L], paw_lk[:L], norm_pos[:L])
        gts = load_live_gt("cal", val_vid)
        matched, tp_sd, tp_pd = greedy_match(algos, gts)
        tp = len(matched); fp = len(algos) - tp; fn = len(gts) - tp
        cal_totals["tp"] += tp; cal_totals["fp"] += fp; cal_totals["fn"] += fn
        cal_start_deltas.extend(tp_sd)
        cal_span_deltas.extend(tp_pd)
        tc = classify_matcher_aware(algos, gts, matched)
        for k, v in tc.items():
            cal_topo[k] += v
        cal_per_video[val_vid] = {"tp": tp, "fp": fp, "fn": fn,
                                    "n_algo": len(algos), "n_gt": len(gts)}
        print(f"  cal fold {fold_idx+1}/{len(train_pool_ids)}: "
              f"{val_vid}  TP={tp} FP={fp} FN={fn}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    print(f"  cal LOOCV done ({time.time()-t_start:.1f}s); "
          f"training full-corpus for holdout...", flush=True)
    t0 = time.time()
    full_train_df = df.loc[df["exhaustive"]]
    clf_full = train_gbm(full_train_df, feat_cols,
                          max_iter=PHASE2_WINNER_MAX_ITER,
                          learning_rate=learning_rate,
                          max_depth=PHASE2_WINNER_MAX_DEPTH)
    print(f"  full-corpus train: {time.time()-t0:.1f}s", flush=True)

    hol_totals = {"tp": 0, "fp": 0, "fn": 0}
    hol_topo = defaultdict(int)
    hol_start_deltas = []
    hol_span_deltas = []
    hol_per_video = {}
    for vid, vd in holdout_data.items():
        Xh = vd["X"]
        proba = clf_full.predict_proba(Xh)[:, 1]
        algos = proba_to_algos(proba, vd["paw_lk"], vd["norm_pos"])
        gts = vd["gts"]
        matched, tp_sd, tp_pd = greedy_match(algos, gts)
        tp = len(matched); fp = len(algos) - tp; fn = len(gts) - tp
        hol_totals["tp"] += tp; hol_totals["fp"] += fp; hol_totals["fn"] += fn
        hol_start_deltas.extend(tp_sd)
        hol_span_deltas.extend(tp_pd)
        tc = classify_matcher_aware(algos, gts, matched)
        for k, v in tc.items():
            hol_topo[k] += v
        hol_per_video[vid] = {"tp": tp, "fp": fp, "fn": fn,
                                "n_algo": len(algos), "n_gt": len(gts)}

    cal_sd_med = int(np.median([abs(d) for d in cal_start_deltas])) if cal_start_deltas else None
    cal_pd_med = int(np.median([abs(d) for d in cal_span_deltas])) if cal_span_deltas else None
    hol_sd_med = int(np.median([abs(d) for d in hol_start_deltas])) if hol_start_deltas else None
    hol_pd_med = int(np.median([abs(d) for d in hol_span_deltas])) if hol_span_deltas else None

    print(f"\n  Cal: TP={cal_totals['tp']} FP={cal_totals['fp']} FN={cal_totals['fn']}  "
          f"abs_med start={cal_sd_med} span={cal_pd_med}")
    print(f"  Cal topology: {dict(cal_topo)}")
    print(f"  Hol: TP={hol_totals['tp']} FP={hol_totals['fp']} FN={hol_totals['fn']}  "
          f"abs_med start={hol_sd_med} span={hol_pd_med}")
    print(f"  Hol topology: {dict(hol_topo)}")
    print(f"  Cell total time: {time.time()-t_start:.1f}s", flush=True)

    return {
        "max_iter": PHASE2_WINNER_MAX_ITER,
        "learning_rate": learning_rate,
        "max_depth": PHASE2_WINNER_MAX_DEPTH,
        "bsw_b": BOUNDARY_BUFFER, "bsw_w": BOUNDARY_WEIGHT,
        "cal": {"totals": cal_totals, "topology": dict(cal_topo),
                "start_delta_abs_median": cal_sd_med,
                "span_delta_abs_median": cal_pd_med,
                "per_video": cal_per_video},
        "hol": {"totals": hol_totals, "topology": dict(hol_topo),
                "start_delta_abs_median": hol_sd_med,
                "span_delta_abs_median": hol_pd_med,
                "per_video": hol_per_video},
        "elapsed_seconds": time.time() - t_start,
    }


def main():
    print("=" * 70)
    print(f"GBM HYPERPARAM PHASE 3 -- learning_rate sweep at "
          f"max_iter={PHASE2_WINNER_MAX_ITER}, depth={PHASE2_WINNER_MAX_DEPTH}")
    print(f"Grid: lr in {LEARNING_RATE_CONFIGS}")
    print(f"BSW fixed at b={BOUNDARY_BUFFER}, w={BOUNDARY_WEIGHT}")
    print("=" * 70)
    print()

    print(f"Loading cal corpus from {CORPUS_DIR}...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads((CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    feat_cols = feature_columns()
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"  Train pool: {len(train_pool_ids)} videos, "
          f"{len(eligible_val)} exhaustive folds", flush=True)
    cal_aux = cal_per_video_aux(df)

    print(f"Loading holdout 19 (DLC h5s)...", flush=True)
    holdout_data = {}
    for dlc_path in sorted(HOLDOUT_DLC_DIR.glob(f"*{DLC_SUFFIX}.h5")):
        vid = dlc_path.stem.replace(DLC_SUFFIX, "")
        dlc = load_dlc_h5(dlc_path)
        feats = extract_features(dlc)
        X = feats[feat_cols].to_numpy(dtype="float32")
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        gts = load_live_gt("hol", vid)
        holdout_data[vid] = {"X": X, "paw_lk": paw_lk, "norm_pos": norm_pos, "gts": gts}
    print(f"  {len(holdout_data)} holdout videos loaded", flush=True)
    print()

    results = {"cells": {}, "run_timestamp_start": datetime.utcnow().isoformat() + "Z"}
    for lr in LEARNING_RATE_CONFIGS:
        cell_result = evaluate_cell_lr(lr, df, feat_cols, eligible_val,
                                          cal_aux, holdout_data)
        results["cells"][f"lr_{lr}"] = cell_result
        results["run_timestamp_last_cell"] = datetime.utcnow().isoformat() + "Z"
        (OUT_DIR / "metrics" / "sweep_results.json").write_text(
            json.dumps(results, indent=2, default=int), encoding="utf-8")

    print()
    print("=" * 140)
    print(f"SUMMARY (Phase 3 lr sweep at iter={PHASE2_WINNER_MAX_ITER} "
          f"depth={PHASE2_WINNER_MAX_DEPTH})")
    print("=" * 140)
    print(f"{'lr':<6} | {'cal_TP':>6} {'cal_FP':>6} {'cal_FN':>6} {'cal_sd':>6} {'cal_pd':>6} "
          f"| {'hol_TP':>6} {'hol_FP':>6} {'hol_FN':>6} {'hol_sd':>6} {'hol_pd':>6} | {'time':>5}s")
    print("-" * 140)
    for lr in LEARNING_RATE_CONFIGS:
        r = results["cells"][f"lr_{lr}"]
        ct = r["cal"]["totals"]; ht = r["hol"]["totals"]
        print(f"{lr:<6} | {ct['tp']:>6} {ct['fp']:>6} {ct['fn']:>6} "
              f"{r['cal']['start_delta_abs_median']:>6} "
              f"{r['cal']['span_delta_abs_median']:>6} | "
              f"{ht['tp']:>6} {ht['fp']:>6} {ht['fn']:>6} "
              f"{r['hol']['start_delta_abs_median']:>6} "
              f"{r['hol']['span_delta_abs_median']:>6} | "
              f"{r['elapsed_seconds']:.0f}")
    print()
    print(f"Wrote: {OUT_DIR / 'metrics' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
