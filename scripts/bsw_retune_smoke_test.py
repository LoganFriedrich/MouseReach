"""Quick smoke test for the BSW retune runner. Trains 1 LOOCV fold at
w=0.8, scores 1 holdout video, prints timing. ~5-15 min if everything
works.

Purpose: catch path/data/import issues before committing to ~10+ hours
of background compute.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the actual sweep module so we test the same code paths
from restart_phase_b_bsw_retune_w_sweep import (
    CORPUS_DIR, HOLDOUT_DLC_DIR, DLC_SUFFIX,
    BOUNDARY_BUFFER,
    train_gbm, proba_to_algos, greedy_match, classify_matcher_aware,
    cal_per_video_aux, load_live_gt,
)
from mousereach.reach.v8.features import (
    feature_columns, extract_features, load_dlc_h5,
)
from mousereach.reach.v8.postprocess import (
    compute_paw_mean_lk, compute_hand_to_boxl_norm_pos,
)


def main():
    print("=" * 70)
    print("BSW RETUNE SMOKE TEST -- 1 cal fold + 1 hol video at b=1 w=0.8")
    print("=" * 70)
    print()

    t_overall = time.time()

    # Load corpus
    print("Loading parquet...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads((CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    feat_cols = feature_columns()
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"  Parquet load: {time.time()-t0:.1f}s. "
          f"Train pool: {len(train_pool_ids)} videos, "
          f"{len(eligible_val)} exhaustive folds. "
          f"Features: {len(feat_cols)}.", flush=True)

    # Cal aux
    print("Computing cal aux (paw_lk, norm_pos)...", flush=True)
    t0 = time.time()
    cal_aux = cal_per_video_aux(df)
    print(f"  Cal aux: {time.time()-t0:.1f}s for {len(cal_aux)} videos", flush=True)

    # Smoke test: train 1 fold at w=0.8
    val_vid = eligible_val[0]
    train_ids = [v for v in eligible_val if v != val_vid]
    train_mask = df["video_id"].isin(train_ids) & df["exhaustive"]
    train_df = df.loc[train_mask]
    val_df = df.loc[df["video_id"] == val_vid].sort_values("frame")
    print(f"\nSmoke test fold: val={val_vid}", flush=True)
    print(f"  Train rows: {len(train_df)}, val rows: {len(val_df)}", flush=True)

    t0 = time.time()
    clf = train_gbm(train_df, feat_cols, b=BOUNDARY_BUFFER, w=0.8)
    train_time = time.time() - t0
    print(f"  TRAIN time: {train_time:.1f}s (= {train_time/60:.1f}min)", flush=True)

    # Inference + postprocess + match on val video
    t0 = time.time()
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
    matched, _, _ = greedy_match(algos, gts)
    inf_time = time.time() - t0
    print(f"  INFERENCE+match: {inf_time:.1f}s", flush=True)
    print(f"  Cal val: n_algo={len(algos)} n_gt={len(gts)} "
          f"TP={len(matched)} FP={len(algos)-len(matched)} "
          f"FN={len(gts)-len(matched)}", flush=True)

    # Quick holdout test on 1 video
    print("\nHoldout test on 1 video...", flush=True)
    dlc_path = sorted(HOLDOUT_DLC_DIR.glob(f"*{DLC_SUFFIX}.h5"))[0]
    vid = dlc_path.stem.replace(DLC_SUFFIX, "")
    print(f"  vid: {vid}", flush=True)
    t0 = time.time()
    dlc = load_dlc_h5(dlc_path)
    feats = extract_features(dlc)
    Xh = feats[feat_cols].to_numpy(dtype="float32")
    paw_lk_h = compute_paw_mean_lk(dlc)
    norm_pos_h = compute_hand_to_boxl_norm_pos(dlc)
    gts_h = load_live_gt("hol", vid)
    proba_h = clf.predict_proba(Xh)[:, 1]
    algos_h = proba_to_algos(proba_h, paw_lk_h, norm_pos_h)
    matched_h, _, _ = greedy_match(algos_h, gts_h)
    hol_time = time.time() - t0
    print(f"  HOLDOUT inference: {hol_time:.1f}s", flush=True)
    print(f"  Hol score: n_algo={len(algos_h)} n_gt={len(gts_h)} "
          f"TP={len(matched_h)} FP={len(algos_h)-len(matched_h)} "
          f"FN={len(gts_h)-len(matched_h)}", flush=True)

    print(f"\nTotal smoke test: {time.time()-t_overall:.1f}s", flush=True)
    print("\n--- Time projection for full Phase 1 ---", flush=True)
    folds_per_cell = len(eligible_val)
    cells = 4
    train_total_min = train_time / 60 * (folds_per_cell + 1) * cells
    hol_total_min = hol_time / 60 * 19 * cells
    print(f"  Per-cell trains: {folds_per_cell} LOOCV + 1 full-corpus = "
          f"{folds_per_cell+1} trains x {train_time:.0f}s = "
          f"{(folds_per_cell+1)*train_time/60:.0f} min per cell", flush=True)
    print(f"  Per-cell holdout: 19 videos x {hol_time:.0f}s = "
          f"{19*hol_time/60:.0f} min per cell", flush=True)
    print(f"  4 cells total estimate: ~{train_total_min + hol_total_min:.0f} min "
          f"= ~{(train_total_min + hol_total_min)/60:.1f} hours", flush=True)


if __name__ == "__main__":
    main()
