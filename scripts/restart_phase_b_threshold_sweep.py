"""
Phase B step 4: threshold sweep across LOOCV folds.

Training is the slow part; threshold is a post-processing knob.
This script trains one model per fold (same as the LOOCV runner), but
evaluates at multiple thresholds without retraining.

Output: per-threshold aggregate metrics, plus a side-by-side figure.
The user's "over-call OK" preference predicts a lower threshold will
trade FNs for FPs and lift overall TP. We pick the lowest threshold
that doesn't bloat FP catastrophically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.eval import (AlgoReach, GTReach, MatchResult,
                                       evaluate_reaches, summarize_results)
from mousereach.reach.v8.features import feature_columns
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
SNAPSHOT_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection"
)
SWEEP_DIR = SNAPSHOT_ROOT / "v8.0.0_dev_threshold_sweep"

THRESHOLDS = [0.20, 0.30, 0.40, 0.50]
MERGE_GAP = 2
MIN_SPAN = 3


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train_pool.parquet ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    feat_cols = feature_columns()

    # Exhaustive subset is what we val against
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    val_video_ids = [v for v in train_pool_ids if v in exh_set]
    print(f"  {len(val_video_ids)} exhaustive videos for LOOCV", flush=True)
    print()

    # results[threshold] = list of MatchResult (across all folds, all videos)
    sweep_results = {t: [] for t in THRESHOLDS}

    for i, val_vid in enumerate(val_video_ids):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        # Train on exhaustive only
        train_mask = df["video_id"].isin(train_ids) & df["exhaustive"]
        train = df.loc[train_mask]
        val = df.loc[df["video_id"] == val_vid]

        X_train = train[feat_cols].to_numpy(dtype=np.float32)
        y_train = train["label"].to_numpy(dtype=np.int8)
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        sw = np.where(
            y_train == 1, len(y_train) / (2.0 * n_pos),
            len(y_train) / (2.0 * n_neg)).astype(np.float32) if (n_pos and n_neg) else None

        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6,
            random_state=42, early_stopping=False,
        )
        clf.fit(X_train, y_train, sample_weight=sw)

        Xv = val[feat_cols].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(Xv)[:, 1]
        sub = val.sort_values("frame")
        rid = sub["reach_id"].to_numpy()
        frames = sub["frame"].to_numpy()
        gt_reaches: List[GTReach] = []
        for ri in sorted(set(rid[rid >= 0].tolist())):
            rmask = rid == ri
            f = frames[rmask]
            gt_reaches.append(GTReach(
                start_frame=int(f.min()), end_frame=int(f.max()),
                video_id=val_vid, index=ri))

        line_parts = [f"  fold {i+1}/{len(val_video_ids)}: val={val_vid}"]
        for thr in THRESHOLDS:
            algo_raw = probabilities_to_reaches(
                proba, threshold=thr, merge_gap=MERGE_GAP, min_span=MIN_SPAN)
            algo = [
                AlgoReach(start_frame=r.start_frame, end_frame=r.end_frame,
                          video_id=val_vid, index=j)
                for j, r in enumerate(algo_raw)
            ]
            res = evaluate_reaches(algo, gt_reaches, video_id=val_vid)
            sweep_results[thr].extend(res)
            n_tp = sum(1 for r in res if r.status == "tp")
            n_fp = sum(1 for r in res if r.status == "fp")
            n_fn = sum(1 for r in res if r.status == "fn")
            line_parts.append(f"thr={thr}: TP={n_tp} FP={n_fp} FN={n_fn}")
        print("  | ".join(line_parts), flush=True)

    print()
    print("=" * 80)
    print("THRESHOLD SWEEP AGGREGATE")
    print("=" * 80)

    aggregate = {}
    for thr in THRESHOLDS:
        s = summarize_results(sweep_results[thr])
        aggregate[str(thr)] = s
        print(f"\nthreshold={thr}")
        print(f"  TP={s['n_tp']}  FP={s['n_fp']}  FN={s['n_fn']}")
        sd = s["tp_start_delta"]
        spd = s["tp_span_delta"]
        print(f"  start_delta (TPs): median={sd['median']}  |median|={sd['abs_median']}  "
              f"p10={sd['p10']}  p90={sd['p90']}  range=[{sd['min']},{sd['max']}]")
        print(f"  span_delta  (TPs): median={spd['median']}  |median|={spd['abs_median']}  "
              f"p10={spd['p10']}  p90={spd['p90']}  range=[{spd['min']},{spd['max']}]")

    # Save raw + aggregate
    (SWEEP_DIR / "metrics").mkdir(exist_ok=True)
    (SWEEP_DIR / "metrics" / "sweep_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8")

    # Render per-threshold figures into a single subdir each
    for thr in THRESHOLDS:
        thr_dir = SWEEP_DIR / f"threshold_{thr:.2f}"
        thr_dir.mkdir(exist_ok=True)
        raw = [
            {"status": r.status, "video_id": r.video_id,
             "gt_index": r.gt_index, "algo_index": r.algo_index,
             "start_delta": r.start_delta, "span_delta": r.span_delta}
            for r in sweep_results[thr]
        ]
        render_v8_reach_figures(
            snapshot_dir=thr_dir,
            raw_results=raw,
            summary=aggregate[str(thr)],
            title_suffix=f" (LOOCV, threshold={thr})",
        )

    print()
    print(f"Saved sweep_aggregate.json + per-threshold figures to {SWEEP_DIR}")


if __name__ == "__main__":
    main()
