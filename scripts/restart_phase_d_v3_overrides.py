"""
Phase D v3: re-run LOOCV with the v1.1.0 extended feature set, then
apply post-prediction override rules and compare accuracy before/after.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.assignment.v1.train import loocv as v1_loocv
from mousereach.assignment.v1.overrides import (
    apply_overrides, summarize_overrides)
from mousereach.assignment.v1.features import (
    feature_columns as baseline_feature_columns,
    causal_feature_columns)


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
SNAPSHOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\assignment\v1.2.0_dev_overrides_loocv"
)


def main():
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    print("Loading v2-causal dataset ...", flush=True)
    df = pd.read_parquet(
        CORPUS_DIR / "phase_d_dataset_v2_causal" / "train_pool.parquet")
    print(f"  {len(df)} reaches across {df['video_id'].nunique()} videos")
    print()

    # Run LOOCV with extended features (same as v1.1.0)
    extended_cols = baseline_feature_columns() + causal_feature_columns()
    print(f"Running LOOCV with {len(extended_cols)} features ...", flush=True)

    import mousereach.assignment.v1.features as v1_features
    original_fc = v1_features.feature_columns
    v1_features.feature_columns = lambda: extended_cols
    try:
        train_pool_ids = sorted(df["video_id"].unique().tolist())
        folds = v1_loocv(df, train_pool_ids,
                         max_iter=200, learning_rate=0.05, max_depth=4)
    finally:
        v1_features.feature_columns = original_fc

    # Collect raw rows
    raw_rows = []
    for f in folds:
        raw_rows.extend(f.rows)

    # Per-segment accuracy BEFORE override (matches v1.1.0 baseline)
    seg_pre = {}
    for r in raw_rows:
        key = (r["video_id"], r["segment_num"])
        if key not in seg_pre:
            seg_pre[key] = {"outcome": r["segment_outcome"],
                            "predicted_causal": None, "gt_causal": None}
        if r["pred_causal"] == 1:
            seg_pre[key]["predicted_causal"] = (r["reach_id"], r["reach_start_frame"])
        if r["gt_causal"] == 1:
            seg_pre[key]["gt_causal"] = (r["reach_id"], r["reach_start_frame"])

    by_outcome_pre = defaultdict(lambda: {"n_segments": 0, "causal_correct": 0})
    for k, s in seg_pre.items():
        out = s["outcome"]
        by_outcome_pre[out]["n_segments"] += 1
        if (s["predicted_causal"] is not None and s["gt_causal"] is not None
                and s["predicted_causal"] == s["gt_causal"]):
            by_outcome_pre[out]["causal_correct"] += 1

    # Apply overrides
    print()
    print("Applying overrides ...", flush=True)
    override_rows = apply_overrides(raw_rows, df, pre_apex_threshold=0.10)
    over_summary = summarize_overrides(override_rows)
    print(f"  Total predictions: {over_summary['n_total']}")
    print(f"  Changed by override: {over_summary['n_changed']}")
    for reason, n in over_summary["by_reason"].items():
        print(f"    {reason}: {n}")
    print()

    # Per-segment accuracy AFTER override
    seg_post = {}
    for r in override_rows:
        key = (r.video_id, r.segment_num)
        if key not in seg_post:
            seg_post[key] = {"outcome": r.segment_outcome,
                             "predicted_causal": None, "gt_causal": None}
        if r.pred_causal_final == 1:
            seg_post[key]["predicted_causal"] = (r.reach_id, r.reach_start_frame)
        if r.gt_causal == 1:
            seg_post[key]["gt_causal"] = (r.reach_id, r.reach_start_frame)

    by_outcome_post = defaultdict(lambda: {"n_segments": 0, "causal_correct": 0})
    for k, s in seg_post.items():
        out = s["outcome"]
        by_outcome_post[out]["n_segments"] += 1
        if (s["predicted_causal"] is not None and s["gt_causal"] is not None
                and s["predicted_causal"] == s["gt_causal"]):
            by_outcome_post[out]["causal_correct"] += 1

    # Report comparison
    print("=" * 80)
    print("CAUSAL ATTRIBUTION (pre vs post overrides)")
    print("=" * 80)
    print(f"{'outcome':<22} {'n_segs':>8} {'pre-override':>16} {'post-override':>16}")
    for outcome in ("retrieved", "displaced_sa"):
        n = by_outcome_pre[outcome]["n_segments"]
        pre_c = by_outcome_pre[outcome]["causal_correct"]
        post_c = by_outcome_post[outcome]["causal_correct"]
        if n > 0:
            print(f"  {outcome:<20} {n:>8} "
                  f"{pre_c}/{n} ({100*pre_c/n:>4.1f}%)   "
                  f"{post_c}/{n} ({100*post_c/n:>4.1f}%)")
    print()

    # Save
    (metrics_dir / "loocv_results.json").write_text(json.dumps({
        "n_folds": len(folds),
        "feature_count": len(extended_cols),
        "by_outcome_pre_override": dict(by_outcome_pre),
        "by_outcome_post_override": dict(by_outcome_post),
        "override_summary": over_summary,
        "raw_rows": raw_rows,
        "override_rows": [
            {"video_id": r.video_id, "segment_num": r.segment_num,
             "reach_id": r.reach_id, "gt_causal": r.gt_causal,
             "pred_causal_raw": r.pred_causal_raw,
             "pred_causal_final": r.pred_causal_final,
             "proba_causal": r.proba_causal,
             "override_reason": r.override_reason,
             "pre_apex_inside_pillar_frac": r.pre_apex_inside_pillar_frac}
            for r in override_rows
        ],
    }, indent=2), encoding="utf-8")
    print(f"Saved: {metrics_dir / 'loocv_results.json'}")


if __name__ == "__main__":
    main()
