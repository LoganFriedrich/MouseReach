"""
Phase D step 2: LOOCV training of the assignment classifier on GT
reaches.

This evaluates the causal-vs-miss classifier directly against GT-known
causal reaches. The cleanest read on whether the classifier learns
anything beyond the "last reach is causal" heuristic.

Reports:
  - Per-segment-outcome causal-attribution recall (% of touched segments
    where the model picked the GT causal reach correctly)
  - Comparison to two baselines:
      (a) "last reach is causal" (the placeholder heuristic)
      (b) "first reach is causal"
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.assignment.v1.train import loocv

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
SNAPSHOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\assignment\v1.0.0_dev_loocv_gt_reaches"
)


def baseline_first_or_last(train_df: pd.DataFrame, mode: str) -> dict:
    """Score the first-reach or last-reach heuristic against GT.

    Returns per-outcome causal_correct / n_segments.
    """
    by_outcome: dict = {}
    for (vid, sn), grp in train_df.groupby(["video_id", "segment_num"]):
        out = grp["segment_outcome"].iloc[0]
        if mode == "last":
            picked = grp["reach_start_frame"].idxmax()
        elif mode == "first":
            picked = grp["reach_start_frame"].idxmin()
        else:
            raise ValueError(f"bad mode {mode}")
        gt_causal_idx = grp[grp["causal"] == 1].index
        b = by_outcome.setdefault(out, {"n_segments": 0, "causal_correct": 0})
        b["n_segments"] += 1
        if len(gt_causal_idx) > 0 and picked in gt_causal_idx:
            b["causal_correct"] += 1
    return by_outcome


def main():
    SNAPSHOT.mkdir(parents=True, exist_ok=True)

    print("Loading train_pool.parquet ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_d_dataset" / "train_pool.parquet")
    train_pool_ids = sorted(df["video_id"].unique().tolist())
    print(f"  {len(df)} per-reach rows across {len(train_pool_ids)} videos")
    print()

    # Baselines for comparison
    print("Computing baselines (no training) ...")
    last_baseline = baseline_first_or_last(df, "last")
    first_baseline = baseline_first_or_last(df, "first")
    print()

    print("Running LOOCV on assignment classifier ...", flush=True)
    folds = loocv(df, train_pool_ids, max_iter=200, learning_rate=0.05, max_depth=4)
    print()

    # Aggregate
    rows = []
    for f in folds:
        rows.extend(f.rows)

    # Per-segment causal attribution accuracy
    seg_results: dict = {}
    for r in rows:
        key = (r["video_id"], r["segment_num"])
        if key not in seg_results:
            seg_results[key] = {
                "outcome": r["segment_outcome"],
                "predicted_causal_idx": None,
                "gt_causal_idx": None,
            }
        if r["pred_causal"] == 1:
            seg_results[key]["predicted_causal_idx"] = (r["reach_id"],
                                                        r["reach_start_frame"])
        if r["gt_causal"] == 1:
            seg_results[key]["gt_causal_idx"] = (r["reach_id"],
                                                 r["reach_start_frame"])

    by_outcome: dict = defaultdict(lambda: {"n_segments": 0, "causal_correct": 0})
    for key, s in seg_results.items():
        out = s["outcome"]
        by_outcome[out]["n_segments"] += 1
        if s["predicted_causal_idx"] is not None and \
           s["gt_causal_idx"] is not None and \
           s["predicted_causal_idx"] == s["gt_causal_idx"]:
            by_outcome[out]["causal_correct"] += 1

    print("=" * 80)
    print("CAUSAL ATTRIBUTION ACCURACY (per segment)")
    print("=" * 80)
    print()
    print(f"{'outcome':<22} {'n_segs':>8} {'classifier':>14} {'last-reach':>14} {'first-reach':>14}")
    for outcome in ("retrieved", "displaced_sa"):
        n = by_outcome[outcome]["n_segments"]
        clf_correct = by_outcome[outcome]["causal_correct"]
        last_correct = last_baseline.get(outcome, {}).get("causal_correct", 0)
        first_correct = first_baseline.get(outcome, {}).get("causal_correct", 0)
        if n > 0:
            print(f"  {outcome:<20} {n:>8} "
                  f"{clf_correct}/{n} ({100*clf_correct/n:>4.1f}%)   "
                  f"{last_correct}/{n} ({100*last_correct/n:>4.1f}%)   "
                  f"{first_correct}/{n} ({100*first_correct/n:>4.1f}%)")
    print()

    # Save
    metrics_dir = SNAPSHOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / "loocv_results.json").write_text(json.dumps({
        "n_folds": len(folds),
        "by_outcome": dict(by_outcome),
        "last_reach_baseline": last_baseline,
        "first_reach_baseline": first_baseline,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {metrics_dir / 'loocv_results.json'}")


if __name__ == "__main__":
    main()
