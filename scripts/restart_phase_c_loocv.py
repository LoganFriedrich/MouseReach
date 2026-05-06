"""
Phase C step 2: LOOCV training for the v5 outcome classifier.

Trains one model per fold (val = single held-out video, train = the
other 36 train_pool videos). Aggregates predictions and renders the
canonical Sankey + directional confusion table.

Reporting per the user-mandated standard
(reach_outcome_evaluation_format.md):
  - LEAD with Sankey + directional confusion (which GT class -> which
    algo class). NEVER F1.
  - Show counts. Precision/recall as supporting only, if at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.outcomes.v5.train import loocv, confusion_matrix, directional_summary
from mousereach.improvement.outcome._run_notebooks import run_sankey


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\outcome\v5.0.0_dev_initial_loocv"
)


def main():
    print("=" * 70)
    print("PHASE C OUTCOME LOOCV")
    print("=" * 70)
    print()

    print("Loading train_pool.parquet ...", flush=True)
    train_df = pd.read_parquet(CORPUS_DIR / "phase_c_dataset" / "train_pool.parquet")
    folds_def = json.loads((CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    print(f"  {len(train_df)} segments across {len(train_pool_ids)} videos", flush=True)
    print()

    print("Running LOOCV (1 fold per video; outcome trains on all kinds) ...", flush=True)
    folds = loocv(
        train_df=train_df,
        train_video_ids=train_pool_ids,
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
    )
    print()

    # Aggregate
    all_rows = []
    for f in folds:
        all_rows.extend(f.rows)

    cm = confusion_matrix(all_rows)
    direction = directional_summary(all_rows)

    # Per-class TP/FP/FN counts
    per_class_counts = {}
    for cls in ["retrieved", "displaced_sa", "untouched", "abnormal_exception"]:
        n_gt = sum(1 for r in all_rows if r.gt_label == cls)
        n_algo = sum(1 for r in all_rows if r.algo_label == cls)
        n_correct = sum(1 for r in all_rows if r.gt_label == cls and r.algo_label == cls)
        per_class_counts[cls] = {
            "n_gt": n_gt,
            "n_algo": n_algo,
            "n_correct": n_correct,
            "precision": (n_correct / n_algo) if n_algo else 0.0,
            "recall": (n_correct / n_gt) if n_gt else 0.0,
        }

    n_total = sum(1 for r in all_rows if r.gt_label is not None)
    n_correct = sum(
        1 for r in all_rows if r.gt_label is not None and r.gt_label == r.algo_label)
    strict_acc = n_correct / n_total if n_total else None

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    scalars = {
        "n_segments_paired": n_total,
        "outcome_label": {
            "strict_accuracy": strict_acc,
            "committed_accuracy": strict_acc,
            "abstention_rate": 0.0,
            "per_class": per_class_counts,
            "confusion_matrix": cm,
        },
        "directional_summary": direction,
    }
    (metrics_dir / "scalars.json").write_text(
        json.dumps(scalars, indent=2), encoding="utf-8")

    # Per-segment CSV
    per_segment_rows = []
    for r in all_rows:
        per_segment_rows.append({
            "video_id": r.video_id,
            "segment_num": r.segment_num,
            "gt_outcome": r.gt_label,
            "algo_outcome": r.algo_label,
            "outcome_label_match": (r.gt_label == r.algo_label),
            "gt_exhaustive": r.gt_exhaustive,
        })
    pd.DataFrame(per_segment_rows).to_csv(
        metrics_dir / "outcome_per_segment.csv", index=False)

    # Render Sankey via the canonical runner
    run_sankey(SNAPSHOT_DIR)

    print("=" * 70)
    print("AGGREGATE LOOCV RESULTS")
    print("=" * 70)
    print(f"Total segments evaluated: {n_total}")
    print(f"Correct (strict): {n_correct} ({100*strict_acc:.1f}%)")
    print()
    print("Per-class counts (GT, algo, correct):")
    for cls, c in per_class_counts.items():
        print(f"  {cls:>22s}: n_gt={c['n_gt']:>3}  n_algo={c['n_algo']:>3}  correct={c['n_correct']:>3}")
    print()
    print("Directional confusion (GT class -> algo class):")
    for gt_cls, dist in sorted(direction["by_gt"].items()):
        total = sum(dist.values())
        if total == 0:
            continue
        rest = sorted(((v, k) for k, v in dist.items()), reverse=True)
        line = f"  {gt_cls:>22s} (n={total}): "
        line += ", ".join(f"{k}={v}" for v, k in rest)
        print(line)
    print()
    print(f"Saved Sankey to: {SNAPSHOT_DIR / 'figures' / 'sankey.png'}")
    print(f"Saved scalars to: {metrics_dir / 'scalars.json'}")


if __name__ == "__main__":
    main()
