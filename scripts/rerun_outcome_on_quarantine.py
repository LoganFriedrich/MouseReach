"""
Re-run PelletOutcomeDetector on the v4.0.0_dev_walkthrough quarantine corpus
with the currently-installed mousereach code, then compute outcome metrics
via the improvement framework.

Inputs (frozen quarantine):
  iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough/
    dlc/{video}DLC_*.h5
    gt/{video}_unified_ground_truth.json
    algo_outputs/{video}_segments.json   (use these segments)
    algo_outputs/{video}_reaches.json    (use these reaches)

Outputs:
  Improvement_Snapshots/outcome/<TAG>/algo_outcomes/{video}_pellet_outcomes.json
  Improvement_Snapshots/outcome/<TAG>/metrics/scalars.json
  Improvement_Snapshots/outcome/<TAG>/metrics/outcome_per_segment.csv
  Improvement_Snapshots/outcome/<TAG>/metrics/per_video.csv

Usage:
  python rerun_outcome_on_quarantine.py <TAG>
  e.g.:
  python rerun_outcome_on_quarantine.py outcome_v4.0.0_step6_post_gt_migration
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure mousereach is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.outcomes.core.pellet_outcome import PelletOutcomeDetector
from mousereach.improvement.outcome.metrics import compute_outcome_metrics


QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
ALGO_INPUT_DIR = QUARANTINE / "algo_outputs"  # source of segments + reaches

SNAPSHOTS_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome"
)


def main(tag: str) -> None:
    snapshot_dir = SNAPSHOTS_ROOT / tag
    out_outcomes = snapshot_dir / "algo_outcomes"
    out_metrics = snapshot_dir / "metrics"
    out_outcomes.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    # Discover videos via DLC files (canonical source)
    dlc_files = sorted(DLC_DIR.glob("*DLC_*.h5"))
    print(f"Found {len(dlc_files)} DLC files in quarantine.")

    detector = PelletOutcomeDetector()
    n_ok = 0
    n_skip = 0

    for dlc_path in dlc_files:
        video = dlc_path.stem.split("DLC_")[0]
        seg_path = ALGO_INPUT_DIR / f"{video}_segments.json"
        reach_path = ALGO_INPUT_DIR / f"{video}_reaches.json"

        if not seg_path.exists():
            print(f"  SKIP {video}: no segments file")
            n_skip += 1
            continue

        try:
            results = detector.detect(
                dlc_path=dlc_path,
                segments_path=seg_path,
                reaches_path=reach_path if reach_path.exists() else None,
            )
            out_path = out_outcomes / f"{video}_pellet_outcomes.json"
            PelletOutcomeDetector.save_results(
                results, out_path, validation_status="auto_approved"
            )
            n_ok += 1
        except Exception as e:
            print(f"  ERROR {video}: {type(e).__name__}: {e}")
            n_skip += 1

    print(f"\nOutcome detection: {n_ok} OK, {n_skip} skipped/errored.")
    print(f"Outputs: {out_outcomes}")

    # Compute metrics via the framework
    print("\nComputing outcome metrics via improvement.outcome.metrics ...")
    scalars = compute_outcome_metrics(
        gt_dir=GT_DIR,
        algo_dir=out_outcomes,
        output_dir=out_metrics,
        reaches_dir=ALGO_INPUT_DIR,
    )

    label = scalars.get("outcome_label", {})
    pc = label.get("per_class", {})
    print()
    print(f"Snapshot: {snapshot_dir}")
    print(f"  n_segments_paired: {scalars.get('n_segments_paired')}")
    print(f"  strict_accuracy:   {label.get('strict_accuracy', 0.0):.4f}")
    print(f"  per-class precision/recall/f1 (n_gt, n_algo):")
    for cls in ("retrieved", "displaced_sa", "displaced_outside", "untouched",
                "abnormal_exception"):
        c = pc.get(cls, {})
        print(
            f"    {cls:>20s}: "
            f"P={c.get('precision', 0.0):.3f}  "
            f"R={c.get('recall', 0.0):.3f}  "
            f"F={c.get('f1', 0.0):.3f}  "
            f"(n_gt={c.get('n_gt', 0)}, n_algo={c.get('n_algo', 0)})"
        )

    cm = label.get("confusion_matrix", {})
    print(f"  confusion matrix entries: {len(cm)}")
    for k, v in sorted(cm.items()):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rerun_outcome_on_quarantine.py <TAG>")
        sys.exit(1)
    main(sys.argv[1])
