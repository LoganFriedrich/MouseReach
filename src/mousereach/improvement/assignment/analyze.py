"""
Assignment analyzer (algo 4) -- per-reach confusion.

Reads:
  <snapshot>/algo_outputs/{video}_pellet_outcomes.json
        (must include `causal_reach_id` for committed touched segments,
         `would_be_causal_reach_id` for triaged segments)
  <snapshot>/algo_outputs/{video}_reaches.json
        (each reach must have a `reach_id` int)
  <gt_dir>/{video}_unified_ground_truth.json

Writes:
  <snapshot>/metrics/scalars.json    (canonical -- run_sankey reads from here)
  <snapshot>/metrics/assignment_scalars.json (alias)

Uses `compute_per_reach_confusion` in
`mousereach.improvement.outcome.metrics`. The labeling rules
(causal-only abnormal_exception, would-be-causal-only triaged) are
enforced inside that function -- DO NOT duplicate the logic.
"""
from __future__ import annotations

import json
from pathlib import Path

from mousereach.improvement.lib.inputs import load_snapshot_paths, write_scalars
from mousereach.improvement.outcome.metrics import compute_per_reach_confusion


def analyze(snapshot_dir: Path) -> dict:
    paths = load_snapshot_paths(snapshot_dir)

    res = compute_per_reach_confusion(
        gt_dir=paths.gt_dir,
        algo_dir=paths.algo_outputs_dir,
        reaches_dir=paths.algo_outputs_dir,
        video_ids=paths.video_ids,
    )

    # Count triage flows for surfacing in the legend
    cm = res["confusion_matrix"]
    triage_flows = {k: v for k, v in cm.items() if k.split("__")[1] == "triaged"}
    triage_count = sum(triage_flows.values())

    out = {
        "n_videos": len(paths.video_ids),
        "n_reaches_universe": res["n_reaches_universe"],
        "triage_count": triage_count,
        "outcome_label": {
            "strict_accuracy": None,
            "committed_accuracy": None,
            "abstention_rate": 0.0,
            "per_class": res["per_class"],
            "confusion_matrix": cm,
        },
        "directional_summary": {
            "by_gt": _by_side(cm, side=0),
            "by_algo": _by_side(cm, side=1),
        },
    }
    write_scalars(paths.metrics_dir, out, "assignment_scalars.json")
    write_scalars(paths.metrics_dir, out, "scalars.json")  # for run_sankey
    return out


def _by_side(cm: dict, side: int) -> dict:
    """Group confusion matrix counts by GT side (0) or algo side (1)."""
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(int))
    for key, v in cm.items():
        parts = key.split("__")
        if len(parts) != 2:
            continue
        if side == 0:
            grouped[parts[0]][parts[1]] += v
        else:
            grouped[parts[1]][parts[0]] += v
    return {k: dict(v) for k, v in grouped.items()}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.assignment.analyze <snapshot_dir>")
        sys.exit(1)
    res = analyze(Path(sys.argv[1]))
    print(f"n_reaches_universe={res['n_reaches_universe']}  triage={res['triage_count']}")
