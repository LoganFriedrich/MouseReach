"""
Outcome analyzer (algo 3) -- per-PELLET (per-segment).

Reads:
  <snapshot>/algo_outputs/{video}_pellet_outcomes.json  (per-segment outcome)
  <gt_dir>/{video}_unified_ground_truth.json           (per-segment GT)

Writes:
  <snapshot>/metrics/outcome_scalars.json

Per-segment confusion. 4 classes: retrieved, displaced_sa, untouched,
abnormal_exception. displaced_outside is collapsed to displaced_sa.
algo can also emit "triaged" -- counted as a separate flow.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from mousereach.improvement.lib.inputs import (
    load_snapshot_paths, load_algo_outcomes, load_gt_segments, write_scalars,
)
from .schema import SegmentOutcomeRow, OutcomeScalars

CLASSES = ["retrieved", "displaced_sa", "untouched", "abnormal_exception", "triaged"]


def _collapse(o):
    if o == "displaced_outside":
        return "displaced_sa"
    return o


def analyze(snapshot_dir: Path) -> dict:
    paths = load_snapshot_paths(snapshot_dir)
    rows: List[SegmentOutcomeRow] = []
    confusion: Dict[str, int] = defaultdict(int)
    triage_count = 0

    for vid in paths.video_ids:
        algo_data = load_algo_outcomes(paths.algo_outputs_dir, vid)
        gt_segs, exhaustive = load_gt_segments(paths.gt_dir, vid)
        gt_by = {s["segment_num"]: s for s in gt_segs}

        for seg in algo_data.get("segments", []) or []:
            sn = seg.get("segment_num")
            gt = gt_by.get(sn)
            if gt is None:
                continue
            algo_label = _collapse(seg.get("outcome"))
            gt_label = _collapse(gt.get("outcome"))
            if algo_label == "triaged":
                triage_count += 1

            ai = seg.get("interaction_frame")
            gi = gt.get("interaction_frame")
            delta = (int(ai) - int(gi)) if (ai is not None and gi is not None) else None

            rows.append(SegmentOutcomeRow(
                video_id=vid, segment_num=int(sn),
                gt_outcome=gt_label, algo_outcome=algo_label,
                gt_interaction_frame=gi, algo_interaction_frame=ai,
                interaction_delta=delta, gt_exhaustive=exhaustive,
            ))
            if gt_label and algo_label:
                confusion[f"{gt_label}__{algo_label}"] += 1

    # Per-class P/R counts
    per_class = {}
    for cls in CLASSES:
        n_gt = sum(1 for r in rows if r.gt_outcome == cls)
        n_algo = sum(1 for r in rows if r.algo_outcome == cls)
        n_correct = sum(1 for r in rows
                        if r.gt_outcome == cls and r.algo_outcome == cls)
        per_class[cls] = {"n_gt": n_gt, "n_algo": n_algo, "n_correct": n_correct}

    n_correct = sum(1 for r in rows if r.gt_outcome == r.algo_outcome and r.gt_outcome)

    out = OutcomeScalars(
        n_videos=len(paths.video_ids),
        n_segments=len(rows),
        n_correct=n_correct,
        triage_count=triage_count,
        confusion_matrix=dict(confusion),
        per_class=per_class,
        rows=[r.to_dict() for r in rows],
    ).to_dict()

    # Also write the legacy outcome_label key the run_sankey reader expects
    out["outcome_label"] = {
        "confusion_matrix": dict(confusion),
        "per_class": per_class,
        "strict_accuracy": (n_correct / len(rows)) if rows else None,
        "committed_accuracy": (n_correct / len(rows)) if rows else None,
        "abstention_rate": 0.0,
    }
    out["n_segments_paired"] = len(rows)

    write_scalars(paths.metrics_dir, out, "outcome_scalars.json")
    # ALSO write the canonical scalars.json the existing run_sankey reads
    write_scalars(paths.metrics_dir, out, "scalars.json")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.outcome.analyze <snapshot_dir>")
        sys.exit(1)
    res = analyze(Path(sys.argv[1]))
    print(f"n_segments={res['n_segments']}  n_correct={res['n_correct']}  triage={res['triage_count']}")
