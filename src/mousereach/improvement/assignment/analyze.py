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

Two-level reporting (parallel to outcome's analyzer):

  - **pre_review**: algo state BEFORE GT auto-resolve fired. For
    segments cleared by gt_auto_resolve / human triage clearing, we
    revert ``outcome`` to ``original_outcome`` (or "triaged" if not
    preserved), clear ``causal_reach_id``, and re-flag the segment.
    The per-reach confusion is then computed against this reverted view.
  - **post_review**: algo state as currently on disk (post resolve).

Both confusion matrices land in ``outcome_label.{pre_review,
post_review}`` so the shared ``run_sankey`` renderer produces the
same two-panel layout it does for the outcome detector.

The labeling rules (causal-only abnormal_exception,
would-be-causal-only triaged) live in
``mousereach.improvement.outcome.metrics.compute_per_reach_confusion``;
do not duplicate that logic here.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict

from mousereach.improvement.lib.inputs import load_snapshot_paths, write_scalars
from mousereach.improvement.outcome.metrics import compute_per_reach_confusion


def _build_pre_review_dir(algo_dir: Path) -> Path:
    """Build a temp dir of pellet_outcomes JSONs with ``triage_cleared``
    segments reverted to their pre-resolution state. Reaches JSONs are
    copied unchanged (algo's reach detector output isn't affected by
    GT auto-resolve at the assignment-eval level)."""
    tmp = Path(tempfile.mkdtemp(prefix="mr_pre_review_"))
    for src in algo_dir.glob("*_pellet_outcomes.json"):
        data = json.loads(src.read_text(encoding="utf-8"))
        for seg in data.get("segments", []) or []:
            if not seg.get("triage_cleared"):
                continue
            orig = seg.get("original_outcome") or "triaged"
            seg["outcome"] = orig
            seg["flagged_for_review"] = True
            # Pre-resolution: cascade hadn't stamped a causal reach.
            seg["causal_reach_id"] = None
            # Preserve `would_be_causal_reach_id` if the cascade had one;
            # absence means the assignment grader will fall back to IFR
            # containment per the labeling rule, which is correct for
            # the pre-review view too.
        (tmp / src.name).write_text(json.dumps(data), encoding="utf-8")
    for src in algo_dir.glob("*_reaches.json"):
        shutil.copy2(src, tmp / src.name)
    return tmp


def _summarize(cm: Dict[str, int]) -> Dict[str, object]:
    """Build the per-level summary block (accuracy / n_correct /
    n_triaged) that ``run_sankey`` reads."""
    total = sum(cm.values()) if cm else 0
    n_correct = sum(v for k, v in cm.items()
                    if "__" in k and k.split("__")[0] == k.split("__")[1])
    n_triaged = sum(v for k, v in cm.items() if k.split("__")[1] == "triaged")
    return {
        "confusion_matrix": cm,
        "accuracy": (n_correct / total) if total else None,
        "n_correct": n_correct,
        "n_triaged": n_triaged,
        "n_total": total,
    }


def analyze(snapshot_dir: Path) -> dict:
    paths = load_snapshot_paths(snapshot_dir)

    # POST-review (current state on disk).
    res_post = compute_per_reach_confusion(
        gt_dir=paths.gt_dir,
        algo_dir=paths.algo_outputs_dir,
        reaches_dir=paths.algo_outputs_dir,
        video_ids=paths.video_ids,
    )

    # PRE-review: build a temp dir of reverted outcome JSONs, run the
    # same matcher against it. Tempdir is cleaned up after.
    pre_dir = _build_pre_review_dir(paths.algo_outputs_dir)
    try:
        res_pre = compute_per_reach_confusion(
            gt_dir=paths.gt_dir,
            algo_dir=pre_dir,
            reaches_dir=pre_dir,
            video_ids=paths.video_ids,
        )
    finally:
        shutil.rmtree(pre_dir, ignore_errors=True)

    cm_post = res_post["confusion_matrix"]
    cm_pre = res_pre["confusion_matrix"]

    # Per-reach "triage flows" surface for the legend.
    triage_flows_post = {k: v for k, v in cm_post.items()
                         if k.split("__")[1] == "triaged"}
    triage_count_post = sum(triage_flows_post.values())
    triage_flows_pre = {k: v for k, v in cm_pre.items()
                        if k.split("__")[1] == "triaged"}
    triage_count_pre = sum(triage_flows_pre.values())

    pre_summary = _summarize(cm_pre)
    post_summary = _summarize(cm_post)

    out = {
        "n_videos": len(paths.video_ids),
        "n_reaches_universe": res_post["n_reaches_universe"],
        "triage_count": triage_count_post,
        "outcome_label": {
            # Back-compat top-level keys mapped to post-review.
            "confusion_matrix": cm_post,
            "per_class": res_post["per_class"],
            "strict_accuracy": post_summary["accuracy"],
            "committed_accuracy": post_summary["accuracy"],
            "abstention_rate": (
                (triage_count_post / post_summary["n_total"])
                if post_summary["n_total"] else 0.0
            ),
            # Two-level reporting consumed by run_sankey for the two-panel render.
            "pre_review": pre_summary,
            "post_review": post_summary,
        },
        "triage_resolution": {
            "n_triaged_pre_review": triage_count_pre,
            "n_resolved_from_gt": max(0, triage_count_pre - triage_count_post),
            "n_resolved_from_human": 0,  # napari clearing isn't tracked at reach level
            "n_still_triaged_post_review": triage_count_post,
            "resolution_rate": (
                ((triage_count_pre - triage_count_post) / triage_count_pre)
                if triage_count_pre else None
            ),
        },
        "directional_summary": {
            "by_gt": _by_side(cm_post, side=0),
            "by_algo": _by_side(cm_post, side=1),
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
    pre = res["outcome_label"]["pre_review"]
    post = res["outcome_label"]["post_review"]
    print(f"n_reaches_universe={res['n_reaches_universe']}")
    if pre['accuracy'] is not None:
        print(f"  pre-review:  acc={pre['accuracy']:.3f} "
              f"({pre['n_correct']}/{pre['n_total']}), triaged={pre['n_triaged']}")
    if post['accuracy'] is not None:
        print(f"  post-review: acc={post['accuracy']:.3f} "
              f"({post['n_correct']}/{post['n_total']}), triaged={post['n_triaged']}")
