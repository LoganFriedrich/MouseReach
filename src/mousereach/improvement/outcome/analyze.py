"""
Outcome analyzer (algo 3) -- per-PELLET (per-segment).

Reads:
  <snapshot>/algo_outputs/{video}_pellet_outcomes.json  (per-segment outcome)
  <gt_dir>/{video}_unified_ground_truth.json           (per-segment GT)

Writes:
  <snapshot>/metrics/outcome_scalars.json
  <snapshot>/metrics/scalars.json  (canonical, read by run_sankey/summary_table)

Per-segment confusion. 4 classes: retrieved, displaced_sa, untouched,
abnormal_exception. displaced_outside is collapsed to displaced_sa.
The algo can also emit "triaged" -- counted as a separate flow.

Two-level reporting
-------------------
The pipeline includes a GT auto-resolve step: when a triaged segment has a
matching unified GT entry, the cascade's flag is lifted and the GT outcome
is stamped in (``triage_cleared=True`` / ``cleared_by="gt_auto_resolve"``).
We report metrics at TWO levels so the algo's in-isolation performance and
its production-pipeline-realistic performance are both visible:

- **pre_review**: the algo's call BEFORE GT auto-resolve fires. For a
  segment that GT auto-resolve cleared, the pre-review call is the
  ``original_outcome`` (the cascade's pre-resolution outcome, typically
  "triaged" / "uncertain") rather than the resolved outcome. This is the
  algo's true in-isolation behavior.
- **post_review**: the algo's call AFTER GT auto-resolve fires. For a
  segment that was auto-resolved, post-review uses the GT-stamped outcome.
  This is what downstream kinematic analysis actually sees.

In production, post_review is the metric that matters; pre_review tells us
how much of the algo's apparent accuracy is "algo got it right" versus
"GT auto-resolve cleaned up after the algo".
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


def _pre_review_outcome(seg: dict) -> str | None:
    """Return the algo's call BEFORE GT auto-resolve fired.

    For segments cleared by gt_auto_resolve, the current ``outcome`` field
    holds the GT-stamped class; the algo's pre-resolution call lives in
    ``original_outcome`` (or, if it equalled GT and so was not preserved,
    the current outcome stands).

    For segments cleared by a human reviewer (napari triage-clearing tool),
    the same convention applies: ``original_outcome`` holds the algo's
    pre-clearing call. Those count as "triaged" pre-review because the
    algo did punt -- a human merely supplied the answer afterward.
    """
    if not seg.get("triage_cleared"):
        return _collapse(seg.get("outcome"))
    orig = seg.get("original_outcome")
    if orig:
        return _collapse(orig)
    # original_outcome was elided because it matched GT, OR the pre-clearing
    # call was literally the "triaged" sentinel. In either case, the algo
    # was not confident at this segment in isolation -- record as triaged.
    return "triaged"


def _accuracy(rows: list, attr: str) -> float | None:
    """Fraction of rows where ``r.gt_outcome == r.<attr>`` (excluding rows
    with no GT label). Returns None if no eligible rows."""
    paired = [r for r in rows if r.gt_outcome]
    if not paired:
        return None
    n_correct = sum(1 for r in paired if getattr(r, attr) == r.gt_outcome)
    return n_correct / len(paired)


def _confusion(rows: list, attr: str) -> Dict[str, int]:
    cm: Dict[str, int] = defaultdict(int)
    for r in rows:
        algo = getattr(r, attr)
        if r.gt_outcome and algo:
            cm[f"{r.gt_outcome}__{algo}"] += 1
    return dict(cm)


def _per_class(rows: list, attr: str) -> Dict[str, dict]:
    per: Dict[str, dict] = {}
    for cls in CLASSES:
        n_gt = sum(1 for r in rows if r.gt_outcome == cls)
        n_algo = sum(1 for r in rows if getattr(r, attr) == cls)
        n_correct = sum(1 for r in rows
                        if r.gt_outcome == cls and getattr(r, attr) == cls)
        precision = (n_correct / n_algo) if n_algo else 0.0
        recall = (n_correct / n_gt) if n_gt else 0.0
        per[cls] = {
            "n_gt": n_gt, "n_algo": n_algo, "n_correct": n_correct,
            "precision": precision, "recall": recall,
        }
    return per


def analyze(snapshot_dir: Path) -> dict:
    paths = load_snapshot_paths(snapshot_dir)
    rows: List[SegmentOutcomeRow] = []
    n_triaged_pre = 0  # algo punted before resolve
    n_resolved_from_gt = 0  # of those, lifted by gt_auto_resolve
    n_resolved_from_human = 0  # of those, lifted by napari triage tool
    n_still_triaged_post = 0  # neither resolved -> downstream still sees triaged

    for vid in paths.video_ids:
        algo_data = load_algo_outcomes(paths.algo_outputs_dir, vid)
        gt_segs, exhaustive = load_gt_segments(paths.gt_dir, vid)
        gt_by = {s["segment_num"]: s for s in gt_segs}

        for seg in algo_data.get("segments", []) or []:
            sn = seg.get("segment_num")
            gt = gt_by.get(sn)
            if gt is None:
                continue
            pre_label = _pre_review_outcome(seg)
            post_label = _collapse(seg.get("outcome"))
            gt_label = _collapse(gt.get("outcome"))

            if pre_label == "triaged":
                n_triaged_pre += 1
                cleared_by = seg.get("cleared_by") if seg.get("triage_cleared") else None
                if cleared_by == "gt_auto_resolve":
                    n_resolved_from_gt += 1
                elif cleared_by:
                    n_resolved_from_human += 1
                else:
                    n_still_triaged_post += 1

            ai = seg.get("interaction_frame")
            gi = gt.get("interaction_frame")
            delta = (int(ai) - int(gi)) if (ai is not None and gi is not None) else None

            row = SegmentOutcomeRow(
                video_id=vid, segment_num=int(sn),
                gt_outcome=gt_label, algo_outcome=post_label,
                gt_interaction_frame=gi, algo_interaction_frame=ai,
                interaction_delta=delta, gt_exhaustive=exhaustive,
            )
            # Stamp pre-review label on the row (attribute, not a constructor
            # arg, to avoid touching the schema and downstream consumers).
            setattr(row, "algo_outcome_pre_review", pre_label)
            rows.append(row)

    # Pre- and post-review confusion matrices + per-class breakdowns.
    confusion_pre = _confusion(rows, "algo_outcome_pre_review")
    confusion_post = _confusion(rows, "algo_outcome")
    per_class_pre = _per_class(rows, "algo_outcome_pre_review")
    per_class_post = _per_class(rows, "algo_outcome")
    acc_pre = _accuracy(rows, "algo_outcome_pre_review")
    acc_post = _accuracy(rows, "algo_outcome")

    n_correct_post = sum(1 for r in rows if r.gt_outcome and r.algo_outcome == r.gt_outcome)
    n_correct_pre = sum(1 for r in rows
                        if r.gt_outcome
                        and getattr(r, "algo_outcome_pre_review") == r.gt_outcome)

    out = OutcomeScalars(
        n_videos=len(paths.video_ids),
        n_segments=len(rows),
        n_correct=n_correct_post,  # back-compat: legacy "n_correct" == post-review
        triage_count=n_triaged_pre,  # back-compat: pre-review triaged count
        confusion_matrix=dict(confusion_post),  # back-compat: post-review
        per_class=per_class_post,  # back-compat: post-review
        rows=[r.to_dict() for r in rows],
    ).to_dict()

    # Augment the row dicts with the pre-review label so notebooks / Sankey
    # readers can render the two-level view from a single artifact.
    for row_dict, row_obj in zip(out["rows"], rows):
        row_dict["algo_outcome_pre_review"] = getattr(row_obj, "algo_outcome_pre_review")

    # Interaction frame summary (median/mean signed delta over rows where
    # both algo and GT have a frame). Consumed by the summary_table header.
    paired_deltas = [r.interaction_delta for r in rows
                     if r.interaction_delta is not None]
    if paired_deltas:
        import statistics as _stats
        abs_deltas = [abs(d) for d in paired_deltas]
        out["interaction_frame"] = {
            "n_paired": len(paired_deltas),
            "median_abs_delta": int(_stats.median(abs_deltas)),
            "mean_signed_delta": sum(paired_deltas) / len(paired_deltas),
        }
    else:
        out["interaction_frame"] = {
            "n_paired": 0,
            "median_abs_delta": None,
            "mean_signed_delta": None,
        }

    out["triage_resolution"] = {
        "n_triaged_pre_review": n_triaged_pre,
        "n_resolved_from_gt": n_resolved_from_gt,
        "n_resolved_from_human": n_resolved_from_human,
        "n_still_triaged_post_review": n_still_triaged_post,
        "resolution_rate": (
            (n_resolved_from_gt + n_resolved_from_human) / n_triaged_pre
            if n_triaged_pre else None
        ),
    }

    # Two-level breakdown -- the headline numbers the user asked for.
    out["outcome_label"] = {
        # Legacy keys preserved for existing readers; mapped to post-review.
        "confusion_matrix": dict(confusion_post),
        "per_class": per_class_post,
        "strict_accuracy": acc_post,
        "committed_accuracy": acc_post,
        "abstention_rate": (n_still_triaged_post / len(rows)) if rows else 0.0,
        # New explicit two-level reporting.
        "pre_review": {
            "confusion_matrix": dict(confusion_pre),
            "per_class": per_class_pre,
            "accuracy": acc_pre,
            "n_correct": n_correct_pre,
            "n_triaged": n_triaged_pre,
        },
        "post_review": {
            "confusion_matrix": dict(confusion_post),
            "per_class": per_class_post,
            "accuracy": acc_post,
            "n_correct": n_correct_post,
            "n_triaged": n_still_triaged_post,
        },
    }
    out["n_segments_paired"] = len(rows)

    write_scalars(paths.metrics_dir, out, "outcome_scalars.json")
    write_scalars(paths.metrics_dir, out, "scalars.json")

    # Per-segment CSV consumed by the interaction-violin + summary-table
    # runners. Column names match the legacy schema those tools expect
    # (`interaction_frame_delta`) plus the new ``algo_outcome_pre_review``.
    import csv
    csv_path = paths.metrics_dir / "outcome_per_segment.csv"
    fieldnames = [
        "video_id", "segment_num",
        "gt_outcome", "algo_outcome", "algo_outcome_pre_review",
        "gt_interaction_frame", "algo_interaction_frame",
        "interaction_frame_delta", "gt_exhaustive",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "video_id": r.video_id,
                "segment_num": r.segment_num,
                "gt_outcome": r.gt_outcome or "",
                "algo_outcome": r.algo_outcome or "",
                "algo_outcome_pre_review": getattr(r, "algo_outcome_pre_review", "") or "",
                "gt_interaction_frame": r.gt_interaction_frame if r.gt_interaction_frame is not None else "",
                "algo_interaction_frame": r.algo_interaction_frame if r.algo_interaction_frame is not None else "",
                "interaction_frame_delta": r.interaction_delta if r.interaction_delta is not None else "",
                "gt_exhaustive": "true" if r.gt_exhaustive else "false",
            })
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.outcome.analyze <snapshot_dir>")
        sys.exit(1)
    res = analyze(Path(sys.argv[1]))
    tr = res["triage_resolution"]
    ol = res["outcome_label"]
    pre = ol["pre_review"]
    post = ol["post_review"]
    n = res["n_segments"]
    print(f"n_segments={n}")
    print(f"  pre-review:  acc={pre['accuracy']:.3f} ({pre['n_correct']}/{n}), "
          f"triaged={pre['n_triaged']}")
    print(f"  post-review: acc={post['accuracy']:.3f} ({post['n_correct']}/{n}), "
          f"triaged={post['n_triaged']}")
    print(f"  resolved from GT: {tr['n_resolved_from_gt']} / {tr['n_triaged_pre_review']}")
