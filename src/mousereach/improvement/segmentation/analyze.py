"""
Segmentation analyzer (algo 1).

Matches algo boundaries to GT boundaries within ±MATCH_TOL frames
(greedy nearest-neighbor). Computes per-pair delta = algo - gt.
Unmatched algo boundaries -> FP. Unmatched GT boundaries -> FN.

Reads:
  <snapshot>/algo_outputs/{video}_segments.json (algo boundaries)
  <gt_dir>/{video}_unified_ground_truth.json (gt boundaries; falls back to
                                              algo boundaries as own GT if
                                              GT lacks a segmentation block)

Writes:
  <snapshot>/metrics/segmentation_scalars.json
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from mousereach.improvement.lib.inputs import (
    load_snapshot_paths, load_algo_boundaries, load_gt_boundaries,
    load_algo_segments, write_scalars,
)
from .schema import BoundaryMatch, SegmentationScalars

MATCH_TOL = 50    # ±frames; permissive so tails are visible on the violin


def match_boundaries(
    algo: Sequence[int], gt: Sequence[int], tol: int = MATCH_TOL,
) -> List[Tuple[str, int, int, int]]:
    """Greedy nearest-neighbor match. Returns list of
    (status, algo_frame, gt_frame, delta) tuples. status in {matched, fp, fn}.
    """
    candidates: List[Tuple[int, int, int]] = []
    for gi, g in enumerate(gt):
        for ai, a in enumerate(algo):
            d = abs(int(a) - int(g))
            if d <= tol:
                candidates.append((d, gi, ai))
    candidates.sort(key=lambda x: x[0])

    matched_g, matched_a = set(), set()
    out: List[Tuple[str, int, int, int]] = []
    for _d, gi, ai in candidates:
        if gi in matched_g or ai in matched_a:
            continue
        matched_g.add(gi)
        matched_a.add(ai)
        delta = int(algo[ai]) - int(gt[gi])
        out.append(("matched", int(algo[ai]), int(gt[gi]), delta))

    for gi, g in enumerate(gt):
        if gi not in matched_g:
            out.append(("fn", -1, int(g), 0))
    for ai, a in enumerate(algo):
        if ai not in matched_a:
            out.append(("fp", int(a), -1, 0))
    return out


def analyze(snapshot_dir: Path) -> dict:
    paths = load_snapshot_paths(snapshot_dir)
    matches: List[BoundaryMatch] = []
    n_with_gt = 0
    triage_count = 0

    for vid in paths.video_ids:
        algo_b = load_algo_boundaries(paths.algo_outputs_dir, vid)
        gt_b = load_gt_boundaries(paths.gt_dir, vid)
        if not gt_b:
            continue
        n_with_gt += 1

        for status, af, gf, delta in match_boundaries(algo_b, gt_b):
            matches.append(BoundaryMatch(
                video_id=vid, status=status,
                algo_frame=af if af >= 0 else None,
                gt_frame=gf if gf >= 0 else None,
                delta=delta if status == "matched" else None,
            ))

        # Triage: algo's own boundary_flags field marks low-confidence boundaries
        algo_seg = load_algo_segments(paths.algo_outputs_dir, vid)
        triage_count += sum(1 for f in (algo_seg.get("boundary_flags", []) or []) if f)

    deltas = np.array([m.delta for m in matches if m.status == "matched"],
                      dtype=int) if matches else np.array([], dtype=int)

    def pct(arr, q):
        return int(np.percentile(arr, q)) if len(arr) else None

    scalars = SegmentationScalars(
        n_videos=len(paths.video_ids),
        n_videos_with_gt_boundaries=n_with_gt,
        n_matched=int(sum(1 for m in matches if m.status == "matched")),
        n_fp=int(sum(1 for m in matches if m.status == "fp")),
        n_fn=int(sum(1 for m in matches if m.status == "fn")),
        delta_median=int(np.median(deltas)) if len(deltas) else None,
        delta_abs_median=int(np.median(np.abs(deltas))) if len(deltas) else None,
        delta_p10=pct(deltas, 10),
        delta_p90=pct(deltas, 90),
        delta_min=int(deltas.min()) if len(deltas) else None,
        delta_max=int(deltas.max()) if len(deltas) else None,
        triage_count=triage_count,
        matches=[m.to_dict() for m in matches],
    )
    out = scalars.to_dict()

    # Back-compat block consumed by the segmentation summary-table runner,
    # which expects scalars["all"] = { n_gt_boundaries, n_algo_boundaries,
    # n_phantom, n_miss } and similar per-subset blocks. Subsets aren't
    # populated yet (the analyzer does not yet stamp boundary-level GT
    # subset tags), so all are aliased to the "all" rollup.
    n_phantom = int(sum(1 for m in matches if m.status == "fp"))
    n_miss = int(sum(1 for m in matches if m.status == "fn"))
    n_matched_local = int(sum(1 for m in matches if m.status == "matched"))
    all_block = {
        "n_gt_boundaries": n_matched_local + n_miss,
        "n_algo_boundaries": n_matched_local + n_phantom,
        "n_phantom": n_phantom,
        "n_miss": n_miss,
    }
    out["all"] = all_block
    # Subset breakdowns (inter_pellet vs endpoint) aren't yet stamped at
    # boundary level. The summary-table runner needs them; until they're
    # implemented the table is skipped (violin still tells the story).

    # Canonical scalars.json the runners read.
    write_scalars(paths.metrics_dir, out, "segmentation_scalars.json")
    write_scalars(paths.metrics_dir, out, "scalars.json")

    # Per-boundary CSV consumed by the violin + summary-table runners.
    # Map status names to what those runners expect: fp -> phantom (algo
    # emitted a boundary GT didn't have), fn -> miss (GT had a boundary
    # algo didn't emit). subset_tag stays "all" for now -- future GT
    # versions can stamp boundary-level subsets and we'll surface them.
    import csv
    status_map = {"matched": "matched", "fp": "phantom", "fn": "miss"}
    deltas_path = paths.metrics_dir / "boundary_deltas.csv"
    with open(deltas_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=[
                "video_id", "status", "signed_delta", "subset_tag",
                "algo_frame", "gt_frame",
            ],
        )
        w.writeheader()
        for m in matches:
            w.writerow({
                "video_id": m.video_id,
                "status": status_map.get(m.status, m.status),
                "signed_delta": m.delta if m.delta is not None else 0,
                "subset_tag": "all",
                "algo_frame": m.algo_frame if m.algo_frame is not None else "",
                "gt_frame": m.gt_frame if m.gt_frame is not None else "",
            })
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.segmentation.analyze <snapshot_dir>")
        sys.exit(1)
    res = analyze(Path(sys.argv[1]))
    print(f"Matched: {res['n_matched']}  FP: {res['n_fp']}  FN: {res['n_fn']}")
    print(f"Delta median: {res['delta_median']}  |median|: {res['delta_abs_median']}")
    print(f"Triage count: {res['triage_count']}")
