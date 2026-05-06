"""
Reach detection analyzer (algo 2).

Matches algo reaches to GT reaches with TP iff |start_delta| <= 2 AND
span tolerance, per `feedback_reach_outcome_evaluation_format.md`.

Reads:
  <snapshot>/algo_outputs/{video}_reaches.json
  <gt_dir>/{video}_unified_ground_truth.json (reaches block)

Writes:
  <snapshot>/metrics/reach_detection_scalars.json
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from mousereach.improvement.lib.inputs import (
    load_snapshot_paths, load_algo_reaches, load_gt_reaches, write_scalars,
)
from .schema import ReachMatch, ReachDetectionScalars

START_TOL = 2
SPAN_TOL_REL = 0.5
SPAN_TOL_ABS = 5


def _iter_algo_reaches(algo_data: dict) -> List[dict]:
    out = []
    for sg in algo_data.get("segments", []) or []:
        for r in sg.get("reaches", []) or []:
            rr = dict(r)
            rr["_seg_num"] = sg.get("segment_num")
            out.append(rr)
    return out


def _match(algo: List[dict], gt: List[dict],
           start_tol: int = START_TOL,
           span_tol_rel: float = SPAN_TOL_REL,
           span_tol_abs: int = SPAN_TOL_ABS) -> List[tuple]:
    """Greedy nearest-Δstart matching with TP iff start AND span pass.
    Returns list of (status, gt_idx, algo_idx, start_delta, span_delta).
    """
    candidates = []
    for gi, g in enumerate(gt):
        gs = g.get("start_frame")
        ge = g.get("end_frame")
        if gs is None or ge is None:
            continue
        gspan = ge - gs + 1
        for ai, a in enumerate(algo):
            as_ = a.get("start_frame")
            ae = a.get("end_frame")
            if as_ is None or ae is None:
                continue
            d_start = abs(int(as_) - int(gs))
            if d_start > start_tol:
                continue
            aspan = ae - as_ + 1
            tol = max(span_tol_rel * gspan, span_tol_abs)
            d_span = abs(aspan - gspan)
            if d_span > tol:
                continue
            candidates.append((d_start, d_span, gi, ai))

    candidates.sort(key=lambda x: (x[0], x[1]))
    matched_g, matched_a = set(), set()
    out = []
    for d_start, d_span, gi, ai in candidates:
        if gi in matched_g or ai in matched_a:
            continue
        matched_g.add(gi); matched_a.add(ai)
        gs = gt[gi]["start_frame"]; ge = gt[gi]["end_frame"]
        as_ = algo[ai]["start_frame"]; ae = algo[ai]["end_frame"]
        out.append(("tp", gi, ai,
                    int(as_) - int(gs),
                    int(ae - as_ + 1) - int(ge - gs + 1)))
    for gi in range(len(gt)):
        if gi not in matched_g and gt[gi].get("start_frame") is not None:
            out.append(("fn", gi, -1, 0, 0))
    for ai in range(len(algo)):
        if ai not in matched_a and algo[ai].get("start_frame") is not None:
            out.append(("fp", -1, ai, 0, 0))
    return out


def analyze(snapshot_dir: Path) -> dict:
    paths = load_snapshot_paths(snapshot_dir)
    matches: List[ReachMatch] = []
    triage_count = 0

    for vid in paths.video_ids:
        algo_data = load_algo_reaches(paths.algo_outputs_dir, vid)
        algo = _iter_algo_reaches(algo_data)
        gt = load_gt_reaches(paths.gt_dir, vid)

        # Triage from algo: any reach with triaged=True or triage_reason set
        for r in algo:
            if r.get("triaged") or r.get("triage_reason"):
                triage_count += 1

        for status, gi, ai, sd, spd in _match(algo, gt):
            mr = ReachMatch(video_id=vid, status=status)
            if gi >= 0:
                mr.gt_start = int(gt[gi]["start_frame"])
                mr.gt_end = int(gt[gi]["end_frame"])
            if ai >= 0:
                mr.algo_start = int(algo[ai]["start_frame"])
                mr.algo_end = int(algo[ai]["end_frame"])
            if status == "tp":
                mr.start_delta = sd
                mr.span_delta = spd
            matches.append(mr)

    sd = np.array([m.start_delta for m in matches if m.status == "tp"], dtype=int) \
        if matches else np.array([], dtype=int)
    spd = np.array([m.span_delta for m in matches if m.status == "tp"], dtype=int) \
        if matches else np.array([], dtype=int)

    def pct(a, q):
        return int(np.percentile(a, q)) if len(a) else None

    out = ReachDetectionScalars(
        n_videos=len(paths.video_ids),
        n_tp=int(sum(1 for m in matches if m.status == "tp")),
        n_fp=int(sum(1 for m in matches if m.status == "fp")),
        n_fn=int(sum(1 for m in matches if m.status == "fn")),
        triage_count=triage_count,
        start_delta_median=int(np.median(sd)) if len(sd) else None,
        start_delta_abs_median=int(np.median(np.abs(sd))) if len(sd) else None,
        start_delta_p10=pct(sd, 10), start_delta_p90=pct(sd, 90),
        start_delta_min=int(sd.min()) if len(sd) else None,
        start_delta_max=int(sd.max()) if len(sd) else None,
        span_delta_median=int(np.median(spd)) if len(spd) else None,
        span_delta_abs_median=int(np.median(np.abs(spd))) if len(spd) else None,
        span_delta_p10=pct(spd, 10), span_delta_p90=pct(spd, 90),
        span_delta_min=int(spd.min()) if len(spd) else None,
        span_delta_max=int(spd.max()) if len(spd) else None,
        matches=[m.to_dict() for m in matches],
    ).to_dict()
    write_scalars(paths.metrics_dir, out, "reach_detection_scalars.json")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.reach_detection.analyze <snapshot_dir>")
        sys.exit(1)
    res = analyze(Path(sys.argv[1]))
    print(f"TP={res['n_tp']}  FP={res['n_fp']}  FN={res['n_fn']}  triage={res['triage_count']}")
