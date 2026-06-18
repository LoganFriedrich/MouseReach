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

START_TOL = 4
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

        # Pre-extract usable starts/ends for nearest-neighbor lookup
        algo_starts_ends = [
            (int(a["start_frame"]), int(a["end_frame"]))
            for a in algo if a.get("start_frame") is not None
            and a.get("end_frame") is not None
        ]
        gt_starts_ends = [
            (int(g["start_frame"]), int(g["end_frame"]))
            for g in gt if g.get("start_frame") is not None
            and g.get("end_frame") is not None
        ]

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
            elif status == "fp" and gt_starts_ends:
                # algo reach with no GT match -> nearest GT by start_frame
                a_s, a_e = mr.algo_start, mr.algo_end
                nearest = min(gt_starts_ends, key=lambda t: abs(t[0] - a_s))
                mr.nearest_opp_start_delta = a_s - nearest[0]
                mr.nearest_opp_span_delta = (a_e - a_s + 1) - (nearest[1] - nearest[0] + 1)
            elif status == "fn" and algo_starts_ends:
                # GT reach with no algo match -> nearest algo by start_frame
                g_s, g_e = mr.gt_start, mr.gt_end
                nearest = min(algo_starts_ends, key=lambda t: abs(t[0] - g_s))
                # Sign convention: signed = (gt - algo) so positive = gt later
                mr.nearest_opp_start_delta = g_s - nearest[0]
                mr.nearest_opp_span_delta = (nearest[1] - nearest[0] + 1) - (g_e - g_s + 1)
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
    # Back-compat: summary-table runner expects scalars["total"] with
    # n_fp/n_fn/n_matched/n_gt/n_algo + top-level n_perfect_videos.
    n_matched = out["n_tp"]
    n_fp = out["n_fp"]
    n_fn = out["n_fn"]
    out["total"] = {
        "n_matched": n_matched,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "n_gt": n_matched + n_fn,
        "n_algo": n_matched + n_fp,
    }
    perfect_videos = set(paths.video_ids)
    for m in matches:
        if m.status in ("fp", "fn"):
            perfect_videos.discard(m.video_id)
    out["n_perfect_videos"] = len(perfect_videos)

    write_scalars(paths.metrics_dir, out, "reach_detection_scalars.json")
    write_scalars(paths.metrics_dir, out, "scalars.json")

    # Per-reach CSV consumed by the violin + summary-table runners.
    # Map status: tp -> matched (runner convention). end_delta = algo_end -
    # gt_end (signed). subset_tag stays "all".
    import csv
    status_map = {"tp": "matched", "fp": "fp", "fn": "fn"}
    csv_path = paths.metrics_dir / "reach_matches.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=[
                "video_id", "status", "start_delta", "end_delta",
                "subset_tag", "algo_start", "algo_end", "gt_start", "gt_end",
                "nearest_opp_start_delta", "nearest_opp_span_delta",
            ],
        )
        w.writeheader()
        for m in matches:
            end_delta = None
            if (m.status == "tp" and m.algo_end is not None
                    and m.gt_end is not None):
                end_delta = int(m.algo_end) - int(m.gt_end)
            w.writerow({
                "video_id": m.video_id,
                "status": status_map.get(m.status, m.status),
                "start_delta": m.start_delta if m.start_delta is not None else 0,
                "end_delta": end_delta if end_delta is not None else 0,
                "subset_tag": "all",
                "algo_start": m.algo_start if m.algo_start is not None else "",
                "algo_end": m.algo_end if m.algo_end is not None else "",
                "gt_start": m.gt_start if m.gt_start is not None else "",
                "gt_end": m.gt_end if m.gt_end is not None else "",
                "nearest_opp_start_delta": (
                    m.nearest_opp_start_delta
                    if m.nearest_opp_start_delta is not None else ""
                ),
                "nearest_opp_span_delta": (
                    m.nearest_opp_span_delta
                    if m.nearest_opp_span_delta is not None else ""
                ),
            })
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.reach_detection.analyze <snapshot_dir>")
        sys.exit(1)
    res = analyze(Path(sys.argv[1]))
    print(f"TP={res['n_tp']}  FP={res['n_fp']}  FN={res['n_fn']}  triage={res['triage_count']}")
