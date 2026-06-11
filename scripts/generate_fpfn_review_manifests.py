"""
Generate per-video FP/FN review manifests for the FP/FN reach review widget.

Schema documented in SCHEMA.md (provided by the widget author). One JSON
per video, grouped by corpus. The widget consumes these to display
TP/FP/FN markers during video playback.

================================================================
SOURCES (v8.0.0 production model on both corpora)
================================================================

  Calibration (LOOCV on the 16-video exhaustive train pool):
    Improvement_Snapshots/reach_detection/
      v8.0.0_dev_boundary_sample_weight_b1_w0.8/
      metrics/loocv_aggregate.json
    Schema: raw_results: [{video_id, status (tp/fp/fn), gt_index,
            algo_index, algo_start_frame, algo_end_frame,
            gt_start_frame, gt_end_frame, start_delta, span_delta}]

  Generalization (v8.0.0 inference on 20 held-out videos, run 2026-05-18):
    Improvement_Snapshots/reach_detection/
      v8.0.0_generalization_20video/metrics/reach_detection_scalars.json
    Schema: matches: [{video_id, status (matched/fp/fn), gt_start,
            gt_end, algo_start, algo_end, start_delta, span_delta}]

================================================================
OUTPUTS
================================================================

  Y:/2_Connectome/Behavior/MouseReach_Improvement/fpfn_review_manifests/
    calibration_loocv/
      <video_id>.json
    holdout_2026_05_11/
      <video_id>.json

================================================================
NOTES
================================================================

  - Categories computed inline (same logic as
    diagnose_v8_failure_modes_refreshed.py) so each event carries a
    failure-mode label the widget can show in tooltips.
  - video_path, n_frames, fps are OMITTED -- the widget has a file
    picker and reads these from the loaded video.
  - matching_criterion is "strict_start2_span" for both corpora (same
    rule the v8 production eval uses).
  - The LOOCV source uses lowercase status "tp/fp/fn"; the generalization
    source uses "matched/fp/fn". Both are normalized to uppercase
    "TP/FP/FN" per the schema.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CAL_SOURCE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_boundary_sample_weight_b1_w0.8"
    r"\metrics\loocv_aggregate.json"
)
CAL_SNAPSHOT_NAME = "v8.0.0_dev_boundary_sample_weight_b1_w0.8"

GEN_SOURCE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_generalization_20video"
    r"\metrics\reach_detection_scalars.json"
)
GEN_SNAPSHOT_NAME = "v8.0.0_generalization_20video"

OUTPUT_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests"
)

# Fixed across all manifests
DETECTOR_VERSION = "v8.0.0_bsw_w0.8"
MATCHING_CRITERION = "strict_start2_span"

# Matching tolerances (same as production v8 eval)
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5

# Category thresholds (same as diagnose_v8_failure_modes_refreshed.py)
MODEL_MISS_WINDOW = 30
RANDOM_WINDOW = 30
NEAR_WINDOW = 10


# ---------------------------------------------------------------------------
# Record loaders (per-source schema normalization)
# ---------------------------------------------------------------------------

def _normalize_kind(raw: str) -> str:
    """Normalize raw status string to schema's TP/FP/FN."""
    s = raw.lower()
    if s in ("tp", "matched"):
        return "TP"
    if s == "fp":
        return "FP"
    if s == "fn":
        return "FN"
    raise ValueError(f"Unknown status string: {raw!r}")


def _load_calibration_records() -> List[Dict[str, Any]]:
    """Normalize loocv_aggregate.json raw_results into a flat record list."""
    data = json.loads(CAL_SOURCE.read_text(encoding="utf-8"))
    out = []
    for r in data["raw_results"]:
        out.append({
            "video_id": r["video_id"],
            "kind": _normalize_kind(r["status"]),
            "algo_start": r.get("algo_start_frame", -1),
            "algo_end": r.get("algo_end_frame", -1),
            "gt_start": r.get("gt_start_frame", -1),
            "gt_end": r.get("gt_end_frame", -1),
            "start_delta": r.get("start_delta"),
            "span_delta": r.get("span_delta"),
        })
    return out


def _load_generalization_records() -> List[Dict[str, Any]]:
    """Normalize reach_detection_scalars.json matches into a flat record list."""
    data = json.loads(GEN_SOURCE.read_text(encoding="utf-8"))
    out = []
    for r in data["matches"]:
        out.append({
            "video_id": r["video_id"],
            "kind": _normalize_kind(r["status"]),
            "algo_start": r.get("algo_start", -1),
            "algo_end": r.get("algo_end", -1),
            "gt_start": r.get("gt_start", -1),
            "gt_end": r.get("gt_end", -1),
            "start_delta": r.get("start_delta"),
            "span_delta": r.get("span_delta"),
        })
    return out


# ---------------------------------------------------------------------------
# Per-video reach lists (dedup by (start, end))
# ---------------------------------------------------------------------------

def reconstruct_per_video(records: List[Dict[str, Any]]
                          ) -> Dict[str, Tuple[List[Reach], List[Reach]]]:
    algo_by_vid: Dict[str, set] = defaultdict(set)
    gt_by_vid: Dict[str, set] = defaultdict(set)
    for r in records:
        vid = r["video_id"]
        if r["algo_start"] is not None and r["algo_start"] >= 0:
            algo_by_vid[vid].add((int(r["algo_start"]), int(r["algo_end"])))
        if r["gt_start"] is not None and r["gt_start"] >= 0:
            gt_by_vid[vid].add((int(r["gt_start"]), int(r["gt_end"])))

    out: Dict[str, Tuple[List[Reach], List[Reach]]] = {}
    for vid in sorted(set(algo_by_vid) | set(gt_by_vid)):
        algo_sorted = sorted(algo_by_vid.get(vid, set()))
        gt_sorted = sorted(gt_by_vid.get(vid, set()))
        algo_reaches = [Reach(start_frame=s, end_frame=e, index=i)
                        for i, (s, e) in enumerate(algo_sorted)]
        gt_reaches = [Reach(start_frame=s, end_frame=e, index=i)
                      for i, (s, e) in enumerate(gt_sorted)]
        out[vid] = (algo_reaches, gt_reaches)
    return out


# ---------------------------------------------------------------------------
# Category logic (mirrors diagnose_v8_failure_modes_refreshed.py)
# ---------------------------------------------------------------------------

def _algos_overlapping_window(algos: List[Reach], a: int, b: int
                              ) -> List[Reach]:
    return [r for r in algos
            if not (r.end_frame < a or r.start_frame > b)]


def _passes_span_tol(a_span: int, g_span: int) -> bool:
    tol = max(STRICT_SPAN_TOL_REL * g_span, STRICT_SPAN_TOL_ABS)
    return abs(a_span - g_span) <= tol


def _nearest_by_start(target: int, candidates: List[Reach]
                      ) -> Tuple[Optional[Reach], Optional[int]]:
    best = None; best_d = None
    for c in candidates:
        d = abs(c.start_frame - target)
        if best_d is None or d < best_d:
            best = c; best_d = d
    return best, best_d


def categorize_fn(gt_start: int, gt_end: int,
                  algos: List[Reach]) -> str:
    """Schema-compatible FN category. Mirrors the failure-mode runner."""
    overlapping = _algos_overlapping_window(algos, gt_start, gt_end)
    if len(overlapping) >= 2:
        return "fragmented"

    near, abs_sd = _nearest_by_start(gt_start, algos)
    if near is None or abs_sd > MODEL_MISS_WINDOW:
        return "miss"

    g_span = gt_end - gt_start + 1
    a_span = near.end_frame - near.start_frame + 1
    start_ok = abs_sd <= STRICT_START_TOL
    span_ok = _passes_span_tol(a_span, g_span)

    if not start_ok and not span_ok:
        return "tolerance_miss_both"
    if not start_ok:
        return "tolerance_miss_start"
    return "tolerance_miss_span"


def categorize_fp(algo_start: int, algo_end: int,
                  gts: List[Reach], matched_gt_keys: set) -> str:
    """Schema-compatible FP category. Mirrors the failure-mode runner."""
    # within_gt: algo start inside any GT window
    for g in gts:
        if g.start_frame <= algo_start <= g.end_frame:
            return "within_gt"

    near, abs_sd = _nearest_by_start(algo_start, gts)
    if near is None or abs_sd > RANDOM_WINDOW:
        return "phantom"

    if abs_sd <= NEAR_WINDOW:
        key = (near.start_frame, near.end_frame)
        if key in matched_gt_keys:
            return "split_twin"
        return "tolerance_miss"

    return "other_near"


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------

def _build_event(rec: Dict[str, Any],
                 algos: List[Reach], gts: List[Reach],
                 matched_gt_keys: set) -> Dict[str, Any]:
    """One event in schema form."""
    kind = rec["kind"]
    if kind == "TP":
        return {
            "kind": "TP",
            "detector": {"start": int(rec["algo_start"]),
                         "end": int(rec["algo_end"])},
            "gt": {"start": int(rec["gt_start"]),
                   "end": int(rec["gt_end"])},
            "category": None,
            "start_delta": rec["start_delta"],
            "span_delta": rec["span_delta"],
        }
    if kind == "FP":
        cat = categorize_fp(
            int(rec["algo_start"]), int(rec["algo_end"]),
            gts, matched_gt_keys,
        )
        return {
            "kind": "FP",
            "detector": {"start": int(rec["algo_start"]),
                         "end": int(rec["algo_end"])},
            "gt": None,
            "category": cat,
        }
    if kind == "FN":
        cat = categorize_fn(
            int(rec["gt_start"]), int(rec["gt_end"]),
            algos,
        )
        return {
            "kind": "FN",
            "detector": None,
            "gt": {"start": int(rec["gt_start"]),
                   "end": int(rec["gt_end"])},
            "category": cat,
        }
    raise ValueError(f"Unknown kind: {kind}")


def build_manifests(records: List[Dict[str, Any]],
                    corpus_label: str,
                    snapshot_name: str) -> Dict[str, Dict[str, Any]]:
    """Group records by video_id, produce manifest dicts."""
    per_video_reaches = reconstruct_per_video(records)
    records_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        records_by_video[r["video_id"]].append(r)

    manifests: Dict[str, Dict[str, Any]] = {}
    for video_id, video_records in records_by_video.items():
        algos, gts = per_video_reaches.get(
            video_id, ([], []))

        # Re-match strict to get which GT keys are matched (for split_twin)
        results = match_reaches(
            algos, gts, strict=True,
            strict_start_tol=STRICT_START_TOL,
            strict_span_tol_rel=STRICT_SPAN_TOL_REL,
            strict_span_tol_abs=STRICT_SPAN_TOL_ABS,
        )
        matched_gt_keys = set()
        for r in results:
            if r.status == "matched" and r.gt_start is not None:
                matched_gt_keys.add((int(r.gt_start), int(r.gt_end)))

        events = [
            _build_event(rec, algos, gts, matched_gt_keys)
            for rec in video_records
        ]
        # Sort events by detector start (or gt start for FNs)
        def _sort_key(ev):
            if ev["kind"] == "FN":
                return ev["gt"]["start"]
            return ev["detector"]["start"]
        events.sort(key=_sort_key)

        manifests[video_id] = {
            "video_id": video_id,
            "detector_version": DETECTOR_VERSION,
            "snapshot": snapshot_name,
            "corpus": corpus_label,
            "matching_criterion": MATCHING_CRITERION,
            "events": events,
        }

    return manifests


def write_manifests(manifests: Dict[str, Dict[str, Any]], corpus_label: str
                    ) -> int:
    out_dir = OUTPUT_ROOT / corpus_label
    out_dir.mkdir(parents=True, exist_ok=True)
    for video_id, manifest in manifests.items():
        out_path = out_dir / f"{video_id}.json"
        out_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    return len(manifests)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_corpus_summary(manifests: Dict[str, Dict[str, Any]],
                          corpus_label: str) -> None:
    total_tp = total_fp = total_fn = 0
    print(f"  [{corpus_label}] {len(manifests)} videos:")
    for video_id, m in manifests.items():
        n_tp = sum(1 for e in m["events"] if e["kind"] == "TP")
        n_fp = sum(1 for e in m["events"] if e["kind"] == "FP")
        n_fn = sum(1 for e in m["events"] if e["kind"] == "FN")
        total_tp += n_tp; total_fp += n_fp; total_fn += n_fn
        print(f"    {video_id:35} TP={n_tp:>4} FP={n_fp:>4} FN={n_fn:>4}")
    print(f"  totals: TP={total_tp} FP={total_fp} FN={total_fn}")


def main():
    print("=" * 70)
    print("Generate per-video FP/FN review manifests (v8.0.0)")
    print("=" * 70)
    print()

    print(f"Loading calibration source: {CAL_SOURCE.name}")
    if not CAL_SOURCE.exists():
        raise FileNotFoundError(f"Missing: {CAL_SOURCE}")
    cal_records = _load_calibration_records()
    print(f"  {len(cal_records)} records")

    print(f"Loading generalization source: {GEN_SOURCE.name}")
    if not GEN_SOURCE.exists():
        raise FileNotFoundError(f"Missing: {GEN_SOURCE}")
    gen_records = _load_generalization_records()
    print(f"  {len(gen_records)} records")
    print()

    print("Building manifests ...")
    cal_manifests = build_manifests(
        cal_records, "calibration_loocv", CAL_SNAPSHOT_NAME)
    gen_manifests = build_manifests(
        gen_records, "holdout_2026_05_11", GEN_SNAPSHOT_NAME)
    print()

    print("Per-video event counts:")
    _print_corpus_summary(cal_manifests, "calibration_loocv")
    print()
    _print_corpus_summary(gen_manifests, "holdout_2026_05_11")
    print()

    n_cal = write_manifests(cal_manifests, "calibration_loocv")
    n_gen = write_manifests(gen_manifests, "holdout_2026_05_11")
    print(f"Wrote {n_cal} manifests to "
          f"{OUTPUT_ROOT / 'calibration_loocv'}/")
    print(f"Wrote {n_gen} manifests to "
          f"{OUTPUT_ROOT / 'holdout_2026_05_11'}/")


if __name__ == "__main__":
    main()
