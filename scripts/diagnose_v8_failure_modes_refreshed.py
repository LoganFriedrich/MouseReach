"""
Diagnostic: failure-mode breakdown on v8.0.0 production model, refreshed.

The 2026-05-04 inspection (snapshot
v8.0.0_dev_failure_mode_breakdown_on_bsw_w08) categorized FNs and FPs
on v8 dev with BSW. v8.0.0 production has BSW b=1/w=0.8 INTEGRATED -- the
analyses are essentially on the same model on the calibration corpus,
but the generalization corpus has never been categorized this way. This
runner refreshes both, side-by-side, so the failure-mode mixture is
visible on the current production model on both corpora.

================================================================
METHOD
================================================================

Same matching primitive as the criterion reconciliation runner. For each
corpus:
  1. Reconstruct per-video (algo_reaches, gt_reaches) from the snapshot.
  2. Run strict matching (start_tol=2, span_tol=max(0.5*gspan, 5)).
  3. Categorize every FN and FP using the canonical heuristic below.

FN categories (per unmatched GT reach):
  - model_miss:      no algo reach within MODEL_MISS_WINDOW frames of GT start
  - fragmented:      2+ algo reaches overlap GT [start, end] window
  - tol_miss_start:  nearest algo by start has |start_delta| > 2 AND span OK
  - tol_miss_span:   nearest algo has |start_delta| <= 2 AND span fails
  - tol_miss_both:   nearest algo fails BOTH start and span tolerance

FP categories (per unmatched algo reach):
  - within_gt:         algo_start_frame inside any GT [start, end] window
  - split_twin:        nearest GT (by start, within +/- 10f) IS matched to
                       some other algo reach -> this FP is the leftover
                       fragment of a split
  - near_unmatched_gt: nearest GT within +/-10f, GT is itself an FN
                       (algo and GT are in the same place but failed
                       both the start and span checks)
  - random:            no GT within +/-RANDOM_WINDOW frames

================================================================
WHAT WE CONCLUDE
================================================================

The refreshed mixture answers: where is v8.0.0 actually failing on the
current corpora? Two specific questions:

1. Has the mixture shifted vs the 05-04 numbers? If model_miss is still
   ~5% of FN, the model is still detecting nearly every reach and the
   problem is boundary precision (where most of our experiments have
   targeted). If model_miss has grown, the model has regressed on
   detection.

2. Does the generalization corpus show a different mixture? If so, the
   v8.0.0 model's failure modes are corpus-specific (worth understanding
   why), not a single uniform behavior. This informs whether a single
   post-filter or feature change can address both corpora.

================================================================
OUTPUTS
================================================================

  Improvement_Snapshots/reach_detection/v8.0.0_dev_failure_modes_refreshed/
    metrics/
      failure_modes.json     # side-by-side counts + percentages
      fn_examples.json       # 20 random FNs per category per corpus, with
                              # video_id + gt_start + nearest algo, for
                              # case-by-case inspection
      fp_examples.json       # same for FPs
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CAL_SNAPSHOT_JSON = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_boundary_sample_weight_b1_w0.8"
    r"\metrics\loocv_aggregate.json"
)

GEN_SNAPSHOT_JSON = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_generalization_20video"
    r"\metrics\reach_detection_scalars.json"
)

OUTPUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_failure_modes_refreshed_v8_only"
)

# Tolerances
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5
MODEL_MISS_WINDOW = 30        # FN: no algo within this many frames -> model_miss
RANDOM_WINDOW = 30            # FP: no GT within this many frames -> random
NEAR_WINDOW = 10              # FP near-GT scope (for split_twin / near_unmatched_gt)
SEED_FOR_EXAMPLES = 13
N_EXAMPLES_PER_CATEGORY = 20


# ---------------------------------------------------------------------------
# Snapshot loaders (same shape as reconciliation runner -- duplicate
# rather than refactor to keep each runner self-contained)
# ---------------------------------------------------------------------------

def _records_from_loocv(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in data["raw_results"]:
        out.append({
            "video_id": r["video_id"],
            "algo_start": r.get("algo_start_frame", -1),
            "algo_end": r.get("algo_end_frame", -1),
            "gt_start": r.get("gt_start_frame", -1),
            "gt_end": r.get("gt_end_frame", -1),
        })
    return out


def _records_from_generalization(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in data["matches"]:
        out.append({
            "video_id": r["video_id"],
            "algo_start": r.get("algo_start", -1),
            "algo_end": r.get("algo_end", -1),
            "gt_start": r.get("gt_start", -1),
            "gt_end": r.get("gt_end", -1),
        })
    return out


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
# Helpers
# ---------------------------------------------------------------------------

def _nearest_algo_by_start(gt: Reach, algos: List[Reach]
                           ) -> Optional[Tuple[Reach, int]]:
    """Return (algo, abs_start_delta) for the algo whose start is closest
    to gt.start. None if algos is empty.
    """
    best = None
    best_d = None
    for a in algos:
        d = abs(a.start_frame - gt.start_frame)
        if best_d is None or d < best_d:
            best = a
            best_d = d
    if best is None:
        return None
    return best, best_d


def _algos_overlapping_window(algos: List[Reach], a: int, b: int
                              ) -> List[Reach]:
    """Algo reaches that have any frame overlap with [a, b] inclusive."""
    return [r for r in algos
            if not (r.end_frame < a or r.start_frame > b)]


def _algo_span(a: Reach) -> int:
    return a.end_frame - a.start_frame + 1


def _passes_span_tol(a_span: int, g_span: int) -> bool:
    tol = max(STRICT_SPAN_TOL_REL * g_span, STRICT_SPAN_TOL_ABS)
    return abs(a_span - g_span) <= tol


def categorize_fn(gt: Reach, algos: List[Reach]) -> Tuple[str, Dict[str, Any]]:
    """Categorize one unmatched GT reach.  Returns (category, info dict)."""
    overlapping = _algos_overlapping_window(algos, gt.start_frame, gt.end_frame)
    if len(overlapping) >= 2:
        return "fragmented", {
            "n_overlapping_algo": len(overlapping),
            "overlapping_algo_spans": [(r.start_frame, r.end_frame) for r in overlapping],
        }

    near = _nearest_algo_by_start(gt, algos)
    if near is None or near[1] > MODEL_MISS_WINDOW:
        return "model_miss", {
            "nearest_algo_start": near[0].start_frame if near else None,
            "nearest_algo_end": near[0].end_frame if near else None,
            "abs_start_delta": near[1] if near else None,
        }

    a, abs_sd = near
    start_ok = abs_sd <= STRICT_START_TOL
    span_ok = _passes_span_tol(_algo_span(a), _algo_span(gt))

    info = {
        "nearest_algo_start": a.start_frame,
        "nearest_algo_end": a.end_frame,
        "abs_start_delta": abs_sd,
        "signed_span_delta": _algo_span(a) - _algo_span(gt),
        "start_ok": bool(start_ok),
        "span_ok": bool(span_ok),
    }
    if not start_ok and not span_ok:
        return "tol_miss_both", info
    if not start_ok:
        return "tol_miss_start", info
    return "tol_miss_span", info


def categorize_fp(algo: Reach, gts: List[Reach], matched_gt_ids: set
                  ) -> Tuple[str, Dict[str, Any]]:
    """Categorize one unmatched algo reach. matched_gt_ids = set of GT
    indices already matched to some other algo reach.
    """
    for g in gts:
        if g.start_frame <= algo.start_frame <= g.end_frame:
            return "within_gt", {
                "containing_gt_start": g.start_frame,
                "containing_gt_end": g.end_frame,
                "containing_gt_matched": g.index in matched_gt_ids,
            }

    # Nearest GT by start
    near = None
    near_d = None
    for g in gts:
        d = abs(g.start_frame - algo.start_frame)
        if near_d is None or d < near_d:
            near = g; near_d = d

    if near is None or near_d > RANDOM_WINDOW:
        return "random", {
            "nearest_gt_start": near.start_frame if near else None,
            "abs_start_delta": near_d if near else None,
        }

    if near_d <= NEAR_WINDOW and near.index in matched_gt_ids:
        return "split_twin", {
            "nearest_gt_start": near.start_frame,
            "nearest_gt_end": near.end_frame,
            "abs_start_delta": near_d,
        }

    if near_d <= NEAR_WINDOW and near.index not in matched_gt_ids:
        return "near_unmatched_gt", {
            "nearest_gt_start": near.start_frame,
            "nearest_gt_end": near.end_frame,
            "abs_start_delta": near_d,
        }

    # Between NEAR_WINDOW and RANDOM_WINDOW
    return "other_near", {
        "nearest_gt_start": near.start_frame,
        "abs_start_delta": near_d,
    }


# ---------------------------------------------------------------------------
# Per-corpus analysis
# ---------------------------------------------------------------------------

def analyze_corpus(per_video: Dict[str, Tuple[List[Reach], List[Reach]]]
                   ) -> Dict[str, Any]:
    fn_counts: Dict[str, int] = defaultdict(int)
    fp_counts: Dict[str, int] = defaultdict(int)
    fn_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    fp_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_tp = total_fp = total_fn = 0
    per_video_summary: Dict[str, Dict[str, Any]] = {}

    for vid, (algos, gts) in per_video.items():
        results = match_reaches(
            algos, gts, strict=True,
            strict_start_tol=STRICT_START_TOL,
            strict_span_tol_rel=STRICT_SPAN_TOL_REL,
            strict_span_tol_abs=STRICT_SPAN_TOL_ABS,
        )
        v_tp = v_fp = v_fn = 0
        matched_gt_ids = {r.gt_reach_index for r in results
                          if r.status == "matched" and r.gt_reach_index is not None}
        for r in results:
            if r.status == "matched":
                v_tp += 1
            elif r.status == "fn":
                v_fn += 1
                gt = gts[r.gt_reach_index]
                cat, info = categorize_fn(gt, algos)
                fn_counts[cat] += 1
                fn_examples[cat].append({
                    "video_id": vid,
                    "gt_start": gt.start_frame,
                    "gt_end": gt.end_frame,
                    **info,
                })
            elif r.status == "fp":
                v_fp += 1
                a = algos[r.algo_reach_index]
                cat, info = categorize_fp(a, gts, matched_gt_ids)
                fp_counts[cat] += 1
                fp_examples[cat].append({
                    "video_id": vid,
                    "algo_start": a.start_frame,
                    "algo_end": a.end_frame,
                    **info,
                })
        per_video_summary[vid] = {
            "n_tp": v_tp, "n_fp": v_fp, "n_fn": v_fn,
            "n_gt": len(gts), "n_algo": len(algos),
        }
        total_tp += v_tp; total_fp += v_fp; total_fn += v_fn

    # Convert to plain dicts and percentages
    fn_total = sum(fn_counts.values())
    fp_total = sum(fp_counts.values())
    fn_summary = {
        cat: {"count": fn_counts[cat],
              "pct_of_fn": (100.0 * fn_counts[cat] / fn_total) if fn_total else None}
        for cat in sorted(fn_counts.keys())
    }
    fp_summary = {
        cat: {"count": fp_counts[cat],
              "pct_of_fp": (100.0 * fp_counts[cat] / fp_total) if fp_total else None}
        for cat in sorted(fp_counts.keys())
    }

    # Sample examples (seeded RNG so reruns produce the same selection)
    rng = np.random.default_rng(SEED_FOR_EXAMPLES)
    def _sample(d: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        out = {}
        for cat, lst in d.items():
            if len(lst) <= N_EXAMPLES_PER_CATEGORY:
                out[cat] = lst
            else:
                idx = rng.choice(len(lst), size=N_EXAMPLES_PER_CATEGORY,
                                 replace=False)
                out[cat] = [lst[i] for i in idx]
        return out

    return {
        "totals": {"n_tp": total_tp, "n_fp": total_fp, "n_fn": total_fn,
                   "n_videos": len(per_video)},
        "fn_categories": fn_summary,
        "fp_categories": fp_summary,
        "fn_examples": _sample(fn_examples),
        "fp_examples": _sample(fp_examples),
        "per_video": per_video_summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DIAGNOSTIC: failure-mode breakdown on v8.0.0 (BSW w=0.8), refreshed")
    print("  Side-by-side: calibration LOOCV vs 2026-05-11 generalization")
    print("=" * 70)
    print()

    print(f"Loading calibration LOOCV: {CAL_SNAPSHOT_JSON}", flush=True)
    if not CAL_SNAPSHOT_JSON.exists():
        raise FileNotFoundError(f"Calibration snapshot not found: {CAL_SNAPSHOT_JSON}")
    cal_records = _records_from_loocv(CAL_SNAPSHOT_JSON)
    cal_per_video = reconstruct_per_video(cal_records)
    print(f"  {len(cal_per_video)} videos", flush=True)

    print(f"Loading generalization: {GEN_SNAPSHOT_JSON}", flush=True)
    if not GEN_SNAPSHOT_JSON.exists():
        raise FileNotFoundError(f"Generalization snapshot not found: {GEN_SNAPSHOT_JSON}")
    gen_records = _records_from_generalization(GEN_SNAPSHOT_JSON)
    gen_per_video = reconstruct_per_video(gen_records)
    print(f"  {len(gen_per_video)} videos", flush=True)
    print()

    print("Analyzing calibration ...", flush=True)
    cal_result = analyze_corpus(cal_per_video)
    print("Analyzing generalization ...", flush=True)
    gen_result = analyze_corpus(gen_per_video)
    print()

    # -------- Print side-by-side summary -----------------------------------
    print("=" * 70)
    print("FN CATEGORY BREAKDOWN")
    print("=" * 70)
    all_fn_cats = sorted(set(cal_result["fn_categories"]) | set(gen_result["fn_categories"]))
    print(f"  {'category':<24} {'cal count':>10} {'cal %':>8}   {'gen count':>10} {'gen %':>8}")
    for cat in all_fn_cats:
        c = cal_result["fn_categories"].get(cat, {"count": 0, "pct_of_fn": 0})
        g = gen_result["fn_categories"].get(cat, {"count": 0, "pct_of_fn": 0})
        c_pct = c.get("pct_of_fn") or 0
        g_pct = g.get("pct_of_fn") or 0
        print(f"  {cat:<24} {c['count']:>10} {c_pct:>7.1f}%   {g['count']:>10} {g_pct:>7.1f}%")
    print(f"  {'TOTAL FN':<24} {cal_result['totals']['n_fn']:>10}            "
          f"{gen_result['totals']['n_fn']:>10}")
    print()

    print("=" * 70)
    print("FP CATEGORY BREAKDOWN")
    print("=" * 70)
    all_fp_cats = sorted(set(cal_result["fp_categories"]) | set(gen_result["fp_categories"]))
    print(f"  {'category':<24} {'cal count':>10} {'cal %':>8}   {'gen count':>10} {'gen %':>8}")
    for cat in all_fp_cats:
        c = cal_result["fp_categories"].get(cat, {"count": 0, "pct_of_fp": 0})
        g = gen_result["fp_categories"].get(cat, {"count": 0, "pct_of_fp": 0})
        c_pct = c.get("pct_of_fp") or 0
        g_pct = g.get("pct_of_fp") or 0
        print(f"  {cat:<24} {c['count']:>10} {c_pct:>7.1f}%   {g['count']:>10} {g_pct:>7.1f}%")
    print(f"  {'TOTAL FP':<24} {cal_result['totals']['n_fp']:>10}            "
          f"{gen_result['totals']['n_fp']:>10}")
    print()

    # -------- Persist -------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Strip examples out into separate files for readability
    cal_top = {k: v for k, v in cal_result.items() if not k.endswith("_examples")}
    gen_top = {k: v for k, v in gen_result.items() if not k.endswith("_examples")}

    (metrics_dir / "failure_modes.json").write_text(
        json.dumps({
            "criterion": {
                "strict_start_tol": STRICT_START_TOL,
                "strict_span_tol_rel": STRICT_SPAN_TOL_REL,
                "strict_span_tol_abs": STRICT_SPAN_TOL_ABS,
            },
            "category_thresholds": {
                "model_miss_window": MODEL_MISS_WINDOW,
                "random_window": RANDOM_WINDOW,
                "near_window": NEAR_WINDOW,
            },
            "calibration": cal_top,
            "generalization": gen_top,
            "inputs": {
                "calibration_snapshot": str(CAL_SNAPSHOT_JSON),
                "generalization_snapshot": str(GEN_SNAPSHOT_JSON),
            },
        }, indent=2), encoding="utf-8")

    (metrics_dir / "fn_examples.json").write_text(
        json.dumps({
            "calibration": cal_result["fn_examples"],
            "generalization": gen_result["fn_examples"],
        }, indent=2), encoding="utf-8")

    (metrics_dir / "fp_examples.json").write_text(
        json.dumps({
            "calibration": cal_result["fp_examples"],
            "generalization": gen_result["fp_examples"],
        }, indent=2), encoding="utf-8")

    print(f"Wrote: {metrics_dir / 'failure_modes.json'}")
    print(f"Wrote: {metrics_dir / 'fn_examples.json'}")
    print(f"Wrote: {metrics_dir / 'fp_examples.json'}")
    print()
    print("NEXT: human reads the FN and FP breakdown side-by-side, compares")
    print("calibration vs generalization mixtures, decides whether v8.0.0's")
    print("failure modes are corpus-stable or corpus-specific. The example")
    print("files give 20 random cases per category to spot-check by eye.")


if __name__ == "__main__":
    main()
