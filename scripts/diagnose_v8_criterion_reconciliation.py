"""
Diagnostic: matching-criterion reconciliation between 2026-05-04
calibration finding and 2026-05-11 generalization finding.

The two existing analyses give contradictory accounts of v8's span
behavior on the unmatched FP tail:
  - 05-04 (calibration corpus): algo OVER-EXTENDS past GT_end on 81.8% of
    FPs (the within_gt span-extension pattern, Pattern B).
  - 05-11 (generalization corpus): algo UNDER-EXTENDS, producing spans
    8-13 frames SHORTER than GT on the unmatched tail; only 26% of FPs
    are within_gt.

Both findings are on top of BSW-style v8 models, but the criteria and
corpora differ. This runner isolates which axis drives the contradiction.

================================================================
METHOD
================================================================

1. Load v8.0.0-equivalent outputs from two snapshots:
   a. Calibration LOOCV (BSW b=1 w=0.8) -- cumulative best.
   b. Generalization 20-video set from 2026-05-11.
2. Reconstruct (algo_reaches, gt_reaches) lists per video from the
   stored per-event records in each snapshot's metrics JSON.
3. Re-score each corpus under BOTH criteria via
   mousereach.improvement.reach_detection.metrics.match_reaches:
     - STRICT:     strict=True,  start_tol=2,  span_tol=max(0.5*gspan, 5)
     - PERMISSIVE: strict=False, window=10,    no span check
4. For each cell of {corpus, criterion}, compute:
     - TP / FP / FN counts
     - within_gt fraction of FPs (algo_start inside any GT window)
     - span-direction characterization on TPs:
         median, p10, p90 of (algo_span - gt_span); signed
5. Write 2x2 summary + a per-corpus span-distribution histogram so the
   over/under contradiction is visible directly.

================================================================
WHAT WE CONCLUDE
================================================================

This runner DOES NOT decide an algorithm change. It produces evidence.
The interpretive rules:

- If FP/FN counts flip dramatically between strict and permissive on the
  same corpus (e.g., strict shows 80% FPs as within_gt; permissive shows
  10%), the matching criterion is doing most of the work. The "FP
  problem" is partly an eval-rule problem.
- If counts don't flip but corpora differ, the algo behavior is corpus-
  dependent (data shift between calibration and generalization).
- The signed-span distribution on TPs (matched reaches) is independent
  of the criterion -- it directly answers "does the algo over- or under-
  extend on matched reaches?" If both corpora show median(algo - gt) > 0,
  v8 over-extends in general. If both show < 0, it under-extends. If
  one shows + and the other -, behavior differs by corpus.

The output is read by a human and feeds the next direction decision.

================================================================
INPUTS (verified 2026-05-15 by file inspection)
================================================================

  Calibration snapshot:
    Y:\\2_Connectome\\Behavior\\MouseReach_Improvement\\Improvement_Snapshots\\
      reach_detection\\v8.0.0_dev_boundary_sample_weight_b1_w0.8\\
      metrics\\loocv_aggregate.json
    Schema: raw_results: [{video_id, status, gt_index, algo_index,
            algo_start_frame, algo_end_frame, gt_start_frame, gt_end_frame,
            start_delta, span_delta}, ...]

  Generalization snapshot:
    Y:\\2_Connectome\\Behavior\\MouseReach_Improvement\\Improvement_Snapshots\\
      reach_detection\\generalization_test_2026-05-11\\
      metrics\\reach_detection_scalars.json
    Schema: matches: [{video_id, status, gt_start, gt_end, algo_start,
            algo_end, start_delta, span_delta}, ...]

  Both schemas: status in {tp, matched, fp, fn}. We treat both "tp" and
  "matched" as matched (the generalization JSON uses "matched", the
  LOOCV JSON uses "tp"; the actual TPs from re-matching come from our
  re-run, not from these source labels).

================================================================
OUTPUT
================================================================

  Improvement_Snapshots/reach_detection/v8.0.0_dev_criterion_reconciliation/
    metrics/
      reconciliation.json    # 2x2 grid + characterizations
      raw_per_video.json     # per-video re-matched results for both criteria

A separate RESULTS.md should be written by the human after inspecting the
output. It should lead with a one-line conclusion on which axis (criterion
vs corpus) drives the contradiction, then surface the signed-span
distribution on TPs as the algo-behavior question.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    Reach, ReachMatchResult, match_reaches,
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
    r"\reach_detection\v8.0.0_dev_criterion_reconciliation_v8_only"
)


# ---------------------------------------------------------------------------
# Loaders -- normalize both snapshot schemas into a per-video
# (algo_reaches, gt_reaches) dict.
# ---------------------------------------------------------------------------

def _records_from_loocv(path: Path) -> List[Dict[str, Any]]:
    """Return list of records with keys: video_id, status, algo_start,
    algo_end, gt_start, gt_end (algo or gt may be -1 / missing)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in data["raw_results"]:
        out.append({
            "video_id": r["video_id"],
            "status": r["status"],
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
            "status": r["status"],
            "algo_start": r.get("algo_start", -1),
            "algo_end": r.get("algo_end", -1),
            "gt_start": r.get("gt_start", -1),
            "gt_end": r.get("gt_end", -1),
        })
    return out


def reconstruct_per_video(records: List[Dict[str, Any]]
                          ) -> Dict[str, Tuple[List[Reach], List[Reach]]]:
    """From per-event records, rebuild the algo + GT reach lists per video.

    A reach is uniquely identified by (video_id, start, end). Status
    flags are ignored -- we want the raw inventory of what the algo
    emitted and what the GT contains.
    """
    algo_by_vid: Dict[str, set] = defaultdict(set)
    gt_by_vid: Dict[str, set] = defaultdict(set)
    for r in records:
        vid = r["video_id"]
        a_s, a_e = r["algo_start"], r["algo_end"]
        g_s, g_e = r["gt_start"], r["gt_end"]
        if a_s is not None and a_s >= 0:
            algo_by_vid[vid].add((int(a_s), int(a_e)))
        if g_s is not None and g_s >= 0:
            gt_by_vid[vid].add((int(g_s), int(g_e)))

    out: Dict[str, Tuple[List[Reach], List[Reach]]] = {}
    all_vids = sorted(set(algo_by_vid) | set(gt_by_vid))
    for vid in all_vids:
        algo_sorted = sorted(algo_by_vid.get(vid, set()))
        gt_sorted = sorted(gt_by_vid.get(vid, set()))
        algo_reaches = [Reach(start_frame=s, end_frame=e, index=i)
                        for i, (s, e) in enumerate(algo_sorted)]
        gt_reaches = [Reach(start_frame=s, end_frame=e, index=i)
                      for i, (s, e) in enumerate(gt_sorted)]
        out[vid] = (algo_reaches, gt_reaches)
    return out


# ---------------------------------------------------------------------------
# Per-cell characterization
# ---------------------------------------------------------------------------

def _fp_within_gt(algo: Reach, gt_reaches: List[Reach]) -> bool:
    """True if algo's start_frame falls inside any GT reach window."""
    for g in gt_reaches:
        if g.start_frame <= algo.start_frame <= g.end_frame:
            return True
    return False


def score_corpus(
    per_video: Dict[str, Tuple[List[Reach], List[Reach]]],
    strict: bool,
) -> Dict[str, Any]:
    """Re-match every video under the given criterion; aggregate counts +
    span characterization + within_gt FP fraction.
    """
    n_tp = n_fp = n_fn = 0
    span_deltas_signed: List[int] = []
    n_within_gt_fp = 0
    per_video_summary: Dict[str, Dict[str, int]] = {}

    for vid, (algo, gt) in per_video.items():
        if strict:
            res = match_reaches(algo, gt, strict=True,
                                strict_start_tol=2,
                                strict_span_tol_rel=0.5,
                                strict_span_tol_abs=5)
        else:
            res = match_reaches(algo, gt, window=10, strict=False)

        v_tp = v_fp = v_fn = 0
        v_within_gt = 0
        for r in res:
            if r.status == "matched":
                v_tp += 1
                # signed span delta: algo_span - gt_span
                a_span = r.algo_end - r.algo_start + 1
                g_span = r.gt_end - r.gt_start + 1
                span_deltas_signed.append(a_span - g_span)
            elif r.status == "fp":
                v_fp += 1
                # determine within_gt
                algo_obj = algo[r.algo_reach_index]
                if _fp_within_gt(algo_obj, gt):
                    v_within_gt += 1
                    n_within_gt_fp += 1
            elif r.status == "fn":
                v_fn += 1

        n_tp += v_tp; n_fp += v_fp; n_fn += v_fn
        per_video_summary[vid] = {
            "n_tp": v_tp, "n_fp": v_fp, "n_fn": v_fn,
            "n_within_gt_fp": v_within_gt,
            "n_gt": len(gt), "n_algo": len(algo),
        }

    if span_deltas_signed:
        arr = np.array(span_deltas_signed)
        span_summary = {
            "n_matched": int(len(arr)),
            "signed_median": float(np.median(arr)),
            "signed_mean": float(np.mean(arr)),
            "signed_p10": float(np.percentile(arr, 10)),
            "signed_p90": float(np.percentile(arr, 90)),
            "signed_min": int(arr.min()),
            "signed_max": int(arr.max()),
            "frac_algo_longer": float((arr > 0).mean()),
            "frac_algo_shorter": float((arr < 0).mean()),
            "frac_exact": float((arr == 0).mean()),
        }
    else:
        span_summary = {"n_matched": 0}

    fp_within_gt_frac = (n_within_gt_fp / n_fp) if n_fp else None

    return {
        "n_videos": len(per_video),
        "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
        "n_within_gt_fp": n_within_gt_fp,
        "fp_within_gt_fraction": fp_within_gt_frac,
        "span_delta_on_matched": span_summary,
        "per_video": per_video_summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DIAGNOSTIC: criterion reconciliation -- v8.0.0 (BSW w=0.8)")
    print("  2x2 grid over {calibration, generalization} x {strict, permissive}")
    print("=" * 70)
    print()

    print(f"Loading calibration LOOCV: {CAL_SNAPSHOT_JSON}", flush=True)
    if not CAL_SNAPSHOT_JSON.exists():
        raise FileNotFoundError(
            f"Calibration snapshot not found: {CAL_SNAPSHOT_JSON}")
    cal_records = _records_from_loocv(CAL_SNAPSHOT_JSON)
    cal_per_video = reconstruct_per_video(cal_records)
    print(f"  Reconstructed {len(cal_per_video)} videos; "
          f"{sum(len(a) for a, _ in cal_per_video.values())} algo reaches, "
          f"{sum(len(g) for _, g in cal_per_video.values())} GT reaches")
    print()

    print(f"Loading generalization 20-video: {GEN_SNAPSHOT_JSON}", flush=True)
    if not GEN_SNAPSHOT_JSON.exists():
        raise FileNotFoundError(
            f"Generalization snapshot not found: {GEN_SNAPSHOT_JSON}")
    gen_records = _records_from_generalization(GEN_SNAPSHOT_JSON)
    gen_per_video = reconstruct_per_video(gen_records)
    print(f"  Reconstructed {len(gen_per_video)} videos; "
          f"{sum(len(a) for a, _ in gen_per_video.values())} algo reaches, "
          f"{sum(len(g) for _, g in gen_per_video.values())} GT reaches")
    print()

    # 2x2 grid
    grid = {}
    for corpus_name, per_video in (("calibration", cal_per_video),
                                   ("generalization", gen_per_video)):
        for criterion_name, strict_flag in (("strict", True),
                                            ("permissive", False)):
            print(f"Scoring [{corpus_name}, {criterion_name}] ...", flush=True)
            cell = score_corpus(per_video, strict=strict_flag)
            grid[f"{corpus_name}__{criterion_name}"] = cell
            sd = cell["span_delta_on_matched"]
            print(f"  TP={cell['n_tp']:5} FP={cell['n_fp']:4} FN={cell['n_fn']:4}  "
                  f"within_gt_fp_frac={cell['fp_within_gt_fraction']}")
            if sd.get("n_matched"):
                print(f"  span_delta on TPs (signed): "
                      f"median={sd['signed_median']:+.1f}f  "
                      f"mean={sd['signed_mean']:+.2f}f  "
                      f"p10={sd['signed_p10']:+.1f}  p90={sd['signed_p90']:+.1f}  "
                      f"algo_longer={sd['frac_algo_longer']:.1%} "
                      f"algo_shorter={sd['frac_algo_shorter']:.1%} "
                      f"exact={sd['frac_exact']:.1%}")
            print()

    # -------- Interpretive summary block (printed and persisted) --------
    print("=" * 70)
    print("INTERPRETIVE READOUT")
    print("=" * 70)

    def _row(corpus):
        s = grid[f"{corpus}__strict"]
        p = grid[f"{corpus}__permissive"]
        return s, p

    for corpus in ("calibration", "generalization"):
        s, p = _row(corpus)
        fp_flip = (s["n_fp"] - p["n_fp"])
        fn_flip = (s["n_fn"] - p["n_fn"])
        print(f"  [{corpus}] strict->permissive  "
              f"FP delta: {fp_flip:+d}   FN delta: {fn_flip:+d}")
        print(f"    within_gt FP fraction:  strict={s['fp_within_gt_fraction']}  "
              f"permissive={p['fp_within_gt_fraction']}")
        sm = s["span_delta_on_matched"]; pm = p["span_delta_on_matched"]
        if sm.get("n_matched") and pm.get("n_matched"):
            print(f"    matched-reach signed span (median):  "
                  f"strict={sm['signed_median']:+.1f}f  "
                  f"permissive={pm['signed_median']:+.1f}f")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    (metrics_dir / "reconciliation.json").write_text(
        json.dumps({
            "grid": grid,
            "inputs": {
                "calibration_snapshot": str(CAL_SNAPSHOT_JSON),
                "generalization_snapshot": str(GEN_SNAPSHOT_JSON),
            },
            "criterion_strict": {
                "start_tol": 2, "span_tol_rel": 0.5, "span_tol_abs": 5,
            },
            "criterion_permissive": {
                "start_window": 10, "span_check": False,
            },
        }, indent=2), encoding="utf-8")

    # Per-video re-matched results dropped for inspection
    raw_per_video = {}
    for corpus, per_video in (("calibration", cal_per_video),
                              ("generalization", gen_per_video)):
        raw_per_video[corpus] = {}
        for vid, (algo, gt) in per_video.items():
            raw_per_video[corpus][vid] = {
                "n_algo": len(algo),
                "n_gt": len(gt),
                "algo_reaches": [(r.start_frame, r.end_frame) for r in algo],
                "gt_reaches": [(r.start_frame, r.end_frame) for r in gt],
            }
    (metrics_dir / "raw_per_video.json").write_text(
        json.dumps(raw_per_video, indent=2), encoding="utf-8")

    print(f"Wrote: {metrics_dir / 'reconciliation.json'}")
    print(f"Wrote: {metrics_dir / 'raw_per_video.json'}")
    print()
    print("NEXT: human reads reconciliation.json, decides whether the FP")
    print("contradiction is criterion-driven (most variance in FP counts")
    print("between strict and permissive on same corpus) or corpus-driven")
    print("(FP behavior stable across criteria, differs across corpora).")
    print("The matched-reach signed span answers the over/under question")
    print("independently of the criterion.")


if __name__ == "__main__":
    main()
