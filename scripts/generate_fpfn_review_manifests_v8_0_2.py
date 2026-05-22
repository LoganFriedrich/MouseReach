"""
Generate per-video FP/FN review manifests for v8.0.2 + asymmetric tolerance.

Successor to generate_fpfn_review_manifests_v8_0_1.py. Differences:

- Algo source: v8.0.1 outputs WITH v8.0.2 trim applied inline (the trim is
  a postprocess on top of v8.0.1 outputs, computed from paw_mean_lk).
- Matcher: asymmetric strict tolerance via metrics.match_reaches default
  (start_delta in [-2, +5]).
- Topology classification: uses STRICT_START_TOL_EARLY / STRICT_START_TOL_LATE
  from metrics module for consistency.
- Output: fpfn_review_manifests/v8.0.2/  (parallel to v8.0.1/)
- Archive: existing v8.0.1/ manifest dir is archived alongside v8.0.2/
  (the v8.0.1 dir is the "prior" manifest population this regen supersedes).

DOES NOT include the ~50 reaches per corpus that v8.0.2 trim drops below
MIN_SPAN. Per Logan 2026-05-22, dropped reaches are intentionally absent
from the manifests; the trim's drop decisions are deterministic and the
review tool's behavior of "missing reach = trim dropped it" is acceptable.

================================================================
SOURCES
================================================================

  Calibration (LOOCV on 16-video exhaustive train pool, v8.0.1 baseline):
    Improvement_Snapshots/reach_detection/
      v8.0.1_model_3_1_baseline_loocv/metrics/loocv_aggregate.json
    (raw_results has v8.0.1 algo reaches; trim applied here)
    paw_mean_lk from train_pool.parquet at:
      Improvement_Snapshots/_corpus/2026-05-21_model_3_1_inventory/
        phase_b_dataset/train_pool.parquet

  Holdout generalization (19 of 20 exhaustive videos, v8.0.1 baseline):
    Improvement_Snapshots/reach_detection/
      v8.0.0_holdout_generalization_merge_gap_0/algo_outputs_v8.0.0_mg0/
        <video_id>_reaches.json
    GT at: iterations/generalization_test_2026-05-11/gt/
    paw_mean_lk from DLC h5 at:
      iterations/generalization_test_2026-05-11/dlc/

================================================================
OUTPUTS
================================================================

  Y:/2_Connectome/Behavior/MouseReach_Improvement/fpfn_review_manifests/v8.0.2/
    calibration_loocv/   <video_id>.json
    holdout_2026_05_11/  <video_id>.json

================================================================
ARCHIVE
================================================================

  The v8.0.1 manifest dir is moved to:
    fpfn_review_manifests/v8.0.1_archive/<timestamp>_<RUN_TAG>/
  Default RUN_TAG = "pre_v8_0_2_and_asym_tol"; override via env var
  MOUSEREACH_MANIFEST_RUN_TAG.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches, MIN_REPORTED_SPAN, is_kinematically_excluded,
    is_outside_gt_segmentation,
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8.postprocess import (
    ReachSpan, trim_leading_sustained_lk, compute_paw_mean_lk,
)


RUN_TAG = os.environ.get("MOUSEREACH_MANIFEST_RUN_TAG",
                          "pre_v8_0_2_and_asym_tol")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CAL_LOOCV_SOURCE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_model_3_1_baseline_loocv\metrics\loocv_aggregate.json"
)
CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
CAL_SNAPSHOT_NAME = "v8.0.2_via_trim_on_v8.0.1_model_3_1_baseline_loocv"

GEN_ALGO_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_holdout_generalization_merge_gap_0\algo_outputs_v8.0.0_mg0"
)
GEN_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)
GEN_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
GEN_SNAPSHOT_NAME = "v8.0.2_via_trim_on_v8.0.0_holdout_generalization_merge_gap_0"

OUTPUT_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests"
    r"\v8.0.2"
)
PRIOR_OUTPUT_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests"
    r"\v8.0.1"
)

# Calibration GT comes from the parquet's reach_id labels (not a separate GT dir),
# so we use the legacy v8.0.1 generator's GT_ROOTS mapping for boundaries only.
GT_ROOTS = {
    "calibration_loocv": Path(
        r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
        r"\validation_runs\DLC_2026_03_27\gt"
    ),
    "holdout_2026_05_11": GEN_GT_DIR,
}

# v8.0.2 + asymmetric tolerance metadata embedded in each manifest
DETECTOR_VERSION = "v8.0.2_bsw_w0.8_mg0_trim_n3_t0.6"
MATCHING_CRITERION = (
    "asymmetric_start_-2_+5_span_50pct_or_5f"
)

SPAN_TOL_REL = 0.5
SPAN_TOL_ABS = 5

# Category thresholds (unchanged from v8.0.1)
MODEL_MISS_WINDOW = 30
NEAR_WINDOW = 10
RANDOM_WINDOW = 30

# v8.0.2 trim parameters (production defaults)
TRIM_THRESHOLD = 0.60
TRIM_SUSTAIN_N = 3
TRIM_MIN_SPAN = 3

# DLC scorer suffix (used for holdout h5 filenames)
DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"


# ---------------------------------------------------------------------------
# v8.0.2 trim application
# ---------------------------------------------------------------------------

def apply_v802_trim(algos: List[Tuple[int, int]], paw_mean_lk: np.ndarray
                    ) -> List[Tuple[int, int]]:
    """Apply v8.0.2 leading-trim to a list of (start, end) reaches.

    Reaches reduced below MIN_SPAN by trimming are dropped (consistent
    with production v8.0.2 behaviour).
    """
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos]
    trimmed = trim_leading_sustained_lk(
        spans, paw_mean_lk,
        threshold=TRIM_THRESHOLD,
        sustain_n=TRIM_SUSTAIN_N,
        min_span=TRIM_MIN_SPAN,
    )
    return [(r.start_frame, r.end_frame) for r in trimmed]


def load_dlc_h5(path: Path) -> pd.DataFrame:
    df = pd.read_hdf(path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df


# ---------------------------------------------------------------------------
# Source data loading (calibration + holdout) with v8.0.2 trim applied
# ---------------------------------------------------------------------------

PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]


def _build_calibration_per_video() -> Dict[str, Dict[str, Any]]:
    """Returns dict[video_id] = {"algos": [(s,e),...], "gts": [(s,e),...]}.
    Algos have v8.0.2 trim applied.
    """
    print(f"Loading calibration LOOCV: {CAL_LOOCV_SOURCE.name}", flush=True)
    data = json.loads(CAL_LOOCV_SOURCE.read_text(encoding="utf-8"))
    raw = data["raw_results"]

    algos_pre = defaultdict(set)
    gts = defaultdict(set)
    for r in raw:
        vid = r["video_id"]
        if r["algo_start_frame"] >= 0:
            algos_pre[vid].add((int(r["algo_start_frame"]),
                                int(r["algo_end_frame"])))
        if r["gt_start_frame"] >= 0:
            gts[vid].add((int(r["gt_start_frame"]),
                          int(r["gt_end_frame"])))

    print(f"  {len(algos_pre)} videos; "
          f"{sum(len(s) for s in algos_pre.values())} v8.0.1 algo reaches; "
          f"{sum(len(s) for s in gts.values())} GT reaches",
          flush=True)

    # Load paw_mean_lk from parquet
    print(f"Loading parquet for paw_mean_lk...", flush=True)
    df = pd.read_parquet(CAL_PARQUET,
                          columns=["video_id", "frame"] + PARQUET_LK_COLS)
    df["paw_mean_lk"] = df[PARQUET_LK_COLS].to_numpy(dtype=np.float32).mean(axis=1)
    lk_by_vid = {}
    for vid, grp in df.groupby("video_id", sort=False):
        g = grp.sort_values("frame")
        mx = int(g["frame"].max())
        arr = np.full(mx + 1, np.nan, dtype=np.float32)
        arr[g["frame"].to_numpy()] = g["paw_mean_lk"].to_numpy()
        lk_by_vid[vid] = arr

    # Apply v8.0.2 trim
    out = {}
    n_dropped = 0
    for vid in sorted(set(algos_pre.keys()) | set(gts.keys())):
        algos_v1 = sorted(algos_pre.get(vid, set()))
        if vid in lk_by_vid:
            algos_v2 = apply_v802_trim(algos_v1, lk_by_vid[vid])
        else:
            algos_v2 = algos_v1
        n_dropped += len(algos_v1) - len(algos_v2)
        out[vid] = {
            "algos": sorted(set(algos_v2)),
            "gts": sorted(gts.get(vid, set())),
            "v801_algo_count": len(algos_v1),
        }
    print(f"  v8.0.2 trim: dropped {n_dropped} reaches across all videos "
          f"(remaining: {sum(len(v['algos']) for v in out.values())})",
          flush=True)
    return out


def _build_holdout_per_video() -> Dict[str, Dict[str, Any]]:
    """Same as calibration but for the holdout corpus."""
    print(f"Loading holdout v8.0.1 algo outputs + GT + DLC...", flush=True)
    out = {}
    n_v1_total = 0
    n_v2_total = 0
    for algo_path in sorted(GEN_ALGO_DIR.glob("*_reaches.json")):
        vid = algo_path.stem.replace("_reaches", "")
        gt_path = GEN_GT_DIR / f"{vid}_unified_ground_truth.json"
        dlc_path = GEN_DLC_DIR / f"{vid}{DLC_SUFFIX}.h5"
        if not gt_path.exists():
            print(f"  [skip] no GT: {vid}")
            continue
        if not dlc_path.exists():
            print(f"  [skip] no DLC: {vid}")
            continue

        # Load v8.0.1 algo reaches
        adata = json.loads(algo_path.read_text(encoding="utf-8"))
        algos_v1 = sorted(set(
            (int(r["start_frame"]), int(r["end_frame"]))
            for r in adata.get("reaches", [])
        ))
        # Load GT reaches
        gdata = json.loads(gt_path.read_text(encoding="utf-8"))
        reaches_obj = gdata.get("reaches", {})
        rlist = (reaches_obj.get("reaches", [])
                 if isinstance(reaches_obj, dict) else [])
        gts = sorted(set(
            (int(r["start_frame"]), int(r["end_frame"]))
            for r in rlist if not r.get("exclude_from_analysis")
        ))
        # Load DLC for paw_mean_lk
        dlc = load_dlc_h5(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        # Apply v8.0.2 trim
        algos_v2 = apply_v802_trim(algos_v1, paw_lk)

        out[vid] = {
            "algos": sorted(set(algos_v2)),
            "gts": gts,
            "v801_algo_count": len(algos_v1),
        }
        n_v1_total += len(algos_v1)
        n_v2_total += len(algos_v2)
    print(f"  {len(out)} holdout videos processed; "
          f"v8.0.1: {n_v1_total} algos; v8.0.2: {n_v2_total} algos "
          f"(dropped {n_v1_total - n_v2_total})",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# Matching + records
# ---------------------------------------------------------------------------

def _build_records_for_video(algos: List[Tuple[int, int]],
                              gts: List[Tuple[int, int]]
                              ) -> List[Dict[str, Any]]:
    """Match algo + GT under asymmetric tolerance, return list of record
    dicts in the same shape as the v8.0.1 manifest generator expects.
    """
    algo_reaches = [Reach(start_frame=s, end_frame=e, index=i)
                    for i, (s, e) in enumerate(algos)]
    gt_reaches = [Reach(start_frame=s, end_frame=e, index=i)
                  for i, (s, e) in enumerate(gts)]
    # match_reaches with strict=True uses asymmetric defaults from metrics
    results = match_reaches(
        algo_reaches, gt_reaches, strict=True,
        strict_start_tol=STRICT_START_TOL_EARLY,
        strict_start_tol_late=STRICT_START_TOL_LATE,
        strict_span_tol_rel=SPAN_TOL_REL,
        strict_span_tol_abs=SPAN_TOL_ABS,
    )
    out = []
    for r in results:
        kind = ("TP" if r.status == "matched"
                else "FP" if r.status == "fp"
                else "FN")
        rec: Dict[str, Any] = {
            "kind": kind,
            "algo_start": r.algo_start if r.algo_start is not None else -1,
            "algo_end": r.algo_end if r.algo_end is not None else -1,
            "gt_start": r.gt_start if r.gt_start is not None else -1,
            "gt_end": r.gt_end if r.gt_end is not None else -1,
            "start_delta": r.start_delta,
            "span_delta": ((r.algo_end - r.algo_start + 1)
                           - (r.gt_end - r.gt_start + 1)
                           if r.status == "matched" else None),
        }
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Topology classification (copied + adapted from v8.0.1 generator)
# ---------------------------------------------------------------------------

def _overlaps(a: Reach, b: Reach) -> bool:
    return not (a.end_frame < b.start_frame or a.start_frame > b.end_frame)


def _passes_span_tol(a_span: int, g_span: int) -> bool:
    tol = max(SPAN_TOL_REL * g_span, SPAN_TOL_ABS)
    return abs(a_span - g_span) <= tol


def build_topology_components(algos: List[Reach], gts: List[Reach]
                              ) -> Tuple[Dict[Tuple[int, int], int],
                                         Dict[Tuple[int, int], int],
                                         List[Tuple[List[Reach], List[Reach]]]]:
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(len(algos)):
        parent[("algo", i)] = ("algo", i)
    for j in range(len(gts)):
        parent[("gt", j)] = ("gt", j)

    for i, a in enumerate(algos):
        for j, g in enumerate(gts):
            if _overlaps(a, g):
                union(("algo", i), ("gt", j))

    comp_nodes: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for node in parent:
        comp_nodes[find(node)].append(node)

    sorted_roots = sorted(comp_nodes.keys())
    components: List[Tuple[List[Reach], List[Reach]]] = []
    algo_to_cid: Dict[Tuple[int, int], int] = {}
    gt_to_cid: Dict[Tuple[int, int], int] = {}
    for cid, root in enumerate(sorted_roots):
        a_list, g_list = [], []
        for node in comp_nodes[root]:
            kind, idx = node
            if kind == "algo":
                a_list.append(algos[idx])
                algo_to_cid[(algos[idx].start_frame, algos[idx].end_frame)] = cid
            else:
                g_list.append(gts[idx])
                gt_to_cid[(gts[idx].start_frame, gts[idx].end_frame)] = cid
        components.append((a_list, g_list))
    return algo_to_cid, gt_to_cid, components


def classify_component(algo_list: List[Reach], gt_list: List[Reach]
                       ) -> Tuple[str, Optional[str]]:
    """Component topology label. Uses asymmetric start tolerance from
    metrics module (STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE).
    """
    n_algo = len(algo_list)
    n_gt = len(gt_list)
    if n_algo == 1 and n_gt == 0:
        return ("FALSE_POSITIVE", None)
    if n_algo == 0 and n_gt == 1:
        return ("FALSE_NEGATIVE", None)
    if n_algo == 1 and n_gt == 1:
        a = algo_list[0]
        g = gt_list[0]
        a_span = a.end_frame - a.start_frame + 1
        g_span = g.end_frame - g.start_frame + 1
        sd = a.start_frame - g.start_frame
        start_ok = (-STRICT_START_TOL_EARLY <= sd <= STRICT_START_TOL_LATE)
        span_ok = _passes_span_tol(a_span, g_span)
        if start_ok and span_ok:
            return ("TP", None)
        if not start_ok and not span_ok:
            return ("TOLERANCE_ERROR", "start_and_span_off")
        if not start_ok:
            return ("TOLERANCE_ERROR", "start_off")
        return ("TOLERANCE_ERROR", "span_off")
    if n_algo == 1 and n_gt >= 2:
        return ("MERGED", f"{n_gt}_gt")
    if n_algo >= 2 and n_gt == 1:
        return ("FRAGMENTED", f"{n_algo}_algo")
    if n_algo >= 2 and n_gt >= 2:
        return ("COMPLEX", f"{n_algo}_algo_{n_gt}_gt")
    return ("UNKNOWN", None)


# ---------------------------------------------------------------------------
# FP / FN category logic (copied verbatim from v8.0.1 generator)
# ---------------------------------------------------------------------------

def _algos_overlapping_window(algos: List[Reach], a: int, b: int
                              ) -> List[Reach]:
    return [r for r in algos
            if not (r.end_frame < a or r.start_frame > b)]


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
    overlapping = _algos_overlapping_window(algos, gt_start, gt_end)
    if len(overlapping) >= 2:
        return "fragmented"
    near, abs_sd = _nearest_by_start(gt_start, algos)
    if near is None or abs_sd > MODEL_MISS_WINDOW:
        return "miss"
    g_span = gt_end - gt_start + 1
    a_span = near.end_frame - near.start_frame + 1
    # Asymmetric early/late tolerance: use the existing strict bounds.
    sd_signed = near.start_frame - gt_start
    start_ok = (-STRICT_START_TOL_EARLY <= sd_signed <= STRICT_START_TOL_LATE)
    span_ok = _passes_span_tol(a_span, g_span)
    if not start_ok and not span_ok:
        return "tolerance_miss_both"
    if not start_ok:
        return "tolerance_miss_start"
    return "tolerance_miss_span"


def categorize_fp(algo_start: int, algo_end: int,
                  gts: List[Reach], matched_gt_keys: set) -> str:
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

def _topology_for_event(algo_key: Optional[Tuple[int, int]],
                        gt_key: Optional[Tuple[int, int]],
                        algo_to_cid: Dict[Tuple[int, int], int],
                        gt_to_cid: Dict[Tuple[int, int], int],
                        components: List[Tuple[List[Reach], List[Reach]]]
                        ) -> Tuple[str, Optional[str], Optional[int]]:
    cid: Optional[int] = None
    if algo_key is not None and algo_key in algo_to_cid:
        cid = algo_to_cid[algo_key]
    elif gt_key is not None and gt_key in gt_to_cid:
        cid = gt_to_cid[gt_key]
    if cid is None:
        return ("UNKNOWN", None, None)
    a_list, g_list = components[cid]
    topo, sub = classify_component(a_list, g_list)
    return (topo, sub, cid)


def _build_event(rec: Dict[str, Any],
                 algos: List[Reach], gts: List[Reach],
                 matched_gt_keys: set,
                 gt_boundaries: List[int],
                 algo_to_cid, gt_to_cid, components
                 ) -> Dict[str, Any]:
    kind = rec["kind"]
    if kind == "TP":
        a_s, a_e = int(rec["algo_start"]), int(rec["algo_end"])
        g_s, g_e = int(rec["gt_start"]), int(rec["gt_end"])
        excl = (is_kinematically_excluded(a_s, a_e)
                or is_kinematically_excluded(g_s, g_e))
        topo, sub, cid = _topology_for_event(
            (a_s, a_e), (g_s, g_e), algo_to_cid, gt_to_cid, components)
        return {
            "kind": "TP",
            "detector": {"start": a_s, "end": a_e},
            "gt": {"start": g_s, "end": g_e},
            "category": None,
            "start_delta": rec["start_delta"],
            "span_delta": rec["span_delta"],
            "kinematically_excluded": bool(excl),
            "outside_gt_segmentation": False,
            "topology": topo,
            "topology_sub": sub,
            "component_id": cid,
        }
    if kind == "FP":
        a_s, a_e = int(rec["algo_start"]), int(rec["algo_end"])
        cat = categorize_fp(a_s, a_e, gts, matched_gt_keys)
        excl = is_kinematically_excluded(a_s, a_e)
        outside = is_outside_gt_segmentation(a_s, gt_boundaries)
        topo, sub, cid = _topology_for_event(
            (a_s, a_e), None, algo_to_cid, gt_to_cid, components)
        return {
            "kind": "FP",
            "detector": {"start": a_s, "end": a_e},
            "gt": None,
            "category": cat,
            "kinematically_excluded": bool(excl),
            "outside_gt_segmentation": bool(outside),
            "topology": topo,
            "topology_sub": sub,
            "component_id": cid,
        }
    if kind == "FN":
        g_s, g_e = int(rec["gt_start"]), int(rec["gt_end"])
        cat = categorize_fn(g_s, g_e, algos)
        excl = is_kinematically_excluded(g_s, g_e)
        topo, sub, cid = _topology_for_event(
            None, (g_s, g_e), algo_to_cid, gt_to_cid, components)
        return {
            "kind": "FN",
            "detector": None,
            "gt": {"start": g_s, "end": g_e},
            "category": cat,
            "kinematically_excluded": bool(excl),
            "outside_gt_segmentation": False,
            "topology": topo,
            "topology_sub": sub,
            "component_id": cid,
        }
    raise ValueError(f"Unknown kind: {kind}")


def _load_gt_boundaries(corpus_label: str, video_id: str) -> List[int]:
    gt_root = GT_ROOTS.get(corpus_label)
    if gt_root is None:
        return []
    gt_path = gt_root / f"{video_id}_unified_ground_truth.json"
    if not gt_path.exists():
        return []
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    seg = data.get("segmentation", {})
    boundaries = seg.get("boundaries", [])
    return sorted(int(b["frame"]) for b in boundaries if "frame" in b)


def build_manifests(per_video: Dict[str, Dict[str, Any]],
                    corpus_label: str,
                    snapshot_name: str) -> Dict[str, Dict[str, Any]]:
    manifests = {}
    for video_id, vd in per_video.items():
        algos_tuples = vd["algos"]
        gts_tuples = vd["gts"]
        algos = [Reach(start_frame=s, end_frame=e, index=i)
                 for i, (s, e) in enumerate(algos_tuples)]
        gts = [Reach(start_frame=s, end_frame=e, index=i)
               for i, (s, e) in enumerate(gts_tuples)]

        # Re-match
        records = _build_records_for_video(algos_tuples, gts_tuples)

        # Topology components
        algo_to_cid, gt_to_cid, components = build_topology_components(algos, gts)

        # GT boundaries for outside_gt_segmentation flag
        gt_boundaries = _load_gt_boundaries(corpus_label, video_id)

        # Matched GT keys (for split_twin check)
        matched_gt_keys = set()
        for rec in records:
            if rec["kind"] == "TP":
                matched_gt_keys.add((int(rec["gt_start"]), int(rec["gt_end"])))

        events = [
            _build_event(rec, algos, gts, matched_gt_keys, gt_boundaries,
                          algo_to_cid, gt_to_cid, components)
            for rec in records
        ]
        events.sort(key=lambda ev: (ev["gt"]["start"] if ev["kind"] == "FN"
                                     else ev["detector"]["start"]))

        topology_summary = defaultdict(int)
        for a_list, g_list in components:
            label, _sub = classify_component(a_list, g_list)
            topology_summary[label] += 1

        manifests[video_id] = {
            "video_id": video_id,
            "detector_version": DETECTOR_VERSION,
            "snapshot": snapshot_name,
            "corpus": corpus_label,
            "matching_criterion": MATCHING_CRITERION,
            "min_reported_span": MIN_REPORTED_SPAN,
            "v801_algo_count": vd.get("v801_algo_count"),
            "v802_algo_count": len(algos_tuples),
            "v802_trim_dropped": (vd.get("v801_algo_count", 0)
                                   - len(algos_tuples)),
            "gt_segmentation": {
                "n_boundaries": len(gt_boundaries),
                "first_frame": gt_boundaries[0] if gt_boundaries else None,
                "last_frame": gt_boundaries[-1] if gt_boundaries else None,
            },
            "topology_summary": dict(topology_summary),
            "events": events,
        }
    return manifests


# ---------------------------------------------------------------------------
# Archive existing manifests
# ---------------------------------------------------------------------------

def archive_existing_manifests() -> Optional[Path]:
    """Move v8.0.1/ manifest dir into v8.0.1_archive/<timestamp>_<tag>/.
    Also archives any pre-existing content in v8.0.2/ (rare; only on re-runs).
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    archived_any = False
    archive_dirs = []

    # Archive v8.0.1/ (the prior manifest population we're superseding)
    if PRIOR_OUTPUT_ROOT.exists() and any(PRIOR_OUTPUT_ROOT.iterdir()):
        archive_root = (PRIOR_OUTPUT_ROOT.parent
                        / f"{PRIOR_OUTPUT_ROOT.name}_archive"
                        / f"{timestamp}_{RUN_TAG}")
        archive_root.mkdir(parents=True, exist_ok=True)
        for corpus in ("calibration_loocv", "holdout_2026_05_11"):
            src = PRIOR_OUTPUT_ROOT / corpus
            if src.exists() and any(src.iterdir()):
                dst = archive_root / corpus
                shutil.move(str(src), str(dst))
                print(f"  archived {corpus}: -> {dst}")
                archived_any = True
        archive_dirs.append(archive_root)

    # Archive v8.0.2/ if it already has content (re-run case)
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        archive_root = (OUTPUT_ROOT.parent
                        / f"{OUTPUT_ROOT.name}_archive"
                        / f"{timestamp}_{RUN_TAG}_rerun")
        archive_root.mkdir(parents=True, exist_ok=True)
        for corpus in ("calibration_loocv", "holdout_2026_05_11"):
            src = OUTPUT_ROOT / corpus
            if src.exists() and any(src.iterdir()):
                dst = archive_root / corpus
                shutil.move(str(src), str(dst))
                print(f"  archived (prior v8.0.2 rerun) {corpus}: -> {dst}")
                archived_any = True
        archive_dirs.append(archive_root)

    return archive_dirs[0] if archive_dirs else None


# ---------------------------------------------------------------------------
# Write + summary
# ---------------------------------------------------------------------------

def write_manifests(manifests: Dict[str, Dict[str, Any]],
                    corpus_label: str) -> int:
    out_dir = OUTPUT_ROOT / corpus_label
    out_dir.mkdir(parents=True, exist_ok=True)
    for video_id, manifest in manifests.items():
        out_path = out_dir / f"{video_id}.json"
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(manifests)


def _event_counts_in(events, kind):
    n_un = sum(1 for e in events if e["kind"] == kind)
    n_span = sum(1 for e in events
                 if e["kind"] == kind
                 and not e.get("kinematically_excluded", False))
    n_full = sum(1 for e in events
                 if e["kind"] == kind
                 and not e.get("kinematically_excluded", False)
                 and not e.get("outside_gt_segmentation", False))
    return n_un, n_span, n_full


def _print_summary(manifests, corpus_label):
    total_tp = total_fp = total_fn = 0
    total_tp_f = total_fp_f = total_fn_f = 0
    print(f"  [{corpus_label}] {len(manifests)} videos:")
    for video_id, m in manifests.items():
        events = m["events"]
        n_tp, _, n_tp_f = _event_counts_in(events, "TP")
        n_fp, _, n_fp_f = _event_counts_in(events, "FP")
        n_fn, _, n_fn_f = _event_counts_in(events, "FN")
        total_tp += n_tp; total_fp += n_fp; total_fn += n_fn
        total_tp_f += n_tp_f; total_fp_f += n_fp_f; total_fn_f += n_fn_f
        print(f"    {video_id:35} unfilt TP={n_tp:>4} FP={n_fp:>4} FN={n_fn:>4}  |  "
              f"filt TP={n_tp_f:>4} FP={n_fp_f:>4} FN={n_fn_f:>4}  "
              f"(v8.0.2 trim dropped {m.get('v802_trim_dropped', 0)} v8.0.1 algos)")
    print(f"  totals (unfiltered):                          TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  totals (span + outside_gt_segmentation filter): TP={total_tp_f} FP={total_fp_f} FN={total_fn_f}")


def main():
    print("=" * 70)
    print("Generate FP/FN review manifests -- v8.0.2 + asymmetric tolerance")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Run tag: {RUN_TAG}")
    print(f"Matcher: STRICT_START_TOL_EARLY={STRICT_START_TOL_EARLY}, "
          f"STRICT_START_TOL_LATE={STRICT_START_TOL_LATE}")
    print(f"Trim:    threshold={TRIM_THRESHOLD}, sustain_n={TRIM_SUSTAIN_N}, min_span={TRIM_MIN_SPAN}")
    print("=" * 70)
    print()

    print("Archiving prior v8.0.1 manifests + any prior v8.0.2 rerun...")
    archive_existing_manifests()
    print()

    cal = _build_calibration_per_video()
    print()
    gen = _build_holdout_per_video()
    print()

    print("Building manifests ...")
    cal_manifests = build_manifests(cal, "calibration_loocv", CAL_SNAPSHOT_NAME)
    gen_manifests = build_manifests(gen, "holdout_2026_05_11", GEN_SNAPSHOT_NAME)
    print()

    print("Per-video event counts:")
    _print_summary(cal_manifests, "calibration_loocv")
    print()
    _print_summary(gen_manifests, "holdout_2026_05_11")
    print()

    n_cal = write_manifests(cal_manifests, "calibration_loocv")
    n_gen = write_manifests(gen_manifests, "holdout_2026_05_11")
    print(f"Wrote {n_cal} manifests to {OUTPUT_ROOT / 'calibration_loocv'}/")
    print(f"Wrote {n_gen} manifests to {OUTPUT_ROOT / 'holdout_2026_05_11'}/")


if __name__ == "__main__":
    main()
