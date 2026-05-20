"""
Generate per-video FP/FN review manifests for v8.0.1 (post-mg=0 ship).

Parallel to the original `generate_fpfn_review_manifests.py` runner, but
points at the v8.0.1 snapshots (mg=0 calibration LOOCV + mg=0 holdout
generalization) and writes to a separate subfolder so the prior v8.0.0
manifests stay intact for comparison.

================================================================
SOURCES (v8.0.1 production = BSW b=1/w=0.8 + merge_gap=0)
================================================================

  Calibration (LOOCV on 16-video exhaustive train pool):
    Improvement_Snapshots/reach_detection/
      v8.0.0_dev_merge_gap_0_candidate/metrics/loocv_aggregate.json
    (raw_results schema, same as prior BSW w=0.8 snapshot)

  Holdout generalization (19 of 20 exhaustive videos):
    Improvement_Snapshots/reach_detection/
      v8.0.0_holdout_generalization_merge_gap_0/metrics/reach_detection_scalars.json
    (matches schema, same as prior 20-video snapshot)

================================================================
OUTPUTS
================================================================

  Y:/2_Connectome/Behavior/MouseReach_Improvement/fpfn_review_manifests/v8.0.1/
    calibration_loocv/
      <video_id>.json
    holdout_2026_05_11/
      <video_id>.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches, MIN_REPORTED_SPAN, is_kinematically_excluded,
    is_outside_gt_segmentation,
)

# Tag for the current generation run. Override via MOUSEREACH_MANIFEST_RUN_TAG.
# Used when archiving prior manifests (the tag describes what's about to CHANGE
# in this regen, so the archived snapshot is labeled by what it pre-dates).
# Examples: "pre_topology_enrichment", "pre_segmentation_filter", "gt_edit_rescore".
RUN_TAG = os.environ.get("MOUSEREACH_MANIFEST_RUN_TAG", "regen")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CAL_SOURCE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_merge_gap_0_candidate_rescored_2026-05-20"
    r"\metrics\loocv_aggregate.json"
)
CAL_SNAPSHOT_NAME = "v8.0.0_dev_merge_gap_0_candidate_rescored_2026-05-20"

GEN_SOURCE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_holdout_generalization_merge_gap_0"
    r"\metrics\reach_detection_scalars.json"
)
GEN_SNAPSHOT_NAME = "v8.0.0_holdout_generalization_merge_gap_0"

OUTPUT_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests"
    r"\v8.0.1"
)

# GT directories per corpus (match the widget's GT_ROOTS auto-resolve mapping).
# Used to load GT segmentation boundaries for the outside_gt_segmentation
# headline filter (see metrics.is_outside_gt_segmentation).
GT_ROOTS = {
    "calibration_loocv": Path(
        r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
        r"\validation_runs\DLC_2026_03_27\gt"
    ),
    "holdout_2026_05_11": Path(
        r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
        r"\iterations\generalization_test_2026-05-11\gt"
    ),
}

# Manifest fields
DETECTOR_VERSION = "v8.0.1_bsw_w0.8_mg0"
MATCHING_CRITERION = "strict_start2_span"

# Matching tolerances (same as production eval)
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5

# Category thresholds (same as prior generator)
MODEL_MISS_WINDOW = 30
RANDOM_WINDOW = 30
NEAR_WINDOW = 10


def _load_gt_boundaries(corpus_label: str, video_id: str) -> List[int]:
    """Load GT segmentation boundary frames for a video.

    Returns the sorted list of boundary frames from the unified GT file's
    `segmentation.boundaries` array. Returns empty list if the GT file is
    missing or has no segmentation -- in that case the outside_gt_segmentation
    filter is a no-op for the video.
    """
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
    frames = [int(b["frame"]) for b in boundaries if "frame" in b]
    return sorted(frames)


# ---------------------------------------------------------------------------
# Record loaders (normalize both snapshot schemas)
# ---------------------------------------------------------------------------

def _normalize_kind(raw: str) -> str:
    s = raw.lower()
    if s in ("tp", "matched"):
        return "TP"
    if s == "fp":
        return "FP"
    if s == "fn":
        return "FN"
    raise ValueError(f"Unknown status string: {raw!r}")


def _load_calibration_records() -> List[Dict[str, Any]]:
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
# Category logic
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


# ---------------------------------------------------------------------------
# Topology classifier (connected components of algo-GT overlap graph)
# ---------------------------------------------------------------------------
#
# Computes a topology label per connected component and assigns a
# component_id + topology fields to each event in the manifest. The 7
# topology labels (renamed from the original 6-event taxonomy by Logan
# 2026-05-20):
#
#   TP              - 1 algo + 1 GT, start AND span tolerance both pass
#   TOLERANCE_ERROR - 1 algo + 1 GT, overlap exists but tolerance fails
#   MERGED          - 1 algo + 2+ GT (algo span covers multiple GT reaches)
#   FRAGMENTED      - 2+ algo + 1 GT (multiple algo split one GT)
#   FALSE_POSITIVE  - 1 algo + 0 GT (algo with no GT overlap anywhere)
#   FALSE_NEGATIVE  - 0 algo + 1 GT (GT with no algo overlap anywhere)
#   COMPLEX         - 2+ algo + 2+ GT in one component (merge AND fragment)
#
# The pre-existing FP/FN sub-category labels (within_gt, tolerance_miss_*,
# phantom, etc.) are preserved on each event in the original `category`
# field. The new `topology` and `component_id` fields are ADDITIVE -- they
# do not replace existing fields. Backwards-compat with current widget.
# Per Logan's "Way 1" preference 2026-05-20.

def _overlaps(reach_a: Reach, reach_b: Reach) -> bool:
    return not (reach_a.end_frame < reach_b.start_frame
                or reach_a.start_frame > reach_b.end_frame)


def build_topology_components(algos: List[Reach], gts: List[Reach]
                              ) -> Tuple[Dict[Tuple[int, int], int],
                                         Dict[Tuple[int, int], int],
                                         List[Tuple[List[Reach], List[Reach]]]]:
    """Connected components of the (algo, gt) overlap graph for one video.

    Returns three things:
      algo_to_cid: maps (algo_start, algo_end) -> component_id
      gt_to_cid:   maps (gt_start, gt_end)     -> component_id
      components:  list of (algo_list, gt_list) tuples in component_id order

    Algorithm: union-find over algo + gt nodes, with an edge between an
    algo node and a gt node iff their frame intervals overlap. Each
    connected component is one "topology event."
    """
    # Build parent dict for union-find. Keys: ("algo", i) or ("gt", i).
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

    # Edges: algo-gt overlap. Could optimize with interval tree but the
    # typical per-video reach count is < 500, so O(n*m) is fine here.
    for i, a in enumerate(algos):
        for j, g in enumerate(gts):
            if _overlaps(a, g):
                union(("algo", i), ("gt", j))

    # Collect components: root -> list of nodes
    comp_nodes: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for node in parent:
        comp_nodes[find(node)].append(node)

    # Assign sequential component_id (stable order: by first node's identity)
    sorted_roots = sorted(comp_nodes.keys())
    components: List[Tuple[List[Reach], List[Reach]]] = []
    algo_to_cid: Dict[Tuple[int, int], int] = {}
    gt_to_cid: Dict[Tuple[int, int], int] = {}

    for cid, root in enumerate(sorted_roots):
        a_list = []
        g_list = []
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
    """Return (topology_label, topology_sub) for a connected component.

    topology_sub is an optional finer label. For multi-reach topologies it
    encodes the counts (e.g. "2_gt" for a MERGED with 2 GTs). For
    TOLERANCE_ERROR the sub label is derived from which of start/span
    tolerance failed. For TP, FALSE_POSITIVE, FALSE_NEGATIVE the sub
    label is None.
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
        start_ok = abs(a.start_frame - g.start_frame) <= STRICT_START_TOL
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
    # Should be unreachable (n_algo=0 and n_gt=0 = no component to classify)
    return ("UNKNOWN", None)


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

def _topology_for_event(kind: str,
                        algo_key: Optional[Tuple[int, int]],
                        gt_key: Optional[Tuple[int, int]],
                        algo_to_cid: Dict[Tuple[int, int], int],
                        gt_to_cid: Dict[Tuple[int, int], int],
                        components: List[Tuple[List[Reach], List[Reach]]]
                        ) -> Tuple[str, Optional[str], Optional[int]]:
    """Return (topology, topology_sub, component_id) for a manifest event.

    The event identifies its component via the algo and/or GT frame keys
    that participate in the event. For TP both keys are used (and should
    map to the same component); for FP only the algo key; for FN only the
    GT key.
    """
    cid: Optional[int] = None
    if algo_key is not None and algo_key in algo_to_cid:
        cid = algo_to_cid[algo_key]
    elif gt_key is not None and gt_key in gt_to_cid:
        cid = gt_to_cid[gt_key]

    if cid is None:
        # Event references a reach not present in algo or gt lists. Should
        # not happen if reconstruct_per_video runs on the same records.
        return ("UNKNOWN", None, None)

    a_list, g_list = components[cid]
    topology, topology_sub = classify_component(a_list, g_list)
    return (topology, topology_sub, cid)


def _build_event(rec: Dict[str, Any],
                 algos: List[Reach], gts: List[Reach],
                 matched_gt_keys: set,
                 gt_boundaries: List[int],
                 algo_to_cid: Dict[Tuple[int, int], int],
                 gt_to_cid: Dict[Tuple[int, int], int],
                 components: List[Tuple[List[Reach], List[Reach]]]
                 ) -> Dict[str, Any]:
    kind = rec["kind"]
    if kind == "TP":
        a_s = int(rec["algo_start"]); a_e = int(rec["algo_end"])
        g_s = int(rec["gt_start"]);   g_e = int(rec["gt_end"])
        # Excluded if EITHER side is below MIN_REPORTED_SPAN.
        excl = (is_kinematically_excluded(a_s, a_e)
                or is_kinematically_excluded(g_s, g_e))
        topology, topology_sub, cid = _topology_for_event(
            "TP", (a_s, a_e), (g_s, g_e),
            algo_to_cid, gt_to_cid, components)
        # TP is by construction inside segmentation (matched to a GT reach,
        # which is always inside segmentation). Flag set False for schema
        # uniformity.
        return {
            "kind": "TP",
            "detector": {"start": a_s, "end": a_e},
            "gt": {"start": g_s, "end": g_e},
            "category": None,
            "start_delta": rec["start_delta"],
            "span_delta": rec["span_delta"],
            "kinematically_excluded": bool(excl),
            "outside_gt_segmentation": False,
            "topology": topology,
            "topology_sub": topology_sub,
            "component_id": cid,
        }
    if kind == "FP":
        a_s = int(rec["algo_start"]); a_e = int(rec["algo_end"])
        cat = categorize_fp(a_s, a_e, gts, matched_gt_keys)
        excl = is_kinematically_excluded(a_s, a_e)
        outside = is_outside_gt_segmentation(a_s, gt_boundaries)
        topology, topology_sub, cid = _topology_for_event(
            "FP", (a_s, a_e), None,
            algo_to_cid, gt_to_cid, components)
        return {
            "kind": "FP",
            "detector": {"start": a_s, "end": a_e},
            "gt": None,
            "category": cat,
            "kinematically_excluded": bool(excl),
            "outside_gt_segmentation": bool(outside),
            "topology": topology,
            "topology_sub": topology_sub,
            "component_id": cid,
        }
    if kind == "FN":
        g_s = int(rec["gt_start"]); g_e = int(rec["gt_end"])
        cat = categorize_fn(g_s, g_e, algos)
        excl = is_kinematically_excluded(g_s, g_e)
        topology, topology_sub, cid = _topology_for_event(
            "FN", None, (g_s, g_e),
            algo_to_cid, gt_to_cid, components)
        # FN is by construction inside segmentation (GT reaches always are).
        # Flag set False for schema uniformity.
        return {
            "kind": "FN",
            "detector": None,
            "gt": {"start": g_s, "end": g_e},
            "category": cat,
            "kinematically_excluded": bool(excl),
            "outside_gt_segmentation": False,
            "topology": topology,
            "topology_sub": topology_sub,
            "component_id": cid,
        }
    raise ValueError(f"Unknown kind: {kind}")


def build_manifests(records: List[Dict[str, Any]],
                    corpus_label: str,
                    snapshot_name: str) -> Dict[str, Dict[str, Any]]:
    per_video_reaches = reconstruct_per_video(records)
    records_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        records_by_video[r["video_id"]].append(r)

    manifests: Dict[str, Dict[str, Any]] = {}
    for video_id, video_records in records_by_video.items():
        algos, gts = per_video_reaches.get(video_id, ([], []))
        gt_boundaries = _load_gt_boundaries(corpus_label, video_id)

        # Build the algo-GT overlap graph and its connected components for
        # the new topology labels (Way 1 per Logan 2026-05-20).
        algo_to_cid, gt_to_cid, components = build_topology_components(
            algos, gts)

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
            _build_event(rec, algos, gts, matched_gt_keys, gt_boundaries,
                         algo_to_cid, gt_to_cid, components)
            for rec in video_records
        ]
        def _sort_key(ev):
            if ev["kind"] == "FN":
                return ev["gt"]["start"]
            return ev["detector"]["start"]
        events.sort(key=_sort_key)

        # Per-manifest topology summary: count of each topology label
        # across the video's components (not events). One component = one
        # topology event, regardless of how many TP/FP/FN rows it spans.
        topology_summary: Dict[str, int] = defaultdict(int)
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
            "gt_segmentation": {
                "n_boundaries": len(gt_boundaries),
                "first_frame": gt_boundaries[0] if gt_boundaries else None,
                "last_frame": gt_boundaries[-1] if gt_boundaries else None,
            },
            "topology_summary": dict(topology_summary),
            "events": events,
        }

    return manifests


def archive_existing_manifests() -> Optional[Path]:
    """Move any existing per-corpus manifest dirs into a dated archive.

    Output layout:
      <OUTPUT_ROOT.parent>/<OUTPUT_ROOT.name>_archive/<YYYY-MM-DDTHH-MM>_<tag>/
        calibration_loocv/   (if it existed and had content)
        holdout_2026_05_11/  (if it existed and had content)

    Returns the archive directory path if anything was archived, else None.

    Atomicity: shutil.move is a rename when src and dst are on the same
    filesystem (both are under OUTPUT_ROOT's parent on Y:\\), so each
    corpus dir is moved atomically. A partial-failure scenario could only
    occur across filesystems, which is not the case here.

    Skipped corpora (no existing dir, or empty dir) are not archived --
    nothing to preserve.
    """
    corpora = ["calibration_loocv", "holdout_2026_05_11"]
    populated = [c for c in corpora
                 if (OUTPUT_ROOT / c).exists()
                 and any((OUTPUT_ROOT / c).iterdir())]
    if not populated:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    archive_root = (OUTPUT_ROOT.parent
                    / f"{OUTPUT_ROOT.name}_archive"
                    / f"{timestamp}_{RUN_TAG}")
    archive_root.mkdir(parents=True, exist_ok=True)

    print(f"Archiving prior manifests to: {archive_root}")
    for corpus in populated:
        src = OUTPUT_ROOT / corpus
        dst = archive_root / corpus
        shutil.move(str(src), str(dst))
        print(f"  {corpus}: {len(list(dst.iterdir()))} manifests archived")

    return archive_root


def write_manifests(manifests: Dict[str, Dict[str, Any]],
                    corpus_label: str) -> int:
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

def _event_counts_in(events, kind):
    """Helper: count events of `kind` under unfiltered / span-filt / fully-filt rules."""
    n_un = sum(1 for e in events if e["kind"] == kind)
    n_span = sum(1 for e in events
                 if e["kind"] == kind
                 and not e.get("kinematically_excluded", False))
    # Fully filtered: also exclude outside_gt_segmentation (only meaningful for FP).
    n_full = sum(1 for e in events
                 if e["kind"] == kind
                 and not e.get("kinematically_excluded", False)
                 and not e.get("outside_gt_segmentation", False))
    return n_un, n_span, n_full


def _print_corpus_summary(manifests: Dict[str, Dict[str, Any]],
                          corpus_label: str) -> None:
    total_tp = total_fp = total_fn = 0
    total_tp_s = total_fp_s = total_fn_s = 0
    total_tp_f = total_fp_f = total_fn_f = 0
    total_outside = 0
    print(f"  [{corpus_label}] {len(manifests)} videos:")
    for video_id, m in manifests.items():
        events = m["events"]
        n_tp, n_tp_s, n_tp_f = _event_counts_in(events, "TP")
        n_fp, n_fp_s, n_fp_f = _event_counts_in(events, "FP")
        n_fn, n_fn_s, n_fn_f = _event_counts_in(events, "FN")
        n_outside = sum(1 for e in events
                        if e.get("outside_gt_segmentation", False))
        total_tp += n_tp; total_fp += n_fp; total_fn += n_fn
        total_tp_s += n_tp_s; total_fp_s += n_fp_s; total_fn_s += n_fn_s
        total_tp_f += n_tp_f; total_fp_f += n_fp_f; total_fn_f += n_fn_f
        total_outside += n_outside
        print(f"    {video_id:35} unfilt TP={n_tp:>4} FP={n_fp:>4} FN={n_fn:>4}  |  "
              f"filt TP={n_tp_f:>4} FP={n_fp_f:>4} FN={n_fn_f:>4}  "
              f"(outside_gt_seg FPs dropped: {n_outside})")
    print(f"  totals (unfiltered):                          TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  totals (span>={MIN_REPORTED_SPAN}f filter only):                TP={total_tp_s} FP={total_fp_s} FN={total_fn_s}")
    print(f"  totals (span + outside_gt_segmentation filter): TP={total_tp_f} FP={total_fp_f} FN={total_fn_f}  (outside_gt_seg FPs dropped: {total_outside})")


def main():
    print("=" * 70)
    print(f"Generate FP/FN review manifests -- v8.0.1 (post-mg=0 ship)")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Run tag: {RUN_TAG}")
    print("=" * 70)
    print()

    # Archive any prior manifests at OUTPUT_ROOT/<corpus>/ into a dated
    # archive subfolder before writing fresh ones. Preserves history per
    # the project's archive-then-write convention.
    archive_existing_manifests()
    print()

    print(f"Loading calibration source: {CAL_SOURCE.name}")
    if not CAL_SOURCE.exists():
        raise FileNotFoundError(f"Missing: {CAL_SOURCE}")
    cal_records = _load_calibration_records()
    print(f"  {len(cal_records)} records")

    print(f"Loading holdout source: {GEN_SOURCE.name}")
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
    print(f"Wrote {n_cal} manifests to {OUTPUT_ROOT / 'calibration_loocv'}/")
    print(f"Wrote {n_gen} manifests to {OUTPUT_ROOT / 'holdout_2026_05_11'}/")


if __name__ == "__main__":
    main()
