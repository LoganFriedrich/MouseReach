"""
Slit-crossing tagging + per-reach kinematic-damage metric for the
MouseReach Improvement Process.

Per the saved north-star rule (`mousereach_north_star.md`): the unit of
success is reach kinematic completeness on MEANINGFUL reaches, defined
in `meaningful_reaches_prioritize_not_exclude.md` as reaches that extend
beyond the slit opening (paw crosses the BOXL/BOXR midline at high DLC
likelihood within the GT reach window).

This module adds two utilities, both intended to slot into the existing
snapshot framework -- they read existing snapshot artifacts and write
into the snapshot's metrics directory; they do NOT replace
``compute_reach_detection_metrics`` from ``metrics.py``.

Functions
---------
tag_slit_crossing(gt_dir, h5_search_dir, output_path)
    Pre-process GT reaches into a {video_id: {reach_id: bool}} map of
    slit-crossing flags. Run once per corpus; reusable across snapshots.

compute_kinematic_damage(snapshot_metrics_dir, algo_dir, gt_dir,
                          slit_crossing_path, fragmentation_tol=5,
                          crop_threshold=5)
    On top of an existing snapshot's reach_matches.csv (produced by
    compute_reach_detection_metrics), compute per-reach kinematic-damage
    categories (true_miss, true_fragmentation, start_cropped, end_cropped,
    any_damage) and break out by subset (all / meaningful /
    non_meaningful). Writes kinematic_damage.json into the snapshot's
    metrics directory.

Damage categories (per saved rule
``feedback_only_report_kinematic_damage.md``):
- true_miss: GT reach with no algo reach within the strict matching window
  (status=='fn' in reach_matches.csv).
- true_fragmentation: GT reach is matched, AND another algo reach starts
  inside [gt_start - fragmentation_tol, gt_end). Adjacent FPs starting at
  or after gt_end do NOT count -- the matched algo piece carries the
  reach's kinematic content intact.
- start_cropped: matched, |start_delta| > crop_threshold (default 5f).
- end_cropped: matched, |end_delta| > crop_threshold.
- any_damage: union of the above.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .metrics import (
    Reach,
    _find_algo_file,
    _find_gt_file,
    _load_algo_reaches,
    _load_gt_reaches,
    match_reaches,
)

logger = logging.getLogger(__name__)

HAND_KEYPOINTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
LIK_THRESH = 0.5


def tag_slit_crossing(
    gt_dir: Path,
    h5_search_dir: Path,
    output_path: Path,
) -> Dict[str, Dict[str, bool]]:
    """For each GT reach, determine whether the paw crossed the slit.

    Slit_y is the mean of BOXL_y and BOXR_y across the reach window.
    Slit-crossing = at least one hand keypoint at likelihood >= 0.5 has
    y > slit_y at any frame in [gt_start, gt_end].

    Output JSON map: ``{video_id: {str(reach_id): bool}}``.
    """
    gt_dir = Path(gt_dir)
    h5_search_dir = Path(h5_search_dir)
    output_path = Path(output_path)

    out: Dict[str, Dict[str, bool]] = {}

    for gt_path in sorted(gt_dir.glob("*_unified_ground_truth.json")):
        vid = gt_path.stem.replace("_unified_ground_truth", "")
        h5: Optional[Path] = None
        for p in h5_search_dir.rglob(f"{vid}*DLC*.h5"):
            h5 = p
            break
        if h5 is None:
            logger.warning("No DLC h5 for %s; skipping slit-crossing tag", vid)
            continue

        df_pose = pd.read_hdf(h5)
        df_pose.columns = df_pose.columns.droplevel(0)

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        reaches = gt.get("reaches", {}).get("reaches", [])
        out_vid: Dict[str, bool] = {}
        n = len(df_pose)
        for r in reaches:
            if r.get("exclude_from_analysis"):
                continue
            gs = int(r["start_frame"])
            ge = int(r["end_frame"])
            s = max(0, gs)
            e = min(n, ge + 1)
            if e <= s:
                out_vid[str(r["reach_id"])] = False
                continue
            boxl_y = float(df_pose[("BOXL", "y")].iloc[s:e].mean())
            boxr_y = float(df_pose[("BOXR", "y")].iloc[s:e].mean())
            slit_y = (boxl_y + boxr_y) / 2.0
            crossed = False
            for kp in HAND_KEYPOINTS:
                if (kp, "y") not in df_pose.columns:
                    continue
                yv = df_pose[(kp, "y")].iloc[s:e].to_numpy()
                lk = df_pose[(kp, "likelihood")].iloc[s:e].to_numpy()
                vis = lk >= LIK_THRESH
                if vis.any() and (yv[vis] > slit_y).any():
                    crossed = True
                    break
            out_vid[str(r["reach_id"])] = bool(crossed)
        out[vid] = out_vid

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def compute_kinematic_damage(
    snapshot_metrics_dir: Path,
    algo_dir: Path,
    gt_dir: Path,
    slit_crossing_path: Path,
    fragmentation_tol: int = 5,
    crop_threshold: int = 5,
    window: int = 10,
) -> Dict[str, Any]:
    """Compute per-reach kinematic damage from raw GT + algo files.

    Re-uses ``match_reaches`` from ``metrics.py`` for matching, then layers
    on the kinematic-damage categorisation. Designed to work for any
    snapshot regardless of how its ``reach_matches.csv`` was generated --
    we re-derive matches from GT + algo files directly. Writes
    ``kinematic_damage.json`` into the snapshot's metrics directory.
    """
    snapshot_metrics_dir = Path(snapshot_metrics_dir)
    algo_dir = Path(algo_dir)
    gt_dir = Path(gt_dir)
    slit_crossing_path = Path(slit_crossing_path)

    slit = json.loads(slit_crossing_path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []

    for gt_path in sorted(gt_dir.glob("*_unified_ground_truth.json")):
        vid = gt_path.stem.replace("_unified_ground_truth", "")
        algo_path = _find_algo_file(algo_dir, vid)
        if algo_path is None:
            continue
        gt_reaches = _load_gt_reaches(gt_path)
        algo_reaches = _load_algo_reaches(algo_path)

        # GT reach_id list (matches the order _load_gt_reaches uses)
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        included = [
            r for r in gt_data.get("reaches", {}).get("reaches", [])
            if not r.get("exclude_from_analysis")
        ]
        gt_reach_ids = [str(r["reach_id"]) for r in included]
        slit_for_vid = slit.get(vid, {})

        # All algo (start, end) for fragmentation lookup
        algo_pairs: List[Tuple[int, int]] = [
            (a.start_frame, a.end_frame) for a in algo_reaches
        ]

        results = match_reaches(algo_reaches, gt_reaches, window=window)
        for mr in results:
            # Only GT-rows count for kinematic damage. FP rows have
            # gt_reach_index = -1 (or similar); skip them.
            if mr.gt_reach_index is None or mr.gt_reach_index < 0:
                continue
            gt_idx = mr.gt_reach_index
            rid = gt_reach_ids[gt_idx] if gt_idx < len(gt_reach_ids) else None
            is_meaningful = bool(slit_for_vid.get(rid, False)) if rid else False
            is_true_miss = mr.status == "fn"
            is_fragmented = False
            is_start_cropped = False
            is_end_cropped = False
            if mr.status == "matched":
                gs, ge = mr.gt_start, mr.gt_end
                inside = [
                    a for a in algo_pairs
                    if (gs - fragmentation_tol) <= a[0] < ge
                ]
                is_fragmented = len(inside) >= 2
                if abs(mr.start_delta) > crop_threshold:
                    is_start_cropped = True
                if abs(mr.end_delta) > crop_threshold:
                    is_end_cropped = True
            is_any_damage = (
                is_true_miss
                or is_fragmented
                or is_start_cropped
                or is_end_cropped
            )
            rows.append({
                "video_id": vid,
                "gt_reach_id": rid,
                "is_meaningful": is_meaningful,
                "is_true_miss": is_true_miss,
                "is_fragmented": is_fragmented,
                "is_start_cropped": is_start_cropped,
                "is_end_cropped": is_end_cropped,
                "is_any_damage": is_any_damage,
            })

    df_gt = pd.DataFrame(rows)

    def _subset_stats(sub: pd.DataFrame) -> Dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {"n": 0}
        return {
            "n": int(n),
            "n_true_miss": int(sub["is_true_miss"].sum()),
            "n_fragmented": int(sub["is_fragmented"].sum()),
            "n_start_cropped_gt_5f": int(sub["is_start_cropped"].sum()),
            "n_end_cropped_gt_5f": int(sub["is_end_cropped"].sum()),
            "n_any_damage": int(sub["is_any_damage"].sum()),
            "true_miss_pct": round(100 * sub["is_true_miss"].mean(), 2),
            "fragmented_pct": round(100 * sub["is_fragmented"].mean(), 2),
            "start_cropped_5f_pct": round(100 * sub["is_start_cropped"].mean(), 2),
            "end_cropped_5f_pct": round(100 * sub["is_end_cropped"].mean(), 2),
            "any_damage_pct": round(100 * sub["is_any_damage"].mean(), 2),
        }

    out: Dict[str, Any] = {
        "all": _subset_stats(df_gt),
        "meaningful": _subset_stats(df_gt[df_gt["is_meaningful"]]),
        "non_meaningful": _subset_stats(df_gt[~df_gt["is_meaningful"]]),
        "params": {
            "fragmentation_tol": fragmentation_tol,
            "crop_threshold": crop_threshold,
            "slit_crossing_path": str(slit_crossing_path),
        },
    }

    out_path = snapshot_metrics_dir / "kinematic_damage.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
