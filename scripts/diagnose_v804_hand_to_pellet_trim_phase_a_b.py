"""Phase A step (b): validate hand_to_pellet_dist trim on corpus-wide TPs + TOLs.

Step (a) found that on CNT0316_P3 end-extended TOL tails, hand_to_pellet_dist
averages ~7 px (hand on pellet) while clean TP tails average ~36 px (hand
retracted away). Step (b) tests whether a "walk inward from algo_end while
hand_to_pellet_dist < threshold AND pellet_lk > min_lk" trim would:

  - Successfully shorten the 6 known TOL targets to land near GT_end
  - NOT damage clean TPs on the same videos
  - NOT damage clean TPs across the rest of the corpus

Sweep the threshold across {6, 8, 10, 12, 15} px.

Outputs per TP:
  - new_end_delta vs gt_end (after the trim would fire)
  - whether the trim would damage this TP (new_end pushed before gt_end)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_Y_SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
if _Y_SRC not in sys.path:
    sys.path.insert(0, _Y_SRC)
for _mod in [m for m in list(sys.modules) if m.startswith("mousereach")]:
    del sys.modules[_mod]

import numpy as np
import pandas as pd

from mousereach.reach.core.geometry import load_dlc
from mousereach.reach.v8.postprocess import compute_paw_mean_lk

HOL_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\dlc"
)
HOL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\holdout_2026_05_11"
)
CAL_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\validation_runs\DLC_2026_03_27\Processing\updated dlc model 3.1"
)
CAL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\calibration_loocv"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\Improvement_Snapshots\reach_detection"
    r"\v8.0.4_dev_hand_to_pellet_trim_phase_a_b"
)
DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000.h5"

HAND_KEYPOINTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
PELLET_LK_MIN = 0.5  # require pellet visible to apply trim

THRESHOLDS_PX = [6.0, 8.0, 10.0, 12.0, 15.0]
SUSTAIN_NS = [1, 2, 3, 5, 8]  # sweep sustain-N gate
# Combined cells: (threshold, sustain_n)
CELLS = [(thr, n) for thr in THRESHOLDS_PX for n in SUSTAIN_NS]


def compute_hand_to_pellet(dlc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (hand_to_pellet_dist, pellet_lk, paw_mean_lk) per frame."""
    hx = np.mean([dlc[f"{kp}_x"].to_numpy() for kp in HAND_KEYPOINTS], axis=0)
    hy = np.mean([dlc[f"{kp}_y"].to_numpy() for kp in HAND_KEYPOINTS], axis=0)
    px = dlc["Pellet_x"].to_numpy()
    py = dlc["Pellet_y"].to_numpy()
    dist = np.sqrt((px - hx) ** 2 + (py - hy) ** 2)
    pellet_lk = dlc["Pellet_likelihood"].to_numpy()
    paw_lk = compute_paw_mean_lk(dlc)
    return dist, pellet_lk, paw_lk


def simulate_trim(
    algo_start: int,
    algo_end: int,
    hand_to_pellet: np.ndarray,
    pellet_lk: np.ndarray,
    threshold: float,
    sustain_n: int = 1,
    min_span: int = 3,
) -> int:
    """Simulate walk-inward trim. Walk back from algo_end while:
      (a) frame is within [algo_start + min_span - 1, algo_end]
      (b) hand_to_pellet_dist[frame] < threshold
      (c) pellet_lk[frame] >= PELLET_LK_MIN

    Returns the new end frame (may equal algo_end if no trim).
    """
    new_end = algo_end
    while new_end > algo_start + min_span - 1:
        # Check sustained window [new_end - sustain_n + 1, new_end]
        ws = new_end - sustain_n + 1
        if ws < algo_start:
            break
        in_window = True
        for f in range(ws, new_end + 1):
            if f >= len(hand_to_pellet):
                in_window = False
                break
            if hand_to_pellet[f] >= threshold:
                in_window = False
                break
            if pellet_lk[f] < PELLET_LK_MIN:
                in_window = False
                break
        if in_window:
            new_end -= 1
        else:
            break
    return new_end


@dataclass
class TrimResult:
    corpus: str
    video: str
    kind: str  # "TOL_end_ext" / "clean_TP" / "other_TP"
    algo: Tuple[int, int]
    gt: Tuple[int, int]
    original_end_delta: int  # algo_end - gt_end
    new_end_per_threshold: Dict[str, int]  # str(threshold) -> new_end
    new_end_delta_per_threshold: Dict[str, int]  # str(threshold) -> new_end - gt_end


def _video_dlc_path(video_id: str, corpus: str) -> Path:
    return (CAL_DLC_DIR if corpus == "calibration" else HOL_DLC_DIR) / f"{video_id}{DLC_SUFFIX}"


def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Discover manifest files
    results: List[TrimResult] = []
    tol_target_outcomes: List[TrimResult] = []
    n_videos = 0
    n_videos_failed_dlc = 0

    for label, dirpath in (("calibration", CAL_MANIFEST_DIR), ("holdout", HOL_MANIFEST_DIR)):
        for f in sorted(dirpath.glob("*.json")):
            with open(f) as fh:
                d = json.load(fh)
            video = d["video_id"]
            dlc_path = _video_dlc_path(video, label)
            if not dlc_path.exists():
                n_videos_failed_dlc += 1
                continue
            n_videos += 1
            print(f"  {label}/{video}")
            dlc = load_dlc(dlc_path)
            dist, pellet_lk, paw_lk = compute_hand_to_pellet(dlc)

            # Collect events
            events = d.get("events", [])
            # Group by component_id to find TOL pairs
            comps: Dict[int, List[Dict[str, Any]]] = {}
            for e in events:
                cid = e.get("component_id")
                if cid is not None:
                    comps.setdefault(cid, []).append(e)

            # End-extended TOL pairs
            for cid, evs in comps.items():
                fp_tols = [
                    e for e in evs
                    if e.get("kind") == "FP" and e.get("topology") == "TOLERANCE_ERROR"
                ]
                fn_tols = [
                    e for e in evs
                    if e.get("kind") == "FN" and e.get("topology") == "TOLERANCE_ERROR"
                ]
                if not (len(fp_tols) == 1 and len(fn_tols) == 1 and len(evs) == 2):
                    continue
                fp = fp_tols[0]
                fn = fn_tols[0]
                algo_s, algo_e = fp["detector"]["start"], fp["detector"]["end"]
                gt_s, gt_e = fn["gt"]["start"], fn["gt"]["end"]
                end_delta = algo_e - gt_e
                if end_delta < 3:
                    continue
                # Simulate trim across (threshold, sustain_n) cells
                new_ends = {}
                new_end_deltas = {}
                for thr, sn in CELLS:
                    ne = simulate_trim(algo_s, algo_e, dist, pellet_lk, thr, sustain_n=sn)
                    key = f"thr{thr:.1f}_sn{sn}"
                    new_ends[key] = ne
                    new_end_deltas[key] = ne - gt_e
                rec = TrimResult(
                    corpus=label,
                    video=video,
                    kind="TOL_end_ext",
                    algo=(algo_s, algo_e),
                    gt=(gt_s, gt_e),
                    original_end_delta=end_delta,
                    new_end_per_threshold=new_ends,
                    new_end_delta_per_threshold=new_end_deltas,
                )
                tol_target_outcomes.append(rec)
                results.append(rec)

            # All TP events
            for e in events:
                if e.get("kind") != "TP":
                    continue
                algo = e.get("detector")
                gt = e.get("gt")
                if not algo or not gt:
                    continue
                algo_s, algo_e = algo["start"], algo["end"]
                gt_s, gt_e = gt["start"], gt["end"]
                start_d = e.get("start_delta")
                span_d = e.get("span_delta")
                kind = "clean_TP" if (start_d == 0 and span_d == 0) else "other_TP"
                # Simulate trim across (threshold, sustain_n) cells
                new_ends = {}
                new_end_deltas = {}
                for thr, sn in CELLS:
                    ne = simulate_trim(algo_s, algo_e, dist, pellet_lk, thr, sustain_n=sn)
                    key = f"thr{thr:.1f}_sn{sn}"
                    new_ends[key] = ne
                    new_end_deltas[key] = ne - gt_e
                rec = TrimResult(
                    corpus=label,
                    video=video,
                    kind=kind,
                    algo=(algo_s, algo_e),
                    gt=(gt_s, gt_e),
                    original_end_delta=algo_e - gt_e,
                    new_end_per_threshold=new_ends,
                    new_end_delta_per_threshold=new_end_deltas,
                )
                results.append(rec)

    print()
    print(f"Processed videos: {n_videos} (failed DLC: {n_videos_failed_dlc})")
    print(f"TOL_end_ext targets: {sum(1 for r in results if r.kind == 'TOL_end_ext')}")
    print(f"clean_TP controls: {sum(1 for r in results if r.kind == 'clean_TP')}")
    print(f"other_TP (TP but with non-zero start_delta or span_delta): "
          f"{sum(1 for r in results if r.kind == 'other_TP')}")
    print()

    # === Grid summary: TOL recoveries vs Clean TP losses vs Other TP losses ===
    clean_tps = [r for r in results if r.kind == "clean_TP"]
    other_tps = [r for r in results if r.kind == "other_TP"]
    tol_recs = [r for r in results if r.kind == "TOL_end_ext"]

    def tol_resolved(key):
        return sum(1 for r in tol_recs if -5 <= r.new_end_delta_per_threshold[key] <= 5)

    def tol_over_trim(key):
        return sum(1 for r in tol_recs if r.new_end_delta_per_threshold[key] < -5)

    def tp_lose(tps, key):
        n = 0
        for r in tps:
            d = r.new_end_delta_per_threshold[key]
            gt_span = r.gt[1] - r.gt[0] + 1
            span_tol = max(5, 0.5 * gt_span)
            if r.kind == "clean_TP":
                if abs(d) > span_tol:
                    n += 1
            else:
                if abs(d) > span_tol and abs(r.original_end_delta) <= span_tol:
                    n += 1
        return n

    print("=" * 92)
    print("Grid: (threshold x sustain_n) -- TOL recoveries vs Clean TP losses vs Net")
    print("=" * 92)
    print(f"{'thr':>5} {'sn':>4} {'TOL_recov':>10s} {'TOL_overtrim':>13s} "
          f"{'cleanTP_lose':>13s} {'otherTP_lose':>13s} {'NET':>6s}")
    for thr in THRESHOLDS_PX:
        for sn in SUSTAIN_NS:
            key = f"thr{thr:.1f}_sn{sn}"
            tr = tol_resolved(key)
            tov = tol_over_trim(key)
            ctl = tp_lose(clean_tps, key)
            otl = tp_lose(other_tps, key)
            net = tr - ctl - otl - tov
            print(
                f"  {thr:>4.1f} {sn:>4d} {tr:>10d} {tov:>13d} "
                f"{ctl:>13d} {otl:>13d} {net:>+6d}"
            )

    print()
    print("=" * 78)
    print("Per-TOL detail across (thr, sustain_n) cells")
    print("=" * 78)
    for r in sorted(
        tol_recs,
        key=lambda x: (x.corpus, x.video, x.algo[0])
    ):
        print(f"  {r.video} algo={r.algo} gt={r.gt} orig_ed={r.original_end_delta:+d}")
        for thr in THRESHOLDS_PX:
            cells_str = ", ".join(
                f"sn{sn}=>{r.new_end_delta_per_threshold[f'thr{thr:.1f}_sn{sn}']:+d}"
                for sn in SUSTAIN_NS
            )
            print(f"    thr={thr:>4.1f}: {cells_str}")

    # Write JSON
    out = {
        "thresholds_px": THRESHOLDS_PX,
        "sustain_ns": SUSTAIN_NS,
        "pellet_lk_min": PELLET_LK_MIN,
        "n_videos_processed": n_videos,
        "results": [asdict(r) for r in results],
    }
    out_path = metrics_dir / "trim_simulation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
