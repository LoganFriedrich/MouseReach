"""Phase A diagnostic: feature signatures in the tail of end-extended TOLs.

Hypothesis (Logan, subjective): the algo extends past GT_end because DLC
keeps tracking the paw after the reach is biologically over. Trailing-trim
doesn't fire because paw_lk stays high. We want to find a SECOND signal
that distinguishes "real reach end" from "DLC still tracking after reach
is over".

Per Logan's request, look at:
  - the TOL pair itself (algo span vs GT span)
  - the next algo reach AFTER this one in the same video (gap distance)
  - all DLC-derived features over the tail window for both TOL and clean-TP
    controls on the same videos

Read-only. No algo changes.

Output: per-case feature trajectories (frame-by-frame) + aggregate
TOL-vs-TP-tail comparison.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
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
from mousereach.reach.v8.postprocess import (
    compute_hand_to_boxl_norm_pos,
    compute_paw_mean_lk,
)

# Paths ----------------------------------------------------------------------
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
    r"\v8.0.4_dev_tolerance_end_tail_features_phase_a"
)
DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000.h5"

# Tail window: extend gt_end + 30 to capture post-reach behavior AND
# transition to next reach.
TAIL_PRE = 5     # frames before gt_end
TAIL_POST = 30   # frames after gt_end (or algo_end, whichever is later)

# Threshold to flag a TOL as end-extended
END_EXTEND_MIN_DELTA = 3


# DLC helpers ----------------------------------------------------------------
HAND_KEYPOINTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
SLIT_KEYPOINTS = ("SABL", "SABR", "SATL", "SATR")


def compute_features(dlc: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute all per-frame features over a full video."""
    n = len(dlc)
    feats: Dict[str, np.ndarray] = {}

    # Hand centroid x, y (mean of 4 hand keypoints)
    hx = np.mean([dlc[f"{kp}_x"].to_numpy() for kp in HAND_KEYPOINTS], axis=0)
    hy = np.mean([dlc[f"{kp}_y"].to_numpy() for kp in HAND_KEYPOINTS], axis=0)
    feats["hand_x"] = hx
    feats["hand_y"] = hy

    # Hand velocity (signed, frame-to-frame)
    feats["hand_vx"] = np.concatenate([[0.0], np.diff(hx)])
    feats["hand_vy"] = np.concatenate([[0.0], np.diff(hy)])
    feats["hand_speed"] = np.sqrt(feats["hand_vx"] ** 2 + feats["hand_vy"] ** 2)

    # paw_mean_lk
    feats["paw_lk"] = compute_paw_mean_lk(dlc)

    # norm_pos (hand-to-BoxL normalized by apparatus width)
    feats["norm_pos"] = compute_hand_to_boxl_norm_pos(dlc)

    # Pellet
    feats["pellet_lk"] = dlc["Pellet_likelihood"].to_numpy()
    feats["pellet_x"] = dlc["Pellet_x"].to_numpy()
    feats["pellet_y"] = dlc["Pellet_y"].to_numpy()

    # Hand-to-pellet distance (raw px)
    feats["hand_to_pellet_dx"] = feats["pellet_x"] - hx
    feats["hand_to_pellet_dy"] = feats["pellet_y"] - hy
    feats["hand_to_pellet_dist"] = np.sqrt(
        feats["hand_to_pellet_dx"] ** 2 + feats["hand_to_pellet_dy"] ** 2
    )

    # Nose
    feats["nose_lk"] = dlc["Nose_likelihood"].to_numpy()
    feats["nose_x"] = dlc["Nose_x"].to_numpy()
    feats["nose_y"] = dlc["Nose_y"].to_numpy()

    # Slit center (avg of 4 slit corners)
    sx = np.mean([dlc[f"{kp}_x"].to_numpy() for kp in SLIT_KEYPOINTS], axis=0)
    sy = np.mean([dlc[f"{kp}_y"].to_numpy() for kp in SLIT_KEYPOINTS], axis=0)
    feats["slit_x"] = sx
    feats["slit_y"] = sy

    # Nose-to-slit (Logan apparatus: y = reach-extension axis; high y = past slit)
    feats["nose_to_slit_dist"] = np.sqrt((feats["nose_x"] - sx) ** 2 + (feats["nose_y"] - sy) ** 2)

    # Hand position relative to slit center
    feats["hand_to_slit_dy"] = hy - sy  # positive = hand past slit
    feats["hand_to_slit_dist"] = np.sqrt((hx - sx) ** 2 + (hy - sy) ** 2)

    return feats


# Manifest parsing -----------------------------------------------------------
@dataclass
class TolPair:
    corpus: str
    video: str
    component_id: int
    algo: Tuple[int, int]
    gt: Tuple[int, int]
    start_delta: int
    end_delta: int
    span_delta: int
    fn_category: str
    next_algo_start: Optional[int] = None  # start of next algo reach after this
    next_algo_end: Optional[int] = None
    next_algo_kind: Optional[str] = None  # TP/FP
    next_algo_gt: Optional[Tuple[int, int]] = None  # GT matched to next, if any


def find_tol_pairs(manifest_path: Path, corpus: str) -> List[TolPair]:
    """Find TOL_ERROR components in a manifest and return as TolPair records."""
    with open(manifest_path) as f:
        d = json.load(f)
    video = d["video_id"]

    # Group events by component
    comps: Dict[int, List[Dict[str, Any]]] = {}
    all_algo_events: List[Dict[str, Any]] = []
    for e in d.get("events", []):
        cid = e.get("component_id")
        if cid is not None:
            comps.setdefault(cid, []).append(e)
        if e.get("detector") is not None:
            all_algo_events.append(e)

    # Sort all algo events by start
    all_algo_events.sort(key=lambda e: e["detector"]["start"])

    results: List[TolPair] = []
    for cid, evs in comps.items():
        # Must be exactly 1 FP + 1 FN both TOLERANCE_ERROR
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
        algo = fp["detector"]
        gt = fn["gt"]
        algo_s, algo_e = algo["start"], algo["end"]
        gt_s, gt_e = gt["start"], gt["end"]

        # Find next algo reach (by start) after this algo_e
        next_event = None
        for e in all_algo_events:
            if e["detector"]["start"] > algo_e:
                next_event = e
                break

        next_start = next_event["detector"]["start"] if next_event else None
        next_end = next_event["detector"]["end"] if next_event else None
        next_kind = next_event["kind"] if next_event else None
        next_gt = None
        if next_event and next_event.get("gt"):
            next_gt = (next_event["gt"]["start"], next_event["gt"]["end"])

        results.append(TolPair(
            corpus=corpus,
            video=video,
            component_id=cid,
            algo=(algo_s, algo_e),
            gt=(gt_s, gt_e),
            start_delta=algo_s - gt_s,
            end_delta=algo_e - gt_e,
            span_delta=(algo_e - algo_s + 1) - (gt_e - gt_s + 1),
            fn_category=fn.get("category", ""),
            next_algo_start=next_start,
            next_algo_end=next_end,
            next_algo_kind=next_kind,
            next_algo_gt=next_gt,
        ))
    return results


def find_clean_tps(manifest_path: Path, corpus: str, n_max: int = 5) -> List[Dict[str, Any]]:
    """Find clean TPs (start_delta=0 AND span_delta=0) for use as controls."""
    with open(manifest_path) as f:
        d = json.load(f)
    video = d["video_id"]
    tps = []
    all_algo_events = []
    for e in d.get("events", []):
        if e.get("detector") is not None:
            all_algo_events.append(e)
    all_algo_events.sort(key=lambda e: e["detector"]["start"])

    for e in d.get("events", []):
        if e.get("kind") != "TP":
            continue
        if e.get("start_delta") == 0 and e.get("span_delta") == 0:
            algo = e["detector"]
            gt = e["gt"]
            algo_e = algo["end"]
            # Find next algo reach
            next_start = None
            for ne in all_algo_events:
                if ne["detector"]["start"] > algo_e:
                    next_start = ne["detector"]["start"]
                    break
            tps.append({
                "corpus": corpus,
                "video": video,
                "algo": (algo["start"], algo_e),
                "gt": (gt["start"], gt["end"]),
                "next_algo_start": next_start,
            })
        if len(tps) >= n_max:
            break
    return tps


# Main probe -----------------------------------------------------------------
FEATURE_NAMES = [
    "paw_lk",
    "norm_pos",
    "hand_x",
    "hand_y",
    "hand_vx",
    "hand_vy",
    "hand_speed",
    "pellet_lk",
    "hand_to_pellet_dist",
    "nose_lk",
    "nose_to_slit_dist",
    "hand_to_slit_dy",
    "hand_to_slit_dist",
]


def _slice_features(
    feats: Dict[str, np.ndarray], gt_end: int, algo_end: int, n_total_frames: int
) -> Dict[str, List[float]]:
    """Slice each feature over [gt_end - TAIL_PRE, max(gt_end, algo_end) + TAIL_POST]."""
    s = max(0, gt_end - TAIL_PRE)
    e = min(n_total_frames - 1, max(gt_end, algo_end) + TAIL_POST)
    out: Dict[str, List[float]] = {"frame": list(range(s, e + 1))}
    for fname in FEATURE_NAMES:
        if fname not in feats:
            continue
        arr = feats[fname][s : e + 1]
        out[fname] = [float(v) for v in arr]
    return out


def _summary_stats_in_tail(
    feats: Dict[str, np.ndarray], gt_end: int, algo_end: int
) -> Dict[str, float]:
    """Compute summary stats over the tail (gt_end+1 .. algo_end)."""
    out: Dict[str, float] = {}
    if algo_end <= gt_end:
        # TP control: tail = [gt_end+1, gt_end+10] for parity
        s, e = gt_end + 1, min(len(feats["paw_lk"]) - 1, gt_end + 10)
    else:
        s, e = gt_end + 1, min(len(feats["paw_lk"]) - 1, algo_end)
    if s > e:
        return out
    for fname in FEATURE_NAMES:
        if fname not in feats:
            continue
        arr = feats[fname][s : e + 1]
        arr_valid = arr[~np.isnan(arr)]
        if len(arr_valid) == 0:
            out[f"{fname}_mean"] = float("nan")
            out[f"{fname}_std"] = float("nan")
            continue
        out[f"{fname}_mean"] = float(np.mean(arr_valid))
        out[f"{fname}_std"] = float(np.std(arr_valid))
    return out


def _video_dlc_path(video_id: str, dlc_root: Path) -> Path:
    return dlc_root / f"{video_id}{DLC_SUFFIX}"


def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Find all end-extended TOL pairs
    tol_pairs: List[TolPair] = []
    for label, dirpath in (("calibration", CAL_MANIFEST_DIR), ("holdout", HOL_MANIFEST_DIR)):
        for f in sorted(dirpath.glob("*.json")):
            for tp in find_tol_pairs(f, label):
                if tp.end_delta >= END_EXTEND_MIN_DELTA:
                    tol_pairs.append(tp)

    print(f"End-extended TOL pairs (end_delta >= {END_EXTEND_MIN_DELTA}): {len(tol_pairs)}")
    by_corpus_video: Counter = Counter()
    for tp in tol_pairs:
        by_corpus_video[(tp.corpus, tp.video)] += 1
    for (c, v), n in by_corpus_video.most_common():
        print(f"  {c}: {v}: {n}")
    print()

    # Find clean TPs for each affected video (controls)
    videos_with_tols: List[Tuple[str, str]] = sorted({(tp.corpus, tp.video) for tp in tol_pairs})
    clean_tps_by_video: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for corpus, video in videos_with_tols:
        mdir = CAL_MANIFEST_DIR if corpus == "calibration" else HOL_MANIFEST_DIR
        clean_tps_by_video[(corpus, video)] = find_clean_tps(
            mdir / f"{video}.json", corpus, n_max=5
        )

    # Load DLC + compute features per video
    feature_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    nframes_cache: Dict[Tuple[str, str], int] = {}
    for corpus, video in videos_with_tols:
        dlc_root = CAL_DLC_DIR if corpus == "calibration" else HOL_DLC_DIR
        dlc_path = _video_dlc_path(video, dlc_root)
        if not dlc_path.exists():
            print(f"WARN: DLC not found for {corpus}/{video} at {dlc_path}")
            continue
        print(f"  loading + features: {corpus}/{video}")
        dlc = load_dlc(dlc_path)
        feats = compute_features(dlc)
        feature_cache[(corpus, video)] = feats
        nframes_cache[(corpus, video)] = len(dlc)

    # Build per-case dumps
    per_case_dumps: List[Dict[str, Any]] = []
    tol_tail_stats: List[Dict[str, Any]] = []
    tp_tail_stats: List[Dict[str, Any]] = []

    print()
    print("=== TOL cases ===")
    for tp in tol_pairs:
        key = (tp.corpus, tp.video)
        if key not in feature_cache:
            continue
        feats = feature_cache[key]
        nframes = nframes_cache[key]
        traj = _slice_features(feats, tp.gt[1], tp.algo[1], nframes)
        stats = _summary_stats_in_tail(feats, tp.gt[1], tp.algo[1])
        rec = {
            "kind": "TOL",
            **asdict(tp),
            "tail_stats": stats,
            "trajectory": traj,
        }
        per_case_dumps.append(rec)
        tol_tail_stats.append({
            "kind": "TOL",
            "corpus": tp.corpus,
            "video": tp.video,
            "cid": tp.component_id,
            "end_delta": tp.end_delta,
            **stats,
        })
        gap_str = (
            f"gap_to_next={tp.next_algo_start - tp.algo[1]:>4}" if tp.next_algo_start
            else "no_next_reach"
        )
        print(
            f"  {tp.corpus:11s} {tp.video:30s} cid={tp.component_id:>4} "
            f"algo={tp.algo} gt={tp.gt} ed={tp.end_delta:+3d} {gap_str}"
        )
        print(
            f"    tail paw_lk_mean={stats.get('paw_lk_mean', float('nan')):.3f} "
            f"norm_pos_mean={stats.get('norm_pos_mean', float('nan')):.3f} "
            f"hand_speed_mean={stats.get('hand_speed_mean', float('nan')):.2f} "
            f"hand_to_pellet_mean={stats.get('hand_to_pellet_dist_mean', float('nan')):.1f}"
        )

    print()
    print("=== Clean-TP controls (per video) ===")
    for (corpus, video), tps in clean_tps_by_video.items():
        if (corpus, video) not in feature_cache:
            continue
        feats = feature_cache[(corpus, video)]
        nframes = nframes_cache[(corpus, video)]
        for tp_c in tps:
            traj = _slice_features(feats, tp_c["gt"][1], tp_c["algo"][1], nframes)
            stats = _summary_stats_in_tail(feats, tp_c["gt"][1], tp_c["algo"][1])
            rec = {
                "kind": "TP",
                "corpus": corpus,
                "video": video,
                "algo": tp_c["algo"],
                "gt": tp_c["gt"],
                "next_algo_start": tp_c["next_algo_start"],
                "tail_stats": stats,
                "trajectory": traj,
            }
            per_case_dumps.append(rec)
            tp_tail_stats.append({
                "kind": "TP",
                "corpus": corpus,
                "video": video,
                "algo_start": tp_c["algo"][0],
                **stats,
            })
        print(f"  {corpus}/{video}: {len(tps)} clean TPs controls")

    # Aggregate tail-feature comparison: TOL vs TP
    print()
    print("=== Aggregate: TOL tail vs TP tail (means / stds) ===")
    print(f"{'feature':25s} {'TOL_mean (n=' + str(len(tol_tail_stats)) + ')':>20s} "
          f"{'TP_mean (n=' + str(len(tp_tail_stats)) + ')':>20s} {'delta':>10s}")
    for fname in FEATURE_NAMES:
        tol_vals = [s.get(f"{fname}_mean", float("nan")) for s in tol_tail_stats]
        tp_vals = [s.get(f"{fname}_mean", float("nan")) for s in tp_tail_stats]
        tol_vals = [v for v in tol_vals if not np.isnan(v)]
        tp_vals = [v for v in tp_vals if not np.isnan(v)]
        if not tol_vals or not tp_vals:
            continue
        tol_m = np.mean(tol_vals)
        tp_m = np.mean(tp_vals)
        delta = tol_m - tp_m
        print(f"  {fname:25s} {tol_m:>20.3f} {tp_m:>20.3f} {delta:>+10.3f}")

    # Write JSON
    out = {
        "n_tol_cases": len([d for d in per_case_dumps if d["kind"] == "TOL"]),
        "n_tp_cases": len([d for d in per_case_dumps if d["kind"] == "TP"]),
        "feature_names": FEATURE_NAMES,
        "tail_pre_frames": TAIL_PRE,
        "tail_post_frames": TAIL_POST,
        "per_case": per_case_dumps,
        "tol_tail_stats": tol_tail_stats,
        "tp_tail_stats": tp_tail_stats,
    }
    out_path = metrics_dir / "feature_dumps.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
