"""For each exhaustive GT video, compute paw bodypart features for
every frame and split by in-reach (real paw) vs out-of-reach (DLC
noise -- per memory: in exhaustive videos, every real reach is
labeled, so any paw detection outside GT reach windows is false
positive).

Goal: identify features that cleanly separate real paw from DLC
noise, so we can build a stricter `paw_past_y` predicate.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.features import load_dlc_h5

QUARANTINE = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")
GT_DIR = QUARANTINE / "gt"
DLC_DIR = QUARANTINE / "dlc"
CORPUS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\_corpus\2026-04-30_restart_inventory")

PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")


def find_dlc(vid):
    return next(DLC_DIR.glob(f"{vid}DLC_*.h5"))


def load_gt(vid):
    return json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))


def main():
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]

    exhaustive_videos = []
    for vid in train_pool_ids:
        gt = load_gt(vid)
        if gt.get("reaches", {}).get("exhaustive"):
            exhaustive_videos.append(vid)
    print(f"Exhaustive videos: {len(exhaustive_videos)} of {len(train_pool_ids)}")

    in_reach_features = {
        "n_above_05": [],
        "n_above_07": [],
        "n_above_09": [],
        "max_lk": [],
        "min_lk": [],
        "mean_lk": [],
        "bbox_width": [],   # px
        "bbox_height": [],
        "max_pair_dist": [],  # max pairwise distance among bodyparts with lk>=0.5
        "median_pair_dist": [],
    }
    out_reach_features = {k: [] for k in in_reach_features}

    n_frames_in = 0
    n_frames_out = 0

    for vid in exhaustive_videos:
        dlc = load_dlc_h5(find_dlc(vid))
        gt = load_gt(vid)
        gt_b = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
        if not gt_b or len(gt_b) < 2:
            continue
        # Build set of in-reach frames (across all reaches in this video)
        reach_mask_total = np.zeros(int(dlc.shape[0]), dtype=bool)
        for r in gt.get("reaches", {}).get("reaches", []) or []:
            s = r.get("start_frame"); e = r.get("end_frame")
            if s is None or e is None:
                continue
            s = max(0, int(s))
            e = min(reach_mask_total.shape[0] - 1, int(e))
            if e >= s:
                reach_mask_total[s:e + 1] = True
        # Restrict analysis to frames inside any segment (clean zone)
        seg_mask = np.zeros_like(reach_mask_total)
        for i in range(len(gt_b) - 1):
            s, e = gt_b[i], gt_b[i + 1] - 1 - 5  # transition zone trim
            if e >= s:
                seg_mask[s:e + 1] = True
        # Per-bodypart arrays
        bp_x = np.stack([dlc[f"{bp}_x"].to_numpy(dtype=float) for bp in PAW_BODYPARTS])
        bp_y = np.stack([dlc[f"{bp}_y"].to_numpy(dtype=float) for bp in PAW_BODYPARTS])
        bp_lk = np.stack([dlc[f"{bp}_likelihood"].to_numpy(dtype=float) for bp in PAW_BODYPARTS])
        n_total = int(seg_mask.sum())

        # Compute features only at relevant frames
        for f in range(reach_mask_total.shape[0]):
            if not seg_mask[f]:
                continue
            lks = bp_lk[:, f]
            xs = bp_x[:, f]
            ys = bp_y[:, f]
            # Skip frame if pellet has all zero detection (very unlikely
            # but safe). All paw bodyparts always have some lk.
            n_05 = int((lks >= 0.5).sum())
            n_07 = int((lks >= 0.7).sum())
            n_09 = int((lks >= 0.9).sum())
            max_lk = float(lks.max())
            min_lk = float(lks.min())
            mean_lk = float(lks.mean())
            # Bbox + pairwise dists at lk >= 0.5
            mask = lks >= 0.5
            if mask.sum() >= 2:
                xs_m = xs[mask]
                ys_m = ys[mask]
                bbox_w = float(xs_m.max() - xs_m.min())
                bbox_h = float(ys_m.max() - ys_m.min())
                # Pairwise distances
                pair_dists = []
                for i in range(len(xs_m)):
                    for j in range(i + 1, len(xs_m)):
                        d = float(np.sqrt((xs_m[i] - xs_m[j]) ** 2
                                          + (ys_m[i] - ys_m[j]) ** 2))
                        pair_dists.append(d)
                max_pd = float(max(pair_dists))
                med_pd = float(np.median(pair_dists))
            else:
                bbox_w = bbox_h = max_pd = med_pd = -1.0  # sentinel: not enough points
            target = in_reach_features if reach_mask_total[f] else out_reach_features
            target["n_above_05"].append(n_05)
            target["n_above_07"].append(n_07)
            target["n_above_09"].append(n_09)
            target["max_lk"].append(max_lk)
            target["min_lk"].append(min_lk)
            target["mean_lk"].append(mean_lk)
            target["bbox_width"].append(bbox_w)
            target["bbox_height"].append(bbox_h)
            target["max_pair_dist"].append(max_pd)
            target["median_pair_dist"].append(med_pd)
            if reach_mask_total[f]:
                n_frames_in += 1
            else:
                n_frames_out += 1

    print(f"\n  in-reach frames: {n_frames_in}")
    print(f"  out-of-reach frames: {n_frames_out}")
    print()

    def stats(arr, key, skip_neg=False):
        vals = sorted(v for v in arr if not (skip_neg and v < 0))
        if not vals: return "(empty)"
        n = len(vals)
        def pct(p): return vals[max(0, min(n - 1, int(p / 100 * (n - 1))))]
        return f"min={pct(0):.2f} p1={pct(1):.2f} p5={pct(5):.2f} p25={pct(25):.2f} p50={pct(50):.2f} p75={pct(75):.2f} p95={pct(95):.2f} p99={pct(99):.2f} max={pct(100):.2f}"

    print("Feature distributions: IN-REACH (real paw) vs OUT-OF-REACH (DLC noise)")
    print()
    for key in ["n_above_05", "n_above_07", "n_above_09",
                "max_lk", "min_lk", "mean_lk",
                "bbox_width", "bbox_height", "max_pair_dist", "median_pair_dist"]:
        skip_neg = key in ("bbox_width", "bbox_height", "max_pair_dist", "median_pair_dist")
        print(f"  {key}:")
        print(f"    in-reach : {stats(in_reach_features[key], key, skip_neg)}")
        print(f"    out-reach: {stats(out_reach_features[key], key, skip_neg)}")


if __name__ == "__main__":
    main()
