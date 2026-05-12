"""Calibrate codetection triage thresholds against the empirical
natural envelope, matching the codetection predicate exactly.

Codetection predicate: there exists ANY 3-consecutive-frame window
where every frame meets pellet_lk >= L AND pillar_lk >= L AND
dist <= D radii. The natural envelope must be measured the same way.

Natural pellet-on-pillar period (per user 2026-05-02): any frame in
the clean zone where GT confirms the pellet is genuinely on the pillar
AND no reach activity is happening. This includes:
  (a) all frames in GT-untouched segments
  (b) clean-zone frames BEFORE the GT interaction_frame in touched
      segments (pellet hasn't been displaced/retrieved yet)
  (c) excludes touched segments where the pellet started off-pillar
      (no pre-event on-pillar period)

For each such frame, we additionally require:
  - paw not past y-line at this frame (no reach in progress)
  - pellet bodypart confident (lk >= 0.7) -- otherwise we have no
    pellet position to compare
  - pillar bodypart confident (lk >= 0.7) -- otherwise we have no
    pillar position to compare
  - pellet near the pillar (dist <= ON_PILLAR_RADII radii) -- else
    the pellet isn't actually on the pillar

For every 3-consecutive-frame window of eligible frames, we record
the WINDOW-MAX(dist) and WINDOW-MIN(joint_lk). The codetection
predicate at thresholds (L, D) fires iff there exists a window with
MAX(dist) <= D AND MIN(joint_lk) >= L.

The triage threshold should sit OUTSIDE this natural envelope so it
never false-fires on real pellet-on-pillar data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.outcomes.v6_cascade.stage_6_pellet_displaced_to_sa import PAW_BODYPARTS
from mousereach.reach.v8.features import load_dlc_h5

QUARANTINE = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
CORPUS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\_corpus\2026-04-30_restart_inventory")

PELLET_LK_THR = 0.7
PILLAR_LK_THR = 0.7
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.5  # distance threshold to consider "pellet is on pillar"
MIN_RUN_FRAMES = 3
TRANSITION_ZONE_HALF = 5


def find_dlc(vid):
    return next(DLC_DIR.glob(f"{vid}DLC_*.h5"))


def load_gt(vid):
    return json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))


def gt_segment_bounds(gt):
    gt_b = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    return {i + 1: (gt_b[i], gt_b[i + 1] - 1) for i in range(len(gt_b) - 1)}


def runs_of_true(arr):
    runs = []
    rs = -1
    for i, v in enumerate(arr):
        if v:
            if rs < 0:
                rs = i
        else:
            if rs >= 0:
                runs.append((rs, i - 1))
                rs = -1
    if rs >= 0:
        runs.append((rs, len(arr) - 1))
    return runs


WINDOW = 3  # codetection sustained-frames threshold


def collect_window_samples(vid, sn, seg_start, seg_end, gt_outcome,
                           gt_ifr, dlc):
    """Return list of (window_max_dist_radii, window_min_joint_lk,
    window_max_dist_px, median_pillar_r) for every 3-frame consecutive
    window of eligible frames in the natural pellet-on-pillar period.

    For untouched: clean zone [seg_start, seg_end - TRANSITION_ZONE_HALF].
    For touched (displaced_sa, retrieved): clean zone capped at GT IFR
    (pre-event period only).
    For abnormal_exception or anything else: skipped (unclear pre-event
    state).
    """
    clean_end = seg_end - TRANSITION_ZONE_HALF
    if gt_outcome == "untouched":
        end = clean_end
    elif gt_outcome in ("retrieved", "displaced_sa"):
        if gt_ifr is None:
            return []
        # Only use frames strictly before GT IFR
        end = min(clean_end, int(gt_ifr) - 1)
        if end <= seg_start:
            return []
    else:
        return []
    if end <= seg_start:
        return []

    sub_raw = dlc.iloc[seg_start:end + 1]
    n = len(sub_raw)
    if n < WINDOW:
        return []
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
    pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
    pillar_r = geom["pillar_r"].to_numpy(dtype=float)
    slit_y_line = pillar_cy + pillar_r

    pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
    pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
    pillar_x = sub_raw["Pillar_x"].to_numpy(dtype=float)
    pillar_y = sub_raw["Pillar_y"].to_numpy(dtype=float)
    pillar_lk = sub_raw["Pillar_likelihood"].to_numpy(dtype=float)

    pp_dist = np.sqrt((pellet_x - pillar_x) ** 2 + (pellet_y - pillar_y) ** 2)
    pp_dist_radii = pp_dist / np.maximum(pillar_r, 1e-6)

    ppd_center = np.sqrt((pellet_x - pillar_cx) ** 2 + (pellet_y - pillar_cy) ** 2)
    ppd_center_radii = ppd_center / np.maximum(pillar_r, 1e-6)
    on_pillar = ppd_center_radii <= ON_PILLAR_RADII

    paw_past = np.zeros(n, dtype=bool)
    for bp in PAW_BODYPARTS:
        py = sub[f"{bp}_y"].to_numpy(dtype=float)
        pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
        paw_past |= (py <= slit_y_line) & (pl >= PAW_LK_THR)

    eligible = (
        (pellet_lk >= PELLET_LK_THR)
        & (pillar_lk >= PILLAR_LK_THR)
        & on_pillar
        & ~paw_past
    )
    joint_lk = np.minimum(pellet_lk, pillar_lk)

    samples = []
    # 3-frame rolling window over eligible frames
    for i in range(n - WINDOW + 1):
        if not eligible[i:i + WINDOW].all():
            continue
        win_max_dist_radii = float(pp_dist_radii[i:i + WINDOW].max())
        win_max_dist_px = float(pp_dist[i:i + WINDOW].max())
        win_min_lk = float(joint_lk[i:i + WINDOW].min())
        win_pillar_r = float(np.median(pillar_r[i:i + WINDOW]))
        samples.append((win_max_dist_radii, win_min_lk, win_max_dist_px,
                        win_pillar_r))
    return samples


def main():
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]

    samples = []   # (window_max_dist_radii, window_min_joint_lk,
                   #  window_max_dist_px, median_pillar_r, video_id, segment_num,
                   #  gt_outcome)
    n_seg_processed = defaultdict(int)

    for vid in train_pool_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        gt = load_gt(vid)
        bounds = gt_segment_bounds(gt)
        gt_outs = {s["segment_num"]: s for s in gt.get("outcomes", {}).get("segments", []) or []}
        for sn, (seg_start, seg_end) in bounds.items():
            entry = gt_outs.get(sn, {})
            outcome = entry.get("outcome")
            ifr = entry.get("interaction_frame")
            n_seg_processed[outcome or "unknown"] += 1
            for ms in collect_window_samples(vid, sn, seg_start, seg_end,
                                             outcome, ifr, dlc):
                samples.append(ms + (vid, sn, outcome))

    print(f"Segments by GT class:")
    for k, v in sorted(n_seg_processed.items()):
        print(f"  {k}: {v}")
    print()
    by_outcome = defaultdict(list)
    for d_r, lk, d_px, pr, vid, sn, oc in samples:
        by_outcome[oc].append((d_r, lk, d_px, pr, vid, sn))
    print(f"Eligible 3-frame windows by source GT class:")
    for k in sorted(by_outcome):
        print(f"  {k}: {len(by_outcome[k])} windows")
    print()
    n_total = len(samples)
    if n_total == 0:
        return
    print(f"Total eligible 3-frame windows: {n_total}")
    print()

    dist_radii = sorted(s[0] for s in samples)
    dist_px = sorted(s[2] for s in samples)
    lks = sorted(s[1] for s in samples)
    print("Window-MAX DISTANCE (radii):")
    for p in [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]:
        i = max(0, min(n_total - 1, int(p / 100 * (n_total - 1))))
        print(f"  p{p:>5}: {dist_radii[i]:.4f}")
    print()
    print("Window-MAX DISTANCE (pixels):")
    for p in [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]:
        i = max(0, min(n_total - 1, int(p / 100 * (n_total - 1))))
        print(f"  p{p:>5}: {dist_px[i]:.4f}")
    print()
    print("Window-MIN JOINT LK:")
    for p in [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]:
        i = max(0, min(n_total - 1, int(p / 100 * (n_total - 1))))
        print(f"  p{p:>5}: {lks[i]:.4f}")
    print()
    print("Joint envelope (codetection rule fires if any window has window-max-dist <= D AND window-min-lk >= L):")
    for d_thr, lk_thr in [(1.0, 0.7), (0.7, 0.7), (0.5, 0.7),
                          (1.0, 0.95), (0.5, 0.95), (0.3, 0.95),
                          (0.5, 0.85), (0.5, 0.99), (0.3, 0.99)]:
        n_in = sum(1 for s in samples if s[0] <= d_thr and s[1] >= lk_thr)
        print(f"  dist <= {d_thr:.2f} radii AND lk >= {lk_thr:.2f}: "
              f"{n_in} windows ({100*n_in/n_total:.3f}%)")
    print()
    # 10 tightest natural windows
    print("10 tightest natural windows:")
    sorted_w = sorted(samples, key=lambda x: (x[0], -x[1]))[:10]
    for d, lk, dpx, pr, vid, sn, oc in sorted_w:
        print(f"  {vid} seg {sn:2d} ({oc:14s})  dist_max={d:.3f} radii "
              f"({dpx:.2f} px, pillar_r={pr:.2f})  win_min_lk={lk:.3f}")


if __name__ == "__main__":
    main()
