"""Probe: does Pellet position teleport between two locations during
20251008_CNT0303_P2 segment 7?

If two pellets are physically present (old displaced + new ASPA load),
DLC's single-pellet output would alternate between them based on
per-frame confidence. Look for:
- Bimodal pellet position distribution (two clusters)
- Frame-to-frame position jumps > some threshold
- Sudden changes in cluster membership over time
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mousereach.reach.v8.features import load_dlc_h5

DLC = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\algo_outputs_current"
    r"\20251008_CNT0303_P2DLC_resnet50_MPSAOct27shuffle1_100000.h5"
)
GT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\gt"
    r"\20251008_CNT0303_P2_unified_ground_truth.json"
)

VIDEO = "20251008_CNT0303_P2"
TARGET_SEG = 7
LOOKBACK_SEGS = 2  # also include prior 2 segments to see the displaced_outside event


def main():
    gt = json.loads(GT.read_text())
    boundaries = [int(b["frame"]) for b in gt["segmentation"]["boundaries"]]
    # segment n is [boundaries[n-1], boundaries[n]-1]
    seg_ranges = {}
    for i in range(len(boundaries) - 1):
        seg_ranges[i + 1] = (boundaries[i], boundaries[i + 1] - 1)

    # Also get GT outcomes for context
    gt_outcomes = {s["segment_num"]: s.get("outcome") for s in gt["outcomes"]["segments"]}

    dlc = load_dlc_h5(DLC)
    plk = dlc["Pellet_likelihood"].to_numpy()
    px = dlc["Pellet_x"].to_numpy()
    py = dlc["Pellet_y"].to_numpy()

    # Report context: prior segments + target
    print(f"Video: {VIDEO}")
    print(f"Target segment: {TARGET_SEG} (GT: {gt_outcomes.get(TARGET_SEG)})")
    print()
    print("Prior segment context:")
    for sn in range(max(1, TARGET_SEG - LOOKBACK_SEGS), TARGET_SEG + 1):
        if sn not in seg_ranges:
            continue
        s, e = seg_ranges[sn]
        seg_plk = plk[s:e + 1]
        seg_px = px[s:e + 1]
        seg_py = py[s:e + 1]
        n_conf = int((seg_plk >= 0.7).sum())
        print(f"  Segment {sn} [{s}, {e}] (GT: {gt_outcomes.get(sn)}): "
              f"n_frames={e - s + 1}, n_confident_pellet={n_conf}")

    # Target segment analysis
    print()
    print(f"=== Pellet position over time in segment {TARGET_SEG} ===")
    s, e = seg_ranges[TARGET_SEG]
    seg_plk = plk[s:e + 1]
    seg_px = px[s:e + 1]
    seg_py = py[s:e + 1]
    n_seg = e - s + 1

    # Filter confident frames
    conf_mask = seg_plk >= 0.7
    conf_idx = np.where(conf_mask)[0]
    if len(conf_idx) == 0:
        print("  No confident pellet observations.")
        return

    cx = seg_px[conf_idx]
    cy = seg_py[conf_idx]
    print(f"  Total frames: {n_seg}")
    print(f"  Confident frames: {len(conf_idx)} ({100*len(conf_idx)/n_seg:.1f}%)")
    print()
    print(f"  Pellet position range:")
    print(f"    X: [{cx.min():.1f}, {cx.max():.1f}], range={cx.max()-cx.min():.1f}px")
    print(f"    Y: [{cy.min():.1f}, {cy.max():.1f}], range={cy.max()-cy.min():.1f}px")
    print(f"    Median X: {np.median(cx):.1f}, Y: {np.median(cy):.1f}")
    print(f"    Std X: {np.std(cx):.1f}, Y: {np.std(cy):.1f}")

    # Look for teleports: frame-to-frame jumps in confident-pellet position
    print()
    print("  Frame-to-frame pellet teleports (jumps > 20 px between consecutive confident frames):")
    teleport_count = 0
    big_jumps = []
    for i in range(1, len(conf_idx)):
        dt = conf_idx[i] - conf_idx[i - 1]
        if dt > 3:
            continue  # skip if we lost confidence for too many frames
        dx = cx[i] - cx[i - 1]
        dy = cy[i] - cy[i - 1]
        jump = float(np.sqrt(dx * dx + dy * dy))
        if jump > 20:
            teleport_count += 1
            big_jumps.append({
                "frame_a": int(s + conf_idx[i - 1]),
                "frame_b": int(s + conf_idx[i]),
                "pos_a": (float(cx[i - 1]), float(cy[i - 1])),
                "pos_b": (float(cx[i]), float(cy[i])),
                "jump": jump,
            })
    print(f"    Found {teleport_count} large frame-to-frame jumps.")
    for bj in big_jumps[:15]:
        print(f"    f{bj['frame_a']} -> f{bj['frame_b']}  "
              f"({bj['pos_a'][0]:.1f},{bj['pos_a'][1]:.1f}) -> "
              f"({bj['pos_b'][0]:.1f},{bj['pos_b'][1]:.1f}) "
              f"jump={bj['jump']:.1f}px")
    if teleport_count > 15:
        print(f"    ... ({teleport_count - 15} more)")
    print()

    # 2D histogram of confident positions to look for bimodality
    print("  Pellet position 2D histogram (5px bins, confident frames only):")
    bins_x = np.arange(cx.min() // 5 * 5, cx.max() // 5 * 5 + 6, 5)
    bins_y = np.arange(cy.min() // 5 * 5, cy.max() // 5 * 5 + 6, 5)
    H, xe, ye = np.histogram2d(cx, cy, bins=[bins_x, bins_y])
    # Find peaks: bins with > 5% of confident frames
    threshold = max(5, 0.05 * len(conf_idx))
    peaks = []
    for ix in range(H.shape[0]):
        for iy in range(H.shape[1]):
            if H[ix, iy] >= threshold:
                cx_mid = (xe[ix] + xe[ix + 1]) / 2.0
                cy_mid = (ye[iy] + ye[iy + 1]) / 2.0
                peaks.append((cx_mid, cy_mid, int(H[ix, iy])))
    peaks.sort(key=lambda p: -p[2])
    print(f"    Bins with >= {threshold:.0f} confident frames (sorted by count):")
    for cx_mid, cy_mid, n in peaks[:10]:
        print(f"      ({cx_mid:6.1f}, {cy_mid:6.1f}): {n} frames")

    # Cluster the positions: if there are 2+ clusters > 20px apart, that's the
    # two-pellet signature.
    if len(peaks) >= 2:
        # Look for clusters spaced > 20 px apart
        cluster_pairs = []
        for i, (x1, y1, n1) in enumerate(peaks):
            for x2, y2, n2 in peaks[i + 1:]:
                d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if d > 20:
                    cluster_pairs.append((x1, y1, x2, y2, d, n1, n2))
        print()
        print(f"    Cluster pairs > 20 px apart:")
        for x1, y1, x2, y2, d, n1, n2 in cluster_pairs[:5]:
            print(f"      ({x1:.1f},{y1:.1f}) [{n1}f] <-> "
                  f"({x2:.1f},{y2:.1f}) [{n2}f]  dist={d:.1f}px")


if __name__ == "__main__":
    main()
