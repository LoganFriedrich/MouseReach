"""Diagnostic: does the hand return toward BOXL at MERGED boundaries?
Extended pass: 6 videos, both Mech A (gap=0, paw-visibility hypothesis) and
Mech B (gap>=3, apparatus-suspected) merge cases.

Hypothesis under test (Logan 2026-05-21): "new reaches start by BOXL and end
at BOXR, so if we were to help the model see this point it could maybe split
some of these merges." v8 already has dist__RightHand__BOXL and BOXL_x/y in
the feature set. The question is whether the signal is actually there.

Round 1 (CNT0413_P4 only): hand does NOT return to BOXL; norm_pos stays at
0.68-0.86 across all 5 boundaries. But hand x-velocity DOES sign-flip
cleanly at the boundary. Suggested direction-reversal features instead.

Round 2 (this script): verify on 3 more paw-visibility mice (Mech A) and
3 apparatus-suspected mice with gap>=3 merges (Mech B). Same diagnostics
per event: hand-to-BOXL distance, normalized position, hand x-velocity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mousereach.reach.core.geometry import load_dlc  # noqa: E402

# DLC root directories
CAL_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
    r"\DLC_2026_03_27\Processing"
)
HOLDOUT_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\dlc"
)
DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000.h5"


def dlc_path(video_id: str, corpus: str) -> Path:
    base = CAL_DLC_DIR if corpus == "cal" else HOLDOUT_DLC_DIR
    return base / f"{video_id}{DLC_SUFFIX}"


# Video configurations: each video has its DLC path, mech label, and event list.
# Events are tuples: (comp_id, gt1_start, gt1_end, gt2_start, gt2_end, algo_start, algo_end)
# Mech A = gap=0 (paw-visibility candidate); Mech B = gap>=3 (apparatus-suspected)
VIDEOS = [
    # Mech A: paw-visibility mice (3 of them; CNT0413_P4 already run separately)
    {
        "video_id": "20250630_CNT0104_P3",
        "corpus": "cal",
        "mech": "A",
        "events": [
            (11,  9750,  9763,  9764,  9778,  9750,  9797),
            (33,  17656, 17673, 17674, 17687, 17656, 17687),
            (39,  25188, 25203, 25204, 25216, 25188, 25217),
            (46,  26267, 26284, 26285, 26303, 26267, 26303),
            (49,  27944, 27961, 27962, 27979, 27944, 27979),
            (51,  28010, 28027, 28028, 28043, 28010, 28043),
            (53,  28277, 28292, 28293, 28303, 28277, 28303),
        ],
    },
    {
        "video_id": "20250710_CNT0215_P4",
        "corpus": "cal",
        "mech": "A",
        "events": [
            (63,   7834,  7844,  7845,  7854,  7834,  7856),
            (104,  18773, 18793, 18794, 18812, 18773, 18812),
            (111,  20963, 20978, 20979, 20997, 20963, 20997),
            (117,  23255, 23281, 23282, 23303, 23255, 23322),
            (126,  24646, 24660, 24661, 24679, 24646, 24679),
        ],
    },
    {
        "video_id": "20250718_CNT0214_P1",
        "corpus": "holdout",
        "mech": "A",
        "events": [
            (9,    666,   679,   680,   711,   666,   711),
            (104,  10204, 10217, 10218, 10230, 10204, 10230),
            (137,  12446, 12472, 12473, 12491, 12446, 12491),
            (153,  14543, 14564, 14565, 14579, 14543, 14579),
        ],
    },
    # Mech B: apparatus-suspected merges (gap>=3 between paired GTs)
    {
        "video_id": "20250812_CNT0301_P3",
        "corpus": "cal",
        "mech": "B",
        "events": [
            (91,   12232, 12243, 12247, 12268, 12230, 12268),  # gap=3
            (110,  14360, 14369, 14374, 14389, 14364, 14389),  # gap=4
            (73,   9468,  9481,  9487,  9499,  9468,  9499),   # gap=5
            (174,  26421, 26434, 26441, 26462, 26416, 26462),  # gap=6
            (56,   7661,  7670,  7679,  7694,  7660,  7694),   # gap=8
            (167,  25899, 25924, 25935, 25954, 25899, 25955),  # gap=10
            (171,  26245, 26264, 26281, 26302, 26243, 26312),  # gap=16
        ],
    },
    {
        "video_id": "20251010_CNT0308_P2",
        "corpus": "cal",
        "mech": "B",
        "events": [
            (79,   13945, 13957, 13965, 13975, 13944, 13996),  # gap=7
            (78,   13277, 13291, 13301, 13316, 13260, 13318),  # gap=9
            (23,   1737,  1750,  1761,  1772,  1734,  1772),   # gap=10
            (81,   14657, 14668, 14681, 14695, 14650, 14753),  # gap=12 (4gt)
            (51,   6586,  6599,  6614,  6630,  6585,  6652),   # gap=14
        ],
    },
    {
        "video_id": "20250806_CNT0316_P3",
        "corpus": "holdout",
        "mech": "B",
        "events": [
            (28,   1851,  1874,  1878,  1895,  1850,  1895),   # gap=3
            (148,  33458, 33469, 33478, 33493, 33442, 33504),  # gap=8
            (101,  14970, 14995, 15006, 15021, 14969, 15021),  # gap=10
            (58,   5856,  5862,  5874,  5888,  5844,  5888),   # gap=11
            (149,  33532, 33546, 33560, 33575, 33530, 33591),  # gap=13
        ],
    },
]

HALF_W = 25

OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\Improvement_Snapshots\reach_detection"
    r"\v8.0.1_dev_hand_position_at_merged_boundaries"
)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)


def smoothed_velocity(arr, dt=2):
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n < 2 * dt + 1:
        return out
    out[dt:n - dt] = (arr[2 * dt:n] - arr[0:n - 2 * dt]) / (2.0 * dt)
    return out


def classify_pos(norm: float) -> str:
    if norm < 0.25: return "AT BOXL"
    if norm < 0.4:  return "near BOXL"
    if norm < 0.7:  return "MID"
    return "near BOXR"


def signflip_within(vx: np.ndarray, frame: int, half_w: int = 3) -> bool:
    lo = max(0, frame - half_w)
    hi = min(len(vx), frame + half_w + 1)
    window = vx[lo:hi]
    if len(window) < 2:
        return False
    has_pos = np.any(window > 0.05)
    has_neg = np.any(window < -0.05)
    return bool(has_pos and has_neg)


def process_video(cfg: dict) -> list:
    vid = cfg["video_id"]
    mech = cfg["mech"]
    events = cfg["events"]
    path = dlc_path(vid, cfg["corpus"])
    print(f"\n=== {vid} (Mech {mech}, {len(events)} events) ===")
    if not path.exists():
        print(f"  [FAIL] missing DLC: {path}")
        return []
    dlc = load_dlc(path)
    print(f"  loaded {len(dlc)} frames")

    hand_x = dlc["RightHand_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    hand_y = dlc["RightHand_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    hand_lk = dlc["RightHand_likelihood"].to_numpy()
    boxl_x = dlc["BOXL_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    boxl_y = dlc["BOXL_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    boxr_x = dlc["BOXR_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    boxr_y = dlc["BOXR_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()

    dist_boxl = np.sqrt((hand_x - boxl_x) ** 2 + (hand_y - boxl_y) ** 2)
    dist_boxr = np.sqrt((hand_x - boxr_x) ** 2 + (hand_y - boxr_y) ** 2)
    apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)
    norm_pos = dist_boxl / np.maximum(apparatus, 1e-3)
    vx = smoothed_velocity(hand_x, dt=2)

    med_apparatus = float(np.median(apparatus))
    print(f"  apparatus median width: {med_apparatus:.1f} px")

    rows = []
    fig, axes = plt.subplots(len(events), 3, figsize=(18, 3.0 * len(events)),
                             squeeze=False)
    for i, (comp, g1s, g1e, g2s, g2e, algos, algoe) in enumerate(events):
        gap = g2s - g1e - 1
        win_lo = max(0, g1e - HALF_W)
        win_hi = min(len(dlc), g2s + HALF_W + 1)
        frames = np.arange(win_lo, win_hi)

        # Numeric features at the boundary
        np_at_end1 = float(norm_pos[g1e])
        np_at_start2 = float(norm_pos[g2s])
        # Min distance to BOXL in a 7-frame window around the boundary
        bw_lo = max(0, g1e - 3)
        bw_hi = min(len(dlc), g2s + 4)
        min_dist_boxl = float(np.min(dist_boxl[bw_lo:bw_hi]))
        min_norm_pos = float(np.min(norm_pos[bw_lo:bw_hi]))
        flip = signflip_within(vx, (g1e + g2s) // 2, half_w=3)
        flip_wide = signflip_within(vx, (g1e + g2s) // 2, half_w=6)
        lk_min = float(np.min(hand_lk[bw_lo:bw_hi]))

        rows.append({
            "video": vid,
            "mech": mech,
            "comp": comp,
            "gap": gap,
            "n_gt_first_pair": 2,
            "norm_pos_at_GT1end": np_at_end1,
            "norm_pos_at_GT2start": np_at_start2,
            "min_norm_pos_in_window": min_norm_pos,
            "min_dist_to_BOXL_in_window": min_dist_boxl,
            "vx_signflip_pm3f": flip,
            "vx_signflip_pm6f": flip_wide,
            "RightHand_lk_min": lk_min,
            "apparatus_width_at_boundary": float(apparatus[g1e]),
        })

        ax = axes[i, 0]
        ax.plot(frames, dist_boxl[win_lo:win_hi], color="C0", lw=1.4, label="hand -> BOXL")
        ax.plot(frames, dist_boxr[win_lo:win_hi], color="C3", lw=1.4, label="hand -> BOXR")
        ax.axhline(apparatus[g1e], color="0.6", lw=0.7, ls=":",
                   label=f"width ({apparatus[g1e]:.0f}px)")
        ax.axvspan(g1s, g1e, alpha=0.15, color="C2", label="GT1")
        ax.axvspan(g2s, g2e, alpha=0.15, color="C4", label="GT2")
        ax.axvline(algos, color="C1", ls="--", lw=0.6)
        ax.axvline(algoe, color="C1", ls="--", lw=0.6, label="algo")
        ax.set_title(f"comp={comp} gap={gap}")
        ax.set_xlabel("frame")
        ax.set_ylabel("dist (px)")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.plot(frames, norm_pos[win_lo:win_hi], color="C0", lw=1.4)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axhline(1, color="0.7", lw=0.5)
        ax.axhline(0.25, color="0.8", ls=":", lw=0.5, label="0.25 (BOXL zone)")
        ax.axvspan(g1s, g1e, alpha=0.15, color="C2")
        ax.axvspan(g2s, g2e, alpha=0.15, color="C4")
        ax.axvline(algos, color="C1", ls="--", lw=0.6)
        ax.axvline(algoe, color="C1", ls="--", lw=0.6)
        ax.set_title(f"norm_pos (0=BOXL, 1=BOXR); min_in_win={min_norm_pos:.2f}")
        ax.set_xlabel("frame")
        ax.set_ylim(-0.2, 1.4)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 2]
        ax.plot(frames, vx[win_lo:win_hi], color="C5", lw=1.4)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.axvspan(g1s, g1e, alpha=0.15, color="C2")
        ax.axvspan(g2s, g2e, alpha=0.15, color="C4")
        ax.axvline(algos, color="C1", ls="--", lw=0.6)
        ax.axvline(algoe, color="C1", ls="--", lw=0.6)
        flip_lbl = ("flip" if flip else "no flip") + " (+-3f)"
        ax.set_title(f"hand_vx; {flip_lbl}")
        ax.set_xlabel("frame")
        ax.set_ylabel("dx/dt")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{vid} (Mech {mech}) -- hand position + velocity at MERGED boundaries",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_fig = OUT_DIR / "figures" / f"{vid}.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] figure: {out_fig.name}")

    return rows


def main():
    all_rows = []
    for cfg in VIDEOS:
        all_rows.extend(process_video(cfg))

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "metrics" / "all_videos_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] aggregated summary: {out_csv}")

    print("\n=== AGGREGATE FINDINGS ===")
    print(f"Total events analyzed: {len(df)}")
    print()
    print("Per-mechanism summary:")
    for mech in ["A", "B"]:
        sub = df[df["mech"] == mech]
        if len(sub) == 0: continue
        print(f"\nMechanism {mech} ({len(sub)} events):")
        print(f"  Median norm_pos at GT1.end:        {sub['norm_pos_at_GT1end'].median():.2f}")
        print(f"  Median norm_pos at GT2.start:      {sub['norm_pos_at_GT2start'].median():.2f}")
        print(f"  Median min_norm_pos in window:     {sub['min_norm_pos_in_window'].median():.2f}")
        print(f"  Events with min_norm_pos < 0.25 (hand briefly at BOXL): "
              f"{int((sub['min_norm_pos_in_window'] < 0.25).sum())} / {len(sub)}")
        print(f"  Events with vx sign-flip in +-3f:  {int(sub['vx_signflip_pm3f'].sum())} / {len(sub)}")
        print(f"  Events with vx sign-flip in +-6f:  {int(sub['vx_signflip_pm6f'].sum())} / {len(sub)}")

    print("\nPer-video breakdown:")
    by_vid = df.groupby("video").agg(
        n=("comp", "count"),
        mech=("mech", "first"),
        med_norm_pos=("norm_pos_at_GT1end", "median"),
        n_flip3=("vx_signflip_pm3f", "sum"),
        n_flip6=("vx_signflip_pm6f", "sum"),
        n_at_BOXL=("min_norm_pos_in_window", lambda s: int((s < 0.25).sum())),
    ).reset_index()
    print(by_vid.to_string(index=False))


if __name__ == "__main__":
    main()
