"""Diagnostic: what distinguishes merged GT pairs from cleanly-split pairs
on CNT0413_P4?

CNT0413_P4 has 19 MERGED components and ~180 cleanly-split adjacent GT
pairs. The question: among consecutive GT pairs, what features predict
which ones the algo merges vs which ones it splits correctly?

Per-pair features computed (using model 3.1 parquet for paw_lk and
positions):
  - Inter-GT gap (frames between gt1.end and gt2.start)
  - GT1 span, GT2 span
  - During inter-GT window: paw_mean_lk, paw_min_lk
  - Hand position at gt1.end and at gt2.start
  - Direction reversal (hand x-velocity sign-flip in inter-GT window)
  - Distance to BoxL at boundary
  - Distance to BoxR at boundary

Loads from v8.0.2 manifest (live GT) and the model 3.1 parquet.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


VIDEO_ID = "20251022_CNT0413_P4"
MANIFEST = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests"
    r"\v8.0.2\calibration_loocv\20251022_CNT0413_P4.json"
)
CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.2_dev_cnt0413_merge_pair_comparison"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)


PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]


def smoothed_velocity(arr, dt=2):
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n < 2 * dt + 1:
        return out
    out[dt:n - dt] = (arr[2 * dt:n] - arr[0:n - 2 * dt]) / (2.0 * dt)
    return out


def main():
    print("=" * 70)
    print(f"CNT0413_P4 merge-pair-vs-clean-split-pair comparison")
    print("=" * 70)

    # ===== Load manifest =====
    print("Loading manifest...", flush=True)
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    events = m["events"]

    # Build GT list with TP/FN status + component_id
    gt_records = []
    for e in events:
        if e["gt"] is None:
            continue
        gt_records.append({
            "start": e["gt"]["start"],
            "end": e["gt"]["end"],
            "span": e["gt"]["end"] - e["gt"]["start"] + 1,
            "kind": e["kind"],
            "topology": e["topology"],
            "component_id": e.get("component_id"),
            "algo_start": e["detector"]["start"] if e.get("detector") else None,
            "algo_end": e["detector"]["end"] if e.get("detector") else None,
        })
    gt_records.sort(key=lambda r: r["start"])
    print(f"  {len(gt_records)} GT reaches loaded")

    # ===== Load parquet for paw_lk + positions =====
    print("Loading parquet for paw_lk + positions...", flush=True)
    needed_cols = (
        ["video_id", "frame"]
        + PARQUET_LK_COLS
        + ["RightHand_x", "RightHand_y", "RHLeft_x", "RHLeft_y",
            "RHOut_x", "RHOut_y", "RHRight_x", "RHRight_y",
            "BOXL_x", "BOXL_y", "BOXR_x", "BOXR_y"]
    )
    df = pd.read_parquet(CAL_PARQUET, columns=needed_cols)
    df = df[df["video_id"] == VIDEO_ID].sort_values("frame").reset_index(drop=True)
    n_frames = len(df)
    print(f"  {n_frames} frames")

    # paw_mean_lk per frame
    paw_lk_matrix = df[PARQUET_LK_COLS].to_numpy(dtype=np.float32)
    paw_mean_lk = paw_lk_matrix.mean(axis=1)
    paw_min_lk = paw_lk_matrix.min(axis=1)

    # Hand centroid (mean of 4 hand keypoints, smoothed)
    def smooth(x, w=5):
        return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
    hand_x_smooth = smooth(np.mean(
        [df[f"{kp}_x"].to_numpy() for kp in ("RightHand", "RHLeft", "RHOut", "RHRight")],
        axis=0))
    hand_y_smooth = smooth(np.mean(
        [df[f"{kp}_y"].to_numpy() for kp in ("RightHand", "RHLeft", "RHOut", "RHRight")],
        axis=0))
    boxl_x = smooth(df["BOXL_x"].to_numpy())
    boxl_y = smooth(df["BOXL_y"].to_numpy())
    boxr_x = smooth(df["BOXR_x"].to_numpy())
    boxr_y = smooth(df["BOXR_y"].to_numpy())
    apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)

    dist_boxl = np.sqrt((hand_x_smooth - boxl_x) ** 2 + (hand_y_smooth - boxl_y) ** 2)
    norm_pos = dist_boxl / np.maximum(apparatus, 1e-3)  # 0 = at BoxL, 1 = at BoxR

    hand_vx = smoothed_velocity(hand_x_smooth, dt=2)

    # ===== Classify GT pairs =====
    # For each consecutive GT pair (gt_i, gt_{i+1}), determine the pair label:
    #   "merged"    -- both GTs are in the same MERGED component
    #   "split"     -- both are TPs (cleanly split by algo)
    #   "other"     -- mixed cases (one TP one FN, both FN but no MERGED match, etc.)
    pairs = []
    for i in range(len(gt_records) - 1):
        g1 = gt_records[i]
        g2 = gt_records[i + 1]
        gap = g2["start"] - g1["end"] - 1

        if (g1["topology"] == "MERGED" and g2["topology"] == "MERGED"
                and g1["component_id"] == g2["component_id"]):
            label = "merged"
        elif g1["kind"] == "TP" and g2["kind"] == "TP":
            label = "split"
        else:
            label = "other"
        pairs.append({
            "label": label,
            "g1": g1, "g2": g2, "gap": gap,
            "boundary_frame": g1["end"],  # last frame of gt1 (gt2 starts at gt1.end+1 if gap=0)
        })

    print(f"\nPair classification:")
    n_per = defaultdict(int)
    for p in pairs:
        n_per[p["label"]] += 1
    for k, v in n_per.items():
        print(f"  {k}: {v}")

    # ===== Compute per-pair features =====
    feature_rows = []
    for p in pairs:
        g1, g2 = p["g1"], p["g2"]
        gap = p["gap"]
        # Inter-gap frames (if gap >= 1)
        # Boundary frame = gt1.end (last frame of GT1)
        # For gap=0, frames immediately around boundary are key

        # Window around the boundary: -3 to +3 inclusive
        b = g1["end"]
        win_lo = max(0, b - 3)
        win_hi = min(n_frames, g2["start"] + 4)

        # paw_mean_lk at gt1.end and gt2.start
        lk_at_g1_end = float(paw_mean_lk[g1["end"]]) if g1["end"] < n_frames else np.nan
        lk_at_g2_start = float(paw_mean_lk[g2["start"]]) if g2["start"] < n_frames else np.nan

        # paw_mean_lk min over the boundary window
        lk_window_min = float(paw_mean_lk[win_lo:win_hi].min()) if win_hi > win_lo else np.nan
        lk_window_mean = float(paw_mean_lk[win_lo:win_hi].mean()) if win_hi > win_lo else np.nan

        # Inter-GT gap stats (only meaningful if gap >= 1)
        if gap >= 1:
            inter_lo = g1["end"] + 1
            inter_hi = g2["start"]  # inclusive
            inter_lk_min = float(paw_mean_lk[inter_lo:inter_hi + 1].min())
            inter_lk_mean = float(paw_mean_lk[inter_lo:inter_hi + 1].mean())
        else:
            inter_lk_min = np.nan
            inter_lk_mean = np.nan

        # Hand position
        np_at_g1_end = float(norm_pos[g1["end"]]) if g1["end"] < n_frames else np.nan
        np_at_g2_start = float(norm_pos[g2["start"]]) if g2["start"] < n_frames else np.nan
        np_window_min = float(norm_pos[win_lo:win_hi].min()) if win_hi > win_lo else np.nan

        # Direction reversal: vx sign change in +/-6f window around boundary
        sf_lo = max(0, b - 6)
        sf_hi = min(n_frames, b + 7)
        win_vx = hand_vx[sf_lo:sf_hi]
        has_pos = np.any(win_vx > 0.05)
        has_neg = np.any(win_vx < -0.05)
        vx_signflip = bool(has_pos and has_neg)

        feature_rows.append({
            "label": p["label"],
            "gap": gap,
            "g1_start": g1["start"], "g1_end": g1["end"], "g1_span": g1["span"],
            "g2_start": g2["start"], "g2_end": g2["end"], "g2_span": g2["span"],
            "boundary_frame": b,
            # paw_lk features
            "lk_at_g1_end": lk_at_g1_end,
            "lk_at_g2_start": lk_at_g2_start,
            "lk_window_min": lk_window_min,
            "lk_window_mean": lk_window_mean,
            "inter_lk_min": inter_lk_min,
            "inter_lk_mean": inter_lk_mean,
            # Position features
            "np_at_g1_end": np_at_g1_end,
            "np_at_g2_start": np_at_g2_start,
            "np_window_min": np_window_min,
            # Velocity feature
            "vx_signflip_pm6f": vx_signflip,
        })

    feat_df = pd.DataFrame(feature_rows)
    out_csv = OUT_DIR / "metrics" / "pair_features.csv"
    feat_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-pair features: {out_csv}")

    # ===== Aggregate stats per label =====
    print("\n=== PER-LABEL FEATURE MEDIANS ===")
    print(f"{'feature':<30}  {'merged':>10}  {'split':>10}  {'other':>10}")
    print("-" * 70)
    for col in ["gap", "g1_span", "g2_span",
                "lk_at_g1_end", "lk_at_g2_start", "lk_window_min", "lk_window_mean",
                "inter_lk_min", "inter_lk_mean",
                "np_at_g1_end", "np_at_g2_start", "np_window_min"]:
        row = f"{col:<30}"
        for label in ("merged", "split", "other"):
            sub = feat_df[feat_df["label"] == label][col].dropna()
            if len(sub):
                row += f"  {sub.median():>10.3f}"
            else:
                row += f"  {'n/a':>10}"
        print(row)

    print(f"\n{'vx_signflip_pm6f rate':<30}", end="")
    for label in ("merged", "split", "other"):
        sub = feat_df[feat_df["label"] == label]["vx_signflip_pm6f"]
        if len(sub):
            print(f"  {sub.mean():>10.2f}", end="")
        else:
            print(f"  {'n/a':>10}", end="")
    print()

    # ===== Gap distribution per label =====
    print("\n=== GAP DISTRIBUTION (consecutive GT pairs) ===")
    for label in ("merged", "split", "other"):
        sub = feat_df[feat_df["label"] == label]
        if not len(sub): continue
        print(f"\n  {label} (n={len(sub)}):")
        gap_counts = sub["gap"].value_counts().sort_index()
        for gap, n in gap_counts.items():
            bar = "#" * min(n, 50)
            print(f"    gap={int(gap):>4}: {n:>4}  {bar}")

    # ===== Figure =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: gap distribution
    ax = axes[0, 0]
    for label, color in (("merged", "C3"), ("split", "C0"), ("other", "C2")):
        sub = feat_df[feat_df["label"] == label]
        if not len(sub): continue
        bins = np.arange(-1, sub["gap"].max() + 2)
        ax.hist(sub["gap"].values, bins=bins, alpha=0.5, color=color,
                label=f"{label} (n={len(sub)})")
    ax.set_xlabel("inter-GT gap (frames)")
    ax.set_ylabel("count")
    ax.set_title("Inter-GT gap distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: paw_mean_lk at gt2.start
    ax = axes[0, 1]
    for label, color in (("merged", "C3"), ("split", "C0")):
        sub = feat_df[feat_df["label"] == label]["lk_at_g2_start"].dropna()
        if not len(sub): continue
        ax.hist(sub.values, bins=np.linspace(0, 1, 21), alpha=0.5, color=color,
                label=f"{label} (n={len(sub)}, med={sub.median():.2f})", density=True)
    ax.set_xlabel("paw_mean_lk at gt2.start")
    ax.set_ylabel("density")
    ax.set_title("Paw confidence at gt2.start")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: lk_window_min (min paw_lk in +/-3f around boundary)
    ax = axes[0, 2]
    for label, color in (("merged", "C3"), ("split", "C0")):
        sub = feat_df[feat_df["label"] == label]["lk_window_min"].dropna()
        if not len(sub): continue
        ax.hist(sub.values, bins=np.linspace(0, 1, 21), alpha=0.5, color=color,
                label=f"{label} (n={len(sub)}, med={sub.median():.2f})", density=True)
    ax.set_xlabel("min paw_mean_lk in +/-3f around boundary")
    ax.set_ylabel("density")
    ax.set_title("Min paw confidence around boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: norm_pos at gt2.start (where is the hand?)
    ax = axes[1, 0]
    for label, color in (("merged", "C3"), ("split", "C0")):
        sub = feat_df[feat_df["label"] == label]["np_at_g2_start"].dropna()
        if not len(sub): continue
        ax.hist(sub.values, bins=np.linspace(0, 1.2, 25), alpha=0.5, color=color,
                label=f"{label} (n={len(sub)}, med={sub.median():.2f})", density=True)
    ax.axvline(0.25, color="0.5", ls=":", label="BoxL zone")
    ax.set_xlabel("norm_pos at gt2.start (0=BoxL, 1=BoxR)")
    ax.set_ylabel("density")
    ax.set_title("Hand position at gt2.start")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: g1_span comparison
    ax = axes[1, 1]
    for label, color in (("merged", "C3"), ("split", "C0")):
        sub = feat_df[feat_df["label"] == label]
        if not len(sub): continue
        ax.scatter(sub["g1_span"].values, sub["g2_span"].values,
                   alpha=0.5, color=color, label=f"{label} (n={len(sub)})")
    ax.set_xlabel("GT1 span (frames)")
    ax.set_ylabel("GT2 span (frames)")
    ax.set_title("Reach spans (GT1 vs GT2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6: lk vs gap (does the lk depend on gap?)
    ax = axes[1, 2]
    for label, color in (("merged", "C3"), ("split", "C0")):
        sub = feat_df[feat_df["label"] == label]
        if not len(sub): continue
        ax.scatter(sub["gap"].values, sub["lk_window_min"].values,
                   alpha=0.5, color=color, label=f"{label} (n={len(sub)})")
    ax.set_xlabel("inter-GT gap")
    ax.set_ylabel("min paw_lk in +/-3f around boundary")
    ax.set_title("paw_lk vs gap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{VIDEO_ID}: merged vs cleanly-split GT pairs", fontsize=12)
    fig.tight_layout()
    out_fig = OUT_DIR / "figures" / "merge_pair_comparison.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {out_fig}")


if __name__ == "__main__":
    main()
