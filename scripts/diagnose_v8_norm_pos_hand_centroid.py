"""Diagnostic 6: TP vs MERGED-GT2+ norm_pos comparison using the HAND CENTROID.

Logan (2026-05-21): "there should be some sort of average number between
all the points on the hand because RHLeft is doing the same sort of bias
trick but the other way now."

The 4 right-hand keypoints are biased per their anatomical position:
  RightHand   ~0.58 (annotated toward BOXR / front of paw)
  RHOut       ~0.54
  RHRight     ~0.70 (annotated toward BOXR side)
  RHLeft      ~0.25 (annotated toward BOXL / back of paw)

The unweighted centroid avoids all of these per-keypoint biases. Re-run
the offset profile and window-min/window-max comparisons.

Also extends the offset window to +10 to catch peak extension during a
typical reach.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mousereach.reach.core.geometry import load_dlc  # noqa: E402

CAL_DLC_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\Processing")
HOLDOUT_DLC_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\dlc")
DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000.h5"
MANIFEST_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests\v8.0.1")

VIDEOS = [
    ("20250630_CNT0104_P3", "cal", "calibration_loocv"),
    ("20250710_CNT0215_P4", "cal", "calibration_loocv"),
    ("20250718_CNT0214_P1", "holdout", "holdout_2026_05_11"),
    ("20250812_CNT0301_P3", "cal", "calibration_loocv"),
    ("20251010_CNT0308_P2", "cal", "calibration_loocv"),
    ("20250806_CNT0316_P3", "holdout", "holdout_2026_05_11"),
    ("20251022_CNT0413_P4", "cal", "calibration_loocv"),
]

HAND_KPS = ["RightHand", "RHLeft", "RHOut", "RHRight"]
OFFSETS = list(range(-5, 11))  # -5 to +10 inclusive
WIN_PRE_LO, WIN_PRE_HI = -2, 5
WIN_POST_LO, WIN_POST_HI = 0, 8

OUT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\reach_detection\v8.0.1_dev_norm_pos_hand_centroid")
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)


def dlc_path(vid, corpus):
    base = CAL_DLC_DIR if corpus == "cal" else HOLDOUT_DLC_DIR
    return base / f"{vid}{DLC_SUFFIX}"


def manifest_path(vid, manifest_corpus):
    return MANIFEST_DIR / manifest_corpus / f"{vid}.json"


def main():
    rows = []
    for vid, dlc_corpus, manifest_corpus in VIDEOS:
        dlc_p = dlc_path(vid, dlc_corpus)
        man_p = manifest_path(vid, manifest_corpus)
        if not dlc_p.exists() or not man_p.exists():
            print(f"[skip] {vid}")
            continue
        dlc = load_dlc(dlc_p)

        # Smoothed apparatus reference channels
        boxl_x = dlc["BOXL_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxl_y = dlc["BOXL_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxr_x = dlc["BOXR_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxr_y = dlc["BOXR_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)

        # Hand centroid: unweighted mean of the 4 right-hand keypoints (smoothed)
        cx_list, cy_list = [], []
        for kp in HAND_KPS:
            cx_list.append(dlc[f"{kp}_x"].rolling(5, center=True, min_periods=1).mean().to_numpy())
            cy_list.append(dlc[f"{kp}_y"].rolling(5, center=True, min_periods=1).mean().to_numpy())
        cx = np.mean(cx_list, axis=0)
        cy = np.mean(cy_list, axis=0)

        d_boxl = np.sqrt((cx - boxl_x) ** 2 + (cy - boxl_y) ** 2)
        norm_pos = d_boxl / np.maximum(apparatus, 1e-3)
        n = len(dlc)

        m = json.loads(man_p.read_text())
        events = m["events"]

        kinds = []
        for e in events:
            if e.get("topology") == "TP" and e.get("gt") and e["gt"].get("start") is not None:
                kinds.append(("TP_start", int(e["gt"]["start"])))
        by_comp = defaultdict(list)
        for e in events:
            if e.get("topology") == "MERGED" and e.get("kind") == "FN":
                by_comp[e["component_id"]].append(e)
        for cid, fns in by_comp.items():
            fns = sorted(fns, key=lambda r: r["gt"]["start"] if r.get("gt") and r["gt"].get("start") is not None else 0)
            for idx, r in enumerate(fns):
                if not (r.get("gt") and r["gt"].get("start") is not None): continue
                gs = int(r["gt"]["start"])
                lbl = "MERGED_GT1_start" if idx == 0 else "MERGED_GT2plus_start"
                kinds.append((lbl, gs))

        for kind, gs in kinds:
            if not (0 <= gs < n): continue
            rec = {"video": vid, "kind": kind, "gt_start": gs}
            pre_vals, post_vals, all_vals = [], [], []
            for off in OFFSETS:
                f = gs + off
                if 0 <= f < n:
                    v = float(norm_pos[f])
                    rec[f"np_off{off:+d}"] = v
                    all_vals.append((off, v))
                    if WIN_PRE_LO <= off <= WIN_PRE_HI:
                        pre_vals.append(v)
                    if WIN_POST_LO <= off <= WIN_POST_HI:
                        post_vals.append(v)
                else:
                    rec[f"np_off{off:+d}"] = np.nan
            rec["win_min_pre"] = float(np.min(pre_vals)) if pre_vals else np.nan
            rec["win_max_post"] = float(np.max(post_vals)) if post_vals else np.nan
            rec["range_full"] = (rec["win_max_post"] - rec["win_min_pre"]
                                 if not np.isnan(rec["win_min_pre"]) and not np.isnan(rec["win_max_post"])
                                 else np.nan)
            rows.append(rec)

        n_tp = sum(1 for r in rows if r["video"] == vid and r["kind"] == "TP_start")
        n_m1 = sum(1 for r in rows if r["video"] == vid and r["kind"] == "MERGED_GT1_start")
        n_m2 = sum(1 for r in rows if r["video"] == vid and r["kind"] == "MERGED_GT2plus_start")
        print(f"[{vid}] TP={n_tp} M1={n_m1} M2+={n_m2}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "metrics" / "norm_pos_hand_centroid.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}")
    print(f"Total records: {len(df)}\n")

    # Aggregate medians per offset
    print("=== Pooled median CENTROID norm_pos at each offset relative to GT.start ===")
    offset_cols = [f"np_off{off:+d}" for off in OFFSETS]
    summary = df.groupby("kind")[offset_cols].median().round(3)
    print(summary.to_string())

    print("\n=== Pooled summary of derived features (CENTROID) ===")
    feats = ["win_min_pre", "win_max_post", "range_full"]
    for kind in ["TP_start", "MERGED_GT1_start", "MERGED_GT2plus_start"]:
        sub = df[df["kind"] == kind]
        if len(sub) == 0: continue
        print(f"\n  {kind} (n={len(sub)}):")
        for f in feats:
            vals = sub[f].dropna()
            print(f"    {f:<14}: median={vals.median():.3f}, 25%={vals.quantile(0.25):.3f}, "
                  f"75%={vals.quantile(0.75):.3f}")

    # Per-video separability for the 3 candidate features
    print("\n=== PER-VIDEO: win_max_post (post-start overshoot) ===")
    print(f"{'video':<22} | {'TP med':>7} {'M2+ med':>8} | {'TP>0.85':>8} {'M2+>0.85':>9} | {'TP>1.00':>8} {'M2+>1.00':>9}")
    print("-" * 95)
    for vid in sorted(df['video'].unique()):
        tp = df[(df['video']==vid) & (df['kind']=='TP_start')].dropna(subset=['win_max_post'])
        m2 = df[(df['video']==vid) & (df['kind']=='MERGED_GT2plus_start')].dropna(subset=['win_max_post'])
        if not len(tp): continue
        tp_med = tp['win_max_post'].median()
        m2_med = m2['win_max_post'].median() if len(m2) else float('nan')
        def frac_gt(s, t): return 100.0 * (s > t).sum() / len(s) if len(s) else float('nan')
        line = f"{vid:<22} | {tp_med:>7.3f} {m2_med:>8.3f} | {frac_gt(tp['win_max_post'], 0.85):>7.1f}% {frac_gt(m2['win_max_post'], 0.85):>8.1f}% | {frac_gt(tp['win_max_post'], 1.0):>7.1f}% {frac_gt(m2['win_max_post'], 1.0):>8.1f}%"
        print(line)

    print("\n=== PER-VIDEO: range_full (post-max minus pre-min) ===")
    print(f"{'video':<22} | {'TP med':>7} {'M2+ med':>8} | {'TP>0.50':>8} {'M2+>0.50':>9} | {'TP>0.70':>8} {'M2+>0.70':>9}")
    print("-" * 95)
    for vid in sorted(df['video'].unique()):
        tp = df[(df['video']==vid) & (df['kind']=='TP_start')].dropna(subset=['range_full'])
        m2 = df[(df['video']==vid) & (df['kind']=='MERGED_GT2plus_start')].dropna(subset=['range_full'])
        if not len(tp): continue
        tp_med = tp['range_full'].median()
        m2_med = m2['range_full'].median() if len(m2) else float('nan')
        def frac_gt(s, t): return 100.0 * (s > t).sum() / len(s) if len(s) else float('nan')
        line = f"{vid:<22} | {tp_med:>7.3f} {m2_med:>8.3f} | {frac_gt(tp['range_full'], 0.50):>7.1f}% {frac_gt(m2['range_full'], 0.50):>8.1f}% | {frac_gt(tp['range_full'], 0.70):>7.1f}% {frac_gt(m2['range_full'], 0.70):>8.1f}%"
        print(line)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for kind, color in [("TP_start", "C0"),
                        ("MERGED_GT1_start", "C2"),
                        ("MERGED_GT2plus_start", "C3")]:
        sub = df[df["kind"] == kind]
        if not len(sub): continue
        meds = [sub[f"np_off{off:+d}"].median() for off in OFFSETS]
        q25 = [sub[f"np_off{off:+d}"].quantile(0.25) for off in OFFSETS]
        q75 = [sub[f"np_off{off:+d}"].quantile(0.75) for off in OFFSETS]
        ax.plot(OFFSETS, meds, color=color, lw=2, marker="o", label=f"{kind} (n={len(sub)})")
        ax.fill_between(OFFSETS, q25, q75, alpha=0.15, color=color)
    ax.axvline(0, color="0.5", lw=0.6, ls="--")
    ax.axhline(1.0, color="0.5", lw=0.6, ls=":", label="BOXR")
    ax.axhline(0.5, color="0.7", lw=0.5, ls=":")
    ax.set_xlabel("frame offset from GT.start")
    ax.set_ylabel("hand-centroid norm_pos (0=at BOXL, 1=at BOXR)")
    ax.set_title("Centroid norm_pos profile around reach start (IQR shaded)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0, 1.6, 33)
    for kind, color in [("TP_start", "C0"),
                        ("MERGED_GT1_start", "C2"),
                        ("MERGED_GT2plus_start", "C3")]:
        sub = df[df["kind"] == kind].dropna(subset=["win_max_post"])
        if not len(sub): continue
        ax.hist(sub["win_max_post"].values, bins=bins, alpha=0.4, color=color, density=True,
                label=f"{kind} (n={len(sub)}, med={sub['win_max_post'].median():.2f})")
    ax.axvline(1.0, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("max centroid norm_pos over [GT.start, GT.start+8]")
    ax.set_ylabel("density")
    ax.set_title("Post-start overshoot distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig = OUT_DIR / "figures" / "centroid_norm_pos.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] figure: {out_fig}")


if __name__ == "__main__":
    main()
