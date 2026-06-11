"""Diagnostic 4: compare all 4 right-hand keypoints for the
norm_pos-at-reach-start question.

Logan (2026-05-21): "what point of the hand are you looking at? because if
you're looking at RightHand that will be about even when a reach starts
since it's already skewed to the BOXR based on how we label it."

The 4 right-hand keypoints in v8's feature set are:
  RightHand, RHLeft, RHOut, RHRight

Run the TP-vs-MERGED-GT2+ norm_pos comparison for each, see which gives the
biggest separation between reach-start and merged-pair-second-reach-start.
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

HAND_KEYPOINTS = ["RightHand", "RHLeft", "RHOut", "RHRight"]

OUT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\reach_detection\v8.0.1_dev_norm_pos_keypoint_comparison")
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)


def dlc_path(vid, corpus):
    base = CAL_DLC_DIR if corpus == "cal" else HOLDOUT_DLC_DIR
    return base / f"{vid}{DLC_SUFFIX}"


def manifest_path(vid, manifest_corpus):
    return MANIFEST_DIR / manifest_corpus / f"{vid}.json"


def main():
    all_records = []
    keypoint_check = {}  # per-video median (likelihood, x, y) at rest baseline

    for vid, dlc_corpus, manifest_corpus in VIDEOS:
        dlc_p = dlc_path(vid, dlc_corpus)
        man_p = manifest_path(vid, manifest_corpus)
        if not dlc_p.exists() or not man_p.exists():
            print(f"[skip] missing files for {vid}")
            continue

        dlc = load_dlc(dlc_p)
        boxl_x = dlc["BOXL_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxl_y = dlc["BOXL_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxr_x = dlc["BOXR_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxr_y = dlc["BOXR_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)

        norm_pos_per_kp = {}
        for kp in HAND_KEYPOINTS:
            kx = dlc[f"{kp}_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
            ky = dlc[f"{kp}_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
            d_boxl = np.sqrt((kx - boxl_x) ** 2 + (ky - boxl_y) ** 2)
            norm_pos_per_kp[kp] = d_boxl / np.maximum(apparatus, 1e-3)

        m = json.loads(man_p.read_text())
        events = m["events"]

        # TP events
        for e in events:
            if e.get("topology") == "TP" and e.get("gt") and e["gt"].get("start") is not None:
                gs = int(e["gt"]["start"])
                if 0 <= gs < len(dlc):
                    for kp in HAND_KEYPOINTS:
                        all_records.append({
                            "video": vid, "kind": "TP_start", "keypoint": kp,
                            "frame": gs, "norm_pos": float(norm_pos_per_kp[kp][gs]),
                        })

        # MERGED events
        by_comp = defaultdict(list)
        for e in events:
            if e.get("topology") == "MERGED" and e.get("kind") == "FN":
                by_comp[e["component_id"]].append(e)
        for comp_id, rows in by_comp.items():
            rows = sorted(rows, key=lambda r: r["gt"]["start"] if r.get("gt") and r["gt"].get("start") is not None else 0)
            for idx, r in enumerate(rows):
                if not (r.get("gt") and r["gt"].get("start") is not None): continue
                gs = int(r["gt"]["start"])
                if not (0 <= gs < len(dlc)): continue
                label = "MERGED_GT1_start" if idx == 0 else "MERGED_GT2plus_start"
                for kp in HAND_KEYPOINTS:
                    all_records.append({
                        "video": vid, "kind": label, "keypoint": kp,
                        "frame": gs, "norm_pos": float(norm_pos_per_kp[kp][gs]),
                    })

        print(f"[{vid}] processed")

    df = pd.DataFrame(all_records)
    out_csv = OUT_DIR / "metrics" / "norm_pos_by_keypoint.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}")
    print(f"Total records: {len(df)}")

    # Summary: per keypoint, median norm_pos for TP_start vs MERGED_GT2+_start
    print("\n=== MEDIAN norm_pos by KEYPOINT and KIND ===")
    summary = df.groupby(["keypoint", "kind"])["norm_pos"].median().unstack()
    print(summary.to_string())

    print("\n=== SHIFT: median MERGED_GT2plus_start - median TP_start, per keypoint ===")
    shifts = {}
    for kp in HAND_KEYPOINTS:
        tp = df[(df["keypoint"] == kp) & (df["kind"] == "TP_start")]["norm_pos"]
        m2 = df[(df["keypoint"] == kp) & (df["kind"] == "MERGED_GT2plus_start")]["norm_pos"]
        if len(tp) and len(m2):
            shift = float(m2.median() - tp.median())
            shifts[kp] = shift
            print(f"  {kp:12s}  TP_med={tp.median():.3f}  M2+_med={m2.median():.3f}  shift={shift:+.3f}")

    print("\n=== SEPARABILITY (fraction with norm_pos < threshold) ===")
    for kp in HAND_KEYPOINTS:
        print(f"\n  Keypoint: {kp}")
        for thresh in (0.10, 0.25, 0.40):
            tp = df[(df["keypoint"] == kp) & (df["kind"] == "TP_start")]
            m2 = df[(df["keypoint"] == kp) & (df["kind"] == "MERGED_GT2plus_start")]
            tp_frac = 100.0 * (tp["norm_pos"] < thresh).sum() / max(len(tp), 1)
            m2_frac = 100.0 * (m2["norm_pos"] < thresh).sum() / max(len(m2), 1)
            print(f"    norm_pos < {thresh:.2f}:  TP={tp_frac:.1f}%   M2+={m2_frac:.1f}%   diff={tp_frac - m2_frac:+.1f}pp")

    # Plot: 4 keypoint panels, each showing TP vs MERGED_GT2+ histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    bins = np.linspace(-0.2, 1.6, 37)
    for i, kp in enumerate(HAND_KEYPOINTS):
        ax = axes[i]
        tp = df[(df["keypoint"] == kp) & (df["kind"] == "TP_start")]["norm_pos"].values
        m1 = df[(df["keypoint"] == kp) & (df["kind"] == "MERGED_GT1_start")]["norm_pos"].values
        m2 = df[(df["keypoint"] == kp) & (df["kind"] == "MERGED_GT2plus_start")]["norm_pos"].values
        ax.hist(tp, bins=bins, alpha=0.4, color="C0", density=True,
                label=f"TP (n={len(tp)}, med={np.median(tp):.2f})")
        if len(m1):
            ax.hist(m1, bins=bins, alpha=0.4, color="C2", density=True,
                    label=f"MERGED_GT1 (n={len(m1)}, med={np.median(m1):.2f})")
        if len(m2):
            ax.hist(m2, bins=bins, alpha=0.4, color="C3", density=True,
                    label=f"MERGED_GT2+ (n={len(m2)}, med={np.median(m2):.2f})")
        ax.axvline(0.25, color="0.5", lw=0.6, ls=":")
        ax.set_title(f"{kp}")
        ax.set_xlabel("norm_pos at reach start (0=at BOXL, 1=at BOXR)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("norm_pos at reach START -- comparison across 4 right-hand keypoints\n"
                 "(pooled across 7 videos)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_fig = OUT_DIR / "figures" / "norm_pos_keypoint_comparison.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] figure: {out_fig}")


if __name__ == "__main__":
    main()
