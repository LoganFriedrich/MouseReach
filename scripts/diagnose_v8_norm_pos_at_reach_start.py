"""Diagnostic 3: where is the hand at reach START for TPs vs at merged GT2.start?

Test of Logan's clarified hypothesis (2026-05-21): "all reaches start going
towards or near BOXL and usually end on the other side after an arcing motion
near BOXR."

The earlier diagnostic confirmed:
- Merged events DON'T have the hand return to BOXL between consecutive reaches
- They DO have a clean velocity sign-flip at the boundary

The question now: do *isolated TP reaches* start with hand near BOXL? If yes,
that's a structural pattern the merged-pair second reach violates, and a
"reach-start near BOXL" gate could distinguish merge boundaries from real
reach starts.

Procedure: aggregate TP events from the manifests, compute norm_pos at
GT.start (norm_pos = dist_to_BOXL / apparatus_width), build distribution.
Compare to the same measurement at:
  - GT1.start of MERGED events (the "first" reach in a merge -- should
    match TP distribution if hypothesis holds)
  - GT2.start of MERGED events (the "second" reach in a merge -- should
    differ if the merged-pair hand doesn't fully retract)

Run on the same 6 videos as the previous diagnostic so the corpora line up.
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
    ("20251022_CNT0413_P4", "cal", "calibration_loocv"),  # the original paw-visibility mouse
]

OUT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\reach_detection\v8.0.1_dev_norm_pos_at_reach_start")
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)


def dlc_path(vid: str, corpus: str) -> Path:
    base = CAL_DLC_DIR if corpus == "cal" else HOLDOUT_DLC_DIR
    return base / f"{vid}{DLC_SUFFIX}"


def manifest_path(vid: str, manifest_corpus: str) -> Path:
    return MANIFEST_DIR / manifest_corpus / f"{vid}.json"


def main():
    all_records = []
    for vid, dlc_corpus, manifest_corpus in VIDEOS:
        dlc_p = dlc_path(vid, dlc_corpus)
        man_p = manifest_path(vid, manifest_corpus)
        if not dlc_p.exists():
            print(f"[skip] no DLC: {vid}")
            continue
        if not man_p.exists():
            print(f"[skip] no manifest: {vid}")
            continue

        dlc = load_dlc(dlc_p)
        # Smoothed position channels (matches v8 features)
        hand_x = dlc["RightHand_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        hand_y = dlc["RightHand_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxl_x = dlc["BOXL_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxl_y = dlc["BOXL_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxr_x = dlc["BOXR_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxr_y = dlc["BOXR_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        dist_boxl = np.sqrt((hand_x - boxl_x) ** 2 + (hand_y - boxl_y) ** 2)
        apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)
        norm_pos = dist_boxl / np.maximum(apparatus, 1e-3)
        n_frames = len(dlc)

        # Process manifest events
        m = json.loads(man_p.read_text())
        events = m["events"]

        # Index events by topology+component_id
        by_comp = defaultdict(list)
        for e in events:
            if e.get("topology") in ("MERGED",):
                by_comp[e["component_id"]].append(e)

        # 1) TP starts: each TP row has GT start in gt.start
        for e in events:
            if e.get("topology") == "TP" and e.get("gt") and e["gt"].get("start") is not None:
                gs = int(e["gt"]["start"])
                if 0 <= gs < n_frames:
                    all_records.append({
                        "video": vid,
                        "kind": "TP_start",
                        "frame": gs,
                        "norm_pos": float(norm_pos[gs]),
                    })

        # 2) MERGED GT1.start (first GT) and GT2.start (second GT) and beyond
        for comp_id, rows in by_comp.items():
            fns = [r for r in rows if r.get("kind") == "FN" and r.get("gt") and r["gt"].get("start") is not None]
            fns = sorted(fns, key=lambda r: r["gt"]["start"])
            for idx, r in enumerate(fns):
                gs = int(r["gt"]["start"])
                if 0 <= gs < n_frames:
                    label = "MERGED_GT1_start" if idx == 0 else "MERGED_GT2plus_start"
                    all_records.append({
                        "video": vid,
                        "kind": label,
                        "frame": gs,
                        "norm_pos": float(norm_pos[gs]),
                    })

        print(f"[{vid}] TPs={sum(1 for r in all_records if r['video']==vid and r['kind']=='TP_start')}  "
              f"MERGED_GT1={sum(1 for r in all_records if r['video']==vid and r['kind']=='MERGED_GT1_start')}  "
              f"MERGED_GT2+={sum(1 for r in all_records if r['video']==vid and r['kind']=='MERGED_GT2plus_start')}")

    df = pd.DataFrame(all_records)
    out_csv = OUT_DIR / "metrics" / "norm_pos_at_reach_start.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}")
    print(f"Total records: {len(df)}")

    # Summary stats per kind, overall
    print("\n=== OVERALL DISTRIBUTION (norm_pos at reach start) ===")
    summary = df.groupby("kind")["norm_pos"].describe()[["count", "mean", "50%", "25%", "75%"]]
    print(summary)

    print("\n=== PER-VIDEO (median norm_pos at start) ===")
    pivot = df.pivot_table(index="video", columns="kind", values="norm_pos",
                           aggfunc="median", margins=False)
    print(pivot)

    # Histogram comparison: TP_start vs MERGED_GT2plus_start vs MERGED_GT1_start
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: aggregate histograms
    ax = axes[0]
    bins = np.linspace(-0.1, 1.5, 33)
    for kind, color in [("TP_start", "C0"),
                        ("MERGED_GT1_start", "C2"),
                        ("MERGED_GT2plus_start", "C3")]:
        vals = df[df["kind"] == kind]["norm_pos"].values
        if len(vals) == 0: continue
        ax.hist(vals, bins=bins, alpha=0.5, color=color,
                label=f"{kind} (n={len(vals)}, med={np.median(vals):.2f})",
                density=True)
    ax.axvline(0.25, color="0.5", lw=0.6, ls=":", label="0.25 (BOXL zone)")
    ax.set_xlabel("norm_pos at reach START frame (0=at BOXL, 1=at BOXR)")
    ax.set_ylabel("density")
    ax.set_title("Where is the hand when a reach STARTS?\n"
                 "All 7 videos pooled")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: per-video boxplots for TP_start and MERGED_GT2plus_start
    ax = axes[1]
    positions = []
    data = []
    labels = []
    pos = 0
    colors = []
    for vid in df["video"].unique():
        tps = df[(df["video"] == vid) & (df["kind"] == "TP_start")]["norm_pos"].values
        merg2 = df[(df["video"] == vid) & (df["kind"] == "MERGED_GT2plus_start")]["norm_pos"].values
        if len(tps) > 0:
            data.append(tps)
            positions.append(pos)
            labels.append(f"{vid.split('_')[1]}_TP")
            colors.append("C0")
            pos += 1
        if len(merg2) > 0:
            data.append(merg2)
            positions.append(pos)
            labels.append(f"{vid.split('_')[1]}_M2+")
            colors.append("C3")
            pos += 1
        pos += 0.5  # gap between videos
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                    showfliers=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.25, color="0.5", lw=0.6, ls=":")
    ax.set_ylabel("norm_pos at reach start")
    ax.set_title("Per-video: TP starts (blue) vs MERGED GT2+ starts (red)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_fig = OUT_DIR / "figures" / "norm_pos_at_reach_start.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] figure: {out_fig}")

    # Quantitative separability: what fraction of TP_starts are < 0.25, vs MERGED_GT2plus?
    print("\n=== SEPARABILITY (fraction with norm_pos < 0.25, i.e. hand in BOXL zone at start) ===")
    for kind in ["TP_start", "MERGED_GT1_start", "MERGED_GT2plus_start"]:
        sub = df[df["kind"] == kind]
        if len(sub) == 0: continue
        n_at_boxl = int((sub["norm_pos"] < 0.25).sum())
        n_total = len(sub)
        print(f"  {kind}: {n_at_boxl} / {n_total} ({100.0 * n_at_boxl / n_total:.1f}%)")

    print("\n=== SEPARABILITY (fraction with norm_pos < 0.40) ===")
    for kind in ["TP_start", "MERGED_GT1_start", "MERGED_GT2plus_start"]:
        sub = df[df["kind"] == kind]
        if len(sub) == 0: continue
        n_at_boxl = int((sub["norm_pos"] < 0.40).sum())
        n_total = len(sub)
        print(f"  {kind}: {n_at_boxl} / {n_total} ({100.0 * n_at_boxl / n_total:.1f}%)")


if __name__ == "__main__":
    main()
