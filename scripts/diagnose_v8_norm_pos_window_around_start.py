"""Diagnostic 5: profile RHLeft norm_pos in a window around reach start.

Logan (2026-05-21): "the first frame is usually more centered in comparison
to the second frame of a reach which usually seems to be heavily skewed
toward BOXL."

Previous diagnostic showed RHLeft at GT.start has median norm_pos 0.25 for
TPs, in the BOXL zone. But the paw orientation may not fully commit on
frame 0. If reaches genuinely "skew heavily toward BOXL" by frame +1 or +2,
then a feature like min(RHLeft_norm_pos) over [GT.start-2, GT.start+5]
could be much more separable than the point measurement at GT.start.

Procedure: for each TP, MERGED_GT1, and MERGED_GT2+ start frame, extract
the RHLeft norm_pos profile across offsets [-3, -2, -1, 0, +1, +2, +3, +5, +8].
Compare medians per offset. Also compute window-min profile to test the
"deepest skew" idea.
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

# Per-frame offsets relative to GT.start to evaluate
OFFSETS = list(range(-5, 11))  # -5 to +10 inclusive
# Window over which to compute min norm_pos (the "deepest BOXL skew" feature)
WIN_LO, WIN_HI = -2, 5  # [GT.start - 2, GT.start + 5] inclusive

OUT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\reach_detection\v8.0.1_dev_norm_pos_window_around_start")
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
        # Smoothed channels matching v8 features
        boxl_x = dlc["BOXL_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxl_y = dlc["BOXL_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxr_x = dlc["BOXR_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        boxr_y = dlc["BOXR_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        kx = dlc["RHLeft_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        ky = dlc["RHLeft_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        d_boxl = np.sqrt((kx - boxl_x) ** 2 + (ky - boxl_y) ** 2)
        apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)
        norm_pos = d_boxl / np.maximum(apparatus, 1e-3)
        n = len(dlc)

        m = json.loads(man_p.read_text())
        events = m["events"]

        # Collect (kind, gt_start_frame) per event start
        kinds = []
        for e in events:
            if e.get("topology") == "TP" and e.get("gt") and e["gt"].get("start") is not None:
                kinds.append(("TP_start", int(e["gt"]["start"])))
        # MERGED
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

        # Profile each event across offsets
        for kind, gs in kinds:
            if not (0 <= gs < n): continue
            rec = {"video": vid, "kind": kind, "gt_start": gs}
            # Per-offset norm_pos values
            offset_vals = []
            for off in OFFSETS:
                f = gs + off
                if 0 <= f < n:
                    rec[f"np_off{off:+d}"] = float(norm_pos[f])
                    offset_vals.append((off, float(norm_pos[f])))
                else:
                    rec[f"np_off{off:+d}"] = np.nan
            # Window min over [WIN_LO, WIN_HI]
            window_vals = [v for o, v in offset_vals if WIN_LO <= o <= WIN_HI]
            if window_vals:
                rec["win_min"] = float(np.min(window_vals))
                rec["win_argmin_offset"] = int(np.argmin([v for o, v in offset_vals if WIN_LO <= o <= WIN_HI]) + WIN_LO)
            else:
                rec["win_min"] = np.nan
                rec["win_argmin_offset"] = np.nan
            rows.append(rec)

        n_tp = sum(1 for r in rows if r.get("video") == vid and r["kind"] == "TP_start")
        n_m1 = sum(1 for r in rows if r.get("video") == vid and r["kind"] == "MERGED_GT1_start")
        n_m2 = sum(1 for r in rows if r.get("video") == vid and r["kind"] == "MERGED_GT2plus_start")
        print(f"[{vid}] TP={n_tp} M1={n_m1} M2+={n_m2}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "metrics" / "norm_pos_window_around_start.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}")
    print(f"Total records: {len(df)}")

    # Median norm_pos at each offset, per kind
    print("\n=== MEDIAN RHLeft norm_pos at each offset relative to GT.start ===")
    offset_cols = [f"np_off{off:+d}" for off in OFFSETS]
    summary = df.groupby("kind")[offset_cols].median()
    print(summary.to_string())

    # Window-min stats
    print("\n=== WINDOW-MIN of RHLeft norm_pos over [GT.start-2, GT.start+5] ===")
    for kind in ["TP_start", "MERGED_GT1_start", "MERGED_GT2plus_start"]:
        sub = df[df["kind"] == kind].dropna(subset=["win_min"])
        if len(sub) == 0: continue
        print(f"  {kind}: n={len(sub)}, median={sub['win_min'].median():.3f}, "
              f"25%={sub['win_min'].quantile(0.25):.3f}, 75%={sub['win_min'].quantile(0.75):.3f}")

    # Separability at different thresholds (using window-min)
    print("\n=== SEPARABILITY (fraction with win_min < threshold) ===")
    for thresh in (0.10, 0.15, 0.20, 0.25, 0.30):
        print(f"\n  win_min < {thresh}:")
        for kind in ["TP_start", "MERGED_GT1_start", "MERGED_GT2plus_start"]:
            sub = df[df["kind"] == kind].dropna(subset=["win_min"])
            if len(sub) == 0: continue
            frac = 100.0 * (sub["win_min"] < thresh).sum() / len(sub)
            print(f"    {kind}: {frac:.1f}%")

    # Which offset minimizes norm_pos per kind?
    print("\n=== argmin OFFSET distribution (which frame in window has deepest BOXL skew) ===")
    for kind in ["TP_start", "MERGED_GT1_start", "MERGED_GT2plus_start"]:
        sub = df[df["kind"] == kind].dropna(subset=["win_argmin_offset"])
        if len(sub) == 0: continue
        counts = sub["win_argmin_offset"].value_counts().sort_index()
        print(f"\n  {kind} (n={len(sub)}):")
        for off, cnt in counts.items():
            print(f"    offset {int(off):+d}: {cnt} ({100.0 * cnt / len(sub):.1f}%)")

    # Plot 1: median profile per kind across offsets
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for kind, color in [("TP_start", "C0"),
                        ("MERGED_GT1_start", "C2"),
                        ("MERGED_GT2plus_start", "C3")]:
        sub = df[df["kind"] == kind]
        if len(sub) == 0: continue
        medians = [sub[f"np_off{off:+d}"].median() for off in OFFSETS]
        q25 = [sub[f"np_off{off:+d}"].quantile(0.25) for off in OFFSETS]
        q75 = [sub[f"np_off{off:+d}"].quantile(0.75) for off in OFFSETS]
        ax.plot(OFFSETS, medians, color=color, lw=2, marker="o",
                label=f"{kind} (n={len(sub)})")
        ax.fill_between(OFFSETS, q25, q75, alpha=0.15, color=color)
    ax.axvline(0, color="0.5", lw=0.6, ls="--")
    ax.axhline(0.25, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("frame offset from GT.start")
    ax.set_ylabel("RHLeft norm_pos (0=at BOXL, 1=at BOXR)")
    ax.set_title("Median norm_pos profile around reach start\n"
                 "(IQR shaded; dotted line = BOXL zone threshold 0.25)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: histogram of window-min values per kind
    ax = axes[1]
    bins = np.linspace(-0.05, 1.2, 26)
    for kind, color in [("TP_start", "C0"),
                        ("MERGED_GT1_start", "C2"),
                        ("MERGED_GT2plus_start", "C3")]:
        sub = df[df["kind"] == kind].dropna(subset=["win_min"])
        if len(sub) == 0: continue
        ax.hist(sub["win_min"].values, bins=bins, alpha=0.4, color=color, density=True,
                label=f"{kind} (n={len(sub)}, med={sub['win_min'].median():.2f})")
    ax.axvline(0.25, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("min RHLeft norm_pos over [GT.start-2, GT.start+5]")
    ax.set_ylabel("density")
    ax.set_title('"Deepest BOXL skew" feature\nover frames around reach start')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig = OUT_DIR / "figures" / "norm_pos_window_around_start.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] figure: {out_fig}")


if __name__ == "__main__":
    main()
