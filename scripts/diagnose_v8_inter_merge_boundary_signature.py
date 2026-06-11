"""Diagnostic 7: does the centroid feature actually fire AT the
inter-GT-merge boundary frame, vs other within-reach frames?

The post-start-max feature distinguishes TP starts from MERGED-GT2+ starts
when measured at GT.start frames. But the model needs a signal that fires
specifically at the *boundary frame* of a merge (within an algo-merged
span), distinguishing it from genuinely-in-reach frames.

Test the "valley between two peaks" signature:
  past_max_8(F)   = max of centroid norm_pos in [F-8, F]
  future_max_8(F) = max of centroid norm_pos in [F, F+8]
  valley_depth(F) = min(past_max_8, future_max_8) - current_norm_pos

At a real reach boundary inside a merged algo span:
  - past 8 frames just had the GT1 apex (high)
  - future 8 frames contain the GT2 apex (high)
  - current frame is mid-retract (low)
  -> valley_depth is large

At a typical in-reach frame:
  - either past max is low (early in reach) or future max is low (late)
  - or current is high (near apex)
  -> valley_depth is small

Verify on actual MERGED algo spans, compare to TP algo spans in same video.
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

HAND_KPS = ["RightHand", "RHLeft", "RHOut", "RHRight"]
WINDOW_HALF = 8   # ±8 frame window for past_max / future_max

# All 7 videos
VIDEOS = [
    ("20250630_CNT0104_P3", "cal", "calibration_loocv"),
    ("20250710_CNT0215_P4", "cal", "calibration_loocv"),
    ("20250718_CNT0214_P1", "holdout", "holdout_2026_05_11"),
    ("20250812_CNT0301_P3", "cal", "calibration_loocv"),
    ("20251010_CNT0308_P2", "cal", "calibration_loocv"),
    ("20250806_CNT0316_P3", "holdout", "holdout_2026_05_11"),
    ("20251022_CNT0413_P4", "cal", "calibration_loocv"),
]

OUT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\reach_detection\v8.0.1_dev_inter_merge_boundary_signature")
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)


def dlc_path(vid, corpus):
    base = CAL_DLC_DIR if corpus == "cal" else HOLDOUT_DLC_DIR
    return base / f"{vid}{DLC_SUFFIX}"


def manifest_path(vid, manifest_corpus):
    return MANIFEST_DIR / manifest_corpus / f"{vid}.json"


def compute_centroid_norm_pos(dlc):
    boxl_x = dlc["BOXL_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    boxl_y = dlc["BOXL_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    boxr_x = dlc["BOXR_x"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    boxr_y = dlc["BOXR_y"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)
    cx_list, cy_list = [], []
    for kp in HAND_KPS:
        cx_list.append(dlc[f"{kp}_x"].rolling(5, center=True, min_periods=1).mean().to_numpy())
        cy_list.append(dlc[f"{kp}_y"].rolling(5, center=True, min_periods=1).mean().to_numpy())
    cx = np.mean(cx_list, axis=0)
    cy = np.mean(cy_list, axis=0)
    d_boxl = np.sqrt((cx - boxl_x) ** 2 + (cy - boxl_y) ** 2)
    norm_pos = d_boxl / np.maximum(apparatus, 1e-3)
    return norm_pos


def rolling_past_max(arr, w):
    """Max over [t-w, t] for each t."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float32)
    for t in range(n):
        lo = max(0, t - w)
        out[t] = float(np.max(arr[lo:t + 1]))
    return out


def rolling_future_max(arr, w):
    """Max over [t, t+w] for each t."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float32)
    for t in range(n):
        hi = min(n, t + w + 1)
        out[t] = float(np.max(arr[t:hi]))
    return out


def main():
    all_rows = []
    fig_examples = []  # collect example MERGED + TP from each video for figure

    for vid, dlc_corpus, manifest_corpus in VIDEOS:
        dlc_p = dlc_path(vid, dlc_corpus)
        man_p = manifest_path(vid, manifest_corpus)
        if not dlc_p.exists() or not man_p.exists():
            print(f"[skip] {vid}")
            continue
        dlc = load_dlc(dlc_p)
        norm_pos = compute_centroid_norm_pos(dlc)
        past_max = rolling_past_max(norm_pos, WINDOW_HALF)
        future_max = rolling_future_max(norm_pos, WINDOW_HALF)
        valley_depth = np.minimum(past_max, future_max) - norm_pos
        n = len(dlc)

        m = json.loads(man_p.read_text())
        events = m["events"]

        # Group MERGED by component
        merged_by_comp = defaultdict(list)
        for e in events:
            if e.get("topology") == "MERGED":
                merged_by_comp[e["component_id"]].append(e)

        # 1) Inter-GT boundary frames within MERGED algo spans
        #    For each MERGED component, between consecutive GTs we pick the boundary frame.
        #    Boundary frame = GT1.end frame (last frame labeled as in reach 1; the next frame
        #    is GT2.start or a gap frame).
        for cid, rows in merged_by_comp.items():
            fns = sorted([r for r in rows if r.get("kind") == "FN"
                          and r.get("gt") and r["gt"].get("start") is not None],
                         key=lambda r: r["gt"]["start"])
            for i in range(len(fns) - 1):
                gt1_end = int(fns[i]["gt"]["end"])
                gt2_start = int(fns[i + 1]["gt"]["start"])
                gap = gt2_start - gt1_end - 1
                # Pick the boundary frame: the midpoint of [gt1_end, gt2_start]
                # For gap=0, boundary = gt1_end (or gt2_start - 1 = gt1_end)
                # For gap>0, boundary = midpoint
                boundary_frame = (gt1_end + gt2_start) // 2
                if not (0 <= boundary_frame < n): continue
                all_rows.append({
                    "video": vid,
                    "kind": "MERGE_BOUNDARY",
                    "frame": boundary_frame,
                    "gap": gap,
                    "current": float(norm_pos[boundary_frame]),
                    "past_max_8": float(past_max[boundary_frame]),
                    "future_max_8": float(future_max[boundary_frame]),
                    "valley_depth": float(valley_depth[boundary_frame]),
                })

        # 2) In-reach frames from TP events: sample the MIDDLE of each TP reach
        for e in events:
            if e.get("topology") == "TP" and e.get("gt") and e["gt"].get("start") is not None:
                gs = int(e["gt"]["start"])
                ge = int(e["gt"]["end"])
                mid = (gs + ge) // 2
                if not (0 <= mid < n): continue
                all_rows.append({
                    "video": vid,
                    "kind": "TP_MID",
                    "frame": mid,
                    "gap": None,
                    "current": float(norm_pos[mid]),
                    "past_max_8": float(past_max[mid]),
                    "future_max_8": float(future_max[mid]),
                    "valley_depth": float(valley_depth[mid]),
                })

        # 3) Between-reach frames (rest frames -- well outside any algo reach)
        #    Pick frames that are at least 20 frames before/after any GT reach.
        #    Use the segmentation boundaries if available; else just pick 30f gaps.
        all_gts = [(int(e["gt"]["start"]), int(e["gt"]["end"])) for e in events
                   if e.get("gt") and e["gt"].get("start") is not None and e["gt"].get("end") is not None]
        all_gts = sorted(set(all_gts))
        # Iterate pairs of consecutive GTs, take a frame in between if gap is large enough.
        rest_samples = []
        for i in range(len(all_gts) - 1):
            g1e = all_gts[i][1]
            g2s = all_gts[i + 1][0]
            if g2s - g1e > 60:  # at least 60 frames of rest -- take midpoint
                rest_samples.append((g1e + g2s) // 2)
        # Subsample to keep balanced (max 200 per video)
        if len(rest_samples) > 200:
            rest_samples = np.random.RandomState(0).choice(rest_samples, 200, replace=False).tolist()
        for rf in rest_samples:
            if not (0 <= rf < n): continue
            all_rows.append({
                "video": vid,
                "kind": "REST",
                "frame": rf,
                "gap": None,
                "current": float(norm_pos[rf]),
                "past_max_8": float(past_max[rf]),
                "future_max_8": float(future_max[rf]),
                "valley_depth": float(valley_depth[rf]),
            })

        # Pick ONE representative MERGED component and one neighboring TP for the figure
        merged_2gt = [(cid, rows) for cid, rows in merged_by_comp.items()
                      if len([r for r in rows if r.get("kind") == "FN"]) == 2]
        if merged_2gt and not any(e[0] == vid for e in fig_examples):
            cid, rows = merged_2gt[0]
            fns = sorted([r for r in rows if r.get("kind") == "FN"],
                         key=lambda r: r["gt"]["start"])
            fp = next(r for r in rows if r.get("kind") == "FP")
            gt1 = (int(fns[0]["gt"]["start"]), int(fns[0]["gt"]["end"]))
            gt2 = (int(fns[1]["gt"]["start"]), int(fns[1]["gt"]["end"]))
            algo = (int(fp["detector"]["start"]), int(fp["detector"]["end"]))
            # Find a TP in same video to compare
            tp_candidates = [e for e in events if e.get("topology") == "TP"
                             and e.get("gt") and e["gt"].get("start") is not None]
            if tp_candidates:
                tp_event = tp_candidates[len(tp_candidates) // 2]  # middle TP
                tp_gt = (int(tp_event["gt"]["start"]), int(tp_event["gt"]["end"]))
                fig_examples.append((vid, gt1, gt2, algo, tp_gt,
                                     norm_pos, past_max, future_max, valley_depth))

        print(f"[{vid}] processed: {sum(1 for r in all_rows if r['video']==vid)} samples")

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "metrics" / "boundary_signature.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}")
    print(f"Total records: {len(df)}\n")

    # Summary
    print("=== POOLED feature distributions (centroid; window ±8) ===")
    for kind in ["MERGE_BOUNDARY", "TP_MID", "REST"]:
        sub = df[df["kind"] == kind]
        if not len(sub): continue
        print(f"\n  {kind} (n={len(sub)}):")
        for f in ["current", "past_max_8", "future_max_8", "valley_depth"]:
            vals = sub[f].dropna()
            print(f"    {f:<14}: median={vals.median():.3f}, "
                  f"25%={vals.quantile(0.25):.3f}, 75%={vals.quantile(0.75):.3f}")

    # Per-video valley_depth at merge boundaries
    print("\n=== PER-VIDEO: valley_depth at MERGE_BOUNDARY vs TP_MID ===")
    print(f"{'video':<22} | {'n_MB':>4} {'n_TP':>5} | {'MB med':>7} {'TP med':>7} | {'MB>0.30':>8} {'TP>0.30':>8} | {'MB>0.50':>8} {'TP>0.50':>8}")
    print("-" * 110)
    for vid in sorted(df['video'].unique()):
        mb = df[(df['video']==vid) & (df['kind']=='MERGE_BOUNDARY')].dropna(subset=['valley_depth'])
        tp = df[(df['video']==vid) & (df['kind']=='TP_MID')].dropna(subset=['valley_depth'])
        if not len(mb): continue
        mb_med = mb['valley_depth'].median()
        tp_med = tp['valley_depth'].median() if len(tp) else float('nan')
        def frac_gt(s, t): return 100.0 * (s > t).sum() / len(s) if len(s) else float('nan')
        line = f"{vid:<22} | {len(mb):>4} {len(tp):>5} | {mb_med:>7.3f} {tp_med:>7.3f} | {frac_gt(mb['valley_depth'], 0.30):>7.1f}% {frac_gt(tp['valley_depth'], 0.30):>7.1f}% | {frac_gt(mb['valley_depth'], 0.50):>7.1f}% {frac_gt(tp['valley_depth'], 0.50):>7.1f}%"
        print(line)

    # Plot per-video example
    if fig_examples:
        fig, axes = plt.subplots(len(fig_examples), 2, figsize=(16, 3.5 * len(fig_examples)),
                                 squeeze=False)
        for i, (vid, gt1, gt2, algo, tp_gt, norm_pos, past_max, future_max, valley_depth) in enumerate(fig_examples):
            # Window: covers algo span + 10 on each side
            win_lo = max(0, algo[0] - 10)
            win_hi = min(len(norm_pos), algo[1] + 10)
            frames = np.arange(win_lo, win_hi)

            ax = axes[i, 0]
            ax.plot(frames, norm_pos[win_lo:win_hi], color="C0", lw=1.5, label="centroid norm_pos")
            ax.plot(frames, past_max[win_lo:win_hi], color="C2", lw=0.8, alpha=0.7, label="past_max_8")
            ax.plot(frames, future_max[win_lo:win_hi], color="C3", lw=0.8, alpha=0.7, label="future_max_8")
            ax.axvspan(gt1[0], gt1[1], alpha=0.15, color="C2", label="GT1")
            ax.axvspan(gt2[0], gt2[1], alpha=0.15, color="C4", label="GT2")
            ax.axvline(algo[0], color="C1", ls="--", lw=0.6)
            ax.axvline(algo[1], color="C1", ls="--", lw=0.6, label="algo")
            ax.axhline(1.0, color="0.5", lw=0.4, ls=":")
            ax.set_title(f"{vid}: MERGED span (algo={algo[0]}-{algo[1]}, gap={gt2[0]-gt1[1]-1})")
            ax.set_xlabel("frame")
            ax.set_ylabel("centroid norm_pos")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

            ax = axes[i, 1]
            ax.plot(frames, valley_depth[win_lo:win_hi], color="C5", lw=1.5, label="valley_depth")
            ax.axhline(0, color="0.5", lw=0.4)
            ax.axhline(0.30, color="0.5", lw=0.4, ls=":", label="0.30 threshold")
            ax.axvspan(gt1[0], gt1[1], alpha=0.15, color="C2")
            ax.axvspan(gt2[0], gt2[1], alpha=0.15, color="C4")
            ax.axvline(algo[0], color="C1", ls="--", lw=0.6)
            ax.axvline(algo[1], color="C1", ls="--", lw=0.6)
            ax.set_title("valley_depth = min(past_max_8, future_max_8) - current")
            ax.set_xlabel("frame")
            ax.set_ylabel("valley_depth")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_fig = OUT_DIR / "figures" / "boundary_signature_examples.png"
        fig.savefig(out_fig, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[OK] figure: {out_fig}")


if __name__ == "__main__":
    main()
