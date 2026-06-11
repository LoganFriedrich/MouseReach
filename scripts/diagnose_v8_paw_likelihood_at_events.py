"""Diagnostic: paw-likelihood profile per event class on model 3.1 DLC.

Tests whether DLC hand-keypoint likelihood separates the failure classes
(early-start tolerance misses, merged-algo-span inter-GT gaps, phantom FPs)
from clean TPs. Uses TWO paw-confidence signals computed from the 4 hand
keypoints' per-frame likelihoods:
  paw_mean_lk[t] = mean(RightHand_lk[t], RHLeft_lk[t], RHOut_lk[t], RHRight_lk[t])
  paw_min_lk[t]  = min(those four)

These are cross-keypoint aggregations the GBM doesn't currently see as
single features (per the architecture discussion 2026-05-21). Computed
on-the-fly from the new-DLC (model 3.1) parquet.

Analysis windows per event use the UNION extent (max of algo + GT spans),
so that:
- Early-start FPs include both the algo's early frames AND the true GT start
- Over-extensions include both the GT end AND algo's trailing frames
- Merges span the full combined window across both GT halves and the gap

Diagnostic-only. Read-only on parquet + LOOCV JSON. No retraining, no model
code touched.
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


PARQUET_PATH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
LOOCV_PATH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_model_3_1_baseline_loocv\metrics\loocv_aggregate.json"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_dev_paw_lk_diagnostic"
)
(OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

HAND_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
BOUNDARY_N_FRAMES = 3  # first/last N frames of union extent for boundary analysis


def classify_event_with_overlap(row, all_algos_by_video, all_gts_by_video):
    """Classify each LOOCV result row using its status PLUS overlap-checks
    against the per-video algo and GT catalogs.

    The strict matcher drops a match when tolerance fails, leaving an algo
    row with gt_start_frame=-1 even though the algo's span overlaps a real
    GT in the video. Same goes for unmatched GTs whose corresponding algo
    span was dropped. We disambiguate by direct overlap.
    """
    def overlap(a_s, a_e, b_s, b_e):
        return not (a_e < b_s or a_s > b_e)

    algo_present = row["algo_start_frame"] >= 0
    gt_present = row["gt_start_frame"] >= 0
    vid = row["video_id"]

    if not algo_present and not gt_present:
        return "no_data"

    if row["status"] == "tp" and algo_present and gt_present:
        # Matched. Apply strict-tolerance check to label clean vs marginal.
        start_delta = row["start_delta"]
        span_delta = row["span_delta"]
        gt_span = row["gt_end_frame"] - row["gt_start_frame"] + 1
        span_tol = max(0.5 * gt_span, 5)
        if start_delta is not None and span_delta is not None:
            if abs(start_delta) <= 2 and abs(span_delta) <= span_tol:
                return "TP_clean"
            else:
                return "TP_tolerance_marginal"
        return "TP_clean"

    # Unmatched algo (status="fp", gt=-1): check overlap with GTs in same video
    if algo_present and not gt_present:
        algo_s = row["algo_start_frame"]
        algo_e = row["algo_end_frame"]
        overlapping_gts = [
            g for g in all_gts_by_video.get(vid, [])
            if overlap(algo_s, algo_e, g[0], g[1])
        ]
        if not overlapping_gts:
            return "true_phantom_FP"
        # Has overlap, so this is a tolerance-failure FP.
        # Subclassify by start position relative to nearest-start GT
        nearest = min(overlapping_gts, key=lambda g: abs(g[0] - algo_s))
        delta_start = algo_s - nearest[0]
        delta_end = algo_e - nearest[1]
        algo_span = algo_e - algo_s + 1
        gt_span = nearest[1] - nearest[0] + 1
        # Detect MERGED-style: algo overlaps multiple GTs
        if len(overlapping_gts) >= 2:
            return "tolerance_FP_merged_algo"
        # Single-GT overlap: tolerance failure
        if delta_start < -2:
            return "tolerance_FP_start_early"  # algo started too early
        if delta_start > 2:
            return "tolerance_FP_start_late"
        # Start within tolerance, so span must have failed
        span_tol = max(0.5 * gt_span, 5)
        span_delta = algo_span - gt_span
        if span_delta > span_tol:
            return "tolerance_FP_span_over"
        if span_delta < -span_tol:
            return "tolerance_FP_span_short"
        return "tolerance_FP_unclassified"

    # Unmatched GT (status="fn", algo=-1): check overlap with algo in same video
    if gt_present and not algo_present:
        gt_s = row["gt_start_frame"]
        gt_e = row["gt_end_frame"]
        overlapping_algos = [
            a for a in all_algos_by_video.get(vid, [])
            if overlap(gt_s, gt_e, a[0], a[1])
        ]
        if not overlapping_algos:
            return "true_miss_FN"
        # GT overlaps an algo span that was matched to a different GT (MERGED/FRAGMENTED)
        # OR algo span was matched to this GT but matcher dropped it (tolerance failure)
        if len(overlapping_algos) >= 2:
            return "tolerance_FN_fragmented_gt"
        return "tolerance_FN_paired_with_FP"

    return "uncategorized"


def union_extent(row):
    """[min_start, max_end] of algo + GT (handle -1 sentinels)."""
    starts = []
    ends = []
    if row["algo_start_frame"] >= 0:
        starts.append(int(row["algo_start_frame"]))
        ends.append(int(row["algo_end_frame"]))
    if row["gt_start_frame"] >= 0:
        starts.append(int(row["gt_start_frame"]))
        ends.append(int(row["gt_end_frame"]))
    return min(starts), max(ends)


def find_merged_components(rows_by_video):
    """Per video, identify algo reaches that overlap >1 GT (MERGED topology).
    Returns dict: video_id -> list of {algo_start, algo_end, gt_list}.
    """
    def overlap(a_s, a_e, b_s, b_e):
        return not (a_e < b_s or a_s > b_e)

    out = {}
    for vid, rows in rows_by_video.items():
        # Collect unique algo and gt entries
        algos = []
        gts = []
        seen_algo = set()
        seen_gt = set()
        for r in rows:
            if r["algo_start_frame"] >= 0:
                key = (r["algo_start_frame"], r["algo_end_frame"])
                if key not in seen_algo:
                    seen_algo.add(key)
                    algos.append({"start": r["algo_start_frame"],
                                  "end": r["algo_end_frame"]})
            if r["gt_start_frame"] >= 0:
                key = (r["gt_start_frame"], r["gt_end_frame"])
                if key not in seen_gt:
                    seen_gt.add(key)
                    gts.append({"start": r["gt_start_frame"],
                                "end": r["gt_end_frame"]})
        merged = []
        for a in algos:
            overlaps = [g for g in gts if overlap(a["start"], a["end"],
                                                  g["start"], g["end"])]
            if len(overlaps) >= 2:
                merged.append({
                    "algo_start": a["start"], "algo_end": a["end"],
                    "gt_list": sorted(overlaps, key=lambda g: g["start"]),
                })
        out[vid] = merged
    return out


def main():
    print("=" * 70)
    print("PAW LIKELIHOOD DIAGNOSTIC -- model 3.1 DLC")
    print("=" * 70)
    print()

    print("Loading LOOCV aggregate...", flush=True)
    loocv = json.loads(LOOCV_PATH.read_text(encoding="utf-8"))
    results = loocv["raw_results"]
    print(f"  {len(results)} events across {len(set(r['video_id'] for r in results))} videos")

    print("Loading parquet (paw lk columns only)...", flush=True)
    cols_needed = ["video_id", "frame"] + HAND_LK_COLS
    df = pd.read_parquet(PARQUET_PATH, columns=cols_needed)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # Build per-frame paw_mean_lk and paw_min_lk; index by (video_id, frame)
    print("Computing paw_mean_lk and paw_min_lk per frame...", flush=True)
    lk_matrix = df[HAND_LK_COLS].to_numpy(dtype=np.float32)
    df["paw_mean_lk"] = lk_matrix.mean(axis=1)
    df["paw_min_lk"] = lk_matrix.min(axis=1)

    # Build a fast lookup: dict[video_id] -> 1D np.array indexed by frame
    print("Building per-video lk arrays...", flush=True)
    mean_lk_by_vid = {}
    min_lk_by_vid = {}
    for vid, grp in df.groupby("video_id", sort=False):
        grp_sorted = grp.sort_values("frame")
        max_frame = int(grp_sorted["frame"].max())
        mean_arr = np.full(max_frame + 1, np.nan, dtype=np.float32)
        min_arr = np.full(max_frame + 1, np.nan, dtype=np.float32)
        mean_arr[grp_sorted["frame"].to_numpy()] = grp_sorted["paw_mean_lk"].to_numpy()
        min_arr[grp_sorted["frame"].to_numpy()] = grp_sorted["paw_min_lk"].to_numpy()
        mean_lk_by_vid[vid] = mean_arr
        min_lk_by_vid[vid] = min_arr

    # ===== Build per-video algo+GT catalogs for overlap checks =====
    print("Building per-video algo and GT catalogs (for overlap-based classification)...", flush=True)
    all_algos_by_video = defaultdict(list)
    all_gts_by_video = defaultdict(list)
    seen_algo = defaultdict(set)
    seen_gt = defaultdict(set)
    for r in results:
        vid = r["video_id"]
        if r["algo_start_frame"] >= 0:
            key = (r["algo_start_frame"], r["algo_end_frame"])
            if key not in seen_algo[vid]:
                seen_algo[vid].add(key)
                all_algos_by_video[vid].append(key)
        if r["gt_start_frame"] >= 0:
            key = (r["gt_start_frame"], r["gt_end_frame"])
            if key not in seen_gt[vid]:
                seen_gt[vid].add(key)
                all_gts_by_video[vid].append(key)

    # ===== Per-event analysis =====
    print("Classifying events and computing per-event paw lk stats...", flush=True)
    event_rows = []
    for r in results:
        vid = r["video_id"]
        if vid not in mean_lk_by_vid:
            continue
        kind = classify_event_with_overlap(r, all_algos_by_video, all_gts_by_video)
        if kind == "no_data":
            continue
        win_lo, win_hi = union_extent(r)
        mean_lk = mean_lk_by_vid[vid][win_lo:win_hi + 1]
        min_lk = min_lk_by_vid[vid][win_lo:win_hi + 1]
        # Drop NaN frames (out-of-range; shouldn't happen but defensive)
        valid_mean = mean_lk[~np.isnan(mean_lk)]
        valid_min = min_lk[~np.isnan(min_lk)]
        if len(valid_mean) == 0:
            continue
        # Boundary-frame stats (first BOUNDARY_N_FRAMES and last BOUNDARY_N_FRAMES)
        n = len(valid_mean)
        n_bnd = min(BOUNDARY_N_FRAMES, n // 2 if n >= 2 * BOUNDARY_N_FRAMES else n // 2)
        if n_bnd < 1:
            n_bnd = 1
        leading_mean_lk = float(np.mean(valid_mean[:n_bnd]))
        leading_min_lk = float(np.mean(valid_min[:n_bnd]))
        trailing_mean_lk = float(np.mean(valid_mean[-n_bnd:]))
        trailing_min_lk = float(np.mean(valid_min[-n_bnd:]))

        event_rows.append({
            "video": vid,
            "kind": kind,
            "extent_start": win_lo,
            "extent_end": win_hi,
            "extent_span": win_hi - win_lo + 1,
            "algo_start": int(r["algo_start_frame"]),
            "algo_end": int(r["algo_end_frame"]),
            "gt_start": int(r["gt_start_frame"]),
            "gt_end": int(r["gt_end_frame"]),
            "start_delta": r.get("start_delta"),
            "span_delta": r.get("span_delta"),
            # Aggregate over union extent
            "paw_mean_lk_mean": float(valid_mean.mean()),
            "paw_mean_lk_min": float(valid_mean.min()),
            "paw_mean_lk_frac_above_0.5": float((valid_mean > 0.5).mean()),
            "paw_mean_lk_frac_above_0.8": float((valid_mean > 0.8).mean()),
            "paw_min_lk_mean": float(valid_min.mean()),
            "paw_min_lk_min": float(valid_min.min()),
            "paw_min_lk_frac_above_0.5": float((valid_min > 0.5).mean()),
            "paw_min_lk_frac_above_0.8": float((valid_min > 0.8).mean()),
            # Boundary frames
            "leading_mean_lk": leading_mean_lk,
            "leading_min_lk": leading_min_lk,
            "trailing_mean_lk": trailing_mean_lk,
            "trailing_min_lk": trailing_min_lk,
        })

    edf = pd.DataFrame(event_rows)
    out_csv = OUT_DIR / "metrics" / "per_event_paw_lk.csv"
    edf.to_csv(out_csv, index=False)
    print(f"  saved per-event records: {out_csv} ({len(edf)} rows)")
    print()

    # ===== Per-kind summary =====
    print("=" * 70)
    print("PER-CLASS DISTRIBUTIONS (median over union extent)")
    print("=" * 70)
    print()
    print(f"{'kind':<28} {'n':>5}  "
          f"{'mean_lk_mean':>13} {'mean_lk_min':>12} {'min_lk_mean':>12} {'min_lk_min':>11}")
    print("-" * 90)
    for kind in sorted(edf["kind"].unique()):
        sub = edf[edf["kind"] == kind]
        if not len(sub): continue
        print(f"{kind:<28} {len(sub):>5}  "
              f"{sub['paw_mean_lk_mean'].median():>13.3f} "
              f"{sub['paw_mean_lk_min'].median():>12.3f} "
              f"{sub['paw_min_lk_mean'].median():>12.3f} "
              f"{sub['paw_min_lk_min'].median():>11.3f}")
    print()

    print("BOUNDARY FRAMES (leading vs trailing first/last 3 frames of union extent)")
    print(f"{'kind':<28} {'n':>5}  "
          f"{'lead_mean_lk':>13} {'trail_mean_lk':>14} {'lead_min_lk':>12} {'trail_min_lk':>13}")
    print("-" * 95)
    for kind in sorted(edf["kind"].unique()):
        sub = edf[edf["kind"] == kind]
        if not len(sub): continue
        print(f"{kind:<28} {len(sub):>5}  "
              f"{sub['leading_mean_lk'].median():>13.3f} "
              f"{sub['trailing_mean_lk'].median():>14.3f} "
              f"{sub['leading_min_lk'].median():>12.3f} "
              f"{sub['trailing_min_lk'].median():>13.3f}")
    print()

    # ===== MERGED-component-specific analysis =====
    print("=" * 70)
    print("MERGED-ALGO-SPAN inter-GT-gap analysis")
    print("=" * 70)
    print()
    rows_by_video = defaultdict(list)
    for r in results:
        rows_by_video[r["video_id"]].append(r)
    merged_components = find_merged_components(rows_by_video)
    total_merged = sum(len(m) for m in merged_components.values())
    print(f"MERGED components found: {total_merged}")
    print()
    merged_rows = []
    for vid, comps in merged_components.items():
        if not comps: continue
        for c in comps:
            algo_start, algo_end = c["algo_start"], c["algo_end"]
            gts = c["gt_list"]
            mean_arr = mean_lk_by_vid[vid]
            min_arr = min_lk_by_vid[vid]
            # GT-overlap frames (any frame that's within at least one GT)
            gt_frames = set()
            for g in gts:
                for f in range(g["start"], g["end"] + 1):
                    if algo_start <= f <= algo_end:
                        gt_frames.add(f)
            # Inter-GT frames (within algo span but NOT in any GT)
            inter_gt_frames = [f for f in range(algo_start, algo_end + 1)
                               if f not in gt_frames]
            if not inter_gt_frames:
                continue  # no inter-GT gap (gap=0 case)
            # Stats
            gt_mean = float(np.nanmean([mean_arr[f] for f in gt_frames]))
            gt_min_lk_mean = float(np.nanmean([min_arr[f] for f in gt_frames]))
            inter_mean = float(np.nanmean([mean_arr[f] for f in inter_gt_frames]))
            inter_min_lk_mean = float(np.nanmean([min_arr[f] for f in inter_gt_frames]))
            delta_mean = inter_mean - gt_mean
            delta_min = inter_min_lk_mean - gt_min_lk_mean
            merged_rows.append({
                "video": vid,
                "algo_start": algo_start,
                "algo_end": algo_end,
                "n_gt": len(gts),
                "n_inter_gt_frames": len(inter_gt_frames),
                "gt_paw_mean_lk_mean": gt_mean,
                "inter_paw_mean_lk_mean": inter_mean,
                "delta_paw_mean_lk": delta_mean,
                "gt_paw_min_lk_mean": gt_min_lk_mean,
                "inter_paw_min_lk_mean": inter_min_lk_mean,
                "delta_paw_min_lk": delta_min,
            })

    mdf = pd.DataFrame(merged_rows)
    out_mcsv = OUT_DIR / "metrics" / "merged_inter_gt_paw_lk.csv"
    mdf.to_csv(out_mcsv, index=False)
    print(f"MERGED components with gap (n_inter_gt > 0): {len(mdf)}")
    if len(mdf):
        print()
        print(f"  Median GT-frame    paw_mean_lk: {mdf['gt_paw_mean_lk_mean'].median():.3f}")
        print(f"  Median inter-GT    paw_mean_lk: {mdf['inter_paw_mean_lk_mean'].median():.3f}")
        print(f"  Median delta (inter - gt):       {mdf['delta_paw_mean_lk'].median():+.3f}")
        print()
        print(f"  Median GT-frame    paw_min_lk:  {mdf['gt_paw_min_lk_mean'].median():.3f}")
        print(f"  Median inter-GT    paw_min_lk:  {mdf['inter_paw_min_lk_mean'].median():.3f}")
        print(f"  Median delta (inter - gt):       {mdf['delta_paw_min_lk'].median():+.3f}")
    print(f"  saved: {out_mcsv}")
    print()

    # ===== Figures =====
    print("Generating figures...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Group kinds for visualization
    plot_kinds = []
    for k in ["TP_clean", "tolerance_FP_start_early", "tolerance_FP_span_over",
              "tolerance_FP_span_short", "tolerance_FP_start_late",
              "tolerance_FP_merged_algo", "true_phantom_FP",
              "true_miss_FN", "tolerance_FN_paired_with_FP", "tolerance_FN_fragmented_gt"]:
        if k in edf["kind"].unique():
            plot_kinds.append(k)

    colors = {"TP_clean": "C0",
              "tolerance_FP_start_early": "C3",
              "tolerance_FP_span_over": "C4",
              "tolerance_FP_span_short": "C5",
              "tolerance_FP_start_late": "C6",
              "tolerance_FP_merged_algo": "C7",
              "true_phantom_FP": "C1",
              "true_miss_FN": "C8",
              "tolerance_FN_paired_with_FP": "C2",
              "tolerance_FN_fragmented_gt": "C9"}

    bins = np.linspace(0, 1.05, 36)

    # Panel 1: paw_mean_lk_mean over union extent
    ax = axes[0, 0]
    for k in plot_kinds:
        vals = edf[edf["kind"] == k]["paw_mean_lk_mean"].values
        if len(vals) == 0: continue
        ax.hist(vals, bins=bins, alpha=0.4, color=colors.get(k, "gray"),
                density=True,
                label=f"{k} (n={len(vals)}, med={np.median(vals):.2f})")
    ax.axvline(0.5, color="0.5", lw=0.5, ls=":")
    ax.axvline(0.8, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("paw_mean_lk (mean over union extent)")
    ax.set_ylabel("density")
    ax.set_title("A: Mean of 4 hand-keypoint likelihoods, aggregated over union extent")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: paw_min_lk_mean over union extent
    ax = axes[0, 1]
    for k in plot_kinds:
        vals = edf[edf["kind"] == k]["paw_min_lk_mean"].values
        if len(vals) == 0: continue
        ax.hist(vals, bins=bins, alpha=0.4, color=colors.get(k, "gray"),
                density=True,
                label=f"{k} (n={len(vals)}, med={np.median(vals):.2f})")
    ax.axvline(0.5, color="0.5", lw=0.5, ls=":")
    ax.axvline(0.8, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("paw_min_lk (mean of per-frame min over union extent)")
    ax.set_ylabel("density")
    ax.set_title("B: Min of 4 hand-keypoint likelihoods, aggregated over union extent")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: leading-frame paw_mean_lk (boundary-specific)
    ax = axes[1, 0]
    for k in plot_kinds:
        vals = edf[edf["kind"] == k]["leading_mean_lk"].values
        if len(vals) == 0: continue
        ax.hist(vals, bins=bins, alpha=0.4, color=colors.get(k, "gray"),
                density=True,
                label=f"{k} (n={len(vals)}, med={np.median(vals):.2f})")
    ax.axvline(0.5, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("leading paw_mean_lk (first 3 frames of union extent)")
    ax.set_ylabel("density")
    ax.set_title("A: Leading-frame mean -- highlights early-start FPs")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: MERGED inter-GT vs GT-overlap paw_mean_lk delta
    ax = axes[1, 1]
    if len(mdf):
        ax.hist(mdf["delta_paw_mean_lk"].values, bins=np.linspace(-0.5, 0.5, 26),
                alpha=0.6, color="C3", density=False,
                label=f"MERGED delta (inter - gt)\n(n={len(mdf)}, med={mdf['delta_paw_mean_lk'].median():+.3f})")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel("paw_mean_lk delta (inter-GT gap minus GT-overlap frames)")
        ax.set_ylabel("count")
        ax.set_title("MERGED algo spans: is inter-GT gap less confident?")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No MERGED gap data", transform=ax.transAxes, ha="center")

    fig.tight_layout()
    out_fig = OUT_DIR / "figures" / "paw_lk_diagnostic.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {out_fig}")

    # Final summary JSON
    summary = {
        "n_events": len(edf),
        "n_merged_components_with_gap": len(mdf),
        "per_kind_medians": {
            kind: {
                "n": int(len(edf[edf["kind"] == kind])),
                "paw_mean_lk_mean_median": float(edf[edf["kind"] == kind]["paw_mean_lk_mean"].median()),
                "paw_min_lk_mean_median": float(edf[edf["kind"] == kind]["paw_min_lk_mean"].median()),
                "leading_mean_lk_median": float(edf[edf["kind"] == kind]["leading_mean_lk"].median()),
                "trailing_mean_lk_median": float(edf[edf["kind"] == kind]["trailing_mean_lk"].median()),
            }
            for kind in sorted(edf["kind"].unique())
        },
        "merged_summary": {
            "n_components_with_gap": len(mdf),
            "gt_paw_mean_lk_median": float(mdf["gt_paw_mean_lk_mean"].median()) if len(mdf) else None,
            "inter_paw_mean_lk_median": float(mdf["inter_paw_mean_lk_mean"].median()) if len(mdf) else None,
            "delta_paw_mean_lk_median": float(mdf["delta_paw_mean_lk"].median()) if len(mdf) else None,
        }
    }
    (OUT_DIR / "metrics" / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved summary: {OUT_DIR / 'metrics' / 'summary.json'}")


if __name__ == "__main__":
    main()
