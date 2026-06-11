"""Diagnostic: paw-likelihood profile per TOPOLOGY COMPONENT.

Replaces diagnose_v8_paw_likelihood_at_events.py which used row-level
classification (legacy FP/FN view). This version operates at the topology
component level per error_taxonomy_definitions_2026-05-19.md:

  Each connected component of the (algo, GT) overlap graph is ONE event.
  Components are labeled by (n_algo, n_gt) shape:
    TP                 : 1 algo + 1 GT, strict tolerance passes
    TOLERANCE_ERROR    : 1 algo + 1 GT, overlap exists, tolerance fails
                         sub: start_early / start_late / span_over / span_short
    MERGED             : 1 algo + 2+ GT
    FRAGMENTED         : 2+ algo + 1 GT
    FALSE_POSITIVE     : 1 algo + 0 GT (true phantom)
    FALSE_NEGATIVE     : 0 algo + 1 GT (true miss; INDEPENDENT of any FP)
    COMPLEX            : 2+ algo + 2+ GT (rare)

Paw likelihood signals computed from the 4 hand keypoints' per-frame
likelihoods:
  paw_mean_lk[t] = mean(RightHand_lk[t], RHLeft_lk[t], RHOut_lk[t], RHRight_lk[t])
  paw_min_lk[t]  = min(those four)

For each component:
  - Union extent = [min start of all participating algos+GTs, max end of all]
  - lk stats over the union extent
  - Leading / trailing boundary frame lk
  - For MERGED specifically: inter-GT gap lk vs GT-overlap lk
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
    r"\reach_detection\v8.0.1_dev_paw_lk_diagnostic_topology"
)
(OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

HAND_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
BOUNDARY_N_FRAMES = 3


def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def build_components(algos, gts):
    """Connected components of the algo-GT overlap graph.

    algos, gts: lists of (start, end) tuples.
    Returns list of dicts:
      {"topology": <label>, "sub": <subtype or None>,
       "algos": [(s,e),...], "gts": [(s,e),...]}
    """
    algo_to_gt = defaultdict(set)
    gt_to_algo = defaultdict(set)
    for i, (a_s, a_e) in enumerate(algos):
        for j, (g_s, g_e) in enumerate(gts):
            if overlap(a_s, a_e, g_s, g_e):
                algo_to_gt[i].add(j)
                gt_to_algo[j].add(i)

    visited_a, visited_g = set(), set()
    comps = []

    for i in range(len(algos)):
        if i in visited_a:
            continue
        if not algo_to_gt[i]:
            # Isolated algo = FALSE_POSITIVE
            comps.append({
                "topology": "FALSE_POSITIVE", "sub": None,
                "algos": [algos[i]], "gts": []})
            visited_a.add(i)
            continue
        # BFS to find connected component
        algo_in, gt_in = set(), set()
        queue = [("a", i)]
        while queue:
            kind, idx = queue.pop()
            if kind == "a":
                if idx in algo_in:
                    continue
                algo_in.add(idx)
                for gj in algo_to_gt[idx]:
                    queue.append(("g", gj))
            else:
                if idx in gt_in:
                    continue
                gt_in.add(idx)
                for ai in gt_to_algo[idx]:
                    queue.append(("a", ai))
        visited_a.update(algo_in)
        visited_g.update(gt_in)

        comp_algos = [algos[k] for k in algo_in]
        comp_gts = [gts[k] for k in gt_in]
        na, ng = len(comp_algos), len(comp_gts)

        if na == 1 and ng == 1:
            a_s, a_e = comp_algos[0]
            g_s, g_e = comp_gts[0]
            start_delta = a_s - g_s
            algo_span = a_e - a_s + 1
            gt_span = g_e - g_s + 1
            span_delta = algo_span - gt_span
            span_tol = max(0.5 * gt_span, 5)
            if abs(start_delta) <= 2 and abs(span_delta) <= span_tol:
                comps.append({"topology": "TP", "sub": None,
                              "algos": comp_algos, "gts": comp_gts})
            else:
                # Sub-classify the TOLERANCE_ERROR
                if start_delta < -2:
                    sub = "start_early"
                elif start_delta > 2:
                    sub = "start_late"
                elif span_delta > span_tol:
                    sub = "span_over"
                elif span_delta < -span_tol:
                    sub = "span_short"
                else:
                    sub = "unclassified"
                comps.append({"topology": "TOLERANCE_ERROR", "sub": sub,
                              "algos": comp_algos, "gts": comp_gts})
        elif na == 1 and ng >= 2:
            comps.append({"topology": "MERGED", "sub": f"{ng}_gt",
                          "algos": comp_algos, "gts": comp_gts})
        elif na >= 2 and ng == 1:
            comps.append({"topology": "FRAGMENTED", "sub": f"{na}_algo",
                          "algos": comp_algos, "gts": comp_gts})
        elif na >= 2 and ng >= 2:
            comps.append({"topology": "COMPLEX", "sub": f"{na}_algo_{ng}_gt",
                          "algos": comp_algos, "gts": comp_gts})

    # Isolated GTs = FALSE_NEGATIVE
    for j in range(len(gts)):
        if j not in visited_g:
            comps.append({"topology": "FALSE_NEGATIVE", "sub": None,
                          "algos": [], "gts": [gts[j]]})

    return comps


def union_extent(comp):
    starts = [a[0] for a in comp["algos"]] + [g[0] for g in comp["gts"]]
    ends = [a[1] for a in comp["algos"]] + [g[1] for g in comp["gts"]]
    return min(starts), max(ends)


def main():
    print("=" * 70)
    print("PAW LIKELIHOOD DIAGNOSTIC -- TOPOLOGY COMPONENTS")
    print("=" * 70)
    print()

    print("Loading LOOCV aggregate...", flush=True)
    loocv = json.loads(LOOCV_PATH.read_text(encoding="utf-8"))
    results = loocv["raw_results"]
    print(f"  {len(results)} row-events across {len(set(r['video_id'] for r in results))} videos")

    print("Loading parquet (paw lk columns only)...", flush=True)
    cols_needed = ["video_id", "frame"] + HAND_LK_COLS
    df = pd.read_parquet(PARQUET_PATH, columns=cols_needed)

    # Per-frame paw confidence signals
    lk_matrix = df[HAND_LK_COLS].to_numpy(dtype=np.float32)
    df["paw_mean_lk"] = lk_matrix.mean(axis=1)
    df["paw_min_lk"] = lk_matrix.min(axis=1)

    mean_lk_by_vid = {}
    min_lk_by_vid = {}
    for vid, grp in df.groupby("video_id", sort=False):
        grp_sorted = grp.sort_values("frame")
        mx = int(grp_sorted["frame"].max())
        mean_arr = np.full(mx + 1, np.nan, dtype=np.float32)
        min_arr = np.full(mx + 1, np.nan, dtype=np.float32)
        mean_arr[grp_sorted["frame"].to_numpy()] = grp_sorted["paw_mean_lk"].to_numpy()
        min_arr[grp_sorted["frame"].to_numpy()] = grp_sorted["paw_min_lk"].to_numpy()
        mean_lk_by_vid[vid] = mean_arr
        min_lk_by_vid[vid] = min_arr

    # ===== Build algo + GT catalogs per video, then components =====
    print("Building topology components per video...", flush=True)
    algos_by_video = defaultdict(set)
    gts_by_video = defaultdict(set)
    for r in results:
        vid = r["video_id"]
        if r["algo_start_frame"] >= 0:
            algos_by_video[vid].add((int(r["algo_start_frame"]),
                                     int(r["algo_end_frame"])))
        if r["gt_start_frame"] >= 0:
            gts_by_video[vid].add((int(r["gt_start_frame"]),
                                   int(r["gt_end_frame"])))

    component_rows = []
    for vid in mean_lk_by_vid:
        algos = sorted(algos_by_video[vid])
        gts = sorted(gts_by_video[vid])
        comps = build_components(algos, gts)
        for c in comps:
            win_lo, win_hi = union_extent(c)
            mean_arr = mean_lk_by_vid[vid][win_lo:win_hi + 1]
            min_arr = min_lk_by_vid[vid][win_lo:win_hi + 1]
            valid_mean = mean_arr[~np.isnan(mean_arr)]
            valid_min = min_arr[~np.isnan(min_arr)]
            if len(valid_mean) == 0:
                continue
            n = len(valid_mean)
            n_bnd = max(1, min(BOUNDARY_N_FRAMES, n // 2))
            # MERGED-specific: inter-GT gap frames vs GT-overlap frames
            gt_frames = set()
            for g in c["gts"]:
                for f in range(g[0], g[1] + 1):
                    if win_lo <= f <= win_hi:
                        gt_frames.add(f)
            algo_frames_set = set()
            for a in c["algos"]:
                for f in range(a[0], a[1] + 1):
                    if win_lo <= f <= win_hi:
                        algo_frames_set.add(f)
            # Inter-GT gap within an algo span (relevant for MERGED)
            inter_gt_in_algo = sorted(algo_frames_set - gt_frames)

            row = {
                "video": vid,
                "topology": c["topology"],
                "sub": c["sub"] or "",
                "n_algo": len(c["algos"]),
                "n_gt": len(c["gts"]),
                "extent_start": win_lo,
                "extent_end": win_hi,
                "extent_span": win_hi - win_lo + 1,
                "paw_mean_lk_mean": float(valid_mean.mean()),
                "paw_mean_lk_min": float(valid_mean.min()),
                "paw_mean_lk_frac_above_0.5": float((valid_mean > 0.5).mean()),
                "paw_mean_lk_frac_above_0.8": float((valid_mean > 0.8).mean()),
                "paw_min_lk_mean": float(valid_min.mean()),
                "leading_mean_lk": float(np.mean(valid_mean[:n_bnd])),
                "trailing_mean_lk": float(np.mean(valid_mean[-n_bnd:])),
                "leading_min_lk": float(np.mean(valid_min[:n_bnd])),
                "trailing_min_lk": float(np.mean(valid_min[-n_bnd:])),
            }
            # MERGED-specific stats
            if c["topology"] == "MERGED" and inter_gt_in_algo:
                inter_mean = float(np.nanmean(
                    [mean_lk_by_vid[vid][f] for f in inter_gt_in_algo]))
                inter_min = float(np.nanmean(
                    [min_lk_by_vid[vid][f] for f in inter_gt_in_algo]))
                gt_overlap_mean = float(np.nanmean(
                    [mean_lk_by_vid[vid][f] for f in gt_frames])) if gt_frames else None
                row["merged_n_inter_gt_frames"] = len(inter_gt_in_algo)
                row["merged_inter_paw_mean_lk"] = inter_mean
                row["merged_inter_paw_min_lk"] = inter_min
                row["merged_gt_overlap_paw_mean_lk"] = gt_overlap_mean
                if gt_overlap_mean is not None:
                    row["merged_delta_paw_mean_lk"] = inter_mean - gt_overlap_mean
            component_rows.append(row)

    cdf = pd.DataFrame(component_rows)
    out_csv = OUT_DIR / "metrics" / "per_component_paw_lk.csv"
    cdf.to_csv(out_csv, index=False)
    print(f"  {len(cdf)} components saved to {out_csv}")
    print()

    # ===== Topology summary =====
    print("=" * 70)
    print("TOPOLOGY-COMPONENT COUNTS + PAW LIKELIHOOD MEDIANS")
    print("=" * 70)
    print()
    print(f"{'topology':<18}{'sub':<16}{'n':>5}  "
          f"{'mean_lk':>9} {'leading':>9} {'trailing':>9}")
    print("-" * 80)

    # Define ordering: TP first, then TOLERANCE_ERROR subtypes, then MERGED, FRAGMENTED, FP, FN, COMPLEX
    topology_order = [
        ("TP", None),
        ("TOLERANCE_ERROR", "start_early"),
        ("TOLERANCE_ERROR", "start_late"),
        ("TOLERANCE_ERROR", "span_over"),
        ("TOLERANCE_ERROR", "span_short"),
        ("TOLERANCE_ERROR", "unclassified"),
        ("MERGED", None),
        ("FRAGMENTED", None),
        ("FALSE_POSITIVE", None),
        ("FALSE_NEGATIVE", None),
        ("COMPLEX", None),
    ]
    for topo, sub in topology_order:
        if sub is not None:
            sub_df = cdf[(cdf["topology"] == topo) & (cdf["sub"] == sub)]
        else:
            sub_df = cdf[cdf["topology"] == topo]
        if not len(sub_df):
            continue
        sub_str = sub if sub else ""
        print(f"{topo:<18}{sub_str:<16}{len(sub_df):>5}  "
              f"{sub_df['paw_mean_lk_mean'].median():>9.3f} "
              f"{sub_df['leading_mean_lk'].median():>9.3f} "
              f"{sub_df['trailing_mean_lk'].median():>9.3f}")
    print()

    # ===== MERGED inter-GT gap specifically =====
    print("MERGED inter-GT gap (n_inter_gt_frames > 0):")
    merged_with_gap = cdf[(cdf["topology"] == "MERGED") &
                          cdf["merged_n_inter_gt_frames"].notna() &
                          (cdf["merged_n_inter_gt_frames"] > 0)]
    if len(merged_with_gap):
        print(f"  n = {len(merged_with_gap)} of {(cdf['topology'] == 'MERGED').sum()} MERGED components")
        print(f"  Median GT-overlap paw_mean_lk: {merged_with_gap['merged_gt_overlap_paw_mean_lk'].median():.3f}")
        print(f"  Median inter-GT paw_mean_lk:    {merged_with_gap['merged_inter_paw_mean_lk'].median():.3f}")
        print(f"  Median delta (inter - gt):       {merged_with_gap['merged_delta_paw_mean_lk'].median():+.3f}")
    print()

    # ===== Figures =====
    print("Generating figures...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    bins = np.linspace(0, 1.05, 36)

    # Group rows by topology label string for plotting
    cdf["label"] = cdf.apply(
        lambda r: f"{r['topology']}({r['sub']})" if r["sub"] else r["topology"],
        axis=1)
    plot_labels = [
        ("TP", "C0"),
        ("TOLERANCE_ERROR(start_early)", "C3"),
        ("TOLERANCE_ERROR(start_late)", "C6"),
        ("TOLERANCE_ERROR(span_over)", "C4"),
        ("TOLERANCE_ERROR(span_short)", "C5"),
        ("MERGED", "C7"),
        ("FRAGMENTED", "C9"),
        ("FALSE_POSITIVE", "C1"),
        ("FALSE_NEGATIVE", "C2"),
    ]
    plot_labels = [(lbl, color) for lbl, color in plot_labels
                   if lbl in cdf["label"].unique()]

    # Panel 1: paw_mean_lk_mean over union extent
    ax = axes[0, 0]
    for lbl, color in plot_labels:
        vals = cdf[cdf["label"] == lbl]["paw_mean_lk_mean"].values
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.45, color=color, density=True,
                label=f"{lbl} (n={len(vals)}, med={np.median(vals):.2f})")
    ax.axvline(0.5, color="0.5", lw=0.5, ls=":")
    ax.axvline(0.8, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("paw_mean_lk over union extent")
    ax.set_ylabel("density")
    ax.set_title("A: paw_mean_lk aggregated over component extent")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: leading_mean_lk
    ax = axes[0, 1]
    for lbl, color in plot_labels:
        vals = cdf[cdf["label"] == lbl]["leading_mean_lk"].values
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.45, color=color, density=True,
                label=f"{lbl} (n={len(vals)}, med={np.median(vals):.2f})")
    ax.axvline(0.65, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("leading paw_mean_lk (first 3 frames of extent)")
    ax.set_ylabel("density")
    ax.set_title("Leading-frame paw_mean_lk")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: trailing_mean_lk
    ax = axes[1, 0]
    for lbl, color in plot_labels:
        vals = cdf[cdf["label"] == lbl]["trailing_mean_lk"].values
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.45, color=color, density=True,
                label=f"{lbl} (n={len(vals)}, med={np.median(vals):.2f})")
    ax.axvline(0.65, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("trailing paw_mean_lk (last 3 frames of extent)")
    ax.set_ylabel("density")
    ax.set_title("Trailing-frame paw_mean_lk")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: MERGED inter-GT delta
    ax = axes[1, 1]
    if len(merged_with_gap):
        ax.hist(merged_with_gap["merged_delta_paw_mean_lk"].values,
                bins=np.linspace(-0.8, 0.3, 23),
                alpha=0.6, color="C7",
                label=f"MERGED delta\n(n={len(merged_with_gap)}, med={merged_with_gap['merged_delta_paw_mean_lk'].median():+.3f})")
        ax.axvline(0, color="0.5", lw=0.7)
        ax.set_xlabel("paw_mean_lk(inter-GT gap) minus paw_mean_lk(GT-overlap)")
        ax.set_ylabel("count")
        ax.set_title("MERGED inter-GT gap: confidence drop relative to GT frames")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig = OUT_DIR / "figures" / "paw_lk_topology_diagnostic.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {out_fig}")


if __name__ == "__main__":
    main()
