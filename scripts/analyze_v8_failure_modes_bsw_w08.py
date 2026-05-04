"""
Diagnostic analysis of v8 failure modes on the BSW w=0.8 cumulative
best snapshot.

Same categorization logic as analyze_v8_failure_modes.py (which ran on
the pure-baseline failure_mode_breakdown snapshot). The reason for a
new versioned script (per feedback_file_editing_rules.md) is to point
at the BSW w=0.8 snapshot's loocv_aggregate.json instead of the pure
baseline's, and to write the diagnostic outputs into a new snapshot
dir alongside the BSW w=0.8 snapshot.

This is purely informational. No algorithm change. No accept/reject
decision attached to the output.

Pre-experiment checklist (walked at design time):
1. Cumulative-stacking check: diagnostic ON cumulative best (BSW w=0.8
   verified ACTIVE 2026-05-04 in v8_pending_integrations.md). Input
   is the cumulative best snapshot's loocv_aggregate.json.
2. Existing-code-modification check: NO. New script.
3. Assumption check: same NEAR_RANGE=10, RANDOM_THRESHOLD=30 as
   pure-baseline diagnostic.
4. FN-direction-reporting check: diagnostic outputs distribution, not
   delta. Will compare against pure-baseline failure-mode breakdown
   for context.
5. Framework-not-adhoc: output to canonical snapshot dir layout.
6. Branch + tag check: feature/v8-failure-mode-breakdown-on-bsw-w08;
   v8-pre-fmb-bsw-w08-2026-05-04
7. Decision-rule check: N/A diagnostic.

Outputs:
  v8.0.0_dev_failure_mode_breakdown_on_bsw_w08/
    metrics/fn_breakdown.json
    metrics/fp_breakdown.json
    metrics/boundary_error_tail.csv
    figures/failure_mode_summary.png
    RESULTS.md
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_failure_mode_breakdown_on_bsw_w08"
)
SOURCE_AGG_PATH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_boundary_sample_weight_b1_w0.8\metrics\loocv_aggregate.json"
)

# Pure-baseline diagnostic for comparison (read-only)
PURE_BASELINE_DIAGNOSTIC = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_failure_mode_breakdown\metrics"
)

START_TOL = 2
NEAR_RANGE = 10
WIDE_RANGE = 50
RANDOM_THRESHOLD = 30


def categorize_fn(fn_entry, video_algo_reaches):
    gt_start = fn_entry["gt_start_frame"]
    gt_end = fn_entry["gt_end_frame"]
    gt_span = gt_end - gt_start + 1
    if not video_algo_reaches:
        return "model_miss"
    distances = [
        (abs(a_start - gt_start), a_start, a_end)
        for (a_start, a_end, _status) in video_algo_reaches
    ]
    distances.sort()
    nearest_dist, near_a_start, near_a_end = distances[0]
    near_a_span = near_a_end - near_a_start + 1
    if nearest_dist > WIDE_RANGE:
        return "model_miss"
    start_ok = nearest_dist <= START_TOL
    span_ratio = near_a_span / gt_span if gt_span > 0 else 0
    span_ok = abs(near_a_span - gt_span) <= max(0.5 * gt_span, 5)
    if span_ratio < 0.5:
        return "fragmented"
    if not start_ok and not span_ok:
        return "tol_miss_both"
    if not start_ok:
        return "tol_miss_start"
    if not span_ok:
        return "tol_miss_span"
    return "matched_within_tol_but_unmatched"


def categorize_fp(fp_entry, video_gt_reaches):
    fp_start = fp_entry["algo_start_frame"]
    fp_end = fp_entry["algo_end_frame"]
    if not video_gt_reaches:
        return "random"
    distances = [
        (abs(g_start - fp_start), g_start, g_end, gstatus)
        for (g_start, g_end, gstatus) in video_gt_reaches
    ]
    distances.sort()
    nearest_dist, g_start, g_end, gstatus = distances[0]
    overlap = (fp_start <= g_end) and (fp_end >= g_start)
    if overlap:
        return "within_gt"
    if nearest_dist <= NEAR_RANGE:
        if gstatus == "tp":
            return "split_twin"
        else:
            return "near_unmatched_gt"
    if fp_end < g_start and (g_start - fp_end) <= NEAR_RANGE:
        return "pre_reach"
    if fp_start > g_end and (fp_start - g_end) <= NEAR_RANGE:
        return "post_reach"
    if nearest_dist > RANDOM_THRESHOLD:
        return "random"
    return "other"


def main():
    print("=" * 70)
    print("v8 FAILURE-MODE DIAGNOSTIC ON BSW W=0.8 CUMULATIVE BEST")
    print("=" * 70)
    print()

    print(f"Loading {SOURCE_AGG_PATH} ...")
    data = json.loads(SOURCE_AGG_PATH.read_text(encoding="utf-8"))
    raw = data["raw_results"]
    print(f"  {len(raw)} events: TP={data['summary']['n_tp']} "
          f"FP={data['summary']['n_fp']} FN={data['summary']['n_fn']}")
    print()

    by_video_algo = defaultdict(list)
    by_video_gt = defaultdict(list)
    for r in raw:
        vid = r["video_id"]
        if r["status"] in ("tp", "fp"):
            by_video_algo[vid].append((
                r["algo_start_frame"], r["algo_end_frame"], r["status"]))
        if r["status"] in ("tp", "fn"):
            by_video_gt[vid].append((
                r["gt_start_frame"], r["gt_end_frame"], r["status"]))

    fn_categories = Counter()
    fn_per_video = defaultdict(Counter)
    for r in raw:
        if r["status"] != "fn":
            continue
        cat = categorize_fn(r, by_video_algo[r["video_id"]])
        fn_categories[cat] += 1
        fn_per_video[r["video_id"]][cat] += 1

    fp_categories = Counter()
    fp_per_video = defaultdict(Counter)
    for r in raw:
        if r["status"] != "fp":
            continue
        cat = categorize_fp(r, by_video_gt[r["video_id"]])
        fp_categories[cat] += 1
        fp_per_video[r["video_id"]][cat] += 1

    tp_rows = []
    for r in raw:
        if r["status"] != "tp":
            continue
        sd = abs(r["start_delta"])
        spd = abs(r["span_delta"])
        worst = max(sd, spd)
        if worst > 0:
            tp_rows.append({
                "video_id": r["video_id"],
                "gt_start_frame": r["gt_start_frame"],
                "gt_end_frame": r["gt_end_frame"],
                "algo_start_frame": r["algo_start_frame"],
                "algo_end_frame": r["algo_end_frame"],
                "start_delta": r["start_delta"],
                "span_delta": r["span_delta"],
                "max_abs_error": worst,
            })
    tp_rows.sort(key=lambda x: x["max_abs_error"], reverse=True)

    n_tps = sum(1 for r in raw if r["status"] == "tp")
    n_perfect_tps = n_tps - len(tp_rows)
    print(f"Boundary errors: {n_perfect_tps}/{n_tps} TPs perfect; "
          f"{len(tp_rows)} have non-zero error.")
    print()

    print("FN breakdown (n=%d):" % sum(fn_categories.values()))
    for cat, n in fn_categories.most_common():
        pct = 100 * n / sum(fn_categories.values())
        print(f"  {cat:30s} {n:>4d}  ({pct:5.1f}%)")
    print()
    print("FP breakdown (n=%d):" % sum(fp_categories.values()))
    for cat, n in fp_categories.most_common():
        pct = 100 * n / sum(fp_categories.values())
        print(f"  {cat:30s} {n:>4d}  ({pct:5.1f}%)")
    print()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    figures_dir = SNAPSHOT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    (metrics_dir / "fn_breakdown.json").write_text(
        json.dumps({
            "n_fn_total": sum(fn_categories.values()),
            "categories": dict(fn_categories),
            "categories_pct": {k: round(100*v/sum(fn_categories.values()), 2)
                               for k, v in fn_categories.items()},
            "per_video": {vid: dict(c) for vid, c in fn_per_video.items()},
        }, indent=2),
        encoding="utf-8",
    )
    (metrics_dir / "fp_breakdown.json").write_text(
        json.dumps({
            "n_fp_total": sum(fp_categories.values()),
            "categories": dict(fp_categories),
            "categories_pct": {k: round(100*v/sum(fp_categories.values()), 2)
                               for k, v in fp_categories.items()},
            "per_video": {vid: dict(c) for vid, c in fp_per_video.items()},
        }, indent=2),
        encoding="utf-8",
    )
    csv_path = metrics_dir / "boundary_error_tail.csv"
    if tp_rows:
        keys = list(tp_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(tp_rows)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("video_id,gt_start_frame,gt_end_frame,algo_start_frame,"
                    "algo_end_frame,start_delta,span_delta,max_abs_error\n")
    print(f"Wrote: {metrics_dir / 'fn_breakdown.json'}")
    print(f"Wrote: {metrics_dir / 'fp_breakdown.json'}")
    print(f"Wrote: {csv_path}  ({len(tp_rows)} non-perfect TPs)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    ax = axes[0, 0]
    cats = list(fn_categories.most_common())
    if cats:
        names, counts = zip(*cats)
        ax.barh(range(len(names)), counts, color="#D32F2F")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        for i, c in enumerate(counts):
            ax.text(c + 1, i, f"{c}", va="center", fontsize=9)
    ax.set_title(f"FN sources (n={sum(fn_categories.values())})", fontweight="bold")
    ax.set_xlabel("count")

    ax = axes[0, 1]
    cats = list(fp_categories.most_common())
    if cats:
        names, counts = zip(*cats)
        ax.barh(range(len(names)), counts, color="#F57C00")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        for i, c in enumerate(counts):
            ax.text(c + 1, i, f"{c}", va="center", fontsize=9)
    ax.set_title(f"FP sources (n={sum(fp_categories.values())})", fontweight="bold")
    ax.set_xlabel("count")

    ax = axes[1, 0]
    sd_abs = [abs(r["start_delta"]) for r in raw if r["status"] == "tp"]
    if sd_abs:
        max_sd = max(sd_abs)
        bins = list(range(max_sd + 2))
        ax.hist(sd_abs, bins=bins, color="#1976D2", edgecolor="black", alpha=0.85)
        for v in (0, 1, 2):
            n = sd_abs.count(v)
            pct = 100 * n / len(sd_abs)
            ax.text(v + 0.4, ax.get_ylim()[1] * 0.95,
                    f"|d|={v}: {n}\n({pct:.1f}%)", fontsize=8, va="top")
    ax.set_title(f"|start_delta| histogram on TPs (n={n_tps})", fontweight="bold")
    ax.set_xlabel("|start_delta| (frames)")
    ax.set_ylabel("count of TPs")

    ax = axes[1, 1]
    spd_abs = [abs(r["span_delta"]) for r in raw if r["status"] == "tp"]
    if spd_abs:
        max_spd = max(spd_abs)
        bins = list(range(max_spd + 2))
        ax.hist(spd_abs, bins=bins, color="#388E3C", edgecolor="black", alpha=0.85)
        for v in (0, 1, 2):
            n = spd_abs.count(v)
            pct = 100 * n / len(spd_abs)
            ax.text(v + 0.4, ax.get_ylim()[1] * 0.95,
                    f"|d|={v}: {n}\n({pct:.1f}%)", fontsize=8, va="top")
    ax.set_title(f"|span_delta| histogram on TPs (n={n_tps})", fontweight="bold")
    ax.set_xlabel("|span_delta| (frames)")
    ax.set_ylabel("count of TPs")

    fig.suptitle("v8 failure-mode diagnostic on BSW w=0.8 cumulative best",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig_path = figures_dir / "failure_mode_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {fig_path}")
    print()
    print("DONE.")

    # ---- Comparison vs pure baseline diagnostic (for context) ----
    print()
    print("=" * 70)
    print("COMPARISON: cumulative-best (BSW w=0.8) vs pure-baseline diagnostic")
    print("=" * 70)
    pure_fn = json.loads((PURE_BASELINE_DIAGNOSTIC / "fn_breakdown.json").read_text(encoding="utf-8"))
    pure_fp = json.loads((PURE_BASELINE_DIAGNOSTIC / "fp_breakdown.json").read_text(encoding="utf-8"))

    print(f"FN total: pure_baseline={pure_fn['n_fn_total']}  cumul_best={sum(fn_categories.values())}  "
          f"(delta {sum(fn_categories.values()) - pure_fn['n_fn_total']:+d})")
    print()
    print(f"FN by category (pure_baseline -> cumul_best, delta):")
    all_fn_cats = sorted(set(list(pure_fn['categories'].keys()) + list(fn_categories.keys())))
    for cat in all_fn_cats:
        pure_v = pure_fn['categories'].get(cat, 0)
        cur_v = fn_categories.get(cat, 0)
        delta = cur_v - pure_v
        print(f"  {cat:30s} {pure_v:>4d} -> {cur_v:>4d}  ({delta:+d})")

    print()
    print(f"FP total: pure_baseline={pure_fp['n_fp_total']}  cumul_best={sum(fp_categories.values())}  "
          f"(delta {sum(fp_categories.values()) - pure_fp['n_fp_total']:+d})")
    print()
    print(f"FP by category (pure_baseline -> cumul_best, delta):")
    all_fp_cats = sorted(set(list(pure_fp['categories'].keys()) + list(fp_categories.keys())))
    for cat in all_fp_cats:
        pure_v = pure_fp['categories'].get(cat, 0)
        cur_v = fp_categories.get(cat, 0)
        delta = cur_v - pure_v
        print(f"  {cat:30s} {pure_v:>4d} -> {cur_v:>4d}  ({delta:+d})")


if __name__ == "__main__":
    main()
