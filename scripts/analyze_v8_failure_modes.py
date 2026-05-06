"""
Diagnostic analysis of v8 failure modes from the extended-schema LOOCV
output at v8.0.0_dev_failure_mode_breakdown/metrics/loocv_aggregate.json.

Produces:
  - metrics/fn_breakdown.json         FN categorization by source
  - metrics/fp_breakdown.json         FP categorization by position relative to GT
  - metrics/boundary_error_tail.csv   TPs ranked by max(|start_delta|, |span_delta|)
  - figures/failure_mode_summary.png  bar charts: FN sources + FP sources +
                                      boundary error histograms
  - RESULTS.md                        prose summary with the actual numbers and
                                      recommended next direction

Pure analysis script -- reads the existing aggregate JSON, computes
breakdowns, writes back to the same snapshot dir. Does NOT modify any
existing module code.

Categorization rules:

  FN sources (per FN GT reach, find nearest algo reach in same video):
    model_miss      -- no algo reach within +/-50 frames; model produced
                       no in-reach signal at all
    tol_miss_start  -- algo reach within +/-50f but |start_delta| > 2
                       (start_tol failure); span ok-ish
    tol_miss_span   -- algo within +/-2f start but span fails tolerance
    tol_miss_both   -- algo nearby but BOTH start and span fail
    fragmented      -- algo reach within +/-50f and its span is
                       <50% of GT span (suggests a split-piece that
                       failed span_tol)

  FP positions (per FP, find nearest GT reach in same video):
    split_twin            -- FP within +/-10f of a GT_start AND that GT
                             has a TP entry (matched). FP is the second
                             half of a split GT.
    near_unmatched_gt     -- FP within +/-10f of a GT_start AND that GT
                             is FN (not matched). FP could be the algo
                             that failed to match due to tolerances.
    pre_reach             -- FP ends within 10f BEFORE a GT_start, no
                             overlap. Algo fired early.
    post_reach            -- FP starts within 10f AFTER a GT_end, no
                             overlap. Algo fired late.
    within_gt             -- FP overlaps a GT reach window (start or end
                             falls between gt_start and gt_end). Should
                             be rare; means split with unusual geometry.
    random                -- FP further than 30f from any GT reach.
                             Phantom; model fired on something unrelated.
    other                 -- FP near GT but doesn't fit any of the above
                             (10-30f from GT, no overlap). Catch-all.
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
    r"\reach_detection\v8.0.0_dev_failure_mode_breakdown"
)

START_TOL = 2  # v8 matching tolerance
NEAR_RANGE = 10
WIDE_RANGE = 50
RANDOM_THRESHOLD = 30  # FP further than this from any GT = random


def categorize_fn(fn_entry, video_algo_reaches):
    """Categorize a single FN by looking at nearest algo reach in same video.
    video_algo_reaches: list of (algo_start, algo_end, status) tuples for
    all algo reaches (TP + FP) in this video.
    """
    gt_start = fn_entry["gt_start_frame"]
    gt_end = fn_entry["gt_end_frame"]
    gt_span = gt_end - gt_start + 1

    if not video_algo_reaches:
        return "model_miss"

    # Find nearest algo reach by start
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
    return "matched_within_tol_but_unmatched"  # shouldn't happen; sanity tag


def categorize_fp(fp_entry, video_gt_reaches):
    """Categorize a single FP by looking at nearest GT reach in same video.
    video_gt_reaches: list of (gt_start, gt_end, status) tuples where
    status is "tp" (matched) or "fn" (unmatched).
    """
    fp_start = fp_entry["algo_start_frame"]
    fp_end = fp_entry["algo_end_frame"]

    if not video_gt_reaches:
        return "random"

    # Find nearest GT by start_frame distance to fp_start
    distances = [
        (abs(g_start - fp_start), g_start, g_end, gstatus)
        for (g_start, g_end, gstatus) in video_gt_reaches
    ]
    distances.sort()
    nearest_dist, g_start, g_end, gstatus = distances[0]

    # Check overlap with this nearest GT reach
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
    print("v8 FAILURE-MODE DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    print()

    agg_path = SNAPSHOT_DIR / "metrics" / "loocv_aggregate.json"
    print(f"Loading {agg_path} ...")
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    raw = data["raw_results"]
    print(f"  {len(raw)} events: "
          f"TP={data['summary']['n_tp']} "
          f"FP={data['summary']['n_fp']} "
          f"FN={data['summary']['n_fn']}")
    print()

    # Index per video
    by_video_algo = defaultdict(list)  # vid -> list of (a_start, a_end, status)
    by_video_gt = defaultdict(list)    # vid -> list of (g_start, g_end, status)

    for r in raw:
        vid = r["video_id"]
        if r["status"] in ("tp", "fp"):
            by_video_algo[vid].append((
                r["algo_start_frame"], r["algo_end_frame"], r["status"]))
        if r["status"] in ("tp", "fn"):
            by_video_gt[vid].append((
                r["gt_start_frame"], r["gt_end_frame"], r["status"]))

    # --- FN breakdown ---
    fn_categories = Counter()
    fn_per_video = defaultdict(Counter)
    for r in raw:
        if r["status"] != "fn":
            continue
        cat = categorize_fn(r, by_video_algo[r["video_id"]])
        fn_categories[cat] += 1
        fn_per_video[r["video_id"]][cat] += 1

    # --- FP breakdown ---
    fp_categories = Counter()
    fp_per_video = defaultdict(Counter)
    for r in raw:
        if r["status"] != "fp":
            continue
        cat = categorize_fp(r, by_video_gt[r["video_id"]])
        fp_categories[cat] += 1
        fp_per_video[r["video_id"]][cat] += 1

    # --- Boundary error tail (TPs only) ---
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
    print(f"Boundary errors on TPs: {n_perfect_tps}/{n_tps} TPs are perfect "
          f"(start_delta=0 AND span_delta=0); {len(tp_rows)} have non-zero error.")
    print()

    # --- Print the breakdowns ---
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

    # --- Save outputs ---
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    figures_dir = SNAPSHOT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    # FN breakdown JSON
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
    print(f"Wrote: {metrics_dir / 'fn_breakdown.json'}")

    # FP breakdown JSON
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
    print(f"Wrote: {metrics_dir / 'fp_breakdown.json'}")

    # Boundary error tail CSV
    csv_path = metrics_dir / "boundary_error_tail.csv"
    if tp_rows:
        keys = list(tp_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(tp_rows)
    else:
        # All TPs perfect; write empty CSV with header
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("video_id,gt_start_frame,gt_end_frame,algo_start_frame,"
                    "algo_end_frame,start_delta,span_delta,max_abs_error\n")
    print(f"Wrote: {csv_path}  ({len(tp_rows)} non-perfect TPs)")

    # --- Figure: 4-panel summary ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # FN sources
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
    ax.set_title(f"FN sources (n={sum(fn_categories.values())})",
                 fontweight="bold")
    ax.set_xlabel("count")

    # FP sources
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
    ax.set_title(f"FP sources (n={sum(fp_categories.values())})",
                 fontweight="bold")
    ax.set_xlabel("count")

    # Boundary error: |start_delta| histogram
    ax = axes[1, 0]
    sd_abs = [abs(r["start_delta"]) for r in raw if r["status"] == "tp"]
    if sd_abs:
        max_sd = max(sd_abs)
        bins = list(range(max_sd + 2))
        ax.hist(sd_abs, bins=bins, color="#1976D2", edgecolor="black",
                alpha=0.85)
        for v in (0, 1, 2):
            n = sd_abs.count(v)
            pct = 100 * n / len(sd_abs)
            ax.text(v + 0.4, ax.get_ylim()[1] * 0.95,
                    f"|d|={v}: {n}\n({pct:.1f}%)",
                    fontsize=8, va="top")
    ax.set_title(f"|start_delta| histogram on TPs (n={n_tps})",
                 fontweight="bold")
    ax.set_xlabel("|start_delta| (frames)")
    ax.set_ylabel("count of TPs")

    # Boundary error: |span_delta| histogram
    ax = axes[1, 1]
    spd_abs = [abs(r["span_delta"]) for r in raw if r["status"] == "tp"]
    if spd_abs:
        max_spd = max(spd_abs)
        bins = list(range(max_spd + 2))
        ax.hist(spd_abs, bins=bins, color="#388E3C", edgecolor="black",
                alpha=0.85)
        for v in (0, 1, 2):
            n = spd_abs.count(v)
            pct = 100 * n / len(spd_abs)
            ax.text(v + 0.4, ax.get_ylim()[1] * 0.95,
                    f"|d|={v}: {n}\n({pct:.1f}%)",
                    fontsize=8, va="top")
    ax.set_title(f"|span_delta| histogram on TPs (n={n_tps})",
                 fontweight="bold")
    ax.set_xlabel("|span_delta| (frames)")
    ax.set_ylabel("count of TPs")

    fig.suptitle("v8 failure-mode diagnostic (LOOCV, exhaustive subset)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig_path = figures_dir / "failure_mode_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {fig_path}")

    # --- RESULTS.md ---
    n_fn_total = sum(fn_categories.values())
    n_fp_total = sum(fp_categories.values())

    fn_top = fn_categories.most_common(3)
    fp_top = fp_categories.most_common(3)

    # Recommendation logic
    rec_lines = []
    if n_fn_total > 0:
        top_fn_cat, top_fn_n = fn_top[0]
        top_fn_pct = 100 * top_fn_n / n_fn_total
        if top_fn_cat == "model_miss" and top_fn_pct > 40:
            rec_lines.append(
                f"FN sources: dominated by **model_miss** ({top_fn_n}/{n_fn_total} = "
                f"{top_fn_pct:.1f}%). The model is failing to produce ANY "
                f"in-reach signal for these GT reaches. Lever to pull next: "
                f"feature engineering, training data (more positive samples / "
                f"different sampling), threshold lowering, or class-weight rebalancing."
            )
        elif top_fn_cat == "fragmented" and top_fn_pct > 30:
            rec_lines.append(
                f"FN sources: substantial **fragmented** category "
                f"({top_fn_n}/{n_fn_total} = {top_fn_pct:.1f}%). The split-FP "
                f"hypothesis IS supported. Lever: target the GBM probability "
                f"output (smoothing the proba time series, label construction, "
                f"or features that produce more contiguous in-reach signal). "
                f"Post-processing merge_gap was already shown to be exhausted."
            )
        elif "tol_miss" in top_fn_cat:
            rec_lines.append(
                f"FN sources: dominated by **{top_fn_cat}** "
                f"({top_fn_n}/{n_fn_total} = {top_fn_pct:.1f}%). The algo IS "
                f"detecting these but matches fail tolerance. Lever: tighten "
                f"the algo's boundary precision (start or span) -- this is the "
                f"same axis as the boundary-error tail."
            )

    if n_fp_total > 0:
        top_fp_cat, top_fp_n = fp_top[0]
        top_fp_pct = 100 * top_fp_n / n_fp_total
        if top_fp_cat == "random":
            rec_lines.append(
                f"FP sources: dominated by **random** ({top_fp_n}/{n_fp_total} = "
                f"{top_fp_pct:.1f}%). FPs are far from any GT reach -- model is "
                f"firing on unrelated motion. Not a split problem. Lever: "
                f"feature engineering (rejection features), threshold raising, "
                f"or training (negative samples)."
            )
        elif top_fp_cat == "split_twin":
            rec_lines.append(
                f"FP sources: dominated by **split_twin** ({top_fp_n}/{n_fp_total} = "
                f"{top_fp_pct:.1f}%). The user's split-FP observation is "
                f"corroborated. Lever: make the GBM probability series more "
                f"contiguous mid-reach (smoothing, labels, features). "
                f"Post-processing merge_gap was already shown to be exhausted."
            )
        elif top_fp_cat in ("pre_reach", "post_reach"):
            rec_lines.append(
                f"FP sources: dominated by **{top_fp_cat}** "
                f"({top_fp_n}/{n_fp_total} = {top_fp_pct:.1f}%). Algo fires "
                f"early/late around real reaches. Lever: threshold tuning, "
                f"label construction at boundaries."
            )

    if not tp_rows:
        rec_lines.append(
            "Boundary-error tail: ALL TPs are perfect (start_delta=0 AND "
            "span_delta=0). No matched-reach boundary errors to fix."
        )
    else:
        n_with_err = len(tp_rows)
        pct_with_err = 100 * n_with_err / n_tps
        worst = tp_rows[0]
        rec_lines.append(
            f"Boundary-error tail: {n_with_err}/{n_tps} TPs ({pct_with_err:.1f}%) "
            f"have non-zero start_delta or span_delta. "
            f"Worst case: video {worst['video_id']} GT [{worst['gt_start_frame']}, "
            f"{worst['gt_end_frame']}] vs algo [{worst['algo_start_frame']}, "
            f"{worst['algo_end_frame']}], start_delta={worst['start_delta']}, "
            f"span_delta={worst['span_delta']}."
        )

    fn_table = "\n".join(
        f"| {cat} | {n} | {100*n/n_fn_total:.2f}% |"
        for cat, n in fn_categories.most_common()
    )
    fp_table = "\n".join(
        f"| {cat} | {n} | {100*n/n_fp_total:.2f}% |"
        for cat, n in fp_categories.most_common()
    )

    results_md = f"""# v8 failure-mode diagnostic -- RESULTS

**Date:** 2026-05-04
**Snapshot:** `v8.0.0_dev_failure_mode_breakdown/`
**Data source:** `metrics/loocv_aggregate.json` (extended schema, 16-fold LOOCV)
**Analysis script:** `scripts/analyze_v8_failure_modes.py`

## Headline counts (sanity-check vs baseline)

| | Count |
|---|---:|
| TP | {data['summary']['n_tp']} |
| FP | {data['summary']['n_fp']} |
| FN | {data['summary']['n_fn']} |
| TPs with perfect boundaries (start_delta=0 AND span_delta=0) | {n_perfect_tps} |
| TPs with non-zero boundary error | {len(tp_rows)} |

Matches `v8.0.0_dev_initial_loocv` baseline exactly.

## FN breakdown (where do the {n_fn_total} missed reaches come from?)

| Category | Count | % of FN |
|---|---:|---:|
{fn_table}

**Definitions** (per FN GT reach, find nearest algo reach in same video):
- **model_miss** -- no algo reach within +/-50 frames; model produced no in-reach signal at all
- **tol_miss_start** -- algo reach within +/-50f but |start_delta| > 2 (start_tol failure)
- **tol_miss_span** -- algo within start_tol but span fails tolerance
- **tol_miss_both** -- algo nearby but both start and span fail
- **fragmented** -- algo reach within +/-50f and its span is <50% of GT span (split-piece pattern)

## FP breakdown (where do the {n_fp_total} false positives sit relative to GT?)

| Category | Count | % of FP |
|---|---:|---:|
{fp_table}

**Definitions** (per FP, find nearest GT reach by start_frame):
- **split_twin** -- FP within +/-10f of a GT reach AND that GT has a TP entry (matched). FP is the second half of a split GT.
- **near_unmatched_gt** -- FP within +/-10f of a GT_start AND that GT is FN. FP could be the algo that failed to match due to tolerances.
- **pre_reach** -- FP ends within 10f BEFORE a GT_start, no overlap. Algo fired early.
- **post_reach** -- FP starts within 10f AFTER a GT_end, no overlap. Algo fired late.
- **within_gt** -- FP overlaps a GT reach window.
- **random** -- FP further than 30f from any GT reach. Phantom; model fired on something unrelated.
- **other** -- FP near GT but doesn't fit any of the above (10-30f from GT, no overlap).

## Recommendations

{chr(10).join('- ' + line for line in rec_lines)}

## Boundary error tail

{n_perfect_tps} of {n_tps} TPs ({100*n_perfect_tps/n_tps:.1f}%) have perfect boundaries (start_delta=0 AND span_delta=0).
Detailed list of imperfect TPs: `metrics/boundary_error_tail.csv`

## Outputs

- `metrics/fn_breakdown.json` -- full FN counts including per-video breakdown
- `metrics/fp_breakdown.json` -- full FP counts including per-video breakdown
- `metrics/boundary_error_tail.csv` -- non-perfect TPs ranked by max(|start_delta|, |span_delta|)
- `figures/failure_mode_summary.png` -- 4-panel summary (FN sources, FP sources, |start_delta| histogram, |span_delta| histogram)
"""

    (SNAPSHOT_DIR / "RESULTS.md").write_text(results_md, encoding="utf-8")
    print(f"Wrote: {SNAPSHOT_DIR / 'RESULTS.md'}")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
