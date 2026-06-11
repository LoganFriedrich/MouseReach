"""Phase A diagnostic: characterize the TOLERANCE_ERROR(start_late)
population after v8.0.2 trim.

Question: are these events "algo missed initial frames" (algo window is a
SUBSET of real reach) or "algo found wrong reach" (large delta, shifted
window covering different behavior)?

If subset: asymmetric tolerance relaxation is defensible (no kinematic
corruption). If shifted: relaxation would let in genuinely wrong matches.

Procedure:
1. Apply v8.0.2 trim to v8.0.1 algo outputs (calibration + holdout)
2. Re-match against GT at strict +/-2 tolerance
3. Extract events where start_delta > +2 with overlap to some GT
4. Compute:
   - start_delta histogram (where do these events cluster?)
   - span_delta histogram (truncated subset vs shifted span)
   - algo_end vs gt_end relationship (is algo window a subset?)
5. Surface representative examples for spot-check

Uses the production v8.0.2 trim from mousereach.reach.v8.postprocess.
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

from mousereach.reach.v8.postprocess import (
    ReachSpan, trim_leading_sustained_lk, compute_paw_mean_lk,
)

# Likelihood columns differ by source:
#   DLC h5 (raw): "RightHand_likelihood", ...   (used by production compute_paw_mean_lk)
#   Parquet (extracted features): "RightHand_lk", ...  (per features.py _lk suffix)
PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]


CAL_LOOCV_JSON = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_model_3_1_baseline_loocv\metrics\loocv_aggregate.json"
)
CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
HOLDOUT_ALGO_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_holdout_generalization_merge_gap_0\algo_outputs_v8.0.0_mg0"
)
HOLDOUT_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)
HOLDOUT_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.2_dev_start_late_diagnostic"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

START_TOL = 2
SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def find_overlapping_gts(algo_start, algo_end, gts):
    return [(gs, ge) for gs, ge in gts if overlap(algo_start, algo_end, gs, ge)]


def load_holdout_algo(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [(int(r["start_frame"]), int(r["end_frame"]))
            for r in data.get("reaches", [])]


def load_holdout_gt(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    reaches_obj = data.get("reaches", {})
    rlist = reaches_obj.get("reaches", []) if isinstance(reaches_obj, dict) else []
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


def load_dlc(path):
    df = pd.read_hdf(path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df


def classify_event(algo_s, algo_e, gts):
    """Return: ('TP'|'TOLERANCE_ERROR(start_late)'|...|'FALSE_POSITIVE',
                start_delta, span_delta, matched_gt or None).
    """
    overlapping = find_overlapping_gts(algo_s, algo_e, gts)
    if not overlapping:
        return ("FALSE_POSITIVE", None, None, None)
    # Pick nearest by start_delta (smallest |start_delta|)
    best_gt = min(overlapping, key=lambda g: abs(algo_s - g[0]))
    gs, ge = best_gt
    start_delta = algo_s - gs
    algo_span = algo_e - algo_s + 1
    gt_span = ge - gs + 1
    span_delta = algo_span - gt_span
    span_tol = max(SPAN_TOL_FRAC * gt_span, SPAN_TOL_MIN)
    if abs(start_delta) <= START_TOL and abs(span_delta) <= span_tol:
        return ("TP", start_delta, span_delta, best_gt)
    if start_delta > START_TOL:
        return ("TOLERANCE_ERROR(start_late)", start_delta, span_delta, best_gt)
    if start_delta < -START_TOL:
        return ("TOLERANCE_ERROR(start_early)", start_delta, span_delta, best_gt)
    if span_delta > span_tol:
        return ("TOLERANCE_ERROR(span_over)", start_delta, span_delta, best_gt)
    if span_delta < -span_tol:
        return ("TOLERANCE_ERROR(span_short)", start_delta, span_delta, best_gt)
    return ("TOLERANCE_ERROR(unclassified)", start_delta, span_delta, best_gt)


def process_video(vid, algos, gts, paw_lk):
    """Apply v8.0.2 trim, classify each algo event, return list of records."""
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos]
    trimmed = trim_leading_sustained_lk(spans, paw_lk,
                                         threshold=0.60, sustain_n=3, min_span=3)
    out = []
    for r in trimmed:
        a_s, a_e = r.start_frame, r.end_frame
        kind, sd, spd, gt = classify_event(a_s, a_e, gts)
        if gt is None:
            out.append({
                "video": vid, "kind": kind,
                "algo_start": a_s, "algo_end": a_e,
                "algo_span": a_e - a_s + 1,
                "gt_start": None, "gt_end": None, "gt_span": None,
                "start_delta": None, "span_delta": None,
                "algo_window_inside_gt": None,
                "algo_end_past_gt_end": None,
            })
        else:
            gs, ge = gt
            out.append({
                "video": vid, "kind": kind,
                "algo_start": a_s, "algo_end": a_e,
                "algo_span": a_e - a_s + 1,
                "gt_start": gs, "gt_end": ge, "gt_span": ge - gs + 1,
                "start_delta": sd, "span_delta": spd,
                "algo_window_inside_gt": (a_s >= gs and a_e <= ge),
                "algo_end_past_gt_end": (a_e - ge),  # positive if past
            })
    return out


def main():
    print("=" * 70)
    print("v8.0.2 start_late population diagnostic")
    print("=" * 70)
    print()

    # ===== Calibration =====
    print("Loading calibration LOOCV outputs...", flush=True)
    loocv = json.loads(CAL_LOOCV_JSON.read_text(encoding="utf-8"))
    raw = loocv["raw_results"]
    # Reconstruct algo and GT per video
    cal_algos = defaultdict(set)
    cal_gts = defaultdict(set)
    for r in raw:
        vid = r["video_id"]
        if r["algo_start_frame"] >= 0:
            cal_algos[vid].add((int(r["algo_start_frame"]),
                                 int(r["algo_end_frame"])))
        if r["gt_start_frame"] >= 0:
            cal_gts[vid].add((int(r["gt_start_frame"]),
                              int(r["gt_end_frame"])))
    print(f"  {len(cal_algos)} videos, "
          f"{sum(len(s) for s in cal_algos.values())} total algo reaches")

    print("Loading parquet for paw_mean_lk...", flush=True)
    df = pd.read_parquet(CAL_PARQUET, columns=["video_id", "frame"] + PARQUET_LK_COLS)
    df["paw_mean_lk"] = df[PARQUET_LK_COLS].to_numpy(dtype=np.float32).mean(axis=1)
    cal_lk = {}
    for vid, grp in df.groupby("video_id", sort=False):
        grp_sorted = grp.sort_values("frame")
        mx = int(grp_sorted["frame"].max())
        arr = np.full(mx + 1, np.nan, dtype=np.float32)
        arr[grp_sorted["frame"].to_numpy()] = grp_sorted["paw_mean_lk"].to_numpy()
        cal_lk[vid] = arr

    print("Applying v8.0.2 trim + classifying calibration events...", flush=True)
    cal_records = []
    for vid in sorted(cal_algos):
        if vid not in cal_lk:
            continue
        algos = sorted(cal_algos[vid])
        gts = sorted(cal_gts.get(vid, set()))
        cal_records.extend(process_video(vid, algos, gts, cal_lk[vid]))

    cal_df = pd.DataFrame(cal_records)
    cal_df["corpus"] = "calibration"
    print(f"  {len(cal_df)} algo events after trim")
    print(f"  Class counts: {cal_df['kind'].value_counts().to_dict()}")
    print()

    # ===== Holdout =====
    print("Loading holdout outputs + DLC + GT...", flush=True)
    holdout_records = []
    for algo_path in sorted(HOLDOUT_ALGO_DIR.glob("*_reaches.json")):
        vid = algo_path.stem.replace("_reaches", "")
        gt_path = HOLDOUT_GT_DIR / f"{vid}_unified_ground_truth.json"
        dlc_path = HOLDOUT_DLC_DIR / f"{vid}DLC_resnet50_MPSAOct27shuffle1_100000.h5"
        if not gt_path.exists() or not dlc_path.exists():
            continue
        algos = load_holdout_algo(algo_path)
        gts = load_holdout_gt(gt_path)
        dlc = load_dlc(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        holdout_records.extend(process_video(vid, algos, gts, paw_lk))

    h_df = pd.DataFrame(holdout_records)
    h_df["corpus"] = "holdout"
    print(f"  {len(h_df)} algo events after trim")
    print(f"  Class counts: {h_df['kind'].value_counts().to_dict()}")
    print()

    all_df = pd.concat([cal_df, h_df], ignore_index=True)
    all_df.to_csv(OUT_DIR / "metrics" / "all_events_v8_0_2.csv", index=False)
    print(f"Saved per-event records: {OUT_DIR / 'metrics' / 'all_events_v8_0_2.csv'}")
    print()

    # ===== Focus: start_late events =====
    print("=" * 70)
    print("FOCUS: TOLERANCE_ERROR(start_late) events")
    print("=" * 70)
    for corpus in ("calibration", "holdout"):
        sub = all_df[(all_df["corpus"] == corpus)
                     & (all_df["kind"] == "TOLERANCE_ERROR(start_late)")]
        if not len(sub):
            continue
        print(f"\n  Corpus: {corpus}  (n={len(sub)} start_late events)")
        sd = sub["start_delta"].astype(int)
        spd = sub["span_delta"].astype(int)
        print(f"    start_delta: min={sd.min()} median={int(sd.median())} mean={sd.mean():.1f} max={sd.max()}")
        print(f"    span_delta:  min={spd.min()} median={int(spd.median())} mean={spd.mean():.1f} max={spd.max()}")
        print(f"    fraction with algo_window inside gt (algo_start>=gt_start AND algo_end<=gt_end): "
              f"{sub['algo_window_inside_gt'].sum()}/{len(sub)} ({100*sub['algo_window_inside_gt'].mean():.0f}%)")
        print(f"    algo_end past gt_end (positive = algo extends past gt): "
              f"min={int(sub['algo_end_past_gt_end'].min())}  median={int(sub['algo_end_past_gt_end'].median())}  max={int(sub['algo_end_past_gt_end'].max())}")
        # start_delta bin counts
        print("    start_delta bins:")
        bins = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 10), (11, 15), (16, 25), (26, 1000)]
        for lo, hi in bins:
            n = ((sd >= lo) & (sd <= hi)).sum()
            print(f"      {lo:>3}-{hi:<3}: {n:>3}  ({100*n/len(sub):.0f}%)")

    # ===== Examples =====
    print("\n=== EXAMPLES (10 per corpus, sorted by start_delta) ===")
    for corpus in ("calibration", "holdout"):
        sub = all_df[(all_df["corpus"] == corpus)
                     & (all_df["kind"] == "TOLERANCE_ERROR(start_late)")]
        if not len(sub):
            continue
        print(f"\n  -- {corpus} --")
        print(f"  {'video':<24} {'algo_start':>10} {'algo_end':>9} {'algo_span':>10} "
              f"{'gt_start':>9} {'gt_end':>8} {'gt_span':>8} {'sd':>4} {'spd':>4} {'inside_gt':>10}")
        for _, r in sub.sort_values("start_delta").head(10).iterrows():
            print(f"  {r['video']:<24} {r['algo_start']:>10} {r['algo_end']:>9} "
                  f"{int(r['algo_span']):>10} {int(r['gt_start']):>9} {int(r['gt_end']):>8} "
                  f"{int(r['gt_span']):>8} {int(r['start_delta']):>+4} {int(r['span_delta']):>+4} "
                  f"{str(r['algo_window_inside_gt']):>10}")
        print()
        print(f"  -- {corpus} tail (largest start_delta) --")
        for _, r in sub.sort_values("start_delta").tail(5).iterrows():
            print(f"  {r['video']:<24} {r['algo_start']:>10} {r['algo_end']:>9} "
                  f"{int(r['algo_span']):>10} {int(r['gt_start']):>9} {int(r['gt_end']):>8} "
                  f"{int(r['gt_span']):>8} {int(r['start_delta']):>+4} {int(r['span_delta']):>+4} "
                  f"{str(r['algo_window_inside_gt']):>10}")

    # ===== Figure =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, corpus in enumerate(("calibration", "holdout")):
        sub = all_df[(all_df["corpus"] == corpus)
                     & (all_df["kind"] == "TOLERANCE_ERROR(start_late)")]
        if not len(sub):
            continue
        sd = sub["start_delta"].astype(int)
        spd = sub["span_delta"].astype(int)
        ax = axes[0, col]
        ax.hist(sd.values, bins=range(int(sd.min()), int(sd.max()) + 2),
                color="C3", alpha=0.7)
        ax.axvline(2, color="0.5", ls="--", label="strict tol +2")
        ax.axvline(5, color="0.4", ls=":", label="K=5 candidate cap")
        ax.set_xlabel("start_delta (algo_start - gt_start)")
        ax.set_ylabel("count")
        ax.set_title(f"{corpus}: start_late (n={len(sub)})  "
                     f"median={int(sd.median())}  max={int(sd.max())}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.scatter(sd.values, spd.values, alpha=0.6, color="C3")
        ax.axvline(2, color="0.5", ls="--")
        ax.axhline(0, color="0.5", ls="-", lw=0.5)
        ax.set_xlabel("start_delta")
        ax.set_ylabel("span_delta (algo_span - gt_span)")
        ax.set_title("start_delta vs span_delta")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig = OUT_DIR / "figures" / "start_late_distribution.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {out_fig}")


if __name__ == "__main__":
    main()
