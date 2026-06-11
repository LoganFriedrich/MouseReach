"""
Diagnostic: per-frame proba inspection for v8.0.0 merge cases on a single video.

For one video (default: 20250718_CNT0214_P1), runs v8.0.0 inference, identifies
"merge candidates" -- algo reach windows that contain >=2 GT reach starts --
and examines the per-frame probability series in the gaps between those GT
reaches. Tells us whether the model produces a detectable confidence dip at
the boundary where it merged two real reaches into one.

Output is read by a human and answers two questions:

  1. How often does the merge pattern occur on this video?
  2. When it occurs, is there a detectable proba dip in the gap between
     the merged GT reaches that a splitter could use?

If gap_min_proba is consistently low (e.g., <0.7) while the surrounding
reach proba is high (~0.95), the model "sees" the boundary and a splitter
based on proba dips could work. If gap_min_proba is essentially identical
to the surrounding reach (e.g., both ~0.95), the model truly merged the
two reaches in feature space and proba can't help.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.core.geometry import load_dlc
from mousereach.reach.v8 import DEFAULT_MODEL_PATH
from mousereach.reach.v8.features import extract_features
from mousereach.reach.v8.postprocess import probabilities_to_reaches


VIDEO = "20250718_CNT0214_P1"

DLC_PATH = Path(
    rf"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    rf"\generalization_test_2026-05-11\algo_outputs_current"
    rf"\{VIDEO}DLC_resnet50_MPSAOct27shuffle1_100000.h5"
)
GT_PATH = Path(
    rf"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    rf"\generalization_test_2026-05-11\gt\{VIDEO}_unified_ground_truth.json"
)

THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3


def main():
    print("=" * 78)
    print(f"MERGE PROBA DIP INSPECTION -- video: {VIDEO}")
    print("=" * 78)
    print()

    # Inference
    print(f"Loading DLC ...", flush=True)
    dlc = load_dlc(DLC_PATH)
    print(f"Loading v8.0.0 model ...", flush=True)
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]
    print(f"Extracting features + running inference ...", flush=True)
    feats = extract_features(dlc)
    X = feats[feat_cols].to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)[:, 1]

    raw_reaches = probabilities_to_reaches(
        proba, threshold=THRESHOLD, merge_gap=MERGE_GAP, min_span=MIN_SPAN)

    # Load GT (with apex)
    gt_data = json.loads(GT_PATH.read_text(encoding="utf-8"))
    raw_gt = gt_data["reaches"]["reaches"]
    gts = []
    for r in raw_gt:
        if r.get("exclude_from_analysis", False):
            continue
        apex_raw = r.get("apex_frame")
        apex = int(apex_raw) if apex_raw is not None else -1
        gts.append((int(r["start_frame"]), int(r["end_frame"]), apex))
    gts.sort()

    print(f"Algo reaches: {len(raw_reaches)}")
    print(f"GT reaches:   {len(gts)}")
    print()

    # Identify merge candidates: algo reach with >=2 GT starts inside
    merges = []
    for r in raw_reaches:
        gts_inside = [(gs, ge, ga) for gs, ge, ga in gts
                       if r.start_frame <= gs <= r.end_frame]
        if len(gts_inside) >= 2:
            merges.append((r.start_frame, r.end_frame, gts_inside))

    print(f"Merge candidates (algo span containing 2+ GT starts): {len(merges)}")
    print()

    if not merges:
        print("No merge cases on this video.")
        return

    # Detailed inspection
    print("=" * 78)
    print("PER-MERGE DETAIL")
    print("=" * 78)
    all_dip_records = []
    for k, (algo_s, algo_e, gts_in) in enumerate(merges, start=1):
        reach_proba = proba[algo_s:algo_e + 1]
        reach_min = float(reach_proba.min())
        reach_mean = float(reach_proba.mean())
        reach_median = float(np.median(reach_proba))
        print(f"\nMerge {k}: algo=[{algo_s},{algo_e}] span={algo_e-algo_s+1}f  "
              f"reach_proba min={reach_min:.3f} mean={reach_mean:.3f} med={reach_median:.3f}")
        print(f"  Contains {len(gts_in)} GT reaches:")
        for i, (gs, ge, ga) in enumerate(gts_in, start=1):
            apex_str = f" apex={ga}" if ga >= 0 else ""
            print(f"    GT{i}: [{gs},{ge}] span={ge-gs+1}f{apex_str}")
        # Gaps between consecutive GTs
        for i in range(len(gts_in) - 1):
            gs1, ge1, _ = gts_in[i]
            gs2, ge2, _ = gts_in[i + 1]
            gap_start = ge1 + 1
            gap_end = gs2 - 1
            if gap_end < gap_start:
                # Adjacent GTs, no gap frames
                print(f"  Gap between GT{i+1} and GT{i+2}: ZERO frames (adjacent)")
                # Check single-frame proba at GT boundary
                boundary_frame = ge1
                if 0 <= boundary_frame < len(proba):
                    bp = float(proba[boundary_frame])
                    bp1 = float(proba[boundary_frame + 1]) if boundary_frame + 1 < len(proba) else float("nan")
                    print(f"    proba at GT{i+1}_end (frame {boundary_frame}): {bp:.3f}")
                    print(f"    proba at GT{i+2}_start (frame {boundary_frame+1}): {bp1:.3f}")
                continue
            gap_proba = proba[gap_start:gap_end + 1]
            gap_min = float(gap_proba.min())
            gap_max = float(gap_proba.max())
            gap_mean = float(gap_proba.mean())
            argmin_offset = int(np.argmin(gap_proba))
            argmin_frame = gap_start + argmin_offset
            below_threshold = int((gap_proba <= THRESHOLD).sum())
            dip_depth = reach_median - gap_min
            print(f"  Gap GT{i+1}<->GT{i+2}: frames [{gap_start},{gap_end}] span={gap_end-gap_start+1}f")
            print(f"    Gap proba: min={gap_min:.3f} (at frame {argmin_frame})  "
                  f"max={gap_max:.3f}  mean={gap_mean:.3f}")
            print(f"    Dip depth vs reach median: {dip_depth:+.3f}  "
                  f"({below_threshold} frames <= {THRESHOLD})")
            all_dip_records.append({
                "merge_idx": k,
                "gap_start": gap_start, "gap_end": gap_end,
                "gap_span": gap_end - gap_start + 1,
                "gap_min": gap_min, "gap_max": gap_max, "gap_mean": gap_mean,
                "argmin_frame": argmin_frame,
                "reach_median": reach_median,
                "dip_depth": dip_depth,
                "n_frames_below_threshold": below_threshold,
            })

    # Summary distribution across all merges
    if all_dip_records:
        print()
        print("=" * 78)
        print(f"AGGREGATE DIP CHARACTERIZATION ({len(all_dip_records)} gaps)")
        print("=" * 78)
        mins = np.array([r["gap_min"] for r in all_dip_records])
        depths = np.array([r["dip_depth"] for r in all_dip_records])
        spans = np.array([r["gap_span"] for r in all_dip_records])
        below = np.array([r["n_frames_below_threshold"] for r in all_dip_records])
        print(f"  Gap span (frames):       median={int(np.median(spans))}  mean={spans.mean():.1f}  range=[{spans.min()},{spans.max()}]")
        print(f"  Gap min proba:           median={np.median(mins):.3f}  mean={mins.mean():.3f}  range=[{mins.min():.3f},{mins.max():.3f}]")
        print(f"  Dip depth vs reach med:  median={np.median(depths):+.3f}  mean={depths.mean():+.3f}  range=[{depths.min():+.3f},{depths.max():+.3f}]")
        print(f"  Frames <= 0.5 in gap:    median={int(np.median(below))}  mean={below.mean():.1f}  range=[{below.min()},{below.max()}]")
        print()
        # Bucket by depth
        deep = sum(1 for d in depths if d > 0.20)
        mid = sum(1 for d in depths if 0.05 < d <= 0.20)
        shallow = sum(1 for d in depths if d <= 0.05)
        print(f"  Dip depth buckets:")
        print(f"    Deep dip (> 0.20 below reach median):   {deep}  ({100*deep/len(depths):.0f}%)  splittable on proba")
        print(f"    Mid dip (0.05-0.20 below):              {mid}  ({100*mid/len(depths):.0f}%)  possibly splittable")
        print(f"    Shallow / no dip (<= 0.05 below):       {shallow}  ({100*shallow/len(depths):.0f}%)  not splittable on proba")
        print()
        # Bucket by sub-threshold frames
        clean_break = sum(1 for r in all_dip_records if r["n_frames_below_threshold"] >= 1)
        print(f"  Gaps with >=1 sub-threshold frame:       {clean_break}  ({100*clean_break/len(depths):.0f}%)")
        print(f"    (these would split if merge_gap=0; current merge_gap={MERGE_GAP} bridges them)")


if __name__ == "__main__":
    main()
