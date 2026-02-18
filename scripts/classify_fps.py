#!/usr/bin/env python
"""Classify FP reaches as likely-real vs likely-genuine-FP using spatial trajectory."""

import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

REACH_TOLERANCE = 10
LIKELIHOOD_THRESHOLD = 0.5

print = __builtins__.__dict__['print']  # ensure we get the real print
import functools
print = functools.partial(print, flush=True)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_dlc_csv(path):
    df = pd.read_csv(path, header=[0, 1, 2])
    df.columns = [f"{bp}_{coord}" for _, bp, coord in df.columns]
    return df


def matches_gt(algo_reach, gt_reaches):
    s, e = algo_reach['start_frame'], algo_reach['end_frame']
    for gt in gt_reaches:
        if abs(s - gt['start_frame']) <= REACH_TOLERANCE and abs(e - gt['end_frame']) <= REACH_TOLERANCE:
            return True
    return False


def get_righthand_x(dlc_df, frame, slit_x):
    """Get RightHand X offset from slit. Returns NaN if not visible."""
    if frame >= len(dlc_df):
        return np.nan
    lik = dlc_df.loc[frame, 'RightHand_likelihood']
    if lik < LIKELIHOOD_THRESHOLD:
        return np.nan
    return dlc_df.loc[frame, 'RightHand_x'] - slit_x


def analyze_fp_trajectory(dlc_df, reach, slit_x):
    """Compute spatial features for a single reach."""
    start = reach['start_frame']
    end = reach['end_frame']
    duration = end - start + 1

    offsets = []
    for f in range(start, min(end + 1, len(dlc_df))):
        x = get_righthand_x(dlc_df, f, slit_x)
        if not np.isnan(x):
            offsets.append(x)

    if not offsets:
        return {
            'duration': duration,
            'n_visible': 0,
            'visibility_pct': 0,
            'max_x': np.nan,
            'mean_x': np.nan,
            'pct_past_slit': 0,
            'classification': 'no_data',
        }

    offsets = np.array(offsets)
    max_x = float(np.max(offsets))
    mean_x = float(np.mean(offsets))
    pct_past = float(np.sum(offsets > 0) / len(offsets))

    # Classification logic:
    # - "likely_reach": max extension > 5px past slit AND >25% of frames past slit
    # - "borderline": max extension > 0 but not strongly past slit
    # - "likely_fp": hand never meaningfully crossed slit
    if max_x > 5 and pct_past > 0.25:
        classification = 'likely_reach'
    elif max_x > 0:
        classification = 'borderline'
    else:
        classification = 'likely_fp'

    return {
        'duration': duration,
        'n_visible': len(offsets),
        'visibility_pct': len(offsets) / duration * 100,
        'max_x': max_x,
        'mean_x': mean_x,
        'pct_past_slit': pct_past * 100,
        'classification': classification,
    }


def main():
    from mousereach.config import require_processing_root
    processing_dir = require_processing_root() / "Processing"

    gt_files = sorted(processing_dir.glob("*_unified_ground_truth.json"))
    print(f"Found {len(gt_files)} GT files\n")

    all_fps = []  # list of dicts with video_name + features
    all_tps = []

    for gt_file in gt_files:
        video_name = gt_file.stem.replace('_unified_ground_truth', '')

        gt_data = load_json(gt_file)
        gt_reaches = gt_data.get('reaches', {}).get('reaches', [])

        algo_file = gt_file.parent / f"{video_name}_reaches.json"
        if not algo_file.exists():
            continue

        algo_data = load_json(algo_file)
        algo_reaches = []
        for seg in algo_data.get('segments', []):
            algo_reaches.extend(seg.get('reaches', []))

        dlc_files = list(processing_dir.glob(f"{video_name}DLC*.csv"))
        if not dlc_files:
            continue

        try:
            dlc_df = load_dlc_csv(dlc_files[0])
        except Exception:
            continue

        boxl_x = dlc_df['BOXL_x'].median()
        boxr_x = dlc_df['BOXR_x'].median()
        slit_x = (boxl_x + boxr_x) / 2

        for reach in algo_reaches:
            is_tp = matches_gt(reach, gt_reaches)
            features = analyze_fp_trajectory(dlc_df, reach, slit_x)
            features['video'] = video_name
            features['start_frame'] = reach['start_frame']
            features['end_frame'] = reach['end_frame']

            if is_tp:
                all_tps.append(features)
            else:
                all_fps.append(features)

    print("=" * 80)
    print(f"FP CLASSIFICATION RESULTS")
    print(f"Total TPs: {len(all_tps)}, Total FPs: {len(all_fps)}")
    print("=" * 80)

    # Count classifications
    classes = {}
    for fp in all_fps:
        c = fp['classification']
        classes[c] = classes.get(c, 0) + 1

    print(f"\nFP Classification Breakdown:")
    for c in ['likely_reach', 'borderline', 'likely_fp', 'no_data']:
        n = classes.get(c, 0)
        pct = n / len(all_fps) * 100 if all_fps else 0
        print(f"  {c:20s}: {n:4d}  ({pct:5.1f}%)")

    # Stats for each class
    print(f"\n{'':20s}  {'max_x':>8s}  {'mean_x':>8s}  {'%past':>6s}  {'dur':>5s}  {'vis%':>5s}")
    print("-" * 70)

    for cls_name in ['likely_reach', 'borderline', 'likely_fp']:
        subset = [f for f in all_fps if f['classification'] == cls_name]
        if not subset:
            continue
        max_xs = [f['max_x'] for f in subset if not np.isnan(f['max_x'])]
        mean_xs = [f['mean_x'] for f in subset if not np.isnan(f['mean_x'])]
        pcts = [f['pct_past_slit'] for f in subset]
        durs = [f['duration'] for f in subset]
        vis = [f['visibility_pct'] for f in subset]

        print(f"{cls_name:20s}  "
              f"{np.median(max_xs):7.1f}px  "
              f"{np.median(mean_xs):7.1f}px  "
              f"{np.median(pcts):5.1f}%  "
              f"{np.median(durs):5.0f}f  "
              f"{np.median(vis):5.1f}%")

    # For comparison: TP stats
    tp_max_xs = [f['max_x'] for f in all_tps if not np.isnan(f['max_x'])]
    tp_mean_xs = [f['mean_x'] for f in all_tps if not np.isnan(f['mean_x'])]
    tp_pcts = [f['pct_past_slit'] for f in all_tps]
    tp_durs = [f['duration'] for f in all_tps]
    tp_vis = [f['visibility_pct'] for f in all_tps]
    print(f"{'(TPs for reference)':20s}  "
          f"{np.median(tp_max_xs):7.1f}px  "
          f"{np.median(tp_mean_xs):7.1f}px  "
          f"{np.median(tp_pcts):5.1f}%  "
          f"{np.median(tp_durs):5.0f}f  "
          f"{np.median(tp_vis):5.1f}%")

    # Per-video breakdown for likely_reach FPs
    print(f"\n\nPer-video: FPs classified as 'likely_reach' (probable unlabeled GT):")
    print("-" * 70)
    video_counts = {}
    for fp in all_fps:
        if fp['classification'] == 'likely_reach':
            v = fp['video']
            video_counts[v] = video_counts.get(v, 0) + 1

    for v, n in sorted(video_counts.items(), key=lambda x: -x[1]):
        total_fp = sum(1 for f in all_fps if f['video'] == v)
        print(f"  {v}: {n} likely-real of {total_fp} FPs")

    # Corrected precision estimate
    n_likely_real = classes.get('likely_reach', 0)
    n_borderline = classes.get('borderline', 0)
    n_likely_fp = classes.get('likely_fp', 0) + classes.get('no_data', 0)

    corrected_tp = len(all_tps) + n_likely_real
    corrected_fp = n_likely_fp + n_borderline  # be conservative with borderline
    corrected_p = corrected_tp / (corrected_tp + corrected_fp) * 100

    print(f"\n\nCorrected precision estimate:")
    print(f"  Measured TPs:     {len(all_tps)}")
    print(f"  + likely_reach:   {n_likely_real} (probable unlabeled TPs)")
    print(f"  = Corrected TPs:  {corrected_tp}")
    print(f"  Remaining FPs:    {corrected_fp} ({n_likely_fp} likely_fp + {n_borderline} borderline)")
    print(f"  Measured precision:  {len(all_tps) / (len(all_tps) + len(all_fps)) * 100:.1f}%")
    print(f"  Corrected precision: {corrected_p:.1f}%")


if __name__ == '__main__':
    main()
