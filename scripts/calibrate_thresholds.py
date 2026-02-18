"""
Calibrate spatial refiner thresholds from ground truth data.

For every GT video, load the DLC data and GT reaches. Compute hand position
distributions for:
  1. Frames INSIDE GT reaches (where hand should be past slit)
  2. Frames OUTSIDE GT reaches (where hand should be behind slit)
  3. First/last N frames of GT reaches (boundary behavior)
  4. Short reaches (4-10 frames) that ARE in GT vs algo FPs

This tells us what threshold actually separates "real reach" from "not reach"
in position space.
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

SRC = Path(r"y:\2_Connectome\Behavior\MouseReach\src")
sys.path.insert(0, str(SRC))

print("Importing...", flush=True)
from mousereach.config import require_processing_root
from mousereach.reach.core.geometry import load_dlc, load_segments

processing_dir = require_processing_root() / "Processing"

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']

def get_mean_hand_x(row, threshold=0.5):
    """Mean X of visible hand points at given threshold."""
    xs = []
    for p in RH_POINTS:
        l = row.get(f'{p}_likelihood', 0)
        if l >= threshold:
            x = row.get(f'{p}_x', np.nan)
            if not np.isnan(x):
                xs.append(float(x))
    return float(np.mean(xs)) if xs else None

def get_best_hand_x(row, threshold=0.15):
    """X of highest-confidence hand point."""
    best_x, best_l = None, 0
    for p in RH_POINTS:
        l = row.get(f'{p}_likelihood', 0)
        if l > best_l and l >= threshold:
            x = row.get(f'{p}_x', np.nan)
            if not np.isnan(x):
                best_x, best_l = float(x), l
    return best_x

# Collect distributions
inside_reach_offsets_slit = []      # hand_x - slit_x during GT reaches
inside_reach_offsets_boxr = []      # hand_x - boxr_x during GT reaches
outside_reach_offsets_slit = []     # hand_x - slit_x outside GT reaches (but nose engaged)
outside_reach_offsets_boxr = []     # hand_x - boxr_x outside GT reaches

first_2_frames_offsets_boxr = []    # hand_x - boxr_x in first 2 frames of GT reaches
last_2_frames_offsets_slit = []     # hand_x - slit_x in last 2 frames of GT reaches

short_gt_offsets_boxr = []          # hand_x - boxr_x for short (4-10 frame) GT reaches
short_gt_past_ratios = []           # fraction of frames past boxr_x for short GT reaches

# Per-video stats for sanity checking
video_stats = []

gt_files = sorted(processing_dir.glob("*_unified_ground_truth.json"))
print(f"Found {len(gt_files)} GT files", flush=True)

for gt_file in gt_files:
    with open(gt_file) as f:
        gt_data = json.load(f)

    video_name = gt_data.get("video_name", gt_file.stem.replace("_unified_ground_truth", ""))

    # Load DLC data
    dlc_files = list(processing_dir.glob(f"{video_name}*DLC*.h5"))
    if not dlc_files:
        print(f"  SKIP {video_name}: no DLC file", flush=True)
        continue

    seg_file = processing_dir / f"{video_name}_segments.json"
    if not seg_file.exists():
        print(f"  SKIP {video_name}: no segments file", flush=True)
        continue

    df = load_dlc(dlc_files[0])

    # Load segment boundaries and build intervals
    with open(seg_file) as f:
        seg_data = json.load(f)
    boundaries = seg_data.get('boundaries', [])
    if len(boundaries) < 2:
        continue

    # Build segment intervals from consecutive boundary pairs
    seg_intervals = [(boundaries[i], boundaries[i+1]) for i in range(0, len(boundaries) - 1, 2)]

    # Get GT reaches
    gt_reaches = gt_data.get('reaches', {}).get('reaches', [])
    if not gt_reaches:
        continue

    # Build set of frames that are inside GT reaches
    gt_reach_frames = set()
    for r in gt_reaches:
        for f in range(r['start_frame'], r['end_frame'] + 1):
            gt_reach_frames.add(f)

    # Process each segment
    for seg_start, seg_end in seg_intervals:

        # Compute geometry for this segment
        segment_df = df.iloc[seg_start:seg_end]
        if len(segment_df) < 10:
            continue

        boxl_x = segment_df['BOXL_x'].median()
        boxr_x = segment_df['BOXR_x'].median()
        slit_x = (boxl_x + boxr_x) / 2
        slit_width = boxr_x - boxl_x

        # For each frame in segment
        for frame in range(seg_start, min(seg_end, len(df))):
            row = df.iloc[frame]

            # Get hand position using both methods
            mean_x = get_mean_hand_x(row, threshold=0.5)
            best_x = get_best_hand_x(row, threshold=0.15)

            # Use mean_x (matches what the spatial refiner does)
            hand_x = mean_x
            if hand_x is None:
                continue

            offset_slit = hand_x - slit_x
            offset_boxr = hand_x - boxr_x

            if frame in gt_reach_frames:
                inside_reach_offsets_slit.append(offset_slit)
                inside_reach_offsets_boxr.append(offset_boxr)
            else:
                # Check if nose is engaged (similar to reach detector logic)
                nose_x = row.get('Nose_x', np.nan)
                nose_l = row.get('Nose_likelihood', 0)
                if nose_l >= 0.3 and not np.isnan(nose_x):
                    nose_dist = abs(nose_x - slit_x)
                    if nose_dist < 25:  # NOSE_ENGAGEMENT_THRESHOLD
                        outside_reach_offsets_slit.append(offset_slit)
                        outside_reach_offsets_boxr.append(offset_boxr)

    # Analyze first/last frames and short reaches
    for r in gt_reaches:
        start, end = r['start_frame'], r['end_frame']
        duration = end - start + 1

        # Find this reach's segment for boxr_x
        seg_for_reach = None
        for s_start, s_end in seg_intervals:
            if s_start <= start <= s_end:
                seg_for_reach = (s_start, s_end)
                break
        if seg_for_reach is None:
            continue

        seg_df = df.iloc[seg_for_reach[0]:seg_for_reach[1]]
        boxr_x = seg_df['BOXR_x'].median()
        slit_x = (seg_df['BOXL_x'].median() + boxr_x) / 2

        # First 2 frames
        for f in range(start, min(start + 2, end + 1, len(df))):
            hand_x = get_mean_hand_x(df.iloc[f])
            if hand_x is not None:
                first_2_frames_offsets_boxr.append(hand_x - boxr_x)

        # Last 2 frames
        for f in range(max(end - 1, start), min(end + 1, len(df))):
            hand_x = get_mean_hand_x(df.iloc[f])
            if hand_x is not None:
                last_2_frames_offsets_slit.append(hand_x - slit_x)

        # Short reaches
        if 4 <= duration <= 10:
            frames_past = 0
            frames_with_pos = 0
            offsets = []
            for f in range(start, min(end + 1, len(df))):
                hand_x = get_mean_hand_x(df.iloc[f])
                if hand_x is not None:
                    frames_with_pos += 1
                    offset = hand_x - boxr_x
                    offsets.append(offset)
                    if hand_x > boxr_x:
                        frames_past += 1
            if frames_with_pos > 0:
                short_gt_offsets_boxr.extend(offsets)
                short_gt_past_ratios.append(frames_past / frames_with_pos)

    print(f"  {video_name}: {len(gt_reaches)} GT reaches", flush=True)

# ========================================================================
# REPORT
# ========================================================================
print(f"\n{'='*70}", flush=True)
print(f"  HAND POSITION CALIBRATION DATA", flush=True)
print(f"{'='*70}", flush=True)

def report(name, values):
    a = np.array(values)
    if len(a) == 0:
        print(f"\n  {name}: NO DATA", flush=True)
        return
    print(f"\n  {name} (n={len(a)}):", flush=True)
    print(f"    Mean: {np.mean(a):+.1f}px  Median: {np.median(a):+.1f}px", flush=True)
    print(f"    Std:  {np.std(a):.1f}px", flush=True)
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{pct:2d}: {np.percentile(a, pct):+.1f}px", flush=True)
    # What fraction is positive (past the reference)?
    frac_positive = np.mean(a > 0)
    print(f"    Fraction > 0: {frac_positive:.1%}", flush=True)

report("INSIDE GT REACHES: offset from slit_x", inside_reach_offsets_slit)
report("INSIDE GT REACHES: offset from boxr_x", inside_reach_offsets_boxr)
report("OUTSIDE GT REACHES (nose engaged): offset from slit_x", outside_reach_offsets_slit)
report("OUTSIDE GT REACHES (nose engaged): offset from boxr_x", outside_reach_offsets_boxr)

print(f"\n{'='*70}", flush=True)
print(f"  BOUNDARY BEHAVIOR", flush=True)
print(f"{'='*70}", flush=True)

report("First 2 frames of GT reaches: offset from boxr_x", first_2_frames_offsets_boxr)
report("Last 2 frames of GT reaches: offset from slit_x", last_2_frames_offsets_slit)

print(f"\n{'='*70}", flush=True)
print(f"  SHORT REACH ANALYSIS (4-10 frames)", flush=True)
print(f"{'='*70}", flush=True)

report("Short GT reach frame offsets from boxr_x", short_gt_offsets_boxr)
if short_gt_past_ratios:
    a = np.array(short_gt_past_ratios)
    print(f"\n  Past-slit ratio distribution for short GT reaches (n={len(a)}):", flush=True)
    print(f"    Mean: {np.mean(a):.2f}  Median: {np.median(a):.2f}", flush=True)
    for pct in [5, 10, 25, 50]:
        print(f"    P{pct:2d}: {np.percentile(a, pct):.2f}", flush=True)
    below_50 = np.mean(a < 0.50)
    below_30 = np.mean(a < 0.30)
    below_20 = np.mean(a < 0.20)
    print(f"    Would be REMOVED at 50% threshold: {below_50:.1%} of short GT reaches", flush=True)
    print(f"    Would be REMOVED at 30% threshold: {below_30:.1%} of short GT reaches", flush=True)
    print(f"    Would be REMOVED at 20% threshold: {below_20:.1%} of short GT reaches", flush=True)

# Key insight: what's the slit width?
print(f"\n{'='*70}", flush=True)
print(f"  SLIT GEOMETRY", flush=True)
print(f"{'='*70}", flush=True)

# Compute slit widths across all segments
slit_widths = []
for gt_file in gt_files:
    with open(gt_file) as f:
        gt_data = json.load(f)
    video_name = gt_data.get("video_name", gt_file.stem.replace("_unified_ground_truth", ""))
    seg_file = processing_dir / f"{video_name}_segments.json"
    if not seg_file.exists():
        continue
    dlc_files = list(processing_dir.glob(f"{video_name}*DLC*.h5"))
    if not dlc_files:
        continue
    df = load_dlc(dlc_files[0])
    with open(seg_file) as f:
        seg_data2 = json.load(f)
    bounds = seg_data2.get('boundaries', [])
    for i in range(0, len(bounds) - 1, 2):
        seg_df = df.iloc[bounds[i]:bounds[i+1]]
        if len(seg_df) > 0:
            w = seg_df['BOXR_x'].median() - seg_df['BOXL_x'].median()
            slit_widths.append(w)

if slit_widths:
    a = np.array(slit_widths)
    print(f"  Slit width (BOXR_x - BOXL_x): mean={np.mean(a):.1f}px, std={np.std(a):.1f}px", flush=True)
    print(f"  Half-slit (slit_x to boxr_x): {np.mean(a)/2:.1f}px", flush=True)

print(f"\nDone.", flush=True)
