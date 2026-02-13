"""
Re-run reach detection with current algorithm on all videos that have GT files.
Saves results to versioned pipeline archive for comparison.
"""

import json
import sys
from pathlib import Path

# Add MouseReach to path
sys.path.insert(0, str(Path(r"Y:\2_Connectome\Behavior\MouseReach\src")))

from mousereach.reach.core.reach_detector import ReachDetector, VERSION
from mousereach.reach.core.geometry import load_dlc, load_segments, get_boxr_reference, compute_segment_geometry

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
version_tag = VERSION.replace('.', '_').split('_')[0] + '_' + VERSION.replace('.', '_').split('_')[1]
OUTPUT_DIR = Path(rf"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_{version_tag}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running reach detection v{VERSION}")
print(f"Output: {OUTPUT_DIR}")
print()

# Find all videos that have unified GT files
gt_files = sorted(DATA_DIR.glob("*_unified_ground_truth.json"))
print(f"Found {len(gt_files)} GT files")

for gt_file in gt_files:
    with open(gt_file) as f:
        gt = json.load(f)
    video = gt['video_name']

    # Find DLC file
    dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
    if not dlc_files:
        dlc_files = list(DATA_DIR.glob(f"*{video.split('_', 1)[1]}*DLC*.h5"))
    if not dlc_files:
        print(f"  SKIP {video}: no DLC file")
        continue

    # Find segments file
    seg_files = list(DATA_DIR.glob(f"{video}_segments.json"))
    if not seg_files:
        print(f"  SKIP {video}: no segments file")
        continue

    print(f"  Processing {video}...", end=" ", flush=True)

    try:
        detector = ReachDetector()
        results = detector.detect(dlc_files[0], seg_files[0])

        # Save results
        output_path = OUTPUT_DIR / f"{video}_reaches.json"
        ReachDetector.save_results(results, output_path, validation_status="auto_approved")

        total = results.summary['total_reaches']
        print(f"OK - {total} reaches detected")

    except Exception as e:
        print(f"ERROR: {e}")

print("\nDone!")
