"""Debug feature extraction."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

video = "20250811_CNT0305_P2"
gt_file = DATA_DIR / f"{video}_unified_ground_truth.json"

with open(gt_file) as f:
    gt = json.load(f)

# Get boundaries
boundaries = [b['frame'] for b in gt.get('segmentation', {}).get('boundaries', [])]
print(f"Boundaries: {boundaries[:5]}...")

# Check segment 1
seg1 = [s for s in gt.get('outcomes', {}).get('segments', []) if s['segment_num'] == 1][0]
print(f"Segment 1: interaction_frame={seg1.get('interaction_frame')}")

# Calculate midpoint for segment 1
start = 0
end = boundaries[0] if boundaries else 1000
mid = (start + end) // 2
print(f"Segment 1 range: {start} to {end}, midpoint: {mid}")

# Load DLC
dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
print(f"DLC files: {len(dlc_files)}")
if dlc_files:
    df = pd.read_hdf(dlc_files[0])
    print(f"DLC shape: {df.shape}")
    print(f"DLC frame range: 0 to {len(df)-1}")

    # Try to get hand data at midpoint
    try:
        # The columns are multi-level: (scorer, bodypart, coord)
        print(f"Column levels: {df.columns.nlevels}")
        scorer = df.columns.get_level_values(0)[0]
        print(f"Scorer: {scorer}")

        # Get hand data - need to slice by (scorer, bodypart)
        hand_x = df[(scorer, 'RightHand', 'x')].values
        hand_y = df[(scorer, 'RightHand', 'y')].values

        window = 30
        start_idx = max(0, mid - window)
        end_idx = min(len(df) - 1, mid + window)
        print(f"Window: {start_idx} to {end_idx}")

        print(f"Hand x at midpoint: {hand_x[mid]:.1f}")
        print(f"Hand y at midpoint: {hand_y[mid]:.1f}")

        # Pellet
        pellet_x = df[(scorer, 'Pellet', 'x')].values
        pellet_y = df[(scorer, 'Pellet', 'y')].values
        pellet_like = df[(scorer, 'Pellet', 'likelihood')].values
        print(f"Pellet x at midpoint: {pellet_x[mid]:.1f}")
        print(f"Pellet y at midpoint: {pellet_y[mid]:.1f}")
        print(f"Pellet likelihood at midpoint: {pellet_like[mid]:.3f}")

        # Hand-pellet distance
        dist = np.sqrt((hand_x[mid] - pellet_x[mid])**2 + (hand_y[mid] - pellet_y[mid])**2)
        print(f"Hand-pellet distance at midpoint: {dist:.1f} pixels")

        # Key insight: pellet likelihood is very low!
        print(f"\n=== KEY INSIGHT ===")
        print(f"Pellet likelihood is only {pellet_like[mid]*100:.1f}%")
        print("DLC can't reliably track the pellet!")
        print("This makes pellet-based outcome detection unreliable.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
