"""Debug data loading for watchdog."""
import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_0_0")

gt_file = list(DATA_DIR.glob("*_unified_ground_truth.json"))[0]
video = gt_file.stem.replace("_unified_ground_truth", "")

print(f"Video: {video}")

# Load GT
with open(gt_file) as f:
    gt = json.load(f)

# Load algo
algo_file = ALGO_DIR / f"{video}_pellet_outcomes.json"
with open(algo_file) as f:
    algo = json.load(f)

print("\nGT outcomes structure:")
outcomes = gt.get("outcomes", {})
print(f"  n_segments: {outcomes.get('n_segments')}")
segs = outcomes.get("segments", [])
print(f"  segments: {len(segs)}")
if segs:
    s = segs[0]
    print(f"  First seg: segment_num={s.get('segment_num')}, determined={s.get('determined')}, interaction_frame={s.get('interaction_frame')}")

# Count determined
determined = [s for s in segs if s.get("determined")]
print(f"  Determined segments: {len(determined)}")

# Check which have interaction_frame
with_frame = [s for s in determined if s.get("interaction_frame") is not None]
print(f"  With interaction_frame: {len(with_frame)}")

print("\nAlgo outcomes structure:")
algo_segs = algo.get("segments", [])
print(f"  segments: {len(algo_segs)}")
if algo_segs:
    s = algo_segs[0]
    print(f"  First seg: segment_num={s.get('segment_num')}, interaction_frame={s.get('interaction_frame')}")

# Load DLC
dlc_file = list(DATA_DIR.glob(f"{video}DLC*.h5"))[0]
dlc_df = pd.read_hdf(dlc_file)
print(f"\nDLC shape: {dlc_df.shape}")

# Check frame range
print(f"Frame range: 0 to {len(dlc_df)-1}")

# Check if interaction frames are in range
for s in with_frame[:3]:
    frame = s.get("interaction_frame")
    print(f"  Segment {s['segment_num']} interaction_frame: {frame} (in range: {0 <= frame < len(dlc_df)})")
