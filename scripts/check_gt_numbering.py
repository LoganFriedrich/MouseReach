"""Check whether each GT JSON uses pellet-numbering (segment_num = N
means pellet N = algo seg N+1) or algo-segment-numbering."""
import json
from pathlib import Path

QUARANTINE = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")

print(f"{'video':<35s} | total | in_segN | in_segN+1 | mismatch?")
print("-" * 90)

for gt_path in sorted(QUARANTINE.joinpath("gt").glob("*_unified_ground_truth.json")):
    vid = gt_path.stem.replace("_unified_ground_truth", "")
    seg_path = QUARANTINE / "algo_outputs" / f"{vid}_segments.json"
    if not seg_path.exists():
        continue
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    algo = json.loads(seg_path.read_text(encoding="utf-8"))
    b = algo["boundaries"]
    bounds = {i+1: (b[i], b[i+1]-1) for i in range(len(b)-1)}
    in_segN = 0
    in_segNp1 = 0
    total = 0
    for s in gt.get("outcomes", {}).get("segments", []) or []:
        ifr = s.get("interaction_frame")
        if ifr is None:
            continue
        sn = s.get("segment_num")
        total += 1
        if sn in bounds and bounds[sn][0] <= ifr <= bounds[sn][1]:
            in_segN += 1
        if (sn+1) in bounds and bounds[sn+1][0] <= ifr <= bounds[sn+1][1]:
            in_segNp1 += 1
    mismatch = "PELLET-NUMBERING" if in_segNp1 > in_segN else ("OK" if in_segN >= in_segNp1 else "?")
    print(f"{vid:<35s} | {total:5d} | {in_segN:7d} | {in_segNp1:9d} | {mismatch}")
