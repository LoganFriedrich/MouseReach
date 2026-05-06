"""For each video in quarantine, compare algo segmentation boundaries
to GT segmentation boundaries and flag any with material drift.

Drift = max abs(algo[i] - gt[i]) > DRIFT_THRESHOLD frames at any
matching index.

Reports each video as one of:
  AGREE     -- all boundaries within tolerance
  DRIFT     -- at least one boundary off by > DRIFT_THRESHOLD
  COUNT_DIFF-- algo and GT have different number of boundaries
"""
import json
from pathlib import Path

QUARANTINE = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")
DRIFT_THRESHOLD = 50  # frames

results = []
for gt_path in sorted(QUARANTINE.joinpath("gt").glob("*_unified_ground_truth.json")):
    vid = gt_path.stem.replace("_unified_ground_truth", "")
    seg_path = QUARANTINE / "algo_outputs" / f"{vid}_segments.json"
    if not seg_path.exists():
        continue
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    algo = json.loads(seg_path.read_text(encoding="utf-8"))
    algo_b = algo.get("boundaries", []) or []
    gt_b = [bd["frame"] for bd in gt.get("segmentation", {}).get("boundaries", [])]
    if not gt_b:
        results.append((vid, "NO_GT_SEG", 0, 0, 0))
        continue
    if len(algo_b) != len(gt_b):
        results.append((vid, "COUNT_DIFF", len(algo_b), len(gt_b),
                        max(abs(a - g) for a, g in zip(algo_b, gt_b))))
        continue
    deltas = [abs(a - g) for a, g in zip(algo_b, gt_b)]
    max_delta = max(deltas)
    n_drift = sum(1 for d in deltas if d > DRIFT_THRESHOLD)
    if max_delta > DRIFT_THRESHOLD:
        results.append((vid, "DRIFT", len(algo_b), len(gt_b), max_delta, n_drift))
    else:
        results.append((vid, "AGREE", len(algo_b), len(gt_b), max_delta))

print(f"{'video':<35s} | {'verdict':<10s} | algo | gt | max_delta | n_drift")
print("-" * 90)
for r in results:
    if r[1] == "DRIFT":
        print(f"{r[0]:<35s} | {r[1]:<10s} | {r[2]:>4d} | {r[3]:>2d} | {r[4]:>9d} | {r[5]}")
    elif r[1] == "COUNT_DIFF":
        print(f"{r[0]:<35s} | {r[1]:<10s} | {r[2]:>4d} | {r[3]:>2d} | {r[4]:>9d} | -")
    else:
        print(f"{r[0]:<35s} | {r[1]:<10s} | {r[2]:>4d} | {r[3]:>2d} | {r[4]:>9d} | -")

print()
print("Summary:")
verdicts = {}
for r in results:
    verdicts[r[1]] = verdicts.get(r[1], 0) + 1
for v, n in sorted(verdicts.items()):
    print(f"  {v}: {n}")

print()
print("Videos with DRIFT or COUNT_DIFF (candidates for exclusion):")
for r in results:
    if r[1] in ("DRIFT", "COUNT_DIFF"):
        print(f"  {r[0]}")
