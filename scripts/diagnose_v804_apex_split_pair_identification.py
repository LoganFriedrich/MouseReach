"""Can we identify apex-split-produced pairs from manifest data alone?

Each video's manifest tells us apex_fires = v803_algo_count - v802_algo_count.
If, per video, we sort consecutive algo pairs by gap (ascending) and take the
top N pairs (N = apex_fires), do those line up with the actual apex-split
fires?

For this to work, the apex-split-produced pairs must consistently have the
smallest gaps in the video. We expect:
  - apex-split pairs: gap 0-2 (apex-split splits at a trough frame, consuming
    0-2 frames)
  - natural rapid-fire reach pairs: gap distribution shifted higher

If per-video, the count of gap<=2 pairs matches the per-video apex_fires
count, then heuristic B works.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_Y_SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
if _Y_SRC not in sys.path:
    sys.path.insert(0, _Y_SRC)

HOL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\holdout_2026_05_11"
)
CAL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\calibration_loocv"
)


def main():
    print(f"{'Corpus':12s} {'Video':32s} {'apex_fires':>10s} "
          f"{'gap<=0':>7s} {'gap<=1':>7s} {'gap<=2':>7s} {'gap<=3':>7s}")
    print("-" * 90)
    matched_count = 0
    mismatched_count = 0
    for label, dirpath in (("calibration", CAL_MANIFEST_DIR), ("holdout", HOL_MANIFEST_DIR)):
        for f in sorted(dirpath.glob("*.json")):
            with open(f) as fh:
                d = json.load(fh)
            video = d["video_id"]
            apex_fires = d.get("v803_apex_splits_added", 0)
            if apex_fires == 0:
                continue

            # Collect algo events sorted by start
            algos = []
            for e in d.get("events", []):
                algo = e.get("detector")
                if algo is None:
                    continue
                algos.append((algo["start"], algo["end"]))
            algos.sort()

            # Compute pair gaps
            gaps = []
            for i in range(len(algos) - 1):
                gap = algos[i + 1][0] - algos[i][1] - 1
                gaps.append(gap)

            n_le_0 = sum(1 for g in gaps if g <= 0)
            n_le_1 = sum(1 for g in gaps if g <= 1)
            n_le_2 = sum(1 for g in gaps if g <= 2)
            n_le_3 = sum(1 for g in gaps if g <= 3)

            match_indicator = ""
            if n_le_2 == apex_fires:
                matched_count += 1
                match_indicator = "  [match @ gap<=2]"
            elif n_le_1 == apex_fires:
                matched_count += 1
                match_indicator = "  [match @ gap<=1]"
            elif n_le_3 == apex_fires:
                matched_count += 1
                match_indicator = "  [match @ gap<=3]"
            else:
                mismatched_count += 1
                match_indicator = "  [mismatch]"

            print(
                f"{label:12s} {video:32s} {apex_fires:>10d} {n_le_0:>7d} "
                f"{n_le_1:>7d} {n_le_2:>7d} {n_le_3:>7d}{match_indicator}"
            )

    print()
    print(f"Match (apex_fires == gap<=K pair count for some small K): {matched_count}")
    print(f"Mismatch: {mismatched_count}")
    print()
    if mismatched_count > 0:
        print("Mismatches mean the gap criterion alone doesn't identify apex-split-")
        print("produced pairs unambiguously -- we'd need inference re-run with tagging.")
    else:
        print("Clean match: heuristic identification works -- we can do Phase A discriminator")
        print("analysis from manifest data alone, no inference re-run needed.")


if __name__ == "__main__":
    main()
