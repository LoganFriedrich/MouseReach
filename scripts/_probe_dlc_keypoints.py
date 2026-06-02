"""One-off: enumerate keypoint columns in a holdout DLC h5 to confirm
naming for nose, pellet, slit, etc. before building the feature probe."""
from __future__ import annotations

import sys
from pathlib import Path

_Y_SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
if _Y_SRC not in sys.path:
    sys.path.insert(0, _Y_SRC)

from mousereach.reach.core.geometry import load_dlc

DLC_PATH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11"
    r"\dlc\20250806_CNT0316_P3DLC_resnet50_MPSAOct27shuffle1_100000.h5"
)


def main():
    dlc = load_dlc(DLC_PATH)
    cols = sorted(dlc.columns.tolist())
    print(f"Total columns: {len(cols)}")
    # Group by keypoint prefix
    from collections import defaultdict
    bp_suffixes = defaultdict(list)
    for c in cols:
        # Split into bp_suffix
        if "_" in c:
            parts = c.rsplit("_", 1)
            bp_suffixes[parts[0]].append(parts[1])
    print(f"Bodyparts ({len(bp_suffixes)}):")
    for bp, sufs in sorted(bp_suffixes.items()):
        print(f"  {bp:25s} suffixes: {sufs}")


if __name__ == "__main__":
    main()
