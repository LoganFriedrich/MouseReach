"""
One-time patch: clean up GT tool bug where interaction_frame couldn't
be deleted when outcome was changed to untouched.

Affected: 6 segments across 5 videos. The user explicitly tried to
delete the interaction_frame and left "Cannot delete" comments.
We set interaction_frame=null for these segments and preserve audit
trail.

Updates ALL copies of each affected GT file (canonical, exhaustive
subset, and any quarantine iteration snapshots).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

CASES = [
    ("20250626_CNT0102_P4", 8),
    ("20250709_CNT0216_P3", 20),
    ("20251007_CNT0314_P3", 16),
    ("20251008_CNT0301_P4", 1),
    ("20251008_CNT0301_P4", 2),
    ("20251009_CNT0309_P1", 10),
]

CORRECTION_NOTE = (
    "Cleared by patch 2026-05-01: GT tool bug prevented user from "
    "nullifying interaction_frame when outcome was set to untouched. "
    "User comments on these segments documented the workaround. See "
    "memory entry untouched_outcome_known_frame_derivation.md."
)

# Where to look for copies of each GT file
SEARCH_ROOTS = [
    Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27"),
]


def patch_one_file(gt_path: Path, segment_num: int) -> bool:
    """Patch one GT file in-place. Returns True if a write happened."""
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    segs = gt.get("outcomes", {}).get("segments", []) or []
    target = next((s for s in segs if s.get("segment_num") == segment_num), None)
    if target is None:
        return False
    if target.get("interaction_frame") is None:
        # Already null -- no need to patch
        return False
    original = target["interaction_frame"]
    target["original_interaction_frame_before_patch"] = original
    target["interaction_frame"] = None
    # Audit trail
    target.setdefault("corrected_by", "gt_tool_bug_patch_2026-05-01")
    target.setdefault("corrected_at", datetime.now().isoformat() + "+00:00")
    existing_comment = target.get("comment", "") or ""
    if CORRECTION_NOTE not in existing_comment:
        target["comment"] = (
            existing_comment + (" | " if existing_comment else "")
            + CORRECTION_NOTE
        )
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return True


def main():
    n_patched = 0
    n_files_touched = 0
    for video_id, segment_num in CASES:
        copies = []
        for root in SEARCH_ROOTS:
            for p in root.rglob(f"{video_id}_unified_ground_truth.json"):
                copies.append(p)
        if not copies:
            print(f"  WARNING: no GT copies found for {video_id}")
            continue
        for p in copies:
            patched = patch_one_file(p, segment_num)
            if patched:
                n_patched += 1
                n_files_touched += 1
                print(f"  PATCHED {p.relative_to(SEARCH_ROOTS[0])} seg {segment_num}")
            else:
                print(f"  unchanged {p.relative_to(SEARCH_ROOTS[0])} seg {segment_num} "
                      f"(already null or segment not found)")

    print()
    print(f"Total patched writes: {n_patched}")


if __name__ == "__main__":
    main()
