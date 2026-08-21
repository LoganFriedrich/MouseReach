"""Give already-segmented videos back the candidate timepoints that were discarded.

WHY THIS EXISTS
---------------
The segmenter proposes candidate tray advances from four tray corners, merges
them by agreement, keeps 21, and used to throw the rest away. It now keeps them,
because they are exactly what a person needs in order to correct a segmentation:
the alternatives the algorithm had. But every video segmented before that change
has no record of them, so the segmentation fixer has nothing to show.

This recovers them WITHOUT re-segmenting. It re-runs the proposers on the same
pose file to regenerate the candidate list, then writes that list beside the
video's EXISTING boundaries, leaving those boundaries exactly as they are.

WHY THE BOUNDARIES MUST NOT MOVE
--------------------------------
Human reviews are anchored to segments. Re-cutting a video that already has
reviews re-aims them at different footage -- the exact failure the frame-span
matching was added to prevent, and one that is only prevented when the frames
stay put. Reviews in progress right now are anchored to these boundaries. So the
boundaries are preserved verbatim and only the candidate list, the used/unused
marking, and the needs_human verdict are added.

If the fresh run disagrees with the stored boundaries, that disagreement is
recorded rather than acted on: it is a fact worth knowing, not a licence to
rewrite the file.

USAGE
-----
  python backfill_segmentation_candidates.py                 # report only
  python backfill_segmentation_candidates.py --apply
  python backfill_segmentation_candidates.py --queue-dir <dir>

ASCII-only console output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ARCHIVE_BASE = Path(r"Y:/2_Connectome/Behavior/MouseReach_Pipeline/_archived")

# The segmenter merges candidates within this many frames, so anything closer
# than this to a stored boundary is the same tray advance.
SAME_BOUNDARY_FRAMES = 30


def find_pose(bundle: Path, stem: str, analyzed: Path) -> Optional[Path]:
    from mousereach.pipeline.manifest import select_pose_file
    hits = [p for p in bundle.glob(f"{stem}DLC*.h5") if p.is_file()]
    if not hits:
        try:
            man = json.loads((bundle / f"{stem}_manifest.json").read_text())
            c = man.get("canonical_dlc_h5_path")
            if c and Path(c).is_file():
                return Path(c)
        except Exception:
            pass
    if not hits:
        hits = [p for p in analyzed.rglob(f"{stem}DLC*.h5") if p.is_file()]
    return select_pose_file(hits) if hits else None


def verdict_for(candidates: List[dict], boundaries: List[int],
                methods: List[str], ref_quality: str) -> List[str]:
    """Why this segmentation wants a person, judged against the STORED cuts.

    Deliberately not the fresh run's own verdict: that describes boundaries this
    video does not have. The counts of invented and discarded boundaries are not
    recoverable for old files, so this is the recoverable subset and errs toward
    saying nothing rather than inventing a reason.
    """
    import numpy as np

    why = []
    # Interval structure, judged against the STORED cuts. The tray advances on a
    # fixed cadence, so a missed advance leaves a segment about twice as long as
    # its neighbours -- and that is the failure that shifts every segment number
    # after it. Unused candidates are deliberately NOT a reason: the proposers
    # over-propose, and selecting 21 out of 60 is the algorithm working.
    lengths = list(np.diff(sorted(boundaries))) if len(boundaries) >= 5 else []
    if lengths:
        med = float(np.median(lengths))
        if med > 0:
            n_long = sum(1 for L in lengths if L >= med * 1.6)
            n_short = sum(1 for L in lengths if L <= med * 0.5)
            if n_long:
                why.append(f"{n_long} segment(s) run about twice the usual "
                           f"length, which is what a missed tray advance looks like")
            if n_short:
                why.append(f"{n_short} segment(s) are less than half the usual "
                           f"length, which is what an extra cut looks like")
            half = len(lengths) // 2
            if half >= 2:
                a = float(np.median(lengths[:half]))
                b = float(np.median(lengths[half:]))
                if a > 0 and b > 0 and (max(a, b) / min(a, b)) >= 1.35:
                    why.append(f"the cadence changes through the video "
                               f"({a:.0f} frames early vs {b:.0f} late)")
    n_not_detected = sum(1 for m in (methods or [])
                         if m in ("interpolated", "fallback"))
    if n_not_detected:
        why.append(f"{n_not_detected} boundary(ies) were interpolated or fell "
                   f"back rather than being detected")
    if ref_quality and ref_quality != "good":
        why.append(f"reference tracking quality is {ref_quality}")
    return why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Recover the candidate timepoints for videos segmented "
                    "before they were recorded, without moving any boundary.")
    ap.add_argument("--queue-dir", type=Path, default=None,
                    help="Bundles to repair (default: the deep-review queue)")
    ap.add_argument("--analyzed", type=Path,
                    default=Path(r"Y:/2_Connectome/Behavior/MouseReach_Pipeline/Analyzed"))
    ap.add_argument("--apply", action="store_true",
                    help="Actually write. Without this, nothing is modified.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from mousereach.config import Paths
    from mousereach.segmentation.core.segmenter_multi import segment_video_multi

    queue = args.queue_dir or getattr(Paths, "DEEP_REVIEW", None)
    if queue is None:
        print("[FAIL] no deep-review queue configured; pass --queue-dir")
        return 1
    queue = Path(queue)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = ARCHIVE_BASE / ("segments_before_candidate_backfill_" + stamp)

    print("Queue   : %s" % queue)
    print("Mode    : %s" % ("APPLY -- archiving to %s" % archive if args.apply
                            else "DRY RUN, nothing will be written"))
    print()

    tally = Counter()
    routed = []
    bundles = [b for b in sorted(queue.iterdir()) if b.is_dir()]
    for i, bundle in enumerate(bundles):
        if args.limit and tally["examined"] >= args.limit:
            break
        stem = bundle.name
        seg_path = bundle / f"{stem}_segments.json"
        if not seg_path.is_file():
            tally["no segmentation file"] += 1
            continue
        try:
            seg = json.loads(seg_path.read_text())
        except Exception:
            tally["unreadable"] += 1
            continue
        if seg.get("candidates"):
            tally["already has candidates"] += 1
            continue

        pose = find_pose(bundle, stem, args.analyzed)
        if pose is None:
            tally["no pose file to re-derive from"] += 1
            continue

        tally["examined"] += 1
        try:
            fresh_b, diag = segment_video_multi(pose)
        except Exception as e:
            print("  [!] %s: %s" % (stem, e))
            tally["segmenter errored"] += 1
            continue

        stored = sorted(int(b) for b in (seg.get("boundaries") or []))
        cands = []
        for c in (diag.candidates or []):
            f = int(c["frame"])
            cands.append({**c,
                          "used": any(abs(f - b) <= SAME_BOUNDARY_FRAMES
                                      for b in stored)})
        n_unused = sum(1 for c in cands if not c["used"])
        why = verdict_for(cands, stored, seg.get("boundary_methods"),
                          str(seg.get("reference_quality") or ""))

        moved = sum(1 for a, b in zip(sorted(fresh_b), stored) if a != b) \
            if len(fresh_b) == len(stored) else None
        if why:
            routed.append((stem, n_unused, len(why)))
            tally["would route to a person" if not args.apply else "routed to a person"] += 1
        else:
            tally["boundaries look found, not forced"] += 1

        if not args.apply:
            continue

        try:
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seg_path, archive / seg_path.name)
        except OSError as e:
            print("  [!] could not archive %s (%s); skipped" % (stem, e))
            tally["archive failed -- skipped"] += 1
            continue

        seg["candidates"] = cands
        seg["needs_human"] = why
        # Recorded, never acted on: the boundaries stay exactly as they were
        # because reviews are anchored to them.
        seg["candidates_backfilled_at"] = datetime.now().isoformat()
        seg["candidates_backfill_note"] = (
            "Candidates re-derived from the same pose file. Boundaries were NOT "
            "changed. A fresh run of the current segmenter produced "
            + ("%d boundary(ies), %s differing from the stored ones."
               % (len(fresh_b), "none" if moved == 0 else str(moved))
               if moved is not None else
               "%d boundary(ies) against %d stored." % (len(fresh_b), len(stored))))
        tmp = seg_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(seg, indent=2))
        os.replace(tmp, seg_path)
        tally["written"] += 1

    print("RESULT")
    print("------")
    for k, n in tally.most_common():
        print("  %5d  %s" % (n, k))
    if routed:
        routed.sort(key=lambda r: -r[1])
        print("\nvideos a person should look at (worst first):")
        for stem, n_unused, n_reasons in routed[:15]:
            print("   %-30s %3d unused candidates" % (stem, n_unused))
        if len(routed) > 15:
            print("   ... and %d more" % (len(routed) - 15))
    if not args.apply:
        print("\nRe-run with --apply to write these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
