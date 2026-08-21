"""Correct the segmenter version recorded against already-processed videos.

WHAT WAS WRONG
--------------
Segmentation is done by ``segment_video_multi`` in segmenter_multi.py, but the
file was written by ``save_segmentation`` in segmenter_robust.py, which stamped
its OWN module constant. The two constants disagreed: segmenter_multi was on
2.2.3 while segmenter_robust said 2.1.3. So every _segments.json on disk claims
2.1.3 no matter which segmenter produced it, and pipeline_versions.json declared
2.1.3 to match -- the wrong one.

The consequence is not cosmetic. The pellet-window gate shipped in segmenter
2.2, changing how boundaries are cut, and no video was ever marked outdated for
it, because the recorded version never moved.

WHY THE CORRECTION IS SAFE TO MAKE
----------------------------------
Segmenter 2.2.3 landed on 2026-07-08. Every finished, current video in the
corpus was segmented after that date, so all of them were produced by 2.2.3 and
mislabelled. Rewriting the stamp to 2.2.3 records what actually ran; it does not
change any boundary, and it means declaring 2.2.3 marks nothing stale.

Files written from now on stamp the running version themselves, plus a
``segmented_at`` timestamp -- this correction had to fall back to file dates
because segmentation output carried no timestamp of its own.

WHAT IT TOUCHES
---------------
Two files per video, both archived before being modified:
  {video}_segments.json            segmenter_version, plus a provenance note
  {video}_processing_manifest.json pipeline_versions.segmenter

Both are needed: the segments file is the record of what ran, the manifest is
what the currency check actually compares against.

A file is only corrected if it currently claims the OLD version AND was produced
after the new version's release date. Anything else is left alone and reported.

USAGE
-----
  python backfill_segmenter_version.py --dry-run
  python backfill_segmenter_version.py --apply
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

ANALYZED = Path(r"Y:/2_Connectome/Behavior/MouseReach_Pipeline/Analyzed")
ARCHIVE_BASE = Path(r"Y:/2_Connectome/Behavior/MouseReach_Pipeline/_archived")

OLD_VERSION = "2.1.3"
NEW_VERSION = "2.2.3"
RELEASED = datetime(2026, 7, 8)
NOTE = ("backfilled {when}: save_segmentation stamped segmenter_robust's constant "
        "({old}) rather than the segmenter that ran; this file was produced {made}, "
        "after {new} was released {rel}")


def produced_at(seg_doc: dict, path: Path) -> datetime:
    """When this segmentation ran. Prefers a timestamp in the file; older files
    have none, so fall back to the file's own date."""
    for key in ("segmented_at", "created_at", "detected_at"):
        ts = seg_doc.get(key)
        if ts:
            try:
                return datetime.fromisoformat(str(ts)[:19])
            except ValueError:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def write_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Record the segmenter version that actually produced each "
                    "already-processed video.")
    ap.add_argument("--root", type=Path, default=ANALYZED)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write. Without this, nothing is modified.")
    ap.add_argument("--archive-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = args.archive_dir or (ARCHIVE_BASE / ("segmenter_version_backfill_" + stamp))

    tally = Counter()
    print("Walking %s" % args.root)
    print("Correcting %s -> %s for files produced after %s"
          % (OLD_VERSION, NEW_VERSION, RELEASED.date()))
    print("Archive: %s" % archive_dir if args.apply else "DRY RUN -- nothing will be written")
    print()

    for seg_path in sorted(args.root.rglob("*_segments.json")):
        stem = seg_path.name[: -len("_segments.json")]
        try:
            seg = json.loads(seg_path.read_text())
        except Exception:
            tally["unreadable segmentation"] += 1
            continue

        current = str(seg.get("segmenter_version"))
        if current == NEW_VERSION:
            tally["already correct"] += 1
            continue
        if current != OLD_VERSION:
            tally["stamped %s -- left alone" % current] += 1
            continue

        made = produced_at(seg, seg_path)
        if made < RELEASED:
            tally["produced before %s was released -- left alone" % NEW_VERSION] += 1
            continue

        tally["corrected" if args.apply else "would correct"] += 1
        if not args.apply:
            continue

        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seg_path, archive_dir / seg_path.name)
        except OSError as e:
            print("  [!] could not archive %s (%s); skipped" % (seg_path.name, e))
            tally["archive failed -- skipped"] += 1
            continue

        seg["segmenter_version"] = NEW_VERSION
        seg["segmenter_version_provenance"] = NOTE.format(
            when=stamp[:8], old=OLD_VERSION, new=NEW_VERSION,
            made=made.date(), rel=RELEASED.date())
        write_atomic(seg_path, seg)

        # The manifest is what the currency check reads, so it has to agree.
        man_path = seg_path.parent / ("%s_processing_manifest.json" % stem)
        if not man_path.is_file():
            tally["no manifest beside it"] += 1
            continue
        try:
            man = json.loads(man_path.read_text())
            if str((man.get("pipeline_versions") or {}).get("segmenter")) == OLD_VERSION:
                shutil.copy2(man_path, archive_dir / man_path.name)
                man.setdefault("pipeline_versions", {})["segmenter"] = NEW_VERSION
                write_atomic(man_path, man)
                tally["manifest corrected"] += 1
                try:
                    from mousereach.pipeline.version_index import VersionIndex
                    VersionIndex().upsert_from_manifest(stem, man, man_path)
                except Exception:
                    pass
        except Exception as e:
            print("  [!] manifest %s: %s" % (man_path.name, e))
            tally["manifest failed"] += 1

    print("RESULT")
    print("------")
    for k, n in tally.most_common():
        print("  %6d  %s" % (n, k))
    if not args.apply:
        print("\nRe-run with --apply to write these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
