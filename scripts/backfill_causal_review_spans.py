"""Give every existing causal review its segment frame spans.

WHY THIS EXISTS
---------------
A human review is a fact about a stretch of video. "At the end of this segment
the pellet is still on the pillar" stays true forever, no matter which DLC model
or algo version runs next. That is the whole reason review files travel with the
video instead of being thrown away when the video is reprocessed.

But the records were anchored only to `segment_num` -- an index the segmenter
hands out fresh every time it re-cuts a video. Re-segment, and the review of
"segment 7" quietly starts describing a different stretch of footage. The fact
was durable; the pointer to it was not.

`build_segment_record` now stores `segment_span` ({"start": .., "end": ..}) on
every record. This script does the same for the reviews written before that
field existed, reading each review's spans out of the segmentation it was
actually made against -- the `_segments.json` sitting next to it.

WHAT IT CHECKS BEFORE WRITING
-----------------------------
The neighbouring segmentation is only usable if it is still the one the human
reviewed. Two checks, both must pass:

  1. Segment count matches between the review and the segmentation.
  2. Every reviewed record's algo causal reach starts inside the span the
     segmentation gives that segment number.

A file failing either has been re-segmented since it was reviewed. Writing spans
from the current segmentation would invent a frame range the human never saw --
exactly the failure this field exists to prevent -- so those are SKIPPED and
listed by name. They need a human to re-anchor them, or a re-review.

The segmenter version recorded in each review's provenance is reported
alongside, as corroboration.

SAFETY
------
  - Originals are copied to an archive tree before anything is modified.
  - Writes are atomic (temp file + replace), so a file is never left half
    written if the watcher moves it mid-run.
  - Records that already have a span are left alone.
  - Files that disappear mid-run (the watcher moved the video) are counted and
    reported, not treated as errors. Re-run to catch them at their new home.

USAGE
-----
  python backfill_causal_review_spans.py --dry-run     # report only, no writes
  python backfill_causal_review_spans.py               # archive, then backfill
  python backfill_causal_review_spans.py --roots D:\some\other\tree

Exit code is 0 if every discovered file was either backfilled, already done, or
explicitly reported as skipped. Non-zero only on an unexpected failure.
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
from typing import Dict, List, Optional, Tuple

SUFFIX = "_causal_review.json"

DEFAULT_ROOTS = [
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Analyzed"),
    Path(r"C:\2_Connectome\Behavior\MouseReach_Pipeline\Processing"),
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing"),
]

ARCHIVE_BASE = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\_archived")


# ---------------------------------------------------------------------------
# Reading segmentation
# ---------------------------------------------------------------------------

def segmentation_path(review_file: Path) -> Optional[Path]:
    """The _segments.json next to a review file (or the _segmentation.json
    alternate that collect_provenance also accepts)."""
    stem = str(review_file)[: -len(SUFFIX)]
    for suffix in ("_segments.json", "_segmentation.json"):
        p = Path(stem + suffix)
        if p.is_file():
            return p
    return None


def spans_from_segmentation(doc: dict) -> Dict[int, Tuple[int, int]]:
    """segment_num -> (start_frame, end_frame), whatever shape the file takes."""
    out: Dict[int, Tuple[int, int]] = {}

    segs = doc.get("segments")
    if isinstance(segs, list) and segs and isinstance(segs[0], dict):
        for s in segs:
            num = s.get("segment_num")
            start = s.get("start_frame")
            if num is None or start is None:
                continue
            out[int(num)] = (int(start), int(s.get("end_frame", start)))
        if out:
            return out

    # Fallback: a bare boundary list.
    bounds = doc.get("boundaries")
    if isinstance(bounds, list) and len(bounds) >= 2:
        frames = sorted(
            int(b.get("frame", b.get("index", 0))) if isinstance(b, dict) else int(b)
            for b in bounds
        )
        for i in range(len(frames) - 1):
            out[i + 1] = (frames[i], frames[i + 1] - 1)
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(review: dict, spans: Dict[int, Tuple[int, int]]) -> Tuple[bool, str]:
    """Is this segmentation still the one the human reviewed?

    Returns (ok, reason). The reason describes the evidence either way.
    """
    records = review.get("segments") or []
    if not records:
        return False, "review has no segment records"
    if len(records) != len(spans):
        return False, (f"segment count differs (review has {len(records)}, "
                       f"segmentation has {len(spans)}) -- re-segmented since review")

    cross_checked = 0
    for rec in records:
        num = rec.get("segment_num")
        if num is None:
            return False, "a record has no segment_num"
        span = spans.get(int(num))
        if span is None:
            return False, f"segmentation has no segment {num}"

        reach = (rec.get("algo") or {}).get("causal_reach")
        if isinstance(reach, dict) and reach.get("start") is not None:
            start = int(reach["start"])
            if not (span[0] <= start <= span[1]):
                return False, (f"segment {num}: the algo causal reach starts at frame "
                               f"{start}, outside this segmentation's span "
                               f"{span[0]}-{span[1]} -- re-segmented since review")
            cross_checked += 1

    if cross_checked:
        return True, f"verified ({cross_checked} causal reaches land inside their spans)"
    return True, "segment count matches (no causal reach available to cross-check)"


def segmenter_version(review: dict) -> str:
    prov = (review.get("provenance") or {}).get("segmenter") or {}
    return str(prov.get("version", "?"))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def apply_spans(review: dict, spans: Dict[int, Tuple[int, int]]) -> int:
    """Add segment_span to records that lack one. Returns how many were set."""
    added = 0
    for rec in review.get("segments") or []:
        if isinstance(rec.get("segment_span"), dict):
            continue  # already has one; never overwrite
        span = spans.get(int(rec["segment_num"]))
        if span is None:
            continue
        rec["segment_span"] = {"start": span[0], "end": span[1]}
        added += 1
    return added


def write_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


def archive(review_file: Path, archive_dir: Path) -> None:
    """Copy the original aside before it is modified. Never delete anything."""
    try:
        rel = review_file.relative_to(review_file.anchor)
    except ValueError:
        rel = Path(review_file.name)
    dest = archive_dir / review_file.drive.replace(":", "") / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(review_file, dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Backfill segment frame spans into existing causal review files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --dry-run first. It writes nothing and prints exactly what "
               "would change and what would be skipped.",
    )
    ap.add_argument("--roots", nargs="*", type=Path, default=None,
                    help="Directories to search (default: Analyzed + both Processing trees)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report only; make no changes")
    ap.add_argument("--archive-dir", type=Path, default=None,
                    help="Where to copy originals (default: a dated dir under "
                         "MouseReach_Pipeline/_archived)")
    args = ap.parse_args(argv)

    roots = args.roots or DEFAULT_ROOTS
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = args.archive_dir or (ARCHIVE_BASE / f"causal_review_pre_span_backfill_{stamp}")

    files: List[Path] = []
    for root in roots:
        if not Path(root).exists():
            print(f"[!] root does not exist, skipping: {root}")
            continue
        files.extend(sorted(Path(root).rglob(f"*{SUFFIX}")))

    print(f"Found {len(files)} causal review files across {len(roots)} root(s)")
    if args.dry_run:
        print("DRY RUN -- nothing will be written")
    else:
        print(f"Originals archived to: {archive_dir}")
    print()

    tally = Counter()
    spans_written = 0
    skipped: List[Tuple[Path, str]] = []
    seg_versions = Counter()

    for rf in files:
        if not rf.is_file():
            tally["moved away mid-run (re-run to catch)"] += 1
            continue

        sf = segmentation_path(rf)
        if sf is None:
            skipped.append((rf, "no segmentation file beside it"))
            tally["SKIPPED"] += 1
            continue

        try:
            review = json.loads(rf.read_text())
            spans = spans_from_segmentation(json.loads(sf.read_text()))
        except (OSError, ValueError) as e:
            skipped.append((rf, f"unreadable: {e}"))
            tally["SKIPPED"] += 1
            continue

        if not spans:
            skipped.append((rf, "could not read spans out of the segmentation file"))
            tally["SKIPPED"] += 1
            continue

        ok, reason = verify(review, spans)
        if not ok:
            skipped.append((rf, reason))
            tally["SKIPPED"] += 1
            continue

        seg_versions[segmenter_version(review)] += 1

        records = review.get("segments") or []
        already = sum(1 for r in records if isinstance(r.get("segment_span"), dict))
        if records and already == len(records):
            tally["already had spans"] += 1
            continue

        if args.dry_run:
            n = sum(1 for r in records
                    if not isinstance(r.get("segment_span"), dict)
                    and spans.get(int(r["segment_num"])) is not None)
            spans_written += n
            tally["would backfill"] += 1
            continue

        try:
            archive(rf, archive_dir)
            n = apply_spans(review, spans)
            write_atomic(rf, review)
        except OSError as e:
            skipped.append((rf, f"write failed: {e}"))
            tally["SKIPPED"] += 1
            continue

        spans_written += n
        tally["backfilled"] += 1

    print("RESULT")
    print("------")
    for key, n in tally.most_common():
        print(f"  {n:5d}  {key}")
    label = "that would be written" if args.dry_run else "written"
    print(f"  {spans_written:5d}  segment spans {label}")

    if seg_versions:
        print("\nSegmenter version recorded in the backfilled reviews:")
        for v, n in seg_versions.most_common():
            print(f"  {n:5d}  {v}")

    if skipped:
        print(f"\nSKIPPED ({len(skipped)}) -- these need a human to re-anchor or re-review:")
        for path, reason in skipped:
            print(f"  {path.name}")
            print(f"      {reason}")
    else:
        print("\nNothing skipped -- every discovered review was backfillable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
