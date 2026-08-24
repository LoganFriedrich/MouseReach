"""
mousereach.watcher.bench_disagreement - Auto-triage on manual-scoring disagreement.

The BenchDisagreementScanner checks each archived video's algorithm-determined
outcomes against the manual bench-sheet score. When a pellet the bench sheet
records as "missed" (still on its pillar) has an algorithm outcome of
"displaced" or "retrieved" -- a pellet cannot climb back onto its pillar, so
exactly one of the two is wrong -- and NO human has ever reviewed that segment
through any review tool, the whole video is routed to the triage queue so a
person can settle it.

WHY "NEVER REVIEWED" IS THE GATE, NOT "STILL DISAGREES"
----------------------------------------------------------
Once a human reviews a segment (GT, causal review, or triage), the paper bench
sheet itself never changes -- but the segment must never be re-flagged for
this reason again. That is handled for free by scoping to
``outcome_source == "algo"``: the moment any review tool answers a segment,
its outcome_source stops being "algo" permanently, so it drops out of this
scan's population on its own. No separate "already flagged" ledger is needed,
mirroring how ReprocessingScanner (`reprocessor.py`) needs no such ledger
either -- a video stops looking stale the moment it is actually current again.

WHY THIS READS A SNAPSHOT, NEVER connectome.db DIRECTLY
----------------------------------------------------------
connectome.db sits on a network share (Y:) in SQLite rollback-journal mode,
not WAL -- deliberately, because WAL is unreliable over network filesystems.
A writer blocks readers outright, and the watcher must never risk stalling
its main loop waiting on that lock, or corrupting a read that dies partway
through. This scan reads the parquet snapshot at
``C:/LAB_ROOT/_analysis_snapshot/`` instead (kept fresh by
``mousedb.exporters.refresh_snapshot``, scheduled separately, NOT run from
inside the watcher) -- the same rule every recipe in
``mousedb.recipes.manual_scoring_accuracy`` already follows.

Usage:
    # Integrated into ProcessingOrchestrator poll loop (automatic) via
    # BaseOrchestrator._maybe_bench_disagreement_scan()
    # Or run standalone:
    python -m mousereach.watcher.bench_disagreement          Report only
    python -m mousereach.watcher.bench_disagreement --route  Also route to triage
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_DIR = Path("C:/LAB_ROOT/_analysis_snapshot")


class BenchDisagreementScanner:
    """Scan archived videos for bench-vs-algorithm disagreements never reviewed
    by a human, and route them to triage."""

    def __init__(self, db, nas_root: Path, snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR):
        r"""
        Args:
            db: WatcherDB instance
            nas_root: NAS root path (e.g. Y:\LAB_ROOT\Behavior\MouseReach_Pipeline)
            snapshot_dir: directory holding reach_data.parquet / pellet_scores.parquet
        """
        self.db = db
        self.nas_root = Path(nas_root)
        self.archive_dir = self.nas_root / "Analyzed"
        self.snapshot_dir = Path(snapshot_dir)

    def scan(self, route: bool = True) -> dict:
        """Scan all archived videos, optionally route disagreements to triage.

        Args:
            route: If True, flag the offending segment(s) and move the video
                into the triage queue. If False, report only.

        Returns:
            Summary dict with counts:
                scanned: archived videos checked
                no_snapshot: snapshot missing/unreadable, scan aborted
                flagged_videos: videos with >=1 never-reviewed disagreement
                flagged_segments: total such segments across all videos
                routed: videos actually moved to triage (route=True only)
                no_bench_score: videos with no bench-sheet rows at all
                errors: per-video scan errors
        """
        summary = {
            "scanned": 0,
            "no_snapshot": False,
            "flagged_videos": 0,
            "flagged_segments": 0,
            "routed": 0,
            "no_bench_score": 0,
            "errors": 0,
            "flagged_details": [],  # [{video_id, segment_nums}]
        }

        pellets_path = self.snapshot_dir / "pellet_scores.parquet"
        reach_path = self.snapshot_dir / "reach_data.parquet"
        if not pellets_path.exists() or not reach_path.exists():
            logger.warning(
                "Bench disagreement scan: snapshot not found at %s -- skipping "
                "(run mousedb-refresh-snapshot first)", self.snapshot_dir)
            summary["no_snapshot"] = True
            return summary

        try:
            import pandas as pd
            from mousedb.recipes.manual_scoring_accuracy.blind_spot import pair_current
        except ImportError:
            logger.warning(
                "Bench disagreement scan: mousedb not importable in this "
                "environment -- skipping")
            summary["no_snapshot"] = True
            return summary

        rd = pd.read_parquet(reach_path)
        ps = pd.read_parquet(pellets_path)

        archived = {v["video_id"] for v in self.db.get_videos_in_state("archived")}
        logger.info("Bench disagreement scan: %d archived videos", len(archived))

        rd = rd[(rd.segment_num > 0) & (rd.video_name.isin(archived))]
        if rd.empty:
            return summary

        paired = pair_current(rd, ps)
        never_reviewed = paired[paired.outcome_source == "algo"]
        disagreements = never_reviewed[
            (never_reviewed.human == "missed") & (never_reviewed.algo.isin(["displaced", "retrieved"]))
        ]

        for video_id, group in disagreements.groupby("video_name"):
            summary["scanned"] += 1
            segment_nums = sorted(int(s) for s in group["segment_num"].unique())
            try:
                flagged = self._flag_segments(video_id, segment_nums)
            except Exception as e:
                summary["errors"] += 1
                logger.error("Bench disagreement scan: %s failed: %s", video_id, e)
                continue

            if not flagged:
                continue

            summary["flagged_videos"] += 1
            summary["flagged_segments"] += len(flagged)
            summary["flagged_details"].append({"video_id": video_id, "segment_nums": flagged})
            logger.info(
                "Bench disagreement: %s segments %s (bench=missed, algo says otherwise, "
                "never reviewed)", video_id, flagged)

            if route:
                try:
                    self._route(video_id)
                    summary["routed"] += 1
                except Exception as e:
                    summary["errors"] += 1
                    logger.error("Bench disagreement scan: routing %s failed: %s", video_id, e)

        return summary

    def _flag_segments(self, video_id: str, segment_nums) -> list:
        """Set flagged_for_review on the given segment numbers in
        {video_id}_pellet_outcomes.json. Returns the segment_nums actually
        found and flagged (may be fewer than requested if the file has moved
        or the segment isn't present)."""
        p = self._find_file(video_id, f"{video_id}_pellet_outcomes.json")
        if p is None:
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        flagged = []
        for s in data.get("segments", []):
            if s.get("segment_num") in segment_nums:
                s["flagged_for_review"] = True
                s["flag_reason"] = (
                    "bench sheet says missed, pipeline says the pellet moved, "
                    "never reviewed -- auto-flagged by BenchDisagreementScanner")
                s["triage_cleared"] = None
                flagged.append(s["segment_num"])
        if flagged:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return flagged

    def _route(self, video_id: str) -> Path:
        from mousereach.config import Paths
        from mousereach.watcher.review_gate import route_to_queue

        source_dir = self._find_file(video_id, f"{video_id}_pellet_outcomes.json")
        if source_dir is None:
            raise FileNotFoundError(f"{video_id}: no pellet_outcomes.json found to route from")
        return route_to_queue(
            video_id, source_dir.parent, Paths.TRIAGE_REVIEW,
            reason="bench_disagreement_never_reviewed",
            db=self.db, db_state="triage",
        )

    def _find_file(self, video_id: str, filename: str) -> Optional[Path]:
        """Locate a video's file in the archive tree. Same two-pass strategy
        as ReprocessingScanner._load_manifest: direct project/cohort glob
        first, recursive fallback second."""
        if not self.archive_dir.exists():
            return None
        for hit in self.archive_dir.glob(f"*/*/{filename}"):
            return hit
        for hit in self.archive_dir.rglob(filename):
            return hit
        return None


def main():
    import argparse
    from mousereach.config import PROCESSING_ROOT
    from mousereach.watcher.db import WatcherDB

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", action="store_true",
                    help="Also flag segments and move videos to triage (default: report only)")
    ap.add_argument("--nas-root", default=None)
    args = ap.parse_args()

    nas_root = Path(args.nas_root) if args.nas_root else Path(PROCESSING_ROOT)
    db = WatcherDB()
    scanner = BenchDisagreementScanner(db, nas_root)
    summary = scanner.scan(route=args.route)

    print(f"scanned videos with a disagreement candidate : {summary['scanned']}")
    print(f"videos with >=1 never-reviewed disagreement   : {summary['flagged_videos']}")
    print(f"segments flagged                               : {summary['flagged_segments']}")
    if args.route:
        print(f"videos routed to triage                        : {summary['routed']}")
    print(f"errors                                          : {summary['errors']}")
    if summary["no_snapshot"]:
        print("\nSnapshot missing or mousedb not importable -- nothing scanned. "
              "Run mousedb-refresh-snapshot first.")
    for item in summary["flagged_details"][:20]:
        print(f"  {item['video_id']:40s} segments {item['segment_nums']}")
    if len(summary["flagged_details"]) > 20:
        print(f"  ... and {len(summary['flagged_details']) - 20} more")


if __name__ == "__main__":
    main()
