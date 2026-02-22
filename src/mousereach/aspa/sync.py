"""
mousereach.aspa.sync - Sync mousereach results from reprocessed videos into ASPA.db.

After reprocessing, results land in:
    Analyzed/ASPA/{cohort}/   (on NAS)

For each video, reads:
    {video_id}_reaches.json
    {video_id}_features.json           (kinematics - optional)
    {video_id}_pellet_outcomes.json    (outcomes per reach)

and inserts rows into mousereach_reaches.

CLI:
    mousereach-aspa-sync [--cohort H] [--all]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from mousereach.aspa.database import ensure_tables, get_connection, get_db_path


# ---------------------------------------------------------------------------
# JSON loader helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Any]:
    """Load JSON file, returning None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"    [!] Could not read {path.name}: {exc}")
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-video sync
# ---------------------------------------------------------------------------

def _extract_meta(reaches_data: dict) -> dict:
    """Pull version / scorer fields from reaches JSON top-level metadata."""
    meta = reaches_data if isinstance(reaches_data, dict) else {}
    return {
        "mousereach_version":      meta.get("mousereach_version"),
        "dlc_scorer":              meta.get("dlc_scorer"),
        "segmenter_version":       meta.get("segmenter_version"),
        "reach_detector_version":  meta.get("reach_detector_version"),
        "processed_by":            meta.get("processed_by"),
    }


def _build_reach_rows(
    video_id: str,
    cohort: str,
    animal_id: str,
    reaches_data: Optional[dict],
    outcomes_data: Optional[dict],
    features_data: Optional[dict],
    outcome_detector_version: Optional[str],
) -> List[dict]:
    """Build list of reach row dicts for mousereach_reaches table.

    Expected reaches_data structure (flexible - handles list or dict-of-segments):
        {
            "segments": [
                {
                    "segment_num": 1,
                    "reaches": [
                        {"reach_num": 1, "start_frame": 100, "end_frame": 200, "apex_frame": 150, ...},
                        ...
                    ]
                }
            ]
        }
    OR flat list at top level.
    """
    if reaches_data is None:
        return []

    meta = _extract_meta(reaches_data)

    # Build outcome lookup: (segment_num, reach_num) -> outcome
    outcome_lookup: Dict[Tuple, str] = {}
    if outcomes_data and isinstance(outcomes_data, dict):
        od_version = outcomes_data.get("outcome_detector_version", outcome_detector_version)
        outcome_detector_version = od_version or outcome_detector_version
        for seg in outcomes_data.get("segments", []):
            seg_num = seg.get("segment_num")
            for r in seg.get("reaches", []):
                key = (seg_num, r.get("reach_num"))
                outcome_lookup[key] = r.get("outcome")
    elif outcomes_data and isinstance(outcomes_data, list):
        for r in outcomes_data:
            key = (r.get("segment_num"), r.get("reach_num"))
            outcome_lookup[key] = r.get("outcome")

    # Build kinematics lookup: (segment_num, reach_num) -> feature dict
    feature_lookup: Dict[Tuple, dict] = {}
    if features_data and isinstance(features_data, dict):
        for seg in features_data.get("segments", []):
            seg_num = seg.get("segment_num")
            for r in seg.get("reaches", []):
                key = (seg_num, r.get("reach_num"))
                feature_lookup[key] = r
    elif features_data and isinstance(features_data, list):
        for r in features_data:
            key = (r.get("segment_num"), r.get("reach_num"))
            feature_lookup[key] = r

    rows = []
    segments = reaches_data.get("segments", [])
    if not segments and isinstance(reaches_data, list):
        # Flat list of reaches
        segments = [{"segment_num": None, "reaches": reaches_data}]

    for seg in segments:
        seg_num = seg.get("segment_num")
        for r in seg.get("reaches", []):
            reach_num = r.get("reach_num")
            key = (seg_num, reach_num)
            feats = feature_lookup.get(key, {})

            rows.append({
                "video_id":                 video_id,
                "cohort":                   cohort,
                "animal_id":                animal_id,
                "segment_num":              _safe_int(seg_num),
                "reach_num":                _safe_int(reach_num),
                "start_frame":              _safe_int(r.get("start_frame")),
                "end_frame":                _safe_int(r.get("end_frame")),
                "apex_frame":               _safe_int(r.get("apex_frame")),
                "duration_frames":          _safe_int(r.get("duration_frames")),
                "outcome":                  outcome_lookup.get(key, r.get("outcome")),
                "max_extent_mm":            _safe_float(feats.get("max_extent_mm",    r.get("max_extent_mm"))),
                "velocity_at_apex":         _safe_float(feats.get("velocity_at_apex", r.get("velocity_at_apex"))),
                "trajectory_straightness":  _safe_float(feats.get("trajectory_straightness", r.get("trajectory_straightness"))),
                "mousereach_version":       meta.get("mousereach_version"),
                "dlc_scorer":               meta.get("dlc_scorer"),
                "segmenter_version":        meta.get("segmenter_version"),
                "reach_detector_version":   meta.get("reach_detector_version"),
                "outcome_detector_version": outcome_detector_version,
                "processed_by":             meta.get("processed_by"),
            })

    return rows


def sync_video(
    video_dir: Path,
    video_id: str,
    cohort: str,
    animal_id: str,
    db_path: Path,
    dry_run: bool = False,
) -> int:
    """Sync one video's mousereach results into ASPA.db.

    Looks for:
        video_dir/{video_id}_reaches.json
        video_dir/{video_id}_pellet_outcomes.json   (optional)
        video_dir/{video_id}_features.json          (optional)

    Returns:
        Number of reach rows inserted (or 0 in dry_run).
    """
    reaches_path  = video_dir / f"{video_id}_reaches.json"
    outcomes_path = video_dir / f"{video_id}_pellet_outcomes.json"
    features_path = video_dir / f"{video_id}_features.json"

    if not reaches_path.exists():
        # Try without video_id prefix - maybe flat directory
        reaches_path = next(video_dir.glob("*_reaches.json"), None)
        if reaches_path is None:
            return 0

    reaches_data  = _load_json(reaches_path)
    outcomes_data = _load_json(outcomes_path) if outcomes_path.exists() else None
    features_data = _load_json(features_path) if features_path.exists() else None

    od_version = None
    if outcomes_data and isinstance(outcomes_data, dict):
        od_version = outcomes_data.get("outcome_detector_version")

    rows = _build_reach_rows(
        video_id, cohort, animal_id,
        reaches_data, outcomes_data, features_data,
        od_version,
    )

    if not rows:
        return 0

    if dry_run:
        print(f"    [DRY-RUN] Would insert {len(rows)} reaches for {video_id}")
        return len(rows)

    conn = get_connection(db_path)
    try:
        with conn:
            # Upsert video record
            conn.execute(
                """
                INSERT INTO videos (video_id, cohort, animal_id, has_mousereach_results)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(video_id) DO UPDATE SET
                    has_mousereach_results = 1,
                    cohort    = excluded.cohort,
                    animal_id = excluded.animal_id
                """,
                (video_id, cohort, animal_id),
            )

            # Delete existing mousereach_reaches for this video to allow re-sync
            conn.execute("DELETE FROM mousereach_reaches WHERE video_id = ?", (video_id,))

            conn.executemany(
                """
                INSERT INTO mousereach_reaches
                    (video_id, cohort, animal_id, segment_num, reach_num,
                     start_frame, end_frame, apex_frame, duration_frames, outcome,
                     max_extent_mm, velocity_at_apex, trajectory_straightness,
                     mousereach_version, dlc_scorer, segmenter_version,
                     reach_detector_version, outcome_detector_version, processed_by)
                VALUES
                    (:video_id, :cohort, :animal_id, :segment_num, :reach_num,
                     :start_frame, :end_frame, :apex_frame, :duration_frames, :outcome,
                     :max_extent_mm, :velocity_at_apex, :trajectory_straightness,
                     :mousereach_version, :dlc_scorer, :segmenter_version,
                     :reach_detector_version, :outcome_detector_version, :processed_by)
                """,
                rows,
            )
    finally:
        conn.close()

    return len(rows)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _animal_id_from_video_id(video_id: str) -> str:
    """Best-effort extract animal_id from video_id."""
    parts = video_id.split("_")
    for part in parts:
        if re.match(r"^[A-Za-z]+\d+", part):
            return part
    return "UNKNOWN"


def find_reprocessed_videos(
    nas_root: Path,
    cohort: Optional[str] = None,
) -> Iterator[Tuple[str, str, str, Path]]:
    """Yield (cohort, video_id, animal_id, video_dir) for reprocessed ASPA videos.

    Scans: nas_root/Analyzed/ASPA/{cohort}/{video_id}/
           OR flat: nas_root/Analyzed/ASPA/{cohort}/*.json patterns

    Args:
        nas_root: Root of NAS output tree.
        cohort:   If given, restrict to that cohort only.
    """
    aspa_dir = nas_root / "Analyzed" / "ASPA"
    if not aspa_dir.exists():
        print(f"[!] ASPA results directory not found: {aspa_dir}")
        return

    if cohort:
        cohort_dirs = [aspa_dir / cohort]
    else:
        cohort_dirs = sorted(d for d in aspa_dir.iterdir() if d.is_dir())

    for cohort_dir in cohort_dirs:
        if not cohort_dir.is_dir():
            continue
        cohort_name = cohort_dir.name

        # Check for per-video subdirectories
        subdirs = [d for d in cohort_dir.iterdir() if d.is_dir()]
        if subdirs:
            for video_subdir in sorted(subdirs):
                video_id   = video_subdir.name
                animal_id  = _animal_id_from_video_id(video_id)
                yield cohort_name, video_id, animal_id, video_subdir
        else:
            # Flat directory: find all _reaches.json files
            for reaches_json in sorted(cohort_dir.glob("*_reaches.json")):
                video_id  = reaches_json.stem.replace("_reaches", "")
                animal_id = _animal_id_from_video_id(video_id)
                yield cohort_name, video_id, animal_id, cohort_dir


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync mousereach reprocessing results into ASPA.db."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cohort", metavar="COHORT",
                       help="Sync single cohort (e.g. H)")
    group.add_argument("--all", action="store_true",
                       help="Sync all cohorts found under Analyzed/ASPA/")

    parser.add_argument("--db-path", metavar="PATH",
                        help="Override ASPA.db path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse results but do not write to database")

    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else get_db_path()

    if not args.dry_run:
        ensure_tables(db_path)

    from mousereach.config import require_nas_drive
    try:
        nas_root = require_nas_drive() / "! DLC Output"
    except Exception as e:
        print(f"[FAIL] Could not resolve NAS root: {e}")
        sys.exit(1)

    cohort_filter = args.cohort if not args.all else None

    total_videos = 0
    total_rows   = 0
    errors       = 0

    for cohort, video_id, animal_id, video_dir in find_reprocessed_videos(
        nas_root, cohort_filter
    ):
        try:
            n = sync_video(
                video_dir, video_id, cohort, animal_id,
                db_path=db_path, dry_run=args.dry_run,
            )
            if n > 0:
                action = "Would insert" if args.dry_run else "Synced"
                print(f"  [OK] {cohort}/{video_id}: {action} {n} reaches")
                total_videos += 1
                total_rows   += n
            else:
                print(f"  [!] {cohort}/{video_id}: no reaches found")
        except Exception as exc:
            print(f"  [FAIL] {cohort}/{video_id}: {exc}")
            errors += 1

    print(f"\nDone. Videos: {total_videos}, Reach rows: {total_rows}, Errors: {errors}")
    if args.dry_run:
        print("(dry-run: no data written)")


if __name__ == "__main__":
    main()
