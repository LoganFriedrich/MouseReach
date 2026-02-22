"""
mousereach.archive.migrate - One-time migration from Sort/ to project/cohort structure.

Moves:
    Analyzed/Sort/CNT/*  ->  Analyzed/Connectome/CNT01/, CNT02/, CNT03/, CNT04/
    Analyzed/Sort/Multi-Animal/*.mkv  ->  Analyzed/Connectome/{cohort}/Multi-Animal/

Usage:
    mousereach-migrate-archive              # Dry run (show what would happen)
    mousereach-migrate-archive --execute    # Actually move files
"""

import argparse
import sqlite3
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mousereach.config import Paths, AnimalID, get_video_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_animal_id_from_video_id(video_id: str) -> Optional[str]:
    """Extract animal_id from a standard single-animal video_id.

    Format: YYYYMMDD_ANIMALID_TRAY[POS]
    Example: 20250704_CNT0101_P1 -> CNT0101

    Returns None if the format is not recognised.
    """
    parts = video_id.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def _parse_first_animal_from_mkv_stem(stem: str) -> Optional[str]:
    """Extract the first animal_id from a multi-animal MKV filename stem.

    Format: YYYYMMDD_CNT0101,CNT0205,..._P1
    Returns the first animal_id string (e.g. 'CNT0101'), or None.
    """
    # Split on underscores; the second part is the comma-separated animal list
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    animal_field = parts[1]
    first = animal_field.split(",")[0].strip()
    return first if first else None


def _get_destination_for_video_id(
    video_id: str,
    nas_root: Path,
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve the destination directory for a single-animal video_id.

    Returns (dest_dir, reason_if_unknown).
    """
    animal_id = _parse_animal_id_from_video_id(video_id)
    if not animal_id:
        return None, f"Cannot parse animal_id from video_id: {video_id}"

    try:
        project, cohort = AnimalID.get_project_and_cohort(animal_id)
    except Exception as exc:
        return None, f"AnimalID.get_project_and_cohort failed for {animal_id!r}: {exc}"

    if project == "UNKNOWN" or cohort == "UNKNOWN":
        return None, f"Unknown project/cohort for animal_id {animal_id!r}"

    dest = nas_root / "Analyzed" / project / cohort
    return dest, None


def _get_destination_for_mkv(
    stem: str,
    nas_root: Path,
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve the destination directory for a multi-animal MKV.

    MKVs land in Analyzed/<project>/<cohort>/Multi-Animal/.
    Returns (dest_dir, reason_if_unknown).
    """
    first_animal = _parse_first_animal_from_mkv_stem(stem)
    if not first_animal:
        return None, f"Cannot parse first animal from MKV stem: {stem!r}"

    try:
        project, cohort = AnimalID.get_project_and_cohort(first_animal)
    except Exception as exc:
        return None, f"AnimalID.get_project_and_cohort failed for {first_animal!r}: {exc}"

    if project == "UNKNOWN" or cohort == "UNKNOWN":
        return None, f"Unknown project/cohort for animal_id {first_animal!r}"

    dest = nas_root / "Analyzed" / project / cohort / "Multi-Animal"
    return dest, None


# ---------------------------------------------------------------------------
# Discovery: Sort/CNT/
# ---------------------------------------------------------------------------

def _discover_cnt_groups(
    sort_cnt: Path,
) -> Tuple[Dict[str, List[Path]], List[Tuple[Path, str]]]:
    """Walk Sort/CNT/ and group files by video_id.

    Returns:
        groups: {video_id: [file, ...]}
        unknowns: [(file, reason)]
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    unknowns: List[Tuple[Path, str]] = []

    if not sort_cnt.exists():
        return groups, unknowns

    for f in sort_cnt.iterdir():
        if not f.is_file():
            continue

        try:
            vid = get_video_id(f.name)
        except Exception as exc:
            unknowns.append((f, f"get_video_id raised: {exc}"))
            continue

        if not vid or vid == f.stem:
            # get_video_id returned the stem unchanged - check if it looks like a video_id
            # A valid video_id has at least DATE_ANIMALID structure
            if re.match(r"^\d{8}_[A-Za-z]+\d+", vid or ""):
                groups[vid].append(f)
            else:
                unknowns.append((f, f"video_id does not match expected pattern: {vid!r}"))
        else:
            groups[vid].append(f)

    return groups, unknowns


# ---------------------------------------------------------------------------
# Discovery: Sort/Multi-Animal/
# ---------------------------------------------------------------------------

def _discover_multi_animal_groups(
    sort_multi: Path,
) -> Tuple[Dict[str, List[Path]], List[Tuple[Path, str]]]:
    """Walk Sort/Multi-Animal/ and group MKV files by stem.

    Returns:
        groups: {stem: [file, ...]}  (usually just one .mkv per stem)
        unknowns: [(file, reason)]
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    unknowns: List[Tuple[Path, str]] = []

    if not sort_multi.exists():
        return groups, unknowns

    for f in sort_multi.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() != ".mkv":
            # Non-MKV files in Multi-Animal - treat as unknown
            unknowns.append((f, "Non-MKV file in Multi-Animal folder"))
            continue
        groups[f.stem].append(f)

    return groups, unknowns


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_cnt_plan(
    groups: Dict[str, List[Path]],
    unknowns: List[Tuple[Path, str]],
    nas_root: Path,
    execute: bool,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Print the migration plan for Sort/CNT/ files.

    Returns a mapping of {cohort_label: [(src, dest), ...]} for execution.
    """
    plan_by_cohort: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    unknown_plan: List[Tuple[Path, str]] = list(unknowns)

    for video_id, files in sorted(groups.items()):
        dest_dir, reason = _get_destination_for_video_id(video_id, nas_root)
        if dest_dir is None:
            for f in files:
                unknown_plan.append((f, reason or "Unknown destination"))
            continue

        cohort_label = dest_dir.name  # e.g. 'CNT01'
        for f in files:
            plan_by_cohort[cohort_label].append((f, dest_dir / f.name))

    # --- Print summary ---
    print("=" * 70)
    print("Sort/CNT/ migration plan")
    print("=" * 70)

    total_files = sum(len(v) for v in plan_by_cohort.values())
    print(f"  Total files: {total_files}")
    print(f"  Total groups (video IDs): {len(groups)}")
    print()

    for cohort_label in sorted(plan_by_cohort.keys()):
        moves = plan_by_cohort[cohort_label]
        print(f"  Cohort {cohort_label}: {len(moves)} file(s)")
        for src, dst in moves:
            rel_src = src.name
            rel_dst = str(dst.relative_to(nas_root)) if nas_root in dst.parents else str(dst)
            print(f"    {rel_src}")
            print(f"      -> {rel_dst}")
        print()

    if unknown_plan:
        print(f"  UNKNOWN / unparseable: {len(unknown_plan)} file(s)")
        unknown_dest = nas_root / "Analyzed" / "UNKNOWN"
        for f, reason in unknown_plan:
            print(f"    {f.name}  [{reason}]")
            print(f"      -> UNKNOWN/")
        print()

    return plan_by_cohort, unknown_plan


def _print_multi_animal_plan(
    groups: Dict[str, List[Path]],
    unknowns: List[Tuple[Path, str]],
    nas_root: Path,
    execute: bool,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Print the migration plan for Sort/Multi-Animal/ files."""
    plan_by_cohort: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    unknown_plan: List[Tuple[Path, str]] = list(unknowns)

    for stem, files in sorted(groups.items()):
        dest_dir, reason = _get_destination_for_mkv(stem, nas_root)
        if dest_dir is None:
            for f in files:
                unknown_plan.append((f, reason or "Unknown destination"))
            continue

        # cohort label is the parent of Multi-Animal
        cohort_label = dest_dir.parent.name
        for f in files:
            plan_by_cohort[cohort_label].append((f, dest_dir / f.name))

    print("=" * 70)
    print("Sort/Multi-Animal/ migration plan")
    print("=" * 70)

    total_files = sum(len(v) for v in plan_by_cohort.values())
    print(f"  Total files: {total_files}")
    print(f"  Total MKV groups: {len(groups)}")
    print()

    for cohort_label in sorted(plan_by_cohort.keys()):
        moves = plan_by_cohort[cohort_label]
        print(f"  Cohort {cohort_label}: {len(moves)} MKV(s)")
        for src, dst in moves:
            rel_src = src.name
            rel_dst = str(dst.relative_to(nas_root)) if nas_root in dst.parents else str(dst)
            print(f"    {rel_src}")
            print(f"      -> {rel_dst}")
        print()

    if unknown_plan:
        print(f"  UNKNOWN / unparseable: {len(unknown_plan)} file(s)")
        for f, reason in unknown_plan:
            print(f"    {f.name}  [{reason}]")
            print(f"      -> UNKNOWN/Multi-Animal/")
        print()

    return plan_by_cohort, unknown_plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _execute_moves(
    plan_by_cohort: Dict[str, List[Tuple[Path, Path]]],
    unknown_plan: List[Tuple[Path, str]],
    nas_root: Path,
    label: str,
) -> Tuple[int, int]:
    """Move files according to plan. Returns (moved, failed)."""
    moved = 0
    failed = 0

    for cohort_label, moves in plan_by_cohort.items():
        for src, dst in moves:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                print(f"  [OK] {src.name} -> {dst.parent.name}/{dst.name}")
                moved += 1
            except Exception as exc:
                print(f"  [FAIL] {src.name}: {exc}")
                failed += 1

    # Move unknowns to UNKNOWN/
    unknown_dir = nas_root / "Analyzed" / "UNKNOWN"
    if label == "Multi-Animal":
        unknown_dir = nas_root / "Analyzed" / "UNKNOWN" / "Multi-Animal"

    for f, reason in unknown_plan:
        try:
            unknown_dir.mkdir(parents=True, exist_ok=True)
            dst = unknown_dir / f.name
            shutil.move(str(f), str(dst))
            print(f"  [OK] {f.name} -> UNKNOWN/ (reason: {reason})")
            moved += 1
        except Exception as exc:
            print(f"  [FAIL] {f.name}: {exc}")
            failed += 1

    return moved, failed


# ---------------------------------------------------------------------------
# Watcher DB path updates
# ---------------------------------------------------------------------------

def _update_watcher_db_paths(
    nas_root: Path,
    dry_run: bool,
) -> int:
    """Update watcher.db entries for archived videos whose current_path is
    under Sort/CNT/.

    Finds all rows where:
      state = 'archived'  AND  current_path LIKE '%Sort/CNT%'

    For each such row, re-derives the new current_path from the video_id
    using get_project_and_cohort, then updates (or prints, in dry-run mode).

    Returns the number of rows updated (or that would be updated).
    """
    from mousereach.config import PROCESSING_ROOT

    if PROCESSING_ROOT is None:
        print("  [!] PROCESSING_ROOT not configured - cannot locate watcher.db")
        return 0

    db_path = Path(PROCESSING_ROOT) / "watcher.db"
    if not db_path.exists():
        # Also check WatcherConfig db_path override
        try:
            from mousereach.config import WatcherConfig
            watcher_cfg = WatcherConfig.load()
            if watcher_cfg.db_path and watcher_cfg.db_path.exists():
                db_path = watcher_cfg.db_path
            else:
                print(f"  [!] watcher.db not found at {db_path}")
                return 0
        except Exception:
            print(f"  [!] watcher.db not found at {db_path}")
            return 0

    print(f"  Checking watcher.db at: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(
            """
            SELECT id, video_id, current_path
            FROM videos
            WHERE state = 'archived'
              AND current_path LIKE '%Sort/CNT%'
            """
        )
        rows = cursor.fetchall()
    except Exception as exc:
        print(f"  [!] Failed to query watcher.db: {exc}")
        conn.close()
        return 0

    if not rows:
        print("  No watcher.db rows need path updates.")
        conn.close()
        return 0

    print(f"  Found {len(rows)} archived row(s) with old Sort/CNT paths")
    updated = 0

    for row in rows:
        row_id = row["id"]
        video_id = row["video_id"]
        old_path = row["current_path"]

        # Determine new path
        dest_dir, reason = _get_destination_for_video_id(video_id, nas_root)
        if dest_dir is None:
            print(f"    [!] {video_id}: cannot resolve new path - {reason}")
            continue

        # Reconstruct filename from old_path (take the filename part)
        old_filename = Path(old_path).name if old_path else f"{video_id}.mp4"
        new_path = str(dest_dir / old_filename)

        if dry_run:
            print(f"    [DRY RUN] {video_id}:")
            print(f"      old: {old_path}")
            print(f"      new: {new_path}")
        else:
            try:
                conn.execute(
                    "UPDATE videos SET current_path = ?, updated_at = datetime('now') "
                    "WHERE id = ?",
                    (new_path, row_id)
                )
                conn.commit()
                print(f"    [OK] {video_id}: {old_path} -> {new_path}")
                updated += 1
            except Exception as exc:
                print(f"    [FAIL] {video_id}: {exc}")

    conn.close()

    if dry_run:
        print(f"  (dry run) Would update {len(rows)} row(s)")
        return len(rows)

    print(f"  Updated {updated} row(s) in watcher.db")
    return updated


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for mousereach-migrate-archive."""
    parser = argparse.ArgumentParser(
        prog="mousereach-migrate-archive",
        description=(
            "One-time migration from Sort/ to project/cohort archive structure.\n\n"
            "Old structure:\n"
            "  Analyzed/Sort/CNT/             (all CNT videos flat)\n"
            "  Analyzed/Sort/Multi-Animal/    (collage MKVs)\n\n"
            "New structure:\n"
            "  Analyzed/Connectome/CNT01/\n"
            "  Analyzed/Connectome/CNT01/Multi-Animal/\n"
            "  Analyzed/Connectome/CNT02/\n"
            "  ...\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mousereach-migrate-archive              # Dry run - show plan\n"
            "  mousereach-migrate-archive --execute    # Actually move files\n"
            "  mousereach-migrate-archive --nas-root X:/DLC_Output\n"
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would happen without moving files (default)",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually move files and update watcher.db",
    )

    parser.add_argument(
        "--nas-root",
        metavar="PATH",
        default=None,
        help="Override NAS root path (default: from config)",
    )

    args = parser.parse_args()

    # --execute takes precedence; otherwise default is dry-run
    execute = args.execute
    dry_run = not execute

    # --- Resolve NAS root ---
    if args.nas_root:
        nas_root = Path(args.nas_root)
    else:
        nas_root = Paths.NAS_ROOT

    if nas_root is None:
        print("[FAIL] NAS root is not configured.")
        print("       Run 'mousereach-setup' or pass --nas-root PATH")
        raise SystemExit(1)

    nas_root = Path(nas_root)
    if not nas_root.exists():
        print(f"[FAIL] NAS root does not exist: {nas_root}")
        raise SystemExit(1)

    sort_cnt = nas_root / "Analyzed" / "Sort" / "CNT"
    sort_multi = nas_root / "Analyzed" / "Sort" / "Multi-Animal"

    print()
    print("MouseReach Archive Migration")
    print("=" * 70)
    print(f"  NAS root    : {nas_root}")
    print(f"  Source (CNT): {sort_cnt}")
    print(f"  Source (MA) : {sort_multi}")
    print(f"  Mode        : {'EXECUTE (files will be moved)' if execute else 'DRY RUN (no changes)'}")
    print()

    # --- Discover CNT files ---
    cnt_groups, cnt_unknowns = _discover_cnt_groups(sort_cnt)
    cnt_plan, cnt_unknown_plan = _print_cnt_plan(cnt_groups, cnt_unknowns, nas_root, execute)

    # --- Discover Multi-Animal files ---
    ma_groups, ma_unknowns = _discover_multi_animal_groups(sort_multi)
    ma_plan, ma_unknown_plan = _print_multi_animal_plan(ma_groups, ma_unknowns, nas_root, execute)

    # --- Summary ---
    total_cnt_files = sum(len(v) for v in cnt_plan.values()) + len(cnt_unknown_plan)
    total_ma_files = sum(len(v) for v in ma_plan.values()) + len(ma_unknown_plan)
    total_files = total_cnt_files + total_ma_files

    print("=" * 70)
    print(f"Total files to migrate: {total_files}")
    print(f"  Sort/CNT/          : {total_cnt_files}")
    print(f"  Sort/Multi-Animal/ : {total_ma_files}")
    print()

    if dry_run:
        print("[DRY RUN] No files moved. Run with --execute to perform migration.")
        print()
        print("Watcher DB paths that would be updated:")
        _update_watcher_db_paths(nas_root, dry_run=True)
        print()
        return

    # --- Execute ---
    print("Executing migration...")
    print()

    print("Moving Sort/CNT/ files:")
    cnt_moved, cnt_failed = _execute_moves(cnt_plan, cnt_unknown_plan, nas_root, label="CNT")
    print()

    print("Moving Sort/Multi-Animal/ files:")
    ma_moved, ma_failed = _execute_moves(ma_plan, ma_unknown_plan, nas_root, label="Multi-Animal")
    print()

    print("Updating watcher.db paths:")
    _update_watcher_db_paths(nas_root, dry_run=False)
    print()

    # --- Final summary ---
    print("=" * 70)
    print("Migration complete")
    print(f"  Sort/CNT/          : {cnt_moved} moved, {cnt_failed} failed")
    print(f"  Sort/Multi-Animal/ : {ma_moved} moved, {ma_failed} failed")
    total_moved = cnt_moved + ma_moved
    total_failed = cnt_failed + ma_failed
    print(f"  Total              : {total_moved} moved, {total_failed} failed")
    if total_failed:
        print(f"  [!] {total_failed} file(s) failed to move - check output above")
    print()


if __name__ == "__main__":
    main()
