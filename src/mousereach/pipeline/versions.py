"""
mousereach.pipeline.versions - Pipeline version tracking and crystallization.

Manages the pipeline_versions.json file on NAS that declares which tool versions
are considered "current". Provides comparison functions to detect outdated
archived videos and crystallization functions to lock videos against reprocessing.

The pipeline_versions.json file lives at:
    {NAS_ROOT}/pipeline_versions.json

Each archived video has a _processing_manifest.json recording the tool versions
used to process it. The ReprocessingScanner (watcher/reprocessor.py) compares
these manifests against pipeline_versions.json to find outdated videos.
"""

import json
import socket
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VERSIONS_SCHEMA = "1.0"


def _get_versions_path(nas_root: Path = None) -> Path:
    """Get path to pipeline_versions.json on NAS."""
    if nas_root is None:
        from mousereach.config import Paths
        nas_root = Paths.NAS_ROOT
    if nas_root is None:
        raise ValueError("NAS_ROOT not configured")
    return Path(nas_root) / "pipeline_versions.json"


def code_stamped_versions() -> dict:
    """The version each stage's RUNNING code stamps into its outputs, imported
    from the constants themselves. Stages whose module cannot be imported are
    simply absent (never raises)."""
    out = {}
    try:
        from mousereach.segmentation.core.segmenter_multi import SEGMENTER_VERSION
        out["segmenter"] = SEGMENTER_VERSION
    except Exception:
        pass
    try:
        from mousereach.reach.v8 import VERSION as _reach_v
        out["reach_detector"] = _reach_v
    except Exception:
        pass
    try:
        from mousereach.outcomes.v6_cascade import VERSION as _out_v
        out["outcome_detector"] = _out_v
    except Exception:
        pass
    try:
        from mousereach.assignment.v2.assign import VERSION as _asn_v
        out["assignment"] = _asn_v
    except Exception:
        pass
    try:
        from mousereach.kinematics.core.feature_extractor import FeatureExtractor
        out["kinematic_extractor"] = FeatureExtractor.VERSION
    except Exception:
        pass
    return out


def declaration_drift(declared: dict) -> list:
    """[(stage, declared, code_stamps)] wherever pipeline_versions.json disagrees
    with what the installed code actually stamps.

    WHY: this drift has gone invisible twice -- the declaration said segmenter
    2.1.3 while 2.2.3 ran (so the pellet-window gate never marked one video
    outdated), and kinematic_extractor stayed declared 2.0.0
    (so videos kept causal-reach-less features and nobody was told).
    Both times a human census found it months later. A machine can find it in
    milliseconds, so it does."""
    code = code_stamped_versions()
    versions = (declared or {}).get("versions") or declared or {}
    return [(k, versions.get(k), v) for k, v in sorted(code.items())
            if versions.get(k) is not None and str(versions.get(k)) != str(v)]


# The last declaration read successfully, per file: in memory, and on this
# machine's own disk (LAST_GOOD_FILENAME in the processing folder) so it
# survives a restart that happens during an outage.
LAST_GOOD_FILENAME = "pipeline_versions.last_good.json"
_LAST_GOOD: Dict[str, dict] = {}


def _last_good_file() -> Optional[Path]:
    try:
        from mousereach.config import require_processing_root
        return require_processing_root() / LAST_GOOD_FILENAME
    except Exception:
        return None


def _remember_good(versions_path: Path, data: dict) -> None:
    key = str(versions_path)
    if _LAST_GOOD.get(key) == data:
        return
    _LAST_GOOD[key] = data
    p = _last_good_file()
    if p is None:
        return
    try:
        p.write_text(json.dumps({"source": key,
                                 "saved_at": datetime.now().isoformat(timespec="seconds"),
                                 "data": data}, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        logger.debug(f"could not keep a local copy of the version declaration: {e}")


def _last_good(versions_path: Path) -> Optional[dict]:
    key = str(versions_path)
    if key in _LAST_GOOD:
        return _LAST_GOOD[key]
    p = _last_good_file()
    if p is None:
        return None
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
        if saved.get("source") == key and isinstance(saved.get("data"), dict):
            _LAST_GOOD[key] = saved["data"]
            return saved["data"]
    except Exception:
        pass
    return None


def get_current_versions(nas_root: Path = None, strict: bool = False) -> dict:
    """Load current pipeline versions from NAS.

    Returns dict with 'versions' key containing component versions,
    or empty dict if file doesn't exist yet.

    A SHARED DRIVE THAT IS AWAY is not "no declaration". The read has a time
    limit (shared_drive.read_text_bounded), and when the drive does not answer
    the last declaration this machine read successfully is returned instead,
    with a warning. WHY: returning {} during an outage would make a
    graphics-card node, finding no declared model, stop posing as if it were
    misconfigured. With no earlier copy to fall back on, ``strict`` callers get
    SharedDriveUnavailable (so they can wait); others get {} as before.
    """
    from mousereach.shared_drive import SharedDriveUnavailable, read_text_bounded
    versions_path = _get_versions_path(nas_root)
    try:
        text = read_text_bounded(versions_path, what="read pipeline_versions.json")
    except SharedDriveUnavailable as e:
        cached = _last_good(versions_path)
        if cached is not None:
            logger.warning(f"pipeline_versions.json cannot be read right now ({e}); "
                           f"using the copy last read on this machine")
            return cached
        if strict:
            raise
        logger.error(f"pipeline_versions.json cannot be read right now ({e}) and "
                     f"this machine has no earlier copy")
        return {}
    except Exception as e:
        logger.error(f"Failed to read pipeline_versions.json: {e}")
        return {}
    if text is None:
        logger.warning(f"pipeline_versions.json not found at {versions_path}")
        return {}
    try:
        data = json.loads(text)
    except ValueError as e:
        logger.error(f"Failed to read pipeline_versions.json: {e}")
        return {}
    if isinstance(data, dict):
        _remember_good(versions_path, data)
    return data


def update_current_versions(updates: dict, nas_root: Path = None,
                            notes: str = None) -> dict:
    """Update specific version fields in pipeline_versions.json.

    Args:
        updates: Dict of version fields to update (e.g. {'segmenter': '1.1.0'})
        nas_root: NAS root path (auto-detected if None)
        notes: Optional notes about this version bump

    Returns:
        Updated versions dict
    """
    versions_path = _get_versions_path(nas_root)

    # Load existing or create new
    if versions_path.exists():
        with open(versions_path) as f:
            data = json.load(f)
    else:
        data = {
            'schema_version': VERSIONS_SCHEMA,
            'versions': {},
        }

    # Update versions
    if 'versions' not in data:
        data['versions'] = {}
    data['versions'].update(updates)

    # Update metadata
    data['updated_at'] = datetime.now().isoformat()
    data['updated_by'] = socket.gethostname()
    if notes:
        data['notes'] = notes

    # Save
    versions_path.parent.mkdir(parents=True, exist_ok=True)
    # Write via a sibling temp file and rename: every node re-reads this
    # declaration each poll, and a reader that caught it half-written got
    # {} -- an empty scorer -- from which every model check drew the wrong
    # conclusion.
    import os as _os
    _tmp = versions_path.with_name(versions_path.name + '.tmp')
    with open(_tmp, 'w') as f:
        json.dump(data, f, indent=2)
    _os.replace(_tmp, versions_path)

    logger.info(f"Updated pipeline_versions.json: {updates}")
    return data


def initialize_versions(nas_root: Path = None) -> dict:
    """Create initial pipeline_versions.json from currently installed tools.

    Reads the current mousereach version and creates the versions file
    if it doesn't exist yet.
    """
    versions_path = _get_versions_path(nas_root)
    if versions_path.exists():
        logger.info(f"pipeline_versions.json already exists at {versions_path}")
        return get_current_versions(nas_root)

    import mousereach

    data = {
        'schema_version': VERSIONS_SCHEMA,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'updated_by': socket.gethostname(),
        'versions': {
            'mousereach': mousereach.__version__,
            'dlc_scorer': '',  # Set after first DLC run
            'segmenter': '',   # Set after first segmentation
            'reach_detector': '',  # Set after first reach detection
            'outcome_detector': '',  # Set after first outcome detection
            'kinematic_extractor': '',  # Set after first feature extraction
        },
        'notes': 'Initial version tracking - update component versions after first run',
    }

    versions_path.parent.mkdir(parents=True, exist_ok=True)
    # Write via a sibling temp file and rename: every node re-reads this
    # declaration each poll, and a reader that caught it half-written got
    # {} -- an empty scorer -- from which every model check drew the wrong
    # conclusion.
    import os as _os
    _tmp = versions_path.with_name(versions_path.name + '.tmp')
    with open(_tmp, 'w') as f:
        json.dump(data, f, indent=2)
    _os.replace(_tmp, versions_path)

    logger.info(f"Created pipeline_versions.json at {versions_path}")
    return data


def compare_manifest_to_current(manifest: dict, current: dict) -> dict:
    """Compare a video's processing manifest against current versions.

    Args:
        manifest: Per-video _processing_manifest.json content
        current: pipeline_versions.json content

    Returns:
        Dict with:
            is_current: bool - True only if every tracked stage recorded a
                version AND every one matches the declared current version
            stale_components: stages that ran at an outdated version
            unrecorded_components: stages the manifest never recorded a version
                for; these make the video uncertifiable, not merely outdated
            needs_full_reprocess: bool - True if DLC model changed (requires full re-run)
    """
    result = {
        'is_current': True,
        'stale_components': [],
        'unrecorded_components': [],
        'needs_full_reprocess': False,
        # Stages that only count as current because compatible_versions says
        # so. Callers that UNDO staleness marks (the reprocessor's two-way
        # door) must leave such rows alone: compat's contract is that the few
        # pathological videos are re-marked BY HAND, and un-marking them
        # because they compare "current" would silently cancel exactly those
        # hand-marks (it did, on its first live run).
        'compat_used': [],
    }

    current_versions = current.get('versions', {})
    if not current_versions:
        return result  # No current versions defined = everything is current

    manifest_versions = manifest.get('pipeline_versions', {})
    manifest_dlc = manifest.get('dlc_model', {})

    # Check DLC scorer
    current_dlc = current_versions.get('dlc_scorer', '')
    manifest_dlc_scorer = manifest_dlc.get('dlc_scorer', '')
    if current_dlc and manifest_dlc_scorer and current_dlc != manifest_dlc_scorer:
        result['is_current'] = False
        result['stale_components'].append('dlc')
        result['needs_full_reprocess'] = True

    # Check every stage the manifest is supposed to record. The list comes from
    # manifest.TRACKED_STAGES rather than a second copy kept here, because the
    # two copies drifted: assignment ran on every video, was in neither list, and
    # so was never checked. A stage the manifest does NOT record counts as stale
    # rather than passing silently -- a video whose provenance has a hole in it
    # cannot be certified as current at every step.
    from mousereach.pipeline.manifest import TRACKED_STAGES, NOT_A_VERSION

    # Versions declared compatible with the current one do NOT stale a video.
    # WHY: a version bump whose output is identical for non-pathological
    # videos (a bugfix that only changes broken cases, with the few affected
    # videos re-marked by hand) must not mark the whole corpus outdated. That
    # exact accident happened twice -- two declaration edits with no
    # algorithm change behind them marked a large part of the corpus stale -- and
    # 'outdated' was a one-way door, so no corrective declaration could undo
    # it. Declare compatibility in pipeline_versions.json beside 'versions':
    #   "compatible_versions": {"segmenter": ["2.2.3"]}
    compat = current.get('compatible_versions', {}) or {}

    for stage in TRACKED_STAGES:
        current_v = current_versions.get(stage, '')
        if not current_v:
            continue  # nothing declared for this stage; nothing to enforce
        manifest_v = manifest_versions.get(stage, '')
        if manifest_v in NOT_A_VERSION:
            result['is_current'] = False
            result['unrecorded_components'].append(stage)
        elif manifest_v != current_v:
            if str(manifest_v) in {str(v) for v in (compat.get(stage) or [])}:
                result['compat_used'].append(stage)
                continue
            result['is_current'] = False
            result['stale_components'].append(stage)

    # Check mousereach version (informational, not always a trigger)
    current_mr = current_versions.get('mousereach', '')
    manifest_mr = manifest_versions.get('mousereach', '')
    if current_mr and manifest_mr and current_mr != manifest_mr:
        # Only flag if other components also changed (mousereach version alone
        # doesn't necessarily mean reprocessing is needed)
        pass

    return result


def publication_lock_gate(config=None):
    """May a publication lock be set or lifted on THIS machine?

    Returns ``(allowed, message)``. ``message`` is the refusal when not
    allowed, a caveat to show when allowed with one, or None.

    WHY: a lock is a record beside the filed results that every machine
    honors (pipeline/publication_lock.py) PLUS the row of the machine that
    ran the command ('crystallized'). Videos are filed -- and re-run when
    outdated -- by the machine that analyses them, which is the processing
    server ('processing_server' mode), so that is where the row belongs and
    where locks are set and lifted. A GPU PC that only poses keeps rows that
    say 'pose handed on'. A lab whose only analyzing machine is a GPU PC with
    ``also_process`` on (no processing server) may lock there, with a note.
    An unreadable configuration refuses: the role cannot be told.
    """
    if config is None:
        try:
            from mousereach.config import WatcherConfig
            config = WatcherConfig.load()
        except Exception as e:
            return False, ("Publication locks are set on the processing server, and this "
                           "machine's role could not be read (%s). Nothing was changed." % e)
    mode = getattr(config, "mode", None)
    from mousereach.watcher.standalone import analyses_and_files
    if analyses_and_files(mode):
        # The processing server, or a single machine (whose processing half
        # holds the filed videos' rows, watcher/standalone.py).
        return True, None
    if mode == "dlc_pc" and getattr(config, "also_process", False):
        return True, ("Note: this machine analyses and files videos itself (also_process), "
                      "so the lock's row is set in its own database. The lock record beside "
                      "the filed results is what every other machine honors.")
    return False, ("Publication locks are set and lifted on the processing server, the "
                   "machine that files and re-runs videos. This machine is configured as "
                   "'%s'%s, whose database does not hold the filed videos' rows. Nothing "
                   "was changed; run the command on the "
                   "processing server." % (mode, "" if getattr(config, "also_process", False)
                                           else " (pose only)"))


def crystallize_videos(
    db,
    video_ids: List[str] = None,
    cohort: str = None,
    label: str = "unnamed_timepoint",
    crystallized_by: str = None,
    failures: Optional[list] = None,
    analyzed_root=None,
) -> int:
    """Lock videos against reprocessing.

    Each video gets a lock RECORD beside its filed results first
    (``<video>_crystallized.json``, pipeline/publication_lock.py) -- the lock
    every machine reads -- and then this machine's row goes to
    'crystallized'. WHY the record first: the row alone was known to one
    machine's database only. A video whose record cannot be written (its
    filed results are not where the archive step files them, or the write
    fails) is not locked at all, and is listed in ``failures``. A video named
    with ``video_ids`` that has no row here still gets its record.

    Args:
        db: WatcherDB instance
        video_ids: Specific videos to lock (or use cohort filter)
        cohort: Lock all archived videos matching this cohort string
        label: Human-readable label (e.g. "PAPER_2026_submission")
        crystallized_by: Username (defaults to hostname)
        failures: if given, ``(video_id, why)`` is appended for each video
            that could not be locked
        analyzed_root: the filed tree (default: the configured Analyzed)

    Returns:
        Number of videos crystallized
    """
    from mousereach.pipeline.publication_lock import write_record, LockRecordError

    def _fail(vid, why):
        logger.warning(f"Cannot crystallize {vid}: {why}")
        if failures is not None:
            failures.append((vid, why))

    if cohort and not video_ids:
        archived = db.get_videos_in_state('archived')
        video_ids = [
            v['video_id'] for v in archived
            if cohort.upper() in (v.get('animal_id', '') or '').upper()
        ]

    if not video_ids:
        logger.warning("No videos to crystallize")
        return 0

    count = 0
    by = crystallized_by or socket.gethostname()
    now = datetime.now().isoformat()

    for vid in video_ids:
        try:
            video = db.get_video(vid) if db is not None else None
            if video and video['state'] == 'crystallized':
                # Already locked here; make sure every machine can see it.
                write_record(vid, video.get('crystallized_label') or label,
                             video.get('crystallized_by') or by,
                             video.get('crystallized_at') or now,
                             analyzed_root=analyzed_root)
                continue
            if video and video['state'] not in ('archived', 'outdated'):
                _fail(vid, f"state={video['state']}, need archived/outdated")
                continue
            try:
                write_record(vid, label, by, now, analyzed_root=analyzed_root)
            except LockRecordError as e:
                _fail(vid, str(e))
                continue
            if video:
                db.force_state(
                    vid, 'crystallized',
                    reason=f"crystallized: locked against reprocessing ({label})",
                    crystallized_at=now,
                    crystallized_by=by,
                    crystallized_label=label,
                )
            else:
                logger.info(f"{vid}: no row on this machine; the lock record alone "
                            f"locks it")
            count += 1
            logger.info(f"Crystallized {vid} (label={label})")
        except Exception as e:
            _fail(vid, str(e))

    return count


def uncrystallize_videos(
    db,
    video_ids: List[str] = None,
    label: str = None,
    failures: Optional[list] = None,
    analyzed_root=None,
) -> int:
    """Unlock crystallized videos (allows reprocessing again).

    Removes the lock record beside the filed results first, then puts this
    machine's row back to 'archived'. A record that cannot be removed leaves
    the video locked (row included) and is listed in ``failures``. With
    ``label``, the videos are those whose row OR record carries the label.

    Args:
        db: WatcherDB instance
        video_ids: Specific videos to unlock
        label: Unlock all videos with this crystallized_label
        failures: if given, ``(video_id, why)`` is appended for each video
            that could not be unlocked
        analyzed_root: the filed tree (default: the configured Analyzed)

    Returns:
        Number of videos uncrystallized
    """
    from mousereach.pipeline.publication_lock import (
        remove_record, records_with_label, LockRecordError)

    def _fail(vid, why):
        logger.error(f"Failed to uncrystallize {vid}: {why}")
        if failures is not None:
            failures.append((vid, why))

    record_paths = {}
    if label and not video_ids:
        crystallized = db.get_videos_in_state('crystallized') if db is not None else []
        video_ids = [
            v['video_id'] for v in crystallized
            if (v.get('crystallized_label', '') or '') == label
        ]
        try:
            for vid, p in records_with_label(label, analyzed_root):
                record_paths[vid] = p
                if vid not in video_ids:
                    video_ids.append(vid)
        except Exception as e:
            _fail(label, f"the lock records could not be listed ({e})")

    if not video_ids:
        logger.warning("No videos to uncrystallize")
        return 0

    count = 0
    for vid in video_ids:
        try:
            video = db.get_video(vid) if db is not None else None
            try:
                removed = remove_record(vid, analyzed_root, path=record_paths.get(vid))
            except LockRecordError as e:
                _fail(vid, str(e))
                continue
            if video and video['state'] == 'crystallized':
                db.force_state(
                    vid, 'archived',
                    reason="uncrystallized: lock removed, back to archived",
                    unlock_crystallized=True,   # the one caller allowed to
                    crystallized_at=None,
                    crystallized_by=None,
                    crystallized_label=None,
                )
            elif not removed:
                continue                    # neither a row lock nor a record
            count += 1
            logger.info(f"Uncrystallized {vid}")
        except Exception as e:
            _fail(vid, str(e))

    return count
