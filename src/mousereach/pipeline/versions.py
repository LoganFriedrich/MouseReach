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


def get_current_versions(nas_root: Path = None) -> dict:
    """Load current pipeline versions from NAS.

    Returns dict with 'versions' key containing component versions,
    or empty dict if file doesn't exist yet.
    """
    versions_path = _get_versions_path(nas_root)
    if not versions_path.exists():
        logger.warning(f"pipeline_versions.json not found at {versions_path}")
        return {}

    try:
        with open(versions_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read pipeline_versions.json: {e}")
        return {}


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
    with open(versions_path, 'w') as f:
        json.dump(data, f, indent=2)

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
        },
        'notes': 'Initial version tracking - update component versions after first run',
    }

    versions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(versions_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Created pipeline_versions.json at {versions_path}")
    return data


def compare_manifest_to_current(manifest: dict, current: dict) -> dict:
    """Compare a video's processing manifest against current versions.

    Args:
        manifest: Per-video _processing_manifest.json content
        current: pipeline_versions.json content

    Returns:
        Dict with:
            is_current: bool - True if all versions match
            stale_components: list of component names that are outdated
            needs_full_reprocess: bool - True if DLC model changed (requires full re-run)
    """
    result = {
        'is_current': True,
        'stale_components': [],
        'needs_full_reprocess': False,
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

    # Check pipeline component versions
    component_map = {
        'segmenter': 'segmenter',
        'reach_detector': 'reach_detector',
        'outcome_detector': 'outcome_detector',
    }

    for current_key, manifest_key in component_map.items():
        current_v = current_versions.get(current_key, '')
        manifest_v = manifest_versions.get(manifest_key, '')
        if current_v and manifest_v and current_v != manifest_v:
            result['is_current'] = False
            result['stale_components'].append(current_key)

    # Check mousereach version (informational, not always a trigger)
    current_mr = current_versions.get('mousereach', '')
    manifest_mr = manifest_versions.get('mousereach', '')
    if current_mr and manifest_mr and current_mr != manifest_mr:
        # Only flag if other components also changed (mousereach version alone
        # doesn't necessarily mean reprocessing is needed)
        pass

    return result


def crystallize_videos(
    db,
    video_ids: List[str] = None,
    cohort: str = None,
    label: str = "unnamed_timepoint",
    crystallized_by: str = None,
) -> int:
    """Lock videos against reprocessing.

    Args:
        db: WatcherDB instance
        video_ids: Specific videos to lock (or use cohort filter)
        cohort: Lock all archived videos matching this cohort string
        label: Human-readable label (e.g. "PNAS_2026_submission")
        crystallized_by: Username (defaults to hostname)

    Returns:
        Number of videos crystallized
    """
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
            video = db.get_video(vid)
            if not video:
                logger.warning(f"Video not found: {vid}")
                continue
            if video['state'] not in ('archived', 'outdated'):
                logger.warning(f"Cannot crystallize {vid} (state={video['state']}, need archived/outdated)")
                continue

            db.force_state(
                vid, 'crystallized',
                crystallized_at=now,
                crystallized_by=by,
                crystallized_label=label,
            )
            count += 1
            logger.info(f"Crystallized {vid} (label={label})")
        except Exception as e:
            logger.error(f"Failed to crystallize {vid}: {e}")

    return count


def uncrystallize_videos(
    db,
    video_ids: List[str] = None,
    label: str = None,
) -> int:
    """Unlock crystallized videos (allows reprocessing again).

    Args:
        db: WatcherDB instance
        video_ids: Specific videos to unlock
        label: Unlock all videos with this crystallized_label

    Returns:
        Number of videos uncrystallized
    """
    if label and not video_ids:
        crystallized = db.get_videos_in_state('crystallized')
        video_ids = [
            v['video_id'] for v in crystallized
            if (v.get('crystallized_label', '') or '') == label
        ]

    if not video_ids:
        logger.warning("No videos to uncrystallize")
        return 0

    count = 0
    for vid in video_ids:
        try:
            video = db.get_video(vid)
            if not video:
                continue
            if video['state'] != 'crystallized':
                continue

            db.force_state(
                vid, 'archived',
                crystallized_at=None,
                crystallized_by=None,
                crystallized_label=None,
            )
            count += 1
            logger.info(f"Uncrystallized {vid}")
        except Exception as e:
            logger.error(f"Failed to uncrystallize {vid}: {e}")

    return count
