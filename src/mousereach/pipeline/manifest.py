"""
manifest.py - Processing provenance manifest for MouseReach videos.

Creates a per-video _processing_manifest.json that records:
  - DLC model identity (extracted from h5 filename)
  - Algorithm versions for each pipeline step
  - Validation statuses and triage reasons
  - Processing timestamps and machine identity

This manifest ships with archived files so downstream consumers can
verify that all videos in a cohort were processed with the same
DLC model and algorithm versions.
"""

import json
import logging
import os
import socket
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mousereach

MANIFEST_VERSION = "1.0"

logger = logging.getLogger(__name__)


def extract_dlc_model_info(h5_path: Path) -> Dict[str, str]:
    """Extract DLC model identity from h5 filename.

    DLC h5 filenames follow the pattern:
        {video_id}DLC_{network}_{model}shuffle{N}_{snapshot}.h5

    Example:
        20250624_CNT0115_P2DLC_resnet50_MPSAOct27shuffle1_100000.h5

    Returns dict with:
        dlc_scorer: full DLC suffix (e.g. "DLC_resnet50_MPSAOct27shuffle1_100000")
        dlc_network: backbone network (e.g. "resnet50")
        dlc_model_name: training project name (e.g. "MPSAOct27")
        dlc_shuffle: shuffle number (e.g. 1)
        dlc_snapshot: iteration snapshot (e.g. 100000)
    """
    stem = h5_path.stem
    info = {
        'dlc_scorer': '',
        'dlc_network': '',
        'dlc_model_name': '',
        'dlc_shuffle': '',
        'dlc_snapshot': '',
    }

    if 'DLC_' not in stem:
        return info

    dlc_suffix = 'DLC_' + stem.split('DLC_')[1]
    info['dlc_scorer'] = dlc_suffix

    # Parse: DLC_{network}_{model}shuffle{N}_{snapshot}
    m = re.match(r'DLC_(\w+?)_(.+?)shuffle(\d+)_(\d+)', dlc_suffix)
    if m:
        info['dlc_network'] = m.group(1)
        info['dlc_model_name'] = m.group(2)
        info['dlc_shuffle'] = int(m.group(3))
        info['dlc_snapshot'] = int(m.group(4))

    return info


_DECLARED_SCORER: Optional[str] = None


def declared_dlc_scorer() -> str:
    """The DLC scorer pipeline_versions.json declares as current ('' if none)."""
    global _DECLARED_SCORER
    if _DECLARED_SCORER is None:
        try:
            from mousereach.pipeline.versions import get_current_versions
            _DECLARED_SCORER = (
                get_current_versions().get('versions', {}).get('dlc_scorer', '') or ''
            )
        except Exception:
            _DECLARED_SCORER = ''
    return _DECLARED_SCORER


def select_pose_file(candidates, expected_scorer: str = None) -> Optional[Path]:
    """Pick which pose file to use when a video has more than one.

    The model is part of the filename, so re-posing a video on a new model
    leaves the old pose file sitting beside the new one. A bare
    ``glob(f"{video_id}DLC*.h5")[0]`` then returns whichever the filesystem
    lists first -- which is how a video can be analyzed on one model while its
    manifest records another, and the version checker then calls it current.

    Prefers the declared model; failing that takes the newest file and says so.

    Args:
        candidates: iterable of candidate .h5 paths (a glob result is fine)
        expected_scorer: scorer to prefer; defaults to the declared one

    Returns:
        The chosen path, or None if there were no candidates.
    """
    paths = [Path(p) for p in candidates]
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]

    if expected_scorer is None:
        expected_scorer = declared_dlc_scorer()

    by_scorer: Dict[str, list] = {}
    for p in paths:
        by_scorer.setdefault(extract_dlc_model_info(p).get('dlc_scorer', ''), []).append(p)

    if expected_scorer and expected_scorer in by_scorer:
        return max(by_scorer[expected_scorer], key=lambda p: p.stat().st_mtime)

    chosen = max(paths, key=lambda p: p.stat().st_mtime)
    if len(by_scorer) > 1:
        logger.warning(
            "%s has pose from %d different models (%s) and none is the declared "
            "%s; using the newest (%s). The algorithms are calibrated per model, "
            "so this video should be re-posed.",
            paths[0].stem.split('DLC_')[0],
            len(by_scorer),
            ', '.join(sorted(s or 'unparseable' for s in by_scorer)),
            expected_scorer or '(none declared)',
            chosen.name,
        )
    return chosen


# Every pipeline stage that must be version-tracked, and the file + key where
# that stage records the version it actually ran at.
#
# This exists as ONE declaration because the alternative kept failing. The
# manifest writer used to name its stages by hand and the currency check used
# its own separate list, so a stage could exist in the pipeline and appear in
# neither. That is exactly how kinematics went untracked until 2026-08-19, and
# how assignment was still untracked after it -- every manifest ever written was
# missing it, so no video could honestly be certified as current at every step.
#
# Both create_processing_manifest() and compare_manifest_to_current() read this
# map, so adding a stage here is the only thing needed to make it tracked, and a
# stage that is not here is visibly absent rather than silently skipped.
TRACKED_STAGES = {
    'segmenter':           ("_segments.json",          "segmenter_version"),
    'reach_detector':      ("_reaches.json",           "detector_version"),
    'outcome_detector':    ("_pellet_outcomes.json",   "detector_version"),
    'assignment':          ("_reach_assignments.json", "version"),
    'kinematic_extractor': ("_features.json",          "extractor_version"),
}

# Values read_version_from_json() returns when it could not find a real version.
# A manifest carrying one of these has not recorded that stage, so the video
# cannot be certified current no matter what the string comparison says.
NOT_A_VERSION = ('not_run', 'unknown', 'error_reading', '')


def read_validation_from_json(json_path: Path) -> Dict[str, str]:
    """Read validation_status and triage_reason from an output JSON."""
    result = {'validation_status': 'not_run', 'triage_reason': ''}
    if not json_path.exists():
        return result
    try:
        with open(json_path) as f:
            data = json.load(f)
        result['validation_status'] = data.get('validation_status', 'unknown')
        result['triage_reason'] = data.get('triage_reason', '')
    except Exception:
        result['validation_status'] = 'error_reading'
    return result


def read_version_from_json(json_path: Path, key: str) -> str:
    """Read an algorithm version field from an output JSON."""
    if not json_path.exists():
        return 'not_run'
    try:
        with open(json_path) as f:
            data = json.load(f)
        # Check top-level and diagnostics
        version = data.get(key) or data.get('diagnostics', {}).get(key, '')
        return str(version) if version else 'unknown'
    except Exception:
        return 'error_reading'


def create_processing_manifest(
    video_id: str,
    processing_dir: Path,
    dlc_path: Optional[Path] = None,
    step_timestamps: Optional[Dict[str, str]] = None,
) -> Dict:
    """Create a processing manifest for a video.

    Args:
        video_id: Video identifier (e.g. "20250624_CNT0115_P2")
        processing_dir: Directory containing all output files
        dlc_path: Path to DLC h5 file (auto-detected if None)
        step_timestamps: Dict of step_name -> ISO timestamp

    Returns:
        Manifest dict (also saved to {video_id}_processing_manifest.json)
    """
    # Auto-detect DLC path
    if dlc_path is None:
        h5_files = list(processing_dir.glob(f"{video_id}DLC*.h5"))
        if h5_files:
            dlc_path = select_pose_file(h5_files)

    # DLC model info
    dlc_info = extract_dlc_model_info(dlc_path) if dlc_path else {}

    # Output file paths
    seg_path = processing_dir / f"{video_id}_segments.json"
    reach_path = processing_dir / f"{video_id}_reaches.json"
    outcome_path = processing_dir / f"{video_id}_pellet_outcomes.json"

    # Read validation statuses
    seg_val = read_validation_from_json(seg_path)
    reach_val = read_validation_from_json(reach_path)
    outcome_val = read_validation_from_json(outcome_path)

    # Read algorithm versions from output files, from the single declaration
    # above so no stage can be left out by hand.
    stage_versions = {
        stage: read_version_from_json(processing_dir / f"{video_id}{suffix}", key)
        for stage, (suffix, key) in TRACKED_STAGES.items()
    }

    manifest = {
        'manifest_version': MANIFEST_VERSION,
        'video_id': video_id,
        'created_at': datetime.now().isoformat(),
        'processed_by': socket.gethostname(),

        # DLC model provenance
        'dlc_model': dlc_info,

        # Algorithm versions
        'pipeline_versions': dict(
            {'mousereach': mousereach.__version__}, **stage_versions),

        # Validation statuses
        'validation': {
            'segmentation': seg_val,
            'reach_detection': reach_val,
            'outcome_detection': outcome_val,
        },

        # Processing timestamps (from watcher DB or caller)
        'timestamps': step_timestamps or {},
    }

    # Save manifest
    manifest_path = processing_dir / f"{video_id}_processing_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Push the same versions to the version index, so readers (the dashboard)
    # answer "is this video current?" from one query instead of re-scanning the
    # archive and re-parsing every manifest off the NAS. This is the ONLY place
    # manifests are written, so hooking here covers the watcher, reprocessing and
    # bring-current paths alike. Best-effort: a bookkeeping index must never be
    # able to fail a video's processing, and it is rebuildable from the manifests.
    try:
        from mousereach.pipeline.version_index import VersionIndex
        VersionIndex().upsert_from_manifest(video_id, manifest, manifest_path)
    except Exception:
        pass

    return manifest


def record_kinematic_version(video_id: str, processing_dir: Path, version: str) -> bool:
    """Stamp the kinematic extractor version onto a video's existing manifest.

    Both pipelines compose the manifest BEFORE running feature extraction --
    they have to, because the review gate decides whether kinematics run at all
    and a triaged video must still get a manifest. So
    read_version_from_json(features_path, ...) always found no features file and
    recorded 'not_run', on every node, for every video ever processed. Kinematics
    were the one stage whose version never reached the provenance record, which
    left them permanently invisible to compare_manifest_to_current().

    Rather than reorder the pipeline (which would break the "no kinematics until
    review is cleared" gate), the extractor pushes its own version the moment it
    succeeds -- the same writers-push model the version index already uses.

    Best-effort by the same reasoning as the index upsert: bookkeeping must never
    fail a video whose kinematics actually succeeded.

    Returns:
        True if the manifest was updated.
    """
    manifest_path = Path(processing_dir) / f"{video_id}_processing_manifest.json"
    if not manifest_path.exists():
        logger.debug("No manifest to stamp for %s", video_id)
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest.setdefault('pipeline_versions', {})['kinematic_extractor'] = version
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        try:
            from mousereach.pipeline.version_index import VersionIndex
            VersionIndex().upsert_from_manifest(video_id, manifest, manifest_path)
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning("Could not record kinematic version for %s: %s", video_id, e)
        return False


def backfill_manifest_versions(root: Path, apply: bool = False,
                               archive_dir: Optional[Path] = None,
                               stages=None) -> Dict:
    """Fill in stage versions a manifest never recorded, from the stage's own output.

    A manifest can be missing a stage's version for two reasons, and only one of
    them is a problem. If the stage never ran, 'not_run' is the correct answer
    and is left alone. If the stage DID run -- its output file is sitting right
    there -- then the manifest is simply wrong, and the output file carries the
    version that actually produced it. That value is recoverable without
    recomputing anything.

    This has happened twice now, for the same structural reason: kinematics
    (fixed 2026-08-19) and assignment (never recorded in any manifest, on any
    node, ever). So this is written once, over TRACKED_STAGES, rather than once
    per stage -- whichever stage is missed next, this repairs it.

    Originals are copied to ``archive_dir`` before anything is written. Nothing
    is ever deleted.

    Args:
        root: directory to walk (e.g. the Analyzed tree)
        apply: False (default) reports what would change and writes nothing
        archive_dir: where to copy manifests before modifying them. Required
            when apply is True.
        stages: restrict to these stage names (default: every tracked stage)

    Returns:
        Counts by outcome, and the versions found on disk per stage.
    """
    import shutil
    from collections import Counter

    stages = list(stages or TRACKED_STAGES)
    stats = Counter()
    found = {s: Counter() for s in stages}
    root = Path(root)

    if apply and archive_dir is None:
        raise ValueError("archive_dir is required when apply=True -- "
                         "manifests are archived before they are modified")

    for manifest_path in root.rglob("*_processing_manifest.json"):
        stem = manifest_path.name[: -len("_processing_manifest.json")]
        stats['manifests'] += 1
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            stats['unreadable'] += 1
            continue

        recorded = manifest.get('pipeline_versions') or {}
        fixes = {}
        for stage in stages:
            suffix, key = TRACKED_STAGES[stage]
            if recorded.get(stage, '') not in NOT_A_VERSION:
                continue  # already has a real version
            out_path = manifest_path.parent / f"{stem}{suffix}"
            if not out_path.exists():
                stats['%s: did not run -- left alone' % stage] += 1
                continue
            actual = read_version_from_json(out_path, key)
            if actual in NOT_A_VERSION:
                stats['%s: output has no version either' % stage] += 1
                continue
            found[stage][actual] += 1
            fixes[stage] = actual

        if not fixes:
            continue

        for stage in fixes:
            stats['%s: %s' % (stage, 'fixed' if apply else 'would fix')] += 1

        if not apply:
            continue

        try:
            dest = Path(archive_dir) / manifest_path.name
            Path(archive_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_path, dest)
        except Exception as e:
            logger.warning("Could not archive %s, skipping it: %s", manifest_path.name, e)
            stats['archive failed -- skipped'] += 1
            continue

        try:
            manifest.setdefault('pipeline_versions', {}).update(fixes)
            tmp = manifest_path.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp, manifest_path)
            try:
                from mousereach.pipeline.version_index import VersionIndex
                VersionIndex().upsert_from_manifest(stem, manifest, manifest_path)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Could not write %s: %s", manifest_path.name, e)
            stats['write failed'] += 1

    return {'stats': dict(stats), 'versions_found': {s: dict(c) for s, c in found.items() if c}}


def main_backfill_manifest_versions():
    """CLI: repair manifests that never recorded a stage's version.

    Reports what it would change and writes nothing unless --apply is given.
    Manifests are copied to an archive directory before being modified.
    """
    import argparse
    from datetime import datetime as _dt

    ap = argparse.ArgumentParser(
        description="Fill in pipeline stage versions a manifest never recorded, "
                    "reading each version from that stage's own output file.")
    ap.add_argument("--root", type=Path, default=None,
                    help="Directory to walk (default: the Analyzed tree)")
    ap.add_argument("--stage", action="append", dest="stages", default=None,
                    help="Repeatable. Restrict to these stages (default: all tracked).")
    ap.add_argument("--archive-dir", type=Path, default=None,
                    help="Where to copy manifests before modifying them "
                         "(default: a dated directory under the pipeline's _archived)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write. Without this, nothing is modified.")
    args = ap.parse_args()

    from mousereach.config import Paths
    root = args.root or Paths.ANALYZED_OUTPUT
    if root is None:
        print("[FAIL] No Analyzed directory configured; pass --root")
        return 1

    archive_dir = args.archive_dir
    if archive_dir is None:
        stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        base = Path(root).parent / "_archived"
        archive_dir = base / ("manifests_pre_version_backfill_%s" % stamp)

    print("Walking %s" % root)
    print("Stages : %s" % ", ".join(args.stages or list(TRACKED_STAGES)))
    if args.apply:
        print("Archive: %s" % archive_dir)
    else:
        print("DRY RUN -- nothing will be written")
    print()

    result = backfill_manifest_versions(root, apply=args.apply,
                                        archive_dir=archive_dir if args.apply else None,
                                        stages=args.stages)
    for key, n in sorted(result['stats'].items(), key=lambda kv: -kv[1]):
        print("  %6d  %s" % (n, key))
    if result['versions_found']:
        print("\nversions found in the stage output files:")
        for stage, hist in result['versions_found'].items():
            for ver, n in sorted(hist.items(), key=lambda kv: -kv[1]):
                print("  %6d  %s = %s" % (n, stage, ver))
    if not args.apply:
        print("\nRe-run with --apply to write these.")
    return 0


def backfill_kinematic_versions(root: Path, apply: bool = False) -> Dict:
    """Repair manifests that say kinematics never ran when the features exist.

    Every manifest ever written records kinematic_extractor as 'not_run',
    because the manifest is composed before feature extraction (see
    record_kinematic_version). The features files sitting next to them carry the
    version that actually produced them, so the true value is recoverable
    without re-extracting anything.

    Reads the version out of each ``*_features.json`` and stamps it onto the
    sibling manifest. Videos with no features file are left alone: 'not_run' is
    the correct answer there.

    Args:
        root: directory to walk (e.g. the Analyzed tree)
        apply: False (default) reports what would change and writes nothing

    Returns:
        Counts by outcome, plus the version histogram found on disk.
    """
    from collections import Counter

    stats = Counter()
    versions = Counter()
    root = Path(root)

    for manifest_path in root.rglob("*_processing_manifest.json"):
        stem = manifest_path.name[: -len("_processing_manifest.json")]
        stats['manifests'] += 1
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            stats['unreadable'] += 1
            continue

        current = (manifest.get('pipeline_versions') or {}).get('kinematic_extractor', '')
        features_path = manifest_path.parent / f"{stem}_features.json"
        if not features_path.exists():
            stats['no features -- not_run is correct'] += 1
            continue

        actual = read_version_from_json(features_path, 'extractor_version')
        versions[actual] += 1
        if current == actual:
            stats['already correct'] += 1
            continue

        stats['would fix' if not apply else 'fixed'] += 1
        if apply:
            manifest.setdefault('pipeline_versions', {})['kinematic_extractor'] = actual
            try:
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                try:
                    from mousereach.pipeline.version_index import VersionIndex
                    VersionIndex().upsert_from_manifest(stem, manifest, manifest_path)
                except Exception:
                    pass
            except Exception:
                stats['write failed'] += 1

    return {'counts': dict(stats), 'versions_found': dict(versions)}


def main_backfill_kinematic_versions():
    """CLI: mousereach-backfill-kinematic-versions [--apply] [--root PATH]"""
    import argparse
    from mousereach.config import Paths

    p = argparse.ArgumentParser(
        prog="mousereach-backfill-kinematic-versions",
        description=(
            "Stamp the real kinematic extractor version onto processing manifests. "
            "Manifests are composed before feature extraction runs, so every one of "
            "them records kinematic_extractor='not_run' even when kinematics "
            "completed. This reads the version from each video's _features.json and "
            "writes it to the sibling manifest, making version currency work for "
            "kinematics without re-extracting anything."
        ),
        epilog=(
            "Reports without changing anything unless --apply is given.\n"
            "Example:\n"
            "  mousereach-backfill-kinematic-versions            # report only\n"
            "  mousereach-backfill-kinematic-versions --apply    # write the fix"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--apply', action='store_true',
                   help="Write the corrections (default: report only)")
    p.add_argument('--root', type=Path, default=None,
                   help="Directory to walk (default: the Analyzed output tree)")
    args = p.parse_args()

    root = args.root or Paths.ANALYZED_OUTPUT
    if not root:
        print("ERROR: no --root given and ANALYZED_OUTPUT is not configured.")
        raise SystemExit(1)
    if not Path(root).exists():
        print(f"ERROR: root does not exist: {root}")
        raise SystemExit(1)

    print(f"{'Applying' if args.apply else 'Checking'}: {root}")
    result = backfill_kinematic_versions(root, apply=args.apply)
    print()
    for k, v in sorted(result['counts'].items(), key=lambda kv: -kv[1]):
        print(f"  {v:6d}  {k}")
    if result['versions_found']:
        print("\n  extractor versions found on disk:")
        for k, v in sorted(result['versions_found'].items(), key=lambda kv: -kv[1]):
            print(f"    {v:6d}  {k}")
    if not args.apply and result['counts'].get('would fix'):
        print("\nRe-run with --apply to write these.")


def check_provenance_consistency(
    manifests: list,
    strict: bool = False,
) -> Dict:
    """Check that a set of manifests share the same DLC model and algo versions.

    Args:
        manifests: List of manifest dicts (or paths to manifest JSONs)
        strict: If True, require exact match on all fields

    Returns:
        Dict with 'consistent' bool and 'mismatches' list
    """
    if not manifests:
        return {'consistent': True, 'mismatches': [], 'n_checked': 0}

    # Load from paths if needed
    loaded = []
    for m in manifests:
        if isinstance(m, (str, Path)):
            with open(m) as f:
                loaded.append(json.load(f))
        else:
            loaded.append(m)

    reference = loaded[0]
    mismatches = []

    for i, m in enumerate(loaded[1:], 1):
        vid = m.get('video_id', f'manifest_{i}')

        # Check DLC model
        ref_scorer = reference.get('dlc_model', {}).get('dlc_scorer', '')
        m_scorer = m.get('dlc_model', {}).get('dlc_scorer', '')
        if ref_scorer and m_scorer and ref_scorer != m_scorer:
            mismatches.append({
                'video_id': vid,
                'field': 'dlc_model',
                'expected': ref_scorer,
                'actual': m_scorer,
            })

        # Check algo versions
        ref_versions = reference.get('pipeline_versions', {})
        m_versions = m.get('pipeline_versions', {})
        for key in ('segmenter', 'reach_detector', 'outcome_detector'):
            ref_v = ref_versions.get(key, '')
            m_v = m_versions.get(key, '')
            if ref_v and m_v and ref_v != m_v:
                mismatches.append({
                    'video_id': vid,
                    'field': f'pipeline_versions.{key}',
                    'expected': ref_v,
                    'actual': m_v,
                })

        if strict:
            ref_mr = ref_versions.get('mousereach', '')
            m_mr = m_versions.get('mousereach', '')
            if ref_mr and m_mr and ref_mr != m_mr:
                mismatches.append({
                    'video_id': vid,
                    'field': 'pipeline_versions.mousereach',
                    'expected': ref_mr,
                    'actual': m_mr,
                })

    return {
        'consistent': len(mismatches) == 0,
        'mismatches': mismatches,
        'n_checked': len(loaded),
        'reference_video': reference.get('video_id', 'unknown'),
    }
