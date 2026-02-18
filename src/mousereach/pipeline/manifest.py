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
import socket
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mousereach

MANIFEST_VERSION = "1.0"


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
            dlc_path = h5_files[0]

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

    # Read algorithm versions from output files
    seg_version = read_version_from_json(seg_path, 'segmenter_version')
    reach_version = read_version_from_json(reach_path, 'detector_version')
    outcome_version = read_version_from_json(outcome_path, 'detector_version')

    manifest = {
        'manifest_version': MANIFEST_VERSION,
        'video_id': video_id,
        'created_at': datetime.now().isoformat(),
        'processed_by': socket.gethostname(),

        # DLC model provenance
        'dlc_model': dlc_info,

        # Algorithm versions
        'pipeline_versions': {
            'mousereach': mousereach.__version__,
            'segmenter': seg_version,
            'reach_detector': reach_version,
            'outcome_detector': outcome_version,
        },

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

    return manifest


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
