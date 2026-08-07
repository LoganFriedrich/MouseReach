"""
batch.py - Batch processing logic for pellet outcome detection
"""

from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional

from .pellet_outcome import PelletOutcomeDetector


def get_associated_files(input_dir: Path, video_name: str) -> List[Path]:
    """Get ALL files associated with a video (everything with video_name prefix)."""
    files = []
    for f in input_dir.iterdir():
        if f.is_file() and f.name.startswith(video_name):
            files.append(f)
    return files


def move_to_folder(files: List[Path], dest_folder: Path, verbose: bool = True):
    """Move files to destination folder."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = dest_folder / f.name
        if f.exists() and f != dest:
            shutil.move(str(f), str(dest))


def find_file_sets(input_dir: Path, skip_if_exists: Optional[List[str]] = None) -> List[Dict]:
    """
    Find matching DLC, segment, and reach files.

    Args:
        input_dir: Directory to search
        skip_if_exists: List of file patterns - skip videos that have matching files.
                       Any glob pattern (e.g., "*outcome_ground_truth.json").
                       Extracts video names from matched files and skips those videos.
    """
    # Find videos to skip based on glob patterns
    skip_video_names = set()
    if skip_if_exists:
        for pattern in skip_if_exists:
            for matched_file in input_dir.glob(pattern):
                # Extract video name from matched file
                # Handle various naming conventions
                stem = matched_file.stem
                # Remove common suffixes to get video name
                for suffix in ['_outcome_ground_truth', '_seg_ground_truth', '_pellet_outcomes',
                              '_reaches', '_segments_v2', '_segments', '_seg_validation']:
                    if stem.endswith(suffix):
                        video_name = stem[:-len(suffix)]
                        skip_video_names.add(video_name)
                        break

    file_sets = []

    for h5_file in input_dir.glob("*DLC_*.h5"):
        video_name = h5_file.stem.split('DLC_')[0]

        # Skip if this video name is in the skip list
        if video_name in skip_video_names:
            continue

        seg_file = None
        for pattern in [f"{video_name}_seg_validation.json", f"{video_name}_segments_v2.json",
                       f"{video_name}_segments.json", f"{video_name}_seg_ground_truth.json"]:
            candidate = input_dir / pattern
            if candidate.exists():
                seg_file = candidate
                break

        reach_file = input_dir / f"{video_name}_reaches.json"
        if not reach_file.exists():
            reach_file = None

        if seg_file:
            file_sets.append({
                'video_name': video_name,
                'dlc_file': h5_file,
                'seg_file': seg_file,
                'reach_file': reach_file
            })

    return file_sets


def _extract_boundaries(seg_data):
    """Frame boundaries from the several segment-JSON shapes."""
    if "segmentation" in seg_data:
        return [int(b["frame"]) for b in seg_data["segmentation"]["boundaries"]]
    if "boundaries" in seg_data:
        return [int(b) for b in seg_data["boundaries"]]
    if "segments" in seg_data and isinstance(seg_data["segments"], list):
        bounds = set()
        for s in seg_data["segments"]:
            if "start" in s:
                bounds.add(int(s["start"]))
            if "end" in s:
                bounds.add(int(s["end"]))
        return sorted(bounds)
    raise ValueError(f"Cannot parse segment boundaries (keys: {list(seg_data.keys())})")


def _extract_reaches(reach_data):
    """(start, end) reach windows from the reach JSON.

    The reach detector nests reaches PER SEGMENT under ``segments[].reaches``
    (each reach carries ``start_frame``/``end_frame``); that is the authoritative
    current format. A flat top-level ``reaches`` list is an older/alternate form.
    Handle both. Historically this only read the flat form -- so against real
    reach JSON it returned [], starving the v6 cascade of every reach and breaking
    the reach-dependent outcome stages (e.g. retrieved-via-unique-vanish-reach),
    which collapsed retrieved->0 and inflated triage. (Fixed 2026-08; verified to
    reproduce the 2026-07-03 v6.1 LIVE per-reach Sankey outputs exactly.)
    """
    def _pair(r):
        s = r.get("start_frame")
        if s is None:
            s = r.get("start")
        e = r.get("end_frame")
        if e is None:
            e = r.get("end")
        return (int(s), int(e)) if s is not None and e is not None else None

    reaches = []
    # Nested per-segment form (current, authoritative).
    if isinstance(reach_data.get("segments"), list):
        for seg in reach_data["segments"]:
            for r in (seg.get("reaches") or []):
                p = _pair(r)
                if p is not None:
                    reaches.append(p)
    # Flat top-level form (legacy) -- only if the nested form yielded nothing.
    if not reaches and isinstance(reach_data.get("reaches"), list):
        for r in reach_data["reaches"]:
            p = _pair(r)
            if p is not None:
                reaches.append(p)
    return reaches


def _find_video_dir(search_dir: Path, video_name: str) -> Optional[Path]:
    """Directory holding the source video (for v6 Stage 98 CV checks), or None."""
    search_dir = Path(search_dir)
    for ext in (".avi", ".mp4", ".mkv"):
        if (search_dir / f"{video_name}{ext}").exists():
            return search_dir
    return None


def process_single(
    dlc_path: Path,
    seg_path: Path,
    reach_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    legacy: bool = False,
    video_dir: Optional[Path] = None,
) -> Dict:
    """Detect pellet outcomes for one video.

    Uses the v6 cascade (VERSION 6.1.0) by DEFAULT -- the current, DLC-4.0-
    calibrated detector that pipeline_versions.json declares and that the review
    tool + kinematics expect. Historically THIS entrypoint (the one the watcher /
    reprocess path calls) still ran the legacy v2.4.4-era detector even though the
    CLI had switched to v6 -- a production wiring gap that made the pipeline emit
    stale-version outcomes. v6 is now the default here too. Pass ``legacy=True``
    for the old detector. ``video_dir`` locates the source mp4 for the cascade's
    CV stage (defaults to searching ``output_dir``)."""
    if output_dir is None:
        output_dir = dlc_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if legacy:
        detector = PelletOutcomeDetector()
        results = detector.detect(dlc_path, seg_path, reach_path)
        output_path = output_dir / f"{results.video_name}_pellet_outcomes.json"
        detector.save_results(results, output_path)
        return {
            'video_name': results.video_name,
            'n_segments': results.n_segments,
            **results.summary,
            'output_file': str(output_path),
        }

    # v6 cascade (current production detector)
    from mousereach.outcomes.v6_cascade import detect_outcomes_v6_cascade
    from mousereach.reach.v8.features import load_dlc_h5

    video_id = Path(dlc_path).stem.split("DLC")[0]
    dlc_df = load_dlc_h5(dlc_path)
    seg_data = json.loads(Path(seg_path).read_text(encoding="utf-8"))
    boundaries = _extract_boundaries(seg_data)
    segments = [(boundaries[j], boundaries[j + 1] - 1)
                for j in range(len(boundaries) - 1)]
    reaches = []
    if reach_path and Path(reach_path).exists():
        reaches = _extract_reaches(
            json.loads(Path(reach_path).read_text(encoding="utf-8")))
    if video_dir is None:
        video_dir = _find_video_dir(output_dir, video_id)

    result = detect_outcomes_v6_cascade(
        dlc_df=dlc_df, segments=segments, reaches=reaches,
        video_id=video_id, video_dir=video_dir)

    output_path = output_dir / f"{video_id}_pellet_outcomes.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    counts: Dict[str, int] = {}
    for s in result.get("segments", []):
        oc = s.get("outcome")
        counts[oc] = counts.get(oc, 0) + 1
    return {
        'video_name': video_id,
        'n_segments': len(result.get("segments", [])),
        **counts,
        'output_file': str(output_path),
    }


def process_batch(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    copy_sources: bool = True,
    verbose: bool = True,
    skip_if_exists: Optional[List[str]] = None
) -> Dict:
    """
    Process all videos in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory (default: same as input)
        copy_sources: Copy source files to output
        verbose: Print progress
        skip_if_exists: List of glob patterns - skip videos with matching files
                       (e.g., ["*outcome_ground_truth.json"])
    """
    if output_dir is None:
        output_dir = input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    file_sets = find_file_sets(input_dir, skip_if_exists)
    
    if not file_sets:
        if verbose:
            print(f"No file sets found in {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0, 'videos': []}
    
    if verbose:
        print(f"Found {len(file_sets)} video(s) to process")
        print("-" * 70)
    
    results = {
        'total': len(file_sets),
        'success': 0,
        'failed': 0,
        'videos': [],
        'processed_at': datetime.now().isoformat()
    }
    
    for i, fs in enumerate(file_sets, 1):
        video_name = fs['video_name']
        
        if verbose:
            print(f"[{i}/{len(file_sets)}] {video_name}...", end=" ")
        
        try:
            video_result = process_single(
                fs['dlc_file'], fs['seg_file'], fs['reach_file'], output_dir
            )
            
            if copy_sources and output_dir != input_dir:
                shutil.copy2(fs['dlc_file'], output_dir / fs['dlc_file'].name)
                shutil.copy2(fs['seg_file'], output_dir / fs['seg_file'].name)
                if fs['reach_file']:
                    shutil.copy2(fs['reach_file'], output_dir / fs['reach_file'].name)
            
            results['success'] += 1
            results['videos'].append({'status': 'success', **video_result})
            
            if verbose:
                s = video_result
                disp = s.get('displaced_sa', 0) + s.get('displaced_outside', 0)
                print(f"OK (R={s.get('retrieved', 0)}/D={disp}/U={s.get('untouched', 0)})")
                
        except Exception as e:
            results['failed'] += 1
            results['videos'].append({'video_name': video_name, 'status': 'failed', 'error': str(e)})
            if verbose:
                print(f"FAILED: {e}")
    
    if verbose:
        print("-" * 70)
        print(f"Complete: {results['success']}/{results['total']} succeeded")
    
    summary_path = output_dir / f"batch_outcomes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
