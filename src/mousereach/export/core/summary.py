#!/usr/bin/env python3
"""
Summary statistics - Generate aggregate statistics

Counts each video's resolved result (``*_features.json``: the algorithm's calls
with every person's answer applied), never the algorithm's own
``*_pellet_outcomes.json`` (export.resolved says why).
"""

from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


def compile_results(input_dir: Path) -> Dict:
    """
    Compile all results from a directory.

    Returns dict with:
    - videos: List of video results (each video's features file)
    - summary: Aggregate statistics
    - metadata: Compilation info, including ``left_out``: the videos with
      algorithm files but no features file, each with the reason
    """
    from mousereach.export.resolved import load_resolved_videos
    videos, left_out = load_resolved_videos(Path(input_dir), pellets_only=True)

    def _count(outcomes):
        return sum(sum(1 for s in v.get('segments', []) if s.get('outcome') in outcomes)
                   for v in videos)

    # Lower-case pellet words, as every outcome file holds them. (The counts
    # once compared against 'RETRIEVED', 'DISPLACED' and 'MISSED', which no
    # file holds, so every count was 0.) 'displaced' counts both kinds;
    # 'missed' is the untouched pellet.
    total_segments = sum(len(v.get('segments', [])) for v in videos)
    total_retrieved = _count(('retrieved',))
    total_displaced = _count(('displaced_sa', 'displaced_outside'))
    total_missed = _count(('untouched',))

    return {
        'videos': videos,
        'summary': {
            'n_videos': len(videos),
            'total_segments': total_segments,
            'total_retrieved': total_retrieved,
            'total_displaced': total_displaced,
            'total_missed': total_missed,
            'overall_success_rate': total_retrieved / total_segments if total_segments > 0 else 0
        },
        'metadata': {
            'compiled_at': datetime.now().isoformat(),
            'source_dir': str(input_dir),
            'source': "features files (every person's answer applied)",
            'left_out': left_out,
        }
    }


def generate_summary(input_dir: Path, output_path: Optional[Path] = None) -> Dict:
    """Generate summary statistics and optionally save."""
    results = compile_results(input_dir)

    if output_path:
        with open(output_path, 'w') as f:
            # Don't include full video data in summary file
            summary_data = {
                'summary': results['summary'],
                'metadata': results['metadata'],
                'video_names': [v.get('video_name') for v in results['videos']]
            }
            json.dump(summary_data, f, indent=2)

    return results
