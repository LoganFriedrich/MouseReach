#!/usr/bin/env python3
"""
Summary statistics - Generate aggregate statistics
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime


def compile_results(input_dir: Path) -> Dict:
    """
    Compile all results from a directory.
    
    Returns dict with:
    - videos: List of video results
    - summary: Aggregate statistics
    - metadata: Compilation info
    """
    videos = []
    
    # Load pellet outcomes
    for outcome_file in sorted(input_dir.glob("*_pellet_outcomes.json")):
        with open(outcome_file) as f:
            videos.append(json.load(f))
    
    # Calculate summary
    total_segments = sum(v.get('n_segments', 0) for v in videos)
    total_retrieved = sum(
        sum(1 for s in v.get('segments', []) if s.get('outcome') == 'RETRIEVED')
        for v in videos
    )
    total_displaced = sum(
        sum(1 for s in v.get('segments', []) if s.get('outcome') == 'DISPLACED')
        for v in videos
    )
    total_missed = sum(
        sum(1 for s in v.get('segments', []) if s.get('outcome') == 'MISSED')
        for v in videos
    )
    
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
            'source_dir': str(input_dir)
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
