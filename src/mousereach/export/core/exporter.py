#!/usr/bin/env python3
"""
Data export - Export MouseReach results to Excel/CSV

Reads each video's ``*_features.json`` -- its result with every person's answer
applied -- never the algorithm's ``*_pellet_outcomes.json`` (export.resolved
says why). Videos without a features file are left out and named.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional


def load_all_results(input_dir: Path, left_out: Optional[list] = None) -> List[dict]:
    """Load every video's resolved result (``*_features.json``) in a directory.

    ``left_out``, when a list, receives each video that has algorithm files
    but no readable features file, with the reason (export.resolved)."""
    from mousereach.export.resolved import load_resolved_videos
    results, skipped = load_resolved_videos(Path(input_dir), pellets_only=True)
    if left_out is not None:
        left_out.extend(skipped)
    return results


def _lab_ids(video_name):
    """(animal_id, session) as the LAB names them, for a human-facing export.

    ASPA animals are re-encoded on the way into the pipeline (J11 -> ASPA1011) so
    every stage sees one id shape. Nothing decoded on the way out, so results
    surfaced as ASPA1011 and a reader had to know the alphabet rule to get back to
    J11. These two columns are that missing half.

    video_name itself is left alone -- it is the key that ties a row back to the
    file it came from. Returns (None, video_name) for a non-ASPA video, so every
    project can go through the same call.
    """
    try:
        from mousereach.aspa import lab_animal_id, decode_video_stem
        return lab_animal_id(video_name), decode_video_stem(video_name)
    except Exception:
        return None, video_name


def results_to_dataframe(results: List[dict]) -> pd.DataFrame:
    """Convert results to a flat DataFrame."""
    rows = []
    
    for video_data in results:
        video_name = video_data.get('video_name', '')
        
        animal_id, session = _lab_ids(video_name)

        for segment in video_data.get('segments', []):
            row = {
                'video_name': video_name,
                'animal_id': animal_id,
                'session': session,
                'segment_num': segment.get('segment_num'),
                'outcome': segment.get('outcome'),
                'confidence': segment.get('outcome_confidence'),
                'n_reaches': segment.get('n_reaches', 0),
                'causal_reach_id': segment.get('causal_reach_id'),
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def export_to_excel(
    results: List[dict],
    output_path: Path,
    include_summary: bool = True
) -> Path:
    """
    Export results to Excel workbook.
    
    Creates sheets:
    - Summary: Per-video statistics
    - Details: Per-segment data
    - Reaches: Per-reach data (if available)
    """
    df_details = results_to_dataframe(results)
    
    # Summary by video. Outcome names are the pipeline's lower-case pellet
    # words (the counts once compared against 'RETRIEVED', 'DISPLACED' and
    # 'MISSED', which no file holds, so every count was 0).
    df_summary = df_details.groupby('video_name').agg({
        'segment_num': 'count',
        'outcome': lambda x: (x == 'retrieved').sum(),
        'confidence': 'mean',
        'n_reaches': 'sum'
    }).rename(columns={
        'segment_num': 'n_segments',
        'outcome': 'n_retrieved',
        'confidence': 'mean_confidence',
        'n_reaches': 'total_reaches'
    })
    
    for outcome in ['displaced_sa', 'displaced_outside', 'untouched']:
        df_summary[f'n_{outcome}'] = df_details[df_details['outcome'] == outcome].groupby('video_name').size()
    df_summary = df_summary.fillna(0)
    
    # Calculate success rate
    df_summary['success_rate'] = df_summary['n_retrieved'] / df_summary['n_segments']
    
    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary')
        df_details.to_excel(writer, sheet_name='Details', index=False)
    
    return output_path


def export_to_csv(results: List[dict], output_dir: Path) -> List[Path]:
    """Export results to CSV files."""
    df_details = results_to_dataframe(results)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    details_path = output_dir / 'mousereach_details.csv'
    df_details.to_csv(details_path, index=False)
    
    return [details_path]
