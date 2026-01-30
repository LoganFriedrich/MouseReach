#!/usr/bin/env python3
"""
Data export - Export MouseReach results to Excel/CSV
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
import json


def load_all_results(input_dir: Path) -> List[dict]:
    """Load all outcome JSON files from a directory."""
    results = []
    
    for outcome_file in input_dir.glob("*_pellet_outcomes.json"):
        with open(outcome_file) as f:
            data = json.load(f)
            results.append(data)
    
    return results


def results_to_dataframe(results: List[dict]) -> pd.DataFrame:
    """Convert results to a flat DataFrame."""
    rows = []
    
    for video_data in results:
        video_name = video_data.get('video_name', '')
        
        for segment in video_data.get('segments', []):
            row = {
                'video_name': video_name,
                'segment_num': segment.get('segment_num'),
                'outcome': segment.get('outcome'),
                'confidence': segment.get('confidence'),
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
    
    # Summary by video
    df_summary = df_details.groupby('video_name').agg({
        'segment_num': 'count',
        'outcome': lambda x: (x == 'RETRIEVED').sum(),
        'confidence': 'mean',
        'n_reaches': 'sum'
    }).rename(columns={
        'segment_num': 'n_segments',
        'outcome': 'n_retrieved',
        'confidence': 'mean_confidence',
        'n_reaches': 'total_reaches'
    })
    
    # Add displaced, missed counts
    for outcome in ['DISPLACED', 'MISSED']:
        df_summary[f'n_{outcome.lower()}'] = df_details[df_details['outcome'] == outcome].groupby('video_name').size()
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
