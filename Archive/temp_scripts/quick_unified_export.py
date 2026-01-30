"""Quick unified reach export for presentation."""
import json
import pandas as pd
from pathlib import Path
import re

pipeline_dir = Path(__file__).parent

# Collect all reaches and outcomes
all_reaches = []
all_outcomes = {}

# Find all reach files (prefer Processing over Archive)
reach_files = list(pipeline_dir.glob('Processing/*_reaches.json'))
print(f"Found {len(reach_files)} reach files")

for rf in reach_files:
    try:
        with open(rf) as f:
            data = json.load(f)

        video_name = data.get('video_name', rf.stem.replace('_reaches', ''))

        # Parse filename: 20250701_CNT0110_P2
        # P2 = second pillar tray run of the day, F1 = first flat tray run, etc.
        match = re.match(r'(\d{8})_([A-Z]+\d+)_([PF])(\d)?', video_name)
        if match:
            date_str, animal, tray_type, run_num = match.groups()
            date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            cohort = f"CNT_{animal[3:5]}"
            tray_type = 'Pillar' if tray_type == 'P' else 'Flat'
            run_num = int(run_num) if run_num else 1
        else:
            date, animal, tray_type, run_num, cohort = None, video_name, None, None, None

        for seg in data.get('segments', []):
            seg_num = seg.get('segment_num')
            for reach in seg.get('reaches', []):
                all_reaches.append({
                    'video_name': video_name,
                    'date': date,
                    'animal': animal,
                    'tray_type': tray_type,
                    'run_num': run_num,
                    'cohort': cohort,
                    'segment_num': seg_num,
                    'segment_start_frame': seg.get('start_frame'),
                    'segment_end_frame': seg.get('end_frame'),
                    'reach_id': reach.get('reach_id'),
                    'reach_num_in_segment': reach.get('reach_num'),
                    'start_frame': reach.get('start_frame'),
                    'apex_frame': reach.get('apex_frame'),
                    'end_frame': reach.get('end_frame'),
                    'duration_frames': reach.get('duration_frames'),
                    'max_extent_pixels': reach.get('max_extent_pixels'),
                    'max_extent_ruler': reach.get('max_extent_ruler'),
                    'source': reach.get('source'),
                    'human_corrected': reach.get('human_corrected', False),
                })
    except Exception as e:
        print(f"  Error {rf.name}: {e}")

print(f"Loaded {len(all_reaches)} reaches")

# Load outcome ground truths (prefer ground truth over pellet_outcomes)
outcome_files = list(pipeline_dir.glob('Processing/*_outcome_ground_truth.json'))
print(f"Found {len(outcome_files)} outcome ground truth files")

for of in outcome_files:
    try:
        with open(of) as f:
            data = json.load(f)
        video_name = data.get('video_name', of.stem.replace('_outcome_ground_truth', ''))
        all_outcomes[video_name] = {seg['segment_num']: seg for seg in data.get('segments', [])}
    except Exception as e:
        print(f"  Error {of.name}: {e}")

# Also load pellet_outcomes for videos without ground truth
pellet_outcome_files = list(pipeline_dir.glob('Processing/*_pellet_outcomes.json'))
for pof in pellet_outcome_files:
    video_name = pof.stem.replace('_pellet_outcomes', '')
    if video_name not in all_outcomes:
        try:
            with open(pof) as f:
                data = json.load(f)
            all_outcomes[video_name] = {seg['segment_num']: seg for seg in data.get('segments', [])}
        except:
            pass

print(f"Loaded outcomes for {len(all_outcomes)} videos")

# Merge outcomes into reaches
for reach in all_reaches:
    video = reach['video_name']
    seg = reach['segment_num']
    if video in all_outcomes and seg in all_outcomes[video]:
        out = all_outcomes[video][seg]
        reach['outcome'] = out.get('outcome')
        reach['interaction_frame'] = out.get('interaction_frame')
        reach['outcome_known_frame'] = out.get('outcome_known_frame')
        reach['causal_reach_id'] = out.get('causal_reach_id')
        reach['human_verified'] = out.get('human_verified', False)
        reach['confidence'] = out.get('confidence')
        reach['pellet_visible_start'] = out.get('pellet_visible_start')
        reach['pellet_visible_end'] = out.get('pellet_visible_end')
        reach['distance_from_pillar_start'] = out.get('distance_from_pillar_start')
        reach['distance_from_pillar_end'] = out.get('distance_from_pillar_end')

# Create DataFrame
df = pd.DataFrame(all_reaches)

# Add derived columns
if len(df) > 0:
    # Is this reach the causal reach for the outcome?
    df['is_causal_reach'] = df.apply(lambda r: r['reach_id'] == r.get('causal_reach_id'), axis=1)

    # Duration in seconds (30 fps)
    df['duration_sec'] = df['duration_frames'] / 30.0

    # Max extent in mm (ruler is in ruler-widths, ruler = 10mm)
    df['max_extent_mm'] = df['max_extent_ruler'] * 10

    # Outcome numeric (0=miss, 1=displaced, 2=retrieved)
    outcome_map = {'untouched': 0, 'displaced_sa': 1, 'dropped': 1, 'retrieved': 2}
    df['outcome_score'] = df['outcome'].map(outcome_map)

# Sort
df = df.sort_values(['date', 'animal', 'segment_num', 'reach_num_in_segment'])

# Export
out_path = pipeline_dir / 'unified_reaches_for_presentation.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    df.to_excel(writer, 'All_Reaches', index=False)

    # Summary by animal
    if len(df) > 0:
        summary = df.groupby(['cohort', 'animal', 'date']).agg({
            'reach_id': 'count',
            'outcome_score': 'mean',
            'max_extent_mm': 'mean',
            'duration_sec': 'mean',
            'is_causal_reach': 'sum'
        }).rename(columns={
            'reach_id': 'n_reaches',
            'outcome_score': 'avg_outcome',
            'max_extent_mm': 'avg_extent_mm',
            'duration_sec': 'avg_duration_sec',
            'is_causal_reach': 'n_causal_reaches'
        }).reset_index()
        summary.to_excel(writer, 'Summary_by_Session', index=False)

print(f"\n{'='*60}")
print(f"EXPORTED: {out_path}")
print(f"{'='*60}")
print(f"Total reaches: {len(df)}")
if len(df) > 0:
    print(f"Animals: {df['animal'].nunique()}")
    print(f"Cohorts: {sorted(df['cohort'].dropna().unique())}")
    print(f"Sessions: {df['video_name'].nunique()}")
    print(f"Outcomes available: {df['outcome'].notna().sum()} ({df['outcome'].notna().mean()*100:.0f}%)")
    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())
