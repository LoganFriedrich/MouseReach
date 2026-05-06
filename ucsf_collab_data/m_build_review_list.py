"""
Build discrete review list of M suspect swipes and sessions.

Outputs:
  - M_suspect_swipes_for_review.csv: every flagged swipe with frame numbers
  - M_sessions_for_review.json: session-level priority list for video inspection
"""

import pandas as pd
import numpy as np
import os
import json

base = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_DIR = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE_MAP = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    'Post-Injury': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test',
                     '3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}

SUSPECTS = ['M05', 'M06', 'M07', 'M13', 'M14']


def load_data():
    df1 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data.csv'))
    df2 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data_2.csv'))
    df1['source_csv'] = 'CSV1'
    df2['source_csv'] = 'CSV2'
    df = pd.concat([df1, df2], ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df = df[df['group'] == 'M'].copy()

    for col in ['Path_over_Frames', 'Swipe_length', 'Swipe_area', 'Path_length',
                'Swipe_Duration_Frames']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    test_to_phase = {}
    for phase, tests in PHASE_MAP.items():
        for t in tests:
            test_to_phase[t] = phase
    df['phase'] = df['Test_Type_Grouped_1'].map(test_to_phase)

    return df


def compute_baselines(df):
    baselines = {}
    for animal in SUSPECTS:
        pre = df[(df['SubjectID'] == animal) & (df['phase'] == 'Pre-Injury')]
        baselines[animal] = {
            'pof_p75': pre['Path_over_Frames'].quantile(0.75) if len(pre) > 0 else 999,
            'area_p75': pre['Swipe_area'].quantile(0.75) if len(pre) > 0 else 999,
            'len_p75': pre['Swipe_length'].quantile(0.75) if len(pre) > 0 else 999,
        }
    return baselines


def flag_swipes(df, baselines):
    """Flag individual swipes that are suspect."""
    rows = []

    for animal in SUSPECTS:
        ad = df[(df['SubjectID'] == animal) &
                (df['phase'].isin(['Post-Injury', 'Post-Rehab']))].copy()
        bl = baselines[animal]

        for _, swipe in ad.iterrows():
            flags = []

            # Flag type 1: CSV1 no-pellet swipe
            if swipe['source_csv'] == 'CSV1' and swipe.get('Reach_outcome') == 'no pellet':
                flags.append('CSV1_NO_PELLET')

            # Flag type 2: kinematic outlier (exceeds pre-injury p75 on 2+ metrics)
            exceeds = []
            if pd.notna(swipe['Path_over_Frames']) and swipe['Path_over_Frames'] > bl['pof_p75']:
                exceeds.append('PoF')
            if pd.notna(swipe['Swipe_area']) and swipe['Swipe_area'] > bl['area_p75']:
                exceeds.append('Area')
            if pd.notna(swipe['Swipe_length']) and swipe['Swipe_length'] > bl['len_p75']:
                exceeds.append('Len')
            if len(exceeds) >= 2:
                flags.append('EXCEEDS_BASELINE(%s)' % '+'.join(exceeds))

            if not flags:
                continue

            frame_range = str(swipe.get('Swipe_Duration', '')).strip()
            start_frame = swipe.get('bodyparts_coords', '')

            rows.append({
                'animal': animal,
                'session_id': swipe['Session_ID'],
                'test_date': swipe['Test_Date'],
                'test_type': swipe['Test_Type'],
                'phase': swipe['phase'],
                'tray': swipe.get('Tray_ID', ''),
                'swipe_id': swipe['Swipe_ID'],
                'frame_range': frame_range,
                'start_frame': int(start_frame) if pd.notna(start_frame) else '',
                'duration_frames': int(swipe['Swipe_Duration_Frames']) if pd.notna(swipe['Swipe_Duration_Frames']) else '',
                'reach_outcome': swipe.get('Reach_outcome', ''),
                'pof': round(float(swipe['Path_over_Frames']), 2) if pd.notna(swipe['Path_over_Frames']) else '',
                'area': round(float(swipe['Swipe_area']), 1) if pd.notna(swipe['Swipe_area']) else '',
                'length': round(float(swipe['Swipe_length']), 1) if pd.notna(swipe['Swipe_length']) else '',
                'source_csv': swipe['source_csv'],
                'flags': ' | '.join(flags),
            })

    return pd.DataFrame(rows)


def print_summary(review_df):
    print('FLAGGED SWIPES FOR VIDEO REVIEW')
    print('=' * 90)
    print()
    print('Total flagged swipes: %d' % len(review_df))
    print()

    for animal in SUSPECTS:
        af = review_df[review_df['animal'] == animal]
        print('%s: %d flagged swipes across %d sessions' % (
            animal, len(af), af['session_id'].nunique()))

        for sid, sgroup in af.groupby('session_id'):
            date = sgroup['test_date'].iloc[0]
            test = sgroup['test_type'].iloc[0]
            n_csv1_np = int(sgroup['flags'].str.contains('CSV1_NO_PELLET').sum())
            n_exceeds = int(sgroup['flags'].str.contains('EXCEEDS_BASELINE').sum())

            frames = [f for f in sgroup['start_frame'] if f != '']
            frame_min = min(frames) if frames else '?'
            frame_max = max(frames) if frames else '?'

            print('  %-28s %s  %-30s  %3d swipes  (NP:%d, Exceed:%d)  frames %s-%s' % (
                sid, date, test, len(sgroup), n_csv1_np, n_exceeds, frame_min, frame_max))
        print()

    # Flag distribution
    print('FLAG DISTRIBUTION')
    print('-' * 40)
    csv1_np = int(review_df['flags'].str.contains('CSV1_NO_PELLET').sum())
    exceeds = int(review_df['flags'].str.contains('EXCEEDS_BASELINE').sum())
    both = int((review_df['flags'].str.contains('CSV1_NO_PELLET') &
                review_df['flags'].str.contains('EXCEEDS_BASELINE')).sum())
    print('CSV1 no-pellet only:     %d' % (csv1_np - both))
    print('Exceeds baseline only:   %d' % (exceeds - both))
    print('Both flags:              %d' % both)
    print('Total:                   %d' % len(review_df))


def build_session_summary(review_df):
    session_summary = []
    for animal in SUSPECTS:
        af = review_df[review_df['animal'] == animal]
        for sid, sgroup in af.groupby('session_id'):
            frames = [f for f in sgroup['start_frame'] if f != '']
            n_flagged = len(sgroup)
            session_summary.append({
                'animal': animal,
                'session_id': sid,
                'test_date': sgroup['test_date'].iloc[0],
                'test_type': sgroup['test_type'].iloc[0],
                'phase': sgroup['phase'].iloc[0],
                'tray': sgroup['tray'].iloc[0],
                'n_flagged_swipes': n_flagged,
                'n_csv1_no_pellet': int(sgroup['flags'].str.contains('CSV1_NO_PELLET').sum()),
                'n_exceeds_baseline': int(sgroup['flags'].str.contains('EXCEEDS_BASELINE').sum()),
                'frame_range': '%s-%s' % (min(frames), max(frames)) if frames else 'unknown',
                'priority': 'HIGH' if n_flagged > 20 else 'MEDIUM' if n_flagged > 5 else 'LOW',
            })
    return session_summary


def print_priority_list(session_summary):
    print('\n\nPRIORITY VIDEO REVIEW LIST')
    print('=' * 110)
    print('Review these sessions in this order (HIGH first):')
    print()
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        sessions = [s for s in session_summary if s['priority'] == priority]
        if sessions:
            print('[%s PRIORITY] (%d sessions)' % (priority, len(sessions)))
            for s in sorted(sessions, key=lambda x: -x['n_flagged_swipes']):
                print('  %s  %-28s  %s  %-30s  %d flagged (NP:%d, Exceed:%d)  frames: %s' % (
                    s['animal'], s['session_id'], s['test_date'], s['test_type'],
                    s['n_flagged_swipes'], s['n_csv1_no_pellet'], s['n_exceeds_baseline'],
                    s['frame_range']))
            print()


def main():
    print('Loading M data...')
    df = load_data()
    print('  %d total M swipes' % len(df))

    baselines = compute_baselines(df)
    print('\nPre-injury 75th percentiles:')
    for animal in SUSPECTS:
        bl = baselines[animal]
        print('  %s: PoF=%.2f, Area=%.1f, Len=%.1f' % (
            animal, bl['pof_p75'], bl['area_p75'], bl['len_p75']))

    print('\nFlagging suspect swipes...')
    review_df = flag_swipes(df, baselines)

    print()
    print_summary(review_df)

    # Save full swipe-level CSV
    out_csv = os.path.join(OUTPUT_DIR, 'M_suspect_swipes_for_review.csv')
    review_df.to_csv(out_csv, index=False)
    print('\nSaved swipe-level CSV: %s' % out_csv)

    # Build and save session-level summary
    session_summary = build_session_summary(review_df)

    out_json = os.path.join(OUTPUT_DIR, 'M_sessions_for_review.json')
    with open(out_json, 'w') as f:
        json.dump(session_summary, f, indent=2)
    print('Saved session-level JSON: %s' % out_json)

    print_priority_list(session_summary)

    # Also save a simple text checklist
    out_txt = os.path.join(OUTPUT_DIR, 'M_video_review_checklist.txt')
    with open(out_txt, 'w') as f:
        f.write('M GROUP (60kD) -- VIDEO REVIEW CHECKLIST\n')
        f.write('Generated by m_build_review_list.py\n')
        f.write('=' * 70 + '\n\n')
        f.write('WHAT TO LOOK FOR:\n')
        f.write('  [ ] Camera position/angle change between pre and post sessions\n')
        f.write('  [ ] ASPA detecting non-reach movements (grooming, exploration)\n')
        f.write('  [ ] DLC tracking jumps or identity swaps\n')
        f.write('  [ ] Animal not actually reaching (body movement only)\n')
        f.write('  [ ] Pellet tray empty but movements still detected\n\n')

        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            sessions = [s for s in session_summary if s['priority'] == priority]
            if not sessions:
                continue
            f.write('[%s PRIORITY]\n' % priority)
            for s in sorted(sessions, key=lambda x: -x['n_flagged_swipes']):
                f.write('  [ ] %s | %s | %s | %s\n' % (
                    s['animal'], s['session_id'], s['test_date'], s['test_type']))
                f.write('      %d flagged swipes (CSV1-no-pellet: %d, exceeds-baseline: %d)\n' % (
                    s['n_flagged_swipes'], s['n_csv1_no_pellet'], s['n_exceeds_baseline']))
                f.write('      Check frames: %s\n' % s['frame_range'])
                f.write('\n')

    print('Saved checklist: %s' % out_txt)
    print('\nDone. All outputs in: %s' % OUTPUT_DIR)


if __name__ == '__main__':
    main()
