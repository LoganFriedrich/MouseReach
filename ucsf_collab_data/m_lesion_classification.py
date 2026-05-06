"""
Classify M group mice by best-fit lesion position using the biomech lesion simulator.

Reads UCSF kinematic data (read-only), maps features to MouseReach equivalents,
runs lesion sweep comparison. No modifications to either codebase.

Feature mapping (direction-only, sign agreement):
    UCSF Swipe_length         -> max_extent_mm          (+ = further)
    UCSF Swipe_speed          -> peak_velocity_px_per_frame  (+ = faster)
    UCSF Path_over_Frames     -> mean_velocity_px_per_frame  (+ = faster)
    UCSF Swipe_Duration_Frames -> duration_frames        (+ = longer)
    UCSF Swipe_breadth (inv)  -> trajectory_straightness (+ = straighter, so invert breadth)
    trajectory_smoothness     -> not available in UCSF, excluded
"""

import pandas as pd
import numpy as np
import os
import sys

# Add mousereach to path (read-only)
sys.path.insert(0, 'Y:/2_Connectome/Behavior/MouseReach/src')

UCSF_BASE = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_DIR = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality')

PHASE_MAP = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    'Post-Injury': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test',
                     '3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
}

# UCSF -> MouseReach feature mapping
# (ucsf_col, mousereach_col, invert)
FEATURE_MAP = [
    ('Swipe_length', 'max_extent_mm', False),
    ('Swipe_speed', 'peak_velocity_px_per_frame', False),
    ('Path_over_Frames', 'mean_velocity_px_per_frame', False),
    ('Swipe_Duration_Frames', 'duration_frames', False),
    ('Swipe_breadth', 'trajectory_straightness', True),  # wider = less straight
]


def load_m_data():
    """Load M group UCSF data, assign phases."""
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['SubjectID'].str.startswith('M', na=False)].copy()

    for col in ['Swipe_length', 'Swipe_speed', 'Path_over_Frames',
                'Swipe_Duration_Frames', 'Swipe_breadth', 'Swipe_area']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    test_to_phase = {}
    for phase, tests in PHASE_MAP.items():
        for t in tests:
            test_to_phase[t] = phase
    df['phase'] = df['Test_Type_Grouped_1'].map(test_to_phase)

    return df[df['phase'].notna()].copy()


def build_mousereach_features(df, phase):
    """Map UCSF features to MouseReach feature names for a given phase.
    Returns DataFrame with subject_id + mousereach feature columns."""
    phase_df = df[df['phase'] == phase].copy()

    rows = []
    for animal in sorted(phase_df['SubjectID'].unique()):
        adf = phase_df[phase_df['SubjectID'] == animal]
        if len(adf) < 5:
            continue
        row = {'subject_id': animal}
        for ucsf_col, mr_col, invert in FEATURE_MAP:
            val = adf[ucsf_col].mean()
            if invert:
                val = -val  # invert so direction matches mousereach convention
            row[mr_col] = val
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print('Loading M group UCSF data...')
    df = load_m_data()
    print('Total rows: %d' % len(df))
    print('Animals: %s' % sorted(df['SubjectID'].unique()))

    # Build pre and post feature DataFrames in MouseReach format
    print('\nBuilding feature DataFrames...')
    pre_features = build_mousereach_features(df, 'Pre-Injury')
    post_features = build_mousereach_features(df, 'Post-Injury')
    print('Pre-injury: %d animals' % len(pre_features))
    print('Post-injury: %d animals' % len(post_features))

    # Show the raw shifts before classification
    print('\nPer-animal feature shifts (post - pre as %% of pre):')
    print('%-6s %10s %10s %10s %10s %10s' % ('Animal', 'Extent', 'PeakVel', 'MeanVel', 'Duration', 'Straight'))
    print('-' * 60)
    suspects = {'M05', 'M06', 'M13', 'M14'}
    for _, pre_row in pre_features.iterrows():
        animal = pre_row['subject_id']
        post_row = post_features[post_features['subject_id'] == animal]
        if len(post_row) == 0:
            continue
        post_row = post_row.iloc[0]
        shifts = []
        for _, mr_col, _ in FEATURE_MAP:
            pre_val = pre_row[mr_col]
            post_val = post_row[mr_col]
            if pre_val != 0:
                pct = (post_val - pre_val) / abs(pre_val) * 100
            else:
                pct = 0
            shifts.append(pct)
        tag = ' ***' if animal in suspects else ''
        print('%-6s %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%%%s' % (
            animal, shifts[0], shifts[1], shifts[2], shifts[3], shifts[4], tag))

    # Run lesion sweep analysis
    print('\nRunning lesion sweep...')
    from mousereach.biomech.analysis.kinematic_predictions import LesionSweepAnalysis
    analysis = LesionSweepAnalysis()
    sweep = analysis.run_sweep()

    # Compare real data to predictions
    print('\nClassifying animals by best-fit lesion position...')
    classification = analysis.compare_to_real_data(
        sweep, post_features, pre_features, animal_col='subject_id'
    )

    # Results
    print('\n=== LESION CLASSIFICATION RESULTS ===')
    print('%-6s %12s %8s %s' % ('Animal', 'Best Fit', 'Score', 'Suspect'))
    print('-' * 45)
    for _, row in classification.sort_values('best_fit_center').iterrows():
        animal = row['animal']
        center = row['best_fit_center']
        # Convert numeric center to segment name
        segment_names = {4: 'C4', 4.5: 'C4-C5', 5: 'C5', 5.5: 'C5-C6',
                         6: 'C6', 6.5: 'C6-C7', 7: 'C7', 7.5: 'C7-T1', 8: 'T1'}
        seg = segment_names.get(center, 'C%.1f' % center)
        tag = ' ***' if animal in suspects else ''
        print('%-6s %12s %+7.2f  %s' % (animal, seg, row['best_fit_score'], tag))

    # Save results
    out = os.path.join(OUTPUT_DIR, 'm_lesion_classification.csv')
    classification.to_csv(out, index=False)
    print('\nSaved: %s' % out)

    # Summary
    print('\n=== SUMMARY ===')
    suspect_df = classification[classification['animal'].isin(suspects)]
    normal_df = classification[~classification['animal'].isin(suspects)]
    if len(suspect_df) > 0:
        print('Suspect mean center: %.1f (n=%d)' % (suspect_df['best_fit_center'].mean(), len(suspect_df)))
    if len(normal_df) > 0:
        print('Normal mean center: %.1f (n=%d)' % (normal_df['best_fit_center'].mean(), len(normal_df)))


if __name__ == '__main__':
    main()
