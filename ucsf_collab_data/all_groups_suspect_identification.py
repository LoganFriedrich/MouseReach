"""
Identify suspect animals in every injury group using the same method:
per-animal pre-to-post kinematic shift on 3 features (Path/Frames, Length, Area).
Animals that improve (positive shift) on 2+ of 3 features are flagged as suspects.

This standardizes the suspect identification that was done manually for M.

Output: CSV with per-animal shifts and suspect flag for all groups.
"""

import pandas as pd
import numpy as np
import os

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

FEATURES = ['Path_over_Frames', 'Swipe_length', 'Swipe_area']
SUSPECT_THRESHOLD = 2  # must be positive on at least this many of 3 features


def load_all_groups():
    """Load all UCSF swipe data, assign group and phase."""
    dfs = []

    # Contusion (K, L, M)
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    dfs.append(df1)
    dfs.append(df2)

    # Pyramidotomy (D)
    dfs.append(pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Pyramidotomy_Data.csv'), low_memory=False))

    # Transection (G, H)
    dfs.append(pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Transection_Data.csv'), low_memory=False))
    dfs.append(pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Transection_Data_2.csv'), low_memory=False))

    df = pd.concat(dfs, ignore_index=True)
    df['group'] = df['SubjectID'].str[0]

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    test_to_phase = {}
    for phase, tests in PHASE_MAP.items():
        for t in tests:
            test_to_phase[t] = phase
    df['phase'] = df['Test_Type_Grouped_1'].map(test_to_phase)

    return df[df['phase'].notna()].copy()


def compute_shifts(df):
    """Compute per-animal pre-to-post % shift on each feature.
    Returns DataFrame with one row per animal."""
    results = []
    for group in sorted(df['group'].unique()):
        gdf = df[df['group'] == group]
        for animal in sorted(gdf['SubjectID'].unique()):
            adf = gdf[gdf['SubjectID'] == animal]
            pre = adf[adf['phase'] == 'Pre-Injury']
            post = adf[adf['phase'] == 'Post-Injury']

            if len(pre) < 5 or len(post) < 5:
                continue

            row = {'group': group, 'animal': animal,
                   'n_pre': len(pre), 'n_post': len(post)}
            n_positive = 0
            for feat in FEATURES:
                pre_mean = pre[feat].mean()
                post_mean = post[feat].mean()
                if pre_mean > 0:
                    pct = (post_mean - pre_mean) / pre_mean * 100
                else:
                    pct = 0.0
                row['%s_pct' % feat] = pct
                if pct > 0:
                    n_positive += 1
            row['n_positive_features'] = n_positive
            row['suspect'] = n_positive >= SUSPECT_THRESHOLD
            results.append(row)

    return pd.DataFrame(results)


def main():
    print('Loading all groups...')
    df = load_all_groups()
    print('Total rows: %d' % len(df))
    print('Groups: %s' % sorted(df['group'].unique()))

    print('\nComputing per-animal shifts...')
    shifts = compute_shifts(df)

    # Summary
    for group in sorted(shifts['group'].unique()):
        gdf = shifts[shifts['group'] == group]
        suspects = gdf[gdf['suspect']]
        print('\n%s: %d animals, %d suspects' % (group, len(gdf), len(suspects)))
        if len(suspects) > 0:
            for _, row in suspects.iterrows():
                print('  %s: PoF=%+.1f%%, Len=%+.1f%%, Area=%+.1f%%' % (
                    row['animal'],
                    row['Path_over_Frames_pct'],
                    row['Swipe_length_pct'],
                    row['Swipe_area_pct']))

    # Save
    out = os.path.join(OUTPUT_DIR, 'all_groups_suspect_identification.csv')
    shifts.to_csv(out, index=False)
    print('\nSaved: %s' % out)

    # Also save just the suspect list for easy loading
    suspect_list = shifts[shifts['suspect']][['group', 'animal']].copy()
    out2 = os.path.join(OUTPUT_DIR, 'suspect_animals.csv')
    suspect_list.to_csv(out2, index=False)
    print('Saved: %s' % out2)


if __name__ == '__main__':
    main()
