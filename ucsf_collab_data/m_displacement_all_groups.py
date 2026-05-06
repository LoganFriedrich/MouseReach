"""
Per-group max displacement distribution for pre-injury swipes.
Goal: identify the right-tail separation between real movements and tracking jumps
for each group independently.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

UCSF_BASE = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)

ASPA_BASE = r'Y:\2_Connectome\Behavior\MouseReach_Pipeline\Analyzed\ASPA'
VIDEO_DIRS = {
    'M': os.path.join(ASPA_BASE, 'M', 'Post-Processing'),
    'K': os.path.join(ASPA_BASE, 'K', 'Post-Processing'),
    'L': os.path.join(ASPA_BASE, 'L', 'Post-Processing'),
    'H': os.path.join(ASPA_BASE, 'H', 'Post-Processing'),
    'D': os.path.join(ASPA_BASE, 'OptD', 'Single_Animal'),
    'G': os.path.join(ASPA_BASE, 'G', 'Single_Animal'),
}

UCSF_FILES = {
    'Contusion': ['Swipe_Contusion_Data.csv', 'Swipe_Contusion_Data_2.csv'],
    'Transection': ['Swipe_Transection_Data.csv', 'Swipe_Transection_Data_2.csv'],
    'Pyramidotomy': ['Swipe_Pyramidotomy_Data.csv'],
}

GROUP_INJURY = {
    'M': 'Contusion', 'K': 'Contusion', 'L': 'Contusion',
    'G': 'Transection', 'H': 'Transection',
    'D': 'Pyramidotomy',
}


def parse_frame_range(s):
    parts = str(s).strip().split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def session_to_stem(session_id):
    parts = session_id.split('-')
    animal = parts[0]
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    tray = parts[2]
    return '%s_%s_%s' % (date_str, animal, tray)


def load_ucsf_data():
    """Load all UCSF swipe data."""
    all_dfs = []
    for injury, fnames in UCSF_FILES.items():
        for fname in fnames:
            fpath = os.path.join(UCSF_BASE, fname)
            if os.path.exists(fpath):
                d = pd.read_csv(fpath, low_memory=False)
                all_dfs.append(d)
    df = pd.concat(all_dfs, ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))
    return df


def main():
    print('Loading UCSF data...')
    df = load_ucsf_data()
    print('  Total swipes: %d' % len(df))

    # Pre-injury only
    pre = df[df['Test_Type_Grouped_1'].isin([
        '2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'
    ])]
    pre = pre[pre['sd_start'].notna()].copy()
    print('  Pre-injury swipes: %d' % len(pre))
    print()

    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100, 150, 250, 500]

    for grp in ['D', 'G', 'H', 'K', 'L', 'M']:
        video_dir = VIDEO_DIRS.get(grp)
        if video_dir is None or not os.path.exists(video_dir):
            print('%s: No video directory found, skipping' % grp)
            continue

        grp_pre = pre[pre['group'] == grp]
        if len(grp_pre) == 0:
            print('%s: No pre-injury swipes' % grp)
            continue

        print('GROUP %s: %d pre-injury swipes' % (grp, len(grp_pre)))

        dlc_cache = {}
        max_disps = []
        loaded = 0
        failed = 0

        for _, row in grp_pre.iterrows():
            s = int(row['sd_start'])
            e = int(row['sd_end'])
            stem = session_to_stem(row['Session_ID'])

            if stem not in dlc_cache:
                h5 = glob.glob(os.path.join(video_dir, stem + 'DLC*.h5'))
                if not h5:
                    # Try with OptD -> D name mapping
                    alt_stem = stem.replace('_D', '_OptD').replace('_G', '_OptG')
                    h5 = glob.glob(os.path.join(video_dir, alt_stem + 'DLC*.h5'))
                if not h5:
                    dlc_cache[stem] = None
                    failed += 1
                    continue
                d = pd.read_hdf(h5[0])
                sc = d.columns.get_level_values(0)[0]
                dlc_cache[stem] = d[sc]
                loaded += 1

            dlc = dlc_cache[stem]
            if dlc is None:
                continue

            if e + 1 > len(dlc):
                continue

            rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
            rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
            disp = np.sqrt(np.diff(rh_x)**2 + np.diff(rh_y)**2)
            if len(disp) > 0:
                max_disps.append(disp.max())

        if not max_disps:
            print('  No displacements computed (loaded=%d, failed=%d)' % (loaded, failed))
            print()
            continue

        md = np.array(max_disps)
        print('  DLC files: %d loaded, %d failed' % (loaded, failed))
        print('  Swipes with displacement data: %d' % len(md))
        print('  Percentiles:')
        for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
            print('    p%-5s = %6.1f px' % ('%.1f' % p, np.percentile(md, p)))
        print('    max    = %6.1f px' % md.max())

        # Right tail: count by bin
        counts, _ = np.histogram(md, bins=bins)
        print('  Right tail (bins with counts):')
        for i in range(len(counts)):
            if bins[i] >= 30 or counts[i] > 0:
                bar = '#' * min(counts[i], 100)
                print('    %3d-%3d px: %5d  %s' % (bins[i], bins[i+1], counts[i], bar))

        # Identify the gap
        for i in range(len(counts) - 1):
            if counts[i] > 0 and counts[i+1] == 0:
                # Check if there are counts further right
                has_more = any(counts[j] > 0 for j in range(i+2, len(counts)))
                if has_more:
                    print('  ** GAP detected: %d-%d px (count=%d) -> %d-%d px (count=0)' % (
                        bins[i], bins[i+1], counts[i], bins[i+1], bins[i+2]))

        print()


if __name__ == '__main__':
    main()
