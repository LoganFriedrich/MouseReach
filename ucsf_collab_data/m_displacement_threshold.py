"""
Establish maximum plausible frame-to-frame RightHand displacement
from each animal's final 3 pre-injury sessions.

Any frame where displacement exceeds this is a DLC tracking jump, not real movement.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

VIDEO_DIR = r'X:\! DLC Output\Analyzed\M\Post-Processing'
UCSF_BASE = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)


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


def main():
    # Load UCSF data
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df = df[df['group'] == 'M'].copy()

    for col in ['Swipe_area', 'Path_length', 'Swipe_Duration_Frames']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    # Get final 3 pre-injury sessions only
    pre = df[df['Test_Type_Grouped_1'].isin(['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'])]
    pre = pre[pre['sd_start'].notna()].copy()

    print('PRE-INJURY FRAME-TO-FRAME DISPLACEMENT ANALYSIS')
    print('=' * 90)
    print('For each animal: load DLC data for every pre-injury swipe,')
    print('compute frame-to-frame RH displacement within each swipe.')
    print()

    dlc_cache = {}
    all_displacements = {}  # per animal

    for animal in sorted(pre['SubjectID'].unique()):
        animal_pre = pre[pre['SubjectID'] == animal]
        displacements = []

        for _, row in animal_pre.iterrows():
            s = int(row['sd_start'])
            e = int(row['sd_end'])
            stem = session_to_stem(row['Session_ID'])

            if stem not in dlc_cache:
                h5 = glob.glob(os.path.join(VIDEO_DIR, stem + 'DLC*.h5'))
                if not h5:
                    continue
                d = pd.read_hdf(h5[0])
                sc = d.columns.get_level_values(0)[0]
                dlc_cache[stem] = d[sc]
            dlc = dlc_cache[stem]

            rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
            rh_y = dlc['RightHand']['y'].iloc[s:e+1].values

            dx = np.diff(rh_x)
            dy = np.diff(rh_y)
            disp = np.sqrt(dx**2 + dy**2)
            displacements.extend(disp.tolist())

        if displacements:
            d = np.array(displacements)
            all_displacements[animal] = d
            print('%s: %d frame-to-frame displacements across %d swipes' % (
                animal, len(d), len(animal_pre)))
            print('  mean=%.1f  median=%.1f  p95=%.1f  p99=%.1f  p99.9=%.1f  max=%.1f px/frame' % (
                d.mean(), np.median(d), np.percentile(d, 95),
                np.percentile(d, 99), np.percentile(d, 99.9), d.max()))
            print()

    # Overall pre-injury displacement distribution
    all_d = np.concatenate(list(all_displacements.values()))
    print('\nALL M ANIMALS COMBINED (pre-injury):')
    print('  N = %d displacements' % len(all_d))
    print('  mean=%.1f  median=%.1f  p95=%.1f  p99=%.1f  p99.9=%.1f  max=%.1f px/frame' % (
        all_d.mean(), np.median(all_d), np.percentile(all_d, 95),
        np.percentile(all_d, 99), np.percentile(all_d, 99.9), all_d.max()))
    print()
    print('At 60fps, 1 frame = 16.7ms')
    print('  p99 displacement = %.1f px/frame = %.0f px/sec' % (
        np.percentile(all_d, 99), np.percentile(all_d, 99) * 60))
    print('  max displacement = %.1f px/frame = %.0f px/sec' % (
        all_d.max(), all_d.max() * 60))
    print()

    # Now check post-injury swipes: how many frames exceed the pre-injury p99?
    threshold = np.percentile(all_d, 99)
    print('PROPOSED THRESHOLD: %.1f px/frame (p99 of pre-injury)' % threshold)
    print()

    post = df[df['Test_Type_Grouped_1'].str.contains('Post-injury', na=False)]
    post = post[post['sd_start'].notna()].copy()

    print('POST-INJURY: frames exceeding threshold')
    print('=' * 90)

    for animal in sorted(post['SubjectID'].unique()):
        animal_post = post[post['SubjectID'] == animal]
        total_frames = 0
        bad_frames = 0
        swipes_with_bad = 0
        total_swipes = 0

        for _, row in animal_post.iterrows():
            s = int(row['sd_start'])
            e = int(row['sd_end'])
            stem = session_to_stem(row['Session_ID'])

            if stem not in dlc_cache:
                h5 = glob.glob(os.path.join(VIDEO_DIR, stem + 'DLC*.h5'))
                if not h5:
                    continue
                d = pd.read_hdf(h5[0])
                sc = d.columns.get_level_values(0)[0]
                dlc_cache[stem] = d[sc]
            dlc = dlc_cache[stem]

            rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
            rh_y = dlc['RightHand']['y'].iloc[s:e+1].values

            dx = np.diff(rh_x)
            dy = np.diff(rh_y)
            disp = np.sqrt(dx**2 + dy**2)

            n_bad = (disp > threshold).sum()
            total_frames += len(disp)
            bad_frames += n_bad
            total_swipes += 1
            if n_bad > 0:
                swipes_with_bad += 1

        if total_frames > 0:
            tag = '*** SUSPECT ***' if animal in ['M05', 'M06', 'M07', 'M13', 'M14'] else ''
            print('%s: %d/%d bad frames (%.1f%%), %d/%d swipes affected (%.1f%%)  %s' % (
                animal, bad_frames, total_frames,
                bad_frames / total_frames * 100,
                swipes_with_bad, total_swipes,
                swipes_with_bad / total_swipes * 100,
                tag))

    # Also check pre-injury for comparison
    print()
    print('PRE-INJURY: frames exceeding threshold (should be ~1%)')
    print('=' * 90)
    for animal in sorted(pre['SubjectID'].unique()):
        animal_pre_data = pre[pre['SubjectID'] == animal]
        total_frames = 0
        bad_frames = 0

        for _, row in animal_pre_data.iterrows():
            s = int(row['sd_start'])
            e = int(row['sd_end'])
            stem = session_to_stem(row['Session_ID'])
            if stem not in dlc_cache:
                continue
            dlc = dlc_cache[stem]

            rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
            rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
            disp = np.sqrt(np.diff(rh_x)**2 + np.diff(rh_y)**2)
            n_bad = (disp > threshold).sum()
            total_frames += len(disp)
            bad_frames += n_bad

        if total_frames > 0:
            print('%s: %d/%d bad frames (%.1f%%)' % (
                animal, bad_frames, total_frames,
                bad_frames / total_frames * 100))


if __name__ == '__main__':
    main()
