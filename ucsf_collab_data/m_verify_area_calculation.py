"""
Verify: compute area from DLC data for a UCSF swipe and compare to reported value.
Tests multiple area calculation methods to figure out which one matches.
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


def shoelace(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main():
    # Load UCSF data
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df['Swipe_area'] = pd.to_numeric(df['Swipe_area'], errors='coerce')
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    # Pick several swipes to test: 2 normal M01, 3 high-area suspects
    m01_pre = df[(df['SubjectID'] == 'M01') &
                 (df['Test_Type_Grouped_1'] == '2_Pre-injury_1') &
                 (df['Swipe_area'] > 80) & (df['Swipe_area'] < 250) &
                 (df['sd_start'].notna())].sort_values('Swipe_area', ascending=False).head(2)

    suspects = df[(df['SubjectID'].isin(['M05', 'M13', 'M14'])) &
                  (df['Test_Type_Grouped_1'].str.contains('Post-injury', na=False)) &
                  (df['sd_start'].notna())].sort_values('Swipe_area', ascending=False).head(3)

    test_swipes = pd.concat([m01_pre, suspects])

    dlc_cache = {}

    print('AREA VERIFICATION: UCSF value vs computed from DLC coordinates')
    print('=' * 110)

    for _, row in test_swipes.iterrows():
        s = int(row['sd_start'])
        e = int(row['sd_end'])
        ucsf_area = row['Swipe_area']
        session = row['Session_ID']
        animal = row['SubjectID']

        stem = session_to_stem(session)
        if stem not in dlc_cache:
            h5 = glob.glob(os.path.join(VIDEO_DIR, stem + 'DLC*.h5'))
            if not h5:
                print('%s: No DLC file for %s' % (animal, stem))
                continue
            d = pd.read_hdf(h5[0])
            sc = d.columns.get_level_values(0)[0]
            dlc_cache[stem] = d[sc]
        dlc = dlc_cache[stem]

        rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
        rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
        rh_lk = dlc['RightHand']['likelihood'].iloc[s:e+1].values
        nose_x = dlc['Nose']['x'].iloc[s:e+1].values
        nose_y = dlc['Nose']['y'].iloc[s:e+1].values

        n_frames = e - s + 1
        mean_lk = np.mean(rh_lk)

        # Method 1: Shoelace on RH only (polygon closes back to first RH point)
        area1 = shoelace(rh_x, rh_y)

        # Method 2: Close polygon at mean nose position
        x2 = np.append(rh_x, np.mean(nose_x))
        y2 = np.append(rh_y, np.mean(nose_y))
        area2 = shoelace(x2, y2)

        # Method 3: Close at nose position at first frame
        x3 = np.append(rh_x, nose_x[0])
        y3 = np.append(rh_y, nose_y[0])
        area3 = shoelace(x3, y3)

        # Method 4: Close at nose positions at first AND last frame
        x4 = np.concatenate([[nose_x[0]], rh_x, [nose_x[-1]]])
        y4 = np.concatenate([[nose_y[0]], rh_y, [nose_y[-1]]])
        area4 = shoelace(x4, y4)

        # Method 5: Only high-confidence frames (lk > 0.5)
        good = rh_lk > 0.5
        if good.sum() > 2:
            area5 = shoelace(rh_x[good], rh_y[good])
        else:
            area5 = 0

        # Method 6: High-confidence frames closed at nose
        if good.sum() > 2:
            x6 = np.append(rh_x[good], np.mean(nose_x))
            y6 = np.append(rh_y[good], np.mean(nose_y))
            area6 = shoelace(x6, y6)
        else:
            area6 = 0

        print('\n%s %s frames %d-%d (%d frames, mean RH lk=%.2f)' % (
            animal, session, s, e, n_frames, mean_lk))
        print('  UCSF area:  %.2f' % ucsf_area)
        print('  Method 1 (RH only, self-close):        %8.2f  ratio=%.1fx' % (area1, area1/ucsf_area if ucsf_area > 0 else 0))
        print('  Method 2 (RH + mean nose close):       %8.2f  ratio=%.1fx' % (area2, area2/ucsf_area if ucsf_area > 0 else 0))
        print('  Method 3 (RH + nose[0] close):         %8.2f  ratio=%.1fx' % (area3, area3/ucsf_area if ucsf_area > 0 else 0))
        print('  Method 4 (nose[0] + RH + nose[-1]):     %8.2f  ratio=%.1fx' % (area4, area4/ucsf_area if ucsf_area > 0 else 0))
        print('  Method 5 (high-conf RH, self-close):   %8.2f  ratio=%.1fx' % (area5, area5/ucsf_area if ucsf_area > 0 else 0))
        print('  Method 6 (high-conf RH + nose close):  %8.2f  ratio=%.1fx' % (area6, area6/ucsf_area if ucsf_area > 0 else 0))


if __name__ == '__main__':
    main()
