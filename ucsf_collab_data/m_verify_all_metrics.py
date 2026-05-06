"""
Verify ALL kinematic metrics against DLC data using UCSF Swipe_Duration frame ranges.
For each metric, test multiple calculation methods to find which matches the UCSF value.
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
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)

    for col in ['Swipe_area', 'Path_length', 'Path_over_Frames', 'Swipe_length',
                'Swipe_breadth', 'Swipe_speed', 'Swipe_Duration_Frames']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    # Pick a mix of swipes to test
    m01_pre = df[(df['SubjectID'] == 'M01') &
                 (df['Test_Type_Grouped_1'] == '2_Pre-injury_1') &
                 (df['Swipe_area'] > 80) & (df['sd_start'].notna())
                 ].sort_values('Swipe_area', ascending=False).head(2)

    suspects = df[(df['SubjectID'].isin(['M05'])) &
                  (df['Test_Type_Grouped_1'].str.contains('Post-injury', na=False)) &
                  (df['sd_start'].notna())].sort_values('Swipe_area', ascending=False).head(2)

    test_swipes = pd.concat([m01_pre, suspects])

    dlc_cache = {}

    for _, row in test_swipes.iterrows():
        s = int(row['sd_start'])
        e = int(row['sd_end'])
        session = row['Session_ID']
        animal = row['SubjectID']
        n_frames = e - s + 1

        stem = session_to_stem(session)
        if stem not in dlc_cache:
            h5 = glob.glob(os.path.join(VIDEO_DIR, stem + 'DLC*.h5'))
            if not h5:
                print('%s: No DLC file' % stem)
                continue
            d = pd.read_hdf(h5[0])
            sc = d.columns.get_level_values(0)[0]
            dlc_cache[stem] = d[sc]
        dlc = dlc_cache[stem]

        rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
        rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
        nose_x = dlc['Nose']['x'].iloc[s:e+1].values
        nose_y = dlc['Nose']['y'].iloc[s:e+1].values

        print('\n%s %s frames %d-%d (%d frames)' % (animal, session, s, e, n_frames))
        print('=' * 80)

        # --- AREA (already confirmed: Method 1 = shoelace on RH, self-close) ---
        area_calc = shoelace(rh_x, rh_y)
        print('Swipe_area:      UCSF=%.2f  calc=%.2f  match=%s' % (
            row['Swipe_area'], area_calc,
            'YES' if abs(area_calc - row['Swipe_area']) < 0.1 else 'NO (%.2fx)' % (area_calc / row['Swipe_area'])))

        # --- PATH_LENGTH: total distance traveled frame-to-frame ---
        dx = np.diff(rh_x)
        dy = np.diff(rh_y)
        path_sum = np.sum(np.sqrt(dx**2 + dy**2))
        print('Path_length:     UCSF=%.2f  calc=%.2f  match=%s' % (
            row['Path_length'], path_sum,
            'YES' if abs(path_sum - row['Path_length']) < 0.1 else 'NO (%.2fx)' % (path_sum / row['Path_length'] if row['Path_length'] > 0 else 0)))

        # --- PATH_OVER_FRAMES: path / duration ---
        pof_calc = path_sum / n_frames if n_frames > 0 else 0
        # Also try path / (n_frames - 1)
        pof_calc2 = path_sum / (n_frames - 1) if n_frames > 1 else 0
        print('Path_over_Frames: UCSF=%.6f' % row['Path_over_Frames'])
        print('  path/N_frames:       %.6f  match=%s' % (
            pof_calc, 'YES' if abs(pof_calc - row['Path_over_Frames']) < 0.001 else 'NO'))
        print('  path/(N_frames-1):   %.6f  match=%s' % (
            pof_calc2, 'YES' if abs(pof_calc2 - row['Path_over_Frames']) < 0.001 else 'NO'))

        # --- SWIPE_DURATION_FRAMES ---
        print('Duration_Frames: UCSF=%d  computed=%d  match=%s' % (
            row['Swipe_Duration_Frames'], n_frames,
            'YES' if n_frames == row['Swipe_Duration_Frames'] else 'NO'))

        # --- SWIPE_LENGTH: max_y - nose_y at max_y frame ---
        max_y_idx = np.argmax(rh_y)
        length_1 = rh_y[max_y_idx] - nose_y[max_y_idx]  # max_rh_y - nose_y at same frame
        length_2 = np.max(rh_y) - np.min(rh_y)  # y range of RH
        length_3 = np.max(rh_y) - np.mean(nose_y)  # max_rh_y - mean nose_y
        print('Swipe_length:    UCSF=%.2f' % row['Swipe_length'])
        print('  max_rh_y - nose_y[same frame]:  %.2f  match=%s' % (
            length_1, 'YES' if abs(length_1 - row['Swipe_length']) < 0.1 else 'NO'))
        print('  rh_y range (max-min):           %.2f  match=%s' % (
            length_2, 'YES' if abs(length_2 - row['Swipe_length']) < 0.1 else 'NO'))
        print('  max_rh_y - mean_nose_y:         %.2f  match=%s' % (
            length_3, 'YES' if abs(length_3 - row['Swipe_length']) < 0.1 else 'NO'))

        # --- SWIPE_BREADTH: max_x - min_x ---
        breadth_1 = np.max(rh_x) - np.min(rh_x)
        print('Swipe_breadth:   UCSF=%.2f  calc=%.2f  match=%s' % (
            row['Swipe_breadth'], breadth_1,
            'YES' if abs(breadth_1 - row['Swipe_breadth']) < 0.1 else 'NO'))

        # --- SWIPE_SPEED: distance / time ---
        time_sec = n_frames / 60.0
        speed_1 = (path_sum / 4) / time_sec  # path in mm / time in sec
        speed_2 = path_sum / time_sec  # path in px / time in sec
        time_sec2 = (n_frames - 1) / 60.0
        speed_3 = (path_sum / 4) / time_sec2 if time_sec2 > 0 else 0
        print('Swipe_speed:     UCSF=%.2f' % row['Swipe_speed'])
        print('  (path_px/4) / (N/60):      %.2f  match=%s' % (
            speed_1, 'YES' if abs(speed_1 - row['Swipe_speed']) < 0.1 else 'NO'))
        print('  path_px / (N/60):          %.2f  match=%s' % (
            speed_2, 'YES' if abs(speed_2 - row['Swipe_speed']) < 0.1 else 'NO'))
        print('  (path_px/4) / ((N-1)/60):  %.2f  match=%s' % (
            speed_3, 'YES' if abs(speed_3 - row['Swipe_speed']) < 0.1 else 'NO'))


if __name__ == '__main__':
    main()
