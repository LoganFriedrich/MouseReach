"""
Full provenance audit for every M swipe in the UCSF CSV.

For each swipe:
1. Parse Session_ID -> animal, date, tray
2. Find the corresponding DLC h5 file
3. Verify the Swipe_Duration frame range is within the video length
4. Pull DLC coordinates at those frames
5. Recompute area, length, breadth, path_length, speed, duration
6. Compare to UCSF CSV values
7. Flag any mismatch, missing file, or out-of-range frames

Output: per-swipe audit CSV with match/mismatch flags.
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
DLC_DIR = r'X:\! DLC Output\Analyzed\M\Single_Animal'
# Also check Y: as fallback
DLC_DIR_Y = r'Y:\2_Connectome\Behavior\MouseReach_Pipeline\Analyzed\ASPA\M\Post-Processing'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_DIR = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality')

TOLERANCE = 0.02  # 2% tolerance for kinematic match


def parse_frame_range(s):
    parts = str(s).strip().split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def session_to_stem(sid):
    parts = sid.split('-')
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    animal = parts[0]
    tray = parts[2]
    return '%s_%s_%s' % (date_str, animal, tray), date_str, animal, tray


def find_dlc_h5(stem):
    """Find DLC h5 file on X: or Y:."""
    for base in [DLC_DIR, DLC_DIR_Y]:
        h5s = glob.glob(os.path.join(base, stem + 'DLC*.h5'))
        if h5s:
            return h5s[0]
    return None


def compute_kinematics(dlc, s, e):
    """Recompute all 6 UCSF kinematics from DLC coordinates."""
    rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
    rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
    nose_x = dlc['Nose']['x'].iloc[s:e+1].values
    nose_y = dlc['Nose']['y'].iloc[s:e+1].values

    n_frames = len(rh_x)

    # Area: shoelace on RH path, self-closing
    area = 0.5 * abs(np.dot(rh_x, np.roll(rh_y, 1)) - np.dot(rh_y, np.roll(rh_x, 1)))

    # Length: max(RH_y) - Nose_y at frame of max RH_y
    peak_idx = np.argmax(rh_y)
    length = rh_y[peak_idx] - nose_y[peak_idx]

    # Breadth: max(RH_x) - min(RH_x)
    breadth = np.max(rh_x) - np.min(rh_x)

    # Path length: sum of frame-to-frame displacements
    path_length = np.sum(np.sqrt(np.diff(rh_x)**2 + np.diff(rh_y)**2))

    # Speed: (path/4) / ((n-1)/60)
    speed = (path_length / 4.0) / ((n_frames - 1) / 60.0) if n_frames > 1 else 0

    # Path over frames
    pof = path_length / n_frames if n_frames > 0 else 0

    # Duration
    duration = n_frames

    return {
        'area': area, 'length': length, 'breadth': breadth,
        'path_length': path_length, 'speed': speed, 'pof': pof,
        'duration': duration,
    }


def check_match(ucsf_val, computed_val, tolerance=TOLERANCE):
    """Check if two values match within tolerance."""
    if pd.isna(ucsf_val) or pd.isna(computed_val):
        return 'NA'
    if ucsf_val == 0 and computed_val == 0:
        return 'MATCH'
    if ucsf_val == 0:
        return 'MISMATCH'
    ratio = computed_val / ucsf_val
    if abs(ratio - 1.0) <= tolerance:
        return 'MATCH'
    else:
        return 'MISMATCH(%.3fx)' % ratio


def main():
    print('Loading UCSF M data...')
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    m = df[df['SubjectID'].str.startswith('M', na=False)].copy()

    for col in ['Swipe_length', 'Swipe_area', 'Swipe_breadth', 'Swipe_speed',
                'Path_length', 'Path_over_Frames', 'Swipe_Duration_Frames']:
        m[col] = pd.to_numeric(m[col], errors='coerce')

    m['sd_start'], m['sd_end'] = zip(*m['Swipe_Duration'].apply(parse_frame_range))

    print('Total M swipes: %d' % len(m))

    # Audit every swipe
    dlc_cache = {}
    results = []
    n_match = 0
    n_mismatch = 0
    n_missing_h5 = 0
    n_bad_frames = 0
    n_no_duration = 0

    for idx, row in m.iterrows():
        s = row['sd_start']
        e = row['sd_end']
        sid = row['Session_ID']

        if pd.isna(s) or pd.isna(e):
            results.append({
                'swipe_id': row.get('Swipe_ID', idx),
                'session_id': sid,
                'status': 'NO_DURATION',
            })
            n_no_duration += 1
            continue

        s, e = int(s), int(e)
        stem, date_str, animal, tray = session_to_stem(sid)

        # Check video file exists
        if stem not in dlc_cache:
            h5_path = find_dlc_h5(stem)
            if h5_path is None:
                dlc_cache[stem] = None
            else:
                d = pd.read_hdf(h5_path)
                sc = d.columns.get_level_values(0)[0]
                dlc_cache[stem] = (d[sc], h5_path)

        cached = dlc_cache[stem]
        if cached is None:
            results.append({
                'swipe_id': row.get('Swipe_ID', idx),
                'session_id': sid,
                'date': date_str,
                'animal': animal,
                'tray': tray,
                'status': 'MISSING_H5',
                'stem': stem,
            })
            n_missing_h5 += 1
            continue

        dlc, h5_path = cached

        # Check frame range
        if e + 1 > len(dlc):
            results.append({
                'swipe_id': row.get('Swipe_ID', idx),
                'session_id': sid,
                'date': date_str,
                'animal': animal,
                'tray': tray,
                'status': 'FRAMES_OUT_OF_RANGE',
                'sd_start': s,
                'sd_end': e,
                'video_length': len(dlc),
            })
            n_bad_frames += 1
            continue

        # Recompute kinematics
        computed = compute_kinematics(dlc, s, e)

        # Compare
        area_match = check_match(row['Swipe_area'], computed['area'])
        length_match = check_match(row['Swipe_length'], computed['length'])
        breadth_match = check_match(row['Swipe_breadth'], computed['breadth'])
        speed_match = check_match(row['Swipe_speed'], computed['speed'])
        path_match = check_match(row['Path_length'], computed['path_length'])
        pof_match = check_match(row['Path_over_Frames'], computed['pof'])
        dur_match = check_match(row['Swipe_Duration_Frames'], computed['duration'])

        all_checks = [area_match, length_match, breadth_match, speed_match,
                       path_match, pof_match, dur_match]
        all_ok = all(c in ('MATCH', 'NA') for c in all_checks)

        if all_ok:
            n_match += 1
            status = 'VERIFIED'
        else:
            n_mismatch += 1
            status = 'MISMATCH'

        results.append({
            'swipe_id': row.get('Swipe_ID', idx),
            'session_id': sid,
            'date': date_str,
            'animal': animal,
            'tray': tray,
            'phase': row.get('Test_Type_Grouped_1', ''),
            'status': status,
            'area_check': area_match,
            'length_check': length_match,
            'breadth_check': breadth_match,
            'speed_check': speed_match,
            'path_check': path_match,
            'pof_check': pof_match,
            'duration_check': dur_match,
            'ucsf_area': row['Swipe_area'],
            'computed_area': computed['area'],
            'ucsf_length': row['Swipe_length'],
            'computed_length': computed['length'],
            'sd_start': s,
            'sd_end': e,
        })

        # Progress
        total = len(m)
        done = n_match + n_mismatch + n_missing_h5 + n_bad_frames + n_no_duration
        if done % 2000 == 0:
            print('  %d/%d (%.0f%%) - %d verified, %d mismatch, %d missing, %d bad frames' % (
                done, total, 100*done/total, n_match, n_mismatch, n_missing_h5, n_bad_frames))

    # Summary
    total = len(m)
    print('\n=== PROVENANCE AUDIT RESULTS ===')
    print('Total swipes: %d' % total)
    print('VERIFIED:          %d (%.1f%%)' % (n_match, 100*n_match/total))
    print('MISMATCH:          %d (%.1f%%)' % (n_mismatch, 100*n_mismatch/total))
    print('MISSING H5:        %d (%.1f%%)' % (n_missing_h5, 100*n_missing_h5/total))
    print('FRAMES OUT OF RANGE: %d (%.1f%%)' % (n_bad_frames, 100*n_bad_frames/total))
    print('NO DURATION:       %d (%.1f%%)' % (n_no_duration, 100*n_no_duration/total))

    # Show mismatches by phase
    rdf = pd.DataFrame(results)
    if n_mismatch > 0:
        print('\nMISMATCHES BY PHASE:')
        mismatches = rdf[rdf['status'] == 'MISMATCH']
        print(mismatches.groupby('phase').size().to_string())
        print('\nFirst 10 mismatches:')
        for _, r in mismatches.head(10).iterrows():
            print('  %s %s: area=%s length=%s breadth=%s' % (
                r['session_id'], r['phase'], r['area_check'], r['length_check'], r['breadth_check']))

    if n_missing_h5 > 0:
        print('\nMISSING H5 FILES:')
        missing = rdf[rdf['status'] == 'MISSING_H5']
        print('  Unique stems: %s' % sorted(missing['stem'].unique()))

    # Save full audit
    out = os.path.join(OUTPUT_DIR, 'm_provenance_audit.csv')
    rdf.to_csv(out, index=False)
    print('\nSaved: %s' % out)


if __name__ == '__main__':
    main()
