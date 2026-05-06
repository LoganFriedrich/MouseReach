"""
Export frame-by-frame reach sequences for suspect mice.

For each suspect (M05, M06, M13, M14):
  - Top 3 area reaches pre-injury
  - Top 3 area reaches at each post-injury phase
  - Every frame in the reach (unless duration > 40 frames, then skip)
  - Export as labeled frame collages with DLC bodypart markers

Each reach becomes a horizontal strip of frames with DLC points overlaid.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

SUSPECTS = ['M05', 'M06', 'M13', 'M14']
MAX_DURATION = 40  # skip reaches longer than this

UCSF_BASE = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)
VIDEO_DIR = r'X:\! DLC Output\Analyzed\M\Single_Animal'
DLC_DIR = r'X:\! DLC Output\Analyzed\M\Single_Animal'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_DIR = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'reach_sequences')

PHASES = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    '1wk Post': ['3_1wk_Post-injury'],
    '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}

# DLC bodypart colors
BP_COLORS = {
    'Nose': (0, 255, 0),        # green
    'RightHand': (255, 0, 0),   # red (BGR: blue)
    'RightEar': (255, 255, 0),  # cyan
    'LeftEar': (0, 255, 255),   # yellow
    'Pellet': (0, 0, 255),      # red
    'Reference': (255, 0, 255), # magenta
}


def parse_frame_range(s):
    parts = str(s).strip().split('-')
    if len(parts) >= 2:
        try:
            return int(parts[0]), int(parts[1].split(' ')[0])
        except ValueError:
            pass
    return None, None


def session_to_stem(sid):
    parts = sid.split('-')
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    return '%s_%s_%s' % (date_str, parts[0], parts[2])


def find_video(stem):
    vids = glob.glob(os.path.join(VIDEO_DIR, stem + '.mp4'))
    return vids[0] if vids else None


def find_dlc(stem):
    h5s = glob.glob(os.path.join(DLC_DIR, stem + 'DLC*.h5'))
    return h5s[0] if h5s else None


def extract_frames_with_dlc(video_path, dlc_path, start_frame, end_frame):
    """Extract video frames and overlay DLC markers."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    dlc = pd.read_hdf(dlc_path)
    sc = dlc.columns.get_level_values(0)[0]
    dlc = dlc[sc]

    frames = []
    for frame_num in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay DLC markers
        if frame_num < len(dlc):
            for bp, color in BP_COLORS.items():
                if bp in dlc.columns.get_level_values(0):
                    x = dlc[bp]['x'].iloc[frame_num]
                    y = dlc[bp]['y'].iloc[frame_num]
                    lk = dlc[bp]['likelihood'].iloc[frame_num]
                    if lk > 0.3:
                        radius = 4 if bp == 'RightHand' else 3
                        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                        if bp == 'RightHand':
                            cv2.circle(frame, (int(x), int(y)), radius + 2, color, 1)

        # Add frame number label
        cv2.putText(frame, str(frame_num), (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def make_reach_strip(frames, title, max_width=2400):
    """Create a horizontal strip of frames for one reach."""
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    n = len(frames)

    # Scale down if total width would be too large
    scale = min(1.0, max_width / (w * n))
    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale < 1.0:
        frames = [cv2.resize(f, (new_w, new_h)) for f in frames]

    strip = np.concatenate(frames, axis=1)
    return strip, title


def main():
    print('Loading UCSF data...')
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['SubjectID'].isin(SUSPECTS)].copy()
    df['Swipe_area'] = pd.to_numeric(df['Swipe_area'], errors='coerce')
    df['Swipe_Duration_Frames'] = pd.to_numeric(df['Swipe_Duration_Frames'], errors='coerce')
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for animal in SUSPECTS:
        print('\n=== %s ===' % animal)
        adf = df[df['SubjectID'] == animal]

        for phase_name, phase_labels in PHASES.items():
            phase_df = adf[adf['Test_Type_Grouped_1'].isin(phase_labels)].copy()
            # Filter to reasonable duration
            phase_df = phase_df[phase_df['Swipe_Duration_Frames'] <= MAX_DURATION]
            phase_df = phase_df.dropna(subset=['sd_start', 'sd_end', 'Swipe_area'])

            if len(phase_df) == 0:
                print('  %s: no valid reaches' % phase_name)
                continue

            # Top 3 by area
            top3 = phase_df.nlargest(3, 'Swipe_area')
            print('  %s: top 3 areas = %s' % (phase_name,
                  ', '.join('%.0f' % a for a in top3['Swipe_area'].values)))

            strips = []
            for rank, (_, row) in enumerate(top3.iterrows()):
                stem = session_to_stem(row['Session_ID'])
                video = find_video(stem)
                dlc = find_dlc(stem)

                if not video or not dlc:
                    print('    #%d: missing video or DLC for %s' % (rank+1, stem))
                    continue

                s, e = int(row['sd_start']), int(row['sd_end'])
                n_frames = e - s + 1
                title = '%s %s #%d: frames %d-%d (%d frames), area=%.0f' % (
                    animal, phase_name, rank+1, s, e, n_frames, row['Swipe_area'])

                print('    #%d: %s frames %d-%d (%d frames)' % (rank+1, stem, s, e, n_frames))

                frames = extract_frames_with_dlc(video, dlc, s, e)
                if frames:
                    result = make_reach_strip(frames, title)
                    if result:
                        strips.append(result)

            if strips:
                # Stack strips vertically
                max_w = max(s[0].shape[1] for s in strips)
                padded = []
                for strip_img, strip_title in strips:
                    if strip_img.shape[1] < max_w:
                        pad = np.zeros((strip_img.shape[0], max_w - strip_img.shape[1], 3), dtype=np.uint8)
                        strip_img = np.concatenate([strip_img, pad], axis=1)
                    padded.append(strip_img)

                combined = np.concatenate(padded, axis=0)

                # Save
                out = os.path.join(OUTPUT_DIR, '%s_%s_top3.png' % (
                    animal, phase_name.replace(' ', '_').replace('-', '')))
                fig, ax = plt.subplots(figsize=(combined.shape[1]/100, combined.shape[0]/100))
                ax.imshow(combined)
                ax.axis('off')
                ax.set_title('%s %s - Top 3 reaches by area\n'
                             'Green=Nose, Red=RightHand, Blue=Pellet, Magenta=Reference' % (
                             animal, phase_name), fontsize=10, fontweight='bold')
                plt.tight_layout()
                plt.savefig(out, dpi=100, bbox_inches='tight')
                plt.close()
                print('    Saved: %s' % out)

    print('\nDone.')


if __name__ == '__main__':
    main()
