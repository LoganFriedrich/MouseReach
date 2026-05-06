"""
Export full-resolution frames at peak reach extension for suspect mice.

For each suspect reach:
  - Find the frame of max RightHand_y (peak extension)
  - Take 4 frames centered on that: peak-1, peak, peak+1, peak+2
  - Only include frames where RH is past the nose (actually extended)
  - Export at full resolution, no scaling

Top 3 reaches by area per phase per suspect.
"""

import pandas as pd
import numpy as np
import os
import glob
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

SUSPECTS = ['M01', 'M05', 'M06', 'M08', 'M13', 'M14']
MAX_DURATION = 40

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

BP_COLORS = {
    'Nose': (0, 255, 0),
    'RightHand': (0, 0, 255),
    'RightEar': (255, 255, 0),
    'LeftEar': (0, 255, 255),
    'Pellet': (255, 0, 0),
    'Reference': (255, 0, 255),
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


def get_peak_frames(dlc, s, e):
    """Find 4 frames around peak extension where paw is past nose."""
    rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
    nose_y = dlc['Nose']['y'].iloc[s:e+1].values
    rh_lk = dlc['RightHand']['likelihood'].iloc[s:e+1].values

    # Find peak extension (max RH_y)
    peak_rel = np.argmax(rh_y)
    peak_abs = s + peak_rel

    # Take peak-1 through peak+2
    candidates = [peak_abs + offset for offset in [-1, 0, 1, 2]]

    # Filter: must be within reach range AND paw past nose AND good likelihood
    valid = []
    for f in candidates:
        if f < s or f > e:
            continue
        rel = f - s
        if rel < len(rh_y) and rh_y[rel] > nose_y[rel] and rh_lk[rel] > 0.3:
            valid.append(f)

    return valid


def extract_frame(cap, dlc, frame_num):
    """Extract one full-resolution frame with DLC overlay."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return None

    # Crop to region around nose + paw
    if frame_num < len(dlc):
        nose_x = dlc['Nose']['x'].iloc[frame_num]
        nose_y = dlc['Nose']['y'].iloc[frame_num]
        rh_x = dlc['RightHand']['x'].iloc[frame_num]
        rh_y = dlc['RightHand']['y'].iloc[frame_num]

        # Crop box: centered between nose and paw, padded
        pad = 60
        cx = int((nose_x + rh_x) / 2)
        cy = int((nose_y + rh_y) / 2)
        half_w = int(abs(rh_x - nose_x) / 2) + pad
        half_h = int(abs(rh_y - nose_y) / 2) + pad
        # Ensure minimum size
        half_w = max(half_w, 80)
        half_h = max(half_h, 80)

        x1 = max(0, cx - half_w)
        x2 = min(frame.shape[1], cx + half_w)
        y1 = max(0, cy - half_h)
        y2 = min(frame.shape[0], cy + half_h)
        frame = frame[y1:y2, x1:x2]

        # Draw small semi-transparent markers on cropped frame
        overlay = frame.copy()
        for bp, color in BP_COLORS.items():
            if bp in dlc.columns.get_level_values(0):
                bx = dlc[bp]['x'].iloc[frame_num]
                by = dlc[bp]['y'].iloc[frame_num]
                lk = dlc[bp]['likelihood'].iloc[frame_num]
                if lk > 0.3:
                    # Adjust coords to crop
                    bx_c = int(bx) - x1
                    by_c = int(by) - y1
                    if 0 <= bx_c < frame.shape[1] and 0 <= by_c < frame.shape[0]:
                        cv2.circle(overlay, (bx_c, by_c), 2, color, -1)
                        if bp == 'RightHand':
                            cv2.circle(overlay, (bx_c, by_c), 4, color, 1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, 'f%d' % frame_num, (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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

    dlc_cache = {}
    video_cache = {}

    for animal in SUSPECTS:
        print('\n=== %s ===' % animal)
        adf = df[df['SubjectID'] == animal]

        all_phase_strips = []

        for phase_name, phase_labels in PHASES.items():
            phase_df = adf[adf['Test_Type_Grouped_1'].isin(phase_labels)].copy()
            phase_df = phase_df[phase_df['Swipe_Duration_Frames'] <= MAX_DURATION]
            phase_df = phase_df.dropna(subset=['sd_start', 'sd_end', 'Swipe_area'])

            if len(phase_df) == 0:
                print('  %s: no valid reaches' % phase_name)
                continue

            top3 = phase_df.nlargest(3, 'Swipe_area')

            for rank, (_, row) in enumerate(top3.iterrows()):
                stem = session_to_stem(row['Session_ID'])
                video_path = find_video(stem)
                dlc_path = find_dlc(stem)

                if not video_path or not dlc_path:
                    continue

                if stem not in dlc_cache:
                    d = pd.read_hdf(dlc_path)
                    sc = d.columns.get_level_values(0)[0]
                    dlc_cache[stem] = d[sc]
                dlc = dlc_cache[stem]

                s, e = int(row['sd_start']), int(row['sd_end'])
                peak_frames = get_peak_frames(dlc, s, e)

                if not peak_frames:
                    continue

                # Open video
                if stem not in video_cache:
                    video_cache[stem] = cv2.VideoCapture(video_path)
                cap = video_cache[stem]

                frames = []
                for f in peak_frames:
                    img = extract_frame(cap, dlc, f)
                    if img is not None:
                        frames.append(img)

                if frames:
                    # Concatenate vertically at full resolution
                    strip = np.concatenate(frames, axis=0)
                    label = '%s %s #%d (area=%.0f)' % (animal, phase_name, rank+1, row['Swipe_area'])
                    all_phase_strips.append((strip, label))
                    print('  %s #%d: %d peak frames from %s' % (phase_name, rank+1, len(frames), stem))

        # Save all phases for this animal — each reach is a vertical column
        if all_phase_strips:
            n_strips = len(all_phase_strips)
            max_h = max(s[0].shape[0] for s in all_phase_strips)

            fig, axes = plt.subplots(1, n_strips,
                                      figsize=(3 * n_strips, max_h / 72 + 1))
            if n_strips == 1:
                axes = [axes]

            for i, (strip, label) in enumerate(all_phase_strips):
                axes[i].imshow(strip)
                axes[i].set_title(label, fontsize=7, fontweight='bold', rotation=0)
                axes[i].axis('off')

            fig.suptitle('%s: Peak Extension Frames (top 3 by area per phase)\n'
                         'Green=Nose, Red=Hand, Blue=Pellet, Magenta=Reference' % animal,
                         fontsize=11, fontweight='bold')
            plt.tight_layout()
            out = os.path.join(OUTPUT_DIR, '%s_peak_frames_cropped.png' % animal)
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print('  Saved: %s' % out)

    # Close video captures
    for cap in video_cache.values():
        cap.release()

    print('\nDone.')


if __name__ == '__main__':
    main()
