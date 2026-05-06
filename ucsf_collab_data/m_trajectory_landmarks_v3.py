"""
Trajectory comparison v3: Using UCSF CSV Swipe_Duration frame ranges.

This uses the actual frame ranges from the UCSF data (Swipe_Duration field),
NOT the per-video ASPA xlsx s_idx/e_idx which we confirmed do NOT match.

Shows raw DLC RightHand positions for every frame in the UCSF swipe range,
with nose, reference, pillar, pellet landmarks.
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from datetime import datetime, timedelta

VIDEO_DIR = r'X:\! DLC Output\Analyzed\M\Post-Processing'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_BASE = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'labeled_frames')
os.makedirs(OUTPUT_BASE, exist_ok=True)

UCSF_BASE = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)


def serial_to_date(serial):
    return (datetime(1899, 12, 30) + timedelta(days=int(serial))).strftime('%Y%m%d')


def parse_frame_range(swipe_duration_str):
    """Parse 'start-end' from Swipe_Duration field."""
    parts = str(swipe_duration_str).strip().split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def session_to_video_stem(session_id):
    """Convert session ID like M05-44987-P1 to video stem like 20230302_M05_P1."""
    parts = session_id.split('-')
    animal = parts[0]
    date_str = serial_to_date(int(parts[1]))
    tray = parts[2]
    return '%s_%s_%s' % (date_str, animal, tray)


def load_dlc(video_stem):
    h5_matches = glob.glob(os.path.join(VIDEO_DIR, video_stem + 'DLC*.h5'))
    if not h5_matches:
        return None
    d = pd.read_hdf(h5_matches[0])
    s = d.columns.get_level_values(0)[0]
    return d[s]


def plot_reach(ax, dlc, start_frame, end_frame, label):
    """Plot raw DLC coordinates for every frame in the UCSF swipe range."""

    rh_x = dlc['RightHand']['x'].iloc[start_frame:end_frame+1].values
    rh_y = dlc['RightHand']['y'].iloc[start_frame:end_frame+1].values
    rh_lk = dlc['RightHand']['likelihood'].iloc[start_frame:end_frame+1].values
    nose_x = dlc['Nose']['x'].iloc[start_frame:end_frame+1].values
    nose_y = dlc['Nose']['y'].iloc[start_frame:end_frame+1].values
    ref_x = dlc['Reference']['x'].iloc[start_frame:end_frame+1].values
    ref_y = dlc['Reference']['y'].iloc[start_frame:end_frame+1].values
    pil_x = dlc['Pillar']['x'].iloc[start_frame:end_frame+1].values
    pil_y = dlc['Pillar']['y'].iloc[start_frame:end_frame+1].values
    pil_lk = dlc['Pillar']['likelihood'].iloc[start_frame:end_frame+1].values
    pel_x = dlc['Pellet']['x'].iloc[start_frame:end_frame+1].values
    pel_y = dlc['Pellet']['y'].iloc[start_frame:end_frame+1].values
    pel_lk = dlc['Pellet']['likelihood'].iloc[start_frame:end_frame+1].values

    mean_nose_y = np.nanmean(nose_y)
    mean_nose_x = np.nanmean(nose_x)

    # Slit line at nose Y
    ax.axhline(y=mean_nose_y, color='red', linewidth=1, linestyle='--', alpha=0.5)

    # Reference (small)
    ax.plot(np.nanmean(ref_x), np.nanmean(ref_y), 'k*', markersize=8, zorder=10)

    # Nose (small)
    ax.plot(mean_nose_x, mean_nose_y, 'o', color='red', markersize=5,
            markeredgecolor='darkred', markeredgewidth=1, zorder=8)

    # Pillar (small, if confident)
    good_pil = pil_lk > 0.3
    if good_pil.any():
        ax.plot(np.nanmean(pil_x[good_pil]), np.nanmean(pil_y[good_pil]), 's',
                color='brown', markersize=6, markeredgecolor='black', markeredgewidth=0.5, zorder=6)

    # Pellet (small, if confident)
    good_pel = pel_lk > 0.3
    if good_pel.any():
        ax.plot(np.nanmean(pel_x[good_pel]), np.nanmean(pel_y[good_pel]), 'D',
                color='orange', markersize=5, markeredgecolor='black', markeredgewidth=0.5, zorder=6)

    # Every RightHand position as a dot, colored by likelihood
    for i in range(len(rh_x)):
        lk = rh_lk[i]
        color = (1 - lk, lk, 0)
        size = 20 if lk > 0.5 else 8
        ax.scatter(rh_x[i], rh_y[i], c=[color], s=size, zorder=5,
                   edgecolors='black', linewidths=0.3)

    # Connect consecutive frames
    for i in range(len(rh_x) - 1):
        lk = min(rh_lk[i], rh_lk[i+1])
        color = (1 - lk, lk, 0)
        ax.plot(rh_x[i:i+2], rh_y[i:i+2], '-', color=color, linewidth=0.8, alpha=0.6)

    # Number every frame
    for i in range(len(rh_x)):
        ax.annotate('%d' % i, (rh_x[i], rh_y[i]), fontsize=4,
                    textcoords='offset points', xytext=(3, 3),
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')])

    # Start/end markers
    ax.plot(rh_x[0], rh_y[0], 'go', markersize=8, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10)
    ax.plot(rh_x[-1], rh_y[-1], 'rs', markersize=8, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10)

    # Mean RH likelihood
    mean_lk = np.mean(rh_lk)
    ax.text(0.02, 0.02, 'RH lk=%.2f' % mean_lk, transform=ax.transAxes, fontsize=6,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=8, fontweight='bold')
    ax.set_xlabel('X (px)', fontsize=7)
    ax.set_ylabel('Y (px)', fontsize=7)
    ax.tick_params(labelsize=6)


def main():
    # Load UCSF data
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df = df[df['group'] == 'M'].copy()
    df['Swipe_area'] = pd.to_numeric(df['Swipe_area'], errors='coerce')
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    # Get 2 normal pre-injury reaches from M01 (moderate area, Pre-injury sessions)
    m01_pre = df[(df['SubjectID'] == 'M01') &
                 (df['Test_Type_Grouped_1'] == '2_Pre-injury_1') &
                 (df['Swipe_area'] > 80) & (df['Swipe_area'] < 250) &
                 (df['sd_start'].notna())].sort_values('Swipe_area', ascending=False)

    # Get problem swipes from suspect animals (highest area, post-injury)
    suspects_post = df[(df['SubjectID'].isin(['M05', 'M13', 'M14'])) &
                       (df['Test_Type_Grouped_1'].str.contains('Post-injury', na=False)) &
                       (df['sd_start'].notna())].sort_values('Swipe_area', ascending=False)

    targets = []

    # 2 normal reaches
    for _, row in m01_pre.head(2).iterrows():
        stem = session_to_video_stem(row['Session_ID'])
        targets.append({
            'session_id': row['Session_ID'],
            'stem': stem,
            'start': int(row['sd_start']),
            'end': int(row['sd_end']),
            'label': 'M01 NORMAL pre-injury\nframes %d-%d, area=%.0f, dur=%d\n%s' % (
                row['sd_start'], row['sd_end'], row['Swipe_area'],
                row['sd_end'] - row['sd_start'] + 1, row['Reach_outcome']),
        })

    # 6 problem swipes (top 2 each from M05, M13, M14)
    for animal in ['M05', 'M13', 'M14']:
        animal_swipes = suspects_post[suspects_post['SubjectID'] == animal].head(2)
        for _, row in animal_swipes.iterrows():
            stem = session_to_video_stem(row['Session_ID'])
            targets.append({
                'session_id': row['Session_ID'],
                'stem': stem,
                'start': int(row['sd_start']),
                'end': int(row['sd_end']),
                'label': '%s post-injury\nframes %d-%d, area=%.0f, dur=%d\n%s' % (
                    animal, row['sd_start'], row['sd_end'], row['Swipe_area'],
                    row['sd_end'] - row['sd_start'] + 1, row['Reach_outcome']),
            })

    n = len(targets)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes = axes.flatten()

    dlc_cache = {}

    for idx, target in enumerate(targets):
        stem = target['stem']
        if stem not in dlc_cache:
            dlc_cache[stem] = load_dlc(stem)
        dlc = dlc_cache[stem]

        if dlc is None:
            print('No DLC for %s' % stem)
            axes[idx].text(0.5, 0.5, 'No DLC: %s' % stem, transform=axes[idx].transAxes, ha='center')
            continue

        print('Plotting %s frames %d-%d (area=%.0f)' % (
            stem, target['start'], target['end'],
            target['end'] - target['start']))
        plot_reach(axes[idx], dlc, target['start'], target['end'], target['label'])

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Reach Trajectories from UCSF Swipe_Duration Frame Ranges\n'
                 'Red dashed = slit (nose Y). Dot color = DLC confidence (green=high, red=low).\n'
                 'Green circle = first frame. Red square = last frame.',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'trajectory_with_landmarks_v3.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
