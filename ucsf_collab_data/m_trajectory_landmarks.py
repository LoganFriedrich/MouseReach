"""
Trajectory comparison with landmarks: Reference, Nose, Ears, Pillar, Pellet.
Shows RightHand path + Nose path relative to cage reference.
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

base = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)


def get_normal_reach():
    """Find a typical M01 pre-injury reach for comparison."""
    df1 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    swipe_df = pd.concat([df1, df2], ignore_index=True)

    m01_pre = swipe_df[(swipe_df['SubjectID'] == 'M01') &
                        (swipe_df['Test_Type_Grouped_1'] == '2_Pre-injury_1')].copy()
    m01_pre['Swipe_area'] = pd.to_numeric(m01_pre['Swipe_area'], errors='coerce')
    m01_pre['Swipe_Duration_Frames'] = pd.to_numeric(m01_pre['Swipe_Duration_Frames'], errors='coerce')
    m01_pre['bodyparts_coords'] = pd.to_numeric(m01_pre['bodyparts_coords'], errors='coerce')
    typical = m01_pre[(m01_pre['Swipe_area'] > 100) & (m01_pre['Swipe_area'] < 200) &
                       (m01_pre['Swipe_Duration_Frames'] > 10) & (m01_pre['Swipe_Duration_Frames'] < 18)]
    row = typical.iloc[0]
    sid = row['Session_ID']
    parts = sid.split('-')
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    video_name = '%s_%s_%s.mp4' % (date_str, parts[0], parts[2])
    return {
        'video': video_name,
        'frame': int(row['bodyparts_coords']),
        'dur': int(row['Swipe_Duration_Frames']),
        'label': 'M01 NORMAL pre-injury (area=%.0f, dur=%d)' % (row['Swipe_area'], row['Swipe_Duration_Frames']),
    }


def load_dlc(video_name):
    video_stem = os.path.splitext(video_name)[0]
    h5_matches = glob.glob(os.path.join(VIDEO_DIR, video_stem + 'DLC*.h5'))
    if not h5_matches:
        return None, None
    d = pd.read_hdf(h5_matches[0])
    s = d.columns.get_level_values(0)[0]
    return d[s], d[s].columns.get_level_values(0).unique().tolist()


def plot_trajectory(ax, dlc, start, dur, label):
    """Plot RightHand + Nose paths with landmarks."""

    # RightHand
    rh_x = dlc['RightHand']['x'].iloc[start:start+dur].values
    rh_y = dlc['RightHand']['y'].iloc[start:start+dur].values

    # Nose
    nose_x = dlc['Nose']['x'].iloc[start:start+dur].values
    nose_y = dlc['Nose']['y'].iloc[start:start+dur].values

    # Reference (cage fixture)
    ref_x = dlc['Reference']['x'].iloc[start:start+dur].values
    ref_y = dlc['Reference']['y'].iloc[start:start+dur].values

    # Ears
    le_x = dlc['LeftEar']['x'].iloc[start:start+dur].values
    le_y = dlc['LeftEar']['y'].iloc[start:start+dur].values
    re_x = dlc['RightEar']['x'].iloc[start:start+dur].values
    re_y = dlc['RightEar']['y'].iloc[start:start+dur].values

    # Pillar + Pellet
    pil_x = dlc['Pillar']['x'].iloc[start:start+dur].values
    pil_y = dlc['Pillar']['y'].iloc[start:start+dur].values
    pil_lk = dlc['Pillar']['likelihood'].iloc[start:start+dur].values
    pel_x = dlc['Pellet']['x'].iloc[start:start+dur].values
    pel_y = dlc['Pellet']['y'].iloc[start:start+dur].values
    pel_lk = dlc['Pellet']['likelihood'].iloc[start:start+dur].values

    # Reference star (mean)
    ax.plot(np.nanmean(ref_x), np.nanmean(ref_y), 'k*', markersize=20, zorder=10,
            label='Reference (cage)')

    # Ear positions (mean)
    ax.plot(np.nanmean(le_x), np.nanmean(le_y), 'o', color='purple', markersize=8,
            markeredgecolor='black', markeredgewidth=1, label='LeftEar', zorder=6)
    ax.plot(np.nanmean(re_x), np.nanmean(re_y), 'o', color='magenta', markersize=8,
            markeredgecolor='black', markeredgewidth=1, label='RightEar', zorder=6)

    # Pillar (mean, if confident)
    good_pil = pil_lk > 0.3
    if good_pil.any():
        ax.plot(np.nanmean(pil_x[good_pil]), np.nanmean(pil_y[good_pil]), 's',
                color='brown', markersize=12, markeredgecolor='black', markeredgewidth=1,
                label='Pillar', zorder=6)

    # Pellet (mean, if confident)
    good_pel = pel_lk > 0.3
    if good_pel.any():
        ax.plot(np.nanmean(pel_x[good_pel]), np.nanmean(pel_y[good_pel]), 'D',
                color='orange', markersize=10, markeredgecolor='black', markeredgewidth=1,
                label='Pellet', zorder=6)

    # Nose path (thin red)
    ax.plot(nose_x, nose_y, '-', color='red', linewidth=2, alpha=0.7, label='Nose path')
    # Nose start/end
    ax.plot(nose_x[0], nose_y[0], 'o', color='red', markersize=8,
            markeredgecolor='darkred', markeredgewidth=1.5, zorder=7)

    # RightHand path (bold blue)
    ax.plot(rh_x, rh_y, '-', color='blue', linewidth=3, alpha=0.8, label='RightHand path')

    # Arrows on hand path
    step = max(1, dur // 8)
    for i in range(0, len(rh_x) - 1, step):
        dx = rh_x[min(i + 1, len(rh_x) - 1)] - rh_x[i]
        dy = rh_y[min(i + 1, len(rh_y) - 1)] - rh_y[i]
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            ax.annotate('', xy=(rh_x[i] + dx, rh_y[i] + dy), xytext=(rh_x[i], rh_y[i]),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Number frames on hand path
    for i in range(0, dur, max(1, dur // 6)):
        ax.plot(rh_x[i], rh_y[i], 'ko', markersize=4, zorder=7)
        ax.annotate('%d' % i, (rh_x[i], rh_y[i]), fontsize=7,
                    textcoords='offset points', xytext=(4, 4),
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Hand start/end
    ax.plot(rh_x[0], rh_y[0], 'go', markersize=14, markeredgecolor='black',
            markeredgewidth=2, zorder=10)
    ax.plot(rh_x[-1], rh_y[-1], 'rs', markersize=14, markeredgecolor='black',
            markeredgewidth=2, zorder=10)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.legend(fontsize=7, loc='best')


def main():
    normal = get_normal_reach()
    targets = [
        normal,
        {'video': '20230406_M07_P2.mp4', 'frame': 34449, 'dur': 66,
         'label': 'M07 EXTREME (area=914, dur=66)'},
        {'video': '20230302_M05_P1.mp4', 'frame': 26051, 'dur': 14,
         'label': 'M05 suspect (area=559, dur=14)'},
        {'video': '20230302_M05_P1.mp4', 'frame': 14531, 'dur': 22,
         'label': 'M05 suspect (area=566, dur=22)'},
        {'video': '20230302_M13_P1.mp4', 'frame': 12423, 'dur': 14,
         'label': 'M13 suspect (area=426, dur=14)'},
        {'video': '20230302_M13_P1.mp4', 'frame': 11796, 'dur': 14,
         'label': 'M13 suspect (area=346, dur=14)'},
    ]

    dlc_cache = {}
    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        if target['video'] not in dlc_cache:
            dlc, bp = load_dlc(target['video'])
            dlc_cache[target['video']] = dlc
            if idx == 0:
                print('Bodyparts: %s' % bp)
        dlc = dlc_cache[target['video']]

        if dlc is None:
            axes[idx].text(0.5, 0.5, 'No DLC', transform=axes[idx].transAxes, ha='center')
            continue

        print('Plotting %s f%d d%d' % (target['video'], target['frame'], target['dur']))
        plot_trajectory(axes[idx], dlc, target['frame'], target['dur'], target['label'])

    fig.suptitle('RightHand + Nose Trajectories with Landmarks\n'
                 'Star=Reference (cage), Circles=Ears, Square=Pillar, Diamond=Pellet\n'
                 'Blue=Hand path, Red=Nose path. Green circle=start, Red square=end.',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'trajectory_with_landmarks.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
