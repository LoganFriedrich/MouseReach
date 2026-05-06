"""
Plot problem reaches: swipes with the largest frame-to-frame tracking jumps.
Uses UCSF Swipe_Duration frame ranges. Shows all RH positions.
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
UCSF_BASE = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_BASE = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'labeled_frames')


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
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    return '%s_%s_%s' % (date_str, parts[0], parts[2])


def load_dlc(stem, cache):
    if stem not in cache:
        h5 = glob.glob(os.path.join(VIDEO_DIR, stem + 'DLC*.h5'))
        if not h5:
            return None
        d = pd.read_hdf(h5[0])
        sc = d.columns.get_level_values(0)[0]
        cache[stem] = d[sc]
    return cache[stem]


def plot_reach(ax, dlc, s, e, label):
    rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
    rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
    rh_lk = dlc['RightHand']['likelihood'].iloc[s:e+1].values
    nose_x = dlc['Nose']['x'].iloc[s:e+1].values
    nose_y = dlc['Nose']['y'].iloc[s:e+1].values
    ref_x = dlc['Reference']['x'].iloc[s:e+1].values
    ref_y = dlc['Reference']['y'].iloc[s:e+1].values
    pil_x = dlc['Pillar']['x'].iloc[s:e+1].values
    pil_y = dlc['Pillar']['y'].iloc[s:e+1].values
    pil_lk = dlc['Pillar']['likelihood'].iloc[s:e+1].values
    pel_x = dlc['Pellet']['x'].iloc[s:e+1].values
    pel_y = dlc['Pellet']['y'].iloc[s:e+1].values
    pel_lk = dlc['Pellet']['likelihood'].iloc[s:e+1].values

    # Displacement per frame
    disp = np.sqrt(np.diff(rh_x)**2 + np.diff(rh_y)**2)

    mean_nose_y = np.nanmean(nose_y)
    mean_nose_x = np.nanmean(nose_x)

    ax.axhline(y=mean_nose_y, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.plot(np.nanmean(ref_x), np.nanmean(ref_y), 'k*', markersize=8, zorder=10)
    ax.plot(mean_nose_x, mean_nose_y, 'o', color='red', markersize=5,
            markeredgecolor='darkred', markeredgewidth=1, zorder=8)

    good_pil = pil_lk > 0.3
    if good_pil.any():
        ax.plot(np.nanmean(pil_x[good_pil]), np.nanmean(pil_y[good_pil]), 's',
                color='brown', markersize=6, markeredgecolor='black', markeredgewidth=0.5, zorder=6)

    good_pel = pel_lk > 0.3
    if good_pel.any():
        ax.plot(np.nanmean(pel_x[good_pel]), np.nanmean(pel_y[good_pel]), 'D',
                color='orange', markersize=5, markeredgecolor='black', markeredgewidth=0.5, zorder=6)

    # Every RH position
    for i in range(len(rh_x)):
        lk = rh_lk[i]
        color = (1 - lk, lk, 0)
        size = 20 if lk > 0.5 else 8
        ax.scatter(rh_x[i], rh_y[i], c=[color], s=size, zorder=5,
                   edgecolors='black', linewidths=0.3)

    # Connect frames, highlight big jumps in red
    for i in range(len(rh_x) - 1):
        if disp[i] > 50:
            ax.plot(rh_x[i:i+2], rh_y[i:i+2], '-', color='red', linewidth=2.5, alpha=0.9)
        elif disp[i] > 23:
            ax.plot(rh_x[i:i+2], rh_y[i:i+2], '-', color='orange', linewidth=1.5, alpha=0.7)
        else:
            lk = min(rh_lk[i], rh_lk[i+1])
            color = (1 - lk, lk, 0)
            ax.plot(rh_x[i:i+2], rh_y[i:i+2], '-', color=color, linewidth=0.8, alpha=0.6)

    # Number every frame
    for i in range(len(rh_x)):
        ax.annotate('%d' % i, (rh_x[i], rh_y[i]), fontsize=4,
                    textcoords='offset points', xytext=(3, 3),
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')])

    ax.plot(rh_x[0], rh_y[0], 'go', markersize=8, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10)
    ax.plot(rh_x[-1], rh_y[-1], 'rs', markersize=8, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10)

    ax.text(0.02, 0.02, 'RH lk=%.2f\nmax jump=%.0fpx' % (np.mean(rh_lk), max(disp) if len(disp) > 0 else 0),
            transform=ax.transAxes, fontsize=6,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=8, fontweight='bold')
    ax.set_xlabel('X (px)', fontsize=7)
    ax.set_ylabel('Y (px)', fontsize=7)
    ax.tick_params(labelsize=6)


def main():
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df = df[df['group'] == 'M'].copy()
    df['Swipe_area'] = pd.to_numeric(df['Swipe_area'], errors='coerce')
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    # Problem swipes (from displacement analysis)
    targets = [
        # Biggest jumps
        {'session': 'M06-45022-P1', 'start': 31098, 'end': 31108, 'max_disp': 221, 'area': 150},
        {'session': 'M14-44987-P1', 'start': 727, 'end': 744, 'max_disp': 184, 'area': 445},
        {'session': 'M14-45001-P1', 'start': 24080, 'end': 24117, 'max_disp': 48, 'area': 108},
        {'session': 'M08-45001-P2', 'start': 19801, 'end': 19833, 'max_disp': 33, 'area': 787},
        {'session': 'M13-44987-P1', 'start': 12419, 'end': 12432, 'max_disp': 33, 'area': 426},
        {'session': 'M05-45001-P2', 'start': 16848, 'end': 16868, 'max_disp': 31, 'area': 675},
    ]

    # Add 2 normal pre-injury for comparison
    pre = df[(df['SubjectID'] == 'M01') &
             (df['Test_Type_Grouped_1'] == '2_Pre-injury_1') &
             (df['Swipe_area'] > 80) & (df['Swipe_area'] < 250) &
             (df['sd_start'].notna())].sort_values('Swipe_area', ascending=False).head(2)

    for _, row in pre.iterrows():
        targets.insert(0, {
            'session': row['Session_ID'],
            'start': int(row['sd_start']),
            'end': int(row['sd_end']),
            'max_disp': 0,
            'area': row['Swipe_area'],
        })

    n = len(targets)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    axes = axes.flatten()

    dlc_cache = {}

    for idx, t in enumerate(targets):
        stem = session_to_stem(t['session'])
        dlc = load_dlc(stem, dlc_cache)
        if dlc is None:
            axes[idx].text(0.5, 0.5, 'No DLC: %s' % stem, transform=axes[idx].transAxes, ha='center')
            continue

        animal = t['session'].split('-')[0]
        label = '%s frames %d-%d\narea=%.0f, max_jump=%.0fpx' % (
            animal, t['start'], t['end'], t['area'], t['max_disp'])

        print('Plotting %s %d-%d' % (stem, t['start'], t['end']))
        plot_reach(axes[idx], dlc, t['start'], t['end'], label)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Problem Reaches: Swipes with Largest Tracking Jumps\n'
                 'Red lines = jumps >50px. Orange = >23px. First 2 = normal pre-injury.\n'
                 'Dot color = DLC confidence (green=high, red=low).',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'problem_reaches.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
