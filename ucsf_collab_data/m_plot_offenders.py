"""
Plot the specific swipes that are dragging M's LASSO results toward 'better after injury'.
These are post-injury swipes with area > pre-injury mean for each suspect animal.
Top 4 per animal = 16 total panels.
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

ASPA_BASE = r'Y:\2_Connectome\Behavior\MouseReach_Pipeline\Analyzed\ASPA'
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


def session_to_stem(session_id):
    parts = session_id.split('-')
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    return '%s_%s_%s' % (date_str, parts[0], parts[2])


def load_dlc(stem, cache):
    if stem not in cache:
        h5 = glob.glob(os.path.join(ASPA_BASE, 'M', 'Post-Processing', stem + 'DLC*.h5'))
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

    disp = np.sqrt(np.diff(rh_x)**2 + np.diff(rh_y)**2)
    dist_nose = np.sqrt((rh_x - nose_x)**2 + (rh_y - nose_y)**2)

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

    for i in range(len(rh_x)):
        lk = rh_lk[i]
        color = (1 - lk, lk, 0)
        size = 20 if lk > 0.5 else 8
        ax.scatter(rh_x[i], rh_y[i], c=[color], s=size, zorder=5,
                   edgecolors='black', linewidths=0.3)

    for i in range(len(rh_x) - 1):
        if disp[i] > 50:
            ax.plot(rh_x[i:i+2], rh_y[i:i+2], '-', color='red', linewidth=2.5, alpha=0.9)
        elif disp[i] > 23:
            ax.plot(rh_x[i:i+2], rh_y[i:i+2], '-', color='orange', linewidth=1.5, alpha=0.7)
        else:
            lk = min(rh_lk[i], rh_lk[i+1])
            color = (1 - lk, lk, 0)
            ax.plot(rh_x[i:i+2], rh_y[i:i+2], '-', color=color, linewidth=0.8, alpha=0.6)

    for i in range(len(rh_x)):
        ax.annotate('%d' % i, (rh_x[i], rh_y[i]), fontsize=4,
                    textcoords='offset points', xytext=(3, 3),
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')])

    ax.plot(rh_x[0], rh_y[0], 'go', markersize=8, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10)
    ax.plot(rh_x[-1], rh_y[-1], 'rs', markersize=8, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10)

    max_disp = max(disp) if len(disp) > 0 else 0
    min_dist = min(dist_nose) if len(dist_nose) > 0 else 0
    ax.text(0.02, 0.02, 'lk=%.2f  maxJ=%.0f  minDN=%.0f' % (
        np.mean(rh_lk), max_disp, min_dist),
            transform=ax.transAxes, fontsize=5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=7, fontweight='bold')
    ax.set_xlabel('X (px)', fontsize=6)
    ax.set_ylabel('Y (px)', fontsize=6)
    ax.tick_params(labelsize=5)


def main():
    # Top offenders from the analysis above
    targets = [
        # M05 top 4
        ('M05', 'M05-45001-P1', 20268, 20284, 812.7),
        ('M05', 'M05-45001-P1', 7254, 7273, 791.7),
        ('M05', 'M05-45001-P1', 25975, 25991, 773.7),
        ('M05', 'M05-45001-P2', 7105, 7126, 762.8),
        # M06 top 4
        ('M06', 'M06-44987-P1', 15474, 15488, 396.2),
        ('M06', 'M06-45001-P1', 13757, 13776, 327.7),
        ('M06', 'M06-45001-P2', 19354, 19369, 326.5),
        ('M06', 'M06-45001-P2', 33793, 33810, 324.9),
        # M13 top 4
        ('M13', 'M13-44987-P1', 12419, 12432, 425.6),
        ('M13', 'M13-45001-P2', 30486, 30506, 424.2),
        ('M13', 'M13-44987-P1', 28942, 28954, 420.8),
        ('M13', 'M13-44987-P1', 13264, 13277, 350.3),
        # M14 top 4
        ('M14', 'M14-45001-P2', 19822, 19847, 822.6),
        ('M14', 'M14-45001-P2', 3851, 3871, 729.9),
        ('M14', 'M14-45001-P2', 34974, 34987, 654.3),
        ('M14', 'M14-45001-P2', 19401, 19431, 624.0),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(28, 28))
    axes = axes.flatten()
    dlc_cache = {}

    for idx, (animal, session, s, e, area) in enumerate(targets):
        stem = session_to_stem(session)
        dlc = load_dlc(stem, dlc_cache)
        if dlc is None:
            axes[idx].text(0.5, 0.5, 'No DLC', transform=axes[idx].transAxes, ha='center')
            continue

        dur = e - s + 1
        label = '%s %d-%d dur=%d\narea=%.0f (session %s)' % (animal, s, e, dur, area, session.split('-')[1])
        print('Plotting %s %d-%d area=%.0f' % (animal, s, e, area))
        plot_reach(axes[idx], dlc, s, e, label)

    fig.suptitle('THE OFFENDERS: Post-injury swipes with highest area for each suspect M animal\n'
                 'These are the data points making LASSO call M "better after injury"\n'
                 'Red lines = jumps >50px. Orange = >23px.',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'offenders.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
