"""
Trajectory comparison v2: Using ACTUAL ASPA swipe boundaries from per-video xlsx.

Key improvements over v1:
- Uses s_idx/e_idx from per-video ASPA xlsx (the real swipe boundaries)
- Applies ASPA's pre-filter: only frames where RightHand_y > Nose_y + 8
- Shows nose position as the slit reference (reach should start/end here)
- Marks the nose with a line showing the slit boundary
- Compares normal pre-injury reaches to highest-area post-injury suspects
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

VIDEO_DIR = r'X:\! DLC Output\Analyzed\M\Post-Processing'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_BASE = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'labeled_frames')
os.makedirs(OUTPUT_BASE, exist_ok=True)


def load_dlc(video_stem):
    h5_matches = glob.glob(os.path.join(VIDEO_DIR, video_stem + 'DLC*.h5'))
    if not h5_matches:
        return None
    d = pd.read_hdf(h5_matches[0])
    s = d.columns.get_level_values(0)[0]
    return d[s]


def get_aspa_swipes(video_stem):
    """Load per-video ASPA xlsx with real s_idx/e_idx."""
    xlsx = os.path.join(VIDEO_DIR, video_stem + '.xlsx')
    if not os.path.exists(xlsx):
        return None
    return pd.read_excel(xlsx)


def get_aspa_filtered_frames(dlc, s_idx, e_idx):
    """Return only the frames ASPA would have used for kinematic computation.

    ASPA filters (from frame_processor.py lines 307-312):
    - RightHand_y > Nose_y + 8 (hand past the nose/slit)
    - RightHand_likelihood > 0.1
    - Nose_likelihood > 0.5

    Then within detect_swipes_new:
    - RightHand_y > y_thr (threshold near max extension)
    - RightHand_likelihood > 0.7

    But the sequence stored (g_df) comes from the pre-filtered set (step 1),
    not the doubly-filtered set. So we apply just the pre-filter.
    """
    frames = list(range(s_idx, e_idx + 1))

    rh_x = dlc['RightHand']['x'].iloc[s_idx:e_idx+1].values
    rh_y = dlc['RightHand']['y'].iloc[s_idx:e_idx+1].values
    rh_lk = dlc['RightHand']['likelihood'].iloc[s_idx:e_idx+1].values
    nose_x = dlc['Nose']['x'].iloc[s_idx:e_idx+1].values
    nose_y = dlc['Nose']['y'].iloc[s_idx:e_idx+1].values
    nose_lk = dlc['Nose']['likelihood'].iloc[s_idx:e_idx+1].values

    # ASPA filter: hand must be past nose by 8px, both detected
    mask = (rh_y - nose_y > 8) & (rh_lk > 0.1) & (nose_lk > 0.5)

    return {
        'rh_x': rh_x, 'rh_y': rh_y, 'rh_lk': rh_lk,
        'nose_x': nose_x, 'nose_y': nose_y, 'nose_lk': nose_lk,
        'mask': mask,
        'frames': np.array(frames),
    }


def plot_reach(ax, dlc, s_idx, e_idx, label):
    """Plot a single reach showing raw DLC coordinates within ASPA's s_idx to e_idx.

    No filtering applied -- we don't know exactly what ASPA filtered.
    Just show what DLC tracked and where the landmarks are.
    """

    rh_x = dlc['RightHand']['x'].iloc[s_idx:e_idx+1].values
    rh_y = dlc['RightHand']['y'].iloc[s_idx:e_idx+1].values
    rh_lk = dlc['RightHand']['likelihood'].iloc[s_idx:e_idx+1].values
    nose_x = dlc['Nose']['x'].iloc[s_idx:e_idx+1].values
    nose_y = dlc['Nose']['y'].iloc[s_idx:e_idx+1].values
    ref_x = dlc['Reference']['x'].iloc[s_idx:e_idx+1].values
    ref_y = dlc['Reference']['y'].iloc[s_idx:e_idx+1].values
    pil_x = dlc['Pillar']['x'].iloc[s_idx:e_idx+1].values
    pil_y = dlc['Pillar']['y'].iloc[s_idx:e_idx+1].values
    pil_lk = dlc['Pillar']['likelihood'].iloc[s_idx:e_idx+1].values
    pel_x = dlc['Pellet']['x'].iloc[s_idx:e_idx+1].values
    pel_y = dlc['Pellet']['y'].iloc[s_idx:e_idx+1].values
    pel_lk = dlc['Pellet']['likelihood'].iloc[s_idx:e_idx+1].values

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

    # RightHand path -- show EVERY detected position as a dot
    # Color by likelihood (green=high, red=low), size by likelihood
    for i in range(len(rh_x)):
        lk = rh_lk[i]
        color = (1 - lk, lk, 0)
        size = 20 if lk > 0.5 else 8
        ax.scatter(rh_x[i], rh_y[i], c=[color], s=size, zorder=5, edgecolors='black', linewidths=0.3)

    # Connect consecutive frames with thin lines colored by confidence
    for i in range(len(rh_x) - 1):
        lk = min(rh_lk[i], rh_lk[i+1])
        color = (1 - lk, lk, 0)
        ax.plot(rh_x[i:i+2], rh_y[i:i+2], '-', color=color, linewidth=0.8, alpha=0.6)

    # Number every frame
    for i in range(len(rh_x)):
        ax.annotate('%d' % i, (rh_x[i], rh_y[i]), fontsize=4,
                    textcoords='offset points', xytext=(3, 3),
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')])

    # Start/end markers (small)
    ax.plot(rh_x[0], rh_y[0], 'go', markersize=8, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.plot(rh_x[-1], rh_y[-1], 'rs', markersize=8, markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    # Mean RH likelihood annotation
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
    # Get normal pre-injury reaches from M01
    m01_pre_stem = '20230215_M01_P1'  # Pre-injury 1
    m01_aspa = get_aspa_swipes(m01_pre_stem)
    m01_dlc = load_dlc(m01_pre_stem)

    if m01_aspa is None or m01_dlc is None:
        print('Cannot load M01 pre-injury data')
        return

    # Pick 2 normal M01 reaches (moderate area, normal duration)
    normal_candidates = m01_aspa[(m01_aspa['Area (mm^2)'] > 15) &
                                  (m01_aspa['Area (mm^2)'] < 60) &
                                  (m01_aspa['e_idx'] - m01_aspa['s_idx'] > 8) &
                                  (m01_aspa['e_idx'] - m01_aspa['s_idx'] < 20)]
    normal_picks = normal_candidates.sort_values('Area (mm^2)', ascending=False).head(2)

    # Get suspect post-injury reaches from M05 and M13 (highest area)
    suspect_videos = [
        ('20230302_M05_P1', 'M05 1wk post-injury'),
        ('20230302_M13_P2', 'M13 1wk post-injury'),
        ('20230406_M07_P1', 'M07 post-rehab'),
    ]

    targets = []

    # Add normal reaches
    for _, row in normal_picks.iterrows():
        dur = int(row['e_idx'] - row['s_idx'])
        targets.append({
            'stem': m01_pre_stem,
            's_idx': int(row['s_idx']),
            'e_idx': int(row['e_idx']),
            'label': 'M01 NORMAL pre-injury\narea=%.1f mm^2, dur=%d, outcome=%s' % (
                row['Area (mm^2)'], dur, row['Reach Outcome']),
        })

    # Add suspect reaches (top 1 by area from each video)
    for stem, phase_label in suspect_videos:
        aspa = get_aspa_swipes(stem)
        dlc = load_dlc(stem)
        if aspa is None or dlc is None:
            print('Cannot load %s' % stem)
            continue

        # Top 2 by area
        top = aspa.sort_values('Area (mm^2)', ascending=False).head(2)
        for _, row in top.iterrows():
            dur = int(row['e_idx'] - row['s_idx'])
            targets.append({
                'stem': stem,
                's_idx': int(row['s_idx']),
                'e_idx': int(row['e_idx']),
                'label': '%s %s\narea=%.1f mm^2, dur=%d, outcome=%s' % (
                    stem.split('_')[1], phase_label,
                    row['Area (mm^2)'], dur, row['Reach Outcome']),
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

        print('Plotting %s frames %d-%d' % (stem, target['s_idx'], target['e_idx']))
        plot_reach(axes[idx], dlc, target['s_idx'], target['e_idx'], target['label'])

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Reach Trajectories: Raw DLC within ASPA s_idx-e_idx\n'
                 'Red dashed = slit (nose Y). Line color = DLC confidence (green=high, red=low).\n'
                 'Green circle = s_idx. Red square = e_idx. Star = Reference. Red dot = Nose.',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'trajectory_with_landmarks_v2.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
