"""
Spaghetti plot for M05: all reach trajectories overlaid, nose-centered.
Pre-injury (final 3) vs Post-injury, side by side.
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ASPA_BASE = r'Y:\2_Connectome\Behavior\MouseReach_Pipeline\Analyzed\ASPA\M\Post-Processing'
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


def session_to_stem(sid):
    parts = sid.split('-')
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    return '%s_%s_%s' % (date_str, parts[0], parts[2])


def extract_trajectories(swipes_df, dlc_cache):
    """Extract nose-centered RH trajectories for all swipes."""
    trajectories = []

    for _, row in swipes_df.iterrows():
        s = row['sd_start']
        e = row['sd_end']
        if pd.isna(s):
            continue
        s, e = int(s), int(e)

        stem = session_to_stem(row['Session_ID'])
        if stem not in dlc_cache:
            h5 = glob.glob(os.path.join(ASPA_BASE, stem + 'DLC*.h5'))
            if not h5:
                dlc_cache[stem] = None
                continue
            d = pd.read_hdf(h5[0])
            sc = d.columns.get_level_values(0)[0]
            dlc_cache[stem] = d[sc]

        dlc = dlc_cache[stem]
        if dlc is None or e + 1 > len(dlc):
            continue

        rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
        rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
        rh_lk = dlc['RightHand']['likelihood'].iloc[s:e+1].values
        nose_x = dlc['Nose']['x'].iloc[s:e+1].values
        nose_y = dlc['Nose']['y'].iloc[s:e+1].values

        # Center on nose position at first frame
        cx = rh_x - nose_x[0]
        cy = rh_y - nose_y[0]

        # Only include if hand has decent tracking
        if np.mean(rh_lk) > 0.3 and len(cx) >= 3:
            trajectories.append((cx, cy))

    return trajectories


def plot_spaghetti(ax, trajectories, color, color_light, title):
    """Plot all trajectories overlaid with mean."""
    n = len(trajectories)

    # Individual traces
    alpha = max(0.02, min(0.15, 20.0 / max(n, 1)))
    for x, y in trajectories:
        ax.plot(x, y, color=color_light, alpha=alpha, linewidth=0.5)

    # Mean trajectory (time-normalize to same length first)
    n_pts = 30
    norm_x = []
    norm_y = []
    for x, y in trajectories:
        t_orig = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, n_pts)
        norm_x.append(np.interp(t_new, t_orig, x))
        norm_y.append(np.interp(t_new, t_orig, y))

    if norm_x:
        mean_x = np.mean(norm_x, axis=0)
        mean_y = np.mean(norm_y, axis=0)
        sem_x = np.std(norm_x, axis=0) / np.sqrt(len(norm_x))
        sem_y = np.std(norm_y, axis=0) / np.sqrt(len(norm_y))

        ax.plot(mean_x, mean_y, color=color, linewidth=3, alpha=0.95, zorder=10)

    ax.plot(0, 0, 'o', color='white', markersize=6, markeredgecolor='black',
            markeredgewidth=1, zorder=11)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('%s (n=%d)' % (title, n), fontsize=12, fontweight='bold')
    ax.set_xlabel('Lateral (px)', fontsize=10)
    ax.set_ylabel('Extension (px)', fontsize=10)
    ax.set_facecolor('black')


def main():
    print('Loading UCSF data...')
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    m05 = df[df['SubjectID'] == 'M05'].copy()

    pre = m05[m05['Test_Type_Grouped_1'].isin([
        '2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'])]
    post = m05[m05['Test_Type_Grouped_1'].str.contains('Post-injury', na=False)]

    dlc_cache = {}

    print('Extracting pre-injury trajectories...')
    pre_trajs = extract_trajectories(pre, dlc_cache)
    print('  Got %d trajectories' % len(pre_trajs))

    print('Extracting post-injury trajectories...')
    post_trajs = extract_trajectories(post, dlc_cache)
    print('  Got %d trajectories' % len(post_trajs))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    plot_spaghetti(ax1, pre_trajs, '#00BFFF', '#4488CC', 'M05 Pre-Injury')
    plot_spaghetti(ax2, post_trajs, '#FF4444', '#CC4444', 'M05 Post-Injury')

    # Match axis limits
    all_trajs = pre_trajs + post_trajs
    all_x = np.concatenate([t[0] for t in all_trajs])
    all_y = np.concatenate([t[1] for t in all_trajs])
    pad = 10
    xlim = (np.percentile(all_x, 1) - pad, np.percentile(all_x, 99) + pad)
    ylim = (np.percentile(all_y, 1) - pad, np.percentile(all_y, 99) + pad)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    fig.suptitle('M05 Reach Trajectories: All Pre-Injury vs All Post-Injury\n'
                 'Nose-centered (origin = nose at first frame). White dot = origin.\n'
                 'Bold line = mean trajectory.',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'M05_spaghetti.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
