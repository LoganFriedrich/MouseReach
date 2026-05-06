"""
M05 spaghetti overlay: pre and post trajectories on the same axes.
Pre in blue, post in red.
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
        cx = rh_x - nose_x[0]
        cy = rh_y - nose_y[0]
        if np.mean(rh_lk) > 0.3 and len(cx) >= 3:
            trajectories.append((cx, cy))
    return trajectories


def compute_mean_traj(trajectories, n_pts=30):
    norm_x = []
    norm_y = []
    for x, y in trajectories:
        t_orig = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, n_pts)
        norm_x.append(np.interp(t_new, t_orig, x))
        norm_y.append(np.interp(t_new, t_orig, y))
    if norm_x:
        return np.mean(norm_x, axis=0), np.mean(norm_y, axis=0)
    return None, None


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

    print('Extracting trajectories...')
    pre_trajs = extract_trajectories(pre, dlc_cache)
    post_trajs = extract_trajectories(post, dlc_cache)
    print('  Pre: %d, Post: %d' % (len(pre_trajs), len(post_trajs)))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('black')

    # Pre traces in blue
    pre_alpha = max(0.02, min(0.15, 20.0 / max(len(pre_trajs), 1)))
    for x, y in pre_trajs:
        ax.plot(x, y, color='#4488CC', alpha=pre_alpha, linewidth=0.5)

    # Post traces in red
    post_alpha = max(0.02, min(0.15, 20.0 / max(len(post_trajs), 1)))
    for x, y in post_trajs:
        ax.plot(x, y, color='#CC4444', alpha=post_alpha, linewidth=0.5)

    # Mean trajectories
    pre_mx, pre_my = compute_mean_traj(pre_trajs)
    post_mx, post_my = compute_mean_traj(post_trajs)

    if pre_mx is not None:
        ax.plot(pre_mx, pre_my, color='#00BFFF', linewidth=4, alpha=0.95, zorder=10,
                label='Pre-injury mean (n=%d)' % len(pre_trajs))
    if post_mx is not None:
        ax.plot(post_mx, post_my, color='#FF4444', linewidth=4, alpha=0.95, zorder=10,
                label='Post-injury mean (n=%d)' % len(post_trajs))

    ax.plot(0, 0, 'o', color='white', markersize=8, markeredgecolor='black',
            markeredgewidth=1, zorder=11)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('Lateral (px)', fontsize=13)
    ax.set_ylabel('Extension (px)', fontsize=13)
    ax.set_title('M05: Pre vs Post Reach Trajectories Overlaid\n'
                 'Blue = Pre-injury, Red = Post-injury. Origin = nose.',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'M05_spaghetti_overlay.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
