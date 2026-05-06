"""
M05 spaghetti plot: pre vs post, with pellet position heatmap as background.
Pellet positions nose-centered the same way as reach trajectories.
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


def extract_trajectories_and_pellet(swipes_df, dlc_cache):
    """Extract nose-centered RH trajectories and pellet positions."""
    trajectories = []
    pellet_positions = []

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
        pel_x = dlc['Pellet']['x'].iloc[s:e+1].values
        pel_y = dlc['Pellet']['y'].iloc[s:e+1].values
        pel_lk = dlc['Pellet']['likelihood'].iloc[s:e+1].values

        # Nose-center using first frame nose position
        n0x = nose_x[0]
        n0y = nose_y[0]

        cx = rh_x - n0x
        cy = rh_y - n0y

        if np.mean(rh_lk) > 0.3 and len(cx) >= 3:
            trajectories.append((cx, cy))

        # Pellet positions (nose-centered, high confidence only)
        good_pel = pel_lk > 0.5
        if good_pel.any():
            px = pel_x[good_pel] - n0x
            py = pel_y[good_pel] - n0y
            pellet_positions.extend(zip(px, py))

    return trajectories, pellet_positions


def compute_mean_traj(trajectories, n_pts=30):
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
        std_x = np.std(norm_x, axis=0)
        std_y = np.std(norm_y, axis=0)
        return mean_x, mean_y, std_x, std_y
    return None, None, None, None


def plot_panel(ax, trajectories, pellet_positions, color, color_light, title):
    n = len(trajectories)

    # Pellet heatmap: transparent at zero, increasing green opacity with density
    if pellet_positions:
        px = np.array([p[0] for p in pellet_positions])
        py = np.array([p[1] for p in pellet_positions])

        # Compute histogram manually
        bins = 50
        h, xedges, yedges = np.histogram2d(px, py, bins=bins)
        h = h.T  # hist2d returns transposed

        # Create RGBA image: green channel scales with count, alpha scales with count
        rgba = np.zeros((*h.shape, 4))
        if h.max() > 0:
            norm_h = h / h.max()
            rgba[:, :, 1] = norm_h  # green channel
            rgba[:, :, 2] = norm_h  # blue channel (green+blue = cyan)
            rgba[:, :, 3] = norm_h * 0.8  # alpha: 0 for empty, up to 0.8 for max

        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]  # inverted y
        ax.imshow(rgba, extent=extent, aspect='auto', interpolation='bilinear', zorder=1)

    # Individual reach traces
    alpha = max(0.02, min(0.15, 20.0 / max(n, 1)))
    for x, y in trajectories:
        ax.plot(x, y, color=color_light, alpha=alpha, linewidth=0.5, zorder=2)

    # Mean trajectory with SEM band perpendicular to path
    mean_x, mean_y, std_x, std_y = compute_mean_traj(trajectories)
    if mean_x is not None:
        n_pts = len(mean_x)
        n_trajs = len(trajectories)

        # Time-normalize all trajectories to match mean
        norm_trajs = []
        for x, y in trajectories:
            t_orig = np.linspace(0, 1, len(x))
            t_new = np.linspace(0, 1, n_pts)
            norm_trajs.append((np.interp(t_new, t_orig, x), np.interp(t_new, t_orig, y)))

        # Compute perpendicular direction at each point along mean
        dx = np.gradient(mean_x)
        dy = np.gradient(mean_y)
        perp_x = -dy
        perp_y = dx
        mag = np.sqrt(perp_x**2 + perp_y**2)
        mag[mag == 0] = 1
        perp_x = perp_x / mag
        perp_y = perp_y / mag

        # For each trajectory, compute signed perpendicular distance from mean at each time point
        perp_dists = np.zeros((n_trajs, n_pts))
        for i, (tx, ty) in enumerate(norm_trajs):
            diff_x = tx - mean_x
            diff_y = ty - mean_y
            perp_dists[i, :] = diff_x * perp_x + diff_y * perp_y

        # SD of perpendicular distances, scaled by sigma multiplier
        SIGMA = 0.5  # adjust this value to change band width
        band_width = np.std(perp_dists, axis=0) * SIGMA

        # Build band polygon
        upper_x = mean_x + perp_x * band_width
        upper_y = mean_y + perp_y * band_width
        lower_x = mean_x - perp_x * band_width
        lower_y = mean_y - perp_y * band_width

        poly_x = np.concatenate([upper_x, lower_x[::-1]])
        poly_y = np.concatenate([upper_y, lower_y[::-1]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.15, zorder=8)

        ax.plot(mean_x, mean_y, color=color, linewidth=4, alpha=0.95, zorder=10)

    # Origin (nose)
    ax.plot(0, 0, 'o', color='white', markersize=8, markeredgecolor='black',
            markeredgewidth=1, zorder=11)

    ax.invert_xaxis()  # negative extent on right
    ax.invert_yaxis()  # negative extent on top
    ax.set_aspect('equal')
    ax.set_title('%s (n=%d)' % (title, n), fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Lateral from nose (px)', fontsize=10)
    ax.set_ylabel('Extension from nose toward pellet (px)', fontsize=10)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')


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

    # Contact reaches: displaced or retrieved only
    pre_contact = pre[pre['Reach_outcome'].isin(['pellet displaced', 'swipe successful'])]
    post_contact = post[post['Reach_outcome'].isin(['pellet displaced', 'swipe successful'])]

    dlc_cache = {}

    print('Extracting pre-injury (all)...')
    pre_trajs, pre_pellets = extract_trajectories_and_pellet(pre, dlc_cache)
    print('  %d trajectories, %d pellet points' % (len(pre_trajs), len(pre_pellets)))

    print('Extracting post-injury (all)...')
    post_trajs, post_pellets = extract_trajectories_and_pellet(post, dlc_cache)
    print('  %d trajectories, %d pellet points' % (len(post_trajs), len(post_pellets)))

    print('Extracting pre-injury (contact only)...')
    pre_contact_trajs, pre_contact_pellets = extract_trajectories_and_pellet(pre_contact, dlc_cache)
    print('  %d trajectories, %d pellet points' % (len(pre_contact_trajs), len(pre_contact_pellets)))

    print('Extracting post-injury (contact only)...')
    post_contact_trajs, post_contact_pellets = extract_trajectories_and_pellet(post_contact, dlc_cache)
    print('  %d trajectories, %d pellet points' % (len(post_contact_trajs), len(post_contact_pellets)))

    fig, axes = plt.subplots(3, 2, figsize=(18, 30))
    fig.patch.set_facecolor('black')

    # CMY scheme: cyan=pellet, magenta=pre, yellow=post
    # Top row: all reaches
    plot_panel(axes[0, 0], pre_trajs, pre_pellets, '#FF00FF', '#993399', 'M05 Pre-Injury (all)')
    plot_panel(axes[0, 1], post_trajs, post_pellets, '#FFFF00', '#999933', 'M05 Post-Injury (all)')

    # Middle row: contact reaches only
    plot_panel(axes[1, 0], pre_contact_trajs, pre_contact_pellets, '#FF00FF', '#993399', 'M05 Pre-Injury (displaced+retrieved)')
    plot_panel(axes[1, 1], post_contact_trajs, post_contact_pellets, '#FFFF00', '#999933', 'M05 Post-Injury (displaced+retrieved)')

    # Bottom row: all reaches with top 5% by area highlighted in white, thick
    plot_panel(axes[2, 0], pre_trajs, pre_pellets, '#FF00FF', '#993399', 'M05 Pre-Injury (all + top 5%% white)')
    plot_panel(axes[2, 1], post_trajs, post_pellets, '#FFFF00', '#999933', 'M05 Post-Injury (all + top 5%% white)')

    # Overlay top 5% as thick white lines
    n_pre_top = max(1, int(len(pre_trajs) * 0.05))
    n_post_top = max(1, int(len(post_trajs) * 0.05))

    # Need areas to sort - recompute from trajectories
    def traj_area(traj):
        x, y = traj
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    pre_areas = [(traj_area(t), t) for t in pre_trajs]
    post_areas = [(traj_area(t), t) for t in post_trajs]
    pre_areas.sort(key=lambda x: -x[0])
    post_areas.sort(key=lambda x: -x[0])

    for area, (x, y) in pre_areas[:n_pre_top]:
        axes[2, 0].plot(x, y, color='white', linewidth=2, alpha=0.8, zorder=9)
    for area, (x, y) in post_areas[:n_post_top]:
        axes[2, 1].plot(x, y, color='white', linewidth=2, alpha=0.8, zorder=9)

    # Match all axis limits
    all_trajs = pre_trajs + post_trajs
    all_x = np.concatenate([t[0] for t in all_trajs])
    all_y = np.concatenate([t[1] for t in all_trajs])
    pad = 10
    xlim = (np.percentile(all_x, 1) - pad, np.percentile(all_x, 99) + pad)
    ylim = (np.percentile(all_y, 1) - pad, np.percentile(all_y, 99) + pad)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    fig.suptitle('M05: Reach Trajectories with Pellet Heatmap (cyan)\n'
                 'Top: all reaches. Middle: displaced+retrieved. Bottom: all + top 5%% area (white).\n'
                 'Magenta = pre. Yellow = post. Cyan = pellet. White = top 5%% by area.',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'M05_spaghetti_pellet.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
