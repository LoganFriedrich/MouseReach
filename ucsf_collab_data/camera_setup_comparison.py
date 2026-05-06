"""
Camera Setup Comparison via DLC Fixed-Object Coordinates

Uses Reference point and Pillar positions (fixed physical objects) as proxies
for camera position/angle/zoom. If recording conditions changed between
2022 (K/L) and 2023 (M), these coordinates will differ systematically.

Also extracts: frame count, animal body centroid, spatial scale proxy.
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# --- Config ---
DLC_ROOT = r'X:\! DLC Output\Analyzed'
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'Databases', 'figures', 'bodypart_tracking_quality'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = {
    'K': {'path': os.path.join(DLC_ROOT, 'K', 'Post-Processing'), 'year': 2022, 'injury': 'Contusion 70kD'},
    'L': {'path': os.path.join(DLC_ROOT, 'L', 'Post-Processing'), 'year': 2022, 'injury': 'Contusion 50kD'},
    'M': {'path': os.path.join(DLC_ROOT, 'M', 'Post-Processing'), 'year': 2023, 'injury': 'Contusion 60kD'},
}

# Fixed objects to check
FIXED_BODYPARTS = ['Reference', 'Pellet', 'Pillar']
# Animal bodyparts for centroid
ANIMAL_BODYPARTS = ['Nose', 'RightHand', 'RightEar', 'LeftEar', 'TailBase']
# All bodyparts for scale
ALL_BODYPARTS = FIXED_BODYPARTS + ANIMAL_BODYPARTS

def extract_camera_metrics(csv_path):
    """Extract fixed-object positions and camera metrics from a DLC CSV."""
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    except Exception:
        return None

    scorer = df.columns.get_level_values(0)[0]
    result = {'total_frames': len(df)}

    for bp in ALL_BODYPARTS:
        try:
            x = df[(scorer, bp, 'x')].values.astype(float)
            y = df[(scorer, bp, 'y')].values.astype(float)
            likelihood = df[(scorer, bp, 'likelihood')].values.astype(float)
        except (KeyError, ValueError):
            continue

        # Use high-confidence frames for fixed objects
        if bp in FIXED_BODYPARTS:
            mask = likelihood > 0.9
        else:
            mask = likelihood > 0.5

        if mask.sum() < 10:
            continue

        xm, ym = x[mask], y[mask]
        result[f'{bp}_x_mean'] = np.mean(xm)
        result[f'{bp}_y_mean'] = np.mean(ym)
        result[f'{bp}_x_std'] = np.std(xm)
        result[f'{bp}_y_std'] = np.std(ym)
        result[f'{bp}_x_median'] = np.median(xm)
        result[f'{bp}_y_median'] = np.median(ym)
        result[f'{bp}_conf_mean'] = np.mean(likelihood)
        result[f'{bp}_n_highconf'] = mask.sum()

    return result


print('=' * 80)
print('CAMERA SETUP COMPARISON VIA DLC FIXED-OBJECT COORDINATES')
print('If camera changed between 2022 (K/L) and 2023 (M), fixed objects move')
print('=' * 80)

all_records = []
for grp, info in GROUPS.items():
    csvs = sorted(glob.glob(os.path.join(info['path'], '*.csv')))
    print(f'\n  Group {grp} ({info["injury"]}, {info["year"]}): processing {len(csvs)} CSVs')

    for i, csv_path in enumerate(csvs):
        if (i + 1) % 200 == 0:
            print(f'    ... {i+1}/{len(csvs)}')
        metrics = extract_camera_metrics(csv_path)
        if metrics is None:
            continue
        metrics['group'] = grp
        metrics['year'] = info['year']
        metrics['filename'] = os.path.basename(csv_path)
        # Extract animal ID from filename
        basename = os.path.basename(csv_path)
        parts = basename.split('_')
        for p in parts:
            if p.startswith(grp) and len(p) <= 4:
                metrics['animal_id'] = p
                break
        all_records.append(metrics)

df = pd.DataFrame(all_records)
print(f'\nTotal sessions extracted: {len(df)}')

# Save raw data
df.to_csv(os.path.join(OUTPUT_DIR, 'camera_setup_raw.csv'), index=False)

# --- Analysis ---

print('\n' + '=' * 80)
print('FIXED OBJECT POSITIONS (camera setup proxy)')
print('=' * 80)

for bp in FIXED_BODYPARTS:
    x_col = f'{bp}_x_mean'
    y_col = f'{bp}_y_mean'
    if x_col not in df.columns:
        print(f'\n  {bp}: no data')
        continue

    print(f'\n  {bp} position (pixels):')
    print(f'    {"Group":6s} {"X mean":>10s} {"X SD":>8s} {"Y mean":>10s} {"Y SD":>8s} {"n":>6s}')
    print(f'    {"-"*6} {"-"*10} {"-"*8} {"-"*10} {"-"*8} {"-"*6}')

    for grp in ['K', 'L', 'M']:
        g = df[df['group'] == grp]
        x = g[x_col].dropna()
        y = g[y_col].dropna()
        print(f'    {grp:6s} {x.mean():10.1f} {x.std():8.1f} {y.mean():10.1f} {y.std():8.1f} {len(x):6d}')

    # M vs K+L comparison
    m_x = df[df['group'] == 'M'][x_col].dropna()
    kl_x = df[df['group'].isin(['K', 'L'])][x_col].dropna()
    m_y = df[df['group'] == 'M'][y_col].dropna()
    kl_y = df[df['group'].isin(['K', 'L'])][y_col].dropna()

    if len(m_x) > 10 and len(kl_x) > 10:
        t_x, p_x = stats.ttest_ind(m_x, kl_x)
        t_y, p_y = stats.ttest_ind(m_y, kl_y)
        d_x = (m_x.mean() - kl_x.mean()) / np.sqrt((m_x.std()**2 + kl_x.std()**2) / 2)
        d_y = (m_y.mean() - kl_y.mean()) / np.sqrt((m_y.std()**2 + kl_y.std()**2) / 2)
        sig_x = '***' if p_x < 0.001 else '**' if p_x < 0.01 else '*' if p_x < 0.05 else 'ns'
        sig_y = '***' if p_y < 0.001 else '**' if p_y < 0.01 else '*' if p_y < 0.05 else 'ns'
        print(f'    M vs K+L: X diff={m_x.mean()-kl_x.mean():+.1f}px d={d_x:.2f} p={p_x:.4f} {sig_x}')
        print(f'              Y diff={m_y.mean()-kl_y.mean():+.1f}px d={d_y:.2f} p={p_y:.4f} {sig_y}')


print('\n' + '=' * 80)
print('FIXED OBJECT STABILITY (should be near-zero SD for truly fixed objects)')
print('=' * 80)

for bp in FIXED_BODYPARTS:
    x_std_col = f'{bp}_x_std'
    y_std_col = f'{bp}_y_std'
    if x_std_col not in df.columns:
        continue

    print(f'\n  {bp} within-session position SD (lower = more stable tracking):')
    for grp in ['K', 'L', 'M']:
        g = df[df['group'] == grp]
        xsd = g[x_std_col].dropna()
        ysd = g[y_std_col].dropna()
        print(f'    {grp}: X_SD={xsd.mean():.2f} (SD={xsd.std():.2f}), Y_SD={ysd.mean():.2f} (SD={ysd.std():.2f})')


print('\n' + '=' * 80)
print('SPATIAL SCALE PROXY: Reference-to-Pillar Distance')
print('(Same physical distance, so pixel distance = zoom/scale)')
print('=' * 80)

# Compute ref-pillar distance from coordinates
ref_x = 'Reference_x_mean'
ref_y = 'Reference_y_mean'
pil_x = 'Pillar_x_mean'
pil_y = 'Pillar_y_mean'

if all(c in df.columns for c in [ref_x, ref_y, pil_x, pil_y]):
    df['ref_pillar_dist'] = np.sqrt(
        (df[ref_x] - df[pil_x])**2 + (df[ref_y] - df[pil_y])**2
    )

    for grp in ['K', 'L', 'M']:
        g = df[df['group'] == grp]['ref_pillar_dist'].dropna()
        print(f'  {grp}: {g.mean():.1f} px (SD={g.std():.1f}, n={len(g)})')

    m_d = df[df['group'] == 'M']['ref_pillar_dist'].dropna()
    kl_d = df[df['group'].isin(['K', 'L'])]['ref_pillar_dist'].dropna()
    t, p = stats.ttest_ind(m_d, kl_d)
    d = (m_d.mean() - kl_d.mean()) / np.sqrt((m_d.std()**2 + kl_d.std()**2) / 2)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  M vs K+L: diff={m_d.mean()-kl_d.mean():+.1f}px d={d:.2f} p={p:.6f} {sig}')
    pct_diff = (m_d.mean() - kl_d.mean()) / kl_d.mean() * 100
    print(f'  Scale difference: {pct_diff:+.1f}% (positive = M has larger field of view or objects farther apart)')


print('\n' + '=' * 80)
print('FRAME COUNT COMPARISON')
print('=' * 80)

for grp in ['K', 'L', 'M']:
    g = df[df['group'] == grp]['total_frames'].dropna()
    print(f'  {grp}: mean={g.mean():.0f} frames (SD={g.std():.0f}, min={g.min():.0f}, max={g.max():.0f})')


print('\n' + '=' * 80)
print('PER-ANIMAL REFERENCE POINT POSITION (detect within-group camera shifts)')
print('=' * 80)

if 'Reference_x_mean' in df.columns and 'animal_id' in df.columns:
    for grp in ['K', 'L', 'M']:
        g = df[df['group'] == grp]
        animal_means = g.groupby('animal_id').agg({
            'Reference_x_mean': 'mean',
            'Reference_y_mean': 'mean',
        }).dropna()
        if len(animal_means) > 0:
            print(f'\n  {grp} animals:')
            for aid, row in animal_means.iterrows():
                print(f'    {aid}: Ref X={row["Reference_x_mean"]:.1f}, Ref Y={row["Reference_y_mean"]:.1f}')


# --- Figures ---
print('\nGenerating figures...')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Camera Setup Comparison: Fixed Object Positions\n'
             'K/L (2022) vs M (2023) -- Do pixel coordinates of fixed objects differ?',
             fontsize=14, fontweight='bold')

colors = {'K': '#E74C3C', 'L': '#3498DB', 'M': '#2ECC71'}

for i, bp in enumerate(FIXED_BODYPARTS):
    x_col = f'{bp}_x_mean'
    y_col = f'{bp}_y_mean'
    if x_col not in df.columns:
        continue

    # Scatter plot: X vs Y position
    ax = axes[0, i]
    for grp in ['K', 'L', 'M']:
        g = df[df['group'] == grp]
        x = g[x_col].dropna()
        y = g[y_col].dropna()
        common = x.index.intersection(y.index)
        ax.scatter(x.loc[common], y.loc[common], alpha=0.3, s=5,
                   c=colors[grp], label=f'{grp} (n={len(common)})')
    ax.set_title(f'{bp} Position')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend(fontsize=8)
    ax.invert_yaxis()  # Image coordinates

    # Box plot: X position by group
    ax = axes[1, i]
    data_by_group = []
    labels = []
    for grp in ['K', 'L', 'M']:
        g = df[df['group'] == grp][x_col].dropna()
        data_by_group.append(g.values)
        labels.append(f'{grp}\n({GROUPS[grp]["year"]})')
    bp_plot = ax.boxplot(data_by_group, labels=labels, patch_artist=True)
    for patch, grp in zip(bp_plot['boxes'], ['K', 'L', 'M']):
        patch.set_facecolor(colors[grp])
        patch.set_alpha(0.6)
    ax.set_title(f'{bp} X-Position Distribution')
    ax.set_ylabel('X (pixels)')

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'fig10_camera_setup_fixed_objects.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'  Saved: {fig_path}')
plt.close()

# Scale comparison figure
if 'ref_pillar_dist' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Camera Scale Comparison: Reference-to-Pillar Distance\n'
                 'Same physical distance -- pixel difference = camera zoom/position change',
                 fontsize=13, fontweight='bold')

    # Histogram
    ax = axes[0]
    for grp in ['K', 'L', 'M']:
        g = df[df['group'] == grp]['ref_pillar_dist'].dropna()
        ax.hist(g, bins=50, alpha=0.5, color=colors[grp],
                label=f'{grp} ({GROUPS[grp]["year"]}, mean={g.mean():.1f})')
    ax.set_xlabel('Reference-to-Pillar Distance (pixels)')
    ax.set_ylabel('Session Count')
    ax.legend()
    ax.set_title('Distribution')

    # Per-animal means
    ax = axes[1]
    if 'animal_id' in df.columns:
        for grp in ['K', 'L', 'M']:
            g = df[df['group'] == grp]
            am = g.groupby('animal_id')['ref_pillar_dist'].mean().dropna().sort_values()
            ax.barh([f'{grp}-{a}' for a in am.index], am.values,
                    color=colors[grp], alpha=0.7)
        ax.set_xlabel('Mean Ref-Pillar Distance (pixels)')
        ax.set_title('Per-Animal Means')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'fig11_camera_scale_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fig_path}')
    plt.close()

print('\nDone.')
