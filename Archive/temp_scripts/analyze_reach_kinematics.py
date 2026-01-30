"""
Reach Kinematics Analysis - How Mice Retrieve Pellets

Aggregates all reach feature data and generates publication-ready graphs
comparing kinematics between successful and failed reach attempts.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats

# ============================================================================
# DATA AGGREGATION
# ============================================================================

def load_all_features(features_dir: Path) -> pd.DataFrame:
    """Load and flatten all reach features into a single DataFrame."""
    all_reaches = []

    for fpath in sorted(features_dir.glob('*_features.json')):
        with open(fpath) as f:
            data = json.load(f)

        video_name = data['video_name']

        for seg in data['segments']:
            for reach in seg['reaches']:
                row = {
                    'video_name': video_name,
                    'segment_num': seg['segment_num'],
                    'segment_outcome': seg['outcome'],
                    **reach
                }
                all_reaches.append(row)

    df = pd.DataFrame(all_reaches)

    # Parse video name for metadata
    df['animal'] = df['video_name'].str.extract(r'_([A-Z]+\d+)_')[0]
    df['cohort'] = df['animal'].str.extract(r'CNT(\d{2})')[0].apply(lambda x: f'CNT_{x}' if pd.notna(x) else None)

    # Create success category
    df['success'] = df['segment_outcome'].map({
        'retrieved': 'Success',
        'displaced_sa': 'Fail (Displaced)',
        'displaced_outside': 'Fail (Displaced)',
        'untouched': 'Fail (Untouched)'
    })

    # Binary success
    df['is_success'] = df['segment_outcome'] == 'retrieved'

    return df

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compare_kinematics(df: pd.DataFrame, feature: str) -> dict:
    """Compare a kinematic feature between success and failure."""
    success = df[df['is_success']][feature].dropna()
    fail = df[~df['is_success']][feature].dropna()

    # T-test
    t_stat, p_value = stats.ttest_ind(success, fail)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((success.std()**2 + fail.std()**2) / 2)
    cohens_d = (success.mean() - fail.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        'feature': feature,
        'success_mean': success.mean(),
        'success_std': success.std(),
        'fail_mean': fail.mean(),
        'fail_std': fail.std(),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'n_success': len(success),
        'n_fail': len(fail)
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_kinematic_figure(df: pd.DataFrame, output_path: Path):
    """Create publication-ready figure comparing kinematics by outcome."""

    # Color scheme
    colors = {
        'Success': '#2ecc71',
        'Fail (Displaced)': '#f39c12',
        'Fail (Untouched)': '#e74c3c'
    }

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # -------------------------------------------------------------------------
    # 1. Reach Extent by Outcome
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for success_cat in ['Success', 'Fail (Displaced)', 'Fail (Untouched)']:
        data = df[df['success'] == success_cat]['max_extent_mm'].dropna()
        if len(data) > 0:
            ax1.hist(data, bins=25, alpha=0.6, label=success_cat, color=colors[success_cat], density=True)
    ax1.set_xlabel('Max Reach Extent (mm)', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('A. Reach Extent Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)

    # Add stats annotation
    stats_extent = compare_kinematics(df, 'max_extent_mm')
    ax1.text(0.98, 0.98, f"Success: {stats_extent['success_mean']:.1f}±{stats_extent['success_std']:.1f} mm\n"
                         f"Fail: {stats_extent['fail_mean']:.1f}±{stats_extent['fail_std']:.1f} mm\n"
                         f"p={stats_extent['p_value']:.3f}, d={stats_extent['cohens_d']:.2f}",
             transform=ax1.transAxes, fontsize=8, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -------------------------------------------------------------------------
    # 2. Peak Velocity by Outcome
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for success_cat in ['Success', 'Fail (Displaced)', 'Fail (Untouched)']:
        data = df[df['success'] == success_cat]['peak_velocity_px_per_frame'].dropna()
        if len(data) > 0:
            ax2.hist(data, bins=25, alpha=0.6, label=success_cat, color=colors[success_cat], density=True)
    ax2.set_xlabel('Peak Velocity (px/frame)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('B. Peak Velocity Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)

    stats_vel = compare_kinematics(df, 'peak_velocity_px_per_frame')
    ax2.text(0.98, 0.98, f"Success: {stats_vel['success_mean']:.2f}±{stats_vel['success_std']:.2f}\n"
                         f"Fail: {stats_vel['fail_mean']:.2f}±{stats_vel['fail_std']:.2f}\n"
                         f"p={stats_vel['p_value']:.3f}, d={stats_vel['cohens_d']:.2f}",
             transform=ax2.transAxes, fontsize=8, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -------------------------------------------------------------------------
    # 3. Trajectory Straightness by Outcome
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    for success_cat in ['Success', 'Fail (Displaced)', 'Fail (Untouched)']:
        data = df[df['success'] == success_cat]['trajectory_straightness'].dropna()
        if len(data) > 0:
            ax3.hist(data, bins=25, alpha=0.6, label=success_cat, color=colors[success_cat], density=True)
    ax3.set_xlabel('Trajectory Straightness (0-1)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('C. Trajectory Straightness', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)

    stats_straight = compare_kinematics(df, 'trajectory_straightness')
    ax3.text(0.98, 0.98, f"Success: {stats_straight['success_mean']:.3f}±{stats_straight['success_std']:.3f}\n"
                         f"Fail: {stats_straight['fail_mean']:.3f}±{stats_straight['fail_std']:.3f}\n"
                         f"p={stats_straight['p_value']:.3f}, d={stats_straight['cohens_d']:.2f}",
             transform=ax3.transAxes, fontsize=8, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -------------------------------------------------------------------------
    # 4. Bar chart - Mean values comparison
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    features_to_compare = ['max_extent_mm', 'peak_velocity_px_per_frame', 'duration_frames']
    feature_labels = ['Extent (mm)', 'Peak Vel (px/f)', 'Duration (frames)']

    x = np.arange(len(features_to_compare))
    width = 0.35

    success_means = []
    fail_means = []
    success_sems = []
    fail_sems = []

    for feat in features_to_compare:
        success_data = df[df['is_success']][feat].dropna()
        fail_data = df[~df['is_success']][feat].dropna()
        success_means.append(success_data.mean())
        fail_means.append(fail_data.mean())
        success_sems.append(success_data.std() / np.sqrt(len(success_data)))
        fail_sems.append(fail_data.std() / np.sqrt(len(fail_data)))

    # Normalize for visualization
    max_vals = [max(s, f) for s, f in zip(success_means, fail_means)]
    success_norm = [s/m for s, m in zip(success_means, max_vals)]
    fail_norm = [f/m for f, m in zip(fail_means, max_vals)]

    bars1 = ax4.bar(x - width/2, success_norm, width, label='Success', color=colors['Success'])
    bars2 = ax4.bar(x + width/2, fail_norm, width, label='Fail', color=colors['Fail (Displaced)'])

    ax4.set_ylabel('Normalized Value', fontsize=10)
    ax4.set_title('D. Kinematic Features Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(feature_labels)
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 1.2)

    # -------------------------------------------------------------------------
    # 5. Extent vs Velocity scatter
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    for success_cat in ['Success', 'Fail (Displaced)', 'Fail (Untouched)']:
        subset = df[df['success'] == success_cat]
        ax5.scatter(subset['max_extent_mm'], subset['peak_velocity_px_per_frame'],
                   alpha=0.4, label=success_cat, color=colors[success_cat], s=15)
    ax5.set_xlabel('Max Extent (mm)', fontsize=10)
    ax5.set_ylabel('Peak Velocity (px/frame)', fontsize=10)
    ax5.set_title('E. Extent vs Velocity', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)

    # -------------------------------------------------------------------------
    # 6. Duration vs Extent scatter
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    for success_cat in ['Success', 'Fail (Displaced)', 'Fail (Untouched)']:
        subset = df[df['success'] == success_cat]
        ax6.scatter(subset['duration_frames'], subset['max_extent_mm'],
                   alpha=0.4, label=success_cat, color=colors[success_cat], s=15)
    ax6.set_xlabel('Duration (frames)', fontsize=10)
    ax6.set_ylabel('Max Extent (mm)', fontsize=10)
    ax6.set_title('F. Duration vs Extent', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)

    # -------------------------------------------------------------------------
    # 7. Outcome distribution pie chart
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 0])
    outcome_counts = df['segment_outcome'].value_counts()
    pie_colors = [colors.get(df[df['segment_outcome']==o]['success'].iloc[0], '#95a5a6')
                  for o in outcome_counts.index]
    ax7.pie(outcome_counts.values, labels=outcome_counts.index, colors=pie_colors,
            autopct='%1.1f%%', startangle=90)
    ax7.set_title('G. Outcome Distribution', fontsize=12, fontweight='bold')

    # -------------------------------------------------------------------------
    # 8. Reach position in segment vs outcome
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 1])
    reach_position_success = df.groupby('reach_num')['is_success'].mean() * 100
    ax8.bar(reach_position_success.index[:10], reach_position_success.values[:10],
            color='#3498db', alpha=0.8)
    ax8.set_xlabel('Reach # in Segment', fontsize=10)
    ax8.set_ylabel('Success Rate (%)', fontsize=10)
    ax8.set_title('H. Success Rate by Reach Position', fontsize=12, fontweight='bold')
    ax8.set_ylim(0, 100)

    # -------------------------------------------------------------------------
    # 9. Summary statistics table
    # -------------------------------------------------------------------------
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary_text = f"""
    KINEMATIC SUMMARY
    ─────────────────────────────────
    Total Reaches: {len(df):,}
    Animals: {df['animal'].nunique()}
    Videos: {df['video_name'].nunique()}

    SUCCESS RATE: {df['is_success'].mean()*100:.1f}%

    RETRIEVED ({df['is_success'].sum()} reaches):
      Extent: {df[df['is_success']]['max_extent_mm'].mean():.2f} ± {df[df['is_success']]['max_extent_mm'].std():.2f} mm
      Duration: {df[df['is_success']]['duration_frames'].mean():.1f} frames
      Peak Vel: {df[df['is_success']]['peak_velocity_px_per_frame'].mean():.2f} px/f

    FAILED ({(~df['is_success']).sum()} reaches):
      Extent: {df[~df['is_success']]['max_extent_mm'].mean():.2f} ± {df[~df['is_success']]['max_extent_mm'].std():.2f} mm
      Duration: {df[~df['is_success']]['duration_frames'].mean():.1f} frames
      Peak Vel: {df[~df['is_success']]['peak_velocity_px_per_frame'].mean():.2f} px/f
    """

    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
             va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    # Title
    fig.suptitle('How Mice Retrieve Pellets: Kinematic Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    return fig

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    pipeline_dir = Path(__file__).parent
    features_dir = pipeline_dir / 'Step5_Features'

    print("=" * 70)
    print("REACH KINEMATICS ANALYSIS")
    print("=" * 70)

    # Load all feature data
    print("\n1. Loading feature data...")
    df = load_all_features(features_dir)
    print(f"   Loaded {len(df):,} reaches from {df['video_name'].nunique()} videos")
    print(f"   Animals: {df['animal'].nunique()}")
    print(f"   Cohorts: {sorted(df['cohort'].dropna().unique())}")

    # Run statistical comparisons
    print("\n2. Running statistical analysis...")
    key_features = ['max_extent_mm', 'peak_velocity_px_per_frame', 'duration_frames',
                   'trajectory_straightness', 'mean_velocity_px_per_frame']

    stats_results = []
    for feat in key_features:
        if feat in df.columns:
            result = compare_kinematics(df, feat)
            stats_results.append(result)
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            print(f"   {feat}: p={result['p_value']:.4f} {sig}, d={result['cohens_d']:.2f}")

    # Save stats to CSV
    stats_df = pd.DataFrame(stats_results)
    stats_path = pipeline_dir / 'kinematic_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\n   Saved statistics to: {stats_path}")

    # Generate figure
    print("\n3. Generating publication figure...")
    fig_path = pipeline_dir / 'kinematic_analysis.png'
    create_kinematic_figure(df, fig_path)

    # Export full dataset
    print("\n4. Exporting full dataset...")
    export_path = pipeline_dir / 'all_reach_features.xlsx'
    df.to_excel(export_path, index=False)
    print(f"   Saved: {export_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
