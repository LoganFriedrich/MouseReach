"""
Reach Kinematics Analysis - PROPER WITHIN-SUBJECT DESIGN

Addresses:
1. Subject distribution (are some mice over-represented?)
2. Within-subject comparisons (paired by mouse)
3. Fatigue/temporal effects (performance across session)
4. Mixed-effects models (mouse as random effect)
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels for mixed effects
try:
    import statsmodels.formula.api as smf
    HAS_MIXED = True
except ImportError:
    HAS_MIXED = False
    print("Note: statsmodels not available, skipping mixed-effects models")

# ============================================================================
# DATA LOADING
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

    # Success categories
    df['success'] = df['segment_outcome'].map({
        'retrieved': 'Success',
        'displaced_sa': 'Fail',
        'displaced_outside': 'Fail',
        'untouched': 'Fail'
    })
    df['is_success'] = df['segment_outcome'] == 'retrieved'

    return df

# ============================================================================
# SUBJECT DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_subject_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Check how reaches are distributed across subjects."""
    subject_stats = df.groupby('animal').agg(
        n_reaches=('reach_id', 'count'),
        n_success=('is_success', 'sum'),
        n_fail=('is_success', lambda x: (~x).sum()),
        success_rate=('is_success', 'mean'),
        n_videos=('video_name', 'nunique'),
        mean_extent=('max_extent_mm', 'mean'),
        mean_velocity=('peak_velocity_px_per_frame', 'mean')
    ).reset_index()

    subject_stats['pct_of_total'] = subject_stats['n_reaches'] / len(df) * 100

    return subject_stats

# ============================================================================
# WITHIN-SUBJECT ANALYSIS
# ============================================================================

def within_subject_comparison(df: pd.DataFrame, feature: str) -> dict:
    """
    Proper within-subject comparison.
    For each mouse, compare their success vs fail reaches, then aggregate.
    """
    # Get per-mouse means for success and fail
    mouse_means = df.groupby(['animal', 'is_success'])[feature].mean().unstack()

    # Only include mice with both success and fail reaches
    paired_data = mouse_means.dropna()

    if len(paired_data) < 3:
        return {
            'feature': feature,
            'n_mice': len(paired_data),
            'paired_possible': False,
            'note': 'Too few mice with both success and fail'
        }

    success_means = paired_data[True]
    fail_means = paired_data[False]

    # Paired t-test (within-subject)
    t_stat, p_value = stats.ttest_rel(success_means, fail_means)

    # Effect size (paired Cohen's d)
    diff = success_means - fail_means
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    return {
        'feature': feature,
        'n_mice': len(paired_data),
        'paired_possible': True,
        'success_mean': success_means.mean(),
        'success_std': success_means.std(),
        'fail_mean': fail_means.mean(),
        'fail_std': fail_means.std(),
        'mean_difference': diff.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d_paired': cohens_d
    }

# ============================================================================
# FATIGUE / TEMPORAL ANALYSIS
# ============================================================================

def analyze_fatigue_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how performance changes across session (fatigue effects)."""
    # Create session position (reach number within video)
    df = df.copy()
    df['session_position'] = df.groupby('video_name').cumcount() + 1

    # Bin into early/middle/late
    df['session_phase'] = pd.cut(
        df['session_position'],
        bins=[0, 30, 60, float('inf')],
        labels=['Early (1-30)', 'Middle (31-60)', 'Late (60+)']
    )

    # Calculate metrics by phase
    phase_stats = df.groupby('session_phase').agg(
        n_reaches=('reach_id', 'count'),
        success_rate=('is_success', 'mean'),
        mean_extent=('max_extent_mm', 'mean'),
        mean_velocity=('peak_velocity_px_per_frame', 'mean'),
        mean_duration=('duration_frames', 'mean')
    ).reset_index()

    return phase_stats, df

# ============================================================================
# MIXED EFFECTS MODEL
# ============================================================================

def run_mixed_effects(df: pd.DataFrame, feature: str, outcome_var: str = 'is_success'):
    """Run mixed-effects logistic regression with mouse as random effect."""
    if not HAS_MIXED:
        return None

    # Prepare data
    model_df = df[[feature, outcome_var, 'animal']].dropna()
    model_df = model_df.rename(columns={feature: 'predictor', outcome_var: 'outcome'})
    model_df['outcome'] = model_df['outcome'].astype(int)

    try:
        # Mixed effects logistic regression
        model = smf.mixedlm("outcome ~ predictor", model_df, groups=model_df["animal"])
        result = model.fit(disp=False)

        return {
            'feature': feature,
            'coefficient': result.params['predictor'],
            'std_err': result.bse['predictor'],
            'z_value': result.tvalues['predictor'],
            'p_value': result.pvalues['predictor'],
            'random_effect_var': result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else None
        }
    except Exception as e:
        return {'feature': feature, 'error': str(e)}

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_proper_analysis_figure(df: pd.DataFrame, subject_stats: pd.DataFrame,
                                   within_results: list, fatigue_stats: pd.DataFrame,
                                   output_path: Path):
    """Create figure with proper within-subject analysis."""

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'Success': '#2ecc71', 'Fail': '#e74c3c'}

    # -------------------------------------------------------------------------
    # 1. Subject Distribution - Reaches per mouse
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(subject_stats)), subject_stats['n_reaches'].sort_values(ascending=False),
                   color='#3498db', alpha=0.8)
    ax1.set_xlabel('Mouse (sorted)', fontsize=10)
    ax1.set_ylabel('# Reaches', fontsize=10)
    ax1.set_title('A. Subject Distribution\n(Reach counts per mouse)', fontsize=11, fontweight='bold')
    ax1.axhline(subject_stats['n_reaches'].mean(), color='red', linestyle='--', label=f"Mean: {subject_stats['n_reaches'].mean():.0f}")
    ax1.legend(fontsize=8)

    # -------------------------------------------------------------------------
    # 2. Subject Distribution - Success rate per mouse
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_stats = subject_stats.sort_values('success_rate', ascending=False)
    ax2.bar(range(len(sorted_stats)), sorted_stats['success_rate'] * 100, color='#2ecc71', alpha=0.8)
    ax2.set_xlabel('Mouse (sorted by success)', fontsize=10)
    ax2.set_ylabel('Success Rate (%)', fontsize=10)
    ax2.set_title('B. Success Rate by Mouse', fontsize=11, fontweight='bold')
    ax2.axhline(df['is_success'].mean() * 100, color='red', linestyle='--',
                label=f"Population: {df['is_success'].mean()*100:.1f}%")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 100)

    # -------------------------------------------------------------------------
    # 3. Within-Subject: Paired comparison (extent)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])

    # Get per-mouse means
    mouse_means = df.groupby(['animal', 'is_success'])['max_extent_mm'].mean().unstack().dropna()

    # Plot paired lines
    for idx, row in mouse_means.iterrows():
        ax3.plot([0, 1], [row[False], row[True]], 'o-', color='gray', alpha=0.5, markersize=4)

    # Add means
    ax3.errorbar([0], [mouse_means[False].mean()], yerr=[mouse_means[False].std()/np.sqrt(len(mouse_means))],
                 fmt='s', color=colors['Fail'], markersize=12, capsize=5, label='Fail', zorder=10)
    ax3.errorbar([1], [mouse_means[True].mean()], yerr=[mouse_means[True].std()/np.sqrt(len(mouse_means))],
                 fmt='s', color=colors['Success'], markersize=12, capsize=5, label='Success', zorder=10)

    extent_result = [r for r in within_results if r['feature'] == 'max_extent_mm'][0]
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Fail', 'Success'])
    ax3.set_ylabel('Max Extent (mm)', fontsize=10)
    ax3.set_title(f'C. Within-Subject: Extent\n(n={extent_result["n_mice"]} mice, p={extent_result["p_value"]:.3f})',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)

    # -------------------------------------------------------------------------
    # 4. Within-Subject: Paired comparison (velocity)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[0, 3])

    mouse_means_vel = df.groupby(['animal', 'is_success'])['peak_velocity_px_per_frame'].mean().unstack().dropna()

    for idx, row in mouse_means_vel.iterrows():
        ax4.plot([0, 1], [row[False], row[True]], 'o-', color='gray', alpha=0.5, markersize=4)

    ax4.errorbar([0], [mouse_means_vel[False].mean()], yerr=[mouse_means_vel[False].std()/np.sqrt(len(mouse_means_vel))],
                 fmt='s', color=colors['Fail'], markersize=12, capsize=5, label='Fail', zorder=10)
    ax4.errorbar([1], [mouse_means_vel[True].mean()], yerr=[mouse_means_vel[True].std()/np.sqrt(len(mouse_means_vel))],
                 fmt='s', color=colors['Success'], markersize=12, capsize=5, label='Success', zorder=10)

    vel_result = [r for r in within_results if r['feature'] == 'peak_velocity_px_per_frame'][0]
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Fail', 'Success'])
    ax4.set_ylabel('Peak Velocity (px/frame)', fontsize=10)
    ax4.set_title(f'D. Within-Subject: Velocity\n(n={vel_result["n_mice"]} mice, p={vel_result["p_value"]:.3f})',
                  fontsize=11, fontweight='bold')

    # -------------------------------------------------------------------------
    # 5. Fatigue Effect - Success rate by session phase
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.bar(range(len(fatigue_stats)), fatigue_stats['success_rate'] * 100,
            color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.8)
    ax5.set_xticks(range(len(fatigue_stats)))
    ax5.set_xticklabels(fatigue_stats['session_phase'], rotation=15)
    ax5.set_ylabel('Success Rate (%)', fontsize=10)
    ax5.set_title('E. Fatigue: Success Rate by Phase', fontsize=11, fontweight='bold')
    ax5.set_ylim(0, max(fatigue_stats['success_rate'] * 100) * 1.2)

    # Add trend test
    early = df[df['session_phase'] == 'Early (1-30)']['is_success']
    late = df[df['session_phase'] == 'Late (60+)']['is_success']
    if len(early) > 10 and len(late) > 10:
        chi2, p_fatigue = stats.chi2_contingency(pd.crosstab(df['session_phase'], df['is_success']))[:2]
        ax5.text(0.98, 0.98, f"χ² test: p={p_fatigue:.3f}", transform=ax5.transAxes,
                 fontsize=9, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -------------------------------------------------------------------------
    # 6. Fatigue Effect - Extent by session phase
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.bar(range(len(fatigue_stats)), fatigue_stats['mean_extent'],
            color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.8)
    ax6.set_xticks(range(len(fatigue_stats)))
    ax6.set_xticklabels(fatigue_stats['session_phase'], rotation=15)
    ax6.set_ylabel('Mean Extent (mm)', fontsize=10)
    ax6.set_title('F. Fatigue: Extent by Phase', fontsize=11, fontweight='bold')

    # -------------------------------------------------------------------------
    # 7. Fatigue Effect - Velocity by session phase
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.bar(range(len(fatigue_stats)), fatigue_stats['mean_velocity'],
            color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.8)
    ax7.set_xticks(range(len(fatigue_stats)))
    ax7.set_xticklabels(fatigue_stats['session_phase'], rotation=15)
    ax7.set_ylabel('Mean Peak Velocity', fontsize=10)
    ax7.set_title('G. Fatigue: Velocity by Phase', fontsize=11, fontweight='bold')

    # -------------------------------------------------------------------------
    # 8. Reach position in segment
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[1, 3])
    reach_pos_success = df.groupby('reach_num')['is_success'].agg(['mean', 'count'])
    reach_pos_success = reach_pos_success[reach_pos_success['count'] >= 20]  # Filter low n
    ax8.bar(reach_pos_success.index[:8], reach_pos_success['mean'].values[:8] * 100,
            color='#3498db', alpha=0.8)
    ax8.set_xlabel('Reach # in Segment', fontsize=10)
    ax8.set_ylabel('Success Rate (%)', fontsize=10)
    ax8.set_title('H. Success by Segment Position', fontsize=11, fontweight='bold')
    ax8.set_ylim(0, 50)

    # -------------------------------------------------------------------------
    # 9. Summary Statistics Table
    # -------------------------------------------------------------------------
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9.axis('off')

    # Build stats table
    stats_text = "WITHIN-SUBJECT STATISTICAL RESULTS\n" + "="*50 + "\n\n"
    stats_text += f"{'Feature':<30} {'n_mice':<8} {'p-value':<12} {'Effect (d)':<12} {'Sig':<6}\n"
    stats_text += "-"*70 + "\n"

    for result in within_results:
        if result.get('paired_possible', False):
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            stats_text += f"{result['feature']:<30} {result['n_mice']:<8} {result['p_value']:<12.4f} {result['cohens_d_paired']:<12.2f} {sig:<6}\n"

    stats_text += "\n" + "-"*70 + "\n"
    stats_text += "Paired t-tests comparing each mouse's success vs fail reaches\n"
    stats_text += "Cohen's d calculated on within-subject differences\n"

    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9,
             va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    # -------------------------------------------------------------------------
    # 10. Data Summary
    # -------------------------------------------------------------------------
    ax10 = fig.add_subplot(gs[2, 2:4])
    ax10.axis('off')

    summary = f"""
    DATA SUMMARY
    {'='*40}

    Total Reaches: {len(df):,}
    Animals: {df['animal'].nunique()}
    Videos: {df['video_name'].nunique()}
    Cohorts: {sorted(df['cohort'].dropna().unique())}

    SUBJECT BALANCE:
      Min reaches/mouse: {subject_stats['n_reaches'].min()}
      Max reaches/mouse: {subject_stats['n_reaches'].max()}
      Median: {subject_stats['n_reaches'].median():.0f}
      CV: {subject_stats['n_reaches'].std()/subject_stats['n_reaches'].mean()*100:.1f}%

    OVERALL SUCCESS RATE: {df['is_success'].mean()*100:.1f}%
      Retrieved: {(df['segment_outcome']=='retrieved').sum()}
      Displaced: {df['segment_outcome'].str.contains('displaced').sum()}
      Untouched: {(df['segment_outcome']=='untouched').sum()}

    MICE WITH BOTH SUCCESS & FAIL:
      {len(df.groupby(['animal', 'is_success']).size().unstack().dropna())} / {df['animal'].nunique()} mice
    """

    ax10.text(0.05, 0.95, summary, transform=ax10.transAxes, fontsize=9,
             va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    # Title
    fig.suptitle('Reach Kinematics: Within-Subject Analysis with Fatigue Effects',
                 fontsize=14, fontweight='bold', y=0.98)

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
    print("REACH KINEMATICS - PROPER WITHIN-SUBJECT ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n1. Loading feature data...")
    df = load_all_features(features_dir)
    print(f"   {len(df):,} reaches, {df['animal'].nunique()} mice")

    # Subject distribution
    print("\n2. Analyzing subject distribution...")
    subject_stats = analyze_subject_distribution(df)
    print(f"   Reaches per mouse: {subject_stats['n_reaches'].min()}-{subject_stats['n_reaches'].max()} "
          f"(median: {subject_stats['n_reaches'].median():.0f})")
    print(f"   Success rates: {subject_stats['success_rate'].min()*100:.1f}%-{subject_stats['success_rate'].max()*100:.1f}%")

    # Within-subject analysis
    print("\n3. Running within-subject paired comparisons...")
    features_to_test = ['max_extent_mm', 'peak_velocity_px_per_frame', 'duration_frames',
                        'trajectory_straightness', 'mean_velocity_px_per_frame']

    within_results = []
    for feat in features_to_test:
        if feat in df.columns:
            result = within_subject_comparison(df, feat)
            within_results.append(result)
            if result.get('paired_possible', False):
                sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
                print(f"   {feat}: p={result['p_value']:.4f} {sig} (d={result['cohens_d_paired']:.2f})")

    # Fatigue analysis
    print("\n4. Analyzing fatigue/temporal effects...")
    fatigue_stats, df = analyze_fatigue_effects(df)
    print(f"   Early success rate: {fatigue_stats[fatigue_stats['session_phase']=='Early (1-30)']['success_rate'].values[0]*100:.1f}%")
    print(f"   Late success rate: {fatigue_stats[fatigue_stats['session_phase']=='Late (60+)']['success_rate'].values[0]*100:.1f}%")

    # Mixed effects (if available)
    if HAS_MIXED:
        print("\n5. Running mixed-effects models...")
        for feat in ['max_extent_mm', 'peak_velocity_px_per_frame']:
            result = run_mixed_effects(df, feat)
            if result and 'error' not in result:
                print(f"   {feat}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Generate figure
    print("\n6. Generating publication figure...")
    fig_path = pipeline_dir / 'kinematic_analysis_within_subject.png'
    create_proper_analysis_figure(df, subject_stats, within_results, fatigue_stats, fig_path)

    # Save results
    print("\n7. Saving results...")
    stats_df = pd.DataFrame(within_results)
    stats_df.to_csv(pipeline_dir / 'within_subject_statistics.csv', index=False)
    subject_stats.to_csv(pipeline_dir / 'subject_distribution.csv', index=False)
    fatigue_stats.to_csv(pipeline_dir / 'fatigue_analysis.csv', index=False)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
