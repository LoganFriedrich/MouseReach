"""
Pellet Likelihood Ceiling Analysis

Measures the peak DLC confidence for pellet detection across animals and sessions.
Key insight: If DLC's best confidence is consistently low for certain animals,
that proves it's a tracking/visibility problem, not real pellet removal.

Author: Claude Code
Date: 2026-02-11
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
DLC_BASE = Path(r"X:\! DLC Output\Analyzed")
OUTPUT_PATH = Path(r"Y:\2_Connectome\MouseDB\exports\pellet_ceiling_analysis.csv")
GROUPS = ['K', 'L', 'M']
FLAGGED_ANIMALS = {'L02', 'L10', 'L12', 'L13'}

def extract_animal_id(filename: str) -> str:
    """Extract animal ID (e.g., 'L02') from filename."""
    match = re.search(r'\d{8}_([A-Z]\d{2})_', filename)
    return match.group(1) if match else None

def load_dlc_csv(csv_path: Path) -> pd.DataFrame:
    """Load DLC CSV with 3-row header."""
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        return df
    except Exception as e:
        print(f"Error loading {csv_path.name}: {e}")
        return None

def get_bodypart_likelihood(df: pd.DataFrame, bodypart: str) -> np.ndarray:
    """Extract likelihood column for a bodypart."""
    try:
        # DLC format: (scorer, bodypart, coord)
        # We want the 'likelihood' column for the bodypart
        for col in df.columns:
            if bodypart.lower() in str(col[1]).lower() and 'likelihood' in str(col[2]).lower():
                return df[col].values
        return None
    except Exception as e:
        print(f"Error extracting {bodypart}: {e}")
        return None

def compute_session_metrics(df: pd.DataFrame) -> Dict:
    """Compute all ceiling metrics for a single session."""
    pellet_lik = get_bodypart_likelihood(df, 'Pellet')
    pillar_lik = get_bodypart_likelihood(df, 'Pillar')

    metrics = {}

    # Pellet metrics
    if pellet_lik is not None and len(pellet_lik) > 0:
        valid_pellet = pellet_lik[~np.isnan(pellet_lik)]
        if len(valid_pellet) > 0:
            metrics['pellet_95pct'] = np.percentile(valid_pellet, 95)
            metrics['pellet_max'] = np.max(valid_pellet)
            metrics['pellet_mean'] = np.mean(valid_pellet)
            metrics['pellet_median'] = np.median(valid_pellet)

            # First 500 frames (pellet definitely on pillar)
            first_500 = pellet_lik[:500]
            first_500_valid = first_500[~np.isnan(first_500)]
            if len(first_500_valid) > 0:
                metrics['pellet_early_mean'] = np.mean(first_500_valid)
                metrics['pellet_early_95pct'] = np.percentile(first_500_valid, 95)
            else:
                metrics['pellet_early_mean'] = np.nan
                metrics['pellet_early_95pct'] = np.nan
        else:
            metrics.update({
                'pellet_95pct': np.nan,
                'pellet_max': np.nan,
                'pellet_mean': np.nan,
                'pellet_median': np.nan,
                'pellet_early_mean': np.nan,
                'pellet_early_95pct': np.nan
            })
    else:
        metrics.update({
            'pellet_95pct': np.nan,
            'pellet_max': np.nan,
            'pellet_mean': np.nan,
            'pellet_median': np.nan,
            'pellet_early_mean': np.nan,
            'pellet_early_95pct': np.nan
        })

    # Pillar metrics (control)
    if pillar_lik is not None and len(pillar_lik) > 0:
        valid_pillar = pillar_lik[~np.isnan(pillar_lik)]
        if len(valid_pillar) > 0:
            metrics['pillar_95pct'] = np.percentile(valid_pillar, 95)
            metrics['pillar_max'] = np.max(valid_pillar)
            metrics['pillar_mean'] = np.mean(valid_pillar)
            metrics['pillar_median'] = np.median(valid_pillar)

            # First 500 frames
            first_500 = pillar_lik[:500]
            first_500_valid = first_500[~np.isnan(first_500)]
            if len(first_500_valid) > 0:
                metrics['pillar_early_mean'] = np.mean(first_500_valid)
                metrics['pillar_early_95pct'] = np.percentile(first_500_valid, 95)
            else:
                metrics['pillar_early_mean'] = np.nan
                metrics['pillar_early_95pct'] = np.nan
        else:
            metrics.update({
                'pillar_95pct': np.nan,
                'pillar_max': np.nan,
                'pillar_mean': np.nan,
                'pillar_median': np.nan,
                'pillar_early_mean': np.nan,
                'pillar_early_95pct': np.nan
            })
    else:
        metrics.update({
            'pillar_95pct': np.nan,
            'pillar_max': np.nan,
            'pillar_mean': np.nan,
            'pillar_median': np.nan,
            'pillar_early_mean': np.nan,
            'pillar_early_95pct': np.nan
        })

    # Ceiling gap (positive = pellet easier to see than pillar)
    if not np.isnan(metrics.get('pellet_95pct', np.nan)) and not np.isnan(metrics.get('pillar_95pct', np.nan)):
        metrics['ceiling_gap'] = metrics['pellet_95pct'] - metrics['pillar_95pct']
    else:
        metrics['ceiling_gap'] = np.nan

    return metrics

def process_group(group: str, max_files: int = 200) -> pd.DataFrame:
    """Process all CSV files for a group."""
    group_dir = DLC_BASE / group / "Post-Processing"
    if not group_dir.exists():
        print(f"Warning: {group_dir} does not exist")
        return pd.DataFrame()

    csv_files = sorted(group_dir.glob("*DLC*.csv"))

    # Prioritize flagged animals - take all their files
    flagged_files = [f for f in csv_files if extract_animal_id(f.name) in FLAGGED_ANIMALS]
    other_files = [f for f in csv_files if extract_animal_id(f.name) not in FLAGGED_ANIMALS]

    # Take all flagged + up to max_files total
    files_to_process = flagged_files + other_files[:max(0, max_files - len(flagged_files))]

    print(f"\nProcessing Group {group}: {len(files_to_process)} files ({len(flagged_files)} flagged)")

    results = []
    for i, csv_file in enumerate(files_to_process, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(files_to_process)} files...")

        animal_id = extract_animal_id(csv_file.name)
        if animal_id is None:
            continue

        df = load_dlc_csv(csv_file)
        if df is None:
            continue

        metrics = compute_session_metrics(df)
        metrics['group'] = group
        metrics['animal_id'] = animal_id
        metrics['filename'] = csv_file.name
        metrics['is_flagged'] = animal_id in FLAGGED_ANIMALS

        results.append(metrics)

    print(f"  Completed: {len(results)} sessions processed")
    return pd.DataFrame(results)

def compute_animal_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-animal summary statistics."""
    summaries = []

    for (group, animal), animal_df in df.groupby(['group', 'animal_id']):
        summary = {
            'group': group,
            'animal_id': animal,
            'is_flagged': animal_df['is_flagged'].iloc[0],
            'n_sessions': len(animal_df),

            # Pellet ceiling metrics
            'pellet_95pct_mean': animal_df['pellet_95pct'].mean(),
            'pellet_95pct_median': animal_df['pellet_95pct'].median(),
            'pellet_95pct_min': animal_df['pellet_95pct'].min(),
            'pellet_max_mean': animal_df['pellet_max'].mean(),

            # Early frames (pellet definitely present)
            'pellet_early_95pct_mean': animal_df['pellet_early_95pct'].mean(),

            # Pillar ceiling (control)
            'pillar_95pct_mean': animal_df['pillar_95pct'].mean(),
            'pillar_95pct_median': animal_df['pillar_95pct'].median(),

            # Ceiling gap
            'ceiling_gap_mean': animal_df['ceiling_gap'].mean(),
            'ceiling_gap_median': animal_df['ceiling_gap'].median(),

            # Flag if ceiling is consistently low
            'ceiling_below_0.8_pct': (animal_df['pellet_95pct'] < 0.8).mean() * 100,
            'has_tracking_problem': animal_df['pellet_95pct'].mean() < 0.8
        }

        summaries.append(summary)

    return pd.DataFrame(summaries).sort_values('pellet_95pct_mean')

def print_summary_tables(session_df: pd.DataFrame, animal_df: pd.DataFrame):
    """Print formatted summary tables."""
    print("\n" + "="*80)
    print("PELLET LIKELIHOOD CEILING ANALYSIS")
    print("="*80)

    # Overall statistics
    print("\n--- OVERALL STATISTICS ---")
    print(f"Total sessions analyzed: {len(session_df)}")
    print(f"Total animals: {animal_df['animal_id'].nunique()}")
    print(f"Flagged animals: {animal_df[animal_df['is_flagged']]['animal_id'].nunique()}")

    print(f"\nGlobal pellet 95th percentile: {session_df['pellet_95pct'].mean():.3f} +/- {session_df['pellet_95pct'].std():.3f}")
    print(f"Global pillar 95th percentile: {session_df['pillar_95pct'].mean():.3f} +/- {session_df['pillar_95pct'].std():.3f}")
    print(f"Global ceiling gap: {session_df['ceiling_gap'].mean():.3f} +/- {session_df['ceiling_gap'].std():.3f}")

    # Per-animal table
    print("\n--- PER-ANIMAL CEILING METRICS (sorted by pellet ceiling quality) ---")
    print(f"{'Animal':<8} {'Grp':<4} {'Flag':<5} {'N':<4} {'Pellet_95pct':<13} {'Early_95pct':<12} {'Pillar_95pct':<13} {'Gap':<8} {'Problem'}")
    print("-" * 100)

    for _, row in animal_df.iterrows():
        flag_str = "YES" if row['is_flagged'] else "NO"
        problem_str = "YES" if row['has_tracking_problem'] else "NO"

        print(f"{row['animal_id']:<8} {row['group']:<4} {flag_str:<5} {row['n_sessions']:<4.0f} "
              f"{row['pellet_95pct_mean']:>6.3f} +/- {row['pellet_95pct_median']:>5.3f} "
              f"{row['pellet_early_95pct_mean']:>6.3f}      "
              f"{row['pillar_95pct_mean']:>6.3f} +/- {row['pillar_95pct_median']:>5.3f} "
              f"{row['ceiling_gap_mean']:>6.3f}  {problem_str}")

    # Group comparisons
    print("\n--- GROUP COMPARISONS ---")
    for group in GROUPS:
        group_data = animal_df[animal_df['group'] == group]
        if len(group_data) == 0:
            continue
        print(f"\nGroup {group} (n={len(group_data)} animals):")
        print(f"  Pellet 95th pct: {group_data['pellet_95pct_mean'].mean():.3f} +/- {group_data['pellet_95pct_mean'].std():.3f}")
        print(f"  Pillar 95th pct: {group_data['pillar_95pct_mean'].mean():.3f} +/- {group_data['pillar_95pct_mean'].std():.3f}")
        print(f"  Ceiling gap:     {group_data['ceiling_gap_mean'].mean():.3f} +/- {group_data['ceiling_gap_mean'].std():.3f}")
        print(f"  Animals with tracking problem: {group_data['has_tracking_problem'].sum()}/{len(group_data)}")

    # Flagged vs unflagged in Group L
    print("\n--- FLAGGED VS UNFLAGGED IN GROUP L ---")
    group_l = animal_df[animal_df['group'] == 'L']
    if len(group_l) > 0:
        flagged = group_l[group_l['is_flagged']]
        unflagged = group_l[~group_l['is_flagged']]

        print(f"Flagged (n={len(flagged)}):")
        print(f"  Pellet 95th pct: {flagged['pellet_95pct_mean'].mean():.3f} +/- {flagged['pellet_95pct_mean'].std():.3f}")
        print(f"  Pillar 95th pct: {flagged['pillar_95pct_mean'].mean():.3f} +/- {flagged['pillar_95pct_mean'].std():.3f}")
        print(f"  Ceiling gap:     {flagged['ceiling_gap_mean'].mean():.3f} +/- {flagged['ceiling_gap_mean'].std():.3f}")

        print(f"\nUnflagged (n={len(unflagged)}):")
        print(f"  Pellet 95th pct: {unflagged['pellet_95pct_mean'].mean():.3f} +/- {unflagged['pellet_95pct_mean'].std():.3f}")
        print(f"  Pillar 95th pct: {unflagged['pillar_95pct_mean'].mean():.3f} +/- {unflagged['pillar_95pct_mean'].std():.3f}")
        print(f"  Ceiling gap:     {unflagged['ceiling_gap_mean'].mean():.3f} +/- {unflagged['ceiling_gap_mean'].std():.3f}")

        # Statistical test
        if len(flagged) > 0 and len(unflagged) > 0:
            u_stat, p_val = stats.mannwhitneyu(
                flagged['pellet_95pct_mean'].dropna(),
                unflagged['pellet_95pct_mean'].dropna(),
                alternative='two-sided'
            )
            print(f"\nMann-Whitney U test (pellet ceiling): U={u_stat:.1f}, p={p_val:.4f}")

    # Interpretation
    print("\n--- INTERPRETATION ---")
    print("Ceiling quality categories:")
    print("  Excellent:  95th pct > 0.95 (DLC reliably sees pellet)")
    print("  Good:       95th pct 0.85-0.95 (DLC mostly sees pellet)")
    print("  Marginal:   95th pct 0.70-0.85 (DLC struggles with pellet)")
    print("  Poor:       95th pct < 0.70 (definitive tracking problem)")

    excellent = animal_df[animal_df['pellet_95pct_mean'] > 0.95]
    good = animal_df[(animal_df['pellet_95pct_mean'] > 0.85) & (animal_df['pellet_95pct_mean'] <= 0.95)]
    marginal = animal_df[(animal_df['pellet_95pct_mean'] > 0.70) & (animal_df['pellet_95pct_mean'] <= 0.85)]
    poor = animal_df[animal_df['pellet_95pct_mean'] <= 0.70]

    print(f"\nAnimal distribution:")
    print(f"  Excellent: {len(excellent)} animals")
    print(f"  Good:      {len(good)} animals")
    print(f"  Marginal:  {len(marginal)} animals")
    print(f"  Poor:      {len(poor)} animals")

    if len(poor) > 0:
        print(f"\nAnimals with definitive tracking problems (95th pct < 0.70):")
        for _, row in poor.iterrows():
            flag_str = " [FLAGGED]" if row['is_flagged'] else ""
            print(f"  {row['animal_id']} (Group {row['group']}): {row['pellet_95pct_mean']:.3f}{flag_str}")

def main():
    """Main analysis pipeline."""
    print("Pellet Likelihood Ceiling Analysis")
    print("=" * 80)

    # Process all groups (limit to 100 files per group for speed)
    all_sessions = []
    for group in GROUPS:
        group_df = process_group(group, max_files=100)
        if len(group_df) > 0:
            all_sessions.append(group_df)

    if len(all_sessions) == 0:
        print("ERROR: No data found!")
        return

    session_df = pd.concat(all_sessions, ignore_index=True)

    # Compute animal summaries
    animal_df = compute_animal_summaries(session_df)

    # Print results
    print_summary_tables(session_df, animal_df)

    # Save detailed results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Combine session and animal data for export
    export_df = session_df.merge(
        animal_df[['animal_id', 'pellet_95pct_mean', 'has_tracking_problem']],
        on='animal_id',
        suffixes=('_session', '_animal_mean')
    )

    export_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Detailed results saved to: {OUTPUT_PATH}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
