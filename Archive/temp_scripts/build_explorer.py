"""Build reach explorer database with proper outcome categories.

Outcome hierarchy:
    - retrieved: Mouse grabbed pellet and brought it into box
    - displaced: Mouse touched pellet but failed to grasp
        - displaced_sa: Knocked into staging area
        - displaced_outside: Knocked out of reach
    - untouched: Mouse missed entirely (no pellet contact)
"""
import sys
import pandas as pd
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict
import numpy as np


@dataclass
class KinematicProfile:
    """Statistics for a set of reaches with proper outcome breakdown."""
    n_reaches: int

    # Outcome counts (3 main categories)
    n_retrieved: int      # Successfully grabbed
    n_displaced: int      # Touched but failed grasp (displaced_sa + displaced_outside)
    n_untouched: int      # Complete miss

    # Subtype counts
    n_displaced_sa: int        # Knocked into staging area
    n_displaced_outside: int   # Knocked out of reach

    # Rates
    retrieval_rate: float      # n_retrieved / n_reaches
    contact_rate: float        # (n_retrieved + n_displaced) / n_reaches (touched pellet)
    grasp_efficiency: float    # n_retrieved / (n_retrieved + n_displaced) (grasped given contact)

    # Kinematic stats
    extent_mean: float
    extent_std: float
    extent_median: float
    extent_q25: float
    extent_q75: float
    duration_mean: float
    duration_std: float
    duration_median: float
    velocity_mean: float
    velocity_std: float
    velocity_median: float
    straightness_mean: Optional[float] = None
    straightness_std: Optional[float] = None

    @classmethod
    def from_dataframe(cls, df):
        n = len(df)
        if n == 0:
            return cls(
                n_reaches=0,
                n_retrieved=0, n_displaced=0, n_untouched=0,
                n_displaced_sa=0, n_displaced_outside=0,
                retrieval_rate=0, contact_rate=0, grasp_efficiency=0,
                extent_mean=0, extent_std=0, extent_median=0, extent_q25=0, extent_q75=0,
                duration_mean=0, duration_std=0, duration_median=0,
                velocity_mean=0, velocity_std=0, velocity_median=0
            )

        # Count outcomes
        outcomes = df['outcome'].value_counts() if 'outcome' in df.columns else pd.Series()
        n_retrieved = int(outcomes.get('retrieved', 0))
        n_displaced_sa = int(outcomes.get('displaced_sa', 0))
        n_displaced_outside = int(outcomes.get('displaced_outside', 0))
        n_untouched = int(outcomes.get('untouched', 0))
        n_displaced = n_displaced_sa + n_displaced_outside

        # Calculate rates
        retrieval_rate = n_retrieved / n if n > 0 else 0
        contact_rate = (n_retrieved + n_displaced) / n if n > 0 else 0
        grasp_efficiency = n_retrieved / (n_retrieved + n_displaced) if (n_retrieved + n_displaced) > 0 else 0

        # Kinematic stats
        extent = df['max_extent_mm'].dropna() if 'max_extent_mm' in df.columns else pd.Series()
        duration = df['duration_sec'].dropna() if 'duration_sec' in df.columns else pd.Series()
        velocity = df['peak_velocity_px_per_frame'].dropna() if 'peak_velocity_px_per_frame' in df.columns else pd.Series()
        straightness = df['trajectory_straightness'].dropna() if 'trajectory_straightness' in df.columns else pd.Series()

        return cls(
            n_reaches=n,
            n_retrieved=n_retrieved,
            n_displaced=n_displaced,
            n_untouched=n_untouched,
            n_displaced_sa=n_displaced_sa,
            n_displaced_outside=n_displaced_outside,
            retrieval_rate=float(retrieval_rate),
            contact_rate=float(contact_rate),
            grasp_efficiency=float(grasp_efficiency),
            extent_mean=float(extent.mean()) if len(extent) > 0 else 0,
            extent_std=float(extent.std()) if len(extent) > 0 else 0,
            extent_median=float(extent.median()) if len(extent) > 0 else 0,
            extent_q25=float(extent.quantile(0.25)) if len(extent) > 0 else 0,
            extent_q75=float(extent.quantile(0.75)) if len(extent) > 0 else 0,
            duration_mean=float(duration.mean()) if len(duration) > 0 else 0,
            duration_std=float(duration.std()) if len(duration) > 0 else 0,
            duration_median=float(duration.median()) if len(duration) > 0 else 0,
            velocity_mean=float(velocity.mean()) if len(velocity) > 0 else 0,
            velocity_std=float(velocity.std()) if len(velocity) > 0 else 0,
            velocity_median=float(velocity.median()) if len(velocity) > 0 else 0,
            straightness_mean=float(straightness.mean()) if len(straightness) > 0 else None,
            straightness_std=float(straightness.std()) if len(straightness) > 0 else None,
        )


@dataclass
class OutcomeKinematics:
    """Kinematic stats broken down by outcome type."""
    outcome: str
    n: int
    extent_mean: float
    extent_std: float
    duration_mean: float
    duration_std: float
    velocity_mean: float
    velocity_std: float

    @classmethod
    def from_dataframe(cls, df, outcome: str):
        subset = df[df['outcome'] == outcome] if 'outcome' in df.columns else pd.DataFrame()
        n = len(subset)
        if n == 0:
            return cls(outcome=outcome, n=0, extent_mean=0, extent_std=0,
                      duration_mean=0, duration_std=0, velocity_mean=0, velocity_std=0)

        return cls(
            outcome=outcome,
            n=n,
            extent_mean=float(subset['max_extent_mm'].mean()) if 'max_extent_mm' in subset.columns else 0,
            extent_std=float(subset['max_extent_mm'].std()) if 'max_extent_mm' in subset.columns else 0,
            duration_mean=float(subset['duration_sec'].mean()) if 'duration_sec' in subset.columns else 0,
            duration_std=float(subset['duration_sec'].std()) if 'duration_sec' in subset.columns else 0,
            velocity_mean=float(subset['peak_velocity_px_per_frame'].mean()) if 'peak_velocity_px_per_frame' in subset.columns else 0,
            velocity_std=float(subset['peak_velocity_px_per_frame'].std()) if 'peak_velocity_px_per_frame' in subset.columns else 0,
        )


def compute_outcome_breakdown(df) -> Dict[str, dict]:
    """Compute kinematic profiles for each outcome type."""
    outcomes = {}
    for outcome in ['retrieved', 'displaced_sa', 'displaced_outside', 'untouched']:
        ok = OutcomeKinematics.from_dataframe(df, outcome)
        if ok.n > 0:
            outcomes[outcome] = asdict(ok)
    return outcomes


if __name__ == '__main__':
    pipeline_dir = Path(__file__).parent

    # Load data
    print('Loading reach data...')
    df = pd.read_excel(pipeline_dir / 'unified_reaches_for_presentation.xlsx', sheet_name='All_Reaches')
    print(f'  {len(df)} reaches from {df["animal"].nunique()} mice')

    # Show outcome distribution
    print('\n  Outcome distribution:')
    for outcome, count in df['outcome'].value_counts().items():
        print(f'    {outcome}: {count} ({count/len(df)*100:.1f}%)')

    # Load features
    features_dir = pipeline_dir / 'Step5_Features'
    all_features = []
    for ff in features_dir.glob('*_features.json'):
        with open(ff) as f:
            data = json.load(f)
        video_name = data.get('video_name')
        for seg in data.get('segments', []):
            for reach in seg.get('reaches', []):
                all_features.append({
                    'video_name': video_name,
                    'segment_num': seg.get('segment_num'),
                    'reach_num': reach.get('reach_num'),
                    'peak_velocity_px_per_frame': reach.get('peak_velocity_px_per_frame'),
                    'trajectory_straightness': reach.get('trajectory_straightness'),
                })
    features_df = pd.DataFrame(all_features)
    print(f'\n  {len(features_df)} feature records')

    # Merge features
    df = df.rename(columns={'reach_num_in_segment': 'reach_num'})
    df = df.merge(features_df, on=['video_name', 'segment_num', 'reach_num'], how='left', suffixes=('', '_feat'))

    # Create SQLite DB
    db_path = pipeline_dir / 'reach_explorer.db'
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)

    # Create tables with proper outcome tracking
    conn.executescript('''
    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);

    CREATE TABLE population_stats (
        id INTEGER PRIMARY KEY,
        profile_json TEXT,
        outcome_breakdown_json TEXT
    );

    CREATE TABLE mouse_stats (
        animal TEXT PRIMARY KEY,
        cohort TEXT,
        n_sessions INTEGER,
        profile_json TEXT,
        outcome_breakdown_json TEXT,
        learning_curve_json TEXT
    );

    CREATE TABLE session_stats (
        video_name TEXT PRIMARY KEY,
        animal TEXT,
        date TEXT,
        tray_type TEXT,
        run_num INTEGER,
        profile_json TEXT,
        outcome_breakdown_json TEXT
    );

    CREATE TABLE reaches (
        reach_id TEXT PRIMARY KEY,
        video_name TEXT,
        animal TEXT,
        segment_num INTEGER,
        reach_num INTEGER,
        outcome TEXT,
        extent_mm REAL,
        duration_sec REAL,
        velocity REAL,
        straightness REAL
    );

    CREATE INDEX idx_reaches_animal ON reaches(animal);
    CREATE INDEX idx_reaches_outcome ON reaches(outcome);
    CREATE INDEX idx_reaches_video ON reaches(video_name);
    ''')

    # Store metadata
    conn.execute('INSERT INTO metadata VALUES (?, ?)', ('build_date', pd.Timestamp.now().isoformat()))
    conn.execute('INSERT INTO metadata VALUES (?, ?)', ('n_reaches', str(len(df))))
    conn.execute('INSERT INTO metadata VALUES (?, ?)', ('n_animals', str(df['animal'].nunique())))

    # Population stats
    print('\nComputing population statistics...')
    pop_profile = KinematicProfile.from_dataframe(df)
    pop_breakdown = compute_outcome_breakdown(df)
    conn.execute('INSERT INTO population_stats VALUES (?, ?, ?)',
                 (1, json.dumps(asdict(pop_profile)), json.dumps(pop_breakdown)))

    # Mouse stats
    print('Computing per-mouse statistics...')
    for animal in df['animal'].unique():
        mouse_df = df[df['animal'] == animal]
        profile = KinematicProfile.from_dataframe(mouse_df)
        breakdown = compute_outcome_breakdown(mouse_df)
        cohort = mouse_df['cohort'].iloc[0] if 'cohort' in mouse_df.columns else None

        # Learning curve with outcome breakdown
        learning = []
        if 'date' in mouse_df.columns:
            for date in sorted(mouse_df['date'].dropna().unique()):
                ddf = mouse_df[mouse_df['date'] == date]
                outcomes = ddf['outcome'].value_counts()
                learning.append({
                    'date': str(date),
                    'n_reaches': len(ddf),
                    'retrieval_rate': float(outcomes.get('retrieved', 0) / len(ddf)) if len(ddf) > 0 else 0,
                    'n_retrieved': int(outcomes.get('retrieved', 0)),
                    'n_displaced': int(outcomes.get('displaced_sa', 0) + outcomes.get('displaced_outside', 0)),
                    'n_untouched': int(outcomes.get('untouched', 0)),
                })

        conn.execute('INSERT INTO mouse_stats VALUES (?, ?, ?, ?, ?, ?)',
                     (animal, cohort, mouse_df['video_name'].nunique(),
                      json.dumps(asdict(profile)), json.dumps(breakdown), json.dumps(learning)))

    # Session stats
    print('Computing per-session statistics...')
    for video in df['video_name'].unique():
        sdf = df[df['video_name'] == video]
        profile = KinematicProfile.from_dataframe(sdf)
        breakdown = compute_outcome_breakdown(sdf)
        conn.execute('INSERT INTO session_stats VALUES (?, ?, ?, ?, ?, ?, ?)',
                     (video, sdf['animal'].iloc[0],
                      str(sdf['date'].iloc[0]) if 'date' in sdf.columns else None,
                      sdf['tray_type'].iloc[0] if 'tray_type' in sdf.columns else None,
                      int(sdf['run_num'].iloc[0]) if 'run_num' in sdf.columns and pd.notna(sdf['run_num'].iloc[0]) else None,
                      json.dumps(asdict(profile)), json.dumps(breakdown)))

    # Individual reaches
    print('Storing reaches...')
    for _, r in df.iterrows():
        conn.execute('INSERT OR REPLACE INTO reaches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                     (r.get('reach_id'), r.get('video_name'), r.get('animal'),
                      r.get('segment_num'), r.get('reach_num'), r.get('outcome'),
                      r.get('max_extent_mm'), r.get('duration_sec'),
                      r.get('peak_velocity_px_per_frame'), r.get('trajectory_straightness')))

    conn.commit()
    conn.close()

    # Print summary
    print(f"""
{'='*70}
DATABASE COMPLETE
{'='*70}
Location: {db_path}
Size: {db_path.stat().st_size / 1024:.1f} KB

POPULATION SUMMARY:
  Total reaches: {pop_profile.n_reaches}
  Retrieved: {pop_profile.n_retrieved} ({pop_profile.retrieval_rate*100:.1f}%)
  Displaced: {pop_profile.n_displaced} ({pop_profile.n_displaced/pop_profile.n_reaches*100:.1f}%)
    - Into staging area: {pop_profile.n_displaced_sa}
    - Outside: {pop_profile.n_displaced_outside}
  Untouched: {pop_profile.n_untouched} ({pop_profile.n_untouched/pop_profile.n_reaches*100:.1f}%)

  Contact rate: {pop_profile.contact_rate*100:.1f}% (reached the pellet)
  Grasp efficiency: {pop_profile.grasp_efficiency*100:.1f}% (retrieved given contact)

Mice: {list(df['animal'].unique())}
""")
