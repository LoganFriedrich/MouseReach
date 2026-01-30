"""
Reach Data Explorer - Pre-computed statistics at every analysis level.

Build once with `mousereach-build-explorer`, then query instantly.

Hierarchy:
    Population -> Mouse -> Session -> Segment -> Reach

Each level stores:
    - Kinematic statistics (mean, std, median, IQR)
    - Success/fail comparisons
    - Temporal patterns (fatigue, learning)
"""
import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class KinematicProfile:
    """Summary statistics for a set of reaches."""
    n_reaches: int
    n_success: int
    n_fail: int
    success_rate: float

    # Extent (mm)
    extent_mean: float
    extent_std: float
    extent_median: float
    extent_q25: float
    extent_q75: float

    # Duration (seconds)
    duration_mean: float
    duration_std: float
    duration_median: float

    # Velocity (px/frame)
    velocity_mean: float
    velocity_std: float
    velocity_median: float

    # Trajectory straightness (0-1)
    straightness_mean: Optional[float] = None
    straightness_std: Optional[float] = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'KinematicProfile':
        """Compute profile from reach dataframe."""
        n = len(df)
        if n == 0:
            return cls(
                n_reaches=0, n_success=0, n_fail=0, success_rate=0,
                extent_mean=0, extent_std=0, extent_median=0, extent_q25=0, extent_q75=0,
                duration_mean=0, duration_std=0, duration_median=0,
                velocity_mean=0, velocity_std=0, velocity_median=0
            )

        is_success = df['outcome'] == 'retrieved' if 'outcome' in df.columns else df.get('is_success', pd.Series([False]*n))
        n_success = is_success.sum()

        extent = df['max_extent_mm'].dropna() if 'max_extent_mm' in df.columns else pd.Series()
        duration = df['duration_sec'].dropna() if 'duration_sec' in df.columns else pd.Series()
        velocity = df['peak_velocity_px_per_frame'].dropna() if 'peak_velocity_px_per_frame' in df.columns else pd.Series()
        straightness = df['trajectory_straightness'].dropna() if 'trajectory_straightness' in df.columns else pd.Series()

        return cls(
            n_reaches=n,
            n_success=int(n_success),
            n_fail=int(n - n_success),
            success_rate=float(n_success / n) if n > 0 else 0,
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
class SuccessFailComparison:
    """Comparison between successful and failed reaches."""
    feature: str
    success_mean: float
    success_std: float
    success_n: int
    fail_mean: float
    fail_std: float
    fail_n: int
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool  # p < 0.05

    @classmethod
    def compute(cls, df: pd.DataFrame, feature: str) -> Optional['SuccessFailComparison']:
        """Compute success vs fail comparison for a feature."""
        if feature not in df.columns:
            return None

        is_success = df['outcome'] == 'retrieved' if 'outcome' in df.columns else df.get('is_success', pd.Series([False]*len(df)))

        success_vals = df[is_success][feature].dropna()
        fail_vals = df[~is_success][feature].dropna()

        if len(success_vals) < 2 or len(fail_vals) < 2:
            return None

        t_stat, p_val = stats.ttest_ind(success_vals, fail_vals)
        pooled_std = np.sqrt((success_vals.std()**2 + fail_vals.std()**2) / 2)
        cohens_d = (success_vals.mean() - fail_vals.mean()) / pooled_std if pooled_std > 0 else 0

        return cls(
            feature=feature,
            success_mean=float(success_vals.mean()),
            success_std=float(success_vals.std()),
            success_n=len(success_vals),
            fail_mean=float(fail_vals.mean()),
            fail_std=float(fail_vals.std()),
            fail_n=len(fail_vals),
            t_statistic=float(t_stat),
            p_value=float(p_val),
            cohens_d=float(cohens_d),
            significant=p_val < 0.05
        )


@dataclass
class TemporalPattern:
    """How performance changes over time within a session or across sessions."""
    phase: str  # 'early', 'middle', 'late' OR session date
    n_reaches: int
    success_rate: float
    extent_mean: float
    velocity_mean: float


# =============================================================================
# EXPLORER DATABASE
# =============================================================================

class ReachExplorer:
    """
    Pre-computed reach statistics database.

    Levels:
        - population: All mice combined
        - mouse: Per-animal statistics
        - session: Per-video/day statistics
        - segment: Per-pellet-attempt statistics
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS population_stats (
        id INTEGER PRIMARY KEY,
        profile_json TEXT,
        comparisons_json TEXT,
        temporal_json TEXT
    );

    CREATE TABLE IF NOT EXISTS mouse_stats (
        animal TEXT PRIMARY KEY,
        cohort TEXT,
        n_sessions INTEGER,
        first_session TEXT,
        last_session TEXT,
        profile_json TEXT,
        comparisons_json TEXT,
        learning_curve_json TEXT
    );

    CREATE TABLE IF NOT EXISTS session_stats (
        video_name TEXT PRIMARY KEY,
        animal TEXT,
        date TEXT,
        tray_type TEXT,
        run_num INTEGER,
        profile_json TEXT,
        comparisons_json TEXT,
        fatigue_json TEXT,
        FOREIGN KEY (animal) REFERENCES mouse_stats(animal)
    );

    CREATE TABLE IF NOT EXISTS reaches (
        reach_id TEXT PRIMARY KEY,
        video_name TEXT,
        animal TEXT,
        segment_num INTEGER,
        reach_num INTEGER,
        outcome TEXT,
        extent_mm REAL,
        duration_sec REAL,
        velocity REAL,
        straightness REAL,
        is_success INTEGER,
        FOREIGN KEY (video_name) REFERENCES session_stats(video_name)
    );

    CREATE INDEX IF NOT EXISTS idx_reaches_animal ON reaches(animal);
    CREATE INDEX IF NOT EXISTS idx_reaches_outcome ON reaches(outcome);
    CREATE INDEX IF NOT EXISTS idx_reaches_video ON reaches(video_name);
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = None

    def connect(self):
        """Open database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def initialize(self):
        """Create database schema."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def set_metadata(self, key: str, value: str):
        """Store metadata."""
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value)
        )
        self.conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        """Retrieve metadata."""
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row['value'] if row else None

    # -------------------------------------------------------------------------
    # Population Level
    # -------------------------------------------------------------------------

    def store_population_stats(self, profile: KinematicProfile,
                               comparisons: List[SuccessFailComparison],
                               temporal: List[TemporalPattern]):
        """Store population-level statistics."""
        self.conn.execute(
            "INSERT OR REPLACE INTO population_stats (id, profile_json, comparisons_json, temporal_json) VALUES (?, ?, ?, ?)",
            (1,
             json.dumps(asdict(profile)),
             json.dumps([asdict(c) for c in comparisons if c]),
             json.dumps([asdict(t) for t in temporal]))
        )
        self.conn.commit()

    def get_population_stats(self) -> Dict[str, Any]:
        """Get population-level statistics."""
        row = self.conn.execute(
            "SELECT * FROM population_stats WHERE id = 1"
        ).fetchone()
        if not row:
            return {}
        return {
            'profile': json.loads(row['profile_json']),
            'comparisons': json.loads(row['comparisons_json']),
            'temporal': json.loads(row['temporal_json'])
        }

    # -------------------------------------------------------------------------
    # Mouse Level
    # -------------------------------------------------------------------------

    def store_mouse_stats(self, animal: str, cohort: str, n_sessions: int,
                          first_session: str, last_session: str,
                          profile: KinematicProfile,
                          comparisons: List[SuccessFailComparison],
                          learning_curve: List[Dict]):
        """Store per-mouse statistics."""
        self.conn.execute(
            """INSERT OR REPLACE INTO mouse_stats
               (animal, cohort, n_sessions, first_session, last_session,
                profile_json, comparisons_json, learning_curve_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (animal, cohort, n_sessions, first_session, last_session,
             json.dumps(asdict(profile)),
             json.dumps([asdict(c) for c in comparisons if c]),
             json.dumps(learning_curve))
        )
        self.conn.commit()

    def get_mouse_stats(self, animal: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific mouse."""
        row = self.conn.execute(
            "SELECT * FROM mouse_stats WHERE animal = ?", (animal,)
        ).fetchone()
        if not row:
            return None
        return {
            'animal': row['animal'],
            'cohort': row['cohort'],
            'n_sessions': row['n_sessions'],
            'first_session': row['first_session'],
            'last_session': row['last_session'],
            'profile': json.loads(row['profile_json']),
            'comparisons': json.loads(row['comparisons_json']),
            'learning_curve': json.loads(row['learning_curve_json'])
        }

    def list_mice(self) -> List[str]:
        """List all mice in database."""
        rows = self.conn.execute("SELECT animal FROM mouse_stats ORDER BY animal").fetchall()
        return [r['animal'] for r in rows]

    # -------------------------------------------------------------------------
    # Session Level
    # -------------------------------------------------------------------------

    def store_session_stats(self, video_name: str, animal: str, date: str,
                            tray_type: str, run_num: int,
                            profile: KinematicProfile,
                            comparisons: List[SuccessFailComparison],
                            fatigue: List[TemporalPattern]):
        """Store per-session statistics."""
        self.conn.execute(
            """INSERT OR REPLACE INTO session_stats
               (video_name, animal, date, tray_type, run_num,
                profile_json, comparisons_json, fatigue_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (video_name, animal, date, tray_type, run_num,
             json.dumps(asdict(profile)),
             json.dumps([asdict(c) for c in comparisons if c]),
             json.dumps([asdict(f) for f in fatigue]))
        )
        self.conn.commit()

    def get_session_stats(self, video_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session."""
        row = self.conn.execute(
            "SELECT * FROM session_stats WHERE video_name = ?", (video_name,)
        ).fetchone()
        if not row:
            return None
        return {
            'video_name': row['video_name'],
            'animal': row['animal'],
            'date': row['date'],
            'tray_type': row['tray_type'],
            'run_num': row['run_num'],
            'profile': json.loads(row['profile_json']),
            'comparisons': json.loads(row['comparisons_json']),
            'fatigue': json.loads(row['fatigue_json'])
        }

    def list_sessions(self, animal: Optional[str] = None) -> List[str]:
        """List sessions, optionally filtered by animal."""
        if animal:
            rows = self.conn.execute(
                "SELECT video_name FROM session_stats WHERE animal = ? ORDER BY date",
                (animal,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT video_name FROM session_stats ORDER BY date"
            ).fetchall()
        return [r['video_name'] for r in rows]

    # -------------------------------------------------------------------------
    # Reach Level
    # -------------------------------------------------------------------------

    def store_reaches(self, reaches: List[Dict]):
        """Store individual reach records."""
        self.conn.executemany(
            """INSERT OR REPLACE INTO reaches
               (reach_id, video_name, animal, segment_num, reach_num,
                outcome, extent_mm, duration_sec, velocity, straightness, is_success)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(r.get('reach_id'), r.get('video_name'), r.get('animal'),
              r.get('segment_num'), r.get('reach_num_in_segment'),
              r.get('outcome'), r.get('max_extent_mm'), r.get('duration_sec'),
              r.get('peak_velocity_px_per_frame'), r.get('trajectory_straightness'),
              1 if r.get('outcome') == 'retrieved' else 0)
             for r in reaches]
        )
        self.conn.commit()

    def query_reaches(self, animal: Optional[str] = None,
                      outcome: Optional[str] = None,
                      session: Optional[str] = None,
                      limit: int = 1000) -> pd.DataFrame:
        """Query individual reaches with filters."""
        query = "SELECT * FROM reaches WHERE 1=1"
        params = []

        if animal:
            query += " AND animal = ?"
            params.append(animal)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        if session:
            query += " AND video_name = ?"
            params.append(session)

        query += f" LIMIT {limit}"

        return pd.read_sql_query(query, self.conn, params=params)

    # -------------------------------------------------------------------------
    # High-Level Queries
    # -------------------------------------------------------------------------

    def get_mouse_success_profile(self, animal: str) -> Dict[str, Any]:
        """
        Get the 'success profile' for a mouse - what does a successful reach look like?
        """
        mouse = self.get_mouse_stats(animal)
        if not mouse:
            return {}

        # Get successful reaches for this mouse
        success_df = self.query_reaches(animal=animal, outcome='retrieved', limit=10000)
        fail_df = self.query_reaches(animal=animal, outcome='displaced_sa', limit=10000)

        return {
            'animal': animal,
            'overall': mouse['profile'],
            'success_profile': {
                'n': len(success_df),
                'extent_mean': float(success_df['extent_mm'].mean()) if len(success_df) > 0 else None,
                'extent_std': float(success_df['extent_mm'].std()) if len(success_df) > 0 else None,
                'duration_mean': float(success_df['duration_sec'].mean()) if len(success_df) > 0 else None,
                'velocity_mean': float(success_df['velocity'].mean()) if len(success_df) > 0 else None,
            },
            'fail_profile': {
                'n': len(fail_df),
                'extent_mean': float(fail_df['extent_mm'].mean()) if len(fail_df) > 0 else None,
                'extent_std': float(fail_df['extent_mm'].std()) if len(fail_df) > 0 else None,
                'duration_mean': float(fail_df['duration_sec'].mean()) if len(fail_df) > 0 else None,
                'velocity_mean': float(fail_df['velocity'].mean()) if len(fail_df) > 0 else None,
            },
            'comparisons': mouse['comparisons'],
            'learning_curve': mouse['learning_curve']
        }

    def compare_mice(self, animals: List[str], feature: str = 'success_rate') -> pd.DataFrame:
        """Compare multiple mice on a feature."""
        data = []
        for animal in animals:
            mouse = self.get_mouse_stats(animal)
            if mouse:
                data.append({
                    'animal': animal,
                    'cohort': mouse['cohort'],
                    'n_reaches': mouse['profile']['n_reaches'],
                    'success_rate': mouse['profile']['success_rate'],
                    'extent_mean': mouse['profile']['extent_mean'],
                    'velocity_mean': mouse['profile']['velocity_mean'],
                })
        return pd.DataFrame(data)


# =============================================================================
# BUILDER
# =============================================================================

def build_explorer_database(reach_df: pd.DataFrame, output_path: Path,
                            features_df: Optional[pd.DataFrame] = None) -> ReachExplorer:
    """
    Build the explorer database from reach data.

    Args:
        reach_df: DataFrame with columns: video_name, animal, segment_num,
                  reach_num_in_segment, outcome, max_extent_mm, duration_sec, etc.
        output_path: Path to save the SQLite database
        features_df: Optional DataFrame with additional kinematic features
                     (trajectory_straightness, peak_velocity, etc.)

    Returns:
        ReachExplorer instance connected to the new database
    """
    print(f"Building explorer database: {output_path}")

    # Merge features if provided
    if features_df is not None and len(features_df) > 0:
        # Merge on video_name + segment_num + reach_num
        merge_cols = ['video_name', 'segment_num']
        if 'reach_num' in features_df.columns:
            merge_cols.append('reach_num')
        if 'reach_num_in_segment' in reach_df.columns:
            reach_df = reach_df.rename(columns={'reach_num_in_segment': 'reach_num'})

        df = reach_df.merge(features_df, on=merge_cols, how='left', suffixes=('', '_feat'))
    else:
        df = reach_df.copy()

    # Ensure required columns
    if 'is_success' not in df.columns:
        df['is_success'] = df['outcome'] == 'retrieved'

    # Create database
    explorer = ReachExplorer(output_path)
    explorer.connect()
    explorer.initialize()

    # Store metadata
    explorer.set_metadata('build_date', pd.Timestamp.now().isoformat())
    explorer.set_metadata('n_reaches', str(len(df)))
    explorer.set_metadata('n_animals', str(df['animal'].nunique()))

    # -------------------------------------------------------------------------
    # Population Level
    # -------------------------------------------------------------------------
    print("  Computing population statistics...")
    pop_profile = KinematicProfile.from_dataframe(df)

    pop_comparisons = []
    for feat in ['max_extent_mm', 'duration_sec', 'peak_velocity_px_per_frame', 'trajectory_straightness']:
        comp = SuccessFailComparison.compute(df, feat)
        if comp:
            pop_comparisons.append(comp)

    # Temporal patterns (fatigue across all data)
    temporal = []
    if 'segment_num' in df.columns:
        for phase, (start, end) in [('early', (1, 7)), ('middle', (8, 14)), ('late', (15, 21))]:
            phase_df = df[(df['segment_num'] >= start) & (df['segment_num'] <= end)]
            if len(phase_df) > 0:
                temporal.append(TemporalPattern(
                    phase=phase,
                    n_reaches=len(phase_df),
                    success_rate=float(phase_df['is_success'].mean()),
                    extent_mean=float(phase_df['max_extent_mm'].mean()) if 'max_extent_mm' in phase_df.columns else 0,
                    velocity_mean=float(phase_df['peak_velocity_px_per_frame'].mean()) if 'peak_velocity_px_per_frame' in phase_df.columns else 0
                ))

    explorer.store_population_stats(pop_profile, pop_comparisons, temporal)

    # -------------------------------------------------------------------------
    # Mouse Level
    # -------------------------------------------------------------------------
    print("  Computing per-mouse statistics...")
    for animal in df['animal'].unique():
        mouse_df = df[df['animal'] == animal]

        profile = KinematicProfile.from_dataframe(mouse_df)

        comparisons = []
        for feat in ['max_extent_mm', 'duration_sec', 'peak_velocity_px_per_frame']:
            comp = SuccessFailComparison.compute(mouse_df, feat)
            if comp:
                comparisons.append(comp)

        # Learning curve (by session date)
        learning_curve = []
        if 'date' in mouse_df.columns:
            for date in sorted(mouse_df['date'].dropna().unique()):
                date_df = mouse_df[mouse_df['date'] == date]
                learning_curve.append({
                    'date': date,
                    'n_reaches': len(date_df),
                    'success_rate': float(date_df['is_success'].mean()),
                    'extent_mean': float(date_df['max_extent_mm'].mean()) if 'max_extent_mm' in date_df.columns else None
                })

        cohort = mouse_df['cohort'].iloc[0] if 'cohort' in mouse_df.columns else None
        dates = mouse_df['date'].dropna()
        first_session = str(dates.min()) if len(dates) > 0 else None
        last_session = str(dates.max()) if len(dates) > 0 else None

        explorer.store_mouse_stats(
            animal=animal,
            cohort=cohort,
            n_sessions=mouse_df['video_name'].nunique(),
            first_session=first_session,
            last_session=last_session,
            profile=profile,
            comparisons=comparisons,
            learning_curve=learning_curve
        )

    # -------------------------------------------------------------------------
    # Session Level
    # -------------------------------------------------------------------------
    print("  Computing per-session statistics...")
    for video_name in df['video_name'].unique():
        session_df = df[df['video_name'] == video_name]

        profile = KinematicProfile.from_dataframe(session_df)

        comparisons = []
        for feat in ['max_extent_mm', 'duration_sec']:
            comp = SuccessFailComparison.compute(session_df, feat)
            if comp:
                comparisons.append(comp)

        # Fatigue within session
        fatigue = []
        if 'segment_num' in session_df.columns:
            for phase, (start, end) in [('early', (1, 7)), ('middle', (8, 14)), ('late', (15, 21))]:
                phase_df = session_df[(session_df['segment_num'] >= start) & (session_df['segment_num'] <= end)]
                if len(phase_df) > 0:
                    fatigue.append(TemporalPattern(
                        phase=phase,
                        n_reaches=len(phase_df),
                        success_rate=float(phase_df['is_success'].mean()),
                        extent_mean=float(phase_df['max_extent_mm'].mean()) if 'max_extent_mm' in phase_df.columns else 0,
                        velocity_mean=0
                    ))

        animal = session_df['animal'].iloc[0] if 'animal' in session_df.columns else None
        date = session_df['date'].iloc[0] if 'date' in session_df.columns else None
        tray_type = session_df['tray_type'].iloc[0] if 'tray_type' in session_df.columns else None
        run_num = int(session_df['run_num'].iloc[0]) if 'run_num' in session_df.columns and pd.notna(session_df['run_num'].iloc[0]) else None

        explorer.store_session_stats(
            video_name=video_name,
            animal=animal,
            date=str(date) if date else None,
            tray_type=tray_type,
            run_num=run_num,
            profile=profile,
            comparisons=comparisons,
            fatigue=fatigue
        )

    # -------------------------------------------------------------------------
    # Individual Reaches
    # -------------------------------------------------------------------------
    print("  Storing individual reaches...")
    explorer.store_reaches(df.to_dict('records'))

    print(f"  Done! Database size: {output_path.stat().st_size / 1024:.1f} KB")

    return explorer
