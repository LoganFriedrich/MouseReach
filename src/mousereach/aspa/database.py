"""
mousereach.aspa.database - ASPA.db schema and connection helpers.

Database lives at Y:/2_Connectome/Behavior/MouseReach_Pipeline/ASPA.db
by default, overridable with ASPA_DB_PATH environment variable.
"""

import os
import sqlite3
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path("Y:/2_Connectome/Behavior/MouseReach_Pipeline/ASPA.db")


def get_db_path() -> Path:
    """Return path to ASPA.db.

    Priority:
        1. ASPA_DB_PATH environment variable
        2. Default: Y:/2_Connectome/Behavior/MouseReach_Pipeline/ASPA.db
    """
    env_path = os.environ.get("ASPA_DB_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_DB


def get_connection(db_path: Path = None) -> sqlite3.Connection:
    """Return a sqlite3 connection to ASPA.db.

    Args:
        db_path: Override path. If None, uses get_db_path().

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row for dict-like access.
    """
    if db_path is None:
        db_path = get_db_path()

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL_VIDEOS = """
CREATE TABLE IF NOT EXISTS videos (
    video_id              TEXT PRIMARY KEY,
    cohort                TEXT NOT NULL,
    animal_id             TEXT NOT NULL,
    session_date          TEXT,
    tray_type             TEXT,
    position              INTEGER,
    has_aspa_results      INTEGER NOT NULL DEFAULT 0,
    has_mousereach_results INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_ASPA_REACHES = """
CREATE TABLE IF NOT EXISTS aspa_reaches (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       TEXT NOT NULL REFERENCES videos(video_id),
    cohort         TEXT NOT NULL,
    animal_id      TEXT NOT NULL,
    reach_num      INTEGER,
    start_frame    INTEGER,
    end_frame      INTEGER,
    duration_s     REAL,
    pellet_num     INTEGER,
    outcome        TEXT,
    outcome_raw    TEXT,
    breadth_mm     REAL,
    reach_mm       REAL,
    distance_mm    REAL,
    speed_mm_s     REAL,
    area_mm2       REAL,
    pillar_visible INTEGER
);
"""

_DDL_MOUSEREACH_REACHES = """
CREATE TABLE IF NOT EXISTS mousereach_reaches (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id                TEXT NOT NULL REFERENCES videos(video_id),
    cohort                  TEXT NOT NULL,
    animal_id               TEXT NOT NULL,
    segment_num             INTEGER,
    reach_num               INTEGER,
    start_frame             INTEGER,
    end_frame               INTEGER,
    apex_frame              INTEGER,
    duration_frames         INTEGER,
    outcome                 TEXT,
    max_extent_mm           REAL,
    velocity_at_apex        REAL,
    trajectory_straightness REAL,
    mousereach_version      TEXT,
    dlc_scorer              TEXT,
    segmenter_version       TEXT,
    reach_detector_version  TEXT,
    outcome_detector_version TEXT,
    processed_by            TEXT
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_aspa_video ON aspa_reaches(video_id);",
    "CREATE INDEX IF NOT EXISTS idx_mr_video    ON mousereach_reaches(video_id);",
    "CREATE INDEX IF NOT EXISTS idx_videos_cohort ON videos(cohort);",
]


def ensure_tables(db_path: Path = None) -> None:
    """Create all ASPA.db tables and indexes if they do not exist.

    Safe to call repeatedly (uses IF NOT EXISTS).

    Args:
        db_path: Override path. If None, uses get_db_path().
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(_DDL_VIDEOS)
            conn.execute(_DDL_ASPA_REACHES)
            conn.execute(_DDL_MOUSEREACH_REACHES)
            for idx_sql in _INDEXES:
                conn.execute(idx_sql)
    finally:
        conn.close()
