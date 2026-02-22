#!/usr/bin/env python3
"""
Database Syncer for MouseReach Pipeline

Syncs _features.json (Step 5) outputs to the central connectome database
at Y:/2_Connectome/MouseDB/connectome.db as flattened per-reach records.

Each reach becomes one row in the reach_data table with:
- Session context (subject, date, tray type, run number)
- Reach identity (segment, reach ID, position)
- Outcome linkage (what this reach did to the pellet)
- 30+ kinematic features (velocity, trajectory, posture, etc.)
- Segment-level context (attention, pellet position)
- Tracking quality metrics

The syncer:
1. Scans Processing folder for _features.json files
2. Extracts subject_id and session metadata from video names
3. Flattens segments -> reaches into one row per reach
4. Inserts/replaces records in reach_data table
5. Exports flat CSV for Excel/R/pandas analysis
"""

import json
import hashlib
import re
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


# Central database location
DB_PATH = Path("Y:/2_Connectome/Databases/connectome.db")

# Only sync features files (Step 5 output with joined reach+outcome data)
FEATURES_SUFFIX = "_features.json"

# Local sync state file (tracks what's been synced)
SYNC_STATE_FILE = ".mousereach_sync_state.json"

# CSV dump location
CSV_DUMP_PATH = DB_PATH.parent / "database_dump" / "reach_data.csv"

# Columns extracted directly from each reach dict in features JSON
REACH_JSON_COLUMNS = [
    'reach_id', 'reach_num', 'segment_num',
    'outcome', 'causal_reach', 'interaction_frame', 'distance_to_interaction',
    'is_first_reach', 'is_last_reach', 'n_reaches_in_segment',
    'start_frame', 'apex_frame', 'end_frame', 'duration_frames',
    'max_extent_pixels', 'max_extent_ruler', 'max_extent_mm',
    'velocity_at_apex_px_per_frame', 'velocity_at_apex_mm_per_sec',
    'peak_velocity_px_per_frame', 'mean_velocity_px_per_frame',
    'trajectory_straightness', 'trajectory_smoothness',
    'hand_angle_at_apex_deg', 'hand_rotation_total_deg',
    'grasp_aperture_max_mm', 'grasp_aperture_at_contact_mm',
    'head_width_at_apex_mm', 'nose_to_slit_at_apex_mm',
    'head_angle_at_apex_deg', 'head_angle_change_deg',
    'apex_distance_to_pellet_mm', 'lateral_deviation_mm',
    'mean_likelihood', 'frames_low_confidence', 'tracking_quality_score',
    'flagged_for_review', 'flag_reason',
]

# Boolean fields that need int conversion for SQLite
BOOL_COLUMNS = {'causal_reach', 'is_first_reach', 'is_last_reach', 'flagged_for_review'}

# All columns in reach_data table (for INSERT)
ALL_COLUMNS = (
    ['subject_id', 'video_name', 'session_date', 'tray_type', 'run_number']
    + REACH_JSON_COLUMNS
    + ['segment_outcome', 'segment_outcome_confidence', 'segment_outcome_flagged',
       'attention_score', 'pellet_position_idealness']
    + ['source_file', 'extractor_version', 'imported_at']
    + ['processed_by', 'mousereach_version', 'dlc_scorer', 'segmenter_version',
       'reach_detector_version', 'outcome_detector_version']
)

# SQL to create reach_data table
CREATE_REACH_DATA_SQL = """
CREATE TABLE IF NOT EXISTS reach_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Session identity (parsed from video name: YYYYMMDD_CNTxxxx_TypeRun)
    subject_id TEXT NOT NULL REFERENCES subjects(subject_id),
    video_name TEXT NOT NULL,
    session_date TEXT NOT NULL,
    tray_type TEXT,
    run_number INTEGER,

    -- Reach identity
    segment_num INTEGER NOT NULL,
    reach_id INTEGER NOT NULL,
    reach_num INTEGER NOT NULL,

    -- Outcome linkage
    outcome TEXT,
    causal_reach INTEGER NOT NULL DEFAULT 0,
    interaction_frame INTEGER,
    distance_to_interaction INTEGER,

    -- Reach context
    is_first_reach INTEGER NOT NULL DEFAULT 0,
    is_last_reach INTEGER NOT NULL DEFAULT 0,
    n_reaches_in_segment INTEGER NOT NULL DEFAULT 0,

    -- Temporal features
    start_frame INTEGER NOT NULL,
    apex_frame INTEGER,
    end_frame INTEGER NOT NULL,
    duration_frames INTEGER NOT NULL,

    -- Extent features
    max_extent_pixels REAL,
    max_extent_ruler REAL,
    max_extent_mm REAL,

    -- Velocity features
    velocity_at_apex_px_per_frame REAL,
    velocity_at_apex_mm_per_sec REAL,
    peak_velocity_px_per_frame REAL,
    mean_velocity_px_per_frame REAL,

    -- Trajectory features
    trajectory_straightness REAL,
    trajectory_smoothness REAL,

    -- Hand orientation
    hand_angle_at_apex_deg REAL,
    hand_rotation_total_deg REAL,

    -- Grasp aperture
    grasp_aperture_max_mm REAL,
    grasp_aperture_at_contact_mm REAL,

    -- Body/posture at apex
    head_width_at_apex_mm REAL,
    nose_to_slit_at_apex_mm REAL,
    head_angle_at_apex_deg REAL,
    head_angle_change_deg REAL,

    -- Spatial context
    apex_distance_to_pellet_mm REAL,
    lateral_deviation_mm REAL,

    -- Tracking quality
    mean_likelihood REAL,
    frames_low_confidence INTEGER DEFAULT 0,
    tracking_quality_score REAL,

    -- Flags
    flagged_for_review INTEGER NOT NULL DEFAULT 0,
    flag_reason TEXT,

    -- Segment-level context (denormalized)
    segment_outcome TEXT,
    segment_outcome_confidence REAL,
    segment_outcome_flagged INTEGER DEFAULT 0,
    attention_score REAL,
    pellet_position_idealness REAL,

    -- Metadata
    source_file TEXT NOT NULL,
    extractor_version TEXT,
    imported_at TEXT NOT NULL,

    -- Provenance (from _processing_manifest.json)
    processed_by TEXT,
    mousereach_version TEXT,
    dlc_scorer TEXT,
    segmenter_version TEXT,
    reach_detector_version TEXT,
    outcome_detector_version TEXT,

    -- One row per reach per video
    UNIQUE(video_name, reach_id)
);
"""

# Legacy patterns (kept for backward compat with sync_file_to_database)
SYNC_PATTERNS = {
    "reaches": "_reaches.json",
    "outcomes": "_pellet_outcomes.json",
    "features": "_features.json",
}


@dataclass
class SyncResult:
    """Result of a sync operation."""
    synced: int = 0
    skipped: int = 0
    errors: int = 0
    reaches_inserted: int = 0
    error_messages: List[str] = None

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


def parse_subject_id(video_name: str) -> Optional[str]:
    """
    Extract subject ID from video name and convert to database format.

    Video format: YYYYMMDD_CNTxxxx_[Type][Run]
    Example: 20250624_CNT0115_P2 -> CNT0115 -> CNT_01_15

    Database format: CNT_CC_SS (project_cohort_subject)
    - CC = cohort (2 digits)
    - SS = subject within cohort (2 digits)
    """
    clean_name = video_name
    for suffix in ['_features', '_reaches', '_pellet_outcomes', '_segments']:
        clean_name = clean_name.replace(suffix, '')

    match = re.search(r'CNT(\d{4})', clean_name)
    if match:
        digits = match.group(1)
        cohort = digits[:2]
        subject = digits[2:]
        return f"CNT_{cohort}_{subject}"

    match = re.search(r'(CNT_\d{2}_\d{2})', clean_name)
    if match:
        return match.group(1)

    return None


def parse_video_metadata(video_name: str) -> Dict[str, Any]:
    """
    Extract session metadata from video name.

    Video format: YYYYMMDD_CNTxxxx_TypeRun
    Example: 20250624_CNT0115_P2

    Returns:
        Dict with session_date, tray_type, run_number (any may be None)
    """
    result = {'session_date': None, 'tray_type': None, 'run_number': None}

    clean_name = video_name
    for suffix in ['_features', '_reaches', '_pellet_outcomes', '_segments']:
        clean_name = clean_name.replace(suffix, '')

    # Extract date from start: YYYYMMDD
    date_match = re.match(r'(\d{8})_', clean_name)
    if date_match:
        d = date_match.group(1)
        try:
            result['session_date'] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        except Exception:
            pass

    # Extract tray type and run number from last part: P2, E1, F3, etc.
    # Pattern: after CNTxxxx_, find letter(s) followed by digit(s)
    tray_match = re.search(r'CNT\d{4}_([A-Za-z]+)(\d+)$', clean_name)
    if tray_match:
        result['tray_type'] = tray_match.group(1).upper()
        try:
            result['run_number'] = int(tray_match.group(2))
        except ValueError:
            pass

    return result


def file_hash(path: Path) -> str:
    """Get SHA256 hash of file for change detection."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


class DatabaseSyncer:
    """
    Syncs MouseReach _features.json files to reach_data table.

    Each features file is flattened into one row per reach with outcome
    linkage, kinematic features, and session context.

    Usage:
        syncer = DatabaseSyncer()
        result = syncer.sync_all()
        print(f"Synced {result.synced} files ({result.reaches_inserted} reaches)")
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        processing_path: Optional[Path] = None,
        dry_run: bool = False
    ):
        self.db_path = db_path or DB_PATH
        self.dry_run = dry_run
        self._engine = None
        self._table_ensured = False

        if processing_path:
            self.processing_path = processing_path
        else:
            try:
                from mousereach.config import Paths
                self.processing_path = Paths.PROCESSING
            except Exception:
                self.processing_path = None

        self._sync_state = self._load_sync_state()

    def _load_sync_state(self) -> Dict[str, str]:
        """Load sync state from local file (tracks file hashes)."""
        if self.processing_path is None:
            return {}
        state_file = self.processing_path / SYNC_STATE_FILE
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_sync_state(self):
        """Save sync state to local file."""
        if self.processing_path is None or self.dry_run:
            return
        state_file = self.processing_path / SYNC_STATE_FILE
        try:
            with open(state_file, 'w') as f:
                json.dump(self._sync_state, f, indent=2)
        except Exception:
            pass

    @property
    def engine(self):
        """Get SQLAlchemy engine (lazy initialization)."""
        if self._engine is None:
            if not HAS_SQLALCHEMY:
                raise ImportError("sqlalchemy is required for database sync")
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")
            self._engine = create_engine(f"sqlite:///{self.db_path}")
        return self._engine

    def ensure_reach_data_table(self):
        """Create reach_data table if it doesn't exist, then run migrations."""
        if self._table_ensured:
            return
        try:
            with self.engine.connect() as conn:
                conn.execute(text(CREATE_REACH_DATA_SQL))
                conn.commit()
            self._migrate_reach_data()
            self._table_ensured = True
        except Exception as e:
            raise RuntimeError(f"Failed to create reach_data table: {e}")

    def _migrate_reach_data(self):
        """Add columns that may not exist in older databases."""
        new_columns = [
            ('processed_by', 'TEXT'),
            ('mousereach_version', 'TEXT'),
            ('dlc_scorer', 'TEXT'),
            ('segmenter_version', 'TEXT'),
            ('reach_detector_version', 'TEXT'),
            ('outcome_detector_version', 'TEXT'),
        ]
        try:
            with self.engine.connect() as conn:
                for column, col_type in new_columns:
                    try:
                        conn.execute(text(
                            f"ALTER TABLE reach_data ADD COLUMN {column} {col_type}"
                        ))
                    except Exception:
                        pass  # Column already exists
                conn.commit()
        except Exception:
            pass  # Migration failure shouldn't block sync

    @staticmethod
    def _load_provenance(processing_dir: Path, video_name: str) -> dict:
        """
        Load provenance info from processing manifest.

        Looks for {video_name}_processing_manifest.json in the same directory
        as the features file.

        Returns:
            Dict with provenance columns (empty strings for missing values)
        """
        defaults = {
            'processed_by': None,
            'mousereach_version': None,
            'dlc_scorer': None,
            'segmenter_version': None,
            'reach_detector_version': None,
            'outcome_detector_version': None,
        }

        manifest_path = processing_dir / f"{video_name}_processing_manifest.json"
        if not manifest_path.exists():
            return defaults

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            defaults['processed_by'] = manifest.get('processed_by')
            defaults['mousereach_version'] = (
                manifest.get('pipeline_versions', {}).get('mousereach')
            )
            defaults['dlc_scorer'] = (
                manifest.get('dlc_model', {}).get('dlc_scorer')
            )
            defaults['segmenter_version'] = (
                manifest.get('pipeline_versions', {}).get('segmenter')
            )
            defaults['reach_detector_version'] = (
                manifest.get('pipeline_versions', {}).get('reach_detector')
            )
            defaults['outcome_detector_version'] = (
                manifest.get('pipeline_versions', {}).get('outcome_detector')
            )

            return defaults

        except Exception:
            return defaults

    def check_database(self) -> Tuple[bool, str]:
        """
        Check database connection and schema.

        Returns:
            (success, message) tuple
        """
        try:
            with self.engine.connect() as conn:
                # Check subjects table exists (for foreign key)
                result = conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='subjects'"
                ))
                if result.fetchone() is None:
                    return False, "subjects table not found"

                # Ensure reach_data table exists
                self.ensure_reach_data_table()

                return True, "Database connection OK"
        except Exception as e:
            return False, f"Database error: {e}"

    def get_known_subjects(self) -> List[str]:
        """Get list of known subject IDs from database."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT subject_id FROM subjects"))
                return [row[0] for row in result.fetchall()]
        except Exception:
            return []

    def find_syncable_files(self) -> List[Tuple[Path, str]]:
        """
        Find all _features.json files in Processing that can be synced.

        Returns:
            List of (path, subject_id) tuples
        """
        if self.processing_path is None or not self.processing_path.exists():
            return []

        syncable = []
        known_subjects = set(self.get_known_subjects())

        for json_file in self.processing_path.glob(f"*{FEATURES_SUFFIX}"):
            video_name = json_file.stem.replace('_features', '')
            subject_id = parse_subject_id(video_name)

            if subject_id and subject_id in known_subjects:
                syncable.append((json_file, subject_id))

        return syncable

    def needs_sync(self, path: Path) -> bool:
        """Check if a file needs to be synced (new or changed)."""
        key = str(path.relative_to(self.processing_path) if self.processing_path else path)
        current_hash = file_hash(path)

        if key not in self._sync_state:
            return True

        return self._sync_state[key] != current_hash

    def sync_features_file(self, path: Path, subject_id: str) -> int:
        """
        Sync a _features.json file to reach_data table.

        Flattens the nested JSON into one row per reach, with session context,
        segment context, and all kinematic features.

        Args:
            path: Path to _features.json file
            subject_id: Subject ID in database format (CNT_01_15)

        Returns:
            Number of reaches inserted

        Raises:
            RuntimeError: If sync fails
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            video_name = data.get('video_name', path.stem.replace('_features', ''))
            extractor_version = data.get('extractor_version', 'unknown')
            now = datetime.now().isoformat()

            # Load provenance from processing manifest (sibling file)
            provenance = self._load_provenance(path.parent, video_name)

            # Parse session metadata from video name
            meta = parse_video_metadata(video_name)

            # Build rows from segments -> reaches
            rows = []
            for segment in data.get('segments', []):
                # Segment-level context
                seg_context = {
                    'segment_outcome': segment.get('outcome'),
                    'segment_outcome_confidence': segment.get('outcome_confidence'),
                    'segment_outcome_flagged': 1 if segment.get('outcome_flagged') else 0,
                    'attention_score': segment.get('attention_score'),
                    'pellet_position_idealness': segment.get('pellet_position_idealness'),
                }

                for reach in segment.get('reaches', []):
                    row = {}

                    # Session identity
                    row['subject_id'] = subject_id
                    row['video_name'] = video_name
                    row['session_date'] = meta['session_date'] or ''
                    row['tray_type'] = meta['tray_type']
                    row['run_number'] = meta['run_number']

                    # Reach fields from JSON
                    for col in REACH_JSON_COLUMNS:
                        val = reach.get(col)
                        if col in BOOL_COLUMNS and val is not None:
                            val = 1 if val else 0
                        row[col] = val

                    # Segment context
                    row.update(seg_context)

                    # Metadata
                    row['source_file'] = path.name
                    row['extractor_version'] = extractor_version
                    row['imported_at'] = now

                    # Provenance
                    row.update(provenance)

                    rows.append(row)

            if not rows:
                return 0

            if self.dry_run:
                return len(rows)

            # Atomic replace: delete old rows for this video, insert new ones
            self.ensure_reach_data_table()

            col_names = ', '.join(ALL_COLUMNS)
            placeholders = ', '.join(f':{col}' for col in ALL_COLUMNS)
            insert_sql = f"INSERT INTO reach_data ({col_names}) VALUES ({placeholders})"

            with self.engine.connect() as conn:
                # Delete existing rows for this video
                conn.execute(
                    text("DELETE FROM reach_data WHERE video_name = :video_name"),
                    {"video_name": video_name}
                )

                # Insert all new rows
                for row in rows:
                    conn.execute(text(insert_sql), row)

                conn.commit()

            # Update sync state
            key = str(path.relative_to(self.processing_path) if self.processing_path else path)
            self._sync_state[key] = file_hash(path)

            return len(rows)

        except Exception as e:
            raise RuntimeError(f"Failed to sync {path.name}: {e}")

    def sync_all(self, force: bool = False) -> SyncResult:
        """
        Sync all _features.json files to reach_data table.

        Args:
            force: If True, sync all files even if unchanged

        Returns:
            SyncResult with counts
        """
        result = SyncResult()

        ok, msg = self.check_database()
        if not ok:
            result.errors = 1
            result.error_messages.append(msg)
            return result

        syncable = self.find_syncable_files()

        for path, subject_id in syncable:
            if not force and not self.needs_sync(path):
                result.skipped += 1
                continue

            try:
                n_reaches = self.sync_features_file(path, subject_id)
                result.synced += 1
                result.reaches_inserted += n_reaches
            except Exception as e:
                result.errors += 1
                result.error_messages.append(str(e))

        self._save_sync_state()

        if result.synced > 0:
            self.export_csv()

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get sync status information."""
        status = {
            "database_path": str(self.db_path),
            "processing_path": str(self.processing_path) if self.processing_path else None,
            "database_ok": False,
            "syncable_files": 0,
            "synced_files": 0,
            "pending_files": 0,
        }

        ok, msg = self.check_database()
        status["database_ok"] = ok
        status["database_message"] = msg

        if not ok:
            return status

        syncable = self.find_syncable_files()
        status["syncable_files"] = len(syncable)

        pending = 0
        for path, _ in syncable:
            if self.needs_sync(path):
                pending += 1

        status["synced_files"] = len(syncable) - pending
        status["pending_files"] = pending

        # Count reach_data records
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM reach_data"))
                status["total_reaches"] = result.fetchone()[0]

                result = conn.execute(text(
                    "SELECT COUNT(DISTINCT video_name) FROM reach_data"
                ))
                status["total_videos"] = result.fetchone()[0]

                result = conn.execute(text(
                    "SELECT COUNT(DISTINCT subject_id) FROM reach_data"
                ))
                status["total_subjects"] = result.fetchone()[0]

                result = conn.execute(text(
                    "SELECT COUNT(*) FROM reach_data WHERE causal_reach = 1"
                ))
                status["causal_reaches"] = result.fetchone()[0]
        except Exception:
            status["total_reaches"] = 0
            status["total_videos"] = 0
            status["total_subjects"] = 0
            status["causal_reaches"] = 0

        return status

    def export_csv(self):
        """
        Export reach_data table to flat CSV for Excel/R/pandas.

        Writes to Y:/2_Connectome/Unified_Data/database_dump/reach_data.csv
        One row per reach with all columns - directly usable for analysis.
        """
        try:
            with self.engine.connect() as conn:
                # Get all columns except id
                result = conn.execute(text("""
                    SELECT subject_id, video_name, session_date, tray_type, run_number,
                           segment_num, reach_id, reach_num,
                           outcome, causal_reach, interaction_frame, distance_to_interaction,
                           is_first_reach, is_last_reach, n_reaches_in_segment,
                           start_frame, apex_frame, end_frame, duration_frames,
                           max_extent_pixels, max_extent_ruler, max_extent_mm,
                           velocity_at_apex_px_per_frame, velocity_at_apex_mm_per_sec,
                           peak_velocity_px_per_frame, mean_velocity_px_per_frame,
                           trajectory_straightness, trajectory_smoothness,
                           hand_angle_at_apex_deg, hand_rotation_total_deg,
                           grasp_aperture_max_mm, grasp_aperture_at_contact_mm,
                           head_width_at_apex_mm, nose_to_slit_at_apex_mm,
                           head_angle_at_apex_deg, head_angle_change_deg,
                           apex_distance_to_pellet_mm, lateral_deviation_mm,
                           mean_likelihood, frames_low_confidence, tracking_quality_score,
                           flagged_for_review, flag_reason,
                           segment_outcome, segment_outcome_confidence, segment_outcome_flagged,
                           attention_score, pellet_position_idealness,
                           source_file, extractor_version, imported_at,
                           processed_by, mousereach_version, dlc_scorer,
                           segmenter_version, reach_detector_version,
                           outcome_detector_version
                    FROM reach_data
                    ORDER BY subject_id, session_date, video_name, segment_num, reach_num
                """))
                rows = result.fetchall()

            CSV_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(CSV_DUMP_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(ALL_COLUMNS)
                for row in rows:
                    writer.writerow(list(row))

        except Exception:
            pass  # Don't break sync if CSV export fails


def sync_file_to_database(output_path: Path) -> bool:
    """
    Sync a single pipeline output file to the central database.

    Call this after saving a _features.json file. Only features files
    are synced (they contain the joined reach+outcome data).

    Silently does nothing if the database is unavailable, the subject
    isn't in the database, or the file isn't a _features.json.

    Args:
        output_path: Path to the JSON file just saved

    Returns:
        True if synced, False if skipped/failed (never raises)
    """
    try:
        path = Path(output_path)

        # Only sync features files
        if not path.name.endswith(FEATURES_SUFFIX):
            return False

        # Parse subject ID
        video_name = path.stem.replace('_features', '')
        subject_id = parse_subject_id(video_name)
        if subject_id is None:
            return False

        # Sync
        syncer = DatabaseSyncer()
        ok, _ = syncer.check_database()
        if not ok:
            return False

        # Check subject exists in database
        known = syncer.get_known_subjects()
        if subject_id not in known:
            return False

        syncer.sync_features_file(path, subject_id)
        syncer._save_sync_state()
        syncer.export_csv()
        return True

    except Exception:
        return False  # Never break the pipeline
