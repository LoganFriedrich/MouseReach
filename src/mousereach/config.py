#!/usr/bin/env python3
"""
MouseReach Pipeline Configuration

Master configuration file with all paths, settings, and constants.
Paths are configurable via ~/.mousereach/config.json (preferred) or environment variables.

To configure: Run 'mousereach-setup' command
"""

import os
import json
from pathlib import Path
from typing import Optional, List


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def _load_config() -> dict:
    """Load configuration from JSON file."""
    config_file = Path.home() / ".mousereach" / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

_config = _load_config()

# =============================================================================
# DRIVE MAPPINGS - CONFIGURABLE VIA CONFIG FILE OR ENVIRONMENT VARIABLES
# =============================================================================
# NOTE: No hardcoded defaults! Users must run 'mousereach-setup' to configure.
# This ensures portability across different machines and labs.

# NAS (network storage) - raw videos and final outputs archive
# Priority: config.json > environment variable > None (requires setup)
_nas_drive = _config.get("nas_drive") or os.getenv("MouseReach_NAS_DRIVE")
NAS_DRIVE: Optional[Path] = Path(_nas_drive) if _nas_drive else None

# Processing root - location of all working pipeline folders
# Priority: config.json > environment variable > None (requires setup)
_processing_root = _config.get("processing_root") or os.getenv("MouseReach_PROCESSING_ROOT")
PROCESSING_ROOT: Optional[Path] = Path(_processing_root) if _processing_root else None


def require_processing_root() -> Path:
    """Get PROCESSING_ROOT, raising helpful error if not configured."""
    if PROCESSING_ROOT is None:
        raise ConfigurationError(
            "MouseReach is not configured.\n\n"
            "Run: mousereach-setup\n\n"
            "This will ask you where your pipeline folders are located."
        )
    return PROCESSING_ROOT


def require_nas_drive() -> Path:
    """Get NAS_DRIVE, raising helpful error if not configured."""
    if NAS_DRIVE is None:
        raise ConfigurationError(
            "NAS drive is not configured.\n\n"
            "Run: mousereach-setup\n\n"
            "This will ask you where your archive/NAS drive is located."
        )
    return NAS_DRIVE

# =============================================================================
# PIPELINE PATHS
# =============================================================================

class Paths:
    """All pipeline directory paths - derived from configurable environment variables.

    NOTE: Paths may be None if not configured. Use validate() to check, or
    use require_processing_root()/require_nas_drive() for safe access with errors.
    """

    # Root drives (from config/env - may be None if not configured)
    NAS_DRIVE = NAS_DRIVE
    PROCESSING_ROOT = PROCESSING_ROOT

    # --- NAS/Archive Paths (derived from MouseReach_NAS_DRIVE) ---
    # These will be None if NAS_DRIVE is not configured
    # nas_root config override allows pointing at a different folder structure
    # (e.g. Y:\2_Connectome\Behavior\MouseReach_Pipeline instead of X:\! DLC Output)
    _nas_root = _config.get("nas_root")
    NAS_ROOT = Path(_nas_root) if _nas_root else (NAS_DRIVE / "! DLC Output" if NAS_DRIVE else None)

    # Raw 8-camera collage videos (ARCHIVE - NEVER DELETE)
    MULTI_ANIMAL_SOURCE = NAS_ROOT / "Unanalyzed" / "Multi-Animal" if NAS_ROOT else None

    # Cropped single-animal videos from Step 0
    SINGLE_ANIMAL_OUTPUT = NAS_ROOT / "Unanalyzed" / "Single_Animal" if NAS_ROOT else None

    # Unsupported tray types (E/F) that got into pipeline by mistake
    # These require different algorithms and should be returned to NAS
    UNSUPPORTED_TRAY_RETURN = NAS_ROOT / "Unanalyzed" / "Unsupported_Tray_Type" if NAS_ROOT else None

    # DLC complete staging area (for processing PC to pick up)
    DLC_STAGING = NAS_ROOT / "DLC_Complete" if NAS_ROOT else None

    # Final validated outputs, organized by project/cohort (Step 6 destination)
    ANALYZED_OUTPUT = NAS_ROOT / "Analyzed" if NAS_ROOT else None

    # --- Processing Pipeline Paths (derived from MouseReach_PROCESSING_ROOT) ---
    # These will be None if PROCESSING_ROOT is not configured
    #
    # NEW ARCHITECTURE (v2.3+):
    #   - Single "Processing" folder for all post-DLC files
    #   - Review status determined by validation_status in JSON files, not folder location
    #   - Files stay co-located (video + DLC + segments + reaches + outcomes)
    #   - Archive only when ALL validation_status == "validated"

    # Step 1: DLC processing queue (videos waiting for GPU machine)
    DLC_QUEUE = PROCESSING_ROOT / "DLC_Queue" if PROCESSING_ROOT else None

    # Step 2+: Single processing folder for ALL post-DLC work
    # Files have validation_status in their JSON: "needs_review", "auto_approved", "validated"
    PROCESSING = PROCESSING_ROOT / "Processing" if PROCESSING_ROOT else None

    # Processing errors requiring investigation
    FAILED = PROCESSING_ROOT / "Failed" if PROCESSING_ROOT else None

    # Performance tracking logs (algorithm vs human comparison metrics)
    PERFORMANCE_LOGS = PROCESSING_ROOT / "performance_logs" if PROCESSING_ROOT else None

    # --- Legacy Aliases (for backwards compatibility) ---
    # These all point to PROCESSING now - status is in JSON metadata, not folder
    DLC_COMPLETE = PROCESSING  # After DLC, files go to Processing
    SEG_AUTO_REVIEW = PROCESSING
    SEG_NEEDS_REVIEW = PROCESSING
    SEG_VALIDATED = PROCESSING
    REACH_NEEDS_REVIEW = PROCESSING
    REACH_VALIDATED = PROCESSING
    OUTCOME_NEEDS_REVIEW = PROCESSING
    OUTCOME_VALIDATED = PROCESSING
    GROUND_TRUTH = PROCESSING  # GT files stored alongside video in Processing
    SCORE_NEEDS_REVIEW = PROCESSING
    SCORE_VALIDATED = PROCESSING
    STEP4_NEEDS_REVIEW = PROCESSING
    STEP4_VALIDATED = PROCESSING

    @classmethod
    def validate(cls) -> List[str]:
        """Check paths are configured correctly. Returns list of problems (empty = OK)."""
        problems = []

        if PROCESSING_ROOT is None:
            problems.append("PROCESSING_ROOT not configured - run 'mousereach-setup'")
        elif not PROCESSING_ROOT.exists():
            problems.append(f"PROCESSING_ROOT does not exist: {PROCESSING_ROOT}")

        if NAS_DRIVE is None:
            # NAS is optional for some workflows, so just note it
            problems.append("NAS_DRIVE not configured (optional for basic pipeline)")
        elif not NAS_DRIVE.exists():
            problems.append(f"NAS_DRIVE does not exist: {NAS_DRIVE}")

        return problems

    @classmethod
    def is_configured(cls) -> bool:
        """Check if minimum required paths are configured (PROCESSING_ROOT)."""
        return PROCESSING_ROOT is not None


# =============================================================================
# VIDEO CROPPING SETTINGS
# =============================================================================

class CropSettings:
    """Video cropping configuration."""
    
    # Collage dimensions (1920x1080)
    COLLAGE_WIDTH = 1920
    COLLAGE_HEIGHT = 1080
    
    # Grid layout: 4 columns x 2 rows
    GRID_COLS = 4
    GRID_ROWS = 2
    
    # Single cell dimensions
    CELL_WIDTH = 480   # 1920 / 4
    CELL_HEIGHT = 540  # 1080 / 2
    
    # Crop coordinates for each position (width, height, x_offset, y_offset)
    # Position mapping:
    #   1  2  3  4   (top row)
    #   5  6  7  8   (bottom row)
    CROP_COORDS = [
        (480, 540, 0, 0),      # Position 1: Top-left
        (480, 540, 480, 0),    # Position 2: Top-center-left
        (480, 540, 960, 0),    # Position 3: Top-center-right
        (480, 540, 1440, 0),   # Position 4: Top-right
        (480, 540, 0, 540),    # Position 5: Bottom-left
        (480, 540, 480, 540),  # Position 6: Bottom-center-left
        (480, 540, 960, 540),  # Position 7: Bottom-center-right
        (480, 540, 1440, 540)  # Position 8: Bottom-right
    ]


# =============================================================================
# DLC BODYPARTS
# =============================================================================

class DLCPoints:
    """DeepLabCut tracked bodyparts."""
    
    # All bodyparts tracked by the model
    ALL = [
        'Reference',
        'SATL', 'SABL', 'SABR', 'SATR',  # Scoring Area corners
        'BOXL', 'BOXR',                    # Box edges (slit)
        'Pellet', 'Pillar',                # Target objects
        'RightHand', 'RHLeft', 'RHOut', 'RHRight',  # Hand tracking
        'Nose', 'RightEar', 'LeftEar',     # Head
        'LeftFoot', 'TailBase'             # Body
    ]
    
    # Reference points (fixed, for calibration)
    REFERENCE = ['BOXL', 'BOXR', 'Reference']
    
    # Scoring area corners
    SCORING_AREA = ['SABL', 'SABR', 'SATL', 'SATR']
    
    # Primary SA anchors (most reliable)
    SA_ANCHORS = ['SABL', 'SABR']
    
    # Hand points for reach detection
    HAND = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
    
    # Primary hand point
    PRIMARY_HAND = 'RightHand'
    
    # Targets
    TARGETS = ['Pellet', 'Pillar']


# =============================================================================
# ANIMAL ID PARSING
# =============================================================================

class AnimalID:
    """Animal ID format: {letters}{cohort:2d}{subject:2d}
    
    Examples:
        CNT0101 = Control, cohort 01, subject 01
        ENCR0312 = ENCR experiment, cohort 03, subject 12
        CNT0001 = Blank (cohort 00 = skip)
    """
    
    # Known experiment prefixes
    EXPERIMENTS = ['CNT', 'ENCR', 'OPT', 'DREADD', 'LAT']
    
    # Cohort "00" means blank/skip position
    BLANK_COHORT = "00"
    
    @staticmethod
    def parse(animal_id: str) -> dict:
        """Parse animal ID into components."""
        # Find where letters end and numbers begin
        for i, c in enumerate(animal_id):
            if c.isdigit():
                letters = animal_id[:i]
                numbers = animal_id[i:]
                break
        else:
            return {'valid': False, 'error': 'No numbers found'}
        
        if len(numbers) < 4:
            return {'valid': False, 'error': 'Need at least 4 digits'}
        
        cohort = numbers[:2]
        subject = numbers[2:4]
        
        return {
            'valid': True,
            'experiment': letters,
            'cohort': cohort,
            'subject': subject,
            'is_blank': cohort == AnimalID.BLANK_COHORT,
            'full_id': animal_id
        }
    
    @staticmethod
    def is_blank(animal_id: str) -> bool:
        """Check if animal ID represents a blank position."""
        parsed = AnimalID.parse(animal_id)
        return parsed.get('is_blank', False)
    
    @staticmethod
    def get_experiment(animal_id: str) -> str:
        """Get experiment code from animal ID."""
        parsed = AnimalID.parse(animal_id)
        return parsed.get('experiment', 'UNKNOWN')

    # Project mapping: experiment code -> project folder name
    PROJECT_MAP = {
        'CNT': 'Connectome',
        'ENCR': 'Enhancer',
        # Old ASPA cohorts use single-letter or short prefixes
        'H': 'ASPA', 'I': 'ASPA', 'J': 'ASPA', 'K': 'ASPA',
        'L': 'ASPA', 'M': 'ASPA', 'N': 'ASPA',
        'G': 'ASPA', 'D': 'ASPA', 'F': 'ASPA',
        'ABS': 'ASPA',
        'Opt': 'ASPA',
    }

    @staticmethod
    def get_project_and_cohort(animal_id: str) -> tuple:
        """Get (project_folder, cohort_folder) for archive destination.

        Examples:
            CNT0304 -> ('Connectome', 'CNT03')
            ENCR0102 -> ('Enhancer', 'ENCR01')
            H01 -> ('ASPA', 'H')
            OptG05 -> ('ASPA', 'OptG')

        Returns:
            (project, cohort) tuple
        """
        parsed = AnimalID.parse(animal_id)
        experiment = parsed.get('experiment', '')

        if not experiment:
            # Fallback: extract letters manually for short IDs like H01
            for i, c in enumerate(animal_id):
                if c.isdigit():
                    experiment = animal_id[:i]
                    break
            if not experiment:
                return ('UNKNOWN', 'UNKNOWN')

        project = AnimalID.PROJECT_MAP.get(experiment, experiment)
        cohort_num = parsed.get('cohort', '')

        if parsed.get('valid') and experiment in ('CNT', 'ENCR'):
            cohort_folder = f"{experiment}{cohort_num}"
        else:
            cohort_folder = experiment

        return (project, cohort_folder)


# =============================================================================
# FILENAME CONVENTIONS
# =============================================================================

class FilePatterns:
    """Filename patterns and conventions."""

    # Multi-animal collage: 20250704_CNT0101,CNT0205,..._P1.mkv
    # Single-animal: 20250704_CNT0101_P1.mp4

    # Tray types
    TRAY_TYPES = {
        'P': 'Pillar',   # Full tracking
        'E': 'Easy',     # Limited tracking
        'F': 'Flat'      # Limited tracking
    }

    # Supported tray types (full DLC tracking available)
    SUPPORTED_TRAY_TYPES = ['P']

    # Unsupported tray types (require different algorithms)
    UNSUPPORTED_TRAY_TYPES = ['E', 'F']

    # NOTE: DLC files are auto-detected using glob patterns (*DLC*.h5)
    # Users must train their own DeepLabCut model for their specific setup.
    # See: https://deeplabcut.github.io/DeepLabCut/docs/intro.html

    # Pipeline output files
    SEGMENTS_SUFFIX = "_segments.json"
    SEG_VALIDATION_SUFFIX = "_seg_validation.json"
    REACHES_SUFFIX = "_reaches.json"
    REACHES_VALIDATION_SUFFIX = "_reaches_validation.json"
    OUTCOMES_SUFFIX = "_pellet_outcomes.json"
    OUTCOMES_VALIDATION_SUFFIX = "_outcomes_validation.json"
    DLC_QUALITY_SUFFIX = "_dlc_quality.json"

    # Ground truth files (human-verified for algorithm validation)
    SEG_GROUND_TRUTH_SUFFIX = "_seg_ground_truth.json"
    REACH_GROUND_TRUTH_SUFFIX = "_reach_ground_truth.json"
    OUTCOME_GROUND_TRUTH_SUFFIX = "_outcome_ground_truth.json"
    UNIFIED_GROUND_TRUTH_SUFFIX = "_unified_ground_truth.json"  # v2.4+ unified GT

    # Legacy ground truth (for backwards compatibility)
    LEGACY_GROUND_TRUTH_SUFFIX = "_ground_truth.json"  # ambiguous, avoid using


# =============================================================================
# ALGORITHM THRESHOLDS
# =============================================================================

class Thresholds:
    """Algorithm thresholds (all in ruler units unless noted)."""
    
    # DLC confidence thresholds
    DLC_HIGH_CONF = 0.9
    DLC_MIN_CONF = 0.6
    
    # Reference point stability (pixels)
    REF_STABLE_STD = 3.0   # Good
    REF_MAX_STD = 5.0      # Acceptable
    
    # Segmentation
    EXPECTED_BOUNDARIES = 21
    SEG_HIGH_CONF = 0.95   # Auto-review
    SEG_MIN_CONF = 0.85    # Needs review below this

    # Segmentation triage thresholds
    SEG_PERFECT_CV = 0.05         # Extremely consistent intervals
    SEG_GOOD_CV = 0.10             # Auto-approve threshold
    SEG_ACCEPTABLE_CV = 0.20       # Minor review threshold
    SEG_HIGH_CONFIDENCE = 0.85     # Mean confidence for auto-approval
    SEG_LOW_CONFIDENCE = 0.50      # Below this is critical
    SEG_PERFECT_COUNT = 21         # Expected boundaries
    SEG_ACCEPTABLE_MIN = 19        # Minimum to attempt segmentation
    SEG_PRIMARY_MIN = 18           # Minimum from primary method for auto-approval
    
    # Motion detection (ruler units)
    WIGGLE_NOISE = 0.02    # ~0.2mm, detection jitter
    STILL_THRESHOLD = 0.03 # ~0.3mm, "not moving"
    SIGNIFICANT_MOTION = 0.10  # ~0.9mm, meaningful
    SNAP_MAGNITUDE = 1.0   # ~9mm, pellet-to-pellet
    
    # Pellet position (ruler units)
    PELLET_ON_PILLAR = 0.25  # Max deviation when "home"

    # Pellet outcome detection (pixel distances)
    OUTCOME_SKIP_START_FRAMES = 20  # Skip first N frames for pellet analysis
    OUTCOME_SKIP_END_FRAMES = 30    # Skip last N frames for pellet analysis
    PAW_PROXIMITY_THRESHOLD = 30    # pixels - hand near pellet detection
    EATING_DISTANCE_THRESHOLD = 30  # pixels - hand-to-nose for eating detection
    STATIONARY_MOVEMENT_THRESHOLD = 3.0  # pixels - pellet stationary threshold
    TOWARD_BOX_THRESHOLD = 5.0      # pixels - pellet moving toward box
    TOWARD_SA_THRESHOLD = 5.0       # pixels - pellet moving toward scoring area
    PELLET_FALLING = 0.30    # Definitely off pillar


# =============================================================================
# PHYSICAL GEOMETRY
# =============================================================================

class Geometry:
    """Physical measurements from STL file (Pillar_Tray.stl)."""
    
    # The ruler: SABL to SABR distance
    RULER_MM = 9.0  # millimeters
    
    # Everything else in ruler units (multiply by 9mm for physical)
    SABR_TO_SATR_RULER = 1.667  # 15mm / 9mm
    PILLAR_DIAMETER_RULER = 0.458  # 4.125mm / 9mm
    PELLET_DIAMETER_RULER = 0.278  # ~2.5mm / 9mm
    PILLAR_TO_SA_RULER = 1.069  # 9.618mm / 9mm
    
    # Pillar geometry: forms 55° isoceles triangle with SABL/SABR
    PILLAR_APEX_ANGLE_DEG = 55.0
    PILLAR_PERP_DISTANCE_RULER = 0.944  # sqrt(1.069² - 0.5²)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_experiment_output_dir(animal_id: str) -> Path:
    """Get the final output directory for a video based on project/cohort.

    Examples:
        CNT0304 -> Analyzed/Connectome/CNT03/
        ENCR0102 -> Analyzed/Enhancer/ENCR01/
        H01 -> Analyzed/ASPA/H/
    """
    project, cohort = AnimalID.get_project_and_cohort(animal_id)
    return Paths.ANALYZED_OUTPUT / project / cohort


def parse_tray_type(filename: str) -> dict:
    """Parse tray type from video filename.

    Filename format: {date}_{animalID}_{tray}{position}.mp4
    Example: 20251021_CNT0401_P4.mp4 -> tray='P', position=4

    Returns:
        dict with keys:
            - tray_type: str ('P', 'E', 'F') or None if not found
            - tray_name: str ('Pillar', 'Easy', 'Flat') or 'Unknown'
            - position: int or None
            - is_supported: bool - True if tray type is supported by MouseReach
            - video_id: str - extracted video ID
    """
    import re

    video_id = get_video_id(filename)

    # Pattern: ends with tray type letter followed by position number
    # e.g., "20251021_CNT0401_P4" or "20251021_CNT0401_E2"
    match = re.search(r'_([PEF])(\d+)$', video_id)

    if not match:
        # Unknown format - assume supported for backwards compatibility
        # Only explicitly identified E (Easy) and F (Flat) trays are unsupported
        return {
            'tray_type': None,
            'tray_name': 'Unknown',
            'position': None,
            'is_supported': True,  # Default to supported for unknown formats
            'video_id': video_id,
        }

    tray_type = match.group(1)
    position = int(match.group(2))

    return {
        'tray_type': tray_type,
        'tray_name': FilePatterns.TRAY_TYPES.get(tray_type, 'Unknown'),
        'position': position,
        'is_supported': tray_type in FilePatterns.SUPPORTED_TRAY_TYPES,
        'video_id': video_id
    }


def is_supported_tray_type(filename: str) -> bool:
    """Check if filename represents a supported tray type (P=Pillar).

    Returns True for supported types (P), False for unsupported (E, F).
    """
    parsed = parse_tray_type(filename)
    return parsed['is_supported']


def get_video_id(filename: str) -> str:
    """Extract video ID from any filename in the pipeline.

    Video ID is everything before the extension or DLC suffix.
    Example: "20250704_CNT0101_P1" from any of:
        - 20250704_CNT0101_P1.mp4
        - 20250704_CNT0101_P1DLC_resnet50_....h5
        - 20250704_CNT0101_P1_segments.json
    """
    stem = Path(filename).stem
    
    # Remove DLC suffix if present
    if 'DLC_' in stem:
        stem = stem.split('DLC_')[0].rstrip('_')
    
    # Remove pipeline suffixes (order matters - longer suffixes first)
    for suffix in ['_seg_ground_truth', '_reach_ground_truth', '_outcome_ground_truth',
                   '_outcomes_ground_truth',  # legacy
                   '_seg_validation', '_reaches_validation', '_outcomes_validation',
                   '_pellet_outcomes', '_dlc_quality', '_ground_truth',
                   '_segments', '_reaches']:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem


# =============================================================================
# PRINT CONFIG SUMMARY
# =============================================================================

def print_config():
    """Print configuration summary."""
    print("=" * 60)
    print("MouseReach Pipeline Configuration")
    print("=" * 60)
    print()
    print("ENVIRONMENT CONFIGURATION:")
    nas_drive_env = os.getenv("MouseReach_NAS_DRIVE", "(not set, using default)")
    proc_root_env = os.getenv("MouseReach_PROCESSING_ROOT", "(not set, using default)")
    print(f"  MouseReach_NAS_DRIVE:       {nas_drive_env}")
    print(f"  MouseReach_PROCESSING_ROOT: {proc_root_env}")
    print()
    print("PIPELINE ARCHITECTURE (v2.3+):")
    print("  Single 'Processing' folder - status in JSON metadata, not folder location")
    print()
    print("RESOLVED PATHS:")
    print(f"  NAS Drive:           {NAS_DRIVE}")
    print(f"  Processing Root:     {PROCESSING_ROOT}")
    print(f"  DLC Queue:           {Paths.DLC_QUEUE}")
    print(f"  Processing:          {Paths.PROCESSING}")
    print(f"  Failed:              {Paths.FAILED}")
    print(f"  Analyzed Output:     {Paths.ANALYZED_OUTPUT}")
    print()
    print("VALIDATION STATUS (in JSON files):")
    print("  needs_review   - Requires human review")
    print("  auto_approved  - High confidence, auto-approved")
    print("  validated      - Human reviewed and approved")
    print()
    print("DLC BODYPARTS:")
    print(f"  Total: {len(DLCPoints.ALL)}")
    print(f"  SA Anchors: {DLCPoints.SA_ANCHORS}")
    print(f"  Hand: {DLCPoints.HAND}")
    print()
    print("VIDEO:")
    print(f"  Collage: {CropSettings.COLLAGE_WIDTH}x{CropSettings.COLLAGE_HEIGHT}")
    print(f"  Single:  {CropSettings.CELL_WIDTH}x{CropSettings.CELL_HEIGHT}")
    print()
    print("CONFIGURATION:")
    print("  To change paths, run: mousereach-setup")
    print()


# =============================================================================
# WATCHER CONFIGURATION
# =============================================================================

class WatcherConfig:
    """Configuration for the automated watcher pipeline.

    Loaded from the 'watcher' section of ~/.mousereach/config.json.
    All fields have sensible defaults — the watcher works out of the box
    after running mousereach-setup with watcher configuration.
    """

    def __init__(self, config_dict: dict = None):
        cfg = config_dict or {}
        self.enabled: bool = cfg.get('enabled', False)
        self.poll_interval_seconds: int = cfg.get('poll_interval_seconds', 30)
        self.stability_wait_seconds: int = cfg.get('stability_wait_seconds', 60)
        self.dlc_config_path: Optional[Path] = (
            Path(cfg['dlc_config_path']) if cfg.get('dlc_config_path') else None
        )
        self.dlc_gpu_device: int = cfg.get('dlc_gpu_device', 0)
        self.auto_archive_approved: bool = cfg.get('auto_archive_approved', False)
        self.quarantine_dir: Optional[Path] = (
            Path(cfg['quarantine_dir']) if cfg.get('quarantine_dir') else None
        )
        self.log_dir: Optional[Path] = (
            Path(cfg['log_dir']) if cfg.get('log_dir') else None
        )
        self.max_retries: int = cfg.get('max_retries', 3)
        self.mode: str = cfg.get('mode', 'dlc_pc')  # 'dlc_pc' or 'processing_server'
        self.max_local_pending: int = cfg.get('max_local_pending', 200)
        self.also_process: bool = cfg.get('also_process', False)  # DLC PCs also run seg/reach/outcomes
        self.db_path: Optional[Path] = (
            Path(cfg['db_path']) if cfg.get('db_path') else None
        )  # Local DB path to avoid SQLite-over-SMB issues
        self.staging_path: Optional[Path] = (
            Path(cfg['staging_path']) if cfg.get('staging_path') else None
        )  # Custom staging path for DLC output

    @classmethod
    def load(cls) -> 'WatcherConfig':
        """Load watcher config from ~/.mousereach/config.json."""
        config = _load_config()
        return cls(config.get('watcher', {}))

    def to_dict(self) -> dict:
        """Serialize to dict for saving to config.json."""
        d = {
            'enabled': self.enabled,
            'poll_interval_seconds': self.poll_interval_seconds,
            'stability_wait_seconds': self.stability_wait_seconds,
            'dlc_gpu_device': self.dlc_gpu_device,
            'auto_archive_approved': self.auto_archive_approved,
            'max_retries': self.max_retries,
            'mode': self.mode,
            'max_local_pending': self.max_local_pending,
            'also_process': self.also_process,
        }
        if self.dlc_config_path:
            d['dlc_config_path'] = str(self.dlc_config_path)
        if self.quarantine_dir:
            d['quarantine_dir'] = str(self.quarantine_dir)
        if self.log_dir:
            d['log_dir'] = str(self.log_dir)
        if self.db_path:
            d['db_path'] = str(self.db_path)
        if self.staging_path:
            d['staging_path'] = str(self.staging_path)
        return d

    def get_quarantine_dir(self) -> Path:
        """Get quarantine directory, defaulting to NAS_ROOT/Quarantine."""
        if self.quarantine_dir:
            return self.quarantine_dir
        if Paths.NAS_ROOT:
            return Paths.NAS_ROOT / "Quarantine"
        return require_processing_root() / "Quarantine"

    def get_log_dir(self) -> Path:
        """Get log directory, defaulting to PROCESSING_ROOT/watcher_logs."""
        if self.log_dir:
            return self.log_dir
        return require_processing_root() / "watcher_logs"


if __name__ == "__main__":
    print_config()
