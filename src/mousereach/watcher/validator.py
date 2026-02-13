#!/usr/bin/env python3
"""
Filename validation module for MouseReach watcher.

Validates video filenames before processing and quarantines anything malformed.
Supports both collage filenames (8-camera) and single-animal cropped videos.

Validation Rules:
- Collage: YYYYMMDD_{ID1,ID2,...ID8}_{TRAY}{RUN}.{ext}
- Single: YYYYMMDD_{ID}_{TRAY}{RUN}.mp4

All rules are strictly enforced - any deviation results in quarantine.
"""

import re
import json
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger(__name__)

# Supported video extensions
VIDEO_EXTENSIONS = {'.mkv', '.avi', '.mp4', '.mov', '.wmv'}

# Try to import AnimalID.parse from config, fallback to standalone implementation
try:
    from mousereach.config import AnimalID
    _has_config = True
except ImportError:
    _has_config = False
    logger.debug("mousereach.config not available, using standalone animal ID parser")


@dataclass
class ValidationResult:
    """Result of filename validation."""
    valid: bool
    filename: str
    error: Optional[str] = None
    warning: Optional[str] = None
    parsed: Optional[dict] = None  # Parsed metadata from filename


# =============================================================================
# ANIMAL ID PARSING (Standalone Fallback)
# =============================================================================

def _parse_animal_id_standalone(animal_id: str) -> dict:
    """Standalone animal ID parser (used when mousereach.config unavailable).

    Format: {LETTERS}{COHORT:2d}{SUBJECT:2d}
    Examples: CNT0101, ENCR0312, Opt0001
    """
    if not animal_id:
        return {'valid': False, 'error': 'Empty animal ID'}

    # Find where letters end and numbers begin
    match = re.match(r'^([A-Za-z]+)(\d{2,})$', animal_id)
    if not match:
        return {'valid': False, 'error': 'Invalid format - expected letters followed by digits'}

    letters = match.group(1)
    numbers = match.group(2)

    if len(numbers) < 4:
        return {'valid': False, 'error': 'Need at least 4 digits (cohort + subject)'}

    cohort = numbers[:2]
    subject = numbers[2:4]

    return {
        'valid': True,
        'experiment': letters,
        'cohort': cohort,
        'subject': subject,
        'is_blank': cohort == '00',
        'full_id': animal_id
    }


def _parse_animal_id(animal_id: str) -> dict:
    """Parse animal ID using config if available, otherwise standalone."""
    if _has_config:
        return AnimalID.parse(animal_id)
    else:
        return _parse_animal_id_standalone(animal_id)


def _is_blank_animal(animal_id: str) -> bool:
    """Check if animal ID is blank (cohort 00)."""
    parsed = _parse_animal_id(animal_id)
    return parsed.get('is_blank', False)


# =============================================================================
# FILENAME VALIDATION
# =============================================================================

def validate_collage_filename(filename: str) -> ValidationResult:
    """Validate multi-animal collage filename.

    Expected format: YYYYMMDD_{ID1,ID2,...,ID8}_{TRAY}{RUN}.ext
    Example: 20250704_CNT0101,CNT0205,CNT0305,CNT0306,CNT0102,CNT0605,CNT0309,CNT0906_P1.mkv

    Rules (ALL must pass):
    - Extension is in VIDEO_EXTENSIONS
    - Matches pattern: YYYYMMDD_{IDs}_{TRAY}{RUN}.ext
    - Split by underscore: parts[0] = date (8 digits, valid date, not in future)
    - parts[1] = comma-separated animal IDs
    - parts[2:] joined = tray suffix
    - Exactly 8 comma-separated animal IDs in parts[1]
    - Each animal ID parseable (letters followed by digits)
    - Tray suffix matches [PEF]\\d+ pattern
    - At least one non-blank animal (not cohort "00")

    Args:
        filename: Filename to validate (with or without path)

    Returns:
        ValidationResult with parsed metadata on success
    """
    filename = Path(filename).name  # Strip path if present

    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Invalid extension '{ext}' - expected one of {VIDEO_EXTENSIONS}"
        )

    # Parse filename parts
    stem = Path(filename).stem
    parts = stem.split('_')

    if len(parts) < 3:
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Expected at least 3 underscore-separated parts, got {len(parts)}"
        )

    # Part 0: Date (YYYYMMDD)
    date_str = parts[0]
    if not re.match(r'^\d{8}$', date_str):
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Date '{date_str}' must be 8 digits (YYYYMMDD)"
        )

    # Validate date is real and not in future
    try:
        date = datetime.strptime(date_str, '%Y%m%d')
        if date > datetime.now():
            return ValidationResult(
                valid=False,
                filename=filename,
                error=f"Date {date_str} is in the future"
            )
    except ValueError as e:
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Invalid date {date_str}: {e}"
        )

    # Part 1: Animal IDs (comma-separated)
    animal_ids_str = parts[1]
    animal_ids = animal_ids_str.split(',')

    if len(animal_ids) != 8:
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Expected exactly 8 comma-separated animal IDs, got {len(animal_ids)}"
        )

    # Validate each animal ID
    parsed_animals = []
    for i, animal_id in enumerate(animal_ids, start=1):
        parsed = _parse_animal_id(animal_id)
        if not parsed.get('valid', False):
            return ValidationResult(
                valid=False,
                filename=filename,
                error=f"Animal ID #{i} '{animal_id}': {parsed.get('error', 'parse failed')}"
            )
        parsed_animals.append(parsed)

    # Part 2+: Tray suffix (P1, E2, F1, etc.)
    tray_suffix = '_'.join(parts[2:])
    tray_match = re.match(r'^([PEF])(\d+)$', tray_suffix)
    if not tray_match:
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Tray suffix '{tray_suffix}' must match pattern [PEF]<digits> (e.g., P1, E2, F1)"
        )

    tray_type = tray_match.group(1)
    tray_run = int(tray_match.group(2))

    # Check for at least one non-blank animal
    n_blanks = sum(1 for p in parsed_animals if p['is_blank'])
    if n_blanks == 8:
        return ValidationResult(
            valid=False,
            filename=filename,
            error="All 8 positions are blank (cohort 00) - no real animals"
        )

    # Success - return parsed metadata
    return ValidationResult(
        valid=True,
        filename=filename,
        parsed={
            'date': date_str,
            'animal_ids': animal_ids,
            'tray_type': tray_type,
            'tray_run': tray_run,
            'n_blanks': n_blanks,
            'experiments': list(set(p['experiment'] for p in parsed_animals if not p['is_blank']))
        }
    )


def validate_single_filename(filename: str) -> ValidationResult:
    """Validate single-animal cropped video filename.

    Expected format: YYYYMMDD_{ID}_{TRAY}{RUN}.mp4
    Example: 20250704_CNT0101_P1.mp4

    Rules:
    - Extension is .mp4 (cropped singles are always mp4)
    - Matches pattern: YYYYMMDD_{ID}_{TRAY}{RUN}.mp4
    - Date valid (8 digits, real date, not in future)
    - Single animal ID parseable
    - Tray suffix valid ([PEF]\\d+)

    Args:
        filename: Filename to validate (with or without path)

    Returns:
        ValidationResult with parsed metadata on success
    """
    filename = Path(filename).name  # Strip path if present

    # Check extension (singles are always .mp4)
    ext = Path(filename).suffix.lower()
    if ext != '.mp4':
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Single-animal videos must be .mp4, got '{ext}'"
        )

    # Parse filename parts
    stem = Path(filename).stem
    parts = stem.split('_')

    if len(parts) < 3:
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Expected at least 3 underscore-separated parts, got {len(parts)}"
        )

    # Part 0: Date (YYYYMMDD)
    date_str = parts[0]
    if not re.match(r'^\d{8}$', date_str):
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Date '{date_str}' must be 8 digits (YYYYMMDD)"
        )

    # Validate date is real and not in future
    try:
        date = datetime.strptime(date_str, '%Y%m%d')
        if date > datetime.now():
            return ValidationResult(
                valid=False,
                filename=filename,
                error=f"Date {date_str} is in the future"
            )
    except ValueError as e:
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Invalid date {date_str}: {e}"
        )

    # Part 1: Animal ID
    animal_id = parts[1]
    parsed = _parse_animal_id(animal_id)
    if not parsed.get('valid', False):
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Animal ID '{animal_id}': {parsed.get('error', 'parse failed')}"
        )

    # Warn if blank animal (unusual for single videos)
    warning = None
    if parsed['is_blank']:
        warning = f"Animal ID '{animal_id}' is blank (cohort 00) - unusual for cropped video"

    # Part 2+: Tray suffix (P1, E2, F1, etc.)
    tray_suffix = '_'.join(parts[2:])
    tray_match = re.match(r'^([PEF])(\d+)$', tray_suffix)
    if not tray_match:
        return ValidationResult(
            valid=False,
            filename=filename,
            error=f"Tray suffix '{tray_suffix}' must match pattern [PEF]<digits> (e.g., P1, E2, F1)"
        )

    tray_type = tray_match.group(1)
    tray_run = int(tray_match.group(2))

    # Success - return parsed metadata
    return ValidationResult(
        valid=True,
        filename=filename,
        warning=warning,
        parsed={
            'date': date_str,
            'animal_id': animal_id,
            'experiment': parsed['experiment'],
            'cohort': parsed['cohort'],
            'subject': parsed['subject'],
            'is_blank': parsed['is_blank'],
            'tray_type': tray_type,
            'tray_run': tray_run
        }
    )


# =============================================================================
# AUTO-FIX SUGGESTIONS
# =============================================================================

def suggest_fix(filename: str) -> Optional[str]:
    """Suggest automatic fix for common filename issues.

    Common fixes:
    - Extra/missing underscores
    - Wrong extension casing (.MKV -> .mkv)
    - Missing tray suffix
    - Spaces in filename

    Args:
        filename: Malformed filename

    Returns:
        Suggested corrected filename, or None if can't suggest
    """
    original = filename
    filename = Path(filename).name  # Strip path

    # Fix: Spaces to underscores
    if ' ' in filename:
        filename = filename.replace(' ', '_')

    # Fix: Wrong extension casing
    stem, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    if ext and ext.upper() in [e.lstrip('.').upper() for e in VIDEO_EXTENSIONS]:
        ext = ext.lower()
        filename = f"{stem}.{ext}"

    # Fix: Double underscores
    while '__' in filename:
        filename = filename.replace('__', '_')

    # Fix: Leading/trailing underscores
    stem = Path(filename).stem
    if stem.startswith('_') or stem.endswith('_'):
        stem = stem.strip('_')
        filename = f"{stem}{Path(filename).suffix}"

    # Only return if we actually changed something
    if filename != original:
        return filename

    # Try to detect missing tray suffix
    # Pattern: YYYYMMDD_{IDs} without tray suffix
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) == 2 and re.match(r'^\d{8}$', parts[0]):
        # Missing tray suffix - suggest P1 as default
        ext = Path(filename).suffix
        return f"{stem}_P1{ext}"

    # Can't suggest a fix
    return None


# =============================================================================
# QUARANTINE MANAGEMENT
# =============================================================================

def quarantine_file(
    file_path: Path,
    quarantine_dir: Path,
    reason: str,
    suggested_fix: Optional[str] = None
) -> None:
    """Move file to quarantine and record reason.

    Creates a .quarantine.json file with metadata about why the file was quarantined.

    Args:
        file_path: Path to file to quarantine
        quarantine_dir: Quarantine directory
        reason: Error message explaining why file was quarantined
        suggested_fix: Optional suggested filename fix
    """
    file_path = Path(file_path)
    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Move file to quarantine
    dest_path = quarantine_dir / file_path.name

    # Handle name collision (shouldn't happen, but be safe)
    counter = 1
    while dest_path.exists():
        stem = file_path.stem
        ext = file_path.suffix
        dest_path = quarantine_dir / f"{stem}_conflict{counter}{ext}"
        counter += 1

    try:
        shutil.move(str(file_path), str(dest_path))
    except Exception as e:
        logger.error(f"Failed to move {file_path} to quarantine: {e}")
        raise

    # Write quarantine metadata
    metadata = {
        'original_path': str(file_path.resolve()),
        'quarantine_path': str(dest_path.resolve()),
        'error_message': reason,
        'timestamp': datetime.now().isoformat(),
        'suggested_fix': suggested_fix
    }

    metadata_path = dest_path.with_suffix(dest_path.suffix + '.quarantine.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.warning(
        f"Quarantined: {file_path.name}\n"
        f"  Reason: {reason}\n"
        f"  Location: {dest_path}\n"
        f"  Suggested fix: {suggested_fix or 'None'}"
    )


def get_quarantined_files(quarantine_dir: Path) -> List[dict]:
    """Get list of all quarantined files with metadata.

    Args:
        quarantine_dir: Quarantine directory

    Returns:
        List of quarantine records (dicts from .quarantine.json files)
    """
    quarantine_dir = Path(quarantine_dir)
    if not quarantine_dir.exists():
        return []

    records = []
    for json_file in quarantine_dir.glob('*.quarantine.json'):
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
                metadata['metadata_file'] = str(json_file)
                records.append(metadata)
        except Exception as e:
            logger.error(f"Failed to read quarantine metadata {json_file}: {e}")

    return records


def release_from_quarantine(
    filename: str,
    quarantine_dir: Path,
    dest_dir: Path
) -> bool:
    """Release file from quarantine back to processing directory.

    Args:
        filename: Name of quarantined file (not full path)
        quarantine_dir: Quarantine directory
        dest_dir: Destination directory to restore file to

    Returns:
        True on success, False on failure
    """
    quarantine_dir = Path(quarantine_dir)
    dest_dir = Path(dest_dir)

    quarantined_file = quarantine_dir / filename
    if not quarantined_file.exists():
        logger.error(f"Quarantined file not found: {quarantined_file}")
        return False

    # Find associated metadata
    metadata_file = quarantined_file.with_suffix(quarantined_file.suffix + '.quarantine.json')

    # Move file back
    dest_path = dest_dir / filename
    try:
        shutil.move(str(quarantined_file), str(dest_path))
    except Exception as e:
        logger.error(f"Failed to release {filename} from quarantine: {e}")
        return False

    # Remove metadata
    if metadata_file.exists():
        metadata_file.unlink()

    logger.info(f"Released from quarantine: {filename} -> {dest_path}")
    return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_filename(filename: str, expect_single: bool = None) -> ValidationResult:
    """Auto-detect file type and validate.

    Args:
        filename: Filename to validate
        expect_single: If True, force single-animal validation.
                      If False, force collage validation.
                      If None, auto-detect based on extension and comma presence.

    Returns:
        ValidationResult
    """
    filename = Path(filename).name

    # Auto-detect if not specified
    if expect_single is None:
        ext = Path(filename).suffix.lower()
        stem = Path(filename).stem

        # .mp4 with no commas = likely single
        # .mkv or has commas = likely collage
        has_commas = ',' in stem
        expect_single = (ext == '.mp4') and not has_commas

    if expect_single:
        return validate_single_filename(filename)
    else:
        return validate_collage_filename(filename)


def validate_and_quarantine(
    file_path: Path,
    quarantine_dir: Path,
    expect_single: bool = None
) -> ValidationResult:
    """Validate filename and quarantine if invalid.

    Convenience function combining validation and quarantine.

    Args:
        file_path: Path to file to validate
        quarantine_dir: Quarantine directory (used if invalid)
        expect_single: Passed to validate_filename()

    Returns:
        ValidationResult (valid=False if quarantined)
    """
    result = validate_filename(file_path.name, expect_single=expect_single)

    if not result.valid:
        # Try to suggest a fix
        suggested = suggest_fix(file_path.name)

        # Quarantine the file
        quarantine_file(
            file_path,
            quarantine_dir,
            reason=result.error,
            suggested_fix=suggested
        )

    return result
