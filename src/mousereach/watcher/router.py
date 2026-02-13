"""
Tray-type-aware pipeline routing for MouseReach watcher.

Determines which processing steps to run based on tray type:
- P (Pillar): Full pipeline — has a calibrated DLC model for all points
- E (Easy): Limited pipeline — mouse/box points OK, pellet/tray unreliable
- F (Flat): Limited pipeline — same limitations as E
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# All pipeline steps in execution order
ALL_STEPS = [
    'crop',
    'dlc',
    'segment',
    'reach_detection',
    'outcome_detection',
    'kinematics',
    'export',
    'archive',
]

# Steps to SKIP for each tray type (empty = run all)
SKIP_STEPS = {
    'P': set(),                      # Pillar: full pipeline
    'E': {'outcome_detection'},       # Easy: skip pellet outcome classification
    'F': {'outcome_detection'},       # Flat: skip pellet outcome classification
}

# Known tray types with descriptions
TRAY_DESCRIPTIONS = {
    'P': 'Pillar (full pipeline, calibrated DLC model)',
    'E': 'Easy (training/rehab, limited pellet tracking)',
    'F': 'Flat (training/rehab, limited pellet tracking)',
}


class TrayRouter:
    """Routes videos through appropriate pipeline steps based on tray type."""

    def get_pipeline_steps(self, tray_type: str) -> List[str]:
        """Return ordered list of pipeline steps for a given tray type."""
        tray = tray_type.upper()
        skip = SKIP_STEPS.get(tray, set())
        if tray not in SKIP_STEPS:
            logger.warning(f"Unknown tray type '{tray}', defaulting to full pipeline")
        return [step for step in ALL_STEPS if step not in skip]

    def should_run_step(self, tray_type: str, step: str) -> bool:
        """Check if a specific step should run for a given tray type."""
        tray = tray_type.upper()
        skip = SKIP_STEPS.get(tray, set())
        return step not in skip

    def get_skip_reason(self, tray_type: str, step: str) -> str:
        """Return human-readable reason why a step is skipped, or empty string if not skipped."""
        tray = tray_type.upper()
        if step not in SKIP_STEPS.get(tray, set()):
            return ""
        if step == 'outcome_detection' and tray in ('E', 'F'):
            return f"Tray type '{tray}' ({TRAY_DESCRIPTIONS.get(tray, 'unknown')}): pellet/tray tracking unreliable, skipping outcome classification"
        return f"Step '{step}' skipped for tray type '{tray}'"

    def is_known_tray(self, tray_type: str) -> bool:
        """Check if the tray type is recognized."""
        return tray_type.upper() in SKIP_STEPS

    def describe_tray(self, tray_type: str) -> str:
        """Get human-readable description of tray type."""
        return TRAY_DESCRIPTIONS.get(tray_type.upper(), f"Unknown tray type '{tray_type}'")
