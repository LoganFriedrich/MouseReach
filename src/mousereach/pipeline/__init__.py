"""
MouseReach Unified Pipeline
=======================

Single widget for running the complete analysis pipeline:
- Stage 1: Segmentation (find pellet presentation boundaries)
- Stage 2: Outcome Detection (classify R/D/M for each segment)
- Stage 3: Reach Detection (find individual reach attempts)

All files stay in Processing/. Status is tracked via validation_status
field in JSON files, not by folder location.
"""

from mousereach.pipeline.batch_widget import UnifiedPipelineWidget
from mousereach.pipeline.core import (
    UnifiedPipelineProcessor,
    UnifiedResults,
    PipelineStatus,
    scan_pipeline_status,
    consolidate_all_to_dlc_complete,
)

__all__ = [
    'UnifiedPipelineWidget',
    'UnifiedPipelineProcessor',
    'UnifiedResults',
    'PipelineStatus',
    'scan_pipeline_status',
    'consolidate_all_to_dlc_complete',
]
