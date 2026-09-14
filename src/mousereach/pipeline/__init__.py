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

# The package's public names load on first use, not at import. WHY: importing
# ANY submodule of this package (pipeline.manifest, pipeline.analyzed_tree,
# pipeline.versions ...) runs this file first, and eagerly importing
# batch_widget loaded napari and Qt -- about six seconds -- into every headless
# command and scheduled job that only needed a small helper (mousereach-reconcile,
# the census, the version scan, the ASPA tools). `from mousereach.pipeline import
# UnifiedPipelineWidget` still works; napari.yaml and the launcher name
# batch_widget directly and are unaffected.
_LAZY = {
    'UnifiedPipelineWidget': 'mousereach.pipeline.batch_widget',
    'UnifiedPipelineProcessor': 'mousereach.pipeline.core',
    'UnifiedResults': 'mousereach.pipeline.core',
    'PipelineStatus': 'mousereach.pipeline.core',
    'scan_pipeline_status': 'mousereach.pipeline.core',
    'consolidate_all_to_dlc_complete': 'mousereach.pipeline.core',
}

__all__ = list(_LAZY)


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
