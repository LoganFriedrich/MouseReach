"""
MouseReach pipeline helpers (manifests, versions, the filed tree, triage) and
the "2 - Run Pipeline" tab (batch_widget), which runs one pass of this
machine's watcher by hand.

The tab's old separate processor (pipeline/core.py: UnifiedPipelineProcessor,
scan_pipeline_status, consolidate_all_to_dlc_complete) is removed. It ran an
older copy of the steps -- the old outcome detector before reach detection,
no gate, no queue, no record -- and its "Reprocess All" deleted every
segments file in Processing, a person's cuts included. batch_widget's
docstring says what replaced it and why.
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
