"""
MouseReach Step 4: Reach Detection
=============================

Detect individual reach attempts within each pellet presentation segment.
"""


def __getattr__(name):
    """Lazy import to avoid loading napari widget at package import time.

    The ReachAnnotatorWidget requires napari/Qt which is slow to import and
    hangs in headless environments. Deferring the import lets CLI tools
    (batch detection, eval) load quickly.
    """
    if name == "ReachAnnotatorWidget":
        from mousereach.reach.review_widget import ReachAnnotatorWidget
        return ReachAnnotatorWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['ReachAnnotatorWidget']
