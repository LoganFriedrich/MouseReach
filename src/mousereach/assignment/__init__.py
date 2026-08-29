"""
Reach assignment -- joins reaches to segments and pellets, identifies
the causal reach for each touched segment's outcome.

This is the 4th algo of the MouseReach pipeline. See
`four_algo_decomposition.md` in cross-session memory for context.

v1 -- cascade-trusted IFR-join (single signal).
v2 -- two-signal agreement gate (IFR + displacement). Commits only
      when both signals agree; triages on disagreement.

Production entry point: ``mousereach.assignment.run.assign_reaches_for_video``
(v2, requires the DLC pose). The watcher, ``pipeline/run_all.py``,
``pipeline/reprocess_to_current.py`` and ``mousereach-assign-reaches`` all go
through it. ``assign_reaches_v1`` stays importable for provenance only --
nothing in the pipeline or the command line tools has called it since 2026-08.
"""
from .v1 import VERSION as V1_VERSION
from .v1 import assign_reaches_v1
from .v2 import VERSION as V2_VERSION
from .v2 import assign_reaches_v2

__all__ = [
    "assign_reaches_v1", "V1_VERSION",
    "assign_reaches_v2", "V2_VERSION",
]
