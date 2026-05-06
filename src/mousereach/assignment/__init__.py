"""
Reach assignment -- joins reaches to segments and pellets, identifies
the causal reach for each touched segment's outcome.

This is the 4th algo of the MouseReach pipeline. See
`four_algo_decomposition.md` in cross-session memory for context.

Production entry point: `mousereach.assignment.v1.assign_reaches_v1`
(re-exported here as `assign_reaches_v1`). It takes the v8 reach
detector + v6 cascade outcome detector outputs and produces a permanent
per-reach output table with outcome labels stamped, so downstream
kinematic analysis never re-derives outcomes from the segment side.
"""
from .v1 import VERSION as V1_VERSION
from .v1 import assign_reaches_v1

__all__ = ["assign_reaches_v1", "V1_VERSION"]
