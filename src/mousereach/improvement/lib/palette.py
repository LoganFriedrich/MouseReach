"""
Shared color palettes for improvement process figures.

Consolidates outcome colors (from mousereach.outcomes.review_widget and
mousereach.analysis.plots) and figure defaults into one importable module.

Usage:
    from mousereach.improvement.lib.palette import OUTCOME_COLORS, PHASE_COLORS, FIGURE_DEFAULTS
"""

# ---------------------------------------------------------------------------
# Outcome palette (canonical -- matches review_widget.py and plots.py)
# ---------------------------------------------------------------------------

OUTCOME_COLORS = {
    "retrieved": "#4CAF50",           # Green
    "displaced_sa": "#FF9800",        # Orange
    "displaced_outside": "#FF5722",   # Deep orange
    "untouched": "#2196F3",           # Blue
    "no_pellet": "#9E9E9E",           # Gray
    "uncertain": "#9C27B0",           # Purple
    # v4.0.0+: per-reach categories
    "miss": "#90A4AE",                 # Blue-gray - reach happened but didn't accomplish anything
    "absent": "#424242",               # Dark gray - reach exists only on the other side
}

# Shorter alias used in some analysis scripts
OUTCOME_COLORS_SHORT = {
    "R": OUTCOME_COLORS["retrieved"],
    "D": OUTCOME_COLORS["displaced_sa"],
    "O": OUTCOME_COLORS["displaced_outside"],
    "U": OUTCOME_COLORS["untouched"],
    "N": OUTCOME_COLORS["no_pellet"],
    "?": OUTCOME_COLORS["uncertain"],
}


# ---------------------------------------------------------------------------
# Phase palette (test phases P1-P4)
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "P1": "#1f77b4",  # Blue - baseline
    "P2": "#ff7f0e",  # Orange - early
    "P3": "#2ca02c",  # Green - mid
    "P4": "#d62728",  # Red - late
}


# ---------------------------------------------------------------------------
# Algorithm diagram colors (matches segmenter.dot conventions)
# ---------------------------------------------------------------------------

DIAGRAM_COLORS = {
    "input_output": "#FFF2CC",     # Light yellow - inputs/outputs
    "phase_quality": "#DEEBF7",    # Light blue - Phase 1 quality
    "phase_primary": "#E2F0D9",    # Light green - Phase 2 primary
    "phase_fallback": "#FFE699",   # Light orange - Phase 2b fallback
    "phase_consensus": "#D5E8D4",  # Light teal-green - Phase 2.5 consensus (new)
    "phase_fit": "#FFF2CC",        # Light yellow - Phase 3 fit grid
    "phase_validate": "#E4DFEC",   # Light purple - Phase 4 validate
    "phase_emit": "#F2F2F2",       # Light grey - Phase 5 emit
}


# ---------------------------------------------------------------------------
# Segmentation subset palette (boundary analysis figures)
# ---------------------------------------------------------------------------

SEGMENTATION_COLORS = {
    "all": "#5C6BC0",               # Indigo - all boundaries combined
    "inter_pellet_B2_B20": "#42A5F5",  # Light blue - interior boundaries
    "endpoint_B1_B21": "#AB47BC",   # Purple - first/last boundaries
}

SEGMENTATION_LABELS = {
    "all": "All boundaries",
    "inter_pellet_B2_B20": "Inter-pellet (B2–B20)",
    "endpoint_B1_B21": "Endpoints (B1 + B21)",
}

# Ordered list for consistent plotting (top-to-bottom in violin)
SEGMENTATION_SUBSET_ORDER = ["all", "inter_pellet_B2_B20", "endpoint_B1_B21"]


# ---------------------------------------------------------------------------
# Reach detection palette (delta type analysis figures)
# ---------------------------------------------------------------------------

REACH_DETECTION_COLORS = {
    "start_delta": "#EF5350",           # Red - start boundary deltas
    "end_delta": "#42A5F5",             # Blue - end boundary deltas
}

REACH_DETECTION_LABELS = {
    "start_delta": "Start delta",
    "end_delta": "End delta",
}

# Ordered list for consistent plotting (left-to-right in violin)
REACH_DETECTION_DELTA_ORDER = ["start_delta", "end_delta"]


# ---------------------------------------------------------------------------
# Outcome verdict palette (improvement process evaluation)
# ---------------------------------------------------------------------------

OUTCOME_VERDICT_COLORS = {
    "label_and_reach_correct": "#4CAF50",       # Green - fully correct
    "label_correct_untouched": "#81C784",        # Light green - correct untouched
    "label_correct_wrong_reach": "#FF9800",      # Orange - right label, wrong reach
    "label_wrong": "#F44336",                    # Red - wrong label
    "abstained": "#9E9E9E",                      # Gray - algo abstained
}

OUTCOME_VERDICT_LABELS = {
    "label_and_reach_correct": "Label + reach correct",
    "label_correct_untouched": "Untouched correct",
    "label_correct_wrong_reach": "Label correct, wrong reach",
    "label_wrong": "Label wrong",
    "abstained": "Abstained (uncertain)",
}

# Ordered list for consistent stacking in sankey/summary
OUTCOME_VERDICT_ORDER = [
    "label_and_reach_correct",
    "label_correct_untouched",
    "label_correct_wrong_reach",
    "label_wrong",
    "abstained",
]

# Outcome class order for confusion matrix / sankey
OUTCOME_CLASS_ORDER = [
    "retrieved",
    "displaced_sa",
    "displaced_outside",
    "untouched",
    "uncertain",
    "miss",      # v4.0.0+: per-reach -- non-causal reaches default here
    "absent",    # v4.0.0+: per-reach -- reach exists only on the other side
]


# ---------------------------------------------------------------------------
# Figure defaults
# ---------------------------------------------------------------------------

FIGURE_DEFAULTS = {
    "figsize": (10, 6),
    "dpi": 300,
    "fontsize_title": 14,
    "fontsize_label": 12,
    "fontsize_tick": 10,
    "fontsize_legend": 10,
    "linewidth": 1.5,
    "marker_size": 6,
    "grid_alpha": 0.3,
    "savefig_bbox": "tight",
    "savefig_pad": 0.1,
}
