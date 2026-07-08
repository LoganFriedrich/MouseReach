<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# review

## Purpose
Unified review widget components that provide algorithm vs ground truth comparison functionality across all MouseReach review tools (segmentation, reach, outcome). Includes a mixin class for split-view comparison, a tabbed unified review tool that combines all three review modes into one interface, and a comparison panel for visualizing differences between algorithm output and ground truth data.

## Key Files
| File | Description |
|------|-------------|
| `base.py` | AlgoGTReviewMixin base class providing split-view (algo output vs GT) comparison, diff summary, and separate "Save Validation" and "Save as GT" buttons |
| `unified_widget.py` | UnifiedReviewWidget - tabbed napari widget combining all three review tools (Boundaries, Reaches, Outcomes) with shared video loading and context-aware navigation |
| `comparison_panel.py` | ComparisonPanel and helper functions for creating side-by-side visual comparisons of algo output vs ground truth |
| `save_panel.py` | SavePanel component for handling validation and ground truth file saving with status updates |
| `__init__.py` | Package exports for all review components |

## Subdirectories
None

## For AI Agents

### Working In This Directory
- **Mixin architecture**: Use AlgoGTReviewMixin to add algo vs GT comparison to any review widget
- The mixin provides two save paths: "Save Validation" (updates algo output for downstream pipeline) and "Save as GT" (creates ground truth files for evaluation)
- **Split view design**: Algorithm output shown on left (read-only), ground truth/editable copy on right
- **Diff tracking**: Differences are highlighted and summarized with accuracy metrics
- **Index integration**: Saving triggers PipelineIndex updates for validation_status and ground truth tracking

### UnifiedReviewWidget Key Features
1. **Shared video layer** - Video loaded once, shared across all tabs for memory efficiency
2. **Progressive initialization** - Tab widgets created in background to avoid blocking UI
3. **Context-aware navigation** - N/P keys and item navigation adapt to active tab (Boundary/Reach/Segment)
4. **Dropdown workflow** - Videos needing review auto-populate from PipelineIndex
5. **Tab dependencies** - Reaches and Outcomes tabs only enabled when segments exist
6. **Help dialog** - Built-in keyboard shortcuts reference (? button)

### Implementing AlgoGTReviewMixin
To add algo vs GT comparison to a widget:
```python
class MyReviewWidget(QWidget, AlgoGTReviewMixin):
    def __init__(self, viewer):
        QWidget.__init__(self)
        AlgoGTReviewMixin.__init__(self)
        # ... setup UI ...
        self._init_algo_gt_panels(layout)  # Call after UI setup

    # Implement required abstract methods:
    def _get_algo_output_path(self) -> Path: ...
    def _get_gt_path(self) -> Path: ...
    def _load_algo_data(self) -> dict: ...
    def _load_gt_data(self) -> Optional[dict]: ...
    def _compute_diff(self) -> DiffSummary: ...
    # ... (see base.py for full list)
```

### DiffSummary Usage
- `total_items`: Number of items compared
- `matching_items`: Items that match between algo and GT
- `differing_items`: Items that differ
- `accuracy`: Property returning match rate (0.0-1.0)
- `summary_text()`: Human-readable summary string

## Dependencies

### Internal
- `mousereach.config` - Paths and video_id utilities (PROCESSING_ROOT, get_video_id)
- `mousereach.index` - PipelineIndex for tracking validation status and ground truth
- `mousereach.segmentation.review_widget` - BoundaryReviewWidget (embedded in unified tool)
- `mousereach.reach.review_widget` - ReachAnnotatorWidget (embedded in unified tool)
- `mousereach.outcomes.review_widget` - PelletOutcomeAnnotatorWidget (embedded in unified tool)
- `mousereach.video_prep.compress` - create_preview() for compressed video loading

### External
- `napari` - GUI viewer and notifications
- `qtpy` - Qt widgets (QSplitter, QListWidget, QTabWidget, etc.)
- `pathlib` - Path handling
- `json` - JSON file I/O
- `cv2` (opencv) - Video loading in UnifiedReviewWidget
- `numpy` - Video frame array handling
- `abc` - Abstract base classes for mixin pattern
- `dataclasses` - Structured data (DiffItem, DiffSummary)

<!-- MANUAL: -->

## Routine triage + QC tool (`mousereach-review-tool`)

The **triage tool** the routine protocol uses is `TriageClearingWidget`
(`triage_clearing.py`, CLI `mousereach-review-tool`) -- NOT the tabbed
`UnifiedReviewWidget`, which is **deprecated legacy**. It is a lean,
worklist-driven surface that ingests ONLY unresolved problems and presents them
one segment at a time. (`CausalReviewWidget` in `causal_review_widget.py` is the
separate deep per-video results-review surface.)

| File | Role |
|------|------|
| `triage_queue.py` | Build the worklist: `scan_corpus_for_triage` (per bundle) + `scan_corpus_root_for_triage` / `TriageWorklist.from_corpus_root` (whole corpus). Triggers: `flagged_for_review` (outcome/reach) AND the **unattributed-causal-reach** trigger (a touched outcome with no committed causal reach in `_reach_assignments.json`). Failed-seg videos -> `needs_reseg` lane, not the worklist. `find_default_corpus_root()` -> `Model40_Review/Pending` (or `MOUSEREACH_ROUTINE_ROOT`). |
| `triage_clearing.py` | The widget + CLI. Per-segment clear saves `triage_cleared`/`human_verified`/`human_corrected` back to the bundle's JSONs; non-causal reaches get `exclude_from_analysis`. |
| `qc_pool.py` | Routine spot-check pool: sample the algo's CONFIDENT calls (stratified rotating by cohort\|date), confirm-the-algo, persist to `<review_root>/_QC/` (`qc_state.json` + `qc_drift_log.jsonl`), report an agreement rate. |
| `clear_guard.py` | Preserve human clears across a re-run (see invariant below). |

### Ingestion layouts (mode-gated)
- **corpus_root** (routine): per-video bundle dirs; the algo JSONs live APART
  from the video and (under deferred 4.0 activation) use a different DLC model
  than the one beside the mp4. `TriageClearingWidget` overrides `_gt_algo_dir()`
  (load JSONs from the bundle) and `_dlc_h5_path()` (load the canonical
  resnet101 pose) so it shows the bundle's real 4.0 data, not the stale
  co-located production outputs. Gated by `worklist.mode == "corpus_root"`.
- **algo_dir** (quarantine/flat): video + h5 + JSONs co-located; base
  `GroundTruthWidget` behavior, unchanged.

`unified_gt.py` and `ground_truth_widget.py` gained backward-compatible
`algo_dir` overrides / `_gt_algo_dir()` + `_dlc_h5_path()` hooks (default None =
co-located) to support the split (bundle) layout without touching base behavior.

### Spot-check CLI
- `--qc-count N` blends N stratified-rotating spot-checks into a corpus-root
  session (kind=="qc" entries; green Confirm / Flag row). Confirm logs
  agreement; Flag sets `flagged_for_review` so the segment re-enters triage.
- `--qc-report` prints the agreement/drift summary without launching napari.

### INVARIANT: clear-before-kinematics (do not clobber human clears)
Routine order is **PROCESS -> TRIAGE-CLEAR (human) -> KINEMATICS**. Kinematics
reads the in-place algo JSONs, so a human clear propagates natively -- but a
re-run of the detectors between clearing and kinematics would overwrite it.
`stage_video(..., preserve_clears=True)` (default) captures human-locked
segments before a re-stage and re-applies them via
`clear_guard.merge_preserving_clears` after the fresh run (human call wins,
matched by `segment_num`). It is intentionally **skipped when boundaries change**
(manual re-seg renumbers segments); locked segments that vanish are reported,
not silently dropped. Any future detector re-run path MUST honor this guard.
