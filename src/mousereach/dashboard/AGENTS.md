<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# dashboard

## Purpose
Pipeline status dashboard napari widget providing a complete overview of all files in the MouseReach pipeline. Shows each video's location, processing stage, analysis versions, validation status for all three steps (segmentation, reach, outcome), timestamps, ground truth files, and overall pipeline health. Uses the PipelineIndex for fast startup instead of slow folder scanning.

## Key Files
| File | Description |
|------|-------------|
| `widget.py` | Main PipelineDashboard widget with two tabs: Pipeline Overview (filterable table) and Statistics (validation status counts). Per-video info opens from the selection-gated **File Details button** on Pipeline Overview, not a tab. |
| `folder_scan.py` | Scans the whole pipeline tree; a video's stage = which folder it sits in. Skips dot-dirs (tool scratch like `.omc` is not a review bundle). |
| `version_currency.py` | Is each video current with the shipped algo versions? The FALLBACK path: builds the manifest index and (via `build_version_maps`) resolves status + DLC model in one pass. Used only when the version index is empty/unavailable. |
| `__init__.py` | Package exports for PipelineDashboard and IndexAdapter |

## Subdirectories
None

## For AI Agents

### Working In This Directory
- **Version data comes from the index, not a scan.** `_resolve_version_maps()`
  reads `mousereach.pipeline.version_index` in ONE query; the writers maintain it
  (`create_processing_manifest` pushes a row per video). The manifest scan is only
  the fallback for an empty/unavailable index. A LAGGING index is not repaired by
  scanning -- that is the cost being removed; those videos read "?" and
  `mousereach-version-index-build` fixes them.
- **NEVER do per-row file I/O in the table build.** `_update_overview_table` runs
  once per video (~3,800 rows) on the GUI thread, so any NAS read inside that loop
  is multiplied by the row count and freezes the dashboard. This actually happened:
  `version_status_for` / `dlc_model_for` each re-read the video's manifest off the
  NAS (~22 ms), costing ~168 s of frozen GUI per repaint. Resolve such data ONCE on
  a worker thread into a dict (see `build_version_maps`) and make the per-row call
  an O(1) lookup.
- **Performance critical**: Always use PipelineIndex (via IndexAdapter) instead of scanning folders directly
- Auto-refresh is **disabled** to prevent multi-hour freezes on network drives - users manually refresh when needed
- The dashboard shows **individual validation status** for each step (seg_status, reach_status, outcome_status) in v2.3+ architecture
- Status cells are **clickable** - clicking a "needs review" cell launches the appropriate review tool and jumps to the issue location
- Tray type filtering: Unsupported trays (type F) are hidden by default and highlighted in red when shown
- Archive-ready videos have all three validation statuses set to "validated"
- Ground truth files are tracked separately (S=seg, R=reach, O=outcome) and displayed compactly

### Key Features
1. **Filter by stage** - Show only DLC_Queue, Processing, Failed, or all files
2. **Needs Review filter** - Checkbox to show only files requiring human review
3. **Clickable status cells** - Launch review tools directly from status indicators
4. **Rebuild index** - Full rescan button for when index is stale
5. **Color-coded status** - Green=validated, Orange=needs_review, Light blue=auto_approved, Gray=pending

### Status Icon Legend
- ✓ (Green) = Validated
- ⏳ (Orange) = Needs Review
- ⚡ (Light Blue) = Auto-approved
- \- (Gray) = Pending
- ? (No color) = Unknown

## Dependencies

### Internal
- `mousereach.config` - Paths configuration (PROCESSING_ROOT)
- `mousereach.index` - PipelineIndex for fast data loading
- `mousereach.segmentation.review_widget` - SegmentationReviewWidget (launched on click)
- `mousereach.reach.review_widget` - ReachAnnotatorWidget (launched on click)
- `mousereach.outcomes.review_widget` - PelletOutcomeAnnotatorWidget (launched on click)

### External
- `napari` - GUI viewer and notifications
- `qtpy` - Qt widgets (QTableWidget, QComboBox, QTabWidget, QCheckBox, etc.)
- `pathlib` - Path handling
- `json` - JSON file I/O for loading metadata
- `datetime` - Timestamp formatting

<!-- MANUAL: -->
