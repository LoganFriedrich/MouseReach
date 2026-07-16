"""
MouseReach Pipeline Dashboard Widget
================================

Overview of all files in the pipeline with their status, versions, and locations.

Shows:
- Each file and where it is in the processing pipeline
- Version numbers of analysis used
- Timestamps for each stage
- Whether ground truth files exist
- Overall status (succeeded, warning, failed)

Performance Note:
Uses PipelineIndex for fast startup. If index is stale, run: mousereach-index-rebuild
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QComboBox, QGroupBox, QProgressBar, QTabWidget, QTextEdit,
    QCheckBox
)
from qtpy.QtCore import Qt, QTimer, QObject, Signal
from qtpy.QtGui import QColor, QBrush


class _VersionCheckWorker(QObject):
    """Carries the 'version index built' signal from the worker thread to the GUI."""
    done = Signal()


class _BackfillWorker(QObject):
    """Carries the archive-backfill summary from the worker thread to the GUI."""
    done = Signal(dict)


class _FolderScanWorker(QObject):
    """Carries folder-scan progress + completion from the worker thread to the GUI."""
    progress = Signal(str)
    done = Signal()


# Plain-language labels for the raw watcher states (which mean nothing to a
# novice). Each: (label, tooltip, category) where category drives the text color.
STAGE_INFO = {
    "discovered":   ("New, not started",           "Just found -- no processing has started yet.", "wait"),
    "quarantined":  ("Held: bad name",             "Set aside -- the filename/file could not be read. Fix it in the Quarantine tab.", "bad"),
    "validated":    ("Ready to process",           "Filename checked -- waiting to be cropped / pose-tracked.", "wait"),
    "dlc_queued":   ("Waiting for pose tracking",  "Queued for DeepLabCut pose estimation on a GPU machine.", "wait"),
    "dlc_running":  ("Pose tracking (running)",    "DeepLabCut pose estimation is running.", "busy"),
    "dlc_complete": ("Pose done, ready for MouseReach", "Pose tracking (DLC) finished -- ready for the MouseReach algorithms.", "wait"),
    "processing":   ("Running MouseReach algos",        "Running the MouseReach algorithms: segmentation, reach detection, outcomes, assignment, and kinematics.", "busy"),
    "processed":    ("MouseReach done, not archived",   "Pose tracking + all 4 MouseReach algorithms + kinematics are done and saved to the database. Files are still in the working folder, not yet copied to the archive. (Human review, if any, is the Review column.)", "done"),
    "archiving":    ("Copying to archive",         "Copying the finished results to the archive.", "busy"),
    "archived":     ("Done (archived)",            "Fully finished -- results are analyzed and stored in the archive.", "done"),
    "outdated":     ("Out of date, reprocess",     "Was processed with older algorithm versions -- reprocess to bring it up to date.", "act"),
    "crystallized": ("Locked (published)",         "Locked against reprocessing (e.g. frozen for a publication).", "done"),
    "triage":       ("Waiting for triage review",  "Held out of the database -- a reviewer must answer a per-element question (Review Queues tab).", "act"),
    "deep_review":  ("Waiting for deep review",    "Held out -- segmentation failed or was escalated; needs the deep-review tools (Review Queues tab).", "act"),
    "failed":       ("Failed, needs a look",       "Processing errored out -- needs investigation.", "bad"),
    # Folder-scan stages (video's stage = which pipeline folder it sits in)
    "raw_collage":  ("Raw collage, needs cropping", "An 8-camera collage in the intake folder -- not yet cropped into single-mouse videos.", "wait"),
    "cropped":      ("Cropped, waiting for pose",   "Cropped to a single-mouse video, waiting for pose tracking (DLC).", "wait"),
    "analyzed":     ("Final output (done)",         "Fully processed -- DLC + all 4 MouseReach algorithms + saved to mousedb. This is the final data output.", "done"),
}

_STAGE_CATEGORY_COLOR = {
    "act": QColor(230, 145, 0),    # orange -- needs a human
    "bad": QColor(211, 47, 47),    # red -- broken
    "busy": QColor(25, 118, 210),  # blue -- in progress
    "done": QColor(56, 142, 60),   # green -- finished
    "wait": None,                  # neutral
}


def stage_label(state: str):
    """(label, tooltip, category) for a raw watcher state, with a safe fallback."""
    return STAGE_INFO.get(state, (state, f"State: {state}", "wait"))


# Meaningful groupings for the overview 'Filter by stage' menu. Each maps a
# plain-language label to a set of raw states (None = show everything).
STAGE_FILTERS = [
    ("All videos", None),
    ("Needs attention", {"triage", "deep_review", "failed", "outdated", "quarantined"}),
    ("Not started (raw/cropped)", {"raw_collage", "cropped"}),
    ("In progress", {"validated", "dlc_queued", "dlc_running", "dlc_complete", "processing",
                     "archiving", "raw_collage", "cropped"}),
    ("Done (final output)", {"processed", "archived", "crystallized", "analyzed"}),
    ("Needs triage review", {"triage"}),
    ("Needs deep review", {"deep_review"}),
    ("Needs reprocessing (outdated)", {"outdated"}),
    ("Failed", {"failed"}),
    ("Quarantined", {"quarantined"}),
]
_STAGE_FILTER_MAP = dict(STAGE_FILTERS)

import napari
from napari.utils.notifications import show_info, show_error

from mousereach.config import Paths


class IndexAdapter:
    """Adapts PipelineIndex to the format expected by PipelineDashboard.

    Provides fast index-based lookups instead of slow folder scanning.
    """

    def __init__(self):
        from mousereach.index import PipelineIndex
        self.index = PipelineIndex()
        self.processing_root = Paths.PROCESSING_ROOT
        self.files: Dict[str, Dict] = {}

    def scan(self) -> Dict[str, Dict]:
        """Load data from index (fast) instead of scanning folders (slow).

        Returns data in the same format as the old PipelineScanner.
        """
        self.files = {}

        if not self.processing_root.exists():
            return self.files

        # Load index (fast: single file read)
        self.index.load()

        # NOTE: Auto-refresh disabled - was causing multi-hour freezes on network drives
        # If index is stale, user can manually run: mousereach-index-rebuild
        # stale = self.index.check_stale_folders()
        # if stale:
        #     for folder in stale:
        #         self.index.refresh_folder(folder)

        # Convert index format to dashboard format
        for video_id, video_data in self.index.get_all_videos().items():
            current_stage = video_data.get("current_stage", "Unknown")
            metadata = video_data.get("metadata", {})
            files_by_stage = video_data.get("files", {})

            # Build locations list from files dict
            locations = []
            for stage, filenames in files_by_stage.items():
                # Find the .mp4 file path
                for fname in filenames:
                    if fname.endswith(".mp4"):
                        locations.append({
                            "stage": stage,
                            "path": str(self.processing_root / stage / fname)
                        })
                        break

            # Extract versions from metadata
            versions = {}
            if metadata.get("segmenter_version"):
                versions["segmenter"] = metadata["segmenter_version"]
            if metadata.get("reach_version"):
                versions["reach_detector"] = metadata["reach_version"]
            if metadata.get("outcomes_version"):
                versions["outcome_detector"] = metadata["outcomes_version"]

            # Individual validation statuses (v2.3+ architecture)
            seg_status = metadata.get("seg_validation", "pending")
            reach_status = metadata.get("reach_validation", "pending")
            outcome_status = metadata.get("outcome_validation", "pending")

            # Determine overall status from validation metadata
            status = "unknown"
            if outcome_status and outcome_status != "pending":
                status = outcome_status
            elif reach_status and reach_status != "pending":
                status = reach_status
            elif seg_status and seg_status != "pending":
                status = seg_status

            # Check if ready for archive (all three validated)
            archive_ready = (
                seg_status == "validated" and
                reach_status == "validated" and
                outcome_status == "validated"
            )

            # Check for ground truth files
            ground_truths = []
            for stage, filenames in files_by_stage.items():
                for fname in filenames:
                    if "ground_truth" in fname:
                        ground_truths.append(fname)

            # Build timestamps from mtimes
            timestamps = {}
            mtimes = video_data.get("mtimes", {})
            for key, mtime in mtimes.items():
                if isinstance(mtime, (int, float)):
                    timestamps[key] = datetime.fromtimestamp(mtime).isoformat()

            self.files[video_id] = {
                "locations": locations,
                "versions": versions,
                "timestamps": timestamps,
                "ground_truths": ground_truths,
                "status": status,
                "current_stage": current_stage,
                "metadata": metadata,
                # Individual validation statuses (v2.3+ architecture)
                "seg_status": seg_status,
                "reach_status": reach_status,
                "outcome_status": outcome_status,
                "archive_ready": archive_ready,
                # Tray type info (v2.3+ - for E/F detection)
                "tray_type": metadata.get("tray_type"),
                "tray_supported": metadata.get("tray_supported", True),
            }

        return self.files

    def get_file_summary(self, filename: str) -> str:
        """Get a human-readable summary of a file's status."""
        if filename not in self.files:
            return "Not found in pipeline"

        info = self.files[filename]
        lines = [f"File: {filename}"]

        # Current stage
        current_stage = info.get("current_stage", "Unknown")
        lines.append(f"Current Stage: {current_stage}")

        # Pipeline flow - show stages in order
        lines.append("\nPipeline Path:")
        if info["locations"]:
            for i, loc in enumerate(info["locations"], 1):
                stage = loc['stage']
                lines.append(f"  {i}. {stage}")
        else:
            lines.append("  (no locations)")

        # Metadata from index
        metadata = info.get("metadata", {})
        if metadata:
            lines.append("\nCached Metadata:")
            if metadata.get("seg_boundaries"):
                lines.append(f"  • Boundaries: {metadata['seg_boundaries']}")
            if metadata.get("seg_confidence"):
                lines.append(f"  • Seg Confidence: {metadata['seg_confidence']:.2f}")
            if metadata.get("reach_count"):
                lines.append(f"  • Reaches: {metadata['reach_count']}")
            if metadata.get("outcome_count"):
                lines.append(f"  • Outcomes: {metadata['outcome_count']}")

        # Versions
        if info["versions"]:
            lines.append("\nAnalysis Versions:")
            for name, version in info["versions"].items():
                lines.append(f"  • {name}: v{version}")
        else:
            lines.append("\nAnalysis Versions: None yet")

        # Ground truth
        if info["ground_truths"]:
            lines.append("\nGround Truth Files:")
            for gt in info["ground_truths"]:
                lines.append(f"  ✓ {gt}")
        else:
            lines.append("\nGround Truth: ⚠️ Not created yet")

        # Status
        status = info['status']
        status_emoji = "✓" if status == "validated" else "⏳" if status == "needs_review" else "⚡" if status == "auto_review" else "?"
        lines.append(f"\nCurrent Status: {status_emoji} {status}")

        return "\n".join(lines)

    def rebuild_index(self, progress_callback=None):
        """Rebuild the index from scratch."""
        from mousereach.index.scanner import rebuild_index
        return rebuild_index(self.index, progress_callback)


class WatcherAdapter:
    """Adapts the watcher SQLite DB to the dashboard format.

    Used when a ``watcher.db`` exists -- that is the AUTHORITATIVE pipeline state
    (the daemon updates it), so the dashboard shows each video's real position
    (including the new ``triage`` / ``deep_review`` human-review holds) instead of
    the folder-derived PipelineIndex. Exposes the same public surface as
    IndexAdapter so PipelineDashboard is source-agnostic.
    """

    # state -> health bucket used for the status-column coloring
    _VALIDATED = {"archived", "archiving", "processed", "crystallized"}
    _NEEDS_REVIEW = {"triage", "deep_review"}
    _FAILED = {"failed", "quarantined"}

    def __init__(self):
        self.processing_root = Paths.PROCESSING_ROOT
        self.db_path = (Paths.PROCESSING_ROOT / "watcher.db") if Paths.PROCESSING_ROOT else None
        self.files: Dict[str, Dict] = {}

    def scan(self) -> Dict[str, Dict]:
        self.files = {}
        # Gate on existence BEFORE constructing WatcherDB (constructing creates it).
        if not self.processing_root or not self.db_path or not self.db_path.exists():
            return self.files
        try:
            from mousereach.watcher.db import WatcherDB, VIDEO_TRANSITIONS
            db = WatcherDB(self.db_path)
            for state in VIDEO_TRANSITIONS.keys():   # read keys so new states appear automatically
                for row in db.get_videos_in_state(state):
                    vid = row.get("video_id")
                    if vid:
                        self.files[vid] = self._row_to_dashboard(dict(row))
        except Exception:
            pass  # fall back handled by caller / silent per _safe_refresh contract
        return self.files

    def _ensure_version_data(self, force: bool = False):
        """Cache the shipped versions + the (slow, archive-wide) manifest index so
        the Version column can judge each video. Built on demand (the 'Check
        versions' button) rather than on every refresh, since the archive rglob is
        slow over the NAS."""
        if force or not hasattr(self, "_current_versions"):
            try:
                from mousereach.pipeline.versions import get_current_versions
                from .version_currency import build_manifest_index
                self._current_versions = get_current_versions()
                self._manifest_index = build_manifest_index(
                    [Paths.PROCESSING, Paths.ANALYZED_OUTPUT]
                )
            except Exception:
                self._current_versions = {}
                self._manifest_index = {}

    def version_status_for(self, video_id: str) -> str:
        """'current' / 'outdated' / 'unknown' -- 'unknown' until the version index
        is built (Check versions)."""
        from .version_currency import version_status
        return version_status(
            video_id, getattr(self, "_manifest_index", {}),
            getattr(self, "_current_versions", None),
        )

    def dlc_model_for(self, video_id: str):
        """The DLC scorer/model from the video's manifest (None until the version
        index is built via Check versions)."""
        mp = getattr(self, "_manifest_index", {}).get(video_id)
        if not mp:
            return None
        try:
            m = json.loads(Path(mp).read_text(encoding="utf-8"))
            return m.get("dlc_model", {}).get("dlc_scorer") or None
        except Exception:
            return None

    def _status_bucket(self, state: str) -> str:
        if state in self._VALIDATED:
            return "validated"
        if state in self._NEEDS_REVIEW:
            return "needs_review"
        if state in self._FAILED:
            return "failed"
        return "in_progress"

    def _row_to_dashboard(self, row: Dict) -> Dict:
        state = row.get("state", "unknown")
        timestamps = {}
        for col in ("discovered_at", "validated_at", "crop_completed_at",
                    "dlc_completed_at", "processing_completed_at", "archived_at", "updated_at"):
            v = row.get(col)
            if v:
                timestamps[col] = str(v)
        locations = []
        cp = row.get("current_path")
        if cp:
            locations.append({"stage": state, "path": str(cp)})
        tray_type = row.get("tray_type")
        seg_status, reach_status, outcome_status, versions, gts = self._hydrate(row.get("video_id"))
        return {
            "locations": locations,
            "versions": versions,
            "timestamps": timestamps,
            "ground_truths": gts,
            "status": self._status_bucket(state),   # health bucket -> status-column color
            "current_stage": state,                  # exact watcher state -> stage column
            "metadata": {
                "state": state,
                "error": row.get("error_message"),
                "reprocess_scope": row.get("reprocess_scope"),
                "animal_id": row.get("animal_id"),
            },
            "seg_status": seg_status,
            "reach_status": reach_status,
            "outcome_status": outcome_status,
            "archive_ready": state in self._VALIDATED,
            "review_status": self._review_status(state, row.get("video_id")),
            "tray_type": tray_type,
            "tray_supported": (tray_type not in ("E", "F")) if tray_type else True,
        }

    def _review_status(self, state, video_id):
        """The video's human-review state for the Review column:
          'triage'/'deep_review' -> a hold waiting on a reviewer;
          'reviewed'             -> a saved *_causal_review.json is co-located;
          'none'                 -> nothing to review."""
        if state == "deep_review":
            return "deep_review"
        if state == "triage":
            return "triage"
        proc = Paths.PROCESSING
        if video_id and proc and (proc / f"{video_id}_causal_review.json").exists():
            return "reviewed"
        return "none"

    def _hydrate(self, video_id):
        """Fill per-step validation + versions + GT from the per-video JSONs in
        Paths.PROCESSING when co-located (the watcher row does not carry them);
        default to 'pending' otherwise."""
        seg = reach = outcome = "pending"
        versions: Dict[str, str] = {}
        gts: List[str] = []
        proc = Paths.PROCESSING
        if not video_id or not proc or not proc.exists():
            return seg, reach, outcome, versions, gts

        def _read(suffix):
            p = proc / f"{video_id}{suffix}"
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return None
            return None

        sd = _read("_segments.json")
        if sd is not None:
            seg = sd.get("validation_status", "pending")
            if sd.get("segmenter_version"):
                versions["segmenter"] = sd["segmenter_version"]
        rd = _read("_reaches.json")
        if rd is not None:
            reach = rd.get("validation_status", "pending")
        od = _read("_pellet_outcomes.json")
        if od is not None:
            outcome = od.get("validation_status", "pending")
            if od.get("detector_version"):
                versions["outcome_detector"] = od["detector_version"]
        if (proc / f"{video_id}_unified_ground_truth.json").exists():
            gts.append(f"{video_id}_unified_ground_truth.json")
        return seg, reach, outcome, versions, gts

    def get_file_summary(self, filename: str) -> str:
        info = self.files.get(filename)
        if not info:
            return "Not found in watcher DB"
        lines = [f"Video: {filename}", f"Stage: {stage_label(info['current_stage'])[0]}  ({info['current_stage']})"]
        if info["locations"]:
            lines.append(f"Path: {info['locations'][0]['path']}")
        err = info.get("metadata", {}).get("error")
        if err:
            lines.append(f"Error: {err}")
        if info["ground_truths"]:
            lines.append("Ground truth: present")
        return "\n".join(lines)

    def rebuild_index(self, progress_callback=None):
        # The watcher DB is maintained by the daemon; there is no folder index to rebuild.
        return self.files


class FolderScanAdapter:
    """Dashboard source that scans the WHOLE MouseReach_Pipeline tree and reports
    every video by the folder it physically sits in. The scan is slow (the
    final-output tree), so refresh_scan() runs on a background thread; scan()
    returns the cached result."""

    def __init__(self):
        self.processing_root = Paths.PROCESSING_ROOT
        self.files: Dict[str, Dict] = {}

    def scan(self) -> Dict[str, Dict]:
        return self.files  # cache -- populated by refresh_scan() on a worker thread

    def refresh_scan(self, progress=None) -> Dict[str, Dict]:
        from .folder_scan import scan_pipeline_folders
        self.files = scan_pipeline_folders(progress=progress)
        return self.files

    def _ensure_version_data(self, force: bool = False):
        if force or not hasattr(self, "_current_versions"):
            try:
                from mousereach.pipeline.versions import get_current_versions
                from .version_currency import build_manifest_index
                self._current_versions = get_current_versions()
                self._manifest_index = build_manifest_index([Paths.PROCESSING, Paths.ANALYZED_OUTPUT])
            except Exception:
                self._current_versions = {}
                self._manifest_index = {}

    def version_status_for(self, video_id: str) -> str:
        from .version_currency import version_status
        return version_status(video_id, getattr(self, "_manifest_index", {}),
                              getattr(self, "_current_versions", None))

    def dlc_model_for(self, video_id: str):
        """The DLC scorer/model from the video's manifest (None until Check versions)."""
        mp = getattr(self, "_manifest_index", {}).get(video_id)
        if not mp:
            return None
        try:
            m = json.loads(Path(mp).read_text(encoding="utf-8"))
            return m.get("dlc_model", {}).get("dlc_scorer") or None
        except Exception:
            return None

    def get_file_summary(self, filename: str) -> str:
        info = self.files.get(filename)
        if not info:
            return "Not found"
        lines = [f"Video: {filename}",
                 f"Stage: {stage_label(info['current_stage'])[0]}  ({info['current_stage']})"]
        if info["locations"]:
            lines.append(f"Location: {info['locations'][0]['path']}")
        return "\n".join(lines)

    def rebuild_index(self, progress_callback=None):
        return self.files


def _make_dashboard_adapter():
    """Pick the dashboard's state source. Prefer the full folder scan (every video
    by physical location); fall back to the watcher DB, then the PipelineIndex."""
    try:
        if Paths.ANALYZED_OUTPUT or Paths.PROCESSING_ROOT:
            return FolderScanAdapter()
    except Exception:
        pass
    try:
        root = Paths.PROCESSING_ROOT
        db_path = (root / "watcher.db") if root else None
        if db_path and db_path.exists():
            return WatcherAdapter()
    except Exception:
        pass
    return IndexAdapter()


class PipelineDashboard(QWidget):
    """Dashboard showing all files in the pipeline."""

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        # Authoritative state source: the watcher DB when it exists, else the
        # fast folder-derived PipelineIndex (graceful fallback).
        self.adapter = _make_dashboard_adapter()
        self.all_files = {}

        self._build_ui()

        # Auto-load from index on startup (index read is fast - single file)
        # Use QTimer to load after UI renders to avoid blocking
        from qtpy.QtCore import QTimer
        QTimer.singleShot(100, self._safe_refresh)

    def _safe_refresh(self):
        """On load: kick off the folder scan (background) for the folder-scan
        adapter, else the fast index/DB refresh."""
        try:
            if isinstance(self.adapter, FolderScanAdapter):
                self._start_folder_scan()
            elif self.adapter.processing_root.exists():
                self._refresh_data()
        except Exception:
            pass

    def _start_folder_scan(self):
        """Scan the whole pipeline tree on a background thread + render when done."""
        if getattr(self, "_scan_thread", None) and self._scan_thread.is_alive():
            show_info("Pipeline scan already running...")
            return
        show_info("Scanning the pipeline folders (~30-60s the first time)...")
        worker = _FolderScanWorker()
        worker.progress.connect(lambda m: show_info(m))
        worker.done.connect(self._on_folder_scan_done)
        self._scan_worker = worker

        import threading

        def job():
            try:
                self.adapter.refresh_scan(progress=lambda m: worker.progress.emit(m))
            except Exception as e:
                worker.progress.emit(f"Scan error: {e}")
            worker.done.emit()

        self._scan_thread = threading.Thread(target=job, daemon=True, name="folder-scan")
        self._scan_thread.start()

    def _on_folder_scan_done(self):
        self._refresh_data()
        show_info(f"Pipeline scan complete: {len(self.all_files)} videos.")

    def _refresh_clicked(self):
        """Refresh button: re-scan the folders (folder adapter) or reload."""
        if isinstance(self.adapter, FolderScanAdapter):
            self._start_folder_scan()
        else:
            self._refresh_data()  # Silently skip if network is unavailable

    def _build_ui(self):
        """Build the dashboard UI."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(main_layout)

        # === Header ===
        header = QLabel("<b>Pipeline Dashboard</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        main_layout.addWidget(header)

        info = QLabel(
            f"Data folder: {self.adapter.processing_root}\n"
            "Lists the videos the auto-processor is tracking. Videos already in the "
            "archive will NOT show until you click 'Import archive' (bottom). To "
            "change where the pipeline lives or looks for files, use the 'Change "
            "folders' button below (or the Setup tab)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 10px;")
        main_layout.addWidget(info)

        # === Tabs ===
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # --- Tab 1: Pipeline Overview ---
        overview_widget = self._build_overview_tab()
        tabs.addTab(overview_widget, "Pipeline Overview")

        # --- Tab 2: File Details ---
        details_widget = self._build_details_tab()
        tabs.addTab(details_widget, "File Details")

        # --- Tab 3: Statistics ---
        stats_widget = self._build_stats_tab()
        tabs.addTab(stats_widget, "Statistics")

    def _build_overview_tab(self) -> QWidget:
        """Build pipeline overview tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # === Filter Row ===
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by stage:"))

        self.stage_filter = QComboBox()
        self.stage_filter.addItems([label for label, _ in STAGE_FILTERS])
        self.stage_filter.setToolTip("Filter which files to show")
        self.stage_filter.currentTextChanged.connect(self._update_overview_table)
        filter_layout.addWidget(self.stage_filter)

        filter_layout.addStretch()

        # Quick filter checkboxes
        self.show_needs_review = QCheckBox("Needs Review Only")
        self.show_needs_review.setToolTip("Show only files that need human review")
        self.show_needs_review.stateChanged.connect(self._update_overview_table)
        filter_layout.addWidget(self.show_needs_review)

        layout.addLayout(filter_layout)

        # === Icon Legend ===
        legend = QLabel(
            '<span style="background-color: #4CAF50; color: white; padding: 2px 6px;">✓</span> Validated  '
            '<span style="background-color: #FFA500; padding: 2px 6px;">⏳</span> Needs Review  '
            '<span style="background-color: #ADD8E6; padding: 2px 6px;">⚡</span> Auto-approved  '
            '<span style="color: #888;">-</span> Pending  '
            '| GT: <b>S</b>=Seg <b>R</b>=Reach <b>O</b>=Outcome  '
            '<i>(Click status icons to launch review tool)</i>'
        )
        legend.setStyleSheet("font-size: 10px; padding: 3px;")
        layout.addWidget(legend)

        # Table showing all files and their validation status
        self.overview_table = QTableWidget()
        self.overview_table.setColumnCount(12)
        self.overview_table.setHorizontalHeaderLabels([
            "File", "Stage", "Tray", "DLC", "Seg", "Reach", "Outcome", "Archive", "GT", "Review", "Version", "Updated"
        ])
        self.overview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 9):
            self.overview_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)

        # Enable clicking on status cells to launch review tools
        self.overview_table.itemClicked.connect(self._on_status_cell_clicked)
        layout.addWidget(self.overview_table)

        # Buttons row
        btn_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Reload from index (fast)")
        refresh_btn.clicked.connect(self._refresh_clicked)
        btn_layout.addWidget(refresh_btn)

        rebuild_btn = QPushButton("Rebuild Index")
        rebuild_btn.setToolTip("Full rescan of all folders (slow, but thorough)")
        rebuild_btn.clicked.connect(self._rebuild_index)
        btn_layout.addWidget(rebuild_btn)

        check_versions_btn = QPushButton("Check versions")
        check_versions_btn.setToolTip(
            "Fill the Version column: compare each video against the shipped "
            "algorithm versions (Current / Outdated). Scans the archive -- may "
            "take a moment the first time."
        )
        check_versions_btn.clicked.connect(self._check_versions)
        btn_layout.addWidget(check_versions_btn)

        reprocess_btn = QPushButton("Reprocess outdated")
        reprocess_btn.setStyleSheet("background:#c80; color:white; font-weight:bold;")
        reprocess_btn.setToolTip(
            "Mark every out-of-date video for reprocessing so the watcher brings "
            "them current with the shipped algorithm versions."
        )
        reprocess_btn.clicked.connect(self._reprocess_outdated)
        btn_layout.addWidget(reprocess_btn)

        import_btn = QPushButton("Import archive")
        import_btn.setToolTip(
            "One-time: register every video already in the archive into the "
            "dashboard's database, so the whole corpus shows here and can be "
            "version-checked + reprocessed. Safe to re-run."
        )
        import_btn.clicked.connect(self._import_archive)
        btn_layout.addWidget(import_btn)

        setup_btn = QPushButton("Change folders")
        setup_btn.setToolTip("Set or change where the pipeline lives and looks for files.")
        setup_btn.clicked.connect(self._open_setup)
        btn_layout.addWidget(setup_btn)

        layout.addLayout(btn_layout)

        widget.setLayout(layout)
        return widget

    def _build_details_tab(self) -> QWidget:
        """Build file details tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # File selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select file:"))
        self.file_combo = QComboBox()
        self.file_combo.currentTextChanged.connect(self._on_file_selected)
        selector_layout.addWidget(self.file_combo)
        layout.addLayout(selector_layout)

        # Details text
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)

        widget.setLayout(layout)
        return widget

    def _build_stats_tab(self) -> QWidget:
        """Build statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Statistics labels
        self.stats_text = QLabel("")
        self.stats_text.setWordWrap(True)
        self.stats_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.stats_text)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _refresh_data(self):
        """Refresh data from index (fast)."""
        self.all_files = self.adapter.scan()
        self._update_overview_table()
        self._update_file_combo()
        self._update_statistics()

    def _rebuild_index(self):
        """Rebuild the pipeline index from scratch (folder-index mode only)."""
        # In watcher-DB mode there is no folder index to rebuild -- the daemon
        # maintains the database. Just refresh (and point at Import archive).
        if isinstance(self.adapter, WatcherAdapter):
            show_info("This dashboard reads the live tracking database -- there is "
                      "no folder index to rebuild. Use Refresh to reload, or "
                      "'Import archive' to add older videos.")
            self._refresh_data()
            return

        show_info("Rebuilding index... This may take a moment.")

        def progress(current, total, message):
            # Could update a progress bar here
            pass

        try:
            stats = self.adapter.rebuild_index(progress)
            n = stats.get('videos_found', 0) if isinstance(stats, dict) else 0
            folders = stats.get('folders_scanned', 0) if isinstance(stats, dict) else 0
            show_info(f"Index rebuilt: {n} videos in {folders} folders")
            self._refresh_data()
        except Exception as e:
            show_error(f"Failed to rebuild index: {e}")

    def _check_versions(self):
        """Fill the Version column by comparing each video against the shipped
        algorithm versions. The manifest scan is slow (archive-wide, ~30s), so it
        runs on a background thread and the column updates when it finishes."""
        adapter = self.adapter
        if not hasattr(adapter, "_ensure_version_data"):
            show_info("Version check needs the watcher database (not available here).")
            return
        if getattr(self, "_version_thread", None) and self._version_thread.is_alive():
            show_info("Version check already running...")
            return
        show_info("Checking versions -- scanning the archive (~30s)...")
        worker = _VersionCheckWorker()
        worker.done.connect(self._on_versions_ready)
        self._version_worker = worker  # keep a reference

        import threading

        def job():
            try:
                adapter._ensure_version_data(force=True)
            except Exception:
                pass
            worker.done.emit()

        self._version_thread = threading.Thread(target=job, daemon=True, name="version-check")
        self._version_thread.start()

    def _on_versions_ready(self):
        self._update_overview_table()
        try:
            statuses = [self.adapter.version_status_for(f) for f in self.all_files]
            show_info(
                f"Versions: {statuses.count('current')} current, "
                f"{statuses.count('outdated')} outdated, "
                f"{statuses.count('unknown')} unknown."
            )
        except Exception:
            pass

    def _reprocess_outdated(self):
        """Mark every out-of-date video for reprocessing (the watcher then brings
        them current). The scan reads every archived manifest (~30s over the NAS),
        so it runs on a background thread."""
        if getattr(self, "_reproc_thread", None) and self._reproc_thread.is_alive():
            show_info("Reprocess check already running...")
            return
        try:
            from mousereach.watcher.db import WatcherDB
            db_path = (Paths.PROCESSING_ROOT / "watcher.db") if Paths.PROCESSING_ROOT else None
            if not db_path or not db_path.exists():
                show_error("No watcher database found on this machine.")
                return
            db = WatcherDB(db_path)
        except Exception as e:
            show_error(f"Could not open the database: {e}")
            return
        show_info("Finding out-of-date videos (scanning the archive, ~30s)...")
        worker = _BackfillWorker()  # reuse: done(dict)
        worker.done.connect(self._on_reprocess_done)
        self._reproc_worker = worker

        import threading

        def job():
            try:
                from mousereach.watcher.reprocessor import ReprocessingScanner
                res = ReprocessingScanner(db, Paths.NAS_ROOT).scan(mark_outdated=True)
            except Exception as e:
                res = {"error": str(e)}
            worker.done.emit(res)

        self._reproc_thread = threading.Thread(target=job, daemon=True, name="reprocess-scan")
        self._reproc_thread.start()

    def _on_reprocess_done(self, res: dict):
        if res.get("error"):
            show_error(f"Could not mark outdated videos: {res['error']}")
            return
        n = res.get("outdated", 0)
        if n:
            show_info(f"Marked {n} video(s) for reprocessing. Start the watcher "
                      f"(Watcher Control tab) to bring them current.")
        else:
            show_info("No archived videos are out of date -- nothing to reprocess.")
        self._refresh_data()

    def _import_archive(self):
        """Register every archived video into the tracking DB (background) so the
        dashboard shows the full corpus + reprocess reaches it. Idempotent."""
        if getattr(self, "_backfill_thread", None) and self._backfill_thread.is_alive():
            show_info("Import already running...")
            return
        try:
            from mousereach.watcher.db import WatcherDB
            db_path = (Paths.PROCESSING_ROOT / "watcher.db") if Paths.PROCESSING_ROOT else None
            if not db_path:
                show_error("Processing root is not configured.")
                return
            db = WatcherDB(db_path)
        except Exception as e:
            show_error(f"Could not open the database: {e}")
            return
        show_info("Importing the archive into the database -- scanning (~30s)...")
        worker = _BackfillWorker()
        worker.done.connect(self._on_backfill_done)
        self._backfill_worker = worker

        import threading

        def job():
            try:
                from mousereach.watcher.backfill import backfill_archive
                res = backfill_archive(db, Paths.ANALYZED_OUTPUT)
            except Exception as e:
                res = {"error": str(e)}
            worker.done.emit(res)

        self._backfill_thread = threading.Thread(target=job, daemon=True, name="archive-backfill")
        self._backfill_thread.start()

    def _on_backfill_done(self, res: dict):
        if res.get("error"):
            show_error(f"Import failed: {res['error']}")
            return
        show_info(
            f"Imported archive: {res.get('new', 0)} newly added, "
            f"{res.get('existing', 0)} already known, {res.get('errors', 0)} errors."
        )
        self._refresh_data()

    def _open_setup(self):
        """Open the folder-setup form (also available as the Setup tab) so paths
        can be set/changed right from the dashboard."""
        try:
            from mousereach.setup.setup_widget import SetupWidget
            self._setup_window = SetupWidget(self.viewer)  # keep a ref so it isn't GC'd
            self._setup_window.setWindowTitle("MouseReach -- Folders / Setup")
            self._setup_window.resize(600, 380)
            self._setup_window.show()
        except Exception as e:
            show_error(f"Could not open Setup: {e}")

    def _update_overview_table(self):
        """Update the overview table with validation status columns."""
        # Filter out unsupported tray types (F) - they shouldn't clutter the view
        filtered_files = {
            k: v for k, v in self.all_files.items()
            if v.get("tray_supported", True)
        }

        # Apply stage filter (plain-language groups -> sets of raw states)
        stage_filter = self.stage_filter.currentText() if hasattr(self, 'stage_filter') else "All videos"
        _states = _STAGE_FILTER_MAP.get(stage_filter)
        if _states is not None:
            filtered_files = {k: v for k, v in filtered_files.items()
                              if v.get("current_stage") in _states}

        # Apply "needs review" filter
        if hasattr(self, 'show_needs_review') and self.show_needs_review.isChecked():
            filtered_files = {k: v for k, v in filtered_files.items()
                            if v.get("seg_status") == "needs_review"
                            or v.get("reach_status") == "needs_review"
                            or v.get("outcome_status") == "needs_review"
                            or v.get("status") == "needs_review"
                            or "NeedsReview" in v.get("current_stage", "")}

        self.overview_table.setRowCount(len(filtered_files))

        def status_item(status: str, step: str = "") -> QTableWidgetItem:
            """Create a color-coded status item."""
            # Use emoji for compact display
            if status == "validated":
                item = QTableWidgetItem("✓")
                item.setBackground(QBrush(QColor(76, 175, 80)))  # Green
                item.setForeground(QBrush(QColor(255, 255, 255)))
                item.setToolTip(f"{status} - Review complete")
            elif status == "needs_review":
                item = QTableWidgetItem("⏳")
                item.setBackground(QBrush(QColor(255, 165, 0)))  # Orange
                item.setToolTip(f"{status} - Click to open {step} review tool")
            elif status == "auto_approved":
                item = QTableWidgetItem("⚡")
                item.setBackground(QBrush(QColor(173, 216, 230)))  # Light blue
                item.setToolTip(f"{status} - Click to verify in {step} review tool")
            elif status == "pending":
                item = QTableWidgetItem("-")
                item.setForeground(QBrush(QColor(128, 128, 128)))  # Gray
                item.setToolTip(f"Not yet processed")
            else:
                item = QTableWidgetItem("?")
                item.setToolTip(status)
            item.setTextAlignment(Qt.AlignCenter)
            return item

        for row, (filename, info) in enumerate(sorted(filtered_files.items())):
            # File name
            name_item = QTableWidgetItem(filename)
            self.overview_table.setItem(row, 0, name_item)

            # Current stage -- plain-language label + explanation tooltip
            current_stage = info.get("current_stage", "Unknown")
            _label, _tip, _cat = stage_label(current_stage)
            # A collage still sitting in the intake folder may already be cropped:
            # its offspring exist downstream. Tell the truth from the rollup rather
            # than a blind "needs cropping".
            _crop = info.get("crop_state")
            if current_stage == "raw_collage" and _crop:
                _n = info.get("offspring_present", 0)
                _m = info.get("offspring_expected", 0)
                if _crop == "cropped":
                    _label = f"Cropped ({_n}/{_m} offspring in pipeline)"
                    _tip = ("This collage has already been cropped -- all its "
                            "single-mouse videos exist downstream. Its file still "
                            "sits in the intake folder (never moved out).")
                    _cat = "done"
                elif _crop == "partial":
                    _label = f"Partly cropped ({_n}/{_m} offspring)"
                    _tip = (f"{_n} of {_m} single-mouse videos exist downstream; "
                            "the rest are missing -- cropping is incomplete.")
                    _cat = "act"
            stage_item = QTableWidgetItem(_label)
            stage_item.setToolTip(f"{_tip}  (state: {current_stage})")
            _sc = _STAGE_CATEGORY_COLOR.get(_cat)
            if _sc is not None:
                stage_item.setForeground(QBrush(_sc))
            self.overview_table.setItem(row, 1, stage_item)

            # Tray type (P/E/F) - highlight unsupported in red
            tray_type = info.get("tray_type", "?")
            tray_supported = info.get("tray_supported", True)
            tray_item = QTableWidgetItem(tray_type if tray_type else "?")
            tray_item.setTextAlignment(Qt.AlignCenter)
            if not tray_supported:
                tray_item.setBackground(QBrush(QColor(244, 67, 54)))  # Red
                tray_item.setForeground(QBrush(QColor(255, 255, 255)))
                tray_item.setToolTip("UNSUPPORTED - Use mousereach-reject-tray to move")
            else:
                tray_item.setToolTip("Pillar tray (supported)")
            self.overview_table.setItem(row, 2, tray_item)

            # DLC (pose tracking) status -- done + which model in the tooltip
            dlc_item = status_item(info.get("dlc_status", "pending"), "DLC")
            if info.get("dlc_status") == "validated":
                dlc_model = getattr(self.adapter, "dlc_model_for", lambda v: None)(filename)
                dlc_item.setToolTip(
                    f"Pose tracking (DLC) done.  Model: {dlc_model}" if dlc_model
                    else "Pose tracking (DLC) done.  (Run 'Check versions' to show the model.)"
                )
            self.overview_table.setItem(row, 3, dlc_item)

            # Seg status (color-coded, clickable to launch review)
            seg_item = status_item(info.get("seg_status", "pending"), "Segmentation")
            seg_item.setData(Qt.UserRole, filename)  # Store video_id for click handler
            seg_item.setData(Qt.UserRole + 1, "seg")  # Store step type
            self.overview_table.setItem(row, 4, seg_item)

            # Reach status (color-coded, clickable to launch review)
            reach_item = status_item(info.get("reach_status", "pending"), "Reach")
            reach_item.setData(Qt.UserRole, filename)  # Store video_id
            reach_item.setData(Qt.UserRole + 1, "reach")  # Store step type
            self.overview_table.setItem(row, 5, reach_item)

            # Outcome status (color-coded, clickable to launch review)
            outcome_item = status_item(info.get("outcome_status", "pending"), "Outcome")
            outcome_item.setData(Qt.UserRole, filename)  # Store video_id
            outcome_item.setData(Qt.UserRole + 1, "outcome")  # Store step type
            self.overview_table.setItem(row, 6, outcome_item)

            # Archive ready
            archive_ready = info.get("archive_ready", False)
            archive_item = QTableWidgetItem("Y" if archive_ready else "-")
            if archive_ready:
                archive_item.setBackground(QBrush(QColor(33, 150, 243)))  # Blue
                archive_item.setForeground(QBrush(QColor(255, 255, 255)))
            archive_item.setTextAlignment(Qt.AlignCenter)
            archive_item.setToolTip("Ready for archive" if archive_ready else "Not ready")
            self.overview_table.setItem(row, 7, archive_item)

            # Ground truth - show which types exist (S=seg, R=reach, O=outcome)
            gt_types = []
            for gt_file in info["ground_truths"]:
                if "_seg_ground_truth" in gt_file:
                    gt_types.append("S")
                elif "_reach_ground_truth" in gt_file:
                    gt_types.append("R")
                elif "_outcome_ground_truth" in gt_file:
                    gt_types.append("O")
            gt_str = " ".join(gt_types) if gt_types else "-"
            gt_item = QTableWidgetItem(gt_str)
            gt_item.setTextAlignment(Qt.AlignCenter)
            if gt_types:
                gt_item.setToolTip(f"Ground truth: {', '.join(info['ground_truths'])}")
            self.overview_table.setItem(row, 8, gt_item)

            # Review status: triage / deep-review holds, or a saved human review
            rs = info.get("review_status", "none")
            if rs == "triage":
                review_item = QTableWidgetItem("Triage")
                review_item.setBackground(QBrush(QColor(255, 165, 0)))  # Orange
                review_item.setToolTip("Held -- waiting on triage review")
            elif rs == "deep_review":
                review_item = QTableWidgetItem("Deep")
                review_item.setBackground(QBrush(QColor(244, 67, 54)))  # Red
                review_item.setForeground(QBrush(QColor(255, 255, 255)))
                review_item.setToolTip("Held -- waiting on deep review (seg-failed / escalated)")
            elif rs == "reviewed":
                review_item = QTableWidgetItem("✓")
                review_item.setBackground(QBrush(QColor(76, 175, 80)))  # Green
                review_item.setForeground(QBrush(QColor(255, 255, 255)))
                review_item.setToolTip("Human review saved")
            else:
                review_item = QTableWidgetItem("-")
                review_item.setForeground(QBrush(QColor(128, 128, 128)))
                review_item.setToolTip("Nothing to review")
            review_item.setTextAlignment(Qt.AlignCenter)
            self.overview_table.setItem(row, 9, review_item)

            # Version currency vs the shipped algo versions (filled by Check versions)
            vs = getattr(self.adapter, "version_status_for", lambda v: "unknown")(filename)
            if vs == "current":
                version_item = QTableWidgetItem("Current")
                version_item.setBackground(QBrush(QColor(76, 175, 80)))  # Green
                version_item.setForeground(QBrush(QColor(255, 255, 255)))
                version_item.setToolTip("Up to date with the shipped algorithm versions")
            elif vs == "outdated":
                version_item = QTableWidgetItem("Outdated")
                version_item.setBackground(QBrush(QColor(255, 165, 0)))  # Orange
                version_item.setToolTip("Older than the shipped versions -- reprocess to bring it current")
            else:
                version_item = QTableWidgetItem("?")
                version_item.setForeground(QBrush(QColor(128, 128, 128)))
                version_item.setToolTip("Click 'Check versions' to compare against the shipped versions")
            version_item.setTextAlignment(Qt.AlignCenter)
            self.overview_table.setItem(row, 10, version_item)

            # Last update (date only, compact)
            if info["timestamps"]:
                last_time = max(info["timestamps"].values())
                # Extract just the date part (YYYY-MM-DD)
                last_item = QTableWidgetItem(last_time[:10])
            else:
                last_item = QTableWidgetItem("-")
            self.overview_table.setItem(row, 11, last_item)

    def _update_file_combo(self):
        """Update file selector combo box."""
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        self.file_combo.addItems(sorted(self.all_files.keys()))
        self.file_combo.blockSignals(False)

        if len(self.all_files) > 0:
            self._on_file_selected(list(self.all_files.keys())[0])

    def _on_file_selected(self, filename: str):
        """Handle file selection."""
        if filename and filename in self.all_files:
            summary = self.adapter.get_file_summary(filename)
            self.details_text.setText(summary)

    def _update_statistics(self):
        """Update statistics: a meaningful by-stage / review / version breakdown
        across the whole shown corpus (the old per-step validation counts only
        applied to videos currently sitting in Processing)."""
        from collections import Counter
        total = len(self.all_files)
        state_counts = Counter(info.get("current_stage", "unknown") for info in self.all_files.values())
        review_counts = Counter(info.get("review_status", "none") for info in self.all_files.values())
        has_gt = sum(1 for info in self.all_files.values() if info.get("ground_truths"))

        lines = [f"Total videos: {total}", "", "By stage:"]
        for state, n in sorted(state_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {stage_label(state)[0]:24} {n}")

        lines += [
            "",
            "Review holds:",
            f"  Needs triage review:   {review_counts.get('triage', 0)}",
            f"  Needs deep review:     {review_counts.get('deep_review', 0)}",
            f"  Human review saved:    {review_counts.get('reviewed', 0)}",
            "",
            f"Ground truth files:      {has_gt}/{total}",
        ]

        # Version currency -- only after 'Check versions' has built the index.
        vs_fn = getattr(self.adapter, "version_status_for", None)
        if vs_fn is not None and getattr(self.adapter, "_manifest_index", None):
            vc = Counter(vs_fn(f) for f in self.all_files)
            lines += [
                "",
                "Versions vs the shipped set:",
                f"  Current:   {vc.get('current', 0)}",
                f"  Outdated:  {vc.get('outdated', 0)}",
                f"  Unknown:   {vc.get('unknown', 0)}",
            ]

        self.stats_text.setText("\n".join(lines))

    def _on_status_cell_clicked(self, item: QTableWidgetItem):
        """Click a status cell: the Review cell opens the triage / deep review tool
        for that video (landing on its first held element); Seg/Reach/Outcome open
        the per-step review."""
        col = item.column()
        row = item.row()

        # Review column (9): open the review tool on this video's first held element
        if col == 9:
            name_item = self.overview_table.item(row, 0)
            if not name_item:
                return
            video_id = name_item.text()
            rtext = item.text()
            if rtext == "Triage":
                self._open_review_for_video(video_id, deep=False)
            elif rtext == "Deep":
                self._open_review_for_video(video_id, deep=True)
            return

        # Seg / Reach / Outcome columns (4, 5, 6): the per-step review launcher
        if col not in [4, 5, 6]:
            return
        status = item.toolTip()
        if status not in ["needs_review", "auto_approved"]:
            return
        video_id = item.data(Qt.UserRole)
        step = item.data(Qt.UserRole + 1)
        if not video_id or not step:
            return
        issue_location = self._find_first_issue(video_id, step)
        self._launch_review(video_id, step, jump_to=issue_location)

    def _open_review_for_video(self, video_id: str, deep: bool = False):
        """Open the triage (or deep) review tool on ONE video's bundle -- it lands
        on the first held/triaged element (triage_only walks triaged first)."""
        from mousereach.config import Paths
        root = Paths.DEEP_REVIEW if deep else Paths.TRIAGE_REVIEW
        bundle = (root / video_id) if root else None
        if not bundle or not bundle.exists():
            show_info(f"No {'deep-review' if deep else 'triage'} bundle found for {video_id}. "
                      f"(Use the Review Queues tab for the full queue.)")
            return
        try:
            import napari
            from mousereach.review.causal_review_io import bundle_manifest_path
            from mousereach.review.causal_review_widget import CausalReviewWidget
            mp = bundle_manifest_path(bundle, video_id)
            if not mp.exists():
                show_info(f"No review bundle manifest for {video_id}.")
                return
            manifest = json.loads(mp.read_text(encoding="utf-8"))
            title = f"{'Deep' if deep else 'Triage'} Review -- {video_id}"
            v = napari.Viewer(title=title)
            w = CausalReviewWidget(v, triage_only=(not deep), deep_review=deep)
            v.window.add_dock_widget(w, name=title, area="right")
            w.load_from_manifest(manifest, bundle)
            self._review_windows = getattr(self, "_review_windows", [])
            self._review_windows.append((v, w))  # keep refs so the windows survive
        except Exception as e:
            show_error(f"Could not open review for {video_id}: {e}")

    def _find_first_issue(self, video_id: str, step: str) -> Optional[Dict]:
        """Find the first issue location for a video/step.

        Returns dict with location info (boundary_idx, segment, frame, etc.)
        or None if no specific issue found.
        """
        proc = Paths.PROCESSING

        try:
            if step == "seg":
                seg_file = proc / f"{video_id}_segments.json"
                if seg_file.exists():
                    with open(seg_file) as f:
                        data = json.load(f)
                    # Find first anomaly
                    for i, anom in enumerate(data.get("anomalies", [])):
                        if isinstance(anom, str):
                            # Parse boundary index from anomaly text
                            if "boundary" in anom.lower():
                                import re
                                match = re.search(r'boundary\s*(\d+)', anom.lower())
                                if match:
                                    return {"boundary_idx": int(match.group(1))}
                            return {"boundary_idx": i}
                    return None

            elif step == "reach":
                reach_file = proc / f"{video_id}_reaches.json"
                if reach_file.exists():
                    with open(reach_file) as f:
                        data = json.load(f)
                    # Find first segment with anomalous reach count
                    for seg in data.get("segments", []):
                        n_reaches = len(seg.get("reaches", []))
                        if n_reaches < 3 or n_reaches > 100:
                            return {"segment": seg.get("segment_num", 1)}
                    return None

            elif step == "outcome":
                outcome_file = proc / f"{video_id}_pellet_outcomes.json"
                if outcome_file.exists():
                    with open(outcome_file) as f:
                        data = json.load(f)
                    # Find first flagged or uncertain segment
                    for seg in data.get("segments", []):
                        if seg.get("flagged_for_review") or seg.get("outcome") == "uncertain":
                            return {"segment": seg.get("segment_num", 1)}
                    return None

        except Exception as e:
            print(f"Error finding issue: {e}")

        return None

    def _launch_review(self, video_id: str, step: str, jump_to: Optional[Dict] = None):
        """Launch the appropriate review tool for a video.

        Args:
            video_id: Video identifier
            step: "seg", "reach", or "outcome"
            jump_to: Optional location dict to jump to after loading
        """
        proc = Paths.PROCESSING

        # Find video file
        video_patterns = [
            proc / f"{video_id}.mp4",
            proc / f"{video_id}_preview.mp4",
        ]
        video_path = None
        for pattern in video_patterns:
            if pattern.exists():
                video_path = pattern
                break

        # Also check with DLC suffix
        if not video_path:
            dlc_videos = list(proc.glob(f"{video_id}*DLC*.mp4"))
            if dlc_videos:
                video_path = dlc_videos[0]

        if not video_path:
            show_error(f"Video file not found for {video_id}")
            return

        try:
            if step == "seg":
                from mousereach.segmentation.review_widget import SegmentationReviewWidget
                widget = SegmentationReviewWidget(self.viewer)
                widget._load_video_from_path(video_path)
                self.viewer.window.add_dock_widget(widget, name="Segmentation Review", area="right")

                # Jump to issue if specified
                location_msg = ""
                if jump_to and hasattr(widget, 'goto_boundary'):
                    boundary_idx = jump_to.get("boundary_idx", 0)
                    widget.goto_boundary(boundary_idx)
                    location_msg = f" → Jumped to boundary {boundary_idx}"

                show_info(f"Opened 'Segmentation Review' tab for {video_id}{location_msg}")

            elif step == "reach":
                from mousereach.reach.review_widget import ReachAnnotatorWidget
                widget = ReachAnnotatorWidget(self.viewer)
                widget._load_video_from_path(video_path)
                self.viewer.window.add_dock_widget(widget, name="Reach Review", area="right")

                # Jump to issue if specified
                location_msg = ""
                if jump_to and hasattr(widget, 'goto_segment'):
                    segment = jump_to.get("segment", 1)
                    widget.goto_segment(segment)
                    location_msg = f" → Jumped to segment {segment}"

                show_info(f"Opened 'Reach Review' tab for {video_id}{location_msg}")

            elif step == "outcome":
                from mousereach.outcomes.review_widget import PelletOutcomeAnnotatorWidget
                widget = PelletOutcomeAnnotatorWidget(self.viewer)
                widget._load_video_from_path(video_path)
                self.viewer.window.add_dock_widget(widget, name="Outcome Review", area="right")

                # Jump to issue if specified
                location_msg = ""
                if jump_to and hasattr(widget, 'goto_segment'):
                    segment = jump_to.get("segment", 1)
                    widget.goto_segment(segment)
                    location_msg = f" → Jumped to segment {segment}"

                show_info(f"Opened 'Outcome Review' tab for {video_id}{location_msg}")

        except Exception as e:
            show_error(f"Failed to launch review: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Launch dashboard standalone."""
    import napari
    viewer = napari.Viewer(title="MouseReach Pipeline Dashboard")
    dashboard = PipelineDashboard(viewer)
    viewer.window.add_dock_widget(dashboard, name="Dashboard", area="right")
    napari.run()


if __name__ == "__main__":
    main()
