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
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QBrush

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


class PipelineDashboard(QWidget):
    """Dashboard showing all files in the pipeline."""

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.adapter = IndexAdapter()  # Uses fast index instead of slow scanning
        self.all_files = {}

        self._build_ui()

        # Auto-load from index on startup (index read is fast - single file)
        # Use QTimer to load after UI renders to avoid blocking
        from qtpy.QtCore import QTimer
        QTimer.singleShot(100, self._safe_refresh)

    def _safe_refresh(self):
        """Refresh data only if processing root is accessible."""
        try:
            if self.adapter.processing_root.exists():
                self._refresh_data()
        except Exception:
            pass  # Silently skip if network is unavailable

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
            "Complete view of all files in the pipeline.\n"
            f"Index: {self.adapter.processing_root}\n"
            "Shows location, version, timestamps, and ground truth status."
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
        self.stage_filter.addItems([
            "All (except DLC_Queue)",  # Default - hide unanalyzed
            "All stages",
            "DLC_Queue",
            "Processing",
            "Failed",
        ])
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
        self.overview_table.setColumnCount(9)
        self.overview_table.setHorizontalHeaderLabels([
            "File", "Stage", "Tray", "Seg", "Reach", "Outcome", "Archive", "GT", "Updated"
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
        refresh_btn.clicked.connect(self._refresh_data)
        btn_layout.addWidget(refresh_btn)

        rebuild_btn = QPushButton("Rebuild Index")
        rebuild_btn.setToolTip("Full rescan of all folders (slow, but thorough)")
        rebuild_btn.clicked.connect(self._rebuild_index)
        btn_layout.addWidget(rebuild_btn)

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
        """Rebuild the pipeline index from scratch."""
        show_info("Rebuilding index... This may take a moment.")

        def progress(current, total, message):
            # Could update a progress bar here
            pass

        try:
            stats = self.adapter.rebuild_index(progress)
            show_info(f"Index rebuilt: {stats['videos_found']} videos in {stats['folders_scanned']} folders")
            self._refresh_data()
        except Exception as e:
            show_error(f"Failed to rebuild index: {e}")

    def _update_overview_table(self):
        """Update the overview table with validation status columns."""
        # Filter out unsupported tray types (F) - they shouldn't clutter the view
        filtered_files = {
            k: v for k, v in self.all_files.items()
            if v.get("tray_supported", True)
        }

        # Apply stage filter
        stage_filter = self.stage_filter.currentText() if hasattr(self, 'stage_filter') else "All (except DLC_Queue)"

        if stage_filter == "All (except DLC_Queue)":
            filtered_files = {k: v for k, v in filtered_files.items()
                            if v.get("current_stage") != "DLC_Queue"}
        elif stage_filter == "All stages":
            pass  # Show everything
        elif stage_filter == "Processing":
            # Show anything past DLC_Queue
            filtered_files = {k: v for k, v in filtered_files.items()
                            if v.get("current_stage") not in ["DLC_Queue", None]}
        elif stage_filter != "All stages":
            # Filter to specific stage
            filtered_files = {k: v for k, v in filtered_files.items()
                            if v.get("current_stage") == stage_filter}

        # Apply "needs review" filter
        if hasattr(self, 'show_needs_review') and self.show_needs_review.isChecked():
            filtered_files = {k: v for k, v in filtered_files.items()
                            if v.get("seg_status") == "needs_review"
                            or v.get("reach_status") == "needs_review"
                            or v.get("outcome_status") == "needs_review"
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

            # Current stage
            current_stage = info.get("current_stage", "Unknown")
            stage_item = QTableWidgetItem(current_stage)
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

            # Seg status (color-coded, clickable to launch review)
            seg_item = status_item(info.get("seg_status", "pending"), "Segmentation")
            seg_item.setData(Qt.UserRole, filename)  # Store video_id for click handler
            seg_item.setData(Qt.UserRole + 1, "seg")  # Store step type
            self.overview_table.setItem(row, 3, seg_item)

            # Reach status (color-coded, clickable to launch review)
            reach_item = status_item(info.get("reach_status", "pending"), "Reach")
            reach_item.setData(Qt.UserRole, filename)  # Store video_id
            reach_item.setData(Qt.UserRole + 1, "reach")  # Store step type
            self.overview_table.setItem(row, 4, reach_item)

            # Outcome status (color-coded, clickable to launch review)
            outcome_item = status_item(info.get("outcome_status", "pending"), "Outcome")
            outcome_item.setData(Qt.UserRole, filename)  # Store video_id
            outcome_item.setData(Qt.UserRole + 1, "outcome")  # Store step type
            self.overview_table.setItem(row, 5, outcome_item)

            # Archive ready
            archive_ready = info.get("archive_ready", False)
            archive_item = QTableWidgetItem("Y" if archive_ready else "-")
            if archive_ready:
                archive_item.setBackground(QBrush(QColor(33, 150, 243)))  # Blue
                archive_item.setForeground(QBrush(QColor(255, 255, 255)))
            archive_item.setTextAlignment(Qt.AlignCenter)
            archive_item.setToolTip("Ready for archive" if archive_ready else "Not ready")
            self.overview_table.setItem(row, 6, archive_item)

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
            self.overview_table.setItem(row, 7, gt_item)

            # Last update (date only, compact)
            if info["timestamps"]:
                last_time = max(info["timestamps"].values())
                # Extract just the date part (YYYY-MM-DD)
                last_item = QTableWidgetItem(last_time[:10])
            else:
                last_item = QTableWidgetItem("-")
            self.overview_table.setItem(row, 8, last_item)

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
        """Update statistics display with validation status breakdown."""
        total = len(self.all_files)

        # Count by validation status for each step
        seg_counts = {"validated": 0, "needs_review": 0, "auto_approved": 0, "pending": 0}
        reach_counts = {"validated": 0, "needs_review": 0, "auto_approved": 0, "pending": 0}
        outcome_counts = {"validated": 0, "needs_review": 0, "auto_approved": 0, "pending": 0}
        archive_ready = 0

        for info in self.all_files.values():
            seg_status = info.get("seg_status", "pending")
            reach_status = info.get("reach_status", "pending")
            outcome_status = info.get("outcome_status", "pending")

            seg_counts[seg_status] = seg_counts.get(seg_status, 0) + 1
            reach_counts[reach_status] = reach_counts.get(reach_status, 0) + 1
            outcome_counts[outcome_status] = outcome_counts.get(outcome_status, 0) + 1

            if info.get("archive_ready", False):
                archive_ready += 1

        has_gt = sum(1 for info in self.all_files.values() if info["ground_truths"])

        stats_lines = [
            f"Total videos: {total}",
            "",
            "Segmentation Status:",
            f"  ✓ Validated:    {seg_counts['validated']}",
            f"  ⏳ Needs Review: {seg_counts['needs_review']}",
            f"  ⚡ Auto-approved: {seg_counts['auto_approved']}",
            f"  - Pending:      {seg_counts['pending']}",
            "",
            "Reach Detection Status:",
            f"  ✓ Validated:    {reach_counts['validated']}",
            f"  ⏳ Needs Review: {reach_counts['needs_review']}",
            f"  ⚡ Auto-approved: {reach_counts['auto_approved']}",
            f"  - Pending:      {reach_counts['pending']}",
            "",
            "Outcome Detection Status:",
            f"  ✓ Validated:    {outcome_counts['validated']}",
            f"  ⏳ Needs Review: {outcome_counts['needs_review']}",
            f"  ⚡ Auto-approved: {outcome_counts['auto_approved']}",
            f"  - Pending:      {outcome_counts['pending']}",
            "",
            f"Archive Ready: {archive_ready}/{total}",
            f"Ground Truth:  {has_gt}/{total}",
        ]

        self.stats_text.setText("\n".join(stats_lines))

    def _on_status_cell_clicked(self, item: QTableWidgetItem):
        """Handle click on status cell - launch review tool if applicable."""
        col = item.column()

        # Only handle clicks on status columns (Seg=3, Reach=4, Outcome=5)
        if col not in [3, 4, 5]:
            return

        # Get status from tooltip
        status = item.toolTip()

        # Only launch for reviewable statuses
        if status not in ["needs_review", "auto_approved"]:
            return

        # Get video_id and step from cell data
        video_id = item.data(Qt.UserRole)
        step = item.data(Qt.UserRole + 1)

        if not video_id or not step:
            return

        # Find the issue location and launch review
        issue_location = self._find_first_issue(video_id, step)
        self._launch_review(video_id, step, jump_to=issue_location)

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
