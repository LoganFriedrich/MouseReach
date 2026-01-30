"""
MouseReach Unified Pipeline Widget
==============================

Single napari widget for running the complete analysis pipeline.

Philosophy: "Push everything forward"
- Click Run with no selection → process everything that's ready
- Optionally select specific files for targeted reprocessing

When you click Run:
1. New DLC files → Segmentation → auto-triage
2. Validated segmentations → Outcomes → Reaches → auto-triage
3. Validated items in *_NeedsReview → advance to next stage

Files needing review are paused but don't block others.
"""

from pathlib import Path
from typing import Optional, List
import threading

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QScrollArea, QFrame,
    QMessageBox
)
from qtpy.QtCore import Qt, Signal, QObject, QTimer

import napari
from napari.utils.notifications import show_info, show_error, show_warning

from mousereach.config import Paths
from mousereach.pipeline.core import (
    scan_pipeline_status, PipelineStatus,
    UnifiedPipelineProcessor, UnifiedResults,
    consolidate_all_to_dlc_complete, count_all_videos_in_pipeline
)


class PipelineSignals(QObject):
    """Signals for pipeline processing."""
    status_scanned = Signal(object)  # PipelineStatus
    stage_progress = Signal(str, int, int, str)  # stage_name, current, total, message
    pipeline_complete = Signal(object)  # UnifiedResults
    consolidate_complete = Signal(dict)  # consolidation stats
    error = Signal(str)


class StageWidget(QFrame):
    """Widget displaying a single pipeline stage."""

    def __init__(self, stage_name: str, description: str):
        super().__init__()
        self.stage_name = stage_name
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(layout)

        # Header
        header = QLabel(f"<b>{stage_name}</b>")
        header.setStyleSheet("font-size: 12px;")
        layout.addWidget(header)

        # Description
        desc = QLabel(description)
        desc.setStyleSheet("color: #888; font-size: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Waiting...")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        # Results summary (hidden until complete)
        self.results_label = QLabel("")
        self.results_label.setVisible(False)
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

    def set_pending(self):
        """Reset to pending state."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Waiting...")
        self.status_label.setStyleSheet("color: #666;")
        self.results_label.setVisible(False)

    def set_running(self, current: int, total: int, message: str):
        """Update progress while running."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #4a9eff;")

    def set_complete(self, summary: str, has_warnings: bool = False):
        """Show completion status."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Complete")
        if has_warnings:
            self.status_label.setStyleSheet("color: #ffa500;")
        else:
            self.status_label.setStyleSheet("color: #4caf50;")
        self.results_label.setText(summary)
        self.results_label.setVisible(True)


class UnifiedPipelineWidget(QWidget):
    """
    Unified widget for running the complete MouseReach analysis pipeline.

    Philosophy: "Push everything forward"
    - Run with no selection = process everything that's ready
    - Select specific files = targeted reprocessing
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.base_dir: Path = Paths.PROCESSING_ROOT
        self.specific_files: Optional[List[Path]] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.signals = PipelineSignals()
        self.current_status: Optional[PipelineStatus] = None

        self._build_ui()
        self._connect_signals()

        # Auto-scan on startup (deferred to avoid blocking)
        if self.base_dir.exists():
            QTimer.singleShot(200, self._scan_status)

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(main_layout)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll)

        inner_widget = QWidget()
        scroll.setWidget(inner_widget)
        layout = QVBoxLayout()
        inner_widget.setLayout(layout)

        # === Header ===
        header = QLabel("<b>MouseReach Analysis Pipeline</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header)

        # === Status Summary ===
        status_group = QGroupBox("Pipeline Status")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)

        self.status_label = QLabel("Scanning pipeline...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        status_layout.addWidget(self.status_label)

        # Refresh button
        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Status")
        self.refresh_btn.setToolTip("Re-scan pipeline folders to see what's ready")
        self.refresh_btn.clicked.connect(self._scan_status)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addStretch()
        status_layout.addLayout(btn_row)

        layout.addWidget(status_group)

        # === Run Section ===
        run_group = QGroupBox("Run Pipeline")
        run_layout = QVBoxLayout()
        run_group.setLayout(run_layout)

        instructions = QLabel(
            "Click <b>Run</b> to process everything that's ready.\n"
            "Or select specific files for targeted reprocessing."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; font-size: 10px;")
        run_layout.addWidget(instructions)

        # Run button
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 12px; font-size: 13px;")
        self.run_btn.clicked.connect(self._run_pipeline)
        self.run_btn.setEnabled(False)
        run_layout.addWidget(self.run_btn)

        # Selection row
        select_row = QHBoxLayout()

        self.select_btn = QPushButton("Select Specific Files...")
        self.select_btn.setToolTip("Choose specific files to reprocess")
        self.select_btn.clicked.connect(self._select_specific_files)
        select_row.addWidget(self.select_btn)

        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.setToolTip("Clear selection and process everything")
        self.clear_btn.clicked.connect(self._clear_selection)
        self.clear_btn.setEnabled(False)
        select_row.addWidget(self.clear_btn)

        run_layout.addLayout(select_row)

        # Selection info
        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("color: #4a9eff; font-size: 10px;")
        self.selection_label.setWordWrap(True)
        run_layout.addWidget(self.selection_label)

        # Separator
        run_layout.addSpacing(10)

        # Reprocess All button
        self.reprocess_btn = QPushButton("Reprocess All...")
        self.reprocess_btn.setToolTip(
            "Delete derived outputs and reprocess from scratch.\n"
            "This deletes segments, outcomes, and reaches files."
        )
        self.reprocess_btn.setStyleSheet("color: #ff6b6b;")
        self.reprocess_btn.clicked.connect(self._reprocess_all)
        run_layout.addWidget(self.reprocess_btn)

        layout.addWidget(run_group)

        # === Pipeline Stages ===
        stages_group = QGroupBox("Progress")
        stages_layout = QVBoxLayout()
        stages_group.setLayout(stages_layout)

        self.stage1 = StageWidget(
            "Segmentation",
            "Find pellet presentation boundaries. Good results continue automatically."
        )
        stages_layout.addWidget(self.stage1)

        self.stage2 = StageWidget(
            "Outcomes & Reaches",
            "Classify segments (R/D/M) and detect individual reach attempts."
        )
        stages_layout.addWidget(self.stage2)

        self.stage3 = StageWidget(
            "Advancing",
            "Move validated files to next pipeline stage."
        )
        stages_layout.addWidget(self.stage3)

        layout.addWidget(stages_group)

        # === Summary ===
        summary_group = QGroupBox("Results")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)

        layout.addWidget(summary_group)

        # Spacer
        layout.addStretch()

    def _connect_signals(self):
        self.signals.status_scanned.connect(self._on_status_scanned)
        self.signals.stage_progress.connect(self._on_stage_progress)
        self.signals.pipeline_complete.connect(self._on_pipeline_complete)
        self.signals.consolidate_complete.connect(self._on_consolidate_complete)
        self.signals.error.connect(self._on_error)

    def _scan_status(self):
        """Scan the pipeline to see what's ready."""
        self.status_label.setText("Scanning pipeline...")
        self.refresh_btn.setEnabled(False)

        # Run in background to avoid blocking UI
        def scan_worker():
            try:
                status = scan_pipeline_status(self.base_dir)
                self.signals.status_scanned.emit(status)
            except Exception as e:
                self.signals.error.emit(f"Scan failed: {e}")

        thread = threading.Thread(target=scan_worker)
        thread.start()

    def _on_status_scanned(self, status: PipelineStatus):
        """Handle status scan completion."""
        self.current_status = status
        self.refresh_btn.setEnabled(True)

        # Update status display
        lines = status.summary_lines()
        if status.total_ready > 0:
            lines.append(f"\n<b>Total: {status.total_ready} ready to process</b>")

        self.status_label.setText("<br>".join(lines))

        # Enable run button if there's work to do
        self.run_btn.setEnabled(status.total_ready > 0 or self.specific_files is not None)

        # Reset stages
        self.stage1.set_pending()
        self.stage2.set_pending()
        self.stage3.set_pending()
        self.summary_label.setText("")

    def _select_specific_files(self):
        """Open dialog to select specific files for reprocessing."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DLC files to process",
            str(self.base_dir),
            "DLC Files (*DLC*.h5);;All Files (*)"
        )

        if files:
            self.specific_files = [Path(f) for f in files]
            self.selection_label.setText(f"Selected: {len(self.specific_files)} file(s)")
            self.clear_btn.setEnabled(True)
            self.run_btn.setEnabled(True)

    def _clear_selection(self):
        """Clear specific file selection."""
        self.specific_files = None
        self.selection_label.setText("")
        self.clear_btn.setEnabled(False)
        # Re-enable based on pipeline status
        if self.current_status:
            self.run_btn.setEnabled(self.current_status.total_ready > 0)

    def _reprocess_all(self):
        """Consolidate all files and reprocess from scratch."""
        # Count videos first
        counts = count_all_videos_in_pipeline(self.base_dir)
        total = counts.get('total', 0)

        if total == 0:
            show_warning("No videos found in pipeline to reprocess.")
            return

        # Show confirmation dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("Reprocess All Videos")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(f"This will reprocess ALL {total} videos from scratch.")
        msg.setInformativeText(
            "All existing analysis files will be deleted:\n"
            "- Segmentation results\n"
            "- Outcome classifications\n"
            "- Reach detections\n\n"
            "DLC files will be kept and videos will be reprocessed.\n\n"
            "This cannot be undone. Continue?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        if msg.exec_() != QMessageBox.Yes:
            return

        # Disable UI
        self.run_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.reprocess_btn.setEnabled(False)

        self.summary_label.setText("Consolidating files...")
        self.stage1.set_pending()
        self.stage2.set_pending()
        self.stage3.set_pending()

        # Run consolidation in background
        self.worker_thread = threading.Thread(target=self._consolidate_worker)
        self.worker_thread.start()

    def _consolidate_worker(self):
        """Background worker for consolidation."""
        try:
            stats = consolidate_all_to_dlc_complete(self.base_dir, delete_outputs=True)
            self.signals.consolidate_complete.emit(stats)
        except Exception as e:
            self.signals.error.emit(f"Consolidation failed: {e}")

    def _on_consolidate_complete(self, stats: dict):
        """Handle consolidation completion - now run the pipeline."""
        videos_moved = stats.get('videos_moved', 0)
        derived_deleted = stats.get('derived_deleted', 0)

        self.summary_label.setText(
            f"Consolidated {videos_moved} videos, deleted {derived_deleted} derived files.\n"
            "Now reprocessing..."
        )

        # Re-enable reprocess button for next time
        self.reprocess_btn.setEnabled(True)

        # Refresh status and then run pipeline
        def after_scan():
            # Disconnect temporary handler
            try:
                self.signals.status_scanned.disconnect(after_scan_handler)
            except (TypeError, RuntimeError):
                pass  # Handler already disconnected or signal doesn't exist
            # Now run the pipeline
            self._run_pipeline()

        def after_scan_handler(status):
            after_scan()

        # Connect temporary handler and scan
        self.signals.status_scanned.connect(after_scan_handler)
        self._scan_status()

    def _run_pipeline(self):
        """Run the unified pipeline."""
        self.run_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.reprocess_btn.setEnabled(False)

        # Reset stages
        self.stage1.set_pending()
        self.stage2.set_pending()
        self.stage3.set_pending()
        self.summary_label.setText("Running...")

        self.worker_thread = threading.Thread(target=self._pipeline_worker)
        self.worker_thread.start()

    def _pipeline_worker(self):
        """Background worker for pipeline processing."""
        try:
            def progress_callback(stage: str, current: int, total: int, message: str):
                self.signals.stage_progress.emit(stage, current, total, message)

            processor = UnifiedPipelineProcessor(
                self.base_dir,
                progress_callback=progress_callback,
                specific_files=self.specific_files
            )

            results = processor.run()
            self.signals.pipeline_complete.emit(results)

        except Exception as e:
            self.signals.error.emit(str(e))

    def _on_stage_progress(self, stage: str, current: int, total: int, message: str):
        """Handle stage progress updates."""
        if stage == 'segmentation':
            self.stage1.set_running(current, total, message)
        elif stage in ('outcomes', 'reaches'):
            self.stage2.set_running(current, total, message)
        elif stage == 'advancing':
            self.stage3.set_running(current, total, message)

    def _on_pipeline_complete(self, results: UnifiedResults):
        """Handle pipeline completion."""
        self.run_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.reprocess_btn.setEnabled(True)
        if self.specific_files:
            self.clear_btn.setEnabled(True)

        # Update Stage 1 summary (Segmentation)
        if results.seg_processed > 0:
            s1_summary = f"Processed: {results.seg_processed}"
            if results.seg_auto_approved > 0:
                s1_summary += f"\nAuto-approved: {results.seg_auto_approved}"
            if results.seg_needs_review > 0:
                s1_summary += f"\nNeeds review: {results.seg_needs_review}"
            if results.seg_failed > 0:
                s1_summary += f"\nFailed: {results.seg_failed}"
            self.stage1.set_complete(s1_summary, results.seg_needs_review > 0 or results.seg_failed > 0)
        else:
            self.stage1.set_complete("No new segmentation needed", False)

        # Update Stage 2 summary (Outcomes & Reaches)
        if results.outcome_processed > 0 or results.reach_processed > 0:
            s2_lines = []
            if results.outcome_processed > 0:
                s2_lines.append(f"Outcomes: {results.outcome_completed} completed")
            if results.reach_processed > 0:
                s2_lines.append(f"Reaches: {results.reach_validated} validated")
            if results.reach_needs_review > 0:
                s2_lines.append(f"Reach review: {results.reach_needs_review}")
            self.stage2.set_complete("\n".join(s2_lines), results.reach_needs_review > 0)
        else:
            self.stage2.set_complete("No outcomes/reaches needed", False)

        # Update Stage 3 summary (Advancing)
        advanced = results.reach_advanced + results.outcome_advanced
        if advanced > 0:
            s3_summary = f"Advanced: {advanced} validated files"
            self.stage3.set_complete(s3_summary, False)
        else:
            self.stage3.set_complete("No files to advance", False)

        # Overall summary
        summary_lines = []
        if results.fully_completed > 0:
            summary_lines.append(f"<span style='color: #4caf50;'><b>{results.fully_completed}</b> completed full pipeline</span>")
        if results.paused_for_review > 0:
            summary_lines.append(f"<span style='color: #ffa500;'><b>{results.paused_for_review}</b> paused for review</span>")
        if results.failed > 0:
            summary_lines.append(f"<span style='color: red;'><b>{results.failed}</b> failed</span>")
        if not summary_lines:
            summary_lines.append("Pipeline run complete - nothing to process")

        self.summary_label.setText("<br>".join(summary_lines))

        # Show notification
        if results.fully_completed > 0 and results.paused_for_review == 0:
            show_info(f"Pipeline complete! {results.fully_completed} videos fully processed.")
        elif results.fully_completed > 0:
            show_warning(
                f"Pipeline complete: {results.fully_completed} fully processed, "
                f"{results.paused_for_review} need review."
            )
        elif results.paused_for_review > 0:
            show_warning(f"Pipeline complete: {results.paused_for_review} files need review.")
        else:
            show_info("Pipeline run complete - nothing needed processing.")

        # Refresh status to show new state
        QTimer.singleShot(500, self._scan_status)

    def _on_error(self, message: str):
        """Handle pipeline error."""
        self.run_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.reprocess_btn.setEnabled(True)
        if self.specific_files:
            self.clear_btn.setEnabled(True)
        self.summary_label.setText(f"<span style='color: red;'>Error: {message}</span>")
        show_error(f"Pipeline failed: {message}")


def main():
    """Launch unified pipeline widget standalone."""
    import napari

    viewer = napari.Viewer(title="MouseReach Analysis Pipeline")
    widget = UnifiedPipelineWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Pipeline", area="right")

    print("\nMouseReach Analysis Pipeline")
    print("=" * 40)
    print("Click Run to process everything that's ready.")
    print("Or select specific files for targeted reprocessing.")
    print()

    napari.run()


if __name__ == "__main__":
    main()
