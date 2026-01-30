"""
MouseReach Step 0: Video Prep Widget
================================

Napari widget for cropping multi-animal collage videos into single-animal videos.

Features:
- Browse and select input folder or single file
- Preview collage layout (8-camera grid)
- Run batch cropping with progress
- Auto-copy to DLC queue

Install as plugin:
    pip install -e .
    # Then: Plugins → MouseReach Step 0 - Video Prep
"""

from pathlib import Path
from typing import Optional
import threading

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QListWidget, QMessageBox,
    QCheckBox, QScrollArea, QListWidgetItem
)
from qtpy.QtCore import Qt, Signal, QObject

import napari
from napari.utils.notifications import show_info, show_error


class WorkerSignals(QObject):
    """Signals for background worker thread"""
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(dict)  # results
    error = Signal(str)


class VideoPrepWidget(QWidget):
    """
    Widget for cropping multi-animal collage videos.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.collage_files: list = []
        self.worker_thread: Optional[threading.Thread] = None
        self.signals = WorkerSignals()

        self._build_ui()
        self._connect_signals()

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
        header = QLabel("<b>Step 0: Video Prep</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header)

        # === Instructions ===
        instructions = QLabel(
            "Crop 8-camera collage videos into individual mouse videos.\n\n"
            "Input: Multi-animal .mkv collages (2x4 grid)\n"
            "Output: Single-animal .mp4 files\n\n"
            "Filename format:\n"
            "  20250704_CNT0101,CNT0205,..._P1.mkv\n"
            "  → 20250704_CNT0101_P1.mp4, etc."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(instructions)

        # === Input Selection ===
        input_group = QGroupBox("1. Select Input")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        self.input_label = QLabel("No input selected")
        self.input_label.setWordWrap(True)
        input_layout.addWidget(self.input_label)

        btn_row = QHBoxLayout()
        self.select_folder_btn = QPushButton("Select Folder...")
        self.select_folder_btn.clicked.connect(self._select_input_folder)
        btn_row.addWidget(self.select_folder_btn)

        self.select_file_btn = QPushButton("Select File...")
        self.select_file_btn.clicked.connect(self._select_input_file)
        btn_row.addWidget(self.select_file_btn)
        input_layout.addLayout(btn_row)

        layout.addWidget(input_group)

        # === File List ===
        files_group = QGroupBox("2. Collages to Process")
        files_layout = QVBoxLayout()
        files_group.setLayout(files_layout)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        files_layout.addWidget(self.file_list)

        self.file_count_label = QLabel("0 collages found")
        files_layout.addWidget(self.file_count_label)

        layout.addWidget(files_group)

        # === Output Selection ===
        output_group = QGroupBox("3. Output Location")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)

        self.output_label = QLabel("Default: same as input")
        self.output_label.setWordWrap(True)
        output_layout.addWidget(self.output_label)

        self.select_output_btn = QPushButton("Change Output Folder...")
        self.select_output_btn.clicked.connect(self._select_output_folder)
        output_layout.addWidget(self.select_output_btn)

        self.copy_to_queue_cb = QCheckBox("Copy outputs to DLC Queue")
        self.copy_to_queue_cb.setChecked(True)
        output_layout.addWidget(self.copy_to_queue_cb)

        layout.addWidget(output_group)

        # === Run ===
        run_group = QGroupBox("4. Process")
        run_layout = QVBoxLayout()
        run_group.setLayout(run_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        run_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        run_layout.addWidget(self.status_label)

        self.run_btn = QPushButton("Run Cropping")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self._run_cropping)
        self.run_btn.setEnabled(False)
        run_layout.addWidget(self.run_btn)

        layout.addWidget(run_group)

        # === Results ===
        self.results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_group.setLayout(results_layout)

        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)

        self.results_group.setVisible(False)
        layout.addWidget(self.results_group)

        # Spacer
        layout.addStretch()

    def _connect_signals(self):
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.error.connect(self._on_error)

    def _select_input_folder(self):
        from mousereach.config import Paths
        folder = QFileDialog.getExistingDirectory(
            self, "Select Collage Folder",
            str(Paths.MULTI_ANIMAL_SOURCE)
        )
        if folder:
            self.input_path = Path(folder)
            self._scan_for_collages()

    def _select_input_file(self):
        from mousereach.config import Paths
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Collage Video",
            str(Paths.MULTI_ANIMAL_SOURCE),
            "Video Files (*.mkv *.mp4 *.avi)"
        )
        if file_path:
            self.input_path = Path(file_path)
            self.collage_files = [self.input_path]
            self._update_file_list()

    def _select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        if folder:
            self.output_path = Path(folder)
            self.output_label.setText(str(self.output_path))

    def _scan_for_collages(self):
        """Find all collage files in input directory"""
        if not self.input_path or not self.input_path.is_dir():
            return

        self.collage_files = []
        for ext in ['*.mkv', '*.mp4', '*.avi']:
            for f in self.input_path.glob(ext):
                # Check if filename contains comma (multi-animal indicator)
                if ',' in f.stem:
                    self.collage_files.append(f)

        self.collage_files.sort(key=lambda x: x.name)
        self._update_file_list()

    def _update_file_list(self):
        self.file_list.clear()
        for f in self.collage_files:
            item = QListWidgetItem(f.name)
            self.file_list.addItem(item)

        count = len(self.collage_files)
        self.file_count_label.setText(f"{count} collage(s) found")
        self.run_btn.setEnabled(count > 0)

        if self.input_path:
            if self.input_path.is_file():
                self.input_label.setText(f"File: {self.input_path.name}")
            else:
                self.input_label.setText(f"Folder: {self.input_path}")

    def _run_cropping(self):
        """Run the cropping in a background thread"""
        if not self.collage_files:
            show_error("No collages to process")
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")

        # Run in background thread
        self.worker_thread = threading.Thread(target=self._crop_worker)
        self.worker_thread.start()

    def _crop_worker(self):
        """Background worker for cropping"""
        try:
            from mousereach.video_prep.core import crop_collage, copy_to_dlc_queue

            total = len(self.collage_files)
            results = {
                'total_collages': total,
                'success': 0,
                'skipped': 0,
                'failed': 0,
                'output_files': []
            }

            output_dir = self.output_path or (
                self.collage_files[0].parent if self.collage_files[0].is_file()
                else self.input_path
            )

            for i, collage in enumerate(self.collage_files):
                self.signals.progress.emit(i + 1, total, f"Processing {collage.name}...")

                try:
                    crop_results = crop_collage(collage, output_dir, verbose=False)

                    for r in crop_results:
                        if r.get('status') == 'success':
                            results['success'] += 1
                            results['output_files'].append(r.get('output_path'))
                        elif r.get('status') == 'skipped':
                            results['skipped'] += 1
                        else:
                            results['failed'] += 1

                except Exception as e:
                    results['failed'] += 8  # Assume all 8 positions failed

            # Copy to queue if requested
            if self.copy_to_queue_cb.isChecked() and results['success'] > 0:
                self.signals.progress.emit(total, total, "Copying to DLC queue...")
                try:
                    copied = copy_to_dlc_queue(output_dir, verbose=False)
                    results['copied_to_queue'] = copied
                except Exception as e:
                    results['queue_error'] = str(e)

            self.signals.finished.emit(results)

        except Exception as e:
            self.signals.error.emit(str(e))

    def _on_progress(self, current: int, total: int, message: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_finished(self, results: dict):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Complete!")

        # Show results
        self.results_group.setVisible(True)

        text = f"""
<b>Cropping Complete</b><br><br>
Collages processed: {results['total_collages']}<br>
Videos created: {results['success']}<br>
Skipped (blank): {results['skipped']}<br>
Failed: {results['failed']}
"""
        if 'copied_to_queue' in results:
            text += f"<br><br>Copied to DLC queue: {results['copied_to_queue']}"
        if 'queue_error' in results:
            text += f"<br><br>Queue copy error: {results['queue_error']}"

        self.results_label.setText(text)

        show_info(f"Cropping complete: {results['success']} videos created")

    def _on_error(self, message: str):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Error!")
        show_error(f"Cropping failed: {message}")


def main():
    """Launch Step 0 widget standalone."""
    import napari

    viewer = napari.Viewer(title="MouseReach Step 0: Video Prep")
    widget = VideoPrepWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Step 0 - Crop Videos", area="right")

    print("\nStep 0: Video Prep")
    print("=" * 40)
    print("Crop 8-camera collage videos into individual mouse videos.")
    print()

    napari.run()


if __name__ == "__main__":
    main()
