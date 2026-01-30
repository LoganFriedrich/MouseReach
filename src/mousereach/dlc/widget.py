"""
MouseReach Step 1: DLC Processing Widget
====================================

Napari widget for DeepLabCut pose estimation.

Features:
- Launch DLC GUI for model creation/training
- Browse and select videos for analysis
- Select DLC project/config
- Run batch analysis with progress
- Quality check results

Install as plugin:
    pip install -e .
    # Then: Plugins → MouseReach Step 1 - DLC Processing
"""

from pathlib import Path
from typing import Optional
import threading
import subprocess
import sys

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QListWidget, QMessageBox,
    QCheckBox, QScrollArea, QListWidgetItem, QComboBox, QSpinBox
)
from qtpy.QtCore import Qt, Signal, QObject

import napari
from napari.utils.notifications import show_info, show_error, show_warning


class WorkerSignals(QObject):
    """Signals for background worker thread"""
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(dict)  # results
    error = Signal(str)


class DLCWidget(QWidget):
    """
    Widget for DeepLabCut pose estimation.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.input_path: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.video_files: list = []
        self.worker_thread: Optional[threading.Thread] = None
        self.signals = WorkerSignals()

        self._build_ui()
        self._connect_signals()
        self._load_default_project()

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
        header = QLabel("<b>Step 1: DLC Processing</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header)

        # === DLC GUI Section ===
        gui_group = QGroupBox("DeepLabCut GUI")
        gui_layout = QVBoxLayout()
        gui_group.setLayout(gui_layout)

        gui_info = QLabel(
            "Use the DLC GUI to:\n"
            "- Create new projects\n"
            "- Label training data\n"
            "- Train models\n"
            "- Evaluate model performance"
        )
        gui_info.setWordWrap(True)
        gui_info.setStyleSheet("color: #888;")
        gui_layout.addWidget(gui_info)

        self.launch_dlc_btn = QPushButton("Launch DLC GUI")
        self.launch_dlc_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #2d5a27;")
        self.launch_dlc_btn.clicked.connect(self._launch_dlc_gui)
        gui_layout.addWidget(self.launch_dlc_btn)

        self.launch_dlc_project_btn = QPushButton("Launch DLC GUI with Selected Project")
        self.launch_dlc_project_btn.setStyleSheet("padding: 6px;")
        self.launch_dlc_project_btn.clicked.connect(self._launch_dlc_gui_with_project)
        self.launch_dlc_project_btn.setEnabled(False)
        gui_layout.addWidget(self.launch_dlc_project_btn)

        layout.addWidget(gui_group)

        # === Instructions ===
        instructions = QLabel(
            "<b>Batch Analysis</b>\n\n"
            "Run pose estimation on multiple videos using a trained model.\n"
            "Input: .mp4 videos\n"
            "Output: .h5 tracking files"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 5px;")
        layout.addWidget(instructions)

        # === DLC Config Selection ===
        config_group = QGroupBox("1. Select DLC Project")
        config_layout = QVBoxLayout()
        config_group.setLayout(config_layout)

        self.config_label = QLabel("No project selected")
        self.config_label.setWordWrap(True)
        config_layout.addWidget(self.config_label)

        config_btn_row = QHBoxLayout()
        self.select_config_btn = QPushButton("Select config.yaml...")
        self.select_config_btn.clicked.connect(self._select_config)
        config_btn_row.addWidget(self.select_config_btn)

        self.select_project_btn = QPushButton("Select Project Folder...")
        self.select_project_btn.clicked.connect(self._select_project)
        config_btn_row.addWidget(self.select_project_btn)
        config_layout.addLayout(config_btn_row)

        layout.addWidget(config_group)

        # === Input Selection ===
        input_group = QGroupBox("2. Select Videos")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        self.input_label = QLabel("No videos selected")
        self.input_label.setWordWrap(True)
        input_layout.addWidget(self.input_label)

        btn_row = QHBoxLayout()
        self.select_folder_btn = QPushButton("Select Folder...")
        self.select_folder_btn.clicked.connect(self._select_input_folder)
        btn_row.addWidget(self.select_folder_btn)

        self.select_files_btn = QPushButton("Select Files...")
        self.select_files_btn.clicked.connect(self._select_input_files)
        btn_row.addWidget(self.select_files_btn)
        input_layout.addLayout(btn_row)

        layout.addWidget(input_group)

        # === File List ===
        files_group = QGroupBox("3. Videos to Process")
        files_layout = QVBoxLayout()
        files_group.setLayout(files_layout)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        files_layout.addWidget(self.file_list)

        self.file_count_label = QLabel("0 videos found")
        files_layout.addWidget(self.file_count_label)

        layout.addWidget(files_group)

        # === GPU Selection ===
        gpu_group = QGroupBox("4. Processing Options")
        gpu_layout = QVBoxLayout()
        gpu_group.setLayout(gpu_layout)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("GPU Device:"))
        self.gpu_selector = QSpinBox()
        self.gpu_selector.setMinimum(0)
        self.gpu_selector.setMaximum(7)
        self.gpu_selector.setValue(0)
        gpu_row.addWidget(self.gpu_selector)
        gpu_row.addStretch()
        gpu_layout.addLayout(gpu_row)

        self.use_cpu_cb = QCheckBox("Use CPU instead (slower)")
        gpu_layout.addWidget(self.use_cpu_cb)

        layout.addWidget(gpu_group)

        # === Run ===
        run_group = QGroupBox("5. Run Analysis")
        run_layout = QVBoxLayout()
        run_group.setLayout(run_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        run_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        run_layout.addWidget(self.status_label)

        self.run_btn = QPushButton("Run DLC Analysis")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setEnabled(False)
        run_layout.addWidget(self.run_btn)

        layout.addWidget(run_group)

        # === Quality Check ===
        quality_group = QGroupBox("6. Quality Check")
        quality_layout = QVBoxLayout()
        quality_group.setLayout(quality_layout)

        self.check_quality_btn = QPushButton("Check DLC Quality...")
        self.check_quality_btn.clicked.connect(self._check_quality)
        quality_layout.addWidget(self.check_quality_btn)

        self.quality_label = QLabel("")
        self.quality_label.setWordWrap(True)
        quality_layout.addWidget(self.quality_label)

        layout.addWidget(quality_group)

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

    def _load_default_project(self):
        """Load the default DLC project from mousereach_config"""
        try:
            from mousereach.config import Paths
            # Check if DLC_CONFIG is defined (only on GPU analysis machine)
            if hasattr(Paths, 'DLC_CONFIG') and Paths.DLC_CONFIG.exists():
                self.config_path = Paths.DLC_CONFIG
                if hasattr(Paths, 'DLC_PROJECT'):
                    self.config_label.setText(f"Project: {Paths.DLC_PROJECT.name}")
                self.launch_dlc_project_btn.setEnabled(True)
                self._update_run_button()
        except (ImportError, AttributeError):
            pass  # Config not available on Analysis PC, user will select manually

    def _launch_dlc_gui(self):
        """Launch the DeepLabCut GUI"""
        try:
            # Try to launch DLC GUI
            self.status_label.setText("Launching DLC GUI...")

            # Run in separate process so it doesn't block napari
            # DLC 2.3+ uses "python -m deeplabcut" to launch GUI
            subprocess.Popen([sys.executable, "-m", "deeplabcut"])

            self.status_label.setText("DLC GUI launched (separate window)")
            show_info("DLC GUI launched in separate window")

        except Exception as e:
            show_error(f"Failed to launch DLC GUI: {e}")
            self.status_label.setText("Failed to launch DLC GUI")

    def _launch_dlc_gui_with_project(self):
        """Launch DLC GUI and open selected project"""
        if not self.config_path or not self.config_path.exists():
            show_error("No project selected")
            return

        try:
            self.status_label.setText("Launching DLC GUI with project...")

            # Create a small script to launch DLC and load the project
            # DLC GUI will open and we use deeplabcut.load_config to set it
            script = f'''
import deeplabcut
config_path = r"{self.config_path}"
# Launch GUI - it will start fresh, user can load project from "Manage Project" tab
deeplabcut.launch_dlc()
'''
            # For now, just launch DLC and show user which project to load
            subprocess.Popen([sys.executable, "-m", "deeplabcut"])

            self.status_label.setText("DLC GUI launched - load project from Manage Project tab")
            show_info(f"DLC GUI launched.\nLoad project: {self.config_path.parent.name}\nPath: {self.config_path}")

        except Exception as e:
            show_error(f"Failed to launch DLC GUI: {e}")
            self.status_label.setText("Failed to launch DLC GUI")

    def _select_config(self):
        # Start from default AIs folder if it exists
        # DLC model stays on A: drive (too large to copy)
        start_dir = ""  # User selects their DLC model location

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DLC config.yaml",
            start_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.config_path = Path(file_path)
            self.config_label.setText(f"Project: {self.config_path.parent.name}")
            self.launch_dlc_project_btn.setEnabled(True)
            self._update_run_button()

    def _select_project(self):
        # DLC model stays on A: drive (too large to copy)
        start_dir = ""  # User selects their DLC model location

        folder = QFileDialog.getExistingDirectory(
            self, "Select DLC Project Folder",
            start_dir
        )
        if folder:
            folder_path = Path(folder)
            # Look for config.yaml first, then any .yaml file
            config = folder_path / "config.yaml"
            if not config.exists():
                yaml_files = list(folder_path.glob("*.yaml")) + list(folder_path.glob("*.yml"))
                if yaml_files:
                    config = yaml_files[0]  # Use first yaml found
                else:
                    show_error("No YAML config file found in selected folder")
                    return

            self.config_path = config
            self.config_label.setText(f"Project: {folder_path.name} ({config.name})")
            self.launch_dlc_project_btn.setEnabled(True)
            self._update_run_button()

    def _select_input_folder(self):
        from mousereach.config import Paths
        folder = QFileDialog.getExistingDirectory(
            self, "Select Video Folder",
            str(Paths.DLC_QUEUE)
        )
        if folder:
            self.input_path = Path(folder)
            self._scan_for_videos()

    def _select_input_files(self):
        from mousereach.config import Paths
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos",
            str(Paths.DLC_QUEUE),
            "Video Files (*.mp4 *.avi *.mov)"
        )
        if files:
            self.video_files = [Path(f) for f in files]
            self._update_file_list()

    def _scan_for_videos(self):
        """Find all video files in input directory"""
        if not self.input_path or not self.input_path.is_dir():
            return

        self.video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            self.video_files.extend(self.input_path.glob(ext))

        # Filter out DLC-generated labeled videos (contain "DLC_" in filename)
        self.video_files = [v for v in self.video_files if 'DLC_' not in v.name]

        # Filter out already-processed videos (have corresponding .h5)
        unprocessed = []
        for v in self.video_files:
            h5_pattern = v.stem + "*DLC*.h5"
            existing_h5 = list(v.parent.glob(h5_pattern))
            if not existing_h5:
                unprocessed.append(v)

        self.video_files = sorted(unprocessed, key=lambda x: x.name)
        self._update_file_list()

    def _update_file_list(self):
        self.file_list.clear()
        for f in self.video_files:
            item = QListWidgetItem(f.name)
            self.file_list.addItem(item)

        count = len(self.video_files)
        self.file_count_label.setText(f"{count} video(s) to process")

        if self.input_path:
            self.input_label.setText(f"Folder: {self.input_path}")

        self._update_run_button()

    def _update_run_button(self):
        enabled = self.config_path is not None and len(self.video_files) > 0
        self.run_btn.setEnabled(enabled)

    def _run_analysis(self):
        """Run DLC analysis in background thread"""
        if not self.config_path or not self.video_files:
            show_error("Select a DLC project and videos first")
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")

        self.worker_thread = threading.Thread(target=self._analysis_worker)
        self.worker_thread.start()

    def _analysis_worker(self):
        """Background worker for DLC analysis"""
        try:
            from mousereach.dlc.core import run_dlc_batch

            total = len(self.video_files)
            gpu = None if self.use_cpu_cb.isChecked() else self.gpu_selector.value()

            results = {
                'total': total,
                'success': 0,
                'failed': 0,
                'outputs': []
            }

            for i, video in enumerate(self.video_files):
                self.signals.progress.emit(i + 1, total, f"Processing {video.name}...")

                try:
                    # Run DLC on single video
                    result = run_dlc_batch([video], self.config_path, video.parent, gpu)

                    if result and result[0].get('status') == 'success':
                        results['success'] += 1
                        results['outputs'].append(result[0].get('output_path'))
                    else:
                        results['failed'] += 1

                except Exception as e:
                    results['failed'] += 1

            self.signals.finished.emit(results)

        except ImportError as e:
            self.signals.error.emit(f"DeepLabCut not installed: {e}")
        except Exception as e:
            self.signals.error.emit(str(e))

    def _check_quality(self):
        """Check quality of DLC outputs"""
        from mousereach.config import Paths
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select DLC .h5 Files",
            str(Paths.DLC_COMPLETE),
            "HDF5 Files (*.h5)"
        )
        if not files:
            return

        try:
            from mousereach.dlc.core import check_dlc_quality

            results = []
            for h5_path in files:
                report = check_dlc_quality(Path(h5_path))
                results.append(f"{report.video_name}: {report.overall_quality.upper()} "
                              f"(mean likelihood: {report.mean_likelihood:.2f})")

            self.quality_label.setText("\n".join(results))
            show_info(f"Checked {len(files)} files")

        except ImportError:
            show_error("Quality check not available - missing dependencies")
        except Exception as e:
            show_error(f"Quality check failed: {e}")

    def _on_progress(self, current: int, total: int, message: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_finished(self, results: dict):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Complete!")

        self.results_group.setVisible(True)

        text = f"""
<b>DLC Analysis Complete</b><br><br>
Videos processed: {results['total']}<br>
Succeeded: {results['success']}<br>
Failed: {results['failed']}
"""
        self.results_label.setText(text)

        if results['success'] > 0:
            show_info(f"DLC complete: {results['success']}/{results['total']} videos processed")
        else:
            show_warning("DLC analysis completed but no videos succeeded")

    def _on_error(self, message: str):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Error!")
        show_error(f"DLC analysis failed: {message}")


def main():
    """Launch Step 1 widget standalone."""
    import napari

    viewer = napari.Viewer(title="MouseReach Step 1: DLC Processing")
    widget = DLCWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Step 1 - DLC Analysis", area="right")

    print("\nStep 1: DLC Processing")
    print("=" * 40)
    print("Run DeepLabCut pose estimation on videos.")
    print()

    napari.run()


if __name__ == "__main__":
    main()
