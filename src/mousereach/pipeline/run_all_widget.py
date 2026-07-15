"""Run All Steps tab -- push one opened video through the whole pipeline.

An operator picks a single-animal video and clicks Run; the driver
(mousereach.pipeline.run_all.run_all_steps) runs segmentation -> reaches ->
outcomes -> causal reach (algo-4) -> gate -> kinematics, streaming per-stage
progress into the log. The video must already have its DLC pose (.h5) beside it
(DLC is a GPU step -- run it in the '1 - DLC Analysis' tab first if missing).

The work runs on a background thread; progress is marshalled back to the GUI via
Qt signals (thread-safe queued connections).

ASCII-only console output (Windows cp1252). Qt text may use Unicode.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTextEdit, QFileDialog,
)
from qtpy.QtCore import QObject, Signal

from napari.utils.notifications import show_info, show_error

logger = logging.getLogger(__name__)


class _Worker(QObject):
    """Carries progress/finished signals from the worker thread to the GUI."""
    progress = Signal(str, str)
    finished = Signal(dict)


class RunAllStepsWidget(QWidget):
    """One-click full-pipeline run for a single opened video."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self._thread = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        intro = QLabel(
            "Run ONE video through every step: segmentation -> reaches -> "
            "outcomes -> causal reach -> kinematics. The video must already have "
            "its DLC pose (.h5) next to it (run DLC first if it does not)."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        row = QHBoxLayout()
        self._path = QLineEdit()
        self._path.setPlaceholderText("Path to a single-animal video (.mp4)")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(self._path)
        row.addWidget(browse)
        root.addLayout(row)

        self._run_btn = QPushButton("Run All Steps")
        self._run_btn.setStyleSheet("background:#1a5; color:white; font-weight:bold;")
        self._run_btn.clicked.connect(self._run)
        root.addWidget(self._run_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        root.addWidget(self._log)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a single-animal video", self._path.text() or "",
            "Videos (*.mp4 *.avi);;All files (*)",
        )
        if path:
            self._path.setText(path)

    def _run(self):
        path = self._path.text().strip()
        if not path or not Path(path).exists():
            show_error("Pick a video file first.")
            return
        if self._thread and self._thread.is_alive():
            show_info("A run is already in progress.")
            return
        self._log.clear()
        self._status.setText("Running...")
        self._status.setStyleSheet("color:#888;")
        self._run_btn.setEnabled(False)

        worker = _Worker()
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        self._worker = worker  # keep a reference

        def job():
            try:
                from mousereach.pipeline.run_all import run_all_steps
                res = run_all_steps(path, progress=lambda s, m: worker.progress.emit(s, m))
            except Exception as e:
                res = {"error": str(e), "stages": [], "held": None, "done": False}
            worker.finished.emit(res)

        self._thread = threading.Thread(target=job, daemon=True, name="run-all-steps")
        self._thread.start()

    def _on_progress(self, stage: str, msg: str):
        self._log.append(f"  {stage}: {msg}")

    def _on_finished(self, result: dict):
        self._run_btn.setEnabled(True)
        if result.get("error"):
            self._status.setText(f"[FAIL] {result['error']}")
            self._status.setStyleSheet("color:#a33; font-weight:bold;")
        elif result.get("held"):
            self._status.setText(
                f"[HOLD] Would be held for review ({result['held']}): "
                f"{result.get('hold_reason', '')}. Kinematics skipped -- resolve in "
                f"the Review Queues tab to finish."
            )
            self._status.setStyleSheet("color:#c80; font-weight:bold;")
        elif result.get("done"):
            self._status.setText("[OK] All steps complete -- kinematics synced to the database.")
            self._status.setStyleSheet("color:#1a5; font-weight:bold;")
        else:
            self._status.setText("Finished.")


def main():
    """Standalone launch of the Run All Steps panel."""
    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(RunAllStepsWidget(viewer), name="Run All Steps", area="right")
    napari.run()


if __name__ == "__main__":
    main()
