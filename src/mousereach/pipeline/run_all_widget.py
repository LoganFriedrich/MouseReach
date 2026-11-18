"""Run All Steps tab -- run one chosen video through the watcher's own steps.

A person picks a single-animal video and clicks Run. The run is the watcher's
own (mousereach.watcher.run_one.run_one_video): take it in from the posed
folder, analyse it, run the quality check, the hand-score comparison and the
review gate, make its kinematics and file it -- or hold it in a review queue,
exactly as the automatic watcher would. A filed video, a video held in a review
queue, a crystallized one, one the watcher does not know, and any run while a
watcher already runs on this machine are refused, with the reason on screen;
nothing is run or moved then. Progress lines are the watcher's own log lines
about that video.

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
            "Run ONE chosen video through the watcher's own steps, now: take it "
            "in, segmentation -> reaches -> outcomes -> causal reach, the quality "
            "check, the hand-score comparison and the review gate, then "
            "kinematics and filing -- or a review queue, as the watcher would. "
            "Runs on the processing machine only, and only when no watcher is "
            "running there. Not run (the reason is shown): a filed video, a video "
            "in a review queue, a crystallized video, or one the watcher does not "
            "know (a posed video waits in the posed folder with its pose)."
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
                res = {"error": str(e), "stages": [], "held": None, "done": False,
                       "refused": None}
            worker.finished.emit(res)

        self._thread = threading.Thread(target=job, daemon=True, name="run-all-steps")
        self._thread.start()

    def _on_progress(self, stage: str, msg: str):
        self._log.append(f"  {stage}: {msg}")

    def _on_finished(self, result: dict):
        self._run_btn.setEnabled(True)
        self._status.setText(status_text(result))
        if result.get("refused") or result.get("held"):
            self._status.setStyleSheet("color:#c80; font-weight:bold;")
        elif result.get("error"):
            self._status.setStyleSheet("color:#a33; font-weight:bold;")
        elif result.get("done"):
            self._status.setStyleSheet("color:#1a5; font-weight:bold;")


def status_text(result: dict) -> str:
    """The one line the tab shows when a run ends. WHY its own function: the
    words say what happened to the video (filed, held where, refused why), and
    a test reads them without a display."""
    if result.get("refused"):
        return f"[NOT RUN] {result['refused']}"
    if result.get("held") in ("triage", "deep_review"):
        where = result["held"].replace("_", " ")
        return (f"[HELD] Sent to the {where} queue: {result.get('hold_reason') or ''}. "
                f"It goes on by itself once reviewed (Review Queues tab).")
    if result.get("held") == "processing":
        return (f"[HELD] Waiting in Processing: {result.get('hold_reason') or ''}. "
                f"Not filed until that is resolved.")
    if result.get("error"):
        return f"[FAIL] {result['error']}"
    if result.get("done"):
        return "[OK] Passed every check and filed."
    return f"Finished (watcher state now '{result.get('state_after') or '?'}')."


def main():
    """Standalone launch of the Run All Steps panel."""
    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(RunAllStepsWidget(viewer), name="Run All Steps", area="right")
    napari.run()


if __name__ == "__main__":
    main()
