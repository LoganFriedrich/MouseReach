"""Where Is My Data tab -- the answer to the question at the end of every cohort.

WHY: the lab's workflow ends with someone asking where the clean data is,
finding a review queue nobody mentioned, and concluding everything failed.
This tab answers, per cohort, in one table: animals, hand-scored sessions,
videos in the database, videos waiting in review, how many outcomes are
still algorithm-only vs human-reviewed, the sheet's import status -- and
the folder of current CSVs (with data dictionaries) a person can open now.

The numbers come from mousedb (its own env) via ``mousedb-data-status
--json``, computed from the analysis snapshot and the queue folders --
never the live database -- so refreshing is always safe.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QTextEdit,
)

logger = logging.getLogger(__name__)

DEFAULT_MOUSEDB_PYTHON = Path("C:/LAB_ROOT/envs/MouseDB/python.exe")

QUICK_GUIDE = """
WHERE IS MY DATA -- one row per cohort:

* Animals: how many animals the database knows for the cohort (a number in
  brackets = created from a video before the sheet named them).
* Sheet: the tracking sheet's import status (see the Tracking Sheets tab).
* Sessions scored: hand-scored animal-days in the database.
* Videos in DB: videos whose reaches are in the database.
* In review: videos waiting for a person -- triage (per-segment questions)
  and deep review (whole-video problems). Open the Review Queues tab to work
  them. Until a video is reviewed and released, its data is NOT final.
* Outcomes algo / human: how many pellet outcomes rest on the algorithm
  alone vs were confirmed or corrected by a person.

THE FILES: the bottom panel names the current export folder. It holds
reach_data.csv (one row per reach), manual_scores.csv (one row per pellet
scored from the tray), ODC_sessions_<cohort>.csv (one row per animal per
session, ODC-SCI shape), each with a DATA_DICTIONARY.csv beside it, plus
MANIFEST.json saying when they were written and whether they are complete
for an ODC-SCI upload. "Open exports folder" opens it in Explorer.

The folder is rewritten every hour on the processing server. "Refresh
exports now" rewrites reach_data and manual_scores immediately from the
latest snapshot (the per-cohort ODC files refresh on the hourly run).
"""


class _Runner(QThread):
    done = Signal(dict)

    def __init__(self, python: Path, module: str, args: list):
        super().__init__()
        self.python, self.module, self.args = python, module, args

    def run(self):
        try:
            r = subprocess.run([str(self.python), "-m", self.module] + self.args,
                               capture_output=True, text=True, timeout=1800)
            out = (r.stdout or "").strip()
            start = out.find("{")
            payload = json.loads(out[start:]) if start >= 0 else {}
            payload.setdefault("_returncode", r.returncode)
            payload.setdefault("_stderr", (r.stderr or "")[-2000:])
            self.done.emit(payload)
        except Exception as e:
            self.done.emit({"problems": ["could not run %s: %s" % (self.module, e)],
                            "_returncode": -1})


class DataStatusWidget(QWidget):
    def __init__(self, napari_viewer=None, mousedb_python: Path = DEFAULT_MOUSEDB_PYTHON):
        super().__init__()
        self.viewer = napari_viewer
        self.python = Path(mousedb_python)
        self._status: dict = {}
        self._runner: Optional[_Runner] = None
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("<b>Where Is My Data</b>")
        title.setStyleSheet("font-size: 14px;")
        head.addWidget(title, 1)
        try:
            from mousereach.review.help_button import attach_help
            attach_help(head, "Where Is My Data", QUICK_GUIDE, self)
        except Exception:
            pass
        root.addLayout(head)

        self.snapshot_label = QLabel("")
        self.snapshot_label.setStyleSheet("color: #888;")
        self.snapshot_label.setWordWrap(True)
        root.addWidget(self.snapshot_label)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Cohort", "Animals", "Sheet", "Sessions scored", "Videos in DB",
             "In review (triage / deep)", "Outcomes algo / human", "Reaches"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table, 2)

        self.exports = QTextEdit()
        self.exports.setReadOnly(True)
        self.exports.setMaximumHeight(170)
        root.addWidget(self.exports)

        row = QHBoxLayout()
        for text, slot, tip, style in (
            ("Refresh", self.refresh, "Re-read the snapshot, the queues and the export manifest.", ""),
            ("Open exports folder", self.open_exports,
             "Open the folder of current CSVs (+ data dictionaries) in Explorer.",
             "background:#16405a; color:white; font-weight:bold;"),
            ("Refresh exports now", self.refresh_exports,
             "Rewrite reach_data.csv and manual_scores.csv from the latest snapshot "
             "right now (ODC session files refresh on the hourly run).", ""),
            ("Open Review Queues", self.open_queues,
             "Videos 'in review' are waiting for a person -- the Review Queues tab is "
             "where they get worked.", ""),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setStyleSheet(style)
            b.clicked.connect(slot)
            row.addWidget(b)
        root.addLayout(row)

    def _run(self, module: str, args: list, on_done):
        if self._runner is not None and self._runner.isRunning():
            return
        if not self.python.exists():
            self.exports.setPlainText("mousedb environment not found at %s -- this machine "
                                      "cannot compute data status." % self.python)
            return
        self._runner = _Runner(self.python, module, args)
        self._runner.done.connect(on_done)
        self._runner.start()

    def refresh(self):
        self.snapshot_label.setText("Reading...")
        self._run("mousedb.data_status", ["--json"], self._on_status)

    def _on_status(self, st: dict):
        self._status = st
        self.snapshot_label.setText("Numbers as of the analysis snapshot taken %s"
                                    % (st.get("snapshot_time") or "?"))
        cohorts = st.get("cohorts", [])
        self.table.setRowCount(len(cohorts))
        for i, c in enumerate(cohorts):
            q = c.get("videos_in_review") or {}
            src = c.get("segments_by_outcome_source") or {}
            human = sum(v for k, v in src.items() if k != "algo")
            animals = "?" if c.get("animals") is None else str(c["animals"])
            if c.get("animals_created_from_video_only"):
                animals += "  [%d from video only]" % c["animals_created_from_video_only"]
            sh = c.get("sheet") or {}
            vals = [c["cohort_id"], animals, sh.get("state") or "-",
                    c.get("sessions_scored", 0), c.get("videos_in_db", 0),
                    "%d / %d" % (q.get("triage", 0), q.get("deep_review", 0)),
                    "%d / %d" % (src.get("algo", 0), human), c.get("reaches_in_db", 0)]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if j == 5 and (q.get("triage", 0) or q.get("deep_review", 0)):
                    it.setForeground(Qt.white)
                    it.setBackground(Qt.darkYellow)
                if j == 2 and sh.get("state") in ("last_import_failed", "never_imported", "sheet_newer"):
                    it.setForeground(Qt.white)
                    it.setBackground(Qt.darkRed if sh.get("state") == "last_import_failed" else Qt.darkYellow)
                self.table.setItem(i, j, it)
        ex = st.get("exports") or {}
        lines = ["CURRENT EXPORTS: %s" % ex.get("folder"),
                 "written: %s    complete for ODC-SCI upload: %s"
                 % (ex.get("generated_at") or "never", ex.get("complete"))]
        for name, rows in (ex.get("files") or {}).items():
            lines.append("  %-38s %s rows" % (name, rows))
        for p in list(st.get("problems", [])) + list(ex.get("problems", [])):
            lines.append("  [!] %s" % p)
        self.exports.setPlainText("\n".join(lines))

    def open_exports(self):
        d = (self._status.get("exports") or {}).get("folder")
        if d and Path(d).is_dir():
            os.startfile(d)
        else:
            self.exports.append("No exports folder yet -- press 'Refresh exports now'.")

    def refresh_exports(self):
        self.exports.append("Rewriting current exports from the latest snapshot...")
        self._run("mousedb.exporters.current", ["--json"], lambda _m: self.refresh())

    def open_queues(self):
        try:
            import napari
            from mousereach.review.queue_launcher_widget import ReviewQueuesWidget
            v = napari.Viewer(title="Review Queues")
            v.window.add_dock_widget(ReviewQueuesWidget(v), name="Review Queues", area="right")
            self._queues_window = v
        except Exception as e:
            self.exports.append("Could not open Review Queues: %s" % e)


def main():
    """Standalone: mousereach-data-status"""
    import napari
    v = napari.Viewer(title="MouseReach -- Where Is My Data")
    v.window.add_dock_widget(DataStatusWidget(v), name="Where Is My Data", area="right")
    napari.run()
