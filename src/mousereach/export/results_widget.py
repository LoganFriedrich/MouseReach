"""The "Export results" tab: the filed results as a spreadsheet, at the level
a person chooses -- the same function as ``mousereach-results``
(export/results.py: export_results). One implementation, two ways in.

On the tab: a level (reach, pellet, session, day, animal), the filed-results
folder (default: Analyzed in this machine's pipeline folder), a format (csv
or xlsx), and "Export...", which asks where to save and writes the file.
What each column means is in the xlsx file's "columns" sheet and in
docs/RESULTS_EXPORT.md.

WHY a tab: a lab that drops its videos in and presses Run must get its
results out without a terminal; before this tab no window wrote any results
file at all.

ASCII-only console output; Qt text may use Unicode.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QFileDialog, QGroupBox,
)

logger = logging.getLogger(__name__)

LEVEL_CHOICES = (
    ("Session (one tray = one video): pellet counts, success rate, reach means", "session"),
    ("Pellet (one segment): its outcome and reaches", "pellet"),
    ("Reach: every reach with its kinematics", "reach"),
    ("Day (one animal on one date, all its trays)", "day"),
    ("Animal (everything filed for one animal)", "animal"),
)

INTRO = (
    "Writes the FILED results (every video that passed every check and was "
    "filed into Analyzed) as one spreadsheet, one row per item of the level you "
    "choose. Session, day and animal are counted from pellets: success rate = "
    "retrieved / pellets presented, and a pellet nobody reached for still "
    "counts. Every row also carries the totals of the levels above it "
    "(columns start with reach_, segment_, session_, day_, animal_). The xlsx "
    "file has a second sheet, 'columns', saying what each column means. Same "
    "as the command mousereach-results."
)


class ResultsExportWidget(QWidget):
    """Pick a level, a folder, a format; Export... writes the file."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        root = QVBoxLayout(self)
        head = QLabel("<b>Export results</b>")
        root.addWidget(head)
        intro = QLabel(INTRO)
        intro.setWordWrap(True)
        root.addWidget(intro)

        g = QGroupBox("What to export")
        gl = QVBoxLayout(g)
        self.level = QComboBox()
        for text, key in LEVEL_CHOICES:
            self.level.addItem(text, key)
        gl.addWidget(QLabel("Level (one row per):"))
        gl.addWidget(self.level)
        gl.addWidget(QLabel("Filed-results folder:"))
        row = QHBoxLayout()
        self.root_edit = QLineEdit()
        try:
            from mousereach.config import Paths
            if Paths.ANALYZED_OUTPUT:
                self.root_edit.setText(str(Paths.ANALYZED_OUTPUT))
        except Exception:
            pass
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_root)
        row.addWidget(self.root_edit)
        row.addWidget(browse)
        gl.addLayout(row)
        gl.addWidget(QLabel("Format:"))
        self.fmt = QComboBox()
        self.fmt.addItem("Excel (.xlsx, with a sheet of column meanings)", "xlsx")
        self.fmt.addItem("CSV (.csv)", "csv")
        gl.addWidget(self.fmt)
        root.addWidget(g)

        self.export_btn = QPushButton("Export...")
        self.export_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.export_btn.clicked.connect(self._export)
        root.addWidget(self.export_btn)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        root.addStretch()

    def _browse_root(self):
        path = QFileDialog.getExistingDirectory(self, "Filed-results folder",
                                                self.root_edit.text() or "")
        if path:
            self.root_edit.setText(path)

    def _export(self):
        from mousereach.export.results import default_output
        level = self.level.currentData()
        fmt = self.fmt.currentData()
        suggested = default_output(level, fmt)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results as", str(suggested),
            "Excel (*.xlsx)" if fmt == "xlsx" else "CSV (*.csv)")
        if not path:
            return
        self.run_export(level, self.root_edit.text().strip() or None, path, fmt)

    def run_export(self, level, root, out, fmt) -> dict:
        """Export and say the result on the tab. Returns export_results'
        summary, or {"error": ...}."""
        from mousereach.export.results import export_results
        try:
            res = export_results(level, root, out, fmt)
        except Exception as e:
            self.status.setText(f"Not exported: {e}")
            self.status.setStyleSheet("color:#c33; font-weight:bold;")
            return {"error": str(e)}
        lines = []
        if res["path"]:
            lines.append(f"Exported {res['rows']} {res['level']} row(s) from "
                         f"{res['videos']} filed video(s) to {res['path']}")
            self.status.setStyleSheet("color:#1a5; font-weight:bold;")
        else:
            lines.append(f"Nothing exported: no filed results under {res['root']}.")
            self.status.setStyleSheet("color:#c80; font-weight:bold;")
        for p in res["problems"]:
            lines.append(f"Left out: {p}")
        self.status.setText("\n".join(lines))
        return res


__all__ = ["ResultsExportWidget", "LEVEL_CHOICES"]
