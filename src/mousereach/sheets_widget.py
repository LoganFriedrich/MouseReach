"""Tracking Sheets tab -- import the lab's spreadsheets from a button.

WHY: the tracking spreadsheets are the hand-kept record of every animal
(weights, ramps, injury, surgery, manual tray scores). They are filled in
"eventually", and the database only knows what was imported. Until
2026-08-28 importing was a terminal command nobody in the lab knew, it ran
hourly in the background with no visible result, and a failure rolled back
in silence for weeks. This tab is the visible, clickable version: per cohort
it shows WHICH file is the sheet, WHEN it was edited, WHEN it was last
imported, and whether that worked -- and lets you import, pick the right
file when several match, and set the folder.

The work is done by mousedb (its own conda env); this tab shells out to
``mousedb-sheets`` with --json, exactly like the watcher's bench-disagreement
scan does, so MouseReach never needs mousedb installed.
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
    QTableWidgetItem, QHeaderView, QAbstractItemView, QFileDialog,
    QInputDialog, QMessageBox, QTextEdit,
)

logger = logging.getLogger(__name__)

DEFAULT_MOUSEDB_PYTHON = Path("C:/LAB_ROOT/envs/MouseDB/python.exe")

STATE_TEXT = {
    "up_to_date": ("Up to date", "#2e7d32"),
    "sheet_newer": ("Sheet edited since last import -- import", "#e08020"),
    "never_imported": ("Never imported -- import", "#e08020"),
    "last_import_failed": ("LAST IMPORT FAILED -- see message", "#c62828"),
    "no_sheet": ("No sheet found", "#888"),
    "error": ("Status error", "#c62828"),
}

QUICK_GUIDE = """
The tracking spreadsheets (one per cohort, e.g. Connectome_05_Animal_Tracking.xlsx)
are the lab's hand-kept record: animals, weights, ramps, manual tray scores, injury
and surgery details. The database only knows what has been IMPORTED from them.

WHAT THE TABLE SHOWS, per cohort: which file is being read as the sheet; when that
file was last edited; when it was last imported and whether that worked; and a
plain verdict -- Up to date / Sheet edited since last import / Never imported /
LAST IMPORT FAILED.

WHAT TO DO:

* Press "Import all sheets" (or select rows and "Import selected"). Progress and
  the outcome per cohort appear in the log box. A failure shows its reason.
* "Sheet edited since last import" means someone changed the spreadsheet after the
  last import -- import again to bring the database up to date. (An hourly
  background job also imports; this tab shows you whether it succeeded.)
* If a cohort says "N files match -- choose", more than one workbook in the
  folder claims to be that cohort's sheet (a copy, a draft, a "(2)"). Select the
  row and press "This is the sheet" to pick the real one. The choice is
  remembered and shown.
* "Set sheets folder" points the system at the folder holding the spreadsheets
  (needed once per machine, and again if the folder ever moves). "Open folder"
  opens it in Explorer.
* "New cohort sheet" creates a correctly formatted, empty workbook for a new
  cohort in that folder.

NOTHING HERE EDITS A SPREADSHEET. Importing only reads them.
"""


class _Runner(QThread):
    """Run one mousedb-sheets command off the GUI thread."""
    done = Signal(dict)

    def __init__(self, python: Path, args: list):
        super().__init__()
        self.python, self.args = python, args

    def run(self):
        try:
            r = subprocess.run([str(self.python), "-m", "mousedb.sheet_sync"] + self.args,
                               capture_output=True, text=True, timeout=3600)
            out = (r.stdout or "").strip()
            # the JSON is the last {...} block; importer chatter precedes it
            start = out.find("{")
            payload = json.loads(out[start:]) if start >= 0 else {}
            payload.setdefault("_returncode", r.returncode)
            payload.setdefault("_stderr", (r.stderr or "")[-2000:])
            payload.setdefault("_chatter", out[:start] if start > 0 else "")
            self.done.emit(payload)
        except Exception as e:
            self.done.emit({"problem": "could not run mousedb-sheets: %s" % e,
                            "_returncode": -1})


class TrackingSheetsWidget(QWidget):
    def __init__(self, napari_viewer=None, mousedb_python: Path = DEFAULT_MOUSEDB_PYTHON):
        super().__init__()
        self.viewer = napari_viewer
        self.python = Path(mousedb_python)
        self._status: dict = {}
        self._runner: Optional[_Runner] = None
        self._build_ui()
        self.refresh()

    # ----------------------------------------------------------------- ui
    def _build_ui(self):
        root = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("<b>Tracking Sheets</b>")
        title.setStyleSheet("font-size: 14px;")
        head.addWidget(title, 1)
        try:
            from mousereach.review.help_button import attach_help
            attach_help(head, "Tracking Sheets", QUICK_GUIDE, self)
        except Exception:
            pass
        root.addLayout(head)

        self.folder_label = QLabel("")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #888;")
        root.addWidget(self.folder_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Cohort", "Status", "Sheet file", "Sheet edited", "Last import"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._show_selected_detail)
        root.addWidget(self.table, 2)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        root.addWidget(self.detail)

        row = QHBoxLayout()
        for text, slot, tip, style in (
            ("Refresh", self.refresh, "Re-read the folder and the import history.", ""),
            ("Import all sheets", self.import_all,
             "Read every cohort's sheet into the database (reads only; never edits a sheet).",
             "background:#16405a; color:white; font-weight:bold;"),
            ("Import selected", self.import_selected,
             "Import only the cohort(s) selected in the table.", ""),
            ("This is the sheet", self.pin_selected,
             "When several files match a cohort, mark the selected cohort's chosen file "
             "as THE sheet (you will be asked which).", ""),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setStyleSheet(style)
            b.clicked.connect(slot)
            row.addWidget(b)
        root.addLayout(row)

        row2 = QHBoxLayout()
        for text, slot, tip in (
            ("Set sheets folder...", self.set_folder,
             "Point the system at the folder holding the Connectome_NN_Animal_Tracking.xlsx "
             "files. Needed once per machine, and again if the folder moves."),
            ("Open folder", self.open_folder, "Open the sheets folder in Explorer."),
            ("New cohort sheet...", self.new_sheet,
             "Create a correctly formatted, empty tracking workbook for a new cohort."),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row2.addWidget(b)
        row2.addStretch()
        root.addLayout(row2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        self.log.setPlaceholderText("Import results appear here.")
        root.addWidget(self.log)

    # ------------------------------------------------------------- actions
    def _run(self, args: list, on_done):
        if self._runner is not None and self._runner.isRunning():
            self._log("(busy -- wait for the current command to finish)")
            return
        if not self.python.exists():
            self._log("mousedb environment not found at %s -- this machine cannot "
                      "import sheets. The processing server can." % self.python)
            return
        self._runner = _Runner(self.python, args)
        self._runner.done.connect(on_done)
        self._runner.start()

    def refresh(self):
        self._log("Reading sheet status...")
        self._run(["status", "--json"], self._on_status)

    def _on_status(self, st: dict):
        self._status = st
        if st.get("problem"):
            self.folder_label.setText("[!] " + st["problem"])
            self.table.setRowCount(0)
            self._log(st["problem"])
            return
        self.folder_label.setText("Sheets folder: %s    (import history: %s)"
                                  % (st.get("cnt_sheets_dir"), st.get("ledger")))
        cohorts = st.get("cohorts", [])
        self.table.setRowCount(len(cohorts))
        for i, c in enumerate(cohorts):
            label, color = STATE_TEXT.get(c.get("state"), (c.get("state"), "#888"))
            if c.get("ambiguous"):
                label = "%d files match -- choose (currently: newest)" % len(c["candidates"])
                color = "#e08020"
            li = c.get("last_import") or {}
            vals = [c["cohort_id"], label, (c.get("sheet") or "-") +
                    ("  [pinned]" if c.get("pinned") else ""),
                    c.get("sheet_edited") or "-", li.get("finished") or "never"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if j == 1:
                    it.setForeground(Qt.white)
                    it.setBackground(Qt.darkGreen if color == "#2e7d32" else
                                     (Qt.darkRed if color == "#c62828" else Qt.darkYellow))
                self.table.setItem(i, j, it)
        n_bad = sum(1 for c in cohorts if c.get("state") in ("last_import_failed", "error"))
        n_stale = sum(1 for c in cohorts if c.get("state") in ("sheet_newer", "never_imported"))
        self._log("%d cohort(s): %d need importing, %d failed last time."
                  % (len(cohorts), n_stale, n_bad))

    def _selected_cohorts(self) -> list:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self._status["cohorts"][r] for r in rows if r < len(self._status.get("cohorts", []))]

    def _show_selected_detail(self):
        sel = self._selected_cohorts()
        if not sel:
            self.detail.setText("")
            return
        c = sel[0]
        lines = ["%s: %s" % (c["cohort_id"], c.get("why", ""))]
        if len(c.get("candidates", [])) > 1:
            lines.append("Files matching this cohort: " + "; ".join(
                "%s (edited %s)" % (x["name"], x["edited"]) for x in c["candidates"]))
        li = c.get("last_import") or {}
        if li.get("error"):
            lines.append("Last error: " + str(li["error"]))
        self.detail.setText("\n".join(lines))

    def import_all(self):
        self._log("Importing all sheets... (this reads every workbook; a minute or two)")
        self._run(["import", "--json", "--triggered-by", "gui"], self._on_import)

    def import_selected(self):
        sel = self._selected_cohorts()
        if not sel:
            self._log("Select one or more cohort rows first.")
            return
        args = ["import", "--json", "--triggered-by", "gui"]
        for c in sel:
            args += ["--cohort", c["cohort_id"]]
        self._log("Importing %s..." % ", ".join(c["cohort_id"] for c in sel))
        self._run(args, self._on_import)

    def _on_import(self, r: dict):
        if r.get("problem"):
            self._log("[!] " + r["problem"])
        for c in r.get("cohorts", []):
            if c.get("success"):
                self._log("OK   %s <- %s  %s" % (c["cohort_id"], c.get("sheet_name"),
                                                 json.dumps(c.get("imported"))))
            else:
                self._log("FAIL %s <- %s  %s" % (c["cohort_id"], c.get("sheet_name"),
                                                 c.get("error")))
        if r.get("_returncode", 0) not in (0, 1):
            self._log(r.get("_stderr") or "(no error text)")
        self.refresh()

    def pin_selected(self):
        sel = self._selected_cohorts()
        if len(sel) != 1:
            self._log("Select exactly one cohort row.")
            return
        c = sel[0]
        names = [x["name"] for x in c.get("candidates", [])]
        if not names:
            self._log("%s has no matching files." % c["cohort_id"])
            return
        choice, ok = QInputDialog.getItem(
            self, "Which file is the sheet?",
            "Files matching %s (newest first):" % c["cohort_id"], names, 0, False)
        if not ok:
            return
        self._log("Pinning %s -> %s" % (c["cohort_id"], choice))
        self._run(["pin", c["cohort_id"], choice], lambda _r: self.refresh())

    def set_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder holding the tracking sheets")
        if not d:
            return
        if not any(n.lower().startswith("connectome_") and n.lower().endswith(".xlsx")
                   for n in os.listdir(d)):
            QMessageBox.warning(self, "Not the sheets folder",
                                "That folder holds no Connectome_NN_Animal_Tracking.xlsx file. "
                                "Pick the folder that contains the tracking workbooks.")
            return
        self._log("Setting sheets folder: %s" % d)
        self._run(["set-dir", d], lambda _r: self.refresh())

    def open_folder(self):
        d = self._status.get("cnt_sheets_dir")
        if d and Path(d).is_dir():
            os.startfile(d)
        else:
            self._log("No sheets folder configured.")

    def new_sheet(self):
        d = self._status.get("cnt_sheets_dir")
        if not d:
            self._log("Set the sheets folder first.")
            return
        cohort, ok = QInputDialog.getText(self, "New cohort sheet",
                                          "Cohort name (e.g. CNT_06):")
        if not ok or not cohort.strip():
            return
        start, ok = QInputDialog.getText(self, "New cohort sheet",
                                         "Food-deprivation start date (YYYY-MM-DD):")
        if not ok or not start.strip():
            return
        self._log("Creating sheet for %s in %s ..." % (cohort.strip(), d))
        try:
            r = subprocess.run(
                [str(self.python), "-m", "mousedb.cohort_tools.make_sheets", "--new",
                 "--cohort", cohort.strip(), "--start-date", start.strip(),
                 "--output-dir", d],
                capture_output=True, text=True, timeout=600)
            self._log((r.stdout or "").strip()[-1500:] or "(no output)")
            if r.returncode != 0:
                self._log("FAILED: " + (r.stderr or "")[-1500:])
        except Exception as e:
            self._log("FAILED: %s" % e)
        self.refresh()

    def _log(self, msg: str):
        self.log.append(msg)


def main():
    """Standalone: mousereach-sheets"""
    import napari
    v = napari.Viewer(title="MouseReach -- Tracking Sheets")
    v.window.add_dock_widget(TrackingSheetsWidget(v), name="Tracking Sheets", area="right")
    napari.run()
