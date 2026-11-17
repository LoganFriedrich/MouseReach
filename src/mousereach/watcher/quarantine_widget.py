"""Quarantine tab -- fix and release held-back files.

Videos whose filenames the pipeline could not parse (bad/typo'd animal IDs, a
single-animal video dropped into the collage folder, etc.) are set aside in the
Quarantine folder with a ``.quarantine.json`` note explaining why. This tab lets
an operator, without a terminal:

  * see every quarantined file and the reason it was held,
  * correct its name in place, and
  * release it back into the intake folder so the pipeline picks it up again.

Release checks the name first (validator.release_from_quarantine, shared with
``mousereach-watch-quarantine --release``): a name the chosen intake would not
accept is refused with the reason and a suggested fix, and nothing moves. The
old name is allowed -- it may be valid now -- and the watcher takes a file
released under its old name back in by itself.

ASCII-only console output (Windows cp1252). Qt text may use Unicode.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from qtpy.QtCore import Qt

from napari.utils.notifications import show_info, show_error

logger = logging.getLogger(__name__)

_META_SUFFIX = ".quarantine.json"


class QuarantineWidget(QWidget):
    """List, rename, and release quarantined files."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self._records: List[dict] = []
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        intro = QLabel(
            "Files held back because their name could not be read (bad animal IDs, "
            "a single video in the collage folder, etc.). Fix the name in the last "
            "column, choose where it goes, and Release it back into the pipeline. "
            "A name the chosen intake cannot read is refused (nothing moves) and "
            "the message says what is wrong."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self._count = QLabel("")
        self._count.setStyleSheet("font-weight:bold;")
        root.addWidget(self._count)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["File", "Why held", "Corrected name (editable)"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        root.addWidget(self._table)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Release to:"))
        self._dest = QComboBox()
        self._dest.addItem("Collage intake (8-camera videos)", "multi")
        self._dest.addItem("Single-animal intake (one mouse)", "single")
        controls.addWidget(self._dest)
        controls.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        release_btn = QPushButton("Release selected")
        release_btn.setStyleSheet("background:#1a5; color:white; font-weight:bold;")
        release_btn.clicked.connect(self._release_selected)
        controls.addWidget(refresh_btn)
        controls.addWidget(release_btn)
        root.addLayout(controls)

    # ------------------------------------------------------------ helpers
    def _quarantine_dir(self) -> Optional[Path]:
        try:
            from mousereach.config import WatcherConfig
            return WatcherConfig.load().get_quarantine_dir()
        except Exception:
            return None

    def _dest_dir(self):
        from mousereach.config import Paths
        if self._dest.currentData() == "single":
            return Paths.SINGLE_ANIMAL_OUTPUT
        return Paths.MULTI_ANIMAL_SOURCE

    def _refresh(self):
        qdir = self._quarantine_dir()
        self._records = []
        self._table.setRowCount(0)
        if not qdir or not Path(qdir).exists():
            self._count.setText("Quarantine folder: none / empty.")
            return
        try:
            from mousereach.watcher.validator import get_quarantined_files
            self._records = get_quarantined_files(Path(qdir))
        except Exception as e:
            show_error(f"Could not read the quarantine folder: {e}")
            return
        self._count.setText(f"{len(self._records)} file(s) held in quarantine.")
        self._table.setRowCount(len(self._records))
        for i, rec in enumerate(self._records):
            fname = self._filename(rec)
            f_item = QTableWidgetItem(fname)
            f_item.setFlags(f_item.flags() & ~Qt.ItemIsEditable)
            reason = QTableWidgetItem(str(rec.get("error_message", "") or ""))
            reason.setFlags(reason.flags() & ~Qt.ItemIsEditable)
            fix = QTableWidgetItem(fname)  # editable, pre-filled with current name
            self._table.setItem(i, 0, f_item)
            self._table.setItem(i, 1, reason)
            self._table.setItem(i, 2, fix)

    def _filename(self, rec: dict) -> str:
        mf = rec.get("metadata_file", "")
        if mf.endswith(_META_SUFFIX):
            return Path(mf).name[: -len(_META_SUFFIX)]
        # fall back to the original path's basename
        return Path(rec.get("original_path") or "").name

    def _release_selected(self):
        row = self._table.currentRow()
        if row < 0 or row >= len(self._records):
            show_info("Select a file first.")
            return
        qdir = self._quarantine_dir()
        dest = self._dest_dir()
        if not qdir or not dest:
            show_error("Quarantine or intake folder is not configured.")
            return
        old_name = self._filename(self._records[row])
        new_name = (self._table.item(row, 2).text() or "").strip() or old_name
        # One release routine for the panel and the command line. It checks
        # the name with the CHOSEN destination's rules before moving anything,
        # so a name the watcher would only quarantine again is refused here,
        # with what is wrong and a suggested fix. WHY: the panel moved any
        # name and always said "the watcher will pick it up".
        from mousereach.watcher.validator import release_from_quarantine
        destination = "single" if self._dest.currentData() == "single" else "collage"
        ok, msg = release_from_quarantine(
            old_name, Path(qdir), Path(dest), destination, new_name=new_name)
        if not ok:
            show_error(msg)
            if not (Path(qdir) / old_name).exists():
                self._refresh()
            return
        show_info(msg)
        self._refresh()


def main():
    """Standalone launch of the Quarantine panel."""
    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(QuarantineWidget(viewer), name="Quarantine", area="right")
    napari.run()


if __name__ == "__main__":
    main()
