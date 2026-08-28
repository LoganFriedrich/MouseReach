"""First-run setup, in the app.

"The GUI is god": configuring where the pipeline folders live must be possible
from the app, not only via the ``mousereach-setup`` terminal command. This
widget collects the two paths mousereach needs -- the local Processing root
(required) and the NAS data root (optional) -- with a Detect button that
auto-fills from this machine's lab profile, and saves via the same
``save_config`` the CLI wizard uses (so the two stay in sync).

Config is read at import time, so a save takes effect on the NEXT launch; the UI
says so.

ASCII-only console output (Windows cp1252). Qt text may use Unicode.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QGroupBox, QFileDialog,
)
from napari.utils.notifications import show_info, show_error

logger = logging.getLogger(__name__)


class SetupWidget(QWidget):
    """Collect + save the pipeline paths (Processing root + NAS root)."""

    def __init__(self, napari_viewer=None, on_saved: Optional[Callable] = None):
        super().__init__()
        self.viewer = napari_viewer
        self._on_saved = on_saved
        self._detected_watcher = None
        self._build_ui()
        self._load_existing()

    def _build_ui(self):
        root = QVBoxLayout(self)

        intro = QLabel(
            "Tell MouseReach where its folders live. This is a one-time setup per "
            "machine. 'Detect' fills these in automatically if this computer is a "
            "known lab machine; otherwise enter them by hand."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        detect_btn = QPushButton("Detect automatically")
        detect_btn.clicked.connect(self._detect)
        root.addWidget(detect_btn)
        self._detected_label = QLabel("")
        self._detected_label.setStyleSheet("color:#888;")
        self._detected_label.setWordWrap(True)
        root.addWidget(self._detected_label)

        # Processing root (required)
        pgrp = QGroupBox("Processing root  (required -- local fast disk for working files)")
        pl = QHBoxLayout(pgrp)
        self._proc = QLineEdit()
        self._proc.setPlaceholderText(r"e.g. C:\MyLab\MouseReach_Pipeline")
        pbrowse = QPushButton("Browse...")
        pbrowse.clicked.connect(lambda: self._browse(self._proc))
        pl.addWidget(self._proc)
        pl.addWidget(pbrowse)
        root.addWidget(pgrp)

        # NAS root (optional)
        ngrp = QGroupBox("NAS data root  (optional -- shared network drive)")
        nl = QHBoxLayout(ngrp)
        self._nas = QLineEdit()
        self._nas.setPlaceholderText(r"e.g. Z:\MouseReach_Pipeline  (leave blank if none)")
        nbrowse = QPushButton("Browse...")
        nbrowse.clicked.connect(lambda: self._browse(self._nas))
        nl.addWidget(self._nas)
        nl.addWidget(nbrowse)
        root.addWidget(ngrp)

        save_btn = QPushButton("Save configuration")
        save_btn.setStyleSheet("background:#1a5; color:white; font-weight:bold;")
        save_btn.clicked.connect(self._save)
        root.addWidget(save_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        root.addStretch()

    def _load_existing(self):
        try:
            from mousereach.setup.wizard import load_config
            cfg = load_config()
            self._proc.setText(str(cfg.get("processing_root", "") or ""))
            self._nas.setText(str(cfg.get("nas_drive", "") or ""))
        except Exception as e:
            logger.debug(f"load_config failed: {e}")

    def _detect(self):
        try:
            from mousereach.setup.wizard import detect_lab_profile
            profile, drives, method = detect_lab_profile()
        except Exception as e:
            show_error(f"Detection failed: {e}")
            return
        if not profile:
            self._detected_label.setText(
                "No lab profile matched this machine -- enter the paths by hand."
            )
            return
        d = profile.get("defaults", {}) or {}
        if d.get("processing_root"):
            self._proc.setText(str(d["processing_root"]))
        if d.get("nas_drive"):
            self._nas.setText(str(d["nas_drive"]))
        self._detected_watcher = d.get("watcher")
        self._detected_label.setText(
            f"Detected: {profile.get('name', '?')} (via {method}). "
            f"{profile.get('description', '')}"
        )

    def _browse(self, field: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select folder", field.text() or "")
        if path:
            field.setText(path)

    def _save(self):
        proc = self._proc.text().strip()
        if not proc:
            show_error("Processing root is required.")
            return
        nas = self._nas.text().strip() or None
        try:
            from mousereach.setup.wizard import save_config
            save_config(nas_drive=nas, processing_root=Path(proc), watcher=self._detected_watcher)
        except Exception as e:
            show_error(f"Could not save configuration: {e}")
            return
        msg = "Saved to ~/.mousereach/config.json. Restart MouseReach for it to take effect."
        show_info(msg)
        self._status.setText("[OK] " + msg)
        self._status.setStyleSheet("color:#1a5; font-weight:bold;")
        if self._on_saved:
            try:
                self._on_saved()
            except Exception:
                pass


def main():
    """Standalone launch of the setup panel."""
    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(SetupWidget(viewer), name="Setup", area="right")
    napari.run()


if __name__ == "__main__":
    main()
