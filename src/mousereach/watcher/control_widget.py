"""Watcher Control tab -- drive the auto-processing daemon from the GUI.

"The GUI is god": everything the command-line watcher does must be doable here.
This panel lets an operator, without touching a terminal:

  * START / STOP the background auto-processor (the watcher daemon),
  * RUN ONCE (one scan+drain, then stop) for a quick manual push,
  * PAUSE / RESUME it (e.g. during filming) without stopping it,
  * watch live pipeline status (how many videos in each state, incl. the
    Triage / Deep-Review holds), and
  * view and edit the watcher configuration.

Design notes (from the daemon's own contract):
  * Start/stop is a threading.Event -- the daemon loops until the event is set.
    We run orchestrator.run(event) on a background thread and STOP by setting the
    event; it halts gracefully AFTER the current work item (which can take
    minutes for DLC/pipeline items).
  * Pause is a sentinel file (watcher_paused.flag) the loop checks every cycle --
    independent of start/stop, and visible to a daemon in any process/node.
  * Config is read from ~/.mousereach/config.json at import time, so edits apply
    on the NEXT (re)start, not live. The UI says so.

ASCII-only for any console output (Windows cp1252). Qt widget text may use
Unicode, but this module prints nothing to the terminal.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QComboBox, QSpinBox, QCheckBox, QLineEdit,
)
from qtpy.QtCore import Qt, QTimer, QObject, Signal
from qtpy.QtGui import QColor, QBrush

from napari.utils.notifications import show_info, show_error

logger = logging.getLogger(__name__)


class _BackupWorker(QObject):
    """Carries the backup-finished message from the worker thread to the GUI."""
    done = Signal(str)

# States that are the reviewer's actionable holds / problems -- highlighted.
_HOLD_STATES = {"triage", "deep_review"}
_BAD_STATES = {"failed", "quarantined"}


class WatcherControlWidget(QWidget):
    """The pipeline's control panel: run/monitor/configure the watcher daemon."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self._thread: Optional[threading.Thread] = None
        self._shutdown_event: Optional[threading.Event] = None
        self._orchestrator = None
        self._run_error: Optional[str] = None

        self._build_ui()

        # Deferred first refresh + periodic polling (guarded; DB may be on a NAS).
        QTimer.singleShot(150, self._refresh)
        self._poll = QTimer(self)
        self._poll.setInterval(3000)
        self._poll.timeout.connect(self._refresh)
        self._poll.start()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- Daemon control ---
        ctrl = QGroupBox("Auto-processor (watcher daemon)")
        cl = QVBoxLayout(ctrl)

        self._status_label = QLabel("Status: unknown")
        self._status_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        cl.addWidget(self._status_label)

        self._mode_label = QLabel("")
        self._mode_label.setStyleSheet("color: #888;")
        cl.addWidget(self._mode_label)

        # Which watcher to launch on this machine
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Run as:"))
        self._mode_select = QComboBox()
        self._mode_select.addItem(
            "MouseReach -- process videos (segmentation/reach/outcome/kinematics)",
            "processing_server")
        self._mode_select.addItem(
            "DLC -- crop collages + run pose estimation (needs a GPU)", "dlc_pc")
        self._mode_select.setToolTip("Pick which of the two watchers to launch here.")
        mode_row.addWidget(self._mode_select, 1)
        cl.addLayout(mode_row)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setStyleSheet("background:#1a5; color:white; font-weight:bold;")
        self._start_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setStyleSheet("background:#a33; color:white; font-weight:bold;")
        self._stop_btn.clicked.connect(self._stop)
        self._once_btn = QPushButton("Run Once")
        self._once_btn.clicked.connect(self._run_once)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._toggle_pause)
        for b in (self._start_btn, self._stop_btn, self._once_btn, self._pause_btn):
            btn_row.addWidget(b)
        cl.addLayout(btn_row)
        root.addWidget(ctrl)

        # --- Live status ---
        stat = QGroupBox("Live pipeline status")
        sl = QVBoxLayout(stat)
        self._totals_label = QLabel("(loading...)")
        sl.addWidget(self._totals_label)

        self._state_table = QTableWidget(0, 2)
        self._state_table.setHorizontalHeaderLabels(["State", "Videos"])
        self._state_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._state_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._state_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._state_table.setMaximumHeight(320)
        sl.addWidget(self._state_table)

        sl.addWidget(QLabel("Recent activity:"))
        self._activity = QTextEdit()
        self._activity.setReadOnly(True)
        self._activity.setMaximumHeight(140)
        sl.addWidget(self._activity)

        refresh_btn = QPushButton("Refresh now")
        refresh_btn.clicked.connect(self._refresh)
        sl.addWidget(refresh_btn)
        root.addWidget(stat)

        # --- Config editor ---
        cfg = QGroupBox("Configuration (applies on next start)")
        form = QFormLayout(cfg)
        self._f_enabled = QCheckBox()
        self._f_mode = QComboBox(); self._f_mode.addItems(["dlc_pc", "processing_server"])
        self._f_poll = QSpinBox(); self._f_poll.setRange(1, 3600)
        self._f_stability = QSpinBox(); self._f_stability.setRange(0, 3600)
        self._f_retries = QSpinBox(); self._f_retries.setRange(0, 20)
        self._f_maxpending = QSpinBox(); self._f_maxpending.setRange(1, 100000)
        self._f_gpu = QSpinBox(); self._f_gpu.setRange(-1, 16)
        self._f_autoarchive = QCheckBox()
        self._f_alsoprocess = QCheckBox()
        self._f_dlccfg = QLineEdit()
        self._f_quarantine = QLineEdit()
        self._f_logdir = QLineEdit()
        self._f_dbpath = QLineEdit()
        self._f_staging = QLineEdit()
        form.addRow("enabled", self._f_enabled)
        form.addRow("mode", self._f_mode)
        form.addRow("poll_interval_seconds", self._f_poll)
        form.addRow("stability_wait_seconds", self._f_stability)
        form.addRow("max_retries", self._f_retries)
        form.addRow("max_local_pending", self._f_maxpending)
        form.addRow("dlc_gpu_device", self._f_gpu)
        form.addRow("auto_archive_approved", self._f_autoarchive)
        form.addRow("also_process", self._f_alsoprocess)
        form.addRow("dlc_config_path", self._f_dlccfg)
        form.addRow("quarantine_dir", self._f_quarantine)
        form.addRow("log_dir", self._f_logdir)
        form.addRow("db_path", self._f_dbpath)
        form.addRow("staging_path", self._f_staging)

        cfg_btns = QHBoxLayout()
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._load_config_into_form)
        save_btn = QPushButton("Save config")
        save_btn.setStyleSheet("font-weight:bold;")
        save_btn.clicked.connect(self._save_config)
        cfg_btns.addWidget(reload_btn)
        cfg_btns.addWidget(save_btn)
        form.addRow(cfg_btns)
        root.addWidget(cfg)

        # --- Shipped algorithm/model versions (defines what counts as "current") ---
        ver = QGroupBox("Shipped algorithm versions (editing marks older videos Outdated)")
        vform = QFormLayout(ver)
        self._ver_fields = {}
        for key in ("dlc_scorer", "segmenter", "reach_detector", "outcome_detector", "assignment"):
            fld = QLineEdit()
            self._ver_fields[key] = fld
            vform.addRow(key, fld)
        vbtns = QHBoxLayout()
        vreload = QPushButton("Reload")
        vreload.clicked.connect(self._load_versions)
        vsave = QPushButton("Save versions")
        vsave.setStyleSheet("font-weight:bold;")
        vsave.clicked.connect(self._save_versions)
        vbtns.addWidget(vreload)
        vbtns.addWidget(vsave)
        vform.addRow(vbtns)
        root.addWidget(ver)

        # --- Backups (copy inputs + final outputs to a second drive) ---
        bkp = QGroupBox("Backups (copy pipeline data to a second drive)")
        bform = QFormLayout(bkp)
        self._bkp_enabled = QCheckBox()
        self._bkp_source = QLineEdit()
        self._bkp_source.setPlaceholderText(r"e.g. the NAS root")
        self._bkp_dest = QLineEdit()
        self._bkp_dest.setPlaceholderText(r"e.g. <backup root>")
        bform.addRow("enabled", self._bkp_enabled)
        bform.addRow("source (this drive)", self._bkp_source)
        bform.addRow("backup drive", self._bkp_dest)
        bbtns = QHBoxLayout()
        bsave = QPushButton("Save backup settings")
        bsave.clicked.connect(self._save_backup)
        brun = QPushButton("Back up now")
        brun.setStyleSheet("font-weight:bold;")
        brun.clicked.connect(self._run_backup)
        bbtns.addWidget(bsave)
        bbtns.addWidget(brun)
        bform.addRow(bbtns)
        self._bkp_status = QLabel("")
        self._bkp_status.setWordWrap(True)
        bform.addRow(self._bkp_status)
        root.addWidget(bkp)

        self._load_config_into_form()
        self._load_versions()
        self._load_backup()

    # ------------------------------------------------------------- helpers
    def _pause_file(self) -> Optional[Path]:
        try:
            from mousereach.config import require_processing_root
            return require_processing_root() / "watcher_paused.flag"
        except Exception:
            return None

    def _is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _is_paused(self) -> bool:
        pf = self._pause_file()
        return bool(pf and pf.exists())

    def _db(self):
        """Open the watcher DB if it exists, else None (do NOT create it)."""
        try:
            from mousereach.config import Paths, WatcherConfig
            from mousereach.watcher.db import WatcherDB
            cfg = WatcherConfig.load()
            db_path = cfg.db_path or (Paths.PROCESSING_ROOT / "watcher.db" if Paths.PROCESSING_ROOT else None)
            if not db_path or not Path(db_path).exists():
                return None
            return WatcherDB(Path(db_path))
        except Exception as e:
            logger.debug(f"watcher db unavailable: {e}")
            return None

    # ------------------------------------------------------------- daemon
    def _start(self):
        if self._is_running():
            show_info("Watcher is already running.")
            return
        try:
            from mousereach.config import WatcherConfig, Paths, require_processing_root
            from mousereach.watcher.db import WatcherDB
            from mousereach.watcher.orchestrator import (
                ProcessingOrchestrator, DLCOrchestrator,
            )
            cfg = WatcherConfig.load()
            cfg.mode = self._mode_select.currentData() or cfg.mode  # explicit choice wins
            db_path = cfg.db_path or (require_processing_root() / "watcher.db")
            db = WatcherDB(Path(db_path))
            if cfg.mode == "processing_server":
                self._orchestrator = ProcessingOrchestrator(cfg, db)
            else:
                self._orchestrator = DLCOrchestrator(cfg, db)
            self._run_error = None
            self._shutdown_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run_orchestrator, args=(self._shutdown_event,),
                daemon=True, name="watcher-daemon",
            )
            self._thread.start()
            show_info(f"Watcher started ({cfg.mode}).")
        except Exception as e:
            show_error(f"Could not start watcher: {e}")
            logger.exception("watcher start failed")
        self._refresh()

    def _run_orchestrator(self, evt: threading.Event):
        # NOTE: no signal handlers here -- signal.signal() is main-thread only.
        try:
            self._orchestrator.run(evt)
        except Exception as e:
            self._run_error = str(e)
            logger.exception("watcher daemon crashed")

    def _stop(self):
        if not self._is_running():
            show_info("Watcher is not running.")
            self._refresh()
            return
        if self._shutdown_event:
            self._shutdown_event.set()
        show_info("Stop requested -- the watcher will halt after the current item finishes.")
        self._refresh()

    def _run_once(self):
        if self._is_running():
            show_info("Watcher is already running; Run Once is for when it is stopped.")
            return
        mode = self._mode_select.currentData()
        def _once():
            try:
                from mousereach.config import WatcherConfig, require_processing_root
                from mousereach.watcher.db import WatcherDB
                from mousereach.watcher.orchestrator import (
                    ProcessingOrchestrator, DLCOrchestrator,
                )
                cfg = WatcherConfig.load()
                cfg.mode = mode or cfg.mode
                db_path = cfg.db_path or (require_processing_root() / "watcher.db")
                db = WatcherDB(Path(db_path))
                orch = (ProcessingOrchestrator(cfg, db) if cfg.mode == "processing_server"
                        else DLCOrchestrator(cfg, db))
                orch.run_once()
            except Exception as e:
                self._run_error = str(e)
                logger.exception("run_once failed")
        self._thread = threading.Thread(target=_once, daemon=True, name="watcher-runonce")
        self._thread.start()
        show_info("Run Once started (one scan + drain).")
        self._refresh()

    def _toggle_pause(self):
        pf = self._pause_file()
        if pf is None:
            show_error("Processing root not configured -- cannot pause.")
            return
        try:
            if pf.exists():
                pf.unlink()
                show_info("Watcher resumed.")
            else:
                pf.parent.mkdir(parents=True, exist_ok=True)
                pf.write_text("Watcher paused via GUI.\n", encoding="utf-8")
                show_info("Watcher paused (it keeps running but skips work).")
        except Exception as e:
            show_error(f"Could not toggle pause: {e}")
        self._refresh()

    # ------------------------------------------------------------- status
    def _refresh(self):
        running = self._is_running()
        paused = self._is_paused()
        if running and paused:
            txt, color = "Status: RUNNING (PAUSED)", "#c80"
        elif running:
            txt, color = "Status: RUNNING", "#1a5"
        else:
            txt, color = "Status: STOPPED", "#a33"
        if self._run_error:
            txt += f"  --  last error: {self._run_error}"
        self._status_label.setText(txt)
        self._status_label.setStyleSheet(f"font-weight:bold; font-size:14px; color:{color};")

        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._once_btn.setEnabled(not running)
        self._pause_btn.setText("Resume" if paused else "Pause")

        try:
            from mousereach.config import WatcherConfig, Paths
            cfg = WatcherConfig.load()
            db_path = cfg.db_path or (Paths.PROCESSING_ROOT / "watcher.db" if Paths.PROCESSING_ROOT else "?")
            self._mode_label.setText(f"mode: {cfg.mode}   |   db: {db_path}")
        except Exception:
            pass

        self._refresh_stats()

    def _refresh_stats(self):
        db = self._db()
        if db is None:
            self._totals_label.setText("No watcher database yet (nothing processed on this node).")
            self._state_table.setRowCount(0)
            self._activity.setPlainText("")
            return
        try:
            summary = db.get_pipeline_summary()
        except Exception as e:
            self._totals_label.setText(f"(status unavailable: {e})")
            return
        vids = summary.get("videos", {})
        by_state = vids.get("by_state", {}) or {}
        self._totals_label.setText(
            f"Videos: {vids.get('total', 0)} total   |   "
            f"failed: {vids.get('failed', 0)}   |   quarantined: {vids.get('quarantined', 0)}"
        )
        rows = sorted(by_state.items(), key=lambda kv: (-kv[1], kv[0]))
        self._state_table.setRowCount(len(rows))
        for i, (state, count) in enumerate(rows):
            s_item = QTableWidgetItem(state)
            c_item = QTableWidgetItem(str(count))
            if state in _HOLD_STATES:
                for it in (s_item, c_item):
                    it.setForeground(QBrush(QColor("#c80")))
            elif state in _BAD_STATES:
                for it in (s_item, c_item):
                    it.setForeground(QBrush(QColor("#a33")))
            self._state_table.setItem(i, 0, s_item)
            self._state_table.setItem(i, 1, c_item)

        try:
            log_rows = db.get_recent_log(limit=15)
            lines = []
            for r in log_rows:
                ts = str(r.get("created_at", ""))[:19]
                lines.append(
                    f"{ts}  {r.get('video_id', '')}  {r.get('step', '')} "
                    f"[{r.get('status', '')}]  {r.get('message', '')}"
                )
            self._activity.setPlainText("\n".join(lines))
        except Exception:
            pass

    # ------------------------------------------------------------- config
    def _load_config_into_form(self):
        try:
            from mousereach.config import WatcherConfig
            d = WatcherConfig.load().to_dict()
        except Exception as e:
            show_error(f"Could not load config: {e}")
            return
        self._f_enabled.setChecked(bool(d.get("enabled", False)))
        self._f_mode.setCurrentText(str(d.get("mode", "dlc_pc")))
        _mi = self._mode_select.findData(str(d.get("mode", "dlc_pc")))
        if _mi >= 0:
            self._mode_select.setCurrentIndex(_mi)
        self._f_poll.setValue(int(d.get("poll_interval_seconds", 30)))
        self._f_stability.setValue(int(d.get("stability_wait_seconds", 60)))
        self._f_retries.setValue(int(d.get("max_retries", 3)))
        self._f_maxpending.setValue(int(d.get("max_local_pending", 200)))
        self._f_gpu.setValue(int(d.get("dlc_gpu_device", 0)))
        self._f_autoarchive.setChecked(bool(d.get("auto_archive_approved", False)))
        self._f_alsoprocess.setChecked(bool(d.get("also_process", False)))
        self._f_dlccfg.setText(str(d.get("dlc_config_path", "") or ""))
        self._f_quarantine.setText(str(d.get("quarantine_dir", "") or ""))
        self._f_logdir.setText(str(d.get("log_dir", "") or ""))
        self._f_dbpath.setText(str(d.get("db_path", "") or ""))
        self._f_staging.setText(str(d.get("staging_path", "") or ""))

    def _form_to_dict(self) -> dict:
        d = {
            "enabled": self._f_enabled.isChecked(),
            "mode": self._f_mode.currentText(),
            "poll_interval_seconds": self._f_poll.value(),
            "stability_wait_seconds": self._f_stability.value(),
            "max_retries": self._f_retries.value(),
            "max_local_pending": self._f_maxpending.value(),
            "dlc_gpu_device": self._f_gpu.value(),
            "auto_archive_approved": self._f_autoarchive.isChecked(),
            "also_process": self._f_alsoprocess.isChecked(),
        }
        # Path fields: include only when set.
        for key, widget in (
            ("dlc_config_path", self._f_dlccfg),
            ("quarantine_dir", self._f_quarantine),
            ("log_dir", self._f_logdir),
            ("db_path", self._f_dbpath),
            ("staging_path", self._f_staging),
        ):
            val = widget.text().strip()
            if val:
                d[key] = val
        return d

    def _save_config(self):
        """Merge-write the watcher section into ~/.mousereach/config.json without
        clobbering nas_drive / processing_root."""
        try:
            cfg_path = Path.home() / ".mousereach" / "config.json"
            existing = {}
            if cfg_path.exists():
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            existing["watcher"] = self._form_to_dict()
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            show_info("Config saved. Restart the watcher for changes to take effect.")
        except Exception as e:
            show_error(f"Could not save config: {e}")

    # ------------------------------------------------------------- versions
    def _load_versions(self):
        try:
            from mousereach.pipeline.versions import get_current_versions
            v = get_current_versions().get("versions", {})
        except Exception as e:
            show_error(f"Could not load versions: {e}")
            return
        for k, fld in self._ver_fields.items():
            fld.setText(str(v.get(k, "") or ""))

    def _save_versions(self):
        updates = {k: fld.text().strip() for k, fld in self._ver_fields.items() if fld.text().strip()}
        if not updates:
            show_error("No versions entered.")
            return
        try:
            from mousereach.pipeline.versions import update_current_versions
            update_current_versions(updates)
        except Exception as e:
            show_error(f"Could not save versions: {e}")
            return
        show_info("Shipped versions updated. Videos processed with older versions now show "
                  "as Outdated -- use the dashboard's 'Reprocess outdated' to bring them current.")

    # ------------------------------------------------------------- backups
    def _load_backup(self):
        try:
            from mousereach.config import _load_config
            b = _load_config().get("backup", {}) or {}
        except Exception:
            b = {}
        self._bkp_enabled.setChecked(bool(b.get("enabled", False)))
        self._bkp_source.setText(str(b.get("source_root", "") or ""))
        self._bkp_dest.setText(str(b.get("backup_root", "") or ""))

    def _save_backup(self):
        import json
        try:
            cfg_path = Path.home() / ".mousereach" / "config.json"
            existing = {}
            if cfg_path.exists():
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            existing["backup"] = {
                "enabled": self._bkp_enabled.isChecked(),
                "source_root": self._bkp_source.text().strip(),
                "backup_root": self._bkp_dest.text().strip(),
            }
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            show_info("Backup settings saved.")
        except Exception as e:
            show_error(f"Could not save backup settings: {e}")

    def _run_backup(self):
        src = self._bkp_source.text().strip()
        dst = self._bkp_dest.text().strip()
        if not src or not dst:
            show_error("Set the source and backup drive first.")
            return
        if getattr(self, "_bkp_thread", None) and self._bkp_thread.is_alive():
            show_info("A backup is already running.")
            return
        self._bkp_status.setText("Backing up (copying changed files)... this can take a while.")
        worker = _BackupWorker()
        worker.done.connect(self._on_backup_done)
        self._bkp_worker = worker

        def job():
            try:
                from mousereach.watcher.backup import BackupWatcher
                BackupWatcher(source_root=src, backup_root=dst).run_once()
                worker.done.emit("Backup complete.")
            except Exception as e:
                worker.done.emit(f"Backup failed: {e}")

        self._bkp_thread = threading.Thread(target=job, daemon=True, name="backup-run")
        self._bkp_thread.start()

    def _on_backup_done(self, msg: str):
        self._bkp_status.setText(msg)


def main():
    """Standalone launch of just the Watcher Control panel."""
    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(WatcherControlWidget(viewer), name="Watcher Control", area="right")
    napari.run()


if __name__ == "__main__":
    main()
