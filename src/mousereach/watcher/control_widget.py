"""Watcher Control tab -- drive the auto-processing daemon from the GUI.

"The GUI is god": everything the command-line watcher does must be doable here.
This panel lets an operator, without touching a terminal:

  * START / STOP the background auto-processor (the watcher daemon),
  * RUN ONCE (one scan+drain, then stop) for a quick manual push,
  * PAUSE / RESUME it (e.g. during filming) without stopping it,
  * see WHY it is paused -- by hand, or because a recording program is open,
  * watch live pipeline status (how many videos in each state, incl. the
    Triage / Deep-Review holds), and
  * view and edit the watcher configuration, including the list of recording
    programs that pause this PC.

Design notes (from the daemon's own contract):
  * Start/stop is a threading.Event -- the daemon loops until the event is set.
    We run orchestrator.run(event) on a background thread and STOP by setting the
    event; it halts gracefully AFTER the current work item (which can take
    minutes for DLC/pipeline items).
  * Pause is a sentinel file (watcher_paused.flag) the loop checks every cycle --
    independent of start/stop, and visible to a daemon in any process/node.
  * The watcher ALSO pauses itself while any program listed in
    ``pause_while_running`` is open (mousereach.watcher.recording_guard). WHY: a
    GPU node in the behaviour room poses videos whenever nobody is filming, but
    it cannot know the filming schedule -- the open recording program is the
    only reliable signal, and recording must always win. The Pause/Resume
    button cannot override it; closing the recording program is what resumes.
  * Config is read from ~/.mousereach/config.json at import time, so edits apply
    on the NEXT (re)start, not live. The UI says so. Two exceptions: the
    recording-program list and the resume time are followed by a RUNNING
    watcher (BaseOrchestrator._refresh_recording_settings), because a program
    added here must protect the very next recording.
  * The RUNNING/STOPPED line also looks for a watcher started OUTSIDE this panel
    (the command line, a scheduled task, the Restart button). WHY: an operator
    told STOPPED while a watcher is running presses Start and gets two watchers
    doing the same work.

ASCII-only for any console output (Windows cp1252). Qt widget text may use
Unicode, but this module prints nothing to the terminal.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Plain helpers (no Qt) -- kept out of the widget so they are testable headless.
# ---------------------------------------------------------------------------

# Default for "Resume after (seconds)" when the config does not say. Mirrors
# WatcherConfig.pause_resume_grace_seconds. WHY a grace at all: recording
# programs are often closed and reopened between animals; resuming the instant
# one closes would start a 14-minute pose that the next recording then has to
# share the machine with.
DEFAULT_RESUME_GRACE_SECONDS = 120

# The one-line explanation shown under the field. Novice-first: says what the
# setting DOES and what the operator must do, with no jargon.
# WHY it names the crop: a pose in progress is stopped when a listed program
# opens, but a collage crop already running cannot be, and an operator told
# "no pipeline work" would start filming on top of it.
PAUSE_PROGRAMS_HELP = ("While any of these programs is open, this PC starts no "
                       "new pipeline work, and a pose in progress is stopped "
                       "(it is posed again later). A collage crop that has "
                       "already started finishes first. Close the program when "
                       "you are not recording.")

# Said after Save. WHY it separates the two: most settings apply at the next
# watcher start, but the recording-program list reaches a running watcher on
# its own -- an operator told "restart" for it would have no way to do that for
# a watcher started outside this panel.
SAVE_APPLIES_TEXT = ("The recording-program list and the resume time reach a "
                     "running watcher by themselves within about a minute. "
                     "Every other setting applies the next time the watcher "
                     "starts.")

PAUSE_PROGRAMS_TOOLTIP = (
    "Type the program's name as Task Manager shows it on its Details tab, "
    "e.g. recorder.exe. Separate several names with commas. Leave empty to "
    "never pause for a program.")

PAUSE_REASON_TOOLTIP = (
    "If a recording program is named here, close it to let this PC work "
    "again. The Resume button cannot override it: recording always wins.")

# How often the panel looks for a watcher started outside it. WHY cached: the
# check walks every process's command line, which is slow enough on a busy
# machine to stutter the GUI if done on every 3-second refresh.
_EXTERNAL_WATCHER_CHECK_S = 15.0


def parse_program_list(text: Optional[str]) -> List[str]:
    """"recorder.exe, Other Recorder.exe" -> ["recorder.exe", "Other Recorder.exe"].

    Commas, semicolons and line breaks separate names; spaces do NOT, because
    a program's name can contain spaces. A pasted full path is reduced to the
    program name (the watcher matches the running program's name, never its
    folder), surrounding quotes are dropped, and a name repeated in different
    capitals is kept once (matching is case-insensitive, so the copies would
    only be noise).
    """
    raw = (text or "").replace(";", ",").replace("\r", ",").replace("\n", ",")
    out: List[str] = []
    seen = set()
    for chunk in raw.split(","):
        name = chunk.strip().strip('"').strip("'").strip()
        # Keep only the last path component: "some folder\recorder.exe".
        name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(name)
    return out


def format_program_list(names: Optional[Iterable[str]]) -> str:
    """["recorder.exe", "b.exe"] -> "recorder.exe, b.exe" (the line-edit text)."""
    return ", ".join(str(n) for n in (names or []) if str(n).strip())


def pause_label_text(reason: Optional[str]) -> str:
    """The text beside the Pause/Resume button: empty when not paused."""
    return "Paused: %s" % reason if reason else ""


def watching_label_text(names: Optional[Iterable[str]], grace: Optional[int]) -> str:
    """The quiet text beside Pause/Resume when programs are listed but none
    holds the watcher. Empty when nothing is listed.

    WHY: with no line at all, "the check is working and nothing is open" looks
    exactly like "the name was mistyped and nothing will ever pause" -- the
    operator needs to see which names are being watched for.
    """
    names = [str(n) for n in (names or []) if str(n).strip()]
    if not names:
        return ""
    grace = DEFAULT_RESUME_GRACE_SECONDS if grace is None else int(grace)
    return ("Watching for: %s (none open now; work waits %d s after one closes)"
            % (", ".join(names), grace))


def recording_check_text(names: Optional[Iterable[str]],
                         running: Optional[List[str]],
                         error: Optional[str] = None) -> str:
    """What Save says about the listed programs, checked right then.

    WHY: a misspelled name ("recoder.exe", or a window title instead of the
    program name) pauses nothing and shows nothing. Checking at Save, while the
    operator still has the recording program open in front of them, is the
    one moment the mistake can be seen.
    """
    names = [str(n) for n in (names or []) if str(n).strip()]
    if not names:
        return "No recording programs are listed, so this PC never pauses for one."
    if error:
        return ("Could not check which programs are running (%s). While it "
                "cannot check, the watcher stays paused." % error)
    if running:
        return ("Running right now: %s -- the watcher is paused while it stays "
                "open." % ", ".join(running))
    return ("None of these programs is running right now. If your recording "
            "program IS open, the name does not match: copy it from Task "
            "Manager, Details tab.")


def check_programs_now(names: List[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """(running names, None) or (None, why the check failed). A module
    function so tests can replace it without listing real processes."""
    try:
        from mousereach.watcher.recording_guard import running_programs
        return running_programs(names), None
    except ImportError:
        return None, "psutil is not installed"
    except Exception as e:
        return None, str(e)


def run_once_refusal_text(reason: str) -> str:
    """Why Run Once did not start, and what to do. WHY the hand pause blocks
    it too: a paused watcher starts NO new work, whichever button asks -- the
    hand pause was the only filming protection before recording programs
    could be listed, and a Run Once that ignored it posed on top of filming."""
    if reason == _HAND_PAUSE_REASON:
        return ("Not started: the watcher is paused by hand. Press Resume "
                "first, then Run Once.")
    if reason.startswith("cannot check"):
        return ("Not started: %s. Nothing can start until that check works, "
                "because recording must always win." % reason)
    return ("Not started: %s. Close the recording program first; Run Once can "
            "start a little after it closes." % reason)


def status_headline(panel_running: bool, external_pid: Optional[int],
                    pause_reason: Optional[str],
                    run_error: Optional[str] = None) -> Tuple[str, str]:
    """(text, colour) for the big RUNNING / STOPPED line.

    ``external_pid`` is a watcher found running outside this panel. WHY it
    counts as RUNNING: the operator must never be told STOPPED while a
    watcher is working -- they would start a second one.
    """
    running = bool(panel_running or external_pid)
    if running and pause_reason:
        txt, color = "Status: RUNNING (PAUSED)", "#c80"
    elif running:
        txt, color = "Status: RUNNING", "#1a5"
    else:
        txt, color = "Status: STOPPED", "#a33"
    if external_pid and not panel_running:
        txt += "  (started outside this panel)"
    if run_error:
        txt += f"  --  last error: {run_error}"
    return txt, color


# Shown when the pause state itself cannot be worked out. Matches the text
# recording_guard.pause_reason uses for the hand-made flag.
_HAND_PAUSE_REASON = "paused by hand (watcher_paused.flag)"


class WatcherControlWidget(QWidget):
    """The pipeline's control panel: run/monitor/configure the watcher daemon."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self._thread: Optional[threading.Thread] = None
        self._shutdown_event: Optional[threading.Event] = None
        self._orchestrator = None
        self._run_error: Optional[str] = None
        # One RecordingGuard kept for the panel's lifetime, rebuilt only when
        # the configured list changes. WHY: the guard remembers WHEN a program
        # closed, which is how it can say "closed 30 s ago; resuming after
        # 120 s". A fresh guard every refresh would forget that.
        self._guard = None
        self._guard_key = None
        self._ext_pid: Optional[int] = None
        self._ext_checked_at = float("-inf")

        self._build_ui()

        # Deferred first refresh + periodic polling (guarded; DB may be on a NAS).
        # The same timer refreshes the pause reason, so "recorder.exe is
        # running" appears within a few seconds of the program opening.
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
        for b in (self._start_btn, self._stop_btn, self._once_btn):
            btn_row.addWidget(b)
        cl.addLayout(btn_row)

        # Pause/Resume sits with the reason it is paused, so the operator sees
        # at a glance whether the button or a recording program is holding it.
        pause_row = QHBoxLayout()
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._toggle_pause)
        pause_row.addWidget(self._pause_btn)
        self._pause_reason_label = QLabel("")
        self._pause_reason_label.setWordWrap(True)
        self._pause_reason_label.setStyleSheet("color:#c80; font-weight:bold;")
        self._pause_reason_label.setToolTip(PAUSE_REASON_TOOLTIP)
        pause_row.addWidget(self._pause_reason_label, 1)
        cl.addLayout(pause_row)
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
        # Recording programs that pause this PC (empty = never pause for one).
        self._f_pause_programs = QLineEdit()
        self._f_pause_programs.setPlaceholderText("e.g. recorder.exe")
        self._f_pause_programs.setToolTip(PAUSE_PROGRAMS_TOOLTIP)
        # 0 is allowed (resume the moment the program closes); a day is the
        # cap because a longer grace is indistinguishable from "never resume".
        self._f_pause_grace = QSpinBox(); self._f_pause_grace.setRange(0, 86400)
        self._f_pause_grace.setToolTip(
            "After the last recording program closes, wait this many seconds "
            "before starting work again -- in case recording starts again.")
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
        form.addRow("Pause while these programs are running", self._f_pause_programs)
        form.addRow("Resume after (seconds)", self._f_pause_grace)
        pause_help = QLabel(PAUSE_PROGRAMS_HELP)
        pause_help.setWordWrap(True)
        pause_help.setStyleSheet("color:#888;")
        form.addRow(pause_help)

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

        # --- Work priority: ONE order, for the whole lab ---
        # This is deliberately not a per-machine setting. The queues it orders
        # are shared, so a copy on each machine meant the processing node and
        # the review tools could disagree about what mattered and nothing said
        # so. Saving here writes priority_order.json on the pipeline drive and
        # every node picks it up.
        pri = QGroupBox("Work priority (whole lab -- processing AND human review)")
        pform = QFormLayout(pri)
        self._pri_projects = QLineEdit()
        self._pri_projects.setPlaceholderText("most important first, e.g.  CNT, ASPA")
        self._pri_cohorts = QLineEdit()
        self._pri_cohorts.setPlaceholderText("per project, e.g.  CNT: 01, 02, 03, 04")
        self._pri_trays = QLineEdit()
        self._pri_trays.setPlaceholderText("tray letters worked first, e.g.  P")
        pform.addRow("projects, best first", self._pri_projects)
        pform.addRow("cohorts within a project", self._pri_cohorts)
        pform.addRow("tray types first", self._pri_trays)
        self._pri_status = QLabel("")
        self._pri_status.setWordWrap(True)
        self._pri_status.setStyleSheet("color:#888;")
        pform.addRow(self._pri_status)
        pbtns = QHBoxLayout()
        preload = QPushButton("Reload")
        preload.clicked.connect(self._load_priority_into_form)
        psave = QPushButton("Save priority order")
        psave.setStyleSheet("font-weight:bold;")
        psave.clicked.connect(self._save_priority)
        pbtns.addWidget(preload)
        pbtns.addWidget(psave)
        pform.addRow(pbtns)
        root.addWidget(pri)

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
        self._load_priority_into_form()
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
        """True when paused BY HAND (the flag file). See _current_pause_reason
        for every reason, including an open recording program."""
        pf = self._pause_file()
        return bool(pf and pf.exists())

    def _external_watcher_pid(self) -> Optional[int]:
        """A watcher running outside this panel on this PC, or None.

        Cached for _EXTERNAL_WATCHER_CHECK_S (the scan is slow). This process's
        own id is ignored so the panel never counts itself.
        """
        now = time.monotonic()
        if now - self._ext_checked_at < _EXTERNAL_WATCHER_CHECK_S:
            return self._ext_pid
        pid = None
        try:
            from mousereach.watcher.health import watcher_running
            pid = watcher_running()
        except Exception as e:
            logger.debug(f"external watcher check unavailable: {e}")
        if pid == os.getpid():
            pid = None
        self._ext_pid = pid
        self._ext_checked_at = now
        return pid

    def _recording_guard(self, cfg):
        """The panel's RecordingGuard for ``cfg``'s program list, or None when
        no programs are listed. Rebuilt only when the list or grace changes."""
        names = list(getattr(cfg, "pause_while_running", None) or [])
        if not names:
            self._guard, self._guard_key = None, None
            return None
        grace = getattr(cfg, "pause_resume_grace_seconds", None)
        grace = DEFAULT_RESUME_GRACE_SECONDS if grace is None else int(grace)
        key = (tuple(n.lower() for n in names), grace)
        if self._guard is None or key != self._guard_key:
            # Lazy import: the panel must still open on an install without it.
            from mousereach.watcher.recording_guard import RecordingGuard
            self._guard = RecordingGuard(names, grace_seconds=grace)
            self._guard_key = key
        return self._guard

    def _recording_reason(self, cfg) -> Optional[str]:
        """Why a recording program blocks work right now, or None.

        Fails SAFE: if programs are listed but cannot be checked, that is a
        reason. WHY: recording must always win, so "cannot tell" means "wait".
        """
        try:
            guard = self._recording_guard(cfg)
            return guard.reason() if guard is not None else None
        except Exception as e:
            return "cannot check for recording programs: %s" % e

    def _current_pause_reason(self, cfg) -> Optional[str]:
        """Every reason the watcher is paused (hand flag first), or None.

        Uses recording_guard.pause_reason so the panel shows the same words
        the watcher itself logs.
        """
        pf = self._pause_file()
        try:
            guard = self._recording_guard(cfg)
            from mousereach.watcher.recording_guard import pause_reason
            return pause_reason(pf.parent if pf is not None else None, cfg, guard)
        except Exception as e:
            try:
                if pf is not None and pf.exists():
                    return _HAND_PAUSE_REASON
            except OSError:
                pass
            if getattr(cfg, "pause_while_running", None):
                return "cannot check for recording programs: %s" % e
            return None

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
    def _refuse_if_external_watcher(self) -> bool:
        """Tell the operator and return True when a watcher already runs
        outside this panel. WHY: the panel's watcher runs inside this program
        and does not take the one-watcher-per-PC lock the command-line watcher
        takes, so starting one here would run two watchers doing the same work.
        """
        self._ext_checked_at = float("-inf")  # a button press deserves a fresh look
        if self._external_watcher_pid():
            show_info("A watcher is already running on this PC (started outside "
                      "this panel), so there is nothing to start here -- it "
                      "keeps working on its own. To restart it, press Restart "
                      "processor on the Dashboard.")
            self._refresh()
            return True
        return False

    def _start(self):
        if self._is_running():
            show_info("Watcher is already running.")
            return
        if self._refuse_if_external_watcher():
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
        if self._refuse_if_external_watcher():
            return
        # Every pause blocks Run Once: the hand pause and a recording program
        # alike. The watcher's run_once refuses the same way, so without this
        # check the panel would say "Run Once started" and then nothing would
        # happen, with no word why. See run_once_refusal_text for the reason.
        try:
            from mousereach.config import WatcherConfig
            cfg = WatcherConfig.load()
        except Exception:
            cfg = None
        reason = self._current_pause_reason(cfg)
        if reason:
            show_info(run_once_refusal_text(reason))
            self._refresh()
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
                # Resume only lifts the HAND pause. Say so when a recording
                # program still holds the watcher, or the operator is told
                # "resumed" and then watches nothing happen.
                try:
                    from mousereach.config import WatcherConfig
                    recording = self._recording_reason(WatcherConfig.load())
                except Exception:
                    recording = None
                if recording:
                    show_info("Hand pause removed, but the watcher stays paused: "
                              "%s. It starts again once the recording program "
                              "is closed." % recording)
                else:
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
        cfg = None
        try:
            from mousereach.config import WatcherConfig, Paths
            cfg = WatcherConfig.load()
            db_path = cfg.db_path or (Paths.PROCESSING_ROOT / "watcher.db" if Paths.PROCESSING_ROOT else "?")
            self._mode_label.setText(f"mode: {cfg.mode}   |   db: {db_path}")
        except Exception:
            pass

        panel_running = self._is_running()
        # Only look outside when the panel's own watcher is not running: the
        # scan is slow, and the panel's thread already answers the question.
        external_pid = None if panel_running else self._external_watcher_pid()
        running = bool(panel_running or external_pid)
        reason = self._current_pause_reason(cfg)

        txt, color = status_headline(panel_running, external_pid, reason, self._run_error)
        self._status_label.setText(txt)
        self._status_label.setStyleSheet(f"font-weight:bold; font-size:14px; color:{color};")

        self._start_btn.setEnabled(not running)
        # Stop acts only on the panel's own watcher; one started elsewhere is
        # stopped where it was started.
        self._stop_btn.setEnabled(panel_running)
        self._once_btn.setEnabled(not running)
        # The button controls only the hand pause, so its label follows the
        # flag; the reason label beside it shows every cause -- or, when
        # nothing holds the watcher, quietly which programs it watches for.
        self._pause_btn.setText("Resume" if self._is_paused() else "Pause")
        if reason:
            self._pause_reason_label.setStyleSheet("color:#c80; font-weight:bold;")
            self._pause_reason_label.setText(pause_label_text(reason))
        else:
            self._pause_reason_label.setStyleSheet("color:#888;")
            self._pause_reason_label.setText(watching_label_text(
                getattr(cfg, "pause_while_running", None),
                getattr(cfg, "pause_resume_grace_seconds", None)))

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
        self._f_pause_programs.setText(format_program_list(d.get("pause_while_running")))
        grace = d.get("pause_resume_grace_seconds")
        self._f_pause_grace.setValue(
            DEFAULT_RESUME_GRACE_SECONDS if grace is None else int(grace))

    # Every watcher setting this form owns. Anything else in the config file
    # is preserved on save. WHY: the form rebuilds the watcher section from
    # its own fields, so a setting with no widget -- work_priority, and today
    # also mode, db_path, max_local_pending, staging_path when they were
    # hand-written -- was silently deleted the first time somebody pressed
    # Save. Listing the owned keys (rather than merging everything) keeps the
    # form's ability to CLEAR a path by blanking it -- and, the same way, to
    # clear the recording-program list, which turns the pause off.
    _FORM_MANAGED_KEYS = frozenset({
        "enabled", "mode", "poll_interval_seconds", "stability_wait_seconds",
        "max_retries", "max_local_pending", "dlc_gpu_device",
        "auto_archive_approved", "also_process", "dlc_config_path",
        "quarantine_dir", "log_dir", "db_path", "staging_path",
        "pause_while_running", "pause_resume_grace_seconds",
    })

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
            "pause_resume_grace_seconds": self._f_pause_grace.value(),
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
        # Written only when non-empty, like WatcherConfig.to_dict. WHY: an
        # absent list is the shipped default (never pause for a program), so a
        # PC nobody configured keeps a config file that says nothing new.
        programs = parse_program_list(self._f_pause_programs.text())
        if programs:
            d["pause_while_running"] = programs
        return d

    def _save_config(self):
        """Merge-write the watcher section into ~/.mousereach/config.json without
        clobbering nas_drive / processing_root."""
        try:
            cfg_path = Path.home() / ".mousereach" / "config.json"
            existing = {}
            if cfg_path.exists():
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            preserved = {k: v for k, v in (existing.get("watcher") or {}).items()
                         if k not in self._FORM_MANAGED_KEYS}
            preserved.update(self._form_to_dict())
            existing["watcher"] = preserved
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception as e:
            show_error(f"Could not save config: {e}")
            return
        programs = preserved.get("pause_while_running") or []
        running, error = check_programs_now(programs) if programs else (None, None)
        show_info("Config saved. %s %s" % (
            SAVE_APPLIES_TEXT, recording_check_text(programs, running, error)))

    # ------------------------------------------------------------- priority
    @staticmethod
    def _parse_cohort_text(text: str) -> dict:
        """"CNT: 01, 02; ASPA: 05" -> {"CNT": ["01","02"], "ASPA": ["05"]}.

        Commas and spaces both separate, so nobody has to remember which.
        A chunk with no project prefix is skipped rather than guessed at.
        """
        out = {}
        for chunk in (text or "").split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            project, rest = chunk.split(":", 1)
            project = project.strip()
            values = [v for v in rest.replace(",", " ").split() if v]
            if project and values:
                out[project] = values
        return out

    @staticmethod
    def _format_cohort_text(cohorts: dict) -> str:
        return "; ".join("%s: %s" % (p, ", ".join(v))
                         for p, v in (cohorts or {}).items() if v)

    @staticmethod
    def _split_list(text: str) -> list:
        return [v for v in (text or "").replace(",", " ").split() if v]

    def _load_priority_into_form(self):
        """Show the lab-wide order, and what it currently resolves to."""
        try:
            from mousereach.watcher.work_priority import (
                lab_priority_path, load_lab_policy, read_lab_priority,
            )
            raw = read_lab_priority() or {}
            self._pri_projects.setText(", ".join(raw.get("projects") or []))
            self._pri_cohorts.setText(self._format_cohort_text(raw.get("cohorts")))
            self._pri_trays.setText(", ".join(raw.get("tray_types") or ["P"]))
            policy = load_lab_policy(max_age_s=0)
            where = lab_priority_path()
            lines = policy.describe()
            if not raw:
                lines.append("no lab file yet at %s -- saving here creates it"
                             % (where or "(no shared drive configured)"))
            self._pri_status.setText("\n".join(lines))
        except Exception as e:
            self._pri_status.setText("Could not read the lab order: %s" % e)

    def _save_priority(self):
        """Write the lab-wide order to the shared drive."""
        try:
            from mousereach.config import FilePatterns
            from mousereach.watcher.work_priority import (
                read_lab_priority, save_lab_priority,
            )
            existing = read_lab_priority() or {}
            trays = [t.upper() for t in self._split_list(self._pri_trays.text())]
            path = save_lab_priority(
                projects=self._split_list(self._pri_projects.text()),
                cohorts=self._parse_cohort_text(self._pri_cohorts.text()),
                tray_types=trays or None,
                # Not on the form: keep whatever the file already said, and
                # fall back to the shipped "unsupported trays wait" rule.
                idle_only_tray_types=(existing.get("idle_only_tray_types")
                                      or list(FilePatterns.UNSUPPORTED_TRAY_TYPES)),
            )
            self._load_priority_into_form()
            show_info("Priority order saved to %s. Every machine reads it -- "
                      "running watchers pick it up within a minute, and the "
                      "review tools use it for the next video they hand out."
                      % path)
        except Exception as e:
            show_error("Could not save the priority order: %s" % e)

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
