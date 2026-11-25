"""The "2 - Run Pipeline" tab: run the pipeline by hand from a window.

WHAT IT DOES
------------
Two buttons. "Run the pipeline now" (everything waiting) runs ONE pass of
this machine's own watcher -- exactly what ``mousereach-watch --once`` and the Watcher Control
tab's "Run Once" run -- and stops when nothing is left waiting. What that pass
does depends on the role this machine is configured for (``watcher.mode`` in
~/.mousereach/config.json, shown on the tab):

On a pose-only graphics-card PC (``dlc_pc``, ``also_process`` false) the pass
is the pose pass: it cuts waiting recordings into single-animal videos, poses
them and stages each pose for the processing machine. It does NOT run
segmentation or anything after it; the analysis below happens on the
processing machine. (A ``dlc_pc`` with ``also_process`` true also analyses
what it posed and files it, as below.)

On the processing machine (``processing_server``):

  * every posed video waiting on this machine goes through segmentation,
    reach detection, the outcome cascade, reach assignment, the quality
    check, the hand-score comparison and the review gate, in the watcher's
    order;
  * a video the gate holds goes to its review queue (triage or manual
    segmentation), with its watcher record, as it would overnight;
  * a video that passed every check gets its kinematics and is filed;
  * videos coming back from review go on from where they were sent.

"Run chosen videos..." asks for one or more video files and runs each one
through the same steps, now (watcher/run_one.py: run_chosen_videos -- the
watcher's own handlers for that one video, as the old Run All Steps tab did).
On a single machine (``standalone``) a chosen video that has no pose yet is
posed first by the machine's posing half, and a chosen multi-animal recording
in the intake folder is cut first and each of its videos run. A filed video, a
video held in a review queue and a crystallized one are refused with the
reason (filed means final; a held video goes on once its review is finished).

No automatic watcher has to be running. The pass refuses to start, with the
reason shown, where the watcher itself would not start work: another watcher
is already running on this PC, the watcher is paused (by hand or because a
recording program is open), a stop was requested, or the machine's role is
not configured.

WHY IT IS BUILT THIS WAY
------------------------
Running the pipeline by hand from a window is the most important capability
the tool has, so the tab stays. It does not run its own copy of the steps
(an older copy would run the old outcome detector BEFORE reach detection, in
the folder the watcher works in, with no reach assignment, no hand-score
check, no gate, no review queue and no watcher record).

So there is ONE implementation and two ways to start it. This tab IS the
Watcher Control panel's run-once (it subclasses WatcherControlWidget and keeps
its guards and its run-once unchanged) with a smaller screen: same steps, same
order, same gates and holds, same outputs, same refusals. A change to how the
watcher runs a video reaches this button with no second copy to keep in step.

CHOOSING WHICH VIDEOS GO FIRST. The "Do one animal's videos first" box puts
one animal first, through the watcher's own priority (watcher/priority_animal.py,
the same file ``mousereach-watch-prioritize`` writes): the next passes take
that animal's videos before any other, with every step, check and hold
unchanged. That is how a person has chosen videos done now from a window.

CHOSEN VIDEOS. Everything must be runnable by hand, easily, from the window
and the command line. "Run chosen videos..." runs each chosen
video through the watcher's own steps; what it does with a filed, held or
crystallized video is what the watcher does (refuse, and say why). There is
no "Reprocess All": it would delete every segments, reaches and outcomes
file in Processing, a person's hand-set cuts included, and filed results are
final (a filed video is run again only by a person's explicit command, e.g.
the dashboard's "Re-run video"). There is no separate Run All Steps tab:
this button does the same for one or many.

ASCII-only console output (Windows cp1252). Qt widget text may use Unicode,
but this module prints nothing to the terminal.
"""
from __future__ import annotations

import logging
import threading

from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
    QTableWidget, QTextEdit, QHeaderView,
)

from qtpy.QtCore import QObject, Signal

from mousereach.watcher.control_widget import WatcherControlWidget, status_headline

logger = logging.getLogger(__name__)


class _ChosenRelay(QObject):
    """Carries lines from the chosen-videos thread to the tab (queued signals)."""
    line = Signal(str)
    done = Signal()


# One line per chosen video: the same words as mousereach-run-videos.
from mousereach.watcher.run_one import chosen_summary_line  # noqa: E402


RUN_PIPELINE_INTRO = (
    "Runs one pass of this machine's watcher on everything waiting here, then "
    "stops: the same steps the automatic watcher runs on this machine, in the "
    "same order, with the same checks. What that is depends on this machine's "
    "role (shown below). On the processing machine: segmentation, reaches, "
    "outcomes, causal reach, quality check, hand-score comparison, review "
    "gate, kinematics, filing -- a video with a problem goes to its review "
    "queue, and a video that passed every check is filed. On a pose-only "
    "graphics-card PC: cutting recordings and posing them, then handing the "
    "poses to the processing machine, which does the analysis. No watcher "
    "has to be running."
)


def role_explanation(mode, also_process=False) -> str:
    """One sentence saying what a pass does on a machine of role ``mode``.

    WHY: the tab runs THIS machine's watcher role. On a pose-only
    graphics-card PC that is the pose pass, not the analysis, and a person
    who pressed the button there must not expect videos to be analysed or
    filed by it."""
    if mode == "standalone":
        return ("This machine is the lab's only MouseReach machine: a pass cuts and "
                "poses waiting recordings, then analyses every posed video, takes "
                "back videos a person has finished reviewing, sends a video with a "
                "problem to its review queue and files a video that passed every "
                "check.")
    if mode == "processing_server":
        return ("This machine is the processing machine: a pass analyses the "
                "posed videos waiting here, sends a video with a problem to its "
                "review queue and files a video that passed every check.")
    if mode == "dlc_pc" and also_process:
        return ("This machine is a graphics-card PC that also analyses: a pass "
                "cuts and poses waiting recordings, then analyses what it posed "
                "and files it (or sends it to its review queue), as the "
                "processing machine would.")
    if mode == "dlc_pc":
        return ("This machine is a pose-only graphics-card PC: a pass cuts waiting "
                "recordings and poses them, and hands each pose to the processing "
                "machine. It does not analyse or file anything; that happens on "
                "the processing machine.")
    return f"This machine's role ({mode}) is not one the pipeline knows."

RUN_PIPELINE_BUTTON_TEXT = "Run the pipeline now"
RUN_CHOSEN_BUTTON_TEXT = "Run chosen videos..."

# Shown under the two buttons (module docstring).
SELECTION_NOTE = (
    "\"Run the pipeline now\" runs everything waiting on this machine. \"Run "
    "chosen videos...\" asks for video files and runs each one now, through the "
    "same steps with every check (on a single machine a video with no pose yet is "
    "posed first, and a multi-animal recording in the intake folder is cut "
    "first). A filed video, a video held in a review queue and a crystallized one "
    "are not run, and the reason is shown: filed results are final, and a held "
    "video goes on by itself once its review is finished. After you finish a "
    "review, press \"Run the pipeline now\" to take the video back (the automatic "
    "watcher does it by itself). Same as: mousereach-watch --once (everything) and "
    "mousereach-run-videos <file> ... (chosen)."
)

PRIORITY_INTRO = (
    "Do one animal's videos first: type the animal (e.g. ABC0101) and press "
    "\"Put first\". The next passes take that animal's videos before any other; "
    "every step and check is the same. It stays in force -- for the automatic "
    "watcher too, overnight included -- until you press \"Back to normal order\". "
    "Same as mousereach-watch-prioritize."
)


class UnifiedPipelineWidget(WatcherControlWidget):
    """Run the pipeline by hand: one pass of this machine's watcher.

    Everything that decides what happens to a video is inherited from
    WatcherControlWidget (``_run_once`` and the guards it calls); this class
    only lays out a smaller screen. See the module docstring for why.
    """

    def _build_ui(self):
        root = QVBoxLayout(self)

        header = QLabel("<b>Run the pipeline by hand</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        root.addWidget(header)

        intro = QLabel(RUN_PIPELINE_INTRO)
        intro.setWordWrap(True)
        root.addWidget(intro)

        self._status_label = QLabel("Status: unknown")
        self._status_label.setStyleSheet("font-weight: bold;")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        self._mode_label = QLabel("")
        self._mode_label.setStyleSheet("font-weight: bold;")
        self._mode_label.setWordWrap(True)
        root.addWidget(self._mode_label)

        # The role is this machine's configured one; _take_start_guards reads
        # the "Run as" box when there is one, and there is none on this tab.
        self._mode_select = None

        self.run_btn = QPushButton(RUN_PIPELINE_BUTTON_TEXT)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 12px; font-size: 13px;")
        self.run_btn.setToolTip(
            "One pass of this machine's watcher (the same as Run Once on the "
            "Watcher Control tab and mousereach-watch --once).")
        self.run_btn.clicked.connect(self._run_once)
        root.addWidget(self.run_btn)

        self._pause_reason_label = QLabel("")
        self._pause_reason_label.setWordWrap(True)
        self._pause_reason_label.setStyleSheet("color:#c80; font-weight:bold;")
        root.addWidget(self._pause_reason_label)

        prio = QGroupBox("Do one animal's videos first")
        pl = QVBoxLayout(prio)
        pi = QLabel(PRIORITY_INTRO)
        pi.setWordWrap(True)
        pl.addWidget(pi)
        prow = QHBoxLayout()
        self.priority_edit = QLineEdit()
        self.priority_edit.setPlaceholderText("animal, e.g. ABC0101")
        prow.addWidget(self.priority_edit)
        self.priority_btn = QPushButton("Put first")
        self.priority_btn.clicked.connect(self._set_priority)
        prow.addWidget(self.priority_btn)
        self.priority_clear_btn = QPushButton("Back to normal order")
        self.priority_clear_btn.clicked.connect(self._clear_priority)
        prow.addWidget(self.priority_clear_btn)
        pl.addLayout(prow)
        self.priority_label = QLabel("")
        self.priority_label.setWordWrap(True)
        pl.addWidget(self.priority_label)
        root.addWidget(prio)

        # Chosen videos, through the same steps (module docstring).
        self.chosen_btn = QPushButton(RUN_CHOSEN_BUTTON_TEXT)
        self.chosen_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.chosen_btn.setToolTip(
            "Pick one or more video files; each is run now through the watcher's own "
            "steps (the same as mousereach-run-videos).")
        self.chosen_btn.clicked.connect(self._pick_and_run_chosen)
        root.addWidget(self.chosen_btn)
        self._chosen_log = QTextEdit()
        self._chosen_log.setReadOnly(True)
        self._chosen_log.setMaximumHeight(140)
        self._chosen_log.setPlaceholderText("What happened to each chosen video appears here.")
        root.addWidget(self._chosen_log)
        self._chosen_thread = None
        self._chosen_relay = _ChosenRelay()
        self._chosen_relay.line.connect(self._chosen_log.append)
        self._chosen_relay.done.connect(self._chosen_finished)

        selection = QLabel(SELECTION_NOTE)
        selection.setWordWrap(True)
        selection.setStyleSheet("color: #888;")
        root.addWidget(selection)

        stats = QGroupBox("Videos on this machine, by state")
        sl = QVBoxLayout(stats)
        self._totals_label = QLabel("")
        self._totals_label.setWordWrap(True)
        sl.addWidget(self._totals_label)
        self._state_table = QTableWidget(0, 2)
        self._state_table.setHorizontalHeaderLabels(["State", "Videos"])
        self._state_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._state_table.verticalHeader().setVisible(False)
        sl.addWidget(self._state_table)
        sl.addWidget(QLabel("Recent steps:"))
        self._activity = QTextEdit()
        self._activity.setReadOnly(True)
        sl.addWidget(self._activity)
        root.addWidget(stats)

        more = QLabel(
            "Start or stop the automatic watcher, pause it, or change its "
            "settings on the Watcher Control tab. Held videos are reviewed from "
            "the Review Queues tab.")
        more.setWordWrap(True)
        more.setStyleSheet("color: #888;")
        root.addWidget(more)
        root.addStretch()

    # ------------------------------------------------------ chosen videos
    def _pick_and_run_chosen(self):
        from qtpy.QtWidgets import QFileDialog
        start = ""
        try:
            from mousereach.config import Paths
            start = str(Paths.NAS_ROOT or "")
        except Exception:
            pass
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose videos to run", start, "Videos (*.mp4 *.avi *.mkv)")
        if files:
            self.run_chosen(files)

    def run_chosen(self, files) -> None:
        """Run ``files`` on a background thread (watcher/run_one.py:
        run_chosen_videos); each line of what happens is shown on the tab."""
        if self._chosen_thread is not None and self._chosen_thread.is_alive():
            self._chosen_log.append("A run of chosen videos is still going; wait for it.")
            return
        relay = self._chosen_relay

        def _work():
            from mousereach.watcher.run_one import run_chosen_videos
            try:
                results = run_chosen_videos(
                    list(files), progress=lambda st, msg: relay.line.emit(f"  {st}: {msg}"))
                for r in results:
                    relay.line.emit(chosen_summary_line(r))
            except Exception as e:
                relay.line.emit(f"[FAIL] {e}")
            relay.done.emit()

        self._chosen_log.append(f"Running {len(files)} chosen file(s)...")
        self.chosen_btn.setEnabled(False)
        self._chosen_thread = threading.Thread(target=_work, daemon=True,
                                               name="run-chosen-videos")
        self._chosen_thread.start()

    def _chosen_finished(self):
        self.chosen_btn.setEnabled(True)
        self._chosen_log.append("Done.")
        try:
            self._refresh()
        except Exception:
            pass

    def _show_priority(self):
        """Say which animal goes first now, if any."""
        try:
            from mousereach.watcher.priority_animal import current_priority
            data = current_priority()
        except Exception as e:
            self.priority_label.setText(f"Could not read the priority animal: {e}")
            return
        if data:
            self.priority_label.setText(
                f"First now: {data['animal_id']} (set {data.get('set_at', '?')}).")
        else:
            self.priority_label.setText("No animal is first: normal order.")

    def _set_priority(self):
        from mousereach.watcher.priority_animal import set_priority_animal
        try:
            animal = set_priority_animal(self.priority_edit.text())
        except Exception as e:
            self.priority_label.setText(f"Not set: {e}")
            return
        self.priority_label.setText(
            f"First now: {animal}. Press \"{RUN_PIPELINE_BUTTON_TEXT}\".")

    def _clear_priority(self):
        from mousereach.watcher.priority_animal import clear_priority
        try:
            clear_priority()
        except Exception as e:
            self.priority_label.setText(f"Could not clear it: {e}")
            return
        self._show_priority()

    def _refresh(self):
        """The Watcher Control panel's refresh, for this tab's fewer controls."""
        cfg = None
        try:
            from mousereach.config import WatcherConfig, Paths
            cfg = WatcherConfig.load()
            # The file the watcher and every tool use (watcher/db_location),
            # or why this login cannot name it.
            from mousereach.watcher.db_location import resolve_watcher_db_path
            try:
                db_path = resolve_watcher_db_path()
            except Exception as e:
                db_path = f"NOT NAMED -- {e}"
            self._mode_label.setText(
                f"{role_explanation(cfg.mode, getattr(cfg, 'also_process', False))}"
                f"   (role: {cfg.mode}   |   db: {db_path})")
        except Exception as e:
            self._mode_label.setText(
                f"Not configured ({e}). Use the Setup tab, then come back.")

        panel_running = self._is_running()
        external_pid = None if panel_running else self._external_watcher_pid()
        reason = self._current_pause_reason(cfg)
        txt, color = status_headline(panel_running, external_pid, reason, self._run_error)
        if panel_running:
            txt = "Status: a pass is running"
        self._status_label.setText(txt)
        self._status_label.setStyleSheet(f"font-weight:bold; color:{color};")
        self.run_btn.setEnabled(not (panel_running or external_pid))
        self._pause_reason_label.setText(f"Paused: {reason}" if reason else "")
        self._show_priority()
        self._refresh_stats()


__all__ = ["UnifiedPipelineWidget", "RUN_PIPELINE_INTRO", "RUN_PIPELINE_BUTTON_TEXT",
           "RUN_CHOSEN_BUTTON_TEXT", "SELECTION_NOTE", "PRIORITY_INTRO",
           "role_explanation", "chosen_summary_line"]


def main():
    """Open the tab on its own in a napari window."""
    import napari
    viewer = napari.Viewer(title="MouseReach - Run Pipeline")
    viewer.window.add_dock_widget(UnifiedPipelineWidget(viewer), name="Run Pipeline",
                                  area="right")
    napari.run()


if __name__ == "__main__":
    main()
