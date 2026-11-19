"""The "2 - Run Pipeline" tab runs the watcher's own pass, not a copy of it.

WHY: running the pipeline by hand from a window must do exactly what the
automatic watcher does -- same steps, order, gates, holds and outputs. The tab
used to run its own older processor: the old outcome detector BEFORE reach
detection, no reach assignment, no hand-score check, no gate and no review
queue, in the folder the watcher works in; and its "Reprocess All" deleted
every segments file in Processing, a person's hand-set cuts included.

The tab now starts the watcher's run-once (the code behind Watcher Control's
Run Once and ``mousereach-watch --once``). These tests build the real tab with
no display (Qt offscreen), stand in the watcher's orchestrator, press the
button, and check what ran and what was left on disk.
"""
import json
import os
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("napari")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def tab(tmp_path, monkeypatch):
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    # Imported BEFORE require_processing_root is stood in below. WHY: a module
    # first imported while the stand-in is in place binds it by name
    # (``from ..config import require_processing_root``) and keeps it after
    # the test, so every later test in the run saw a configured machine.
    from mousereach.watcher import orchestrator as _orch, control_widget as _cw  # noqa: F401
    from mousereach.pipeline import batch_widget as _bw  # noqa: F401

    from mousereach.config import Paths, WatcherConfig
    processing = tmp_path / "Processing"
    processing.mkdir()
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(Paths, "PROCESSING", processing, raising=False)
    cfg = SimpleNamespace(mode="processing_server", db_path=tmp_path / "watcher.db",
                          pause_while_running=[], pause_during_recording_hours=False,
                          pause_resume_grace_seconds=None, recording_hours=None,
                          recording_days=None)
    monkeypatch.setattr(WatcherConfig, "load", classmethod(lambda cls: cfg))

    import mousereach.config as mr_config
    monkeypatch.setattr(mr_config, "require_processing_root", lambda: tmp_path)

    # Nothing else is running, nothing pauses the watcher, the lock is free.
    from mousereach.watcher import singleton, health, recording_guard
    monkeypatch.setattr(singleton, "acquire_watcher_lock", lambda: object())
    monkeypatch.setattr(singleton, "release_watcher_lock", lambda h: None)
    monkeypatch.setattr(singleton, "stop_flag_left", lambda p: None)
    monkeypatch.setattr(health, "watcher_running", lambda: None)
    monkeypatch.setattr(recording_guard, "pause_reason", lambda *a, **k: None)

    # The watcher's own pass, stood in: it records that it ran.
    ran = []
    from mousereach.watcher import orchestrator, db as wdb

    class FakeOrchestrator:
        def __init__(self, config, db):
            self.config = config

        def run_once(self):
            ran.append(self.config.mode)

    monkeypatch.setattr(orchestrator, "ProcessingOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(orchestrator, "DLCOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(wdb, "WatcherDB", lambda path: SimpleNamespace(path=path))

    # Any older in-tab processor that might still be there is stood in too, so
    # the test never runs real algorithms; it records that it was used.
    legacy = []
    try:
        from mousereach.pipeline import core as legacy_core

        class LegacyProcessor:
            def __init__(self, *a, **k):
                legacy.append("constructed")

            def run(self, *a, **k):
                legacy.append("ran")
                return SimpleNamespace()

        monkeypatch.setattr(legacy_core, "UnifiedPipelineProcessor", LegacyProcessor)
    except ImportError:
        pass

    shown = []
    from mousereach.watcher import control_widget
    monkeypatch.setattr(control_widget, "show_info", shown.append)
    monkeypatch.setattr(control_widget, "show_error", shown.append)

    from mousereach.pipeline.batch_widget import UnifiedPipelineWidget
    widget = UnifiedPipelineWidget(None)
    app.processEvents()
    yield SimpleNamespace(widget=widget, ran=ran, legacy=legacy, shown=shown,
                          processing=processing, app=app, cfg=cfg)
    widget.close()


def _wait(pred, app, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_the_run_button_runs_the_watchers_own_pass(tab):
    """Pressing Run starts one pass of this machine's watcher, and nothing else."""
    w = tab.widget
    w.run_btn.setEnabled(True)
    w.run_btn.click()
    assert _wait(lambda: tab.ran or tab.legacy, tab.app), "nothing ran"
    thread = getattr(w, "_thread", None) or getattr(w, "worker_thread", None)
    if thread is not None:
        thread.join(10)
    assert tab.ran == ["processing_server"]
    assert tab.legacy == []


def test_the_tab_leaves_a_persons_cuts_alone(tab):
    """No control on the tab deletes outputs in Processing: a hand-cut
    segments file sitting there is untouched after a run. The old "Reprocess
    All" button, which deleted them, is gone (2026-11-19)."""
    seg = tab.processing / "20990101_PRJ0101_P1_segments.json"
    seg.write_text(json.dumps({"boundaries": [10, 20], "boundary_source": "human"}),
                   encoding="utf-8")
    before = seg.read_bytes()
    assert not hasattr(tab.widget, "reprocess_btn")
    tab.widget.run_btn.setEnabled(True)
    tab.widget.run_btn.click()
    _wait(lambda: tab.ran or tab.legacy, tab.app)
    assert seg.read_bytes() == before


def test_the_tab_says_it_runs_the_same_steps_as_the_watcher():
    from mousereach.pipeline import batch_widget
    intro = batch_widget.RUN_PIPELINE_INTRO
    for word in ("same steps", "hand-score", "review queue", "filed"):
        assert word in intro
    intro.encode("ascii")


def test_on_a_pose_only_pc_the_tab_says_it_poses_and_does_not_analyse(tab):
    """The tab runs THIS machine's role. On a pose-only graphics-card PC that is
    the pose pass, so the tab must say so: a person there must not expect the
    button to analyse or file videos."""
    tab.cfg.mode = "dlc_pc"
    tab.cfg.also_process = False
    tab.widget._refresh()
    said = tab.widget._mode_label.text()
    assert "pose" in said and "does not analyse or file" in said, said
    assert "processing machine" in said

    tab.cfg.mode = "processing_server"
    tab.widget._refresh()
    said = tab.widget._mode_label.text()
    assert "files a video that passed every check" in said, said

    from mousereach.pipeline import batch_widget
    assert "pose-only" in batch_widget.RUN_PIPELINE_INTRO


def test_chosen_videos_run_through_the_watchers_own_steps(tab, monkeypatch):
    """The owner's answer (2026-11-19, MRR-15a): chosen videos are run by hand
    again, through the same code as the watcher (run_one.run_chosen_videos).
    The greyed-out "Select Specific Files...", "Clear Selection" and
    "Reprocess All..." are gone; the tab says what is not run and why."""
    from qtpy.QtWidgets import QLabel
    from mousereach.pipeline import batch_widget
    from mousereach.watcher import run_one
    got = []

    def _run(files, progress=None):
        got.append(list(files))
        progress("pipeline", "running")
        return [{"video": "A", "done": True}, {"video": "B", "refused": "filed"},
                {"video": "C", "held": "triage", "hold_reason": "a pellet to decide"}]
    monkeypatch.setattr(run_one, "run_chosen_videos", _run)
    w = tab.widget
    for name in ("select_btn", "clear_btn", "reprocess_btn"):
        assert not hasattr(w, name)
    assert w.chosen_btn.text() == batch_widget.RUN_CHOSEN_BUTTON_TEXT
    w.run_chosen(["x/A.mp4", "x/B.mp4", "x/C.mp4"])
    assert _wait(lambda: "Done." in w._chosen_log.toPlainText(), tab.app)
    log = w._chosen_log.toPlainText()
    assert got == [["x/A.mp4", "x/B.mp4", "x/C.mp4"]]
    assert "[OK] A: filed" in log and "[NOT RUN] B: filed" in log
    assert "[HELD] C: a pellet to decide" in log
    labels = " ".join(l.text() for l in w.findChildren(QLabel))
    assert "Run chosen videos" in labels and "crystallized" in labels
    for text in (batch_widget.SELECTION_NOTE, batch_widget.PRIORITY_INTRO):
        text.encode("ascii")


def test_putting_an_animal_first_from_the_tab_is_the_watchers_own_priority(tab, monkeypatch):
    """Choosing which videos go first from the window writes the same priority
    the command writes and the watcher reads; "Back to normal order" removes
    it; an id that is not an animal writes nothing."""
    from mousereach.watcher.orchestrator import BaseOrchestrator
    w = tab.widget
    pfile = tab.processing.parent / "priority_animal.json"

    w.priority_edit.setText("abc0101")
    w.priority_btn.click()
    assert json.loads(pfile.read_text(encoding="utf-8"))["animal_id"] == "ABC0101"
    # The watcher's own reading of it (BaseOrchestrator, which both
    # orchestrators inherit; the fixture stands in only the two subclasses).
    import mousereach.watcher.orchestrator as orch
    monkeypatch.setattr(orch, "require_processing_root", lambda: tab.processing.parent)
    assert BaseOrchestrator._get_priority_animal(None) == "ABC0101"
    assert "ABC0101" in w.priority_label.text()

    w.priority_clear_btn.click()
    assert not pfile.exists()
    assert "normal order" in w.priority_label.text()

    w.priority_edit.setText("not an animal")
    w.priority_btn.click()
    assert not pfile.exists()
    assert w.priority_label.text().startswith("Not set")


def test_the_command_and_the_tab_write_the_same_priority(tab, monkeypatch, capsys):
    import sys
    from mousereach.watcher import cli
    from mousereach.watcher.priority_animal import current_priority
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-prioritize", "ABC0102"])
    cli.main_prioritize()
    assert current_priority()["animal_id"] == "ABC0102"
    tab.widget._show_priority()
    assert "ABC0102" in tab.widget.priority_label.text()
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-prioritize", "--clear"])
    cli.main_prioritize()
    assert current_priority() is None
