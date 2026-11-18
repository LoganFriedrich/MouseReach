"""Every stopped video is on a screen with its reason, and can be re-run there.

WHY: a video whose row is 'failed' or 'unresolvable' is in no review queue.
The Pipeline Dashboard finds videos by the folders they sit in, so a stopped
video whose file is in no scanned folder could not be seen or selected
anywhere; its reason was only in a terminal command on the machine holding
the row; a graphics-card PC's stopped videos were invisible from the
processing server; and the status command listed failed rows but never
unresolvable ones.

Built in tmp_path with a real watcher database and invented ids.
"""
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

import mousereach.config as config
from mousereach.watcher.db import WatcherDB

FAILED = "20990301_ABC0301_P1"
UNRES = "20990301_ABC0302_P2"
FILED = "20990301_ABC0303_P3"
WHY_FAILED = "segmentation crashed: test reason"
WHY_UNRES = "no file on this node: test reason"


@pytest.fixture
def db(tmp_path):
    d = WatcherDB(tmp_path / "watcher.db")
    for vid in (FAILED, UNRES, FILED):
        d.register_video(vid, source_path=str(pathlib.Path("x") / f"{vid}.mp4"))
    d.force_state(FAILED, "processing", reason="test")
    d.mark_failed(FAILED, WHY_FAILED)
    d.mark_unresolvable(UNRES, WHY_UNRES)
    d.force_state(FILED, "archived", reason="test")
    return d


_APP = []


def _qt():
    """A QApplication kept alive for the whole module (one that is garbage
    collected while widgets exist crashes the interpreter)."""
    from qtpy.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


# ------------------------------------------------------------ the Stopped tab

def test_the_dashboard_lists_every_stopped_video_with_its_reason(db, monkeypatch):
    _qt()
    from qtpy.QtWidgets import QLabel, QTableWidget
    from mousereach.dashboard.widget import PipelineDashboard
    import mousereach.watcher.health as health
    monkeypatch.setattr(health, "watcher_running", lambda: 1234)
    monkeypatch.setattr(health, "_current_pause_reason", lambda: None)
    dash = SimpleNamespace(health_label=QLabel(), stopped_table=QTableWidget(),
                           stopped_note=QLabel(), _watcher_db=lambda: db)
    dash._update_stopped = lambda d=None: PipelineDashboard._update_stopped(dash, d)
    PipelineDashboard._update_health(dash)
    t = dash.stopped_table
    rows = [[t.item(r, c).text() for c in range(t.columnCount())] for r in range(t.rowCount())]
    assert [r[0] for r in rows] == [FAILED, UNRES]          # failed first
    assert WHY_FAILED in rows[0][3] and WHY_UNRES in rows[1][3]
    assert FILED not in [r[0] for r in rows]


def test_the_list_is_read_from_a_copy_of_the_database(db, monkeypatch):
    """The dashboard never holds a read on the database the watcher writes:
    the rows come from a snapshot, deleted after reading."""
    import mousereach.watcher.stopped as stopped
    made = []
    real = stopped.snapshot_db

    def spy(path):
        p = real(path)
        made.append(p)
        return p
    monkeypatch.setattr(stopped, "snapshot_db", spy)
    rows = stopped.read_stopped(db=db)
    assert {r["video_id"] for r in rows} == {FAILED, UNRES}
    assert made and made[0] != pathlib.Path(db.db_path) and not made[0].exists()


def test_the_failed_filter_shows_a_failed_video_no_folder_holds(db):
    """The overview is built from the folders; a failed video whose file is
    in none of them must still be listed under Failed, by its row state."""
    _qt()
    from qtpy.QtWidgets import QComboBox, QTableWidget
    from mousereach.dashboard.widget import PipelineDashboard, STAGE_FILTERS
    combo = QComboBox()
    combo.addItems([label for label, _ in STAGE_FILTERS])
    combo.setCurrentText("Failed")
    table = QTableWidget()
    table.setColumnCount(12)
    dash = SimpleNamespace(all_files={}, stage_filter=combo, overview_table=table,
                           adapter=SimpleNamespace(),
                           # what the Stopped tab read from the database
                           _stopped_rows=[
                               {"video_id": FAILED, "state": "failed", "row_state": "failed",
                                "reason": WHY_FAILED, "since": "", "what_to_do": ""},
                               {"video_id": UNRES, "state": "unresolvable",
                                "row_state": "unresolvable", "reason": WHY_UNRES,
                                "since": "", "what_to_do": ""}])
    PipelineDashboard._update_overview_table(dash)
    names = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert names == [FAILED]


def test_a_failed_video_in_the_working_folder_is_shown_as_failed(db):
    import mousereach.watcher.stopped as stopped
    files = {FAILED: {"current_stage": "processing", "locations": [], "metadata": {}}}
    merged = stopped.merge_row_states(files, stopped.read_stopped(db=db))
    assert merged[FAILED]["current_stage"] == "failed"
    assert stopped.matches(merged[UNRES], {"unresolvable"})


# ------------------------------------------------------------ the Re-run action

def test_rerun_of_a_failed_video_goes_through_the_reset(db, tmp_path, monkeypatch):
    """The existing reset path: the restart step is read from the files this
    machine holds, and the automatic retries are given back."""
    proc = tmp_path / "Processing"
    proc.mkdir()
    (proc / f"{FAILED}.mp4").write_bytes(b"v")
    (proc / f"{FAILED}DLC_resnet50_TestModelshuffle1_100.h5").write_bytes(b"p")
    monkeypatch.setattr(config.Paths, "PROCESSING", proc, raising=False)
    monkeypatch.setattr(config.Paths, "DLC_QUEUE", tmp_path / "none", raising=False)
    import mousereach.watcher.stopped as stopped
    ok, msg = stopped.rerun(db, FAILED)
    row = db.get_video(FAILED)
    assert ok and row["state"] == "processing" and not row["error_count"]


def test_the_overview_rerun_button_resets_a_failed_video_instead_of_marking_it(
        db, tmp_path, monkeypatch):
    """The overview's Re-run button used to mark a failed video 'outdated'
    (a re-run from its filed folder) -- which a failed video does not have."""
    _qt()
    proc = tmp_path / "Processing"
    proc.mkdir()
    (proc / f"{FAILED}.mp4").write_bytes(b"v")
    (proc / f"{FAILED}DLC_resnet50_TestModelshuffle1_100.h5").write_bytes(b"p")
    monkeypatch.setattr(config.Paths, "PROCESSING", proc, raising=False)
    monkeypatch.setattr(config.Paths, "DLC_QUEUE", tmp_path / "none", raising=False)
    import mousereach.dashboard.widget as w
    monkeypatch.setattr(w.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("cause fixed", True)))
    monkeypatch.setattr(w.QMessageBox, "question",
                        staticmethod(lambda *a, **k: w.QMessageBox.Ok))
    monkeypatch.setattr(w.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    dash = SimpleNamespace(_selected_file=f"{FAILED}.mp4", _watcher_db=lambda: db,
                           _update_stopped=lambda *a: None)
    dash._rerun_stopped_video = lambda vid: w.PipelineDashboard._rerun_stopped_video(dash, vid)
    w.PipelineDashboard._rerun_selected(dash)
    assert db.get_video(FAILED)["state"] == "processing"


def test_rerun_reads_the_live_row_and_leaves_a_retried_video_alone(db):
    import mousereach.watcher.stopped as stopped
    db.force_state(FAILED, "processing", reason="the watcher retried it")
    ok, msg = stopped.rerun(db, FAILED)
    assert not ok and "processing" in msg
    assert db.get_video(FAILED)["state"] == "processing"


def test_rerun_of_a_failed_video_with_no_files_here_asks_for_the_step(db, tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr(config.Paths, "PROCESSING", tmp_path / "none1", raising=False)
    monkeypatch.setattr(config.Paths, "DLC_QUEUE", tmp_path / "none2", raising=False)
    import mousereach.watcher.stopped as stopped
    ok, msg = stopped.rerun(db, FAILED)
    assert not ok and msg.startswith("Choose the step")
    ok, msg = stopped.rerun(db, FAILED, from_step="dlc_complete")
    assert ok and db.get_video(FAILED)["state"] == "dlc_complete"


def test_rerun_of_an_unresolvable_video_on_a_server_changes_nothing(db):
    import mousereach.watcher.stopped as stopped
    ok, msg = stopped.rerun(db, UNRES, server=True)
    assert not ok and "processing server" in msg
    assert db.get_video(UNRES)["state"] == "unresolvable"


# ------------------------------------------------------------ other machines

def test_a_gpu_pcs_failed_videos_reach_the_servers_red_panel(db, tmp_path, monkeypatch):
    """Each node's status report carries its stopped counts; the server's
    dashboard shows a node's failed videos in red."""
    from mousereach.watcher import node_status
    from mousereach.watcher.orchestrator import BaseOrchestrator
    share = tmp_path / "share"
    share.mkdir()
    node = SimpleNamespace(db=db)
    report = BaseOrchestrator._work_report(node)
    assert node_status.write(share, node_status.WORKING, hostname="GPU-NODE-1", extra=report)
    text = node_status.describe(share)
    assert "1 failed, 1 unresolvable" in text

    _qt()
    from qtpy.QtWidgets import QLabel
    from mousereach.dashboard.widget import PipelineDashboard
    monkeypatch.setattr(config.Paths, "NAS_ROOT", share, raising=False)
    dash = SimpleNamespace(attention_label=QLabel(), waiting_label=QLabel(),
                           ATTENTION_SHOWN=PipelineDashboard.ATTENTION_SHOWN)
    PipelineDashboard._update_attention(dash)
    assert "GPU-NODE-1: 1 video(s) FAILED" in dash.attention_label.text()


def test_a_report_without_counts_reads_not_reported_never_zero(tmp_path):
    from mousereach.watcher import node_status
    share = tmp_path / "share"
    share.mkdir()
    node_status.write(share, node_status.WORKING, hostname="OLD-NODE",
                      extra={"doing": None, "last_done": None, "done_last_hour": 0})
    assert "not reported" in node_status.describe(share)


# ------------------------------------------------------------ the status command

def test_status_lists_unresolvable_videos_with_their_reason(db, tmp_path, monkeypatch, capsys):
    import mousereach.watcher.cli as cli
    share = tmp_path / "share2"
    (share / "attention").mkdir(parents=True)
    monkeypatch.setattr(config.Paths, "NAS_ROOT", share, raising=False)
    cli._print_held_and_failed(db)
    out = capsys.readouterr().out
    out.encode("ascii")
    assert f"{UNRES}: {WHY_UNRES}" in out


def test_an_unresolvable_video_with_filed_results_is_still_rerun_from_them(
        db, tmp_path, monkeypatch):
    """No new harm: before, the overview's button re-ran an unresolvable video
    from its filed folder; that stays so when its filed results are on disk."""
    _qt()
    import mousereach.dashboard.widget as w
    import mousereach.watcher.stopped as stopped
    monkeypatch.setattr(stopped, "filed_results_present", lambda vid: vid == UNRES)
    monkeypatch.setattr(w.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("look again", True)))
    monkeypatch.setattr(w.QMessageBox, "question",
                        staticmethod(lambda *a, **k: w.QMessageBox.Ok))
    monkeypatch.setattr(w.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    dash = SimpleNamespace(_selected_file=f"{UNRES}.mp4", _watcher_db=lambda: db,
                           _update_stopped=lambda *a: None)
    dash._rerun_stopped_video = lambda vid: w.PipelineDashboard._rerun_stopped_video(dash, vid)
    dash._rerun_selected = lambda: w.PipelineDashboard._rerun_selected(dash)
    w.PipelineDashboard._rerun_stopped_video(dash, UNRES)       # from the Stopped tab
    row = db.get_video(UNRES)
    assert (row["state"], row["reprocess_scope"]) == ("outdated", "segmentation")


def test_a_snapshot_that_cannot_be_taken_leaves_no_copy_behind(tmp_path, monkeypatch):
    import tempfile
    import mousereach.watcher.stopped as stopped
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(Exception):
        stopped.snapshot_db(tmp_path / "no_such.db")
    assert not list(tmp_path.glob("watcher_copy_*"))
