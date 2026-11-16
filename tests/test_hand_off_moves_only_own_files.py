"""The hand-off to the posed folder moves only files this GPU node holds.

The stage step finds a 'dlc_complete' video by name and MOVES every file of it
it finds into Processing/Posed. Its search also looks at the top of the
shared singles folder, and a recorded path inside the filed folder was
returned even with the archive search switched off -- so a row with no local
copy moved a shared, unposed video (or a filed video's results) into Posed.
Now: the stage step takes only a copy in this node's DLC queue or its own
claim folder; a recorded path inside the filed folder is ignored when the
archive is not searched; startup recovery does not count a file at the top of
the shared folder as this node's; and --sweep on a GPU node never records a filed,
posed or shared path against a working state (a processing server's sweep
still records a posed copy, and the filed copy of a re-run). tmp_path only.
"""
import sys
from pathlib import Path

import pytest

from mousereach.watcher import cli, coordination, single_claim
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.locate import locate_video_file

from test_singles_intake_claim import HOST, _as, _node, env  # noqa: F401

VID = "20240101_ABC9901_P1"


def _filed_copy(env):
    d = env.analyzed_output / "PROJ" / "PROJ_01"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{VID}.mp4"
    p.write_bytes(b"filed")
    return p


def test_a_recorded_path_in_the_filed_folder_is_ignored_without_the_archive(env):
    filed = _filed_copy(env)
    assert locate_video_file(VID, raw=str(filed), search_archive=False,
                             search_staging=False) is None
    assert locate_video_file(VID, raw=str(filed)) == filed     # a reader still sees it


def _stage(env, a):
    _as(env, a)
    return a._stage_to_nas({"id": VID, "data": a.db.get_video(VID)})


def test_an_unposed_shared_copy_is_not_moved_into_posed(env):
    a = _node(env, HOST)
    shared = env.door / f"{VID}.mp4"
    shared.write_bytes(b"video")
    a.db.register_video(VID, str(shared), current_path=str(shared))
    a.db.force_state(VID, "dlc_complete", reason="test: adopted by recovery")

    assert _stage(env, a) is False
    assert shared.read_bytes() == b"video"                       # left where it was
    assert not any(env.dlc_staging.iterdir())                    # nothing moved
    row = a.db.get_video(VID)
    assert row["state"] == "unresolvable"
    assert row["error_message"].startswith(single_claim.SHARED_REASON)


def test_a_filed_video_is_never_moved_into_posed(env):
    a = _node(env, HOST)
    filed = _filed_copy(env)
    (filed.parent / f"{VID}DLC_resnet50_x.h5").write_bytes(b"pose")
    a.db.register_video(VID, str(filed), current_path=str(filed))
    a.db.force_state(VID, "dlc_complete", reason="test: a swept row")

    assert _stage(env, a) is False
    assert filed.read_bytes() == b"filed"
    assert (filed.parent / f"{VID}DLC_resnet50_x.h5").is_file()
    assert not any(env.dlc_staging.iterdir())
    assert a.db.get_video(VID)["state"] == "unresolvable"


def test_recovery_does_not_take_a_shared_copy_as_this_nodes(env, monkeypatch):
    shared = env.door / f"{VID}.mp4"
    shared.write_bytes(b"video")
    db = WatcherDB(db_path=env.tmp / "recover.db")
    c = coordination.PipelineCoordinator.__new__(coordination.PipelineCoordinator)
    monkeypatch.setattr(c, "get_all_video_states", lambda: {
        VID: {"state": "dlc_complete", "hostname": "NODE-OTHER",
              "source_path": str(shared), "collage_id": None}}, raising=False)
    monkeypatch.setattr(c, "get_all_collage_states", lambda: {}, raising=False)
    c.recover_local_db(db, HOST)
    assert db.get_video(VID)["state"] == "unresolvable"


def _role(monkeypatch, mode):
    from types import SimpleNamespace
    from mousereach import config
    monkeypatch.setattr(config.WatcherConfig, "load",
                        classmethod(lambda cls: SimpleNamespace(mode=mode)))


def _sweep(env, monkeypatch, state, mode):
    db = WatcherDB(db_path=env.tmp / "sweep.db")
    db.register_video(VID, db.NO_FILE_HERE)
    db.force_state(VID, state, reason="test: a phantom row")
    _role(monkeypatch, mode)
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: db.db_path)
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-unresolvable", "--sweep"])
    cli.main_unresolvable()
    return db.get_video(VID)


def _filed_where_the_archive_puts_it(env):
    from mousereach.archive.core import get_archive_destination
    d = Path(get_archive_destination(VID))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{VID}.mp4"
    p.write_bytes(b"filed")
    return p


def test_on_a_server_the_sweep_records_a_posed_copy(env, monkeypatch):
    """The posed folder is where a processing server works a posed video
    from: a pathless 'dlc_complete' row whose files wait there gets its path,
    not 'unresolvable'."""
    posed = env.dlc_staging / f"{VID}.mp4"
    posed.write_bytes(b"posed for the server")
    row = _sweep(env, monkeypatch, "dlc_complete", "processing_server")
    assert row["state"] == "dlc_complete"
    assert row["current_path"] == str(posed)


def test_on_a_server_the_sweep_records_the_filed_copy_of_a_rerun(env, monkeypatch):
    filed = _filed_where_the_archive_puts_it(env)
    row = _sweep(env, monkeypatch, "outdated", "processing_server")
    assert row["state"] == "outdated"
    assert row["current_path"] == str(filed)


def test_on_a_server_the_sweep_never_points_another_state_at_a_filed_copy(env, monkeypatch):
    _filed_where_the_archive_puts_it(env)
    row = _sweep(env, monkeypatch, "processing", "processing_server")
    assert row["state"] == "unresolvable"


def test_on_a_gpu_node_the_sweep_ignores_a_filed_copy_of_a_rerun(env, monkeypatch):
    _filed_where_the_archive_puts_it(env)
    row = _sweep(env, monkeypatch, "outdated", "dlc_pc")
    assert row["state"] == "unresolvable"


def test_sweep_never_records_a_posed_path_against_a_working_state(env, monkeypatch, capsys):
    _role(monkeypatch, "dlc_pc")
    db = WatcherDB(db_path=env.tmp / "sweep.db")
    db.register_video(VID, db.NO_FILE_HERE)
    db.force_state(VID, "dlc_complete", reason="test: a phantom row")
    (env.dlc_staging / f"{VID}.mp4").write_bytes(b"posed for the server")
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: db.db_path)
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-unresolvable", "--sweep"])
    cli.main_unresolvable()
    row = db.get_video(VID)
    assert row["state"] == "unresolvable"
    assert not str(row.get("current_path") or "").startswith(str(env.dlc_staging))
