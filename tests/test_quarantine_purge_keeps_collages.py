"""mousereach-watch-quarantine --purge never deletes a multi-animal recording,
and deletes a single only while the collage it was cut from exists.

WHY: --purge used to delete EVERY quarantined file after one y/N question --
a misnamed multi-animal recording (the camera's only recording of that tray,
a piece of the experiment lost for ever) included -- and left each file's row
saying it was still quarantined. The rule now (watcher/quarantine_purge.py):
a collage is never deleted; a single is redundant only while its source
collage exists, which is decided by finding that collage, never by comparing
bytes (two cuts of one collage differ in their bytes); --purge alone is a dry
run; a crystallized video is never touched; a deletion retires the video's
rows (this machine's and the shared database's) and is written to the purge
log.

tmp_path only: the pipeline root, the watcher databases and the config are
stand-ins built here.
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.watcher import cli
from mousereach.watcher.db import WatcherDB

IDS = ",".join(f"ABC99{n:02d}" for n in range(1, 9))
COLLAGE = f"20240102_{IDS}_P1.mkv"
SINGLE = "20240102_ABC9901_P1.mp4"
VID = "20240102_ABC9901_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    import mousereach.config as mc
    pipe = tmp_path / "Pipeline"
    quarantine = pipe / "Processing" / "Quarantine"
    analyzed = pipe / "Analyzed"
    filed_collages = analyzed / "Grant" / "ABC" / "ABC_99" / "Multi-Animal"
    for d in (quarantine, filed_collages):
        d.mkdir(parents=True)
    db_path = tmp_path / "watcher.db"
    db = WatcherDB(db_path=db_path)
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: db_path)
    monkeypatch.setattr(mc.WatcherConfig, "load",
                        classmethod(lambda cls, *a, **k: SimpleNamespace(
                            get_quarantine_dir=lambda: quarantine)))
    monkeypatch.setattr(mc.Paths, "NAS_ROOT", pipe, raising=False)
    monkeypatch.setattr(mc.Paths, "ANALYZED_OUTPUT", analyzed, raising=False)
    answers = []
    monkeypatch.setattr("builtins.input", lambda *a: answers.pop(0) if answers else "y")
    return SimpleNamespace(pipe=pipe, quarantine=quarantine, analyzed=analyzed,
                           filed_collages=filed_collages, db=db, answers=answers)


def _quarantine(env, name, data=b"x" * 100, row=True):
    f = env.quarantine / name
    f.write_bytes(data)
    (env.quarantine / (name + ".quarantine.json")).write_text(
        json.dumps({"error_message": "wrong intake"}))
    if row and name.endswith(".mp4"):
        from mousereach.config import get_video_id
        vid = get_video_id(name)
        env.db.register_video(video_id=vid, source_path=str(f))
        env.db.mark_quarantined(vid, reason="wrong intake", is_collage=False)
    return f


def _collage_at(folder, name=COLLAGE, size=10_000):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"c" * size)
    return folder / name


def _run(monkeypatch, *args):
    monkeypatch.setattr(cli.sys, "argv", ["mousereach-watch-quarantine", *args])
    try:
        cli.main_quarantine()
    except SystemExit as e:
        return e.code or 0
    return 0


def _central(env):
    from mousereach.watcher.coordination import PipelineCoordinator
    return PipelineCoordinator(env.pipe / "watcher_central.db")


# --- collages are never deleted ----------------------------------------------

def test_a_collage_is_never_deleted(env, monkeypatch, capsys):
    good_name = _quarantine(env, COLLAGE, b"c" * 10_000)
    one_id = _quarantine(env, "20240102_ABC9901_P1.mkv", b"c" * 10_000)  # misnamed tray recording
    unreadable = _quarantine(env, "tray recording 2024-01-02.mkv", b"c" * 10_000)
    _collage_at(env.filed_collages)          # even with a collage of that name filed
    code = _run(monkeypatch, "--purge", "--delete")
    assert code == 0
    for f in (good_name, one_id, unreadable):
        assert f.exists(), f"{f.name} was deleted"
        assert (env.quarantine / (f.name + ".quarantine.json")).exists()
    assert "never deleted" in capsys.readouterr().out


def test_a_single_as_large_as_its_collage_is_kept_like_a_collage(env, monkeypatch):
    f = _quarantine(env, SINGLE, b"x" * 9_000)
    _collage_at(env.filed_collages, size=10_000)
    _run(monkeypatch, "--purge", "--delete")
    assert f.exists()
    assert env.db.get_video(VID)["state"] == "quarantined"


# --- a single goes only while its source collage exists -----------------------

def test_a_single_whose_collage_exists_is_deleted_its_rows_retired_and_logged(env, monkeypatch):
    f = _quarantine(env, SINGLE)
    collage = _collage_at(env.filed_collages)
    central = _central(env)
    central.sync_video_state(VID, "OTHER-NODE", "quarantined")
    import hashlib
    digest = hashlib.sha256(f.read_bytes()).hexdigest()

    code = _run(monkeypatch, "--purge", "--delete")

    assert code == 0
    assert not f.exists(), "a single whose collage exists was not deleted"
    assert not (env.quarantine / (SINGLE + ".quarantine.json")).exists()
    assert env.db.get_video(VID) is None, "its quarantined row was left behind"
    assert VID not in central.get_all_video_states(), "its shared row was left behind"
    log = env.db.get_recent_log(50)
    assert any(r["video_id"] == VID and r["step"] == "quarantine_purge" for r in log)
    lines = (env.pipe / "watcher_logs" / "quarantine_purge.jsonl").read_text().splitlines()
    entry = json.loads(lines[-1])
    assert entry["action"] == "deleted"
    assert entry["video_id"] == VID
    assert entry["sha256"] == digest
    assert entry["source_collage"] == str(collage)


def test_a_collage_waiting_in_the_intake_counts(env, monkeypatch):
    f = _quarantine(env, SINGLE)
    _collage_at(env.pipe / "Unanalyzed" / "Multi-Animal")
    _run(monkeypatch, "--purge", "--delete")
    assert not f.exists()


def test_a_single_without_a_traceable_collage_is_kept(env, monkeypatch, capsys):
    f = _quarantine(env, SINGLE)
    # A collage of another tray run, and one without this animal: neither is its source.
    _collage_at(env.filed_collages, name=COLLAGE.replace("_P1.mkv", "_P2.mkv"))
    _collage_at(env.filed_collages, name=COLLAGE.replace("ABC9901", "ABC9909"))
    code = _run(monkeypatch, "--purge", "--delete")
    assert code == 0
    assert f.exists()
    assert env.db.get_video(VID)["state"] == "quarantined"
    assert "only copy -- kept" in capsys.readouterr().out


def test_a_collage_only_in_quarantine_is_not_a_source(env, monkeypatch):
    f = _quarantine(env, SINGLE)
    _quarantine(env, COLLAGE, b"c" * 10_000)
    _run(monkeypatch, "--purge", "--delete")
    assert f.exists()


def test_bytes_never_decide_only_the_source_collage_does(env, monkeypatch):
    # A same-name single from a DIFFERENT cut (other bytes) elsewhere, and the
    # collage exists: deleted, because the collage is what makes it redundant.
    f = _quarantine(env, SINGLE, b"a" * 100)
    other_cut = env.pipe / "Unanalyzed" / "Unsupported_Tray_Type"
    other_cut.mkdir(parents=True)
    (other_cut / SINGLE).write_bytes(b"b" * 120)
    _collage_at(env.filed_collages)
    _run(monkeypatch, "--purge", "--delete")
    assert not f.exists()
    assert (other_cut / SINGLE).exists()


def test_an_identical_copy_without_the_collage_does_not_make_a_single_deletable(env, monkeypatch):
    f = _quarantine(env, SINGLE, b"a" * 100)
    copy_dir = env.analyzed / "Grant" / "ABC" / "ABC_99"
    (copy_dir / SINGLE).write_bytes(b"a" * 100)
    _run(monkeypatch, "--purge", "--delete")
    assert f.exists()


# --- dry run by default --------------------------------------------------------

@pytest.mark.parametrize("args", [("--purge",), ("--purge", "--dry-run"),
                                  ("--purge", "--delete", "--dry-run")])
def test_the_dry_run_changes_nothing(env, monkeypatch, capsys, args):
    single = _quarantine(env, SINGLE)
    collage = _quarantine(env, COLLAGE, b"c" * 10_000)
    _collage_at(env.filed_collages)
    before = sorted(p.name for p in env.quarantine.iterdir())
    code = _run(monkeypatch, *args)        # a "y" is waiting: it must not be asked
    assert code == 0
    assert sorted(p.name for p in env.quarantine.iterdir()) == before
    assert single.exists() and collage.exists()
    assert env.db.get_video(VID)["state"] == "quarantined"
    assert not (env.pipe / "watcher_logs").exists()
    out = capsys.readouterr().out
    assert "Would delete (1)" in out and SINGLE in out
    assert "Dry run: nothing was changed" in out


def test_answering_no_changes_nothing(env, monkeypatch):
    f = _quarantine(env, SINGLE)
    _collage_at(env.filed_collages)
    env.answers.append("n")
    _run(monkeypatch, "--purge", "--delete")
    assert f.exists()
    assert env.db.get_video(VID)["state"] == "quarantined"


# --- rows that are not the quarantined file's are never touched -------------------

def test_a_crystallized_video_is_refused(env, monkeypatch, capsys):
    f = _quarantine(env, SINGLE, row=False)
    env.db.register_video(video_id=VID, source_path="elsewhere")
    env.db.force_state(VID, "crystallized", reason="test lock")
    _collage_at(env.filed_collages)
    _run(monkeypatch, "--purge", "--delete")
    assert f.exists(), "a crystallized video's file was deleted"
    assert env.db.get_video(VID)["state"] == "crystallized"
    assert "crystallized" in capsys.readouterr().out


def test_a_live_row_of_the_same_name_is_left_alone(env, monkeypatch):
    f = _quarantine(env, SINGLE, row=False)
    env.db.register_video(video_id=VID, source_path="elsewhere")
    env.db.update_state(VID, "validated")
    _collage_at(env.filed_collages)
    _run(monkeypatch, "--purge", "--delete")
    assert not f.exists()
    assert env.db.get_video(VID)["state"] == "validated"


def test_no_pipeline_root_deletes_nothing(env, monkeypatch):
    import mousereach.config as mc
    monkeypatch.setattr(mc.Paths, "NAS_ROOT", None, raising=False)
    f = _quarantine(env, SINGLE)
    _collage_at(env.filed_collages)
    assert _run(monkeypatch, "--purge", "--delete") == 1
    assert f.exists()


# --- the purge's flags never reach another option -------------------------------

@pytest.mark.parametrize("flag", ["--dry-run", "--delete"])
def test_purge_flags_are_refused_without_purge(env, monkeypatch, flag):
    # "--release ... --dry-run" must not really move the file.
    f = _quarantine(env, SINGLE)
    code = _run(monkeypatch, "--release", SINGLE, "--to", "single", flag)
    assert code == 2
    assert f.exists()


# --- a collage row counts only when its file is under the pipeline root ---------

def test_a_collage_row_outside_the_pipeline_root_does_not_count(env, monkeypatch, tmp_path):
    f = _quarantine(env, SINGLE)
    scratch = _collage_at(tmp_path / "local_scratch")      # e.g. a machine's own disk
    env.db.register_collage(COLLAGE, str(scratch))
    _run(monkeypatch, "--purge", "--delete")
    assert f.exists()


def test_a_collage_row_under_the_pipeline_root_counts(env, monkeypatch):
    f = _quarantine(env, SINGLE)
    elsewhere = _collage_at(env.pipe / "Some_Other_Folder")
    env.db.register_collage(COLLAGE, str(elsewhere))
    _run(monkeypatch, "--purge", "--delete")
    assert not f.exists()
