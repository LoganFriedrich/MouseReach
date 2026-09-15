"""mousereach-route-to-queue writes state to the daemon's OWN watcher database,
or routes nothing.

WHY: route_cli.main opened a bare WatcherDB(), which defaults to
<processing_root>/watcher.db and ignores the node's watcher.db_path override.
On a node with that override every state write went to an unused decoy
database while the bundles moved on disk -- the live database never learned of
any move ("Disk and DB now disagree", once per video). And when the database
could not be opened it routed "on disk only", which is that same disagreement
by design.

The rule under test (db_location.resolve_watcher_db_path, shared with
watcher/cli.py): the configured db_path when set, else processing_root /
watcher.db. route_cli refuses (exit 1) without flagging or moving anything, and
without creating a database file, when that file is missing, cannot be
resolved, or cannot be opened.

Everything lives under tmp_path; the config is patched (WatcherConfig.load,
require_processing_root), never read from or written to the real one.
"""
import json
from pathlib import Path

import pytest

import mousereach.config as config
import mousereach.pipeline.manifest as manifest
import mousereach.watcher.db as dbmod
from mousereach.config import Paths
from mousereach.watcher import cli, route_cli
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.db_location import resolve_watcher_db_path

VID = "20250101_CNT0101_P1"
POSE = f"{VID}DLC_resnet50_TestOct1shuffle1_1000.h5"


def _set_config(monkeypatch, processing_root, db_path=None):
    """Patch the two config reads the resolver makes."""
    watcher = {"db_path": str(db_path)} if db_path else {}
    monkeypatch.setattr(config.WatcherConfig, "load",
                        staticmethod(lambda: config.WatcherConfig(watcher)))
    monkeypatch.setattr(config, "require_processing_root", lambda: processing_root)


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A minimal node: processing root, Analyzed tree holding one video,
    and the two review queues."""
    proc = tmp_path / "proc"
    proc.mkdir()
    analyzed = tmp_path / "Analyzed"
    results = analyzed / "CNT" / "CNT_01"
    results.mkdir(parents=True)
    (results / f"{VID}.mp4").write_bytes(b"video")
    (results / POSE).write_bytes(b"pose")
    outcomes = results / f"{VID}_pellet_outcomes.json"
    outcomes.write_text(json.dumps({"segments": [
        {"segment_num": 3, "flagged_for_review": False},
        {"segment_num": 7, "flagged_for_review": False}]}), encoding="utf-8")
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    # Never let a pose lookup wander to the real staging area or version file.
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(manifest, "declared_dlc_scorer", lambda max_age_s=None: None)
    return {"tmp": tmp_path, "proc": proc, "results": results,
            "outcomes": outcomes, "triage": triage}


def _live_db(path: Path, results: Path) -> WatcherDB:
    """A real watcher database holding VID as archived, like the live one."""
    db = WatcherDB(path)
    db.register_video(VID, source_path=str(results / f"{VID}.mp4"))
    db.force_state(VID, "archived", reason="test setup")
    return db


def _assert_nothing_routed(site, before):
    assert site["outcomes"].read_bytes() == before        # no flags written
    assert (site["results"] / f"{VID}.mp4").is_file()     # nothing moved
    assert (site["results"] / POSE).is_file()
    assert list(site["triage"].iterdir()) == []            # queue untouched


# ------------------------------------------------------------------ resolver

def test_resolver_uses_the_configured_db_path(site, monkeypatch):
    configured = site["tmp"] / "local" / "watcher_local.db"
    _set_config(monkeypatch, site["proc"], db_path=configured)

    assert resolve_watcher_db_path() == configured


def test_resolver_falls_back_to_processing_root(site, monkeypatch):
    _set_config(monkeypatch, site["proc"])

    assert resolve_watcher_db_path() == site["proc"] / "watcher.db"


def test_resolver_treats_an_unreadable_config_as_no_override(site, monkeypatch):
    def boom():
        raise ValueError("bad config")
    monkeypatch.setattr(config.WatcherConfig, "load", staticmethod(boom))
    monkeypatch.setattr(config, "require_processing_root", lambda: site["proc"])

    assert resolve_watcher_db_path() == site["proc"] / "watcher.db"


def test_cli_resolver_delegates_and_still_prints(site, monkeypatch, capsys):
    configured = site["tmp"] / "local" / "watcher_local.db"
    _set_config(monkeypatch, site["proc"], db_path=configured)

    assert cli._resolve_db_path() == configured
    assert f"[watcher db] {configured}" in capsys.readouterr().out


# ------------------------------------------------------------ route refusal

@pytest.mark.parametrize("mode", ["single", "worklist"])
def test_missing_db_refuses_and_creates_nothing(site, monkeypatch, capsys, mode):
    configured = site["tmp"] / "local" / "watcher_local.db"
    _set_config(monkeypatch, site["proc"], db_path=configured)
    before = site["outcomes"].read_bytes()
    constructed = []
    monkeypatch.setattr(dbmod, "WatcherDB",
                        lambda *a, **k: constructed.append(a) or WatcherDB(*a, **k))

    rc = route_cli.main(_argv(site, mode))

    assert rc == 1
    out = capsys.readouterr().out
    assert f"[watcher db] {configured}" in out
    assert "not found" in out and str(configured) in out
    out.encode("ascii")                                    # ASCII-only output
    assert constructed == []                               # old code: WatcherDB()
    assert not configured.exists()                         # no decoy created
    assert not (site["proc"] / "watcher.db").exists()      # nor the default one
    _assert_nothing_routed(site, before)


def test_db_that_cannot_be_opened_refuses(site, monkeypatch, capsys):
    configured = site["tmp"] / "watcher_local.db"
    _live_db(configured, site["results"])
    _set_config(monkeypatch, site["proc"], db_path=configured)
    before = site["outcomes"].read_bytes()

    def broken(*a, **k):
        raise OSError("database is locked")
    monkeypatch.setattr(dbmod, "WatcherDB", broken)

    rc = route_cli.main(_argv(site, "single"))

    # Old code: logged a warning and routed on disk only.
    assert rc == 1
    assert "cannot open" in capsys.readouterr().out
    _assert_nothing_routed(site, before)


def test_unresolvable_db_refuses(site, monkeypatch, capsys):
    monkeypatch.setattr(config.WatcherConfig, "load",
                        staticmethod(lambda: config.WatcherConfig({})))

    def unconfigured():
        raise config.ConfigurationError("MouseReach is not configured.")
    monkeypatch.setattr(config, "require_processing_root", unconfigured)
    before = site["outcomes"].read_bytes()

    rc = route_cli.main(_argv(site, "single"))

    assert rc == 1
    assert "cannot resolve" in capsys.readouterr().out
    _assert_nothing_routed(site, before)


# ------------------------------------------------------------ route success

@pytest.mark.parametrize("mode", ["single", "worklist"])
def test_existing_configured_db_routes_and_records_state(site, monkeypatch, capsys, mode):
    configured = site["tmp"] / "local" / "watcher_local.db"
    configured.parent.mkdir()
    live = _live_db(configured, site["results"])
    # The decoy the old bare WatcherDB() wrote to: same video, never updated.
    decoy = _live_db(site["proc"] / "watcher.db", site["results"])
    _set_config(monkeypatch, site["proc"], db_path=configured)

    rc = route_cli.main(_argv(site, mode))

    assert rc == 0
    out = capsys.readouterr().out
    assert f"[watcher db] {configured}" in out
    bundle = site["triage"] / VID
    assert (bundle / f"{VID}.mp4").is_file()
    assert (bundle / POSE).is_file()
    flagged = json.loads((bundle / f"{VID}_pellet_outcomes.json").read_text(encoding="utf-8"))
    assert [s["segment_num"] for s in flagged["segments"] if s["flagged_for_review"]] == [3, 7]
    assert not site["outcomes"].exists()
    assert live.get_video(VID)["state"] == "triage"      # the daemon's database
    assert decoy.get_video(VID)["state"] == "archived"   # the decoy left alone


def test_fallback_db_routes_and_json_stdout_stays_parseable(site, monkeypatch, capsys):
    live = _live_db(site["proc"] / "watcher.db", site["results"])
    _set_config(monkeypatch, site["proc"])

    rc = route_cli.main(_argv(site, "single") + ["--json"])

    assert rc == 0
    captured = capsys.readouterr()
    results = json.loads(captured.out)                    # nothing else on stdout
    assert results[0]["routed"] is True and results[0]["flagged"] == [3, 7]
    assert "[watcher db]" in captured.err
    assert live.get_video(VID)["state"] == "triage"


@pytest.mark.parametrize("state", ["dlc_running", "processing", "processed", "archiving"])
def test_video_in_flight_on_the_watcher_is_deferred_untouched(site, monkeypatch, capsys, state):
    """Now that this command writes the daemon's live database, routing a video
    the daemon is working on races it: a 'processing' row set to 'triage'
    underneath the pipeline makes its own 'processed' write illegal and the
    video is marked failed. Deferred: nothing flagged, moved or written, and
    not an exit-1 failure (the integrator offers it again on a later run)."""
    configured = site["tmp"] / "local" / "watcher_local.db"
    configured.parent.mkdir()
    live = _live_db(configured, site["results"])
    live.force_state(VID, state, reason="test setup")
    _set_config(monkeypatch, site["proc"], db_path=configured)
    before = site["outcomes"].read_bytes()

    rc = route_cli.main(_argv(site, "single"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "wait " in out and "in flight" in out
    out.encode("ascii")
    _assert_nothing_routed(site, before)
    assert live.get_video(VID)["state"] == state


def _argv(site, mode):
    base = ["--queue", "triage", "--reason", "bench disagreement"]
    if mode == "single":
        return [VID, "--flag-segments", "3,7"] + base
    worklist = site["tmp"] / "worklist.json"
    worklist.write_text(json.dumps([{"video_id": VID, "segment_nums": [3, 7]}]),
                        encoding="utf-8")
    return ["--worklist", str(worklist)] + base
