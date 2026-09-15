"""Every reader of the singles folder still sees a single a GPU node has claimed.

WHY: a GPU node claims a single for pose by MOVING it from
Unanalyzed/Single_Animal/<stem>.mp4 into
Unanalyzed/Single_Animal/.inflight/<host>/<stem>.mp4 (watcher/single_claim.py),
so only one node poses it. Each reader below listed only the TOP of the singles
folder, so a claimed video dropped out of every report for as long as the node
held it -- it looked like it had vanished. The rules these tests hold:

  (a) the .inflight dot-folder is never read as a video, nor walked as a
      folder of videos with names of its own ("NODE-A", ".inflight");
  (b) a claimed single still counts as waiting work, in the same bucket as an
      unclaimed one: reconcile "waiting:Unanalyzed/Single_Animal" naming the
      machine, census crop_dlc, dashboard "cropped";
  (c) a video found ONLY in a host's .inflight is not a mismatch;
  (d) no node finds another node's claim by searching for the video's name.

Readers ask census.runner.claimed_singles, which asks the claim module and
lists the folder itself when that module is missing -- both are tested.

tmp_path only; Paths is patched, nothing touches a shared drive.
"""
import json
import sys

import pytest

import mousereach.review.causal_review_io as cr_io
from mousereach.census import runner
from mousereach.dashboard import folder_scan
from mousereach.video_prep.core import collage_provenance as cp
from mousereach.watcher import locate
from mousereach.watcher import reconcile as rc

CLAIMED = "20250101_CNT0101_P1"     # claimed by NODE-A, only in .inflight
WAITING = "20250101_CNT0102_P1"     # an ordinary unclaimed single
COLLAGE = ("20250101_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,"
           "CNT0107,CNT0108_P1.mkv")
HOST = "NODE-A"


def _paths():
    from mousereach.config import Paths
    return Paths


@pytest.fixture(params=["claim_module", "fallback"])
def claim_source(request, monkeypatch):
    """Run each reader test twice: once through the claim module's own
    inflight_ids, once with that module unimportable so the reader's own
    listing is used. Either way the reader must give the same answer."""
    if request.param == "fallback":
        monkeypatch.setitem(sys.modules, "mousereach.watcher.single_claim", None)
    else:
        mod = pytest.importorskip("mousereach.watcher.single_claim")
        if not hasattr(mod, "inflight_ids"):
            pytest.skip("single_claim.inflight_ids not written yet")
    return request.param


@pytest.fixture
def env(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    work = tmp_path / "work"
    dirs = {
        "NAS_ROOT": nas,
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "MULTI_ANIMAL_SOURCE": nas / "Unanalyzed" / "Multi-Animal",
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": nas / "Processing" / "Posed",
        "FAILED": nas / "Processing" / "Failed",
        "TRIAGE_REVIEW": nas / "Processing" / "Review" / "triage",
        "DEEP_REVIEW": nas / "Processing" / "Review" / "deep_review",
        "PROCESSING_ROOT": work,
        "PROCESSING": work / "Processing",
        "DLC_QUEUE": work / "DLC_Queue",
    }
    P = _paths()
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(P, name, d, raising=False)
    # No fallback to another configured drive for the CLI's folder lookup.
    monkeypatch.setattr(P, "NAS_DRIVE", None, raising=False)

    import mousereach.config as cfg

    class _NoWatcherConfig:
        @staticmethod
        def load():
            raise RuntimeError("no watcher config in tests")

    monkeypatch.setattr(cfg, "WatcherConfig", _NoWatcherConfig)
    quarantine = nas / "Processing" / "Quarantine"
    quarantine.mkdir(parents=True)
    monkeypatch.setattr(rc, "_quarantine_dir", lambda: quarantine)
    monkeypatch.setattr(cr_io, "resolve_review_path",
                        lambda stem, primary_dir=None: None)
    monkeypatch.setattr(cr_io, "durable_review_dir", lambda: None)
    (nas / "pipeline_versions.json").write_text(
        json.dumps({"versions": {"dlc_scorer": "NEW", "segmenter": "2.0"}}))
    singles = dirs["SINGLE_ANIMAL_OUTPUT"]
    return type("Env", (), {"tmp": tmp_path, "singles": singles,
                            "inflight": singles / ".inflight", **dirs})


def claim(env, stem=CLAIMED, host=HOST):
    """What a claim leaves on disk: the video moved under .inflight/<host>/."""
    d = env.inflight / host
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.mp4"
    p.write_bytes(b"video")
    return p


def unclaimed(env, stem=WAITING):
    p = env.singles / f"{stem}.mp4"
    p.write_bytes(b"video")
    return p


# --------------------------------------------------------------------------
# census.runner.claimed_singles -- the one lookup every reader uses
# --------------------------------------------------------------------------

def test_claimed_singles_names_the_host(env, claim_source):
    claim(env)
    unclaimed(env)
    assert runner.claimed_singles(env.singles) == {CLAIMED: HOST}
    assert runner.claimed_singles() == {CLAIMED: HOST}     # defaults to Paths


def test_claimed_singles_empty_without_claims(env, claim_source):
    unclaimed(env)
    assert runner.claimed_singles(env.singles) == {}


def test_fallback_listing_skips_strays_and_missing_folders(env, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mousereach.watcher.single_claim", None)
    claim(env)
    (env.inflight / "README.txt").write_text("not a host folder")
    (env.inflight / HOST / f"{WAITING}.json").write_text("{}")
    assert runner.claimed_singles(env.singles) == {CLAIMED: HOST}
    assert runner.claimed_singles(tmp_path / "missing") == {}
    guard = tmp_path / "guarded"
    guard.mkdir()
    (guard / ".inflight").write_text("a file, not a folder")
    assert runner.claimed_singles(guard) == {}


def test_fallback_names_both_hosts_if_two_hold_one_name(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "mousereach.watcher.single_claim", None)
    claim(env, host="NODE-A")
    claim(env, host="NODE-B")
    assert runner.claimed_singles(env.singles) == {CLAIMED: "NODE-A, NODE-B"}


def test_a_broken_claim_module_falls_back_to_listing(env, monkeypatch):
    """A claim module that raises must not make claimed singles disappear."""
    import types
    broken = types.ModuleType("mousereach.watcher.single_claim")
    broken.INFLIGHT_DIR = ".inflight"

    def boom():
        raise RuntimeError("half-written module")

    broken.inflight_ids = boom
    monkeypatch.setitem(sys.modules, "mousereach.watcher.single_claim", broken)
    claim(env)
    assert runner.claimed_singles(env.singles) == {CLAIMED: HOST}


def test_claim_module_is_the_authority_for_the_configured_folder(env, monkeypatch):
    """For the configured singles folder the claim module's answer is used as
    given (it owns the claim layout); any other folder is listed directly."""
    import types
    fake = types.ModuleType("mousereach.watcher.single_claim")
    fake.INFLIGHT_DIR = ".inflight"
    fake.inflight_ids = lambda: {WAITING: "NODE-B"}
    monkeypatch.setitem(sys.modules, "mousereach.watcher.single_claim", fake)
    claim(env)                                   # on disk, but the module says otherwise
    assert runner.claimed_singles(env.singles) == {WAITING: "NODE-B"}
    other = env.tmp / "elsewhere"
    (other / ".inflight" / HOST).mkdir(parents=True)
    (other / ".inflight" / HOST / f"{CLAIMED}.mp4").write_bytes(b"video")
    assert runner.claimed_singles(other) == {CLAIMED: HOST}


def test_ids_in_dir_never_reads_the_claim_folder_as_a_video(env):
    claim(env)
    unclaimed(env)
    assert runner.ids_in_dir(env.singles) == {WAITING}


# --------------------------------------------------------------------------
# watcher.reconcile
# --------------------------------------------------------------------------

def test_reconcile_claimed_single_is_waiting_and_names_the_host(env, claim_source):
    claim(env)
    unclaimed(env)
    result = rc.reconcile()
    rows = {r["video_id"]: r for r in result["rows"]}
    assert set(rows) == {CLAIMED, WAITING}          # no ".inflight", no "NODE-A"
    assert rows[CLAIMED]["verdict"] == rc.WAITING
    assert rows[CLAIMED]["found_in"] == ["waiting:Unanalyzed/Single_Animal"]
    assert rows[CLAIMED]["detail"] == f"claimed by {HOST} for pose"
    assert rows[WAITING]["verdict"] == rc.WAITING
    assert rows[WAITING]["detail"] == ""
    assert result["mismatches"] == []


def test_reconcile_notes_a_second_unclaimed_copy(env, claim_source):
    claim(env)
    unclaimed(env, stem=CLAIMED)
    row = {r["video_id"]: r for r in rc.reconcile()["rows"]}[CLAIMED]
    assert row["verdict"] == rc.WAITING
    assert row["found_in"] == ["waiting:Unanalyzed/Single_Animal"]
    assert f"claimed by {HOST} for pose" in row["detail"]
    assert "another copy" in row["detail"]


def test_reconcile_says_when_a_claim_has_not_been_refreshed(env):
    """A claim held by a stopped machine must not read exactly like a live
    one: its holder refreshes it every poll, so its age tells them apart."""
    import os
    import time
    pytest.importorskip("mousereach.watcher.single_claim")
    fresh = claim(env)
    old = claim(env, stem=WAITING)
    then = time.time() - 5 * 3600
    os.utime(old, (then, then))
    rows = {r["video_id"]: r for r in rc.reconcile()["rows"]}
    assert rows[CLAIMED]["detail"] == f"claimed by {HOST} for pose"
    assert rows[WAITING]["verdict"] == rc.WAITING
    assert "not refreshed for 5 h" in rows[WAITING]["detail"]
    assert fresh.exists()


# --------------------------------------------------------------------------
# census.runner.run_census
# --------------------------------------------------------------------------

def test_census_counts_claimed_single_in_crop_dlc(env, claim_source):
    claim(env)
    unclaimed(env)
    c = runner.run_census()
    assert c["sessions"][CLAIMED]["element"] == "crop_dlc"
    assert c["sessions"][WAITING]["element"] == "crop_dlc"
    assert c["by_element"].get("crop_dlc") == 2
    assert not any(s.startswith(".") or s == HOST for s in c["sessions"])


# --------------------------------------------------------------------------
# dashboard.folder_scan.scan_pipeline_folders
# --------------------------------------------------------------------------

def test_dashboard_shows_claimed_single_as_cropped(env, claim_source):
    claimed_path = claim(env)
    unclaimed(env)
    out = folder_scan.scan_pipeline_folders()
    assert set(out) == {CLAIMED, WAITING}
    assert out[CLAIMED]["current_stage"] == "cropped"
    assert out[CLAIMED]["metadata"]["claimed_by"] == HOST
    assert out[CLAIMED]["metadata"]["path"] == str(claimed_path)
    assert out[WAITING]["current_stage"] == "cropped"
    assert "claimed_by" not in out[WAITING]["metadata"]


def test_dashboard_prefers_a_later_stage_over_a_claim(env, claim_source):
    claim(env)
    (env.DLC_STAGING / f"{CLAIMED}.mp4").write_bytes(b"video")
    out = folder_scan.scan_pipeline_folders()
    assert out[CLAIMED]["current_stage"] == "dlc_complete"
    assert "claimed_by" not in out[CLAIMED]["metadata"]


# --------------------------------------------------------------------------
# video_prep.core.collage_provenance.build_downstream_index
# --------------------------------------------------------------------------

def test_downstream_index_counts_claimed_offspring_as_cropped(env):
    """A collage whose child is out on a claim must not look uncut."""
    claim(env)
    unclaimed(env)
    idx = cp.build_downstream_index()
    assert idx[CLAIMED] == "cropped"
    assert idx[WAITING] == "cropped"
    assert HOST not in idx and ".inflight" not in idx


# --------------------------------------------------------------------------
# watcher.locate
# --------------------------------------------------------------------------

def test_locate_never_finds_another_nodes_claim_by_name(env):
    claimed_path = claim(env)
    assert locate.locate_video_file(CLAIMED, search_archive=False) is None
    # A node reaches its OWN claim through the path recorded on its row.
    assert locate.locate_video_file(CLAIMED, raw=str(claimed_path),
                                    search_archive=False) == claimed_path


# --------------------------------------------------------------------------
# watcher.cli mousereach-watch-process-animal (dry run)
# --------------------------------------------------------------------------

def test_process_animal_lists_claim_and_does_not_recut_its_collage(
        env, claim_source, monkeypatch, capsys):
    from mousereach.watcher import cli
    claim(env)
    (env.MULTI_ANIMAL_SOURCE / COLLAGE).write_bytes(b"collage")
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-process-animal",
                                      "CNT0101", "--dry-run"])
    with pytest.raises(SystemExit) as ei:
        cli.main_process_animal()
    assert ei.value.code == 0
    printed = capsys.readouterr().out
    assert CLAIMED in printed
    assert f"claimed by {HOST}" in printed
    # The collage the claimed single was cut from is not queued to be cut again.
    assert "No videos found for CNT0101" in printed
