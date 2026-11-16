"""The hand-scores file and its manifest are read as a pair.

WHY: the tool that writes them replaces the scores file first and the
manifest second. The reader cached the pair keyed on the scores file alone,
so a read between the two renames kept the old manifest (its import times)
beside the new scores until the next import; and a video held for its scores
was checked again only when the scores file changed, not when the manifest
it was waiting for arrived.
"""
import hashlib
import json
import os

import pytest

import mousereach.watcher.hand_scores as hs
import mousereach.watcher.route_cli as rc
from mousereach.watcher.db import WatcherDB

VID = "20250709_ABC0207_P1"
CSV = "video_key,pellet,score\n%s,1,2\n" % VID


@pytest.fixture
def pair(tmp_path, monkeypatch):
    f = tmp_path / "hand.csv"
    f.write_bytes(CSV.encode("utf-8"))      # exactly these bytes: the hash is of them
    hs._CACHE.update(key=None)
    monkeypatch.setattr(hs, "PAIR_RETRY_WAIT_S", 0.0, raising=False)
    base = f.stat().st_mtime
    writes = []

    def manifest(**keys):
        # Each write gets a strictly later time (base + 5 s per write), so two
        # manifests of the same length never share a size-and-time key, however
        # coarse the file system's clock or however fast the test runs.
        writes.append(1)
        m = f.with_suffix(".json")
        m.write_text(json.dumps(keys), encoding="utf-8")
        t = base + 5 * len(writes)
        os.utime(m, (t, t))
    return f, manifest


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_a_new_manifest_beside_the_same_scores_file_is_read(pair):
    f, manifest = pair
    manifest(last_successful_import={"ABC_02": "2026-01-01T20:00:00"})
    assert hs.load(f)[1]["last_successful_import"]["ABC_02"] == "2026-01-01T20:00:00"
    manifest(last_successful_import={"ABC_02": "2026-01-02T20:00:00"})
    assert hs.load(f)[1]["last_successful_import"]["ABC_02"] == "2026-01-02T20:00:00"


def test_a_manifest_written_for_another_scores_file_is_refused_and_not_kept(pair):
    f, manifest = pair
    manifest(scores_sha256=_sha("something else"), last_successful_import={})
    with pytest.raises(Exception) as err:
        hs.load(f)
    assert "does not match" in str(err.value)
    # nothing was cached: once the pair matches, it is read
    manifest(scores_sha256=_sha(CSV), last_successful_import={"ABC_02": "x"})
    scores, man = hs.load(f)
    assert scores[VID] == {1: 2} and man["last_successful_import"] == {"ABC_02": "x"}


def test_a_manifest_with_no_hash_is_accepted(pair):
    f, manifest = pair
    manifest(last_successful_import={})
    assert hs.load(f)[0][VID] == {1: 2}


def test_the_check_holds_the_video_while_the_pair_does_not_match(pair, tmp_path, monkeypatch):
    f, manifest = pair
    manifest(scores_sha256=_sha("older scores"), last_successful_import={})
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f), "required_projects": ["ABC"]})
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / f"{VID}_pellet_outcomes.json").write_text(
        json.dumps({"segments": [{"segment_num": 1, "outcome": "retrieved"}]}), encoding="utf-8")
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())
    import mousereach.watcher.orchestrator as o
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.db.force_state(VID, "processing", reason="test fixture")
    orch.hostname = "HOST"
    assert orch._hand_score_precheck(VID, folder) is False


def test_a_held_video_is_checked_again_when_only_the_manifest_changes(pair, tmp_path, monkeypatch):
    """The wait ends when the cohort's import arrives -- in the manifest."""
    f, manifest = pair
    f.write_text("video_key,pellet,score\n", encoding="utf-8")       # no scores yet
    manifest(last_successful_import={"ABC_02": "2000-01-01T00:00:00"})
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f), "required_projects": ["ABC"]})
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / f"{VID}_pellet_outcomes.json").write_text(
        json.dumps({"segments": [{"segment_num": 1, "outcome": "retrieved"}]}), encoding="utf-8")
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())
    import mousereach.watcher.orchestrator as o
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.db.force_state(VID, "processing", reason="test fixture")
    orch.hostname = "HOST"
    assert orch._hand_score_precheck(VID, folder) is False
    assert orch._waiting_for_hand_scores(VID)
    manifest(last_successful_import={"ABC_02": "2099-01-01T00:00:00"})
    assert not orch._waiting_for_hand_scores(VID)
