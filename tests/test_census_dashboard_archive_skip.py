"""The census, dashboard and collage-provenance walks of Analyzed never read
the superseded-output archive as live data.

WHY: superseded outputs sit under ``Analyzed/Archive/`` with their ORIGINAL
names (``{stem}_reaches.json``, ``{stem}_processing_manifest.json``,
``{stem}DLC_...h5``). Each walk below used ``rglob`` and so counted an older
generation beside -- or instead of -- the live one, silently. Every test puts
a same-named decoy under Analyzed/Archive/ and calls the REAL function; each
assertion on the decoy fails against the old rglob code (noted per test).
Live data, collages inside Analyzed and ``DLC Model <N>/`` poses must still
be found.

tmp_path only; nothing touches a shared drive.
"""
import json
import os
import time

import pytest

import mousereach.review.causal_review_io as cr_io
from mousereach.census.runner import walk_analyzed
from mousereach.dashboard import folder_scan
from mousereach.dashboard.version_currency import build_manifest_index
from mousereach.video_prep.core import collage_provenance as cp

LIVE = "20250101_CNT0101_P1"      # live outputs in its cohort folder
GHOST = "20250101_CNT0102_P1"     # exists ONLY in the archive
REDO = "20250101_CNT0103_P1"      # back in Processing; old outputs archived
POSE = "20250101_CNT0104_P1"      # live pose only, under DLC Model 4
H5_TAIL = "DLC_resnet50_MouseReachJan1shuffle1_100000.h5"


def _paths():
    from mousereach.config import Paths
    return Paths


def _write(path, text="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _age(path, seconds_ago):
    t = time.time() - seconds_ago
    os.utime(path, (t, t))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tmp pipeline with every stage folder of the stage layout present and
    empty, so a test only sees what it puts there, and a plain guard FILE at
    each retired folder name, as the migrated share has. The quarantine lookup
    and review resolver are stubbed so nothing reads this machine's config or
    shared drive."""
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
        "PROCESSING": work / "Processing",
    }
    P = _paths()
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(P, name, d)
    # WHY guard files at the retired names: code under test that hardcoded a
    # retired join off NAS_ROOT (instead of reading Paths) then raises or finds
    # nothing, rather than passing on folders built for it here.
    for rel in (("Processing", "Single_Animal"), ("Processing", "DLC_Complete"),
                ("Processing", "Review", "flagged_for_review")):
        nas.joinpath(*rel).write_text("retired folder", encoding="ascii")

    import mousereach.config as cfg

    class _NoQuarantine:
        @staticmethod
        def load():
            raise RuntimeError("no watcher config in tests")

    monkeypatch.setattr(cfg, "WatcherConfig", _NoQuarantine)
    monkeypatch.setattr(cr_io, "resolve_review_path",
                        lambda stem, primary_dir=None: None)
    (nas / "pipeline_versions.json").write_text(
        json.dumps({"versions": {"dlc_scorer": "NEW", "segmenter": "2.0"}}))
    return type("Env", (), {"tmp": tmp_path, **dirs})


def _manifest(folder, stem, seg="2.0", scorer="NEW"):
    return _write(folder / f"{stem}_processing_manifest.json", json.dumps({
        "pipeline_versions": {"segmenter": seg},
        "dlc_model": {"dlc_scorer": scorer}}))


# --------------------------------------------------------------------------
# census.runner.walk_analyzed
# --------------------------------------------------------------------------

def test_census_walk_ignores_archive_but_keeps_live_and_collages(env):
    A = env.ANALYZED_OUTPUT
    cohort = A / "Connectome" / "CNT01"
    for suf in ("_reaches.json", "_pellet_outcomes.json"):
        _write(cohort / f"{LIVE}{suf}")
    # Collages living inside Analyzed: a cohort straggler and a retired one.
    straggler = _write(cohort / "Multi-Animal" / "20250101_CNT0101,CNT0102_P1.mkv", "v")
    retired = _write(A / "Multi-Animal" / "20250102_CNT0105,CNT0106_P1.mkv", "v")
    # Decoys: the same live stem with an EXTRA suffix, a stem only the archive
    # has, and an archived collage -- all under Analyzed/Archive/.
    arch = A / "Archive" / "superseded_processing_root_3.1"
    _write(arch / f"{LIVE}_features.json")
    _write(arch / f"{GHOST}_reaches.json")
    _write(A / "Archive" / "Multi-Animal" / "20250103_CNT0107,CNT0108_P1.mkv", "v")

    index, mtimes, collages = walk_analyzed(A)

    # Old rglob code: LIVE would also carry _features.json, GHOST would be
    # indexed, and the archived collage would be in the list.
    assert index == {LIVE: {"_reaches.json", "_pellet_outcomes.json"}}
    assert set(mtimes) == {LIVE}
    assert set(mtimes[LIVE]) == {"_reaches.json", "_pellet_outcomes.json"}
    assert sorted(collages) == sorted([straggler, retired])


def test_census_walk_keeps_matching_directory_names(env):
    """rglob("*") yielded directories too and the walk tests NAMES only; a
    non-archive folder named like an output still contributes its name."""
    A = env.ANALYZED_OUTPUT
    (A / "Connectome" / f"{LIVE}_segments.json").mkdir(parents=True)
    (A / "Archive" / f"{GHOST}_segments.json").mkdir(parents=True)
    index, _, _ = walk_analyzed(A)
    assert index == {LIVE: {"_segments.json"}}


def test_census_walk_raises_when_a_folder_cannot_be_listed(env, monkeypatch):
    """A NAS hiccup (WinError 59) listing one cohort folder must raise, as
    rglob("*") did. Old os.walk code (default onerror=None) skipped the folder
    silently and returned a census index missing that cohort, so this test
    failed against it."""
    A = env.ANALYZED_OUTPUT
    _write(A / "Connectome" / "CNT01" / f"{LIVE}_reaches.json")
    bad = A / "Connectome" / "CNT02"
    _write(bad / f"{GHOST}_reaches.json")
    real = os.scandir
    target = os.path.normcase(os.path.normpath(str(bad)))

    def fake(path="."):
        if os.path.normcase(os.path.normpath(os.fspath(path))) == target:
            raise OSError(22, "The specified network name is no longer available",
                          str(bad), 59)
        return real(path)

    monkeypatch.setattr(os, "scandir", fake)
    with pytest.raises(OSError) as ei:
        walk_analyzed(A)
    assert getattr(ei.value, "winerror", None) == 59


# --------------------------------------------------------------------------
# dashboard.folder_scan.scan_pipeline_folders (the Analyzed pass)
# --------------------------------------------------------------------------

def test_dashboard_scan_does_not_mark_archived_outputs_analyzed(env):
    A = env.ANALYZED_OUTPUT
    _write(A / "Connectome" / "CNT01" / f"{LIVE}_reaches.json")
    arch = A / "Archive" / "superseded_processing_root_3.1"
    _write(arch / f"{GHOST}_reaches.json")
    # REDO is back in Processing; its old generation was superseded into the
    # archive. analyzed (100) outranks processing (50), so the old code
    # showed it as done.
    _write(env.PROCESSING / f"{REDO}.mp4", "v")
    _write(arch / f"{REDO}_reaches.json")

    out = folder_scan.scan_pipeline_folders()

    assert out[LIVE]["current_stage"] == "analyzed"
    assert GHOST not in out                                  # old: "analyzed"
    assert out[REDO]["current_stage"] == "processing"        # old: "analyzed"
    assert "Archive" not in out[LIVE]["metadata"]["path"]


# --------------------------------------------------------------------------
# dashboard.version_currency.build_manifest_index
# --------------------------------------------------------------------------

def test_manifest_index_never_picks_an_archived_manifest(env):
    A = env.ANALYZED_OUTPUT
    live = _manifest(A / "Connectome" / "CNT01", LIVE)
    _age(live, 3600)
    # The archived copy is NEWER (the supersede move touched it), so
    # newest-mtime-wins picked it in the old code.
    decoy = _manifest(A / "Archive" / "superseded_processing_root_3.1", LIVE, seg="1.0")
    _age(decoy, 10)
    _manifest(A / "Archive" / "superseded_processing_root_3.1", GHOST)
    # A processing-root manifest is still indexed alongside Analyzed.
    work = _manifest(env.PROCESSING, REDO)

    idx = build_manifest_index([env.PROCESSING, A, None, env.tmp / "missing"])

    assert idx[LIVE] == live                                 # old: decoy
    assert GHOST not in idx                                  # old: indexed
    assert idx[REDO] == work
    assert set(idx) == {LIVE, REDO}


# --------------------------------------------------------------------------
# video_prep.core.collage_provenance.build_downstream_index
# --------------------------------------------------------------------------

def test_downstream_index_ignores_archive_keeps_dlc_model_poses(env):
    A = env.ANALYZED_OUTPUT
    _write(A / "Connectome" / "CNT01" / f"{LIVE}.mp4", "v")
    # Live pose storage: DLC Model <N>/ must stay visible.
    _write(A / "Connectome" / "DLC Model 4" / "CNT01" / f"{POSE}{H5_TAIL}", "h5")
    arch = A / "Archive"
    _write(arch / "DLC Model 4.0" / "CNT01" / f"{GHOST}{H5_TAIL}", "h5")
    _write(arch / "superseded_processing_root_3.1" / f"{REDO}.mp4", "v")
    _write(env.PROCESSING / f"{REDO}.mp4", "v")

    idx = cp.build_downstream_index()

    assert idx[LIVE] == "analyzed"
    assert idx[POSE] == "analyzed"
    assert GHOST not in idx                                  # old: "analyzed"
    assert idx[REDO] == "processing"                         # old: "analyzed"


# --------------------------------------------------------------------------
# video_prep.core.collage_provenance.build_complete_stems
# --------------------------------------------------------------------------

def test_complete_stems_ignores_archived_current_manifest(env):
    A = env.ANALYZED_OUTPUT
    _manifest(A / "Connectome" / "CNT01", LIVE)
    # An archived manifest that happens to compare current: only the archive
    # has GHOST, so the old code called a stem with no live output complete
    # and let its collage retire.
    _manifest(A / "Archive" / "superseded_processing_root_3.1", GHOST)
    # Live REDO is outdated; its archived twin is current. Either way REDO
    # must not count (old code: added it via the archived twin).
    _manifest(A / "Connectome" / "CNT01", REDO, seg="1.0")
    _manifest(A / "Archive" / "superseded_processing_root_3.1", REDO)

    complete = cp.build_complete_stems()

    assert complete == {LIVE}
    # Explicit roots take the same path.
    assert cp.build_complete_stems(analyzed_root=A, nas_root=env.NAS_ROOT) == {LIVE}
