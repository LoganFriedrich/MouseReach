"""A filing refused over a person's cuts moves NOTHING, and is listed for a person.

WHY: filing first sweeps the previous generation of a filed video into the
versioned Archive, then moves the new run in. When the filed folder holds a
person's cuts (a ``<video>_segments.json`` marked ``boundary_source:
"human"``) and the new run's segments file is the algorithm's, filing must
refuse -- the person's answer is never replaced by the algorithm's. A
refusal decided AFTER the sweep would leave every other filed file (reaches,
outcomes, kinematics) already in the Archive, the new ones still in
Processing, and the filed folder holding only the person's cuts and the
manifest, refused again on every retry. So the question is asked before
anything moves, the filed folder stays exactly as it was, and the
video is listed in the attention panel (the error string alone reached only a
log).

Video ids are invented.
"""
import json

V = "20990101_CNT9901_P1"
HUMAN = {"boundaries": [100, 200, 300], "boundary_source": "human", "video_name": V}
KIND = "filing_held_persons_cuts"   # the attention list kind
ALGO = {"boundaries": [110, 210, 310], "segmenter_version": "9.9.9", "video_name": V}


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _setup(tmp_path, monkeypatch):
    import mousereach.archive.core as core
    import mousereach.index as index_pkg
    from mousereach.config import Paths
    analyzed = tmp_path / "nas" / "Analyzed"
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed, raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    dest = analyzed / "PRJ" / "C01"
    proc = tmp_path / "Processing"
    # A full filed generation: the person's cuts and the results made on them.
    _write(dest / f"{V}_segments.json", HUMAN)
    _write(dest / f"{V}_reaches.json", {"segments": [], "gen": "old"})
    _write(dest / f"{V}_pellet_outcomes.json", {"segments": [], "gen": "old"})
    _write(dest / f"{V}_reach_assignments.json", {"gen": "old"})
    _write(dest / f"{V}_features.json", {"gen": "old"})
    _write(dest / f"{V}_processing_manifest.json",
           {"pipeline_versions": {"segmenter": "1"}, "dlc_model": {"dlc_scorer": "S"}})
    (dest / f"{V}.mp4").write_bytes(b"video")
    # This run: the algorithm's cuts and everything made on them.
    _write(proc / f"{V}_segments.json", ALGO)
    _write(proc / f"{V}_reaches.json", {"segments": [], "gen": "new"})
    _write(proc / f"{V}_pellet_outcomes.json", {"segments": [], "gen": "new"})
    _write(proc / f"{V}_features.json", {"gen": "new"})
    _write(proc / f"{V}_processing_manifest.json", {"pipeline_versions": {"segmenter": "2"}})
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)

    class _NoIndex:
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass
    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)
    return core, dest, proc, analyzed / "Archive", tmp_path / "nas" / "attention"


def test_refused_filing_leaves_the_filed_folder_byte_for_byte(tmp_path, monkeypatch):
    core, dest, proc, archive, attention = _setup(tmp_path, monkeypatch)
    filed_before, proc_before = _snapshot(dest), _snapshot(proc)

    r = core.archive_video(V, dry_run=False, verbose=False, skip_ready_check=True,
                           source_dir=proc)

    assert not r["success"]
    assert "person's" in (r["error"] or "")
    assert _snapshot(dest) == filed_before, "the filed folder changed"
    assert _snapshot(proc) == proc_before, "this run's files changed"
    assert not archive.exists() or not any(p.is_file() for p in archive.rglob("*")), \
        "something was swept into the Archive"

    # A retry refuses the same way and still moves nothing.
    r2 = core.archive_video(V, dry_run=False, verbose=False, skip_ready_check=True,
                            source_dir=proc)
    assert not r2["success"] and _snapshot(dest) == filed_before


def test_refused_filing_is_listed_for_a_person_and_cleared_once_filed(tmp_path, monkeypatch):
    core, dest, proc, archive, attention = _setup(tmp_path, monkeypatch)
    r = core.archive_video(V, dry_run=False, verbose=False, skip_ready_check=True,
                           source_dir=proc)
    assert r.get("held_for_person", {}).get("kind") == KIND

    from mousereach.watcher.attention import read_items
    items = [i for i in read_items(tmp_path / "nas")
             if i.get("kind") == KIND]
    assert len(items) == 1 and items[0]["video_id"] == V and items[0]["needs_person"]
    assert "nothing moved" in items[0]["says"] or "before anything moved" in items[0]["says"]

    # The run's cuts are made the person's (e.g. the run picked them up): it
    # files, and the item goes away.
    _write(proc / f"{V}_segments.json", HUMAN)
    r = core.archive_video(V, dry_run=False, verbose=False, skip_ready_check=True,
                           source_dir=proc)
    assert r["success"], r
    assert [i for i in read_items(tmp_path / "nas")
            if i.get("kind") == KIND] == []
