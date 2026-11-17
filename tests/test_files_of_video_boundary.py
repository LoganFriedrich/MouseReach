"""Every mover takes a video's files by one rule that respects the name boundary.

WHY: several movers chose a video's files by a bare "name starts with the
stem" test. Tray numbers are any number of digits, so <date>_<animal>_P1 is a
prefix of <date>_<animal>_P10: filing, staging or cleaning up the first would
move or delete the second's files, leaving that video in no place.
"""
import mousereach.archive.core as core
from mousereach.watcher import orchestrator as o

P1 = "20990101_ZZZ0101_P1"
P10 = "20990101_ZZZ0101_P10"


def _two_videos(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for vid in (P1, P10):
        (folder / f"{vid}.mp4").write_bytes(b"v")
        (folder / f"{vid}DLC_resnet50_X.h5").write_bytes(b"p")
        (folder / f"{vid}_features.json").write_text("{}")


def test_the_rule():
    from mousereach.pipeline.fsutil import is_file_of_video
    assert is_file_of_video(f"{P1}.mp4", P1)
    assert is_file_of_video(f"{P1}_segments.json", P1)
    assert is_file_of_video(f"{P1}DLC_resnet50_X.h5", P1)
    assert not is_file_of_video(f"{P10}.mp4", P1)
    assert not is_file_of_video(f"{P1}.mp4.part", P1)


def test_filing_one_video_leaves_the_other_alone(tmp_path, monkeypatch):
    proc, dest = tmp_path / "Processing", tmp_path / "Analyzed" / "ZZZ01"
    _two_videos(proc)
    dest.mkdir(parents=True)
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)
    import mousereach.index as index_pkg

    class _NoIndex:
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass

    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)
    r = core.archive_video(P1, dry_run=False, verbose=False,
                           skip_ready_check=True, source_dir=proc)
    assert r["success"], r
    left = sorted(p.name for p in proc.iterdir())
    assert left == sorted([f"{P10}.mp4", f"{P10}DLC_resnet50_X.h5",
                           f"{P10}_features.json"]), left


def test_the_watcher_helpers_use_the_same_rule(tmp_path):
    _two_videos(tmp_path / "d")
    d = tmp_path / "d"
    got = {p.name for p in o.BaseOrchestrator._get_associated_files(None, d, P1)}
    assert got == {f"{P1}.mp4", f"{P1}DLC_resnet50_X.h5", f"{P1}_features.json"}
    staged = {p.name for p in o._reprocess_copy_set([d], d, P1)}
    assert not any(n.startswith(P10) for n in staged), staged


def test_routing_into_a_queue_uses_the_same_rule(tmp_path):
    from mousereach.watcher.review_routing import _bundle_files
    _two_videos(tmp_path / "d")
    d = tmp_path / "d"
    (d / f"{P1}.mp4.part").write_bytes(b"half")
    got = {p.name for p in _bundle_files(d, P1)}
    assert got == {f"{P1}.mp4", f"{P1}DLC_resnet50_X.h5", f"{P1}_features.json"}, got
