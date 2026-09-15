"""A new review bundle must never be visible half-built.

WHY: move_video_bundle used to create the empty ``{stem}`` folder in the queue
first and then fill it file by file (cross-volume copies take seconds). The
running watcher's return scan saw the empty folder, retired it to _Problematic
mid-route, and every remaining move failed with "No such file or directory"
(2026-09-14). A new bundle is now assembled in the hidden
``<queue>/.incoming/{stem}/`` and published with one directory rename.
"""
import json

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_routing as routing

STEM = "20250101_CNT0101_P1"


def _make_source(tmp_path, stem=STEM):
    src = tmp_path / "Processing"
    src.mkdir(exist_ok=True)
    names = [
        f"{stem}.mp4",
        f"{stem}DLC_resnet50_modelShuffle1_100000.h5",
        f"{stem}_segments.json",
        f"{stem}_pellet_outcomes.json",
    ]
    for n in names:
        (src / n).write_text("x", encoding="utf-8")
    return src, names


def _routing_manifest(bundle, stem=STEM):
    return json.loads((bundle / f"{stem}_routing.json").read_text(encoding="utf-8"))


def test_new_bundle_published_whole_with_no_staging_left(tmp_path):
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"

    bundle, moved = routing.move_video_bundle(STEM, src, queue, "why")

    assert bundle == queue / STEM
    assert sorted(moved) == sorted(names)
    for n in names:
        assert (bundle / n).exists()
        assert not (src / n).exists()
    man = _routing_manifest(bundle)
    assert man["bundle_dir"] == str(queue / STEM)   # the FINAL folder
    assert man["failed_files"] == []
    assert not (queue / routing.INCOMING_DIR_NAME / STEM).exists()
    assert not (queue / routing.INCOMING_DIR_NAME).exists()  # empty parent removed


def test_final_folder_absent_while_files_move(tmp_path, monkeypatch):
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"
    real_move = routing._safe_move
    seen = []

    def _watching_move(s, d):
        # the return scan must have nothing to see until the bundle is whole
        assert not (queue / STEM).exists()
        assert d.parent == queue / routing.INCOMING_DIR_NAME / STEM
        seen.append(s.name)
        real_move(s, d)

    monkeypatch.setattr(routing, "_safe_move", _watching_move)
    bundle, _ = routing.move_video_bundle(STEM, src, queue, "why")

    assert sorted(seen) == sorted(names)
    assert bundle.is_dir()
    assert (bundle / f"{STEM}_routing.json").exists()


def test_existing_bundle_still_merges_in_place(tmp_path, monkeypatch):
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"
    bundle = queue / STEM
    bundle.mkdir(parents=True)
    (bundle / f"{STEM}_segments.json").write_text("old", encoding="utf-8")
    (bundle / f"{STEM}_manifest.json").write_text("{}", encoding="utf-8")
    real_move = routing._safe_move

    def _direct_move(s, d):
        assert d.parent == bundle   # already visible: no staging detour
        real_move(s, d)

    monkeypatch.setattr(routing, "_safe_move", _direct_move)
    out, moved = routing.move_video_bundle(STEM, src, queue, "why")

    assert out == bundle
    assert sorted(moved) == sorted(names)
    assert (bundle / f"{STEM}_segments.json").read_text(encoding="utf-8") == "x"
    assert (bundle / f"{STEM}_manifest.json").exists()   # untouched
    assert not (queue / routing.INCOMING_DIR_NAME).exists()


def test_interrupted_incoming_is_reused(tmp_path):
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"
    leftover = queue / routing.INCOMING_DIR_NAME / STEM
    leftover.mkdir(parents=True)
    # an earlier route moved this file in, then died before publishing
    (leftover / f"{STEM}_reaches.json").write_text("carried", encoding="utf-8")

    bundle, moved = routing.move_video_bundle(STEM, src, queue, "why")

    assert (bundle / f"{STEM}_reaches.json").read_text(encoding="utf-8") == "carried"
    for n in names:
        assert (bundle / n).exists()
    assert not leftover.exists()
    assert not (queue / routing.INCOMING_DIR_NAME).exists()


def test_failed_move_is_recorded_in_manifest(tmp_path, monkeypatch):
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"
    stuck = f"{STEM}.mp4"
    real_move = routing._safe_move

    def _flaky_move(s, d):
        if s.name == stuck:
            raise OSError("file is locked")
        real_move(s, d)

    monkeypatch.setattr(routing, "_safe_move", _flaky_move)
    bundle, moved = routing.move_video_bundle(STEM, src, queue, "why")

    assert stuck not in moved
    assert (src / stuck).exists()            # never deleted, left where it was
    man = _routing_manifest(bundle)
    assert man["failed_files"] == [{"name": stuck, "error": "file is locked"}]
    assert stuck not in man["moved_files"]


def test_bundle_appearing_mid_build_is_merged(tmp_path, monkeypatch):
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"
    real_move = routing._safe_move
    calls = []

    def _racing_move(s, d):
        real_move(s, d)
        calls.append(s.name)
        if len(calls) == len(names):
            # a concurrent route published the same stem meanwhile
            other = queue / STEM
            other.mkdir()
            (other / f"{STEM}_manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(routing, "_safe_move", _racing_move)
    bundle, moved = routing.move_video_bundle(STEM, src, queue, "why")

    assert bundle == queue / STEM
    for n in names:
        assert (bundle / n).exists()
    assert (bundle / f"{STEM}_manifest.json").exists()
    man = _routing_manifest(bundle)
    assert man["bundle_dir"] == str(bundle)
    assert man["failed_files"] == []
    assert not (queue / routing.INCOMING_DIR_NAME).exists()


def test_other_route_publishing_staging_mid_build_loses_nothing(tmp_path, monkeypatch):
    """Two routes of one stem share .incoming/<stem>. If the other route
    publishes it while this route is still filling it, this route's remaining
    files and its routing record go into the published bundle. They used to
    fail, and the manifest write raised out of the route."""
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"
    build = queue / routing.INCOMING_DIR_NAME / STEM
    real_move = routing._safe_move
    raced = []

    def _move_racing_publish(s, d):
        if not raced:
            raced.append(True)
            # the other route finishes: its file is in, and it renames staging
            (build / f"{STEM}_reaches.json").write_text("other", encoding="utf-8")
            build.rename(queue / STEM)
        real_move(s, d)

    monkeypatch.setattr(routing, "_safe_move", _move_racing_publish)
    bundle, moved = routing.move_video_bundle(STEM, src, queue, "why")

    assert bundle == queue / STEM
    assert sorted(moved) == sorted(names)
    for n in names + [f"{STEM}_reaches.json"]:
        assert (bundle / n).is_file(), n
        assert not (src / n).exists(), n
    man = _routing_manifest(bundle)
    assert man["failed_files"] == []
    assert man["moved_files"] == sorted(names)
    assert not (queue / routing.INCOMING_DIR_NAME).exists()


def test_manifest_write_failure_still_publishes_the_bundle(tmp_path, monkeypatch):
    """Every file has already left its source when the routing manifest is
    written. A failed write there must not strand them in the hidden staging
    folder: the bundle is published and the manifest written into it."""
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"
    real_write = routing._write_routing_manifest
    calls = []

    def _flaky_write(folder, vid, man):
        calls.append(folder)
        if len(calls) == 1:
            raise OSError("share hiccup")
        real_write(folder, vid, man)

    monkeypatch.setattr(routing, "_write_routing_manifest", _flaky_write)
    bundle, moved = routing.move_video_bundle(STEM, src, queue, "why")

    assert calls == [queue / routing.INCOMING_DIR_NAME / STEM, queue / STEM]
    for n in names:
        assert (bundle / n).is_file(), n
    assert _routing_manifest(bundle)["moved_files"] == sorted(names)
    assert not (queue / routing.INCOMING_DIR_NAME).exists()


def test_prefix_boundary_respected(tmp_path):
    src, names = _make_source(tmp_path)
    longer = "20250101_CNT0101_P11"
    other = [f"{longer}.mp4", f"{longer}_segments.json"]
    for n in other:
        (src / n).write_text("y", encoding="utf-8")
    queue = tmp_path / "triage"

    bundle, moved = routing.move_video_bundle(STEM, src, queue, "why")

    assert sorted(moved) == sorted(names)
    for n in other:
        assert (src / n).exists()
        assert not (bundle / n).exists()


def test_route_to_queue_writes_review_manifest_into_final_folder(tmp_path, monkeypatch):
    src, names = _make_source(tmp_path)
    queue = tmp_path / "triage"
    written = []
    monkeypatch.setattr(
        rg, "_write_review_manifest",
        lambda b, v, r: written.append(b) or (b / f"{v}_manifest.json").write_text(
            "{}", encoding="utf-8"))

    bundle = rg.route_to_queue(STEM, src, queue, "why", db=None)

    assert bundle == queue / STEM
    assert written == [queue / STEM]
    assert (bundle / f"{STEM}_manifest.json").exists()
    assert (bundle / f"{STEM}_routing.json").exists()
    assert not (queue / routing.INCOMING_DIR_NAME).exists()
