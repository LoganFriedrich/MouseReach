"""The version scan reconciles disk against db, and diverts leave no husks.

WHY: videos archived by another node had no row on this server, so no
version scan anywhere covered them -- 907 accumulated invisibly and were
adopted by hand once (2026-09-08). The scan now adopts them itself. And
route_to_queue used to leave the emptied per-video source dir behind, which
read as a seg-failed bundle and re-diverted "(0 files)" forever.
"""
import json

import mousereach.watcher.review_gate as rg
import mousereach.pipeline.versions as versions_mod
from mousereach.watcher.reprocessor import ReprocessingScanner


class StubConn:
    def execute(self, *a):
        return []
    def close(self):
        pass


class StubDB:
    def __init__(self):
        self.registered = []
        self.forced = []
    def _get_connection(self):
        return StubConn()
    def get_videos_in_state(self, state):
        return []
    def register_video(self, video_id, source_path=None, current_path=None, **k):
        self.registered.append((video_id, source_path))
    def force_state(self, video_id, state, **k):
        self.forced.append((video_id, state))


def test_scan_adopts_disk_archived_videos_without_rows(tmp_path, monkeypatch):
    coh = tmp_path / "Analyzed" / "Connectome" / "CNT99"
    coh.mkdir(parents=True)
    a = "20240101_ABC0101_P1"
    b = "20240101_ABC0102_P1"
    (coh / (a + "_processing_manifest.json")).write_text("{}", encoding="utf-8")
    (coh / (a + ".mp4")).write_text("x", encoding="utf-8")
    (coh / (b + "_processing_manifest.json")).write_text("{}", encoding="utf-8")
    # b deliberately has NO mp4 -> named, not adopted

    monkeypatch.setattr(versions_mod, "get_current_versions",
                        lambda root: {"versions": {"dlc_scorer": "X"}})
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])

    sc = object.__new__(ReprocessingScanner)
    sc.db = StubDB()
    sc.nas_root = tmp_path
    sc.archive_dir = tmp_path / "Analyzed"
    sc._manifest_cache = {}

    summary = sc.scan(mark_outdated=True)

    assert summary["adopted"] == 1
    assert sc.db.registered and sc.db.registered[0][0] == a
    assert str(coh / (a + ".mp4")) in sc.db.registered[0][1]
    assert (a, "archived") in sc.db.forced
    assert summary["adopt_no_mp4"] == [b]


def test_route_to_queue_removes_emptied_per_video_source(tmp_path, monkeypatch):
    stem = "20240101_ABC0101_P1"
    src = tmp_path / stem                      # per-video bundle dir, emptied
    src.mkdir()
    bundle = tmp_path / "queue" / stem
    bundle.mkdir(parents=True)
    monkeypatch.setattr(rg, "move_video_bundle",
                        lambda *a, **k: (bundle, 0))
    monkeypatch.setattr(rg, "_write_review_manifest", lambda *a, **k: None)
    rg.route_to_queue(stem, src, tmp_path / "queue", "why", db=None)
    assert not src.exists()                    # husk removed


def test_route_to_queue_never_removes_a_shared_dir(tmp_path, monkeypatch):
    stem = "20240101_ABC0101_P1"
    src = tmp_path / "Processing"              # shared dir, not stem-named
    src.mkdir()
    bundle = tmp_path / "queue" / stem
    bundle.mkdir(parents=True)
    monkeypatch.setattr(rg, "move_video_bundle",
                        lambda *a, **k: (bundle, 0))
    monkeypatch.setattr(rg, "_write_review_manifest", lambda *a, **k: None)
    rg.route_to_queue(stem, src, tmp_path / "queue", "why", db=None)
    assert src.exists()                        # shared dirs are sacred


def test_unmark_never_clears_a_hand_mark(tmp_path, monkeypatch):
    """A person's re-run mark must survive the two-way door even when the
    manifest compares current OUTRIGHT -- the usual reason for a hand-mark is
    precisely that the current-version outputs are wrong. The protection used
    to live only in the compat branch, so the dashboard Re-run button's
    promise self-cancelled within one scan for every version-current video
    (2026-09-08, caught when a repair re-mark was wiped mid-repair)."""
    coh = tmp_path / "Analyzed" / "Connectome" / "CNT99"
    coh.mkdir(parents=True)
    stem = "20240101_ABC0101_P1"
    (coh / (stem + "_processing_manifest.json")).write_text("{}", encoding="utf-8")
    (coh / (stem + ".mp4")).write_text("x", encoding="utf-8")

    monkeypatch.setattr(versions_mod, "get_current_versions",
                        lambda root: {"versions": {"dlc_scorer": "X"}})
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])
    monkeypatch.setattr(versions_mod, "compare_manifest_to_current",
                        lambda m, c: {"is_current": True})

    class DB(StubDB):
        def __init__(self):
            super().__init__()
            self.unmarked = []
        def _get_connection(self):
            class C:
                def execute(self, *a):
                    return [(stem,)]          # row exists -> no adoption
                def close(self):
                    pass
            return C()
        def get_videos_in_state(self, state):
            if state == "outdated":
                return [{"video_id": stem, "mark_reason": "human said so",
                         "reprocess_scope": "segmentation"}]
            return []
        def update_state(self, vid, st, **k):
            self.unmarked.append(vid)

    sc = object.__new__(ReprocessingScanner)
    sc.db = DB()
    sc.nas_root = tmp_path
    sc.archive_dir = tmp_path / "Analyzed"
    sc._manifest_cache = {}

    summary = sc.scan(mark_outdated=True)
    assert summary["unmarked_current"] == 0
    assert not sc.db.unmarked
    assert not sc.db.forced
