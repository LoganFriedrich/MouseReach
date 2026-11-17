"""The archive step never files a Pillar video whose kinematics still hold a
pellet nobody decided.

WHY: the review gate passed videos whose open pellets ground truth beside
the video answered, while the kinematics did not find that ground truth and
wrote the algorithm's "triaged" into the results -- which were then filed and
exported as an undecided pellet. Whatever makes two readers disagree, the
filing step checks the file it is about to file.

Invented ids, tmp_path only.
"""
import json

from mousereach.watcher.orchestrator import refuse_filing_without_features


class FakeDB:
    def __init__(self):
        self.failed = {}
        self.steps = []

    def mark_failed(self, vid, msg):
        self.failed[vid] = msg

    def log_step(self, vid, step, status, message=None, **_):
        self.steps.append((vid, step, status, message))

    def get_video(self, vid):
        return {}


def _features(folder, vid, outcomes):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{vid}.mp4").write_bytes(b"x")
    # The run's provenance: the filing step refuses a video without it.
    (folder / f"{vid}_processing_manifest.json").write_text("{}", encoding="utf-8")
    (folder / f"{vid}_features.json").write_text(json.dumps({
        "video_name": vid,
        "segments": [{"segment_num": i + 1, "outcome": o, "reaches": []}
                     for i, o in enumerate(outcomes)]}), encoding="utf-8")


def test_a_pillar_video_with_a_triaged_pellet_is_not_filed(tmp_path):
    vid = "20990101_ABC0101_P1"
    _features(tmp_path, vid, ["retrieved", "triaged", "untouched", None])
    db = FakeDB()
    assert refuse_filing_without_features(db, vid, tmp_path) is True
    assert vid in db.failed
    assert "2, 4" in db.failed[vid]          # the open pellets are named
    assert "undecided" in db.failed[vid]


def test_a_pillar_video_with_every_pellet_decided_is_filed(tmp_path):
    vid = "20990101_ABC0101_P2"
    _features(tmp_path, vid, ["retrieved", "displaced_sa", "untouched"])
    db = FakeDB()
    assert refuse_filing_without_features(db, vid, tmp_path) is False
    assert db.failed == {}


def test_an_easy_tray_is_never_held_for_outcomes(tmp_path):
    vid = "20990101_ABC0101_E1"
    _features(tmp_path, vid, [None, "triaged"])
    db = FakeDB()
    assert refuse_filing_without_features(db, vid, tmp_path) is False
