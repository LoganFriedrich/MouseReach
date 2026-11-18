"""Bring-current (pipeline/reprocess_to_current.py) keeps the watcher's rules.

WHY: the routine files into Analyzed and holds videos in the review queues,
but it did so with none of the watcher's rules: it built queue bundles
itself (a triage bundle beside a deep-review one, no watcher record), let a
quality check that could not run count as a pass, never compared with the
hand scores, finalized a crystallized video or one the watcher was working
on, finalized a clean video with no kinematics, and pushed to a database.

The algorithm steps are stood in (they are tested elsewhere); what is tested
is what the routine does with their results. Video ids are invented.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

V = "20990101_CNT9901_P1"


@pytest.fixture
def world(tmp_path, monkeypatch):
    from mousereach.config import Paths
    queues = {"triage": tmp_path / "Review" / "Pending",
              "deep": tmp_path / "Review" / "Deep_Review"}
    for q in queues.values():
        q.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", queues["triage"], raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", queues["deep"], raising=False)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path / "Review", raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)

    filed = tmp_path / "Analyzed" / "PRJ" / "C01"
    filed.mkdir(parents=True)
    mp4 = filed / f"{V}.mp4"
    mp4.write_bytes(b"video")
    (filed / f"{V}_features.json").write_text('{"old": true}', encoding="utf-8")
    pose = tmp_path / "pose" / f"{V}DLC_resnet101_X_shuffle3_100000.h5"
    pose.parent.mkdir()
    pose.write_bytes(b"pose")

    def out(d, name, doc):
        (d / f"{V}_{name}.json").write_text(json.dumps(doc), encoding="utf-8")

    import mousereach.segmentation.core.batch as sb
    import mousereach.reach.core.batch as rb
    import mousereach.outcomes.core.batch as ob
    import mousereach.assignment.run as ar
    import mousereach.pipeline.manifest as man
    import mousereach.pipeline.triage as tri
    import mousereach.watcher.review_gate as rg
    import mousereach.kinematics.core.feature_extractor as fe
    monkeypatch.setattr(sb, "process_single",
                        lambda h5, output_dir=None: out(output_dir, "segments", {"boundaries": [1]}))
    monkeypatch.setattr(rb, "process_single",
                        lambda h5, seg, output_dir=None: out(output_dir, "reaches", {"segments": []}))
    monkeypatch.setattr(ob, "process_single",
                        lambda h5, seg, reach, output_dir=None, video_dir=None:
                        out(output_dir, "pellet_outcomes", {"segments": []}))
    monkeypatch.setattr(ar, "assign_reaches_for_video", lambda d, vid, h5: {})
    monkeypatch.setattr(man, "create_processing_manifest", lambda **k: (
        k["processing_dir"] / f"{V}_processing_manifest.json").write_text(
            '{"pipeline_versions": {}}', encoding="utf-8"))
    import mousereach.archive.core as acore
    monkeypatch.setattr(acore, "get_archive_destination", lambda vid: filed)

    state = SimpleNamespace(triage_raises=False, gate="clean", hand="agrees",
                            row={"state": "archived"}, pushed=[],
                            features={"new": True}, extract_raises=False)

    class FakeTriage:
        verdict = "auto_approved"

        def save(self, path):
            path.write_text("{}", encoding="utf-8")

    def fake_triage(**k):
        if state.triage_raises:
            raise RuntimeError("the quality check broke")
        return FakeTriage()

    monkeypatch.setattr(tri, "triage_video", fake_triage)
    monkeypatch.setattr(rg, "evaluate_gate", lambda vid, d, qc_verdict=None:
                        (state.gate, "test reason", {}))

    class FakeFeatures:
        n_segments = 0

        def to_dict(self):
            return dict(state.features)

    class FakeExtractor:
        def extract(self, *a, **k):
            if state.extract_raises:
                raise ValueError("no frame rate for this video")
            return FakeFeatures()

    monkeypatch.setattr(fe, "FeatureExtractor", FakeExtractor)
    import mousereach.kinematics.analysis.reach_export as rx
    monkeypatch.setattr(rx, "write_video_results_csv", lambda p: None)

    from mousereach.watcher import hand_scores
    monkeypatch.setattr(hand_scores, "check", lambda vid, d: hand_scores.Verdict(
        {"agrees": hand_scores.AGREES, "waiting": hand_scores.WAITING}[state.hand], "test"))
    monkeypatch.setattr(hand_scores, "record", lambda *a, **k: True)

    from mousereach.watcher import db_location
    fake_db = SimpleNamespace(get_video=lambda vid: dict(state.row),
                              update_state=lambda *a, **k: None,
                              set_fields=lambda *a, **k: None,
                              log_step=lambda *a, **k: None)
    monkeypatch.setattr(db_location, "open_watcher_db_if_present", lambda: (fake_db, None))

    import mousereach.sync.database as sdb
    monkeypatch.setattr(sdb, "sync_file_to_database",
                        lambda p, *a, **k: state.pushed.append(p) or True)

    def run(finalize=True):
        from mousereach.pipeline import reprocess_to_current as rtc
        return rtc.reprocess_video_to_current(
            V, pose, video_file=mp4, work_dir=tmp_path / "work",
            finalize=finalize, skip_if_current=False, archive_root=tmp_path / "Archive")

    return SimpleNamespace(run=run, state=state, filed=filed, queues=queues, pose=pose,
                           work=tmp_path / "work", nas=tmp_path / "nas",
                           archive=tmp_path / "Archive")


def _filed_untouched(w):
    return json.loads((w.filed / f"{V}_features.json").read_text()) == {"old": True}


def _queued(w):
    return [p.name for q in w.queues.values() for p in q.iterdir()]


def test_a_quality_check_that_cannot_run_finalizes_nothing(world):
    world.state.triage_raises = True
    out = world.run()
    assert "quality check could not run" in (out.get("error") or ""), out
    assert _filed_untouched(world) and _queued(world) == []


def test_a_session_without_hand_scores_is_held_not_filed(world):
    world.state.hand = "waiting"
    out = world.run()
    assert "no hand scores" in (out.get("error") or ""), out
    assert _filed_untouched(world) and _queued(world) == []


def test_a_crystallized_video_is_not_finalized(world):
    world.state.row = {"state": "crystallized", "crystallized_label": "PAPER_X"}
    out = world.run()
    assert "not finalized" in (out.get("error") or ""), out
    assert _filed_untouched(world) and _queued(world) == []


def test_a_video_the_watcher_is_working_on_is_not_finalized(world):
    world.state.row = {"state": "processing"}
    out = world.run()
    assert "working on it" in (out.get("error") or ""), out
    assert _filed_untouched(world)


def test_a_clean_video_is_filed_and_nothing_is_pushed_to_a_database(world):
    out = world.run()
    assert out.get("error") is None, out
    assert json.loads((world.filed / f"{V}_features.json").read_text()) == {"new": True}
    assert world.state.pushed == []


def test_a_stale_outputs_gate_answer_is_reported_not_sent_to_a_queue(world):
    """The gate's "stale outputs" answer means the reaches or outcomes describe
    cuts that no longer exist: no reviewer may be shown them and nothing may
    be filed. Bring-current used to send every non-clean answer to triage."""
    world.state.gate = "stale_outputs"
    out = world.run()
    assert "gate answered 'stale_outputs'" in (out.get("error") or ""), out
    assert _queued(world) == [], _queued(world)
    assert _filed_untouched(world)


def test_a_held_video_never_gets_a_triage_copy_beside_its_deep_review_bundle(world):
    """Deep review beats triage: the routine goes through the watcher's router."""
    deep = world.queues["deep"] / V
    deep.mkdir()
    (deep / f"{V}_segments.json").write_text("{}", encoding="utf-8")
    world.state.gate = "triage"
    out = world.run()
    assert not (world.queues["triage"] / V).exists(), out
    assert _filed_untouched(world)


# --- finalize is all or nothing; holds are visible -----------------------------

def _attention(w):
    """Every item of this machine's bring-current attention list (empty when none)."""
    items = []
    for f in (w.nas / "attention").glob("*_bring_current_held.json"):
        items += json.loads(f.read_text(encoding="utf-8")).get("items") or []
    return items


def _filed_features(w):
    p = w.filed / f"{V}_features.json"
    return json.loads(p.read_text()) if p.is_file() else None


def test_a_failed_filing_part_way_puts_the_old_results_back_and_is_an_error(
        world, monkeypatch):
    """Finalize moved the files itself, features FIRST, and dropped a failed
    move silently: the filed folder could hold new kinematics without the
    reaches they were computed from, reported as success. It now files
    through the watcher's own one-step filing."""
    import mousereach.pipeline.fsutil as fsutil
    real = fsutil.rename_with_retry

    def failing_rename(src, dst, *a, **k):
        if Path(dst).name == f"{V}_reaches.json":
            raise OSError("the share went away")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(fsutil, "rename_with_retry", failing_rename)
    out = world.run()
    assert out.get("error"), out
    assert _filed_features(world) == {"old": True}, "the previous generation must stand"
    assert [i["video_id"] for i in _attention(world)] == [V]


def test_the_features_file_is_put_in_place_last(world, monkeypatch):
    import mousereach.pipeline.fsutil as fsutil
    real, order = fsutil.rename_with_retry, []

    def recording_rename(src, dst, *a, **k):
        if not Path(dst).name.endswith(".tmp"):
            order.append(Path(dst).name)
        return real(src, dst, *a, **k)

    monkeypatch.setattr(fsutil, "rename_with_retry", recording_rename)
    out = world.run()
    assert out.get("error") is None, out
    assert order and order[-1] == f"{V}_features.json", order
    assert _filed_features(world) == {"new": True}


def test_an_old_file_that_cannot_be_kept_stops_finalize_before_anything_changes(
        world, monkeypatch):
    """The previous generation is kept in the Archive first; a file that could
    not be kept was overwritten by the new one, with no copy."""
    import mousereach.archive.supersede as sup
    real = sup._checksummed_move
    monkeypatch.setattr(sup, "_checksummed_move", lambda s, d, **k: (
        None if s.name.endswith("_features.json") else real(s, d, **k)))
    out = world.run()
    assert out.get("error"), out
    assert _filed_untouched(world), "the old features file was overwritten"


def test_a_run_without_its_own_manifest_is_not_filed(world, monkeypatch):
    """Filing replaces only what the run brings, so the old manifest would
    stay beside the new kinematics."""
    import mousereach.pipeline.manifest as man
    monkeypatch.setattr(man, "create_processing_manifest", lambda **k: {})
    out = world.run()
    assert "provenance manifest" in (out.get("error") or ""), out
    assert _filed_untouched(world)


def test_an_undecided_pellet_in_the_new_features_is_not_filed(world):
    world.state.features = {"segments": [{"segment_num": 1, "outcome": "triaged"}]}
    out = world.run()
    assert "no decided outcome" in (out.get("error") or ""), out
    assert _filed_untouched(world)


def test_failed_measures_in_the_new_features_are_not_filed(world):
    world.state.features = {"segments": [{"segment_num": 1, "outcome": "retrieved", "reaches": [
        {"reach_id": 2, "extended": {"_extended_features_error": "boom"}}]}]}
    out = world.run()
    assert "_extended_features_error" in (out.get("error") or ""), out
    assert _filed_untouched(world)


def test_a_hand_score_hold_is_printed_and_listed_for_a_person(world, capsys):
    """R4: hand scores always exist, so a missing one is held VISIBLY."""
    world.state.hand = "waiting"
    out = world.run()
    assert "no hand scores" in (out.get("error") or ""), out
    printed = capsys.readouterr().out
    assert V in printed and "no hand scores" in printed, printed
    printed.encode("ascii")                   # terminal output is ASCII only
    items = _attention(world)
    assert [i["video_id"] for i in items] == [V], items
    assert items[0]["needs_person"] is True
    assert _filed_untouched(world)


def test_a_list_that_could_not_be_written_is_not_reported_as_listed(world, capsys, monkeypatch):
    """summary['listed'] must say what was done: when the attention list
    cannot be written, it is False and the printed line says so."""
    import mousereach.watcher.attention as att
    monkeypatch.setattr(att, "note_video", lambda *a, **k: False)
    world.state.hand = "waiting"
    out = world.run()
    assert out.get("listed") is False, out
    assert "could NOT be put on the attention list" in capsys.readouterr().out
    assert _filed_untouched(world)


def test_a_dry_run_hold_prints_but_writes_no_list(world, capsys):
    world.state.hand = "waiting"
    world.run(finalize=False)
    assert "no hand scores" in capsys.readouterr().out
    assert not (world.nas / "attention").exists()


def test_filing_the_video_later_takes_it_off_the_list(world):
    world.state.hand = "waiting"
    world.run()
    assert _attention(world)
    world.state.hand = "agrees"
    out = world.run()
    assert out.get("error") is None, out
    assert _attention(world) == []



def test_the_watchers_filing_takes_a_bring_current_hold_off_the_list(tmp_path, monkeypatch):
    """A video held by bring-current and later filed by the watcher (its
    re-run) must not stay listed for a person."""
    import socket
    import mousereach.archive.core as core
    from mousereach.watcher.attention import note_video
    nas = tmp_path / "nas"
    proc, dest = tmp_path / "Processing", tmp_path / "Analyzed" / "C01"
    proc.mkdir(parents=True)
    dest.mkdir(parents=True)
    monkeypatch.setattr(core.Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(core.Paths, "PROCESSING", proc)
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)
    import mousereach.index as index_pkg

    class _NoIndex:
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass

    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)
    note_video(nas, socket.gethostname(), "bring_current_held", V, "held for a test")
    (proc / f"{V}.mp4").write_bytes(b"v")
    (proc / f"{V}_features.json").write_text("{}")
    r = core.archive_video(V, verbose=False, skip_ready_check=True)
    assert r["success"], r
    items = json.loads((nas / "attention" / f"{socket.gethostname()}_bring_current_held.json")
                       .read_text(encoding="utf-8"))["items"]
    assert items == []


def test_a_failed_pose_record_copy_is_named_in_the_reason(world, monkeypatch):
    """The frame-rate record's copy result was ignored: a failed copy then
    read as "no frame rate", sending a person after the wrong problem."""
    meta = world.pose.with_name(world.pose.stem + "_meta.pickle")
    meta.write_bytes(b"meta")
    import mousereach.watcher.transfer as tr
    real = tr.safe_copy
    monkeypatch.setattr(tr, "safe_copy", lambda s, d, verify=True: (
        False if s.name.endswith("_meta.pickle") else real(s, d, verify=verify)))
    world.state.extract_raises = True
    out = world.run()
    assert "frame-rate record" in (out.get("error") or ""), out
    assert _filed_untouched(world)


# --- every finalize stop is printed, and listed unless nothing would clear it --

def _shown_and_listed(w, capsys, out, words):
    """The stop is in the summary, printed in plain ASCII, and listed once."""
    assert words in (out.get("error") or ""), out
    printed = capsys.readouterr().out
    assert V in printed and words in printed, printed
    printed.encode("ascii")
    assert [i["video_id"] for i in _attention(w)] == [V], _attention(w)


def test_a_run_without_its_own_manifest_is_shown_to_a_person(world, monkeypatch, capsys):
    import mousereach.pipeline.manifest as man
    monkeypatch.setattr(man, "create_processing_manifest", lambda **k: {})
    out = world.run()
    _shown_and_listed(world, capsys, out, "provenance manifest")
    assert _filed_untouched(world)


def test_an_undecided_pellet_refusal_is_shown_to_a_person(world, capsys):
    world.state.features = {"segments": [{"segment_num": 1, "outcome": "triaged"}]}
    out = world.run()
    _shown_and_listed(world, capsys, out, "no decided outcome")
    assert _filed_untouched(world)


def test_a_failed_measures_refusal_is_shown_to_a_person(world, capsys):
    world.state.features = {"segments": [{"segment_num": 1, "outcome": "retrieved", "reaches": [
        {"reach_id": 2, "extended": {"_extended_features_error": "boom"}}]}]}
    out = world.run()
    _shown_and_listed(world, capsys, out, "_extended_features_error")
    assert _filed_untouched(world)


def test_a_video_outside_its_filing_folder_is_shown_to_a_person(
        world, monkeypatch, tmp_path, capsys):
    import mousereach.archive.core as acore
    elsewhere = tmp_path / "Analyzed" / "PRJ" / "C02"
    elsewhere.mkdir(parents=True)
    monkeypatch.setattr(acore, "get_archive_destination", lambda vid: elsewhere)
    out = world.run()
    _shown_and_listed(world, capsys, out, "where the filing rules file it")
    assert _filed_untouched(world) and list(elsewhere.iterdir()) == []


def test_a_pose_copy_that_cannot_be_set_aside_is_shown_to_a_person(
        world, monkeypatch, capsys):
    real = Path.replace

    def refusing_replace(self, target):
        if self.suffix == ".h5":
            raise OSError("the pose copy is in use")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", refusing_replace)
    out = world.run()
    _shown_and_listed(world, capsys, out, "could not set the pose copy aside")
    assert _filed_untouched(world)


def test_a_failed_filing_is_printed_too(world, monkeypatch, capsys):
    import mousereach.pipeline.fsutil as fsutil
    real = fsutil.rename_with_retry

    def failing_rename(src, dst, *a, **k):
        if Path(dst).name == f"{V}_reaches.json":
            raise OSError("the share went away")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(fsutil, "rename_with_retry", failing_rename)
    out = world.run()
    _shown_and_listed(world, capsys, out, "not filed")
    assert _filed_features(world) == {"old": True}


def test_a_video_that_could_not_be_put_in_a_queue_is_shown_to_a_person(
        world, monkeypatch, capsys):
    """Neither filed nor held: it used to be only inside summary['held'],
    with no error at all."""
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", None, raising=False)
    world.state.gate = "triage"
    out = world.run()
    _shown_and_listed(world, capsys, out, "not held for review")
    assert _filed_untouched(world)


def test_a_crystallized_video_is_printed_but_never_listed(world, capsys):
    """The lock is deliberate and no filing will ever clear the item, so it
    would stay on the red panel for good."""
    world.state.row = {"state": "crystallized", "crystallized_label": "PAPER_X"}
    out = world.run()
    assert "not finalized" in (out.get("error") or ""), out
    printed = capsys.readouterr().out
    assert V in printed and "publication lock is deliberate" in printed, printed
    assert "Fix that" not in printed, printed
    assert _attention(world) == []
    assert out.get("listed") is False


def test_a_video_the_watcher_is_working_on_is_printed_but_never_listed(world, capsys):
    world.state.row = {"state": "processing"}
    out = world.run()
    assert "working on it" in (out.get("error") or ""), out
    printed = capsys.readouterr().out
    assert V in printed and "Left to the watcher" in printed, printed
    assert "Fix that" not in printed, printed
    assert _attention(world) == []


def test_a_machine_without_a_watcher_database_is_printed_but_never_listed(
        world, monkeypatch, capsys):
    from mousereach.watcher import db_location
    monkeypatch.setattr(db_location, "open_watcher_db_if_present",
                        lambda: (None, "no watcher database here"))
    out = world.run()
    assert "cannot tell whether it is crystallized" in (out.get("error") or ""), out
    printed = capsys.readouterr().out
    assert V in printed and "machine whose watcher database holds it" in printed, printed
    assert _attention(world) == []
    assert _filed_untouched(world)


def test_older_files_left_beside_a_filed_generation_are_printed_but_never_listed(
        world, monkeypatch, capsys):
    """No later filing clears such an item (bring-current skips a video whose
    filed results are current), so it would stay on the red panel for good."""
    (world.filed / f"{V}_reach_assignments.json").write_text('{"old": true}', encoding="utf-8")
    import mousereach.archive.supersede as sup
    real = sup.supersede_video_outputs

    def sweep(video_id, src, archive_root=None, **k):
        if k.get("only") and not k.get("dry_run") and not k.get("incoming"):
            return {"error": "the Archive is unreachable"}
        return real(video_id, src, archive_root, **k)

    monkeypatch.setattr(sup, "supersede_video_outputs", sweep)
    out = world.run()
    assert _filed_features(world) == {"new": True}
    assert "older files beside it could not be moved" in (out.get("error") or ""), out
    printed = capsys.readouterr().out
    assert "Move them by hand" in printed, printed
    assert _attention(world) == []


def test_a_call_that_raises_is_shown_to_a_person_and_raised_on(world, monkeypatch, capsys):
    """A call outside the steps' own guards (here the gate) used to end
    bring-current with nothing printed or listed."""
    import mousereach.watcher.review_gate as rg

    def broken_gate(*a, **k):
        raise RuntimeError("the gate broke")

    monkeypatch.setattr(rg, "evaluate_gate", broken_gate)
    with pytest.raises(RuntimeError):
        world.run()
    printed = capsys.readouterr().out
    assert V in printed and "the gate broke" in printed, printed
    assert [i["video_id"] for i in _attention(world)] == [V], _attention(world)
    assert _filed_untouched(world)


def test_filed_with_a_work_copy_left_is_not_reported_as_not_filed(world, monkeypatch, capsys):
    """Every file was filed; only a copy in bring-current's own work folder
    could not be removed afterward. That used to read 'not filed ... run
    bring-current again', and the older files beside the filed results were
    then left unswept."""
    leftover = world.filed / f"{V}_reach_assignments.json"
    leftover.write_text('{"old": true}', encoding="utf-8")
    import mousereach.pipeline.fsutil as fsutil
    real = fsutil.retry_transient

    def failing_unlink(fn, *a, **k):
        target = getattr(fn, "__self__", None)
        if (getattr(fn, "__name__", "") == "unlink" and isinstance(target, Path)
                and target.name == f"{V}_features.json" and world.work in target.parents):
            raise OSError("the work copy is in use")
        return real(fn, *a, **k)

    monkeypatch.setattr(fsutil, "retry_transient", failing_unlink)
    out = world.run()
    assert _filed_features(world) == {"new": True}
    err = out.get("error") or ""
    assert not err.startswith("not filed"), out
    printed = capsys.readouterr().out
    assert "run bring-current again" not in printed, printed
    assert "could not be removed" in printed, printed
    assert out.get("work_copy_left"), out
    assert f"{V}_features.json" in (out.get("finalized_moved") or []), out
    assert f"{V}_reaches.json" in (out.get("finalized_moved") or []), out
    assert out.get("decision_final") == "clean", out
    assert not leftover.exists(), "the older generation's file beside it was not swept"
    assert _attention(world) == []


def test_clearing_the_bring_current_list_never_skips_the_persons_cuts_list(
        tmp_path, monkeypatch):
    """The filing clears two lists; a failure clearing one must not keep the
    other from being cleared."""
    import socket
    import mousereach.archive.core as core
    import mousereach.watcher.attention as att
    nas = tmp_path / "nas"
    proc, dest = tmp_path / "Processing", tmp_path / "Analyzed" / "C01"
    proc.mkdir(parents=True)
    dest.mkdir(parents=True)
    monkeypatch.setattr(core.Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(core.Paths, "PROCESSING", proc)
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)
    import mousereach.index as index_pkg

    class _NoIndex:
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass

    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)
    att.note_video(nas, socket.gethostname(), core.HELD_PERSONS_CUTS_KIND, V, "held")
    real_drop = att.drop_video

    def drop(nas_root, host, kind, vid):
        if kind == "bring_current_held":
            raise RuntimeError("that list is broken")
        return real_drop(nas_root, host, kind, vid)

    monkeypatch.setattr(att, "drop_video", drop)
    (proc / f"{V}.mp4").write_bytes(b"v")
    (proc / f"{V}_features.json").write_text("{}")
    r = core.archive_video(V, verbose=False, skip_ready_check=True)
    assert r["success"], r
    path = nas / "attention" / f"{socket.gethostname()}_{core.HELD_PERSONS_CUTS_KIND}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["items"] == []
