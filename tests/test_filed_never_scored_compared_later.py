"""A video filed before its hand scores existed is compared once they arrive,
and every disagreement is listed for a person -- the filed video is never
moved, re-filed or written to.

WHY: for a while a video with no hand scores in a project that requires them
was filed "never scored by hand", uncompared (its outcomes file keeps
"hand_scores": {"verdict": "never_scored"}). Missing scores are the
pipeline's defect, and when a later sheet import brought them nothing ever
compared them: a pipeline-vs-bench disagreement stayed in the analysis data
with no flag. Whether such a filed video is re-opened is not decided, so it
is only listed.

tmp_path only, invented ids.
"""
import json
import os
import time
from types import SimpleNamespace

import pytest

import mousereach.watcher.hand_scores as hs
import mousereach.watcher.route_cli as rc
from mousereach.watcher.attention import read_items

VID = "20990709_ABC0207_P1"          # filed never_scored; scores arrive, pellet 2 differs
AGREE = "20990709_ABC0208_P1"        # filed never_scored; scores arrive and agree
MISS = "20990709_ABC0209_P1"         # filed never_scored; import ran, still no scores
LATE = "20990709_ABC0210_P1"         # filed never_scored; no import since
COMPARED = "20990709_ABC0211_P1"     # filed and compared already: never looked at again


def _outcomes(folder, vid, verdict, read_at="2099-07-10T10:00:00"):
    doc = {"segments": [{"segment_num": 1, "outcome": "retrieved"},
                        {"segment_num": 2, "outcome": "untouched"}],
           "hand_scores": {"pellets": {}, "verdict": verdict, "source_mtime": 1.0,
                           "cohort_last_import": None, "read_at": read_at}}
    p = folder / f"{vid}_pellet_outcomes.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    from mousereach.config import Paths
    nas = tmp_path / "nas"
    analyzed = nas / "Analyzed" / "ABC" / "ABC_02"
    analyzed.mkdir(parents=True)
    (nas / "Analyzed" / "Archive" / "old").mkdir(parents=True)
    for name, val in {"NAS_ROOT": nas, "ANALYZED_OUTPUT": nas / "Analyzed",
                      "REPOSE_QUEUE": None, "DLC_STAGING": None}.items():
        monkeypatch.setattr(Paths, name, val, raising=False)
    files = {v: _outcomes(analyzed, v, "never_scored") for v in (VID, AGREE, MISS)}
    files[LATE] = _outcomes(analyzed, LATE, "never_scored", read_at="2099-07-30T10:00:00")
    files[COMPARED] = _outcomes(analyzed, COMPARED, "agrees")
    # a superseded copy under Archive/ is never read
    _outcomes(nas / "Analyzed" / "Archive" / "old", "20990709_ABC0212_P1", "never_scored")
    scores = tmp_path / "hand.csv"
    scores.write_text("video_key,pellet,score\n"
                      f"{VID},1,2\n{VID},2,1\n"
                      f"{AGREE},1,2\n{AGREE},2,0\n"
                      f"{COMPARED},1,0\n", encoding="utf-8")
    scores.with_suffix(".json").write_text(json.dumps(
        {"last_successful_import": {"ABC_02": "2099-07-20T20:00:00"},
         "sheets_seen": ["ABC_02"]}), encoding="utf-8")
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(scores),
                                                 "required_projects": ["ABC"]})
    hs._CACHE.update(key=None)
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())
    before = {v: (p.read_bytes(), os.stat(p).st_mtime_ns) for v, p in files.items()}
    return SimpleNamespace(nas=nas, analyzed=analyzed, files=files, before=before)


def _scan(env, monkeypatch):
    """One scan of the processing server's watcher, everything else stubbed."""
    import mousereach.watcher.orchestrator as o
    orch = object.__new__(o.ProcessingOrchestrator)
    now = time.monotonic()
    for k, v in {"staging_dir": env.nas / "Posed", "processing_dir": None,
                 "hostname": "HOST", "db": SimpleNamespace(get_video=lambda v: None),
                 "state": SimpleNamespace(discover_dlc_staged=lambda d: []),
                 "_last_review_return_mono": now, "_review_return_min_s": 1e9,
                 "_last_shared_record_mono": now, "_shared_record_min_s": 1e9,
                 "_last_retire_mono": now, "_retire_min_s": 1e9}.items():
        setattr(orch, k, v)
    (env.nas / "Posed").mkdir(exist_ok=True)
    orch._sweep_posed_claims = lambda: None
    orch._rescue_misfiled_singles = lambda: None
    orch._scan_phase()
    r = getattr(orch, "_filed_score_rechecker", None)
    if r is not None and r._thread is not None:
        r._thread.join(30)
    return [i for i in read_items(env.nas) if i.get("listed_in") == "HOST_filed_hand_scores.json"]


def test_a_disagreement_found_after_filing_is_listed_for_a_person(env, monkeypatch):
    items = _scan(env, monkeypatch)
    dis = [i for i in items if i.get("video_id") == VID]
    assert dis and dis[0]["needs_person"] is True
    assert "pellet 2: the hand score says displaced; the algorithm says untouched" in dis[0]["says"]
    dis[0]["says"].encode("ascii")


def test_filed_videos_are_never_moved_or_written(env, monkeypatch):
    _scan(env, monkeypatch)
    for v, p in env.files.items():
        assert (p.read_bytes(), os.stat(p).st_mtime_ns) == env.before[v], v


def test_only_what_needs_a_person_is_listed(env, monkeypatch):
    items = _scan(env, monkeypatch)
    named = {i.get("video_id") for i in items} | {
        v for i in items for v in i.get("video_ids", [])}
    assert AGREE not in named            # agrees: nothing to do
    assert LATE not in named             # no import since it was filed: not missing yet
    assert COMPARED not in named         # compared at filing: never looked at again
    assert "20990709_ABC0212_P1" not in named      # superseded copy under Archive/
    missing = [i for i in items if i["kind"] == "filed_hand_scores_missing"]
    assert len(missing) == 1 and missing[0]["video_ids"] == [MISS]
    assert "ABC_02" in missing[0]["says"] and "Fix the sheet" in missing[0]["says"]


def test_it_runs_again_only_when_the_scores_file_changes(env, monkeypatch):
    from mousereach.watcher import filed_hand_scores as fhs
    r = fhs.Rechecker()
    assert r.maybe_start(env.nas, "HOST", env.nas / "Analyzed")
    r._thread.join(30)
    assert not r.maybe_start(env.nas, "HOST", env.nas / "Analyzed")
    p = hs.settings()["path"]
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert r.maybe_start(env.nas, "HOST", env.nas / "Analyzed")
    r._thread.join(30)


def test_numbering_in_doubt_is_listed_as_not_compared(env, monkeypatch):
    def doubt(video_id, folder):
        raise hs.PelletNumbersUnknown("a reviewer said segment 2 shows pellet 3")
    monkeypatch.setattr(hs, "check", doubt)
    from mousereach.watcher import filed_hand_scores as fhs
    items = fhs.compare(fhs.never_scored_filed(env.nas / "Analyzed"))
    assert {i["kind"] for i in items} == {"filed_hand_scores_not_compared"}
    assert all("stays filed and unchanged" in i["says"] for i in items)


def test_only_the_sheet_fix_names_its_cohort(env, monkeypatch):
    """A database tool copies a cohort's red items into a file beside its
    sheet; only the item whose remedy is the sheet may name the cohort."""
    items = _scan(env, monkeypatch)
    by_kind = {i["kind"]: i for i in items}
    assert by_kind["filed_hand_score_disagreement"]["cohort_id"] is None
    assert by_kind["filed_hand_scores_missing"]["cohort_id"] == "ABC_02"


def test_a_list_that_could_not_be_written_is_tried_again(env, monkeypatch):
    """When the share cannot be written at that moment the run is not
    counted as done: the next scan tries again instead of leaving the old
    list until the scores file changes."""
    from mousereach.watcher import attention
    from mousereach.watcher import filed_hand_scores as fhs
    real = attention.write_list
    monkeypatch.setattr(attention, "write_list", lambda *a, **k: False)
    r = fhs.Rechecker()
    assert r.maybe_start(env.nas, "HOST", env.nas / "Analyzed")
    r._thread.join(30)
    assert not [i for i in read_items(env.nas)
                if i.get("listed_in") == "HOST_filed_hand_scores.json"]
    monkeypatch.setattr(attention, "write_list", real)
    assert r.maybe_start(env.nas, "HOST", env.nas / "Analyzed")   # same stamp, retried
    r._thread.join(30)
    assert [i for i in read_items(env.nas) if i.get("video_id") == VID]


@pytest.mark.parametrize("gone", ["missing_root", "no_outcomes_files"])
def test_an_unreadable_or_empty_tree_never_wipes_the_list(env, monkeypatch, gone):
    """An empty walk (share unreachable, folder gone) is not "nothing to
    list": the red items stay, and the run is tried again on the next scan."""
    from mousereach.watcher import filed_hand_scores as fhs
    r = fhs.Rechecker()
    assert r.maybe_start(env.nas, "HOST", env.nas / "Analyzed")
    r._thread.join(30)
    before = [i for i in read_items(env.nas)
              if i.get("listed_in") == "HOST_filed_hand_scores.json"]
    assert before
    if gone == "missing_root":
        root = env.nas / "Gone"
    else:
        root = env.nas / "EmptyAnalyzed"
        (root / "ABC" / "ABC_02").mkdir(parents=True)
    with pytest.raises(fhs.ListNotWritten):
        fhs.run(env.nas, "HOST", root)
    after = [i for i in read_items(env.nas)
             if i.get("listed_in") == "HOST_filed_hand_scores.json"]
    assert [i["says"] for i in after] == [i["says"] for i in before]
    # through the Rechecker: the stamp is not marked done
    r2 = fhs.Rechecker()
    assert r2.maybe_start(env.nas, "HOST", root)
    r2._thread.join(30)
    assert r2.maybe_start(env.nas, "HOST", root)
    r2._thread.join(30)


@pytest.mark.parametrize("status", [hs.LABEL_MISMATCH, hs.SHEET_ERROR, hs.NOT_CHECKED])
def test_a_check_that_ran_but_did_not_compare_is_listed_by_verdict(env, monkeypatch, status):
    """A verdict other than agrees/disagrees/waiting/suspended must never fall
    through silently: it would read as "compared, agrees"."""
    monkeypatch.setattr(hs, "check", lambda v, f: hs.Verdict(status, "test reason"))
    from mousereach.watcher import filed_hand_scores as fhs
    items = fhs.compare(fhs.never_scored_filed(env.nas / "Analyzed"))
    assert items, "nothing listed for a verdict that did not compare"
    assert {i["kind"] for i in items} == {"filed_hand_scores_not_compared"}
    assert all(status in i["says"] and "stays filed and unchanged" in i["says"]
               for i in items)
