"""One work order for the whole lab, honoured by processing AND human review.

WHY: the ordering policy was per-machine (~/.mousereach/config.json) while the
queues it orders are shared, and the human review tools read no policy at all
-- so a reviewer was handed a uniformly random video while the processing node
worked strictly by preference, and two machines could disagree with nothing to
say so. These tests pin the shared file, the two-list shape a lab actually
thinks in ("these projects first, these cohorts within them"), and the
reconciliation that keeps review sampling unbiased INSIDE a tier while still
working the important tier first.
"""
import json

import pytest

import mousereach.config as cfg
import mousereach.watcher.work_priority as wp
from mousereach.config import Paths

CNT01 = "20250624_CNT0101_P1"     # named cohort, pillar   -> most preferred
CNT01B = "20250625_CNT0102_P1"    # same tier as CNT01
CNT05 = "20250624_CNT0505_P1"     # CNT, unnamed cohort
ASPA = "20220318_ASPA0501_P4"     # second project
OTHER = "20250101_ZZZ0101_P1"     # no project preference
EASY = "20250624_CNT0101_E1"      # unsupported tray -> deferred


@pytest.fixture(autouse=True)
def _no_cached_policy():
    wp.invalidate_lab_policy()
    yield
    wp.invalidate_lab_policy()


@pytest.fixture
def nas(tmp_path, monkeypatch):
    """A shared drive of our own, and no inherited machine config: otherwise
    these tests would pass or fail according to whose laptop ran them."""
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path)

    class _NoLocalSetting:
        work_priority = None

    monkeypatch.setattr(cfg.WatcherConfig, "load",
                        classmethod(lambda cls: _NoLocalSetting()))
    return tmp_path


def _save(**kw):
    kw.setdefault("projects", ["CNT", "ASPA"])
    kw.setdefault("cohorts", {"CNT": ["01", "02", "03", "04"]})
    kw.setdefault("tray_types", ["P"])
    kw.setdefault("idle_only_tray_types", ["E", "F"])
    return wp.save_lab_priority(**kw)


def _tier(policy, video_id):
    return policy.tier({"video_id": video_id})


# ---------------------------------------------------------------- compiling

def test_a_cohort_is_named_both_bare_and_qualified():
    """A video's cohort resolves to "01" and "CNT01"; a lab may write either."""
    order = wp.compile_order(["CNT"], {"CNT": ["01"]}, ["P"])
    assert order[0]["cohort"] == ["01", "CNT01"]
    assert order[0]["project"] == ["CNT"]
    assert order[0]["tray_type"] == ["P"]


def test_an_already_qualified_cohort_is_not_doubled():
    order = wp.compile_order(["CNT"], {"CNT": ["CNT01"]}, ["P"])
    assert order[0]["cohort"] == ["CNT01"]


def test_named_cohorts_come_before_the_rest_of_their_project():
    order = wp.compile_order(["CNT", "ASPA"], {"CNT": ["01", "02"]}, ["P"])
    rendered = [(s.get("project"), s.get("cohort")) for s in order]
    assert rendered == [
        (["CNT"], ["01", "CNT01"]),
        (["CNT"], ["02", "CNT02"]),
        (["CNT"], None),
        (["ASPA"], None),
        (None, None),          # the bare tray tier
    ]


def test_identical_selectors_are_not_repeated():
    order = wp.compile_order(["CNT", "CNT"], {}, ["P"])
    assert len(order) == 2      # the CNT tier and the bare tray tier


def test_no_tray_preference_means_no_tray_key():
    order = wp.compile_order(["CNT"], {}, [])
    assert order == [{"project": ["CNT"]}]


# ---------------------------------------------------------------- the file

def test_save_then_read_round_trip(nas):
    path = _save()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["projects"] == ["CNT", "ASPA"]
    assert body["cohorts"] == {"CNT": ["01", "02", "03", "04"]}
    assert body["schema_version"] == wp.LAB_PRIORITY_SCHEMA
    assert body["updated_at"] and body["updated_by"]
    assert wp.read_lab_priority()["projects"] == ["CNT", "ASPA"]


def test_save_leaves_no_half_written_file_behind(nas):
    _save()
    assert not list(nas.glob("*.tmp"))
    assert (nas / wp.LAB_PRIORITY_FILENAME).is_file()


def test_saving_needs_a_shared_drive(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "NAS_ROOT", None)
    with pytest.raises(ValueError):
        wp.save_lab_priority(projects=["CNT"])


def test_an_unreadable_lab_file_falls_back_instead_of_failing(nas):
    (nas / wp.LAB_PRIORITY_FILENAME).write_text("{not json", encoding="utf-8")
    assert wp.read_lab_priority() is None
    policy = wp.load_lab_policy(max_age_s=0)      # shipped default, no raise
    assert _tier(policy, CNT01) == 0              # pillar-first default


# ---------------------------------------------------------------- precedence

def test_the_lab_file_beats_this_machines_own_setting(nas):
    _save(projects=["CNT"], cohorts={})
    policy = wp.load_lab_policy({"order": [{"project": ["ASPA"]}]}, max_age_s=0)
    assert _tier(policy, CNT01) < _tier(policy, ASPA)
    assert "lab file" in policy.source


def test_this_machines_setting_is_used_until_a_lab_file_exists(nas):
    policy = wp.load_lab_policy({"order": [{"project": ["ASPA"]}]}, max_age_s=0)
    assert _tier(policy, ASPA) == 0
    assert "this machine only" in policy.source


def test_the_shipped_default_when_there_is_neither(nas):
    policy = wp.load_lab_policy(max_age_s=0)
    assert _tier(policy, CNT01) == 0        # pillar
    assert policy.is_deferred({"video_id": EASY})


def test_raw_selectors_in_the_lab_file_are_honoured(nas):
    (nas / wp.LAB_PRIORITY_FILENAME).write_text(
        json.dumps({"order": [{"project": ["ASPA"]}]}), encoding="utf-8")
    policy = wp.load_lab_policy(max_age_s=0)
    assert _tier(policy, ASPA) == 0
    assert "raw selectors" in policy.source


# ---------------------------------------------------------------- caching

def test_the_policy_is_cached_and_zero_age_reads_now(nas, monkeypatch):
    _save()
    reads = []
    real = wp.read_lab_priority
    monkeypatch.setattr(wp, "read_lab_priority",
                        lambda *a, **k: (reads.append(1), real(*a, **k))[1])
    wp.load_lab_policy(max_age_s=0)
    assert len(reads) == 1
    wp.load_lab_policy()                 # inside the TTL -> served from memory
    assert len(reads) == 1
    wp.load_lab_policy(max_age_s=0)      # asked for now -> read again
    assert len(reads) == 2


def test_saving_invalidates_the_cache(nas):
    _save(projects=["CNT"], cohorts={})
    first = wp.load_lab_policy()
    assert _tier(first, CNT01) < _tier(first, ASPA)
    _save(projects=["ASPA"], cohorts={})
    second = wp.load_lab_policy()        # no explicit refresh asked for
    assert _tier(second, ASPA) < _tier(second, CNT01)


# ---------------------------------------------------------------- the lab's ask

def test_the_order_the_lab_asked_for(nas):
    """Pillar first, CNT first, CNT cohorts 01-04 before anything else."""
    _save()
    p = wp.load_lab_policy(max_age_s=0)
    assert _tier(p, CNT01) < _tier(p, CNT05) < _tier(p, ASPA) < _tier(p, OTHER)
    assert p.is_deferred({"video_id": EASY})
    assert not p.is_deferred({"video_id": CNT01})


def test_order_items_sorts_by_tier_and_is_stable_within_one(nas):
    _save()
    items = [OTHER, CNT01B, ASPA, CNT01, CNT05]
    assert wp.order_items(items) == [CNT01B, CNT01, CNT05, ASPA, OTHER]


def test_best_tier_choice_only_ever_returns_the_top_tier(nas):
    _save()
    items = [OTHER, CNT01B, ASPA, CNT01, CNT05]
    picked = {wp.best_tier_choice(items) for _ in range(60)}
    assert picked == {CNT01, CNT01B}     # random INSIDE the tier, never below it


def test_best_tier_choice_on_an_empty_queue(nas):
    assert wp.best_tier_choice([]) is None


def test_paths_adapt_without_a_caller_writing_an_adapter(nas, tmp_path):
    _save()
    bundles = [tmp_path / OTHER, tmp_path / CNT01]
    assert wp.order_items(bundles)[0].name == CNT01
    assert wp.as_item(tmp_path / CNT01) == {"video_id": CNT01}


# ---------------------------------------------------------------- review side

def test_cleared_review_bundles_come_back_in_priority_order(nas, tmp_path):
    """Only a bounded number re-enter the pipeline per scan, so WHICH ones is
    a real decision -- it used to be raw filesystem order."""
    from mousereach.watcher.review_return import _bundles
    _save()
    queue = tmp_path / "triage"
    queue.mkdir()
    for name in (OTHER, ASPA, CNT01, CNT05):
        (queue / name).mkdir()
    (queue / ".hidden").mkdir()          # never a bundle
    got = [b.name for b in _bundles(queue)]
    assert got == [CNT01, CNT05, ASPA, OTHER]


def test_review_return_still_works_with_no_shared_drive(tmp_path, monkeypatch):
    """Ordering must never be the reason a cleared video cannot come back."""
    from mousereach.watcher.review_return import _bundles
    monkeypatch.setattr(Paths, "NAS_ROOT", None)
    queue = tmp_path / "triage"
    queue.mkdir()
    for name in (OTHER, CNT01):
        (queue / name).mkdir()
    assert sorted(b.name for b in _bundles(queue)) == sorted([CNT01, OTHER])


def test_bundles_of_a_missing_queue_is_empty(tmp_path):
    from mousereach.watcher.review_return import _bundles
    assert _bundles(tmp_path / "nope") == []
    assert _bundles(None) == []
