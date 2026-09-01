"""Which work the watcher takes next, and which work waits for an idle node.

Three things are under test, and they fail in different ways:

  1. The RESOLVERS. The tray filter used to read videos.tray_type and nothing
     else. That column is NULL on every video re-registered on its way back
     from a human review queue, which on a node with a review backlog is most
     of the reprocess pool -- so the filter silently stopped filtering and the
     "Pillar first" rule quietly became "pick at random". A filter that
     degrades in silence is worse than no filter, because nobody looks at it
     again.

  2. The POLICY. What an unset key does, what a broken key does, and that no
     project preference is shipped.

  3. The TWO-PASS work loop and its ADMIT/DRAIN split. Easy (E) and Flat (F)
     tray sessions must never be TAKEN ON by a machine that could be running a
     Pillar (P) session -- not "are less preferred", never. But a node must
     always FINISH what is already on its disk, or the deferred videos strand
     and their slots throttle everything else. TestAdmitVersusDrain is the
     part that would have caught both of the failures that shape this design:
     the intake stampede and the stranded local pool.

Naming note: the tests below use PRJA / PRJB as animal-id prefixes and
PROJECT_A / PROJECT_B as selector values. They are placeholders. A real
project code belongs in a lab's own ~/.mousereach/config.json, never here.

None of these tests touch the network, a real database, or a running watcher.
"""

import pytest

from mousereach.config import AnimalID
from mousereach.watcher import work_priority as wp


# =============================================================================
# RESOLVERS -- the DB column is never trusted on its own
# =============================================================================

class TestEffectiveTrayType:

    def test_the_column_is_used_when_it_has_a_value(self):
        assert wp.effective_tray_type(
            {"video_id": "20250704_PRJA0101_E2", "tray_type": "P"}) == "P"

    def test_a_null_column_falls_back_to_the_filename(self):
        """The whole point. A returning review video has no metadata at all."""
        assert wp.effective_tray_type(
            {"video_id": "20250704_PRJA0101_E2", "tray_type": None}) == "E"
        assert wp.effective_tray_type({"video_id": "20250704_PRJA0101_F1"}) == "F"

    def test_an_unparseable_id_is_unknown_not_a_tray(self):
        """Unknown must never be reported as a particular tray."""
        assert wp.effective_tray_type({"video_id": "video1"}) is None

    def test_a_collage_uses_its_suffix_then_its_filename(self):
        assert wp.effective_tray_type(
            {"filename": "20250704_PRJA0101,PRJA0205_E1.mkv", "tray_suffix": "E1"},
            is_collage=True) == "E"
        assert wp.effective_tray_type(
            {"filename": "20250704_PRJA0101,PRJA0205_P1.mkv"}, is_collage=True) == "P"
        assert wp.effective_tray_type({"filename": "collage.mkv"},
                                      is_collage=True) is None

    def test_the_old_substring_check_would_have_been_wrong_here(self):
        """The previous rule was `'_P' in filename`, which is not a tray test.

        It matches any '_P' anywhere in the name -- including one inside an
        animal id or a hand-added suffix -- and for collages it never looked at
        the database at all.
        """
        item = {"filename": "20250704_PRJA0101_Pilot_E1.mkv"}
        assert "_P" in item["filename"]                      # old rule: "Pillar"
        assert wp.effective_tray_type(item, is_collage=True) == "E"

    def test_case_does_not_matter_and_an_odd_value_survives(self):
        assert wp.effective_tray_type({"video_id": "x", "tray_type": "e"}) == "E"
        assert wp.effective_tray_type({"video_id": "x", "tray_type": "X"}) == "X"


class TestProjectAndCohort:

    def test_a_project_is_named_by_its_code_and_by_its_folder(self):
        """Whatever mapping the tool ships, both spellings select the same video.

        Read out of AnimalID.PROJECT_MAP rather than written down here, so this
        test names no project and cannot go stale when the map changes.
        """
        code, folder = next((c, f) for c, f in AnimalID.PROJECT_MAP.items()
                            if len(c) >= 3 and f.upper() != c.upper())
        labels = wp.project_labels({"video_id": "20250704_%s0101_P1" % code})
        assert {code.upper(), folder.upper()} <= labels

    def test_a_short_lettered_id_resolves_without_listing_letters(self):
        """Old lettered cohort ids are selected by their project name."""
        letter, folder = next((c, f) for c, f in AnimalID.PROJECT_MAP.items()
                              if len(c) == 1)
        assert folder.upper() in wp.project_labels({"animal_id": letter + "11"})

    def test_an_unmapped_code_still_names_itself(self):
        assert wp.project_labels({"video_id": "20250704_PRJA0101_P1"}) == {"PRJA"}

    def test_an_unknown_id_claims_no_project(self):
        assert wp.project_labels({"video_id": "video1"}) == set()

    def test_a_collage_carries_every_animal(self):
        item = {"filename": "20250704_PRJA0101,PRJB0205_P1.mkv"}
        assert wp.animal_ids(item) == ["PRJA0101", "PRJB0205"]

    def test_cohort_matches_the_bare_number_and_the_code(self):
        labels = wp.cohort_labels({"video_id": "20250704_PRJA0401_P1"})
        assert {"04", "PRJA04"} <= labels


class TestSelectors:

    def test_every_named_field_must_match(self):
        item = {"video_id": "20250704_PRJA0101_E1"}
        assert wp.selector_matches({"project": ["PRJA"], "tray_type": ["E"]}, item)
        assert not wp.selector_matches({"project": ["PRJA"], "tray_type": ["P"]}, item)

    def test_an_unknown_value_never_matches_a_named_field(self):
        assert not wp.selector_matches({"tray_type": ["P"]}, {"video_id": "video1"})

    def test_an_empty_selector_matches_everything(self):
        assert wp.selector_matches({}, {"video_id": "video1"})


# =============================================================================
# POLICY -- defaults, and what a broken config does
# =============================================================================

class TestPolicy:

    def test_the_shipped_default_is_todays_behaviour_plus_the_deferral(self):
        policy = wp.load_policy(None)
        assert policy.order == [{"tray_type": ["P"]}]
        assert policy.idle_only == [{"tray_type": ["E", "F"]}]
        assert policy.complaints == []

    def test_no_project_preference_is_shipped(self):
        """A project preference is a lab fact and lives only in a lab's config.

        Checked by field rather than by name: the shipped default may only
        speak about tray types, so there is nowhere for a project to hide.
        """
        fields = set()
        for selector in wp.DEFAULT_ORDER + wp.DEFAULT_IDLE_ONLY:
            fields |= set(selector)
        assert fields == {"tray_type"}

    def test_unknown_work_is_last_but_never_unreachable(self):
        policy = wp.load_policy({"order": [{"project": ["PRJA"]},
                                           {"project": ["PRJB"]}]})
        assert policy.tier({"video_id": "20250704_PRJA0101_P1"}) == 0
        assert policy.tier({"video_id": "20250704_PRJB0101_P1"}) == 1
        assert policy.tier({"video_id": "20250704_PRJC0101_P1"}) == 2

    def test_a_broken_key_complains_and_keeps_working(self):
        """A preference must never stop a node from starting.

        Unlike a path, an ordering has a safe substitute: what the watcher did
        before the key existed. So this falls back and says so, loudly, naming
        the field and the file.
        """
        policy = wp.load_policy({"order": [{"traytype": ["P"]}], "idle_only": "E"})
        # Every entry was unusable, so the default is used rather than "no
        # ordering at all" -- which is not what anybody who wrote a list meant.
        assert policy.order == [{"tray_type": ["P"]}]
        assert policy.idle_only == [{"tray_type": ["E", "F"]}]
        assert len(policy.complaints) == 3
        assert "traytype" in policy.complaints[0]
        assert "none of its 1 entries" in policy.complaints[1]
        assert "idle_only" in policy.complaints[2]

    def test_an_unknown_setting_is_named_not_swallowed(self):
        """A removed or misspelled setting must not look like it took effect."""
        policy = wp.load_policy({"idle_only_exempt_stages": []})
        assert any("idle_only_exempt_stages" in c for c in policy.complaints)
        assert policy.order == [{"tray_type": ["P"]}]

    def test_a_bare_string_is_accepted_as_a_one_element_list(self):
        """Forgiving on shape, strict on field names."""
        policy = wp.load_policy({"order": [{"tray_type": "P"}]})
        assert policy.order == [{"tray_type": ["P"]}]
        assert policy.complaints == []

    def test_a_whole_broken_key_falls_back_to_the_default(self):
        policy = wp.load_policy(["not", "an", "object"])
        assert policy.order == [{"tray_type": ["P"]}]
        assert policy.complaints

    def test_an_empty_order_is_honoured_not_replaced(self):
        assert wp.load_policy({"order": []}).order == []

    def test_an_empty_idle_only_defers_nothing(self):
        policy = wp.load_policy({"idle_only": []})
        assert policy.idle_only == []
        assert not policy.is_deferred({"video_id": "20250704_PRJA0101_E1"})

    def test_describe_is_ascii(self):
        """It is printed to a Windows console by --dry-run."""
        for line in wp.load_policy(None).describe():
            line.encode("ascii")


# =============================================================================
# THE WORK LOOP -- two passes, ADMIT gated and DRAIN ungated
# =============================================================================

class FakeDB:
    """Just enough database to drive _select_work_item."""

    def __init__(self, videos_by_state=None, collages_by_state=None):
        self.videos = videos_by_state or {}
        self.collages = collages_by_state or {}

    def get_videos_in_state(self, state):
        return [dict(v) for v in self.videos.get(state, [])]

    def get_collages_in_state(self, state):
        return [dict(c) for c in self.collages.get(state, [])]


def video(video_id, **extra):
    row = {"video_id": video_id, "tray_type": None, "animal_id": None}
    row.update(extra)
    return row


def make_server(videos_by_state, work_priority=None, max_local_pending=200,
                priority_animal=None):
    """A ProcessingOrchestrator with no filesystem and no constructor.

    The real __init__ restores a database from the NAS and runs cross-node
    recovery; none of that is under test here.
    """
    from types import SimpleNamespace
    from mousereach.watcher.orchestrator import ProcessingOrchestrator

    orch = object.__new__(ProcessingOrchestrator)
    orch.config = SimpleNamespace(max_local_pending=max_local_pending,
                                  also_process=False,
                                  work_priority=work_priority)
    orch.db = FakeDB(videos_by_state)
    orch._get_priority_animal = lambda: priority_animal
    return orch


def make_dlc_node(videos_by_state=None, collages_by_state=None,
                  work_priority=None, also_process=False, priority_animal=None):
    """A DLCOrchestrator with no filesystem and no constructor."""
    from types import SimpleNamespace
    from mousereach.watcher.orchestrator import DLCOrchestrator

    orch = object.__new__(DLCOrchestrator)
    orch.config = SimpleNamespace(also_process=also_process,
                                  work_priority=work_priority)
    orch.db = FakeDB(videos_by_state, collages_by_state)
    orch._get_priority_animal = lambda: priority_animal
    orch._archive_backoff_active = lambda video_id: False
    return orch


class TestDeferral:

    def test_preferred_work_beats_deferred_work_in_the_same_bucket(self):
        orch = make_server({"processing": [video("20250704_PRJA0101_E1"),
                                           video("20250704_PRJA0102_P1")]})
        work = orch._get_next_work_item()
        assert work["id"] == "20250704_PRJA0102_P1"

    def test_deferred_work_in_an_EARLIER_bucket_does_not_beat_preferred_work(self):
        """The one thing a per-bucket preference could never do.

        The Easy video is in the intake bucket, which outranks reprocessing.
        Both buckets ADMIT work, so on the first pass the Easy video is
        invisible and the Pillar reprocess wins -- rather than the node taking
        in a session nobody scores per-pellet.
        """
        orch = make_server({
            "dlc_complete": [video("20250704_PRJA0101_E1")],
            "outdated": [video("20250704_PRJA0102_P1",
                               reprocess_scope="segmentation")],
        })
        work = orch._get_next_work_item()
        assert work["type"] == "reprocess"
        assert work["id"] == "20250704_PRJA0102_P1"

    def test_deferred_work_IS_taken_on_when_there_is_nothing_else(self):
        """Deferred, not dropped. E/F sessions are wanted -- last."""
        orch = make_server({"dlc_complete": [video("20250704_PRJA0101_E1")]})
        work = orch._get_next_work_item()
        assert work["type"] == "intake"
        assert work["id"] == "20250704_PRJA0101_E1"

    def test_the_tray_is_read_from_the_filename_when_the_column_is_null(self):
        """Both rows have tray_type NULL, as every returning review video does."""
        orch = make_server({"dlc_complete": [video("20250704_PRJA0101_E1"),
                                             video("20250704_PRJA0102_F1")],
                            "outdated": [video("20250704_PRJA0103_P1")]})
        assert orch._get_next_work_item()["id"] == "20250704_PRJA0103_P1"

    def test_a_video_whose_tray_cannot_be_read_is_not_deferred(self):
        """Unknown is not Easy. Deferring unknown work would hide it forever."""
        orch = make_server({"dlc_complete": [video("mystery")]})
        assert orch._get_next_work_item()["id"] == "mystery"


class TestAdmitVersusDrain:
    """The two failures that shaped this design, one test each.

    Gating every bucket -- the first version of this feature -- fails in two
    ways that only show up when the loop is driven repeatedly against a
    database that changes underneath it. Both are reproduced here.
    """

    def test_admitting_deferred_work_does_not_become_a_stampede(self):
        """FAILURE 1: intake runs BEFORE pipeline, so gating both stampedes.

        On an idle node with nothing but deferred work waiting, a loop that
        gates intake and pipeline together copies a whole cap's worth of
        videos onto local disk before analysing one of them: every cycle, the
        first pass finds nothing, the second pass reaches intake before
        pipeline, and intake wins again.

        With intake gated and the pipeline ungated, exactly one deferred video
        is on the disk at a time: it is taken in, and the next cycle finishes
        it before another is taken.
        """
        cap = 5
        state = {"processed": [], "dlc_complete": [
            video("20250704_PRJA01%02d_E1" % i) for i in range(30)],
            "processing": [], "outdated": []}
        orch = make_server(state, max_local_pending=cap)

        seen = []
        high_water = 0
        for _ in range(12):
            work = orch._get_next_work_item()
            assert work is not None
            seen.append(work["type"])
            row = work["data"]
            if work["type"] == "intake":
                state["dlc_complete"] = [v for v in state["dlc_complete"]
                                         if v["video_id"] != row["video_id"]]
                state["processing"].append(row)
            elif work["type"] == "pipeline":
                state["processing"] = [v for v in state["processing"]
                                       if v["video_id"] != row["video_id"]]
                state["processed"].append(row)
            elif work["type"] == "archive":
                state["processed"] = [v for v in state["processed"]
                                      if v["video_id"] != row["video_id"]]
            high_water = max(high_water, len(state["processing"]))

        assert high_water == 1, (
            "deferred videos piled up on local disk before any was analysed: "
            "%d at once (sequence: %s)" % (high_water, seen))
        assert "pipeline" in seen[:3], seen

    def test_a_local_pool_of_deferred_work_drains_while_other_work_waits(self):
        """FAILURE 2: gating the pipeline bucket strands the local pool.

        Every locally staged video is deferred and the pool is at the cap, so
        intake is blocked. If the pipeline bucket were gated too, the reprocess
        bucket would keep the first pass non-empty for weeks, the second pass
        would never run, the deferred videos would never drain, and no new
        video could ever be taken in.

        This is also the case the old deadlock guard could not cover, because
        intake is not the only door into 'processing': review_return forces a
        human-cleared video straight into that state, so the pool can fill
        without the guard's precondition ever holding.

        Ungated draining fixes it without a guard: the pool empties on the
        very first pass.
        """
        state = {
            "processed": [],
            "dlc_complete": [video("20250704_PRJA0201_P1")],
            "processing": [video("20250704_PRJA0101_E1"),
                           video("20250704_PRJA0102_E1")],
            "outdated": [video("20250704_PRJA0301_P1",
                               reprocess_scope="segmentation")],
        }
        orch = make_server(state, max_local_pending=2)

        # The pool is at the cap, so intake is blocked. The very first thing
        # the node does is drain a slot -- not sit on the reprocess bucket
        # forever, which is what the gated version did.
        first = orch._get_next_work_item()
        assert first["type"] == "pipeline", first
        assert first["id"].endswith("_E1"), first

        # Drive it to completion. Nothing is stranded: both deferred videos
        # are analysed, and the preferred video that was blocked behind the
        # full pool is taken in and analysed too.
        seen = []
        for _ in range(12):
            work = orch._get_next_work_item()
            if work is None:
                break
            seen.append((work["type"], work["id"]))
            row = work["data"]
            for bucket in ("dlc_complete", "processing", "outdated", "processed"):
                state[bucket] = [v for v in state[bucket]
                                 if v["video_id"] != row["video_id"]]
            if work["type"] == "intake":
                state["processing"].append(row)
            elif work["type"] in ("pipeline", "reprocess"):
                state["processed"].append(row)

        analysed = {vid for kind, vid in seen if kind in ("pipeline", "reprocess")}
        assert "20250704_PRJA0102_E1" in analysed, seen   # the other deferred one
        assert "20250704_PRJA0201_P1" in analysed, seen   # was blocked by the cap
        assert "20250704_PRJA0301_P1" in analysed, seen   # the reprocess work
        assert state["processing"] == [], state["processing"]

    def test_a_deferred_video_already_here_is_finished_not_stranded(self):
        """Draining is not "starting new analysis": it is already on the disk."""
        orch = make_server({"processing": [video("20250704_PRJA0101_E1")],
                            "outdated": [video("20250704_PRJA0102_P1",
                                               reprocess_scope="segmentation")]})
        work = orch._get_next_work_item()
        assert work["type"] == "pipeline"
        assert work["id"] == "20250704_PRJA0101_E1"

    def test_archiving_a_deferred_video_is_never_held_back(self):
        """Filing is not analysis; refusing to file strands the result.

        A node with a backlog is never idle, so a deferred archive is a
        permanently stranded result -- off the NAS, out of every downstream
        pull, holding local disk and a row in a working state.
        """
        orch = make_server({"processed": [video("20250704_PRJA0101_E1")],
                            "dlc_complete": [video("20250704_PRJA0102_P1")]})
        work = orch._get_next_work_item()
        assert work["type"] == "archive"
        assert work["id"] == "20250704_PRJA0101_E1"

    def test_but_preferred_work_is_still_filed_first_within_the_bucket(self):
        """Ungated does not mean unordered."""
        orch = make_server({"processed": [video("20250704_PRJA0101_E1"),
                                          video("20250704_PRJA0102_P1")]})
        assert orch._get_next_work_item()["id"] == "20250704_PRJA0102_P1"

    @pytest.mark.parametrize("state,expected", [
        ("processed", "archive"),
        ("processing", "pipeline"),
    ])
    def test_the_drain_buckets_run_on_the_first_pass(self, state, expected):
        orch = make_server({state: [video("20250704_PRJA0101_E1")]})
        work = orch._select_work_item(admit_deferred=False)
        assert work is not None and work["type"] == expected

    @pytest.mark.parametrize("state", ["dlc_complete", "outdated"])
    def test_the_admit_buckets_yield_nothing_on_the_first_pass(self, state):
        """A bucket can be non-empty and still yield nothing.

        If either call site skipped the None test, this would raise TypeError
        on pick['video_id'] -- every cycle, forever, on any node that ever sees
        an Easy video.
        """
        orch = make_server({state: [video("20250704_PRJA0101_E1",
                                          reprocess_scope="segmentation")]})
        assert orch._select_work_item(admit_deferred=False) is None
        assert orch._select_work_item(admit_deferred=True) is not None


class TestDLCNodeAdmitVersusDrain:
    """The same split on the GPU node, where the buckets are different."""

    def test_running_dlc_on_a_deferred_video_waits(self):
        """ADMIT: the GPU is the resource being protected."""
        orch = make_dlc_node({"dlc_queued": [video("20250704_PRJA0101_E1"),
                                             video("20250704_PRJA0102_P1")]})
        assert orch._get_next_work_item()["id"] == "20250704_PRJA0102_P1"

    def test_cropping_a_deferred_collage_waits(self):
        """ADMIT: cropping creates new local files and new DLC work."""
        orch = make_dlc_node(
            collages_by_state={"stable": [
                {"filename": "20250704_PRJA0101_E1.mkv", "animal_ids": "PRJA0101"}]},
            videos_by_state={"dlc_queued": [video("20250704_PRJA0102_P1")]})
        assert orch._get_next_work_item()["type"] == "single_dlc"

    def test_a_deferred_collage_is_cropped_when_nothing_else_waits(self):
        orch = make_dlc_node(collages_by_state={"stable": [
            {"filename": "20250704_PRJA0101_E1.mkv", "animal_ids": "PRJA0101"}]})
        work = orch._get_next_work_item()
        assert work["type"] == "collage"
        assert work["id"] == "20250704_PRJA0101_E1.mkv"

    def test_staging_a_finished_deferred_pose_is_never_held_back(self):
        """DRAIN: the pose exists; holding it hides it from every other node."""
        orch = make_dlc_node({"dlc_complete": [video("20250704_PRJA0101_E1")],
                              "dlc_queued": [video("20250704_PRJA0102_P1")]})
        work = orch._get_next_work_item()
        assert work["type"] == "stage_to_nas"
        assert work["id"] == "20250704_PRJA0101_E1"

    def test_a_local_pipeline_on_a_deferred_video_is_never_held_back(self):
        """DRAIN: also_process node, video already here with its pose."""
        orch = make_dlc_node({"processing": [video("20250704_PRJA0101_E1")],
                              "dlc_queued": [video("20250704_PRJA0102_P1")]},
                             also_process=True)
        work = orch._get_next_work_item()
        assert work["type"] == "local_pipeline"
        assert work["id"] == "20250704_PRJA0101_E1"

    def test_archiving_locally_is_never_held_back(self):
        orch = make_dlc_node({"processed": [video("20250704_PRJA0101_E1")],
                              "dlc_queued": [video("20250704_PRJA0102_P1")]},
                             also_process=True)
        assert orch._get_next_work_item()["type"] == "archive_local"

    def test_an_idle_dlc_node_returns_none(self):
        assert make_dlc_node({})._get_next_work_item() is None


class TestConfiguredOrder:

    ORDER = {"order": [{"project": ["PRJA"]}, {"project": ["PRJB"]}]}

    def test_a_project_preference_is_configuration_only(self):
        orch = make_server({"processing": [video("20250704_PRJB0101_P1"),
                                           video("20250704_PRJA0101_P1")]},
                           work_priority=self.ORDER)
        assert orch._get_next_work_item()["id"] == "20250704_PRJA0101_P1"

    def test_without_that_configuration_neither_project_is_preferred(self):
        """Shipped behaviour: no project outranks another."""
        orch = make_server({"processing": [video("20250704_PRJB0101_P1"),
                                           video("20250704_PRJA0101_P1")]})
        picks = {orch._get_next_work_item()["id"] for _ in range(40)}
        assert picks == {"20250704_PRJB0101_P1", "20250704_PRJA0101_P1"}

    def test_the_project_order_does_not_outrank_the_deferral(self):
        """A preferred project's Easy session still waits for a Pillar session."""
        orch = make_server({"dlc_complete": [video("20250704_PRJA0101_E1"),
                                             video("20250704_PRJB0101_P1")]},
                           work_priority=self.ORDER)
        assert orch._get_next_work_item()["id"] == "20250704_PRJB0101_P1"

    def test_ordering_also_applies_to_the_buckets_that_took_videos_zero(self):
        """Archive and intake used to take videos[0] with no preference at all."""
        orch = make_server({"processed": [video("20250704_PRJB0101_P1"),
                                          video("20250704_PRJA0101_P1")]},
                           work_priority=self.ORDER)
        assert orch._get_next_work_item()["id"] == "20250704_PRJA0101_P1"


class TestPriorityAnimal:

    def test_an_explicit_priority_animal_outranks_the_deferral(self):
        """A person asked for this animal, now.

        Checked in an ADMIT bucket, which is the only place the deferral could
        have blocked it. The alternative is a button that silently does nothing
        for a rehab animal, with no message saying why.
        """
        orch = make_server({"dlc_complete": [video("20250704_PRJA0101_E1",
                                                   animal_id="PRJA0101"),
                                             video("20250704_PRJA0102_P1",
                                                   animal_id="PRJA0102")]},
                           priority_animal="PRJA0101")
        assert orch._get_next_work_item()["id"] == "20250704_PRJA0101_E1"

    def test_the_priority_animals_own_preferred_work_still_comes_first(self):
        orch = make_server({"dlc_complete": [video("20250704_PRJA0101_E1",
                                                   animal_id="PRJA0101"),
                                             video("20250704_PRJA0101_P1",
                                                   animal_id="PRJA0101")]},
                           priority_animal="PRJA0101")
        assert orch._get_next_work_item()["id"] == "20250704_PRJA0101_P1"


class TestDiskCap:

    def test_intake_still_stops_at_the_cap(self):
        orch = make_server({"dlc_complete": [video("20250704_PRJA0101_P1")],
                            "processing": [video("20250704_PRJA0102_P1"),
                                           video("20250704_PRJA0103_P1")]},
                           max_local_pending=2)
        work = orch._get_next_work_item()
        assert work["type"] == "pipeline"      # not intake

    def test_below_the_cap_deferred_work_is_still_not_admitted(self):
        orch = make_server({"dlc_complete": [video("20250704_PRJA0102_E1")],
                            "outdated": [video("20250704_PRJA0104_P1",
                                               reprocess_scope="segmentation")]},
                           max_local_pending=200)
        assert orch._get_next_work_item()["id"] == "20250704_PRJA0104_P1"


class TestNothingToDo:

    def test_an_empty_node_returns_none_from_both_passes(self):
        assert make_server({})._get_next_work_item() is None
