"""Shared guards a GPU node needs before more of them join (locate, collage claims).

WHY these exist:

  * Processing/Posed (Paths.DLC_STAGING) is the processing server's intake.
    A GPU node that resolves a file there copies another node's hand-off into
    its own queue, runs DLC with its output landing in the shared folder, or
    counts the server's copy as its own in-flight work. locate therefore has
    an opt-out (include_staging / search_staging) that drops the folder AND
    refuses a recorded path pointing into it; the re-pose GPU side uses it.
  * Collage claims failed OPEN and never expired: a claim error let a node
    crop anyway, and a claim left by a node whose crop failed or crashed
    blocked that collage for every node forever. try_claim_collage now raises
    on database errors, takes over a claim left in 'cropping' for a day
    (exactly one taker wins), never reopens a finished crop, and the holder
    can release a claim after a failed crop.

Everything is built under tmp_path; the coordinator uses a temp SQLite file.
"""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import mousereach.watcher.repose as repose
from mousereach.config import Paths
from mousereach.watcher import locate

VID = "20250101_CNT0101_P1"
VID2 = "20250102_CNT0102_P1"
COLLAGE = ("20250101_CNT0101,CNT0102,CNT0103,CNT0104,CNT0105,CNT0106,"
           "CNT0107,CNT0108_P1.mkv")


# ----------------------------------------------------------------- fixtures

@pytest.fixture
def node(tmp_path, monkeypatch):
    """Every path locate and repose consult, pointed at a temp tree."""
    layout = {
        "PROCESSING_ROOT": "proc",
        "PROCESSING": "proc/Processing",
        "DLC_QUEUE": "proc/DLC_Queue",
        "NAS_ROOT": "nas",
        "DLC_STAGING": "nas/Processing/Posed",
        "REPOSE_QUEUE": "nas/Processing/Repose_Queue",
        "SINGLE_ANIMAL_OUTPUT": "nas/Unanalyzed/Single_Animal",
        "MULTI_ANIMAL_SOURCE": "nas/Unanalyzed/Multi-Animal",
        "ANALYZED_OUTPUT": "nas/Analyzed",
    }
    made = {}
    for name, sub in layout.items():
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
        made[name] = d
    return made


@pytest.fixture
def coordinator(tmp_path):
    pytest.importorskip("sqlalchemy")
    from mousereach.watcher.coordination import PipelineCoordinator
    c = PipelineCoordinator(db_path=tmp_path / "central" / "watcher_central.db")
    c.ensure_tables()
    return c


def _claim_row(coordinator, filename=COLLAGE):
    return coordinator.get_all_collage_states().get(filename)


def _age_claim(coordinator, hours, filename=COLLAGE):
    """Backdate a claim, written in the same form _now() writes."""
    from sqlalchemy import text
    old = (datetime.now() - timedelta(hours=hours)).isoformat()
    with coordinator.engine.connect() as conn:
        conn.execute(text("UPDATE pipeline_collages SET claimed_at = :t "
                          "WHERE filename = :f"), {"t": old, "f": filename})
        conn.commit()
    return old


# ----------------------------------------------------------------- locate

class TestLocateStagingOptOut:

    def test_default_search_keeps_staging_in_its_place(self, node):
        """Defaults are today's behaviour: the processing server finds its
        intake there, in the same order as before."""
        assert locate.node_search_dirs() == [
            node["PROCESSING"], node["DLC_QUEUE"], node["DLC_STAGING"],
            node["SINGLE_ANIMAL_OUTPUT"]]

    def test_include_staging_false_drops_only_staging(self, node):
        assert locate.node_search_dirs(include_staging=False) == [
            node["PROCESSING"], node["DLC_QUEUE"], node["SINGLE_ANIMAL_OUTPUT"]]

    def test_video_only_in_staging_is_not_found_by_a_gpu_caller(self, node):
        (node["DLC_STAGING"] / f"{VID}.mp4").write_bytes(b"v")
        assert locate.locate_video_file(VID, search_archive=False) is not None
        assert locate.locate_video_file(VID, search_archive=False,
                                        search_staging=False) is None

    def test_raw_path_in_staging_is_ignored_when_staging_is_excluded(self, node):
        staged = node["DLC_STAGING"] / f"{VID}.mp4"
        staged.write_bytes(b"v")
        assert locate.locate_video_file(VID, raw=str(staged),
                                        search_archive=False) == staged
        assert locate.locate_video_file(VID, raw=str(staged), search_archive=False,
                                        search_staging=False) is None

    def test_raw_path_in_staging_falls_through_to_this_nodes_copy(self, node):
        staged = node["DLC_STAGING"] / f"{VID}.mp4"
        staged.write_bytes(b"v")
        mine = node["DLC_QUEUE"] / f"{VID}.mp4"
        mine.write_bytes(b"v")
        assert locate.locate_video_file(VID, raw=str(staged), search_archive=False,
                                        search_staging=False) == mine

    def test_staging_passed_as_an_extra_dir_is_still_excluded(self, node):
        """An extra_dirs entry must not smuggle the folder back in."""
        (node["DLC_STAGING"] / f"{VID}.mp4").write_bytes(b"v")
        assert locate.locate_video_file(
            VID, extra_dirs=[node["DLC_STAGING"]], search_archive=False,
            search_staging=False) is None

    def test_pose_file_in_staging_is_not_found_by_a_gpu_caller(self, node):
        h5 = node["DLC_STAGING"] / f"{VID}DLC_resnet50_TestShuffle1_1000.h5"
        h5.write_bytes(b"pose")
        assert locate.locate_pose_file(VID, search_archive=False) == h5
        assert locate.locate_pose_file(VID, search_archive=False,
                                       search_staging=False) is None
        assert locate.locate_pose_file(VID, raw=str(h5), search_archive=False,
                                       search_staging=False) is None

    def test_raw_pose_outside_staging_still_resolves(self, node):
        h5 = node["DLC_QUEUE"] / f"{VID}DLC_resnet50_TestShuffle1_1000.h5"
        h5.write_bytes(b"pose")
        assert locate.locate_pose_file(VID, raw=str(h5), search_archive=False,
                                       search_staging=False) == h5

    def test_is_in_staging_covers_subfolders_and_nothing_else(self, node):
        assert locate.is_in_staging(node["DLC_STAGING"] / f"{VID}.mp4")
        assert locate.is_in_staging(node["DLC_STAGING"] / ".claims" / f"{VID}.claimed")
        assert not locate.is_in_staging(node["DLC_QUEUE"] / f"{VID}.mp4")
        assert not locate.is_in_staging(node["NAS_ROOT"] / "Processing" / f"{VID}.mp4")
        assert not locate.is_in_staging(None)
        assert not locate.is_in_staging("")

    def test_no_staging_configured_changes_nothing(self, node, monkeypatch):
        monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
        mine = node["DLC_QUEUE"] / f"{VID}.mp4"
        mine.write_bytes(b"v")
        assert not locate.is_in_staging(mine)
        assert locate.locate_video_file(VID, raw=str(mine), search_archive=False,
                                        search_staging=False) == mine


# ----------------------------------------------------------------- repose, GPU side

class TestReposeIgnoresPosed:

    def test_local_file_exists_does_not_count_the_servers_posed_copy(self, node):
        staged = node["DLC_STAGING"] / f"{VID}.mp4"
        staged.write_bytes(b"v")
        row = {"video_id": VID, "current_path": str(staged)}
        assert repose._local_file_exists(VID, row, node["DLC_QUEUE"]) is False

    def test_local_file_exists_still_sees_this_nodes_queue_copy(self, node):
        (node["DLC_STAGING"] / f"{VID}.mp4").write_bytes(b"v")
        (node["DLC_QUEUE"] / f"{VID}.mp4").write_bytes(b"v")
        row = {"video_id": VID, "current_path": None}
        assert repose._local_file_exists(VID, row, node["DLC_QUEUE"]) is True

    def test_request_naming_a_posed_mp4_is_not_served_from_posed(
            self, node, tmp_path, monkeypatch):
        empty_archive = tmp_path / "archive_empty"
        empty_archive.mkdir()
        monkeypatch.setattr(repose, "archive_folder", lambda vid: empty_archive)
        staged = node["DLC_STAGING"] / f"{VID}.mp4"
        staged.write_bytes(b"v")
        body = {"video_id": VID, "video_rel": f"Processing/Posed/{VID}.mp4",
                "video_path": str(staged)}
        assert repose.resolve_request_video(VID, body) is None

    def test_request_is_served_from_the_archive_when_it_has_the_mp4(
            self, node, tmp_path, monkeypatch):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / f"{VID}.mp4").write_bytes(b"v")
        monkeypatch.setattr(repose, "archive_folder", lambda vid: archive)
        (node["DLC_STAGING"] / f"{VID}.mp4").write_bytes(b"v")
        body = {"video_id": VID, "video_rel": f"Processing/Posed/{VID}.mp4"}
        assert repose.resolve_request_video(VID, body) == archive / f"{VID}.mp4"

    def test_publisher_does_not_name_a_posed_mp4_as_the_archived_video(
            self, node, tmp_path, monkeypatch):
        empty_archive = tmp_path / "archive_empty"
        empty_archive.mkdir()
        monkeypatch.setattr(repose, "archive_folder", lambda vid: empty_archive)
        (node["DLC_STAGING"] / f"{VID2}.mp4").write_bytes(b"v")
        assert repose.archived_video(VID2) is None


# ----------------------------------------------------------------- collage claims

class TestCollageClaims:

    def test_first_writer_wins(self, coordinator):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        row = _claim_row(coordinator)
        assert row["hostname"] == "NODE-A" and row["state"] == "cropping"

    def test_second_host_is_refused_and_the_holder_keeps_it(self, coordinator):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is False
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        assert _claim_row(coordinator)["hostname"] == "NODE-A"

    def test_stale_cropping_claim_is_taken_over_exactly_once(self, coordinator):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        _age_claim(coordinator, hours=25)
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is True
        assert coordinator.try_claim_collage(COLLAGE, "NODE-C") is False
        row = _claim_row(coordinator)
        assert row["hostname"] == "NODE-B" and row["state"] == "cropping"
        # the takeover restarts the clock, so the new holder gets its full day
        assert datetime.fromisoformat(row["claimed_at"]) > datetime.now() - timedelta(hours=1)

    def test_two_hosts_that_both_saw_the_stale_claim_cannot_both_win(self, coordinator):
        """Both racers read the stale row before either wrote: the conditional
        update names the old holder, so only the first one matches."""
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        _age_claim(coordinator, hours=25)
        first = coordinator._take_over_stale_claim(COLLAGE, "NODE-A", "NODE-B")
        second = coordinator._take_over_stale_claim(COLLAGE, "NODE-A", "NODE-C")
        assert (first, second) == (True, False)
        assert _claim_row(coordinator)["hostname"] == "NODE-B"

    def test_fresh_cropping_claim_is_not_taken(self, coordinator):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        _age_claim(coordinator, hours=23)
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is False
        assert coordinator._take_over_stale_claim(COLLAGE, "NODE-A", "NODE-B") is False
        assert _claim_row(coordinator)["hostname"] == "NODE-A"

    @pytest.mark.parametrize("done_state", ["cropped", "archived"])
    def test_finished_claim_is_never_taken_however_old(self, coordinator, done_state):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        coordinator.update_collage_state(COLLAGE, done_state)
        _age_claim(coordinator, hours=24 * 30)
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is False
        assert coordinator._take_over_stale_claim(COLLAGE, "NODE-A", "NODE-B") is False
        row = _claim_row(coordinator)
        assert row["hostname"] == "NODE-A" and row["state"] == done_state

    def test_unreadable_claimed_at_is_not_treated_as_stale(self, coordinator):
        from sqlalchemy import text
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        with coordinator.engine.connect() as conn:
            conn.execute(text("UPDATE pipeline_collages SET claimed_at = 'unknown'"))
            conn.commit()
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is False

    def test_release_only_by_the_holder(self, coordinator):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        assert coordinator.release_collage_claim(COLLAGE, "NODE-B") is False
        assert _claim_row(coordinator)["hostname"] == "NODE-A"
        assert coordinator.release_collage_claim(COLLAGE, "NODE-A") is True
        assert _claim_row(coordinator) is None
        # released: any node may claim it again
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is True

    @pytest.mark.parametrize("done_state", ["cropped", "archived"])
    def test_release_refuses_a_finished_crop(self, coordinator, done_state):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        coordinator.update_collage_state(COLLAGE, done_state)
        assert coordinator.release_collage_claim(COLLAGE, "NODE-A") is False
        assert _claim_row(coordinator)["state"] == done_state

    def test_release_of_an_unclaimed_collage_is_false(self, coordinator):
        assert coordinator.release_collage_claim(COLLAGE, "NODE-A") is False


class TestTakeoverNeverRecropsWorkThatExists:
    """Age alone is not enough to take a claim over."""

    def test_holder_reusing_its_old_claim_restarts_the_clock(self, coordinator):
        """Otherwise a holder back after a day crops behind a claim every other
        host reads as abandoned, and a second host crops it too."""
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        _age_claim(coordinator, hours=30)
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        row = _claim_row(coordinator)
        assert datetime.fromisoformat(row["claimed_at"]) > datetime.now() - timedelta(hours=1)
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is False

    @pytest.mark.parametrize("how", ["collage_id", "child_name"])
    def test_stale_claim_with_children_on_record_is_not_taken_over(
            self, coordinator, how, caplog):
        import logging
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        _age_claim(coordinator, hours=25)
        if how == "collage_id":
            coordinator.sync_video_state("20250101_XYZ0101_P1", "NODE-A", "dlc_queued",
                                         collage_id=COLLAGE)
        else:                          # a child named by the collage, synced bare
            coordinator.sync_video_state(VID, "NODE-A", "archived")

        with caplog.at_level(logging.WARNING, logger="mousereach.watcher.coordination"):
            assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is False

        assert _claim_row(coordinator)["hostname"] == "NODE-A"
        assert any("NOT taken over" in r.getMessage() for r in caplog.records)

    def test_children_on_record_of_an_unrelated_collage_do_not_count(self, coordinator):
        coordinator.sync_video_state(VID2, "NODE-A", "dlc_queued", collage_id="other.mkv")
        assert coordinator.collage_children_on_record(COLLAGE) == []
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        _age_claim(coordinator, hours=25)
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is True

    def test_a_takeover_that_finds_the_row_gone_claims_it_instead_of_giving_up(
            self, coordinator, monkeypatch):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        _age_claim(coordinator, hours=25)

        def released_meanwhile(filename, old, new, cutoff=None):
            coordinator.release_collage_claim(filename, old)
            return False

        monkeypatch.setattr(coordinator, "_take_over_stale_claim", released_meanwhile)
        assert coordinator.try_claim_collage(COLLAGE, "NODE-B") is True
        assert _claim_row(coordinator)["hostname"] == "NODE-B"

    def test_cropped_is_written_only_while_the_claim_is_still_this_hosts(self, coordinator):
        assert coordinator.try_claim_collage(COLLAGE, "NODE-A") is True
        assert coordinator.update_collage_state(COLLAGE, "cropped",
                                                only_if_held_by="NODE-B") is False
        assert _claim_row(coordinator)["state"] == "cropping"
        assert coordinator.update_collage_state(COLLAGE, "cropped", only_if_held_by="NODE-A",
                                                singles_created=2) is True
        row = coordinator.get_collage_claim(COLLAGE)
        assert row["state"] == "cropped" and row["singles_created"] == 2
        assert coordinator.get_collage_claim("20250101_NONE_P1.mkv") is None


class TestRecoveryIgnoresPosed:

    def test_a_video_whose_only_copy_is_in_posed_is_not_adopted_as_in_flight(
            self, node, coordinator, tmp_path, monkeypatch):
        from mousereach.watcher.db import WatcherDB
        staged = node["DLC_STAGING"] / f"{VID}.mp4"
        staged.write_bytes(b"v")
        coordinator.sync_video_state(VID, "NODE-B", "dlc_complete", source_path=str(staged))
        monkeypatch.setattr(coordinator, "_get_mousedb_video_names", lambda: set())
        local = WatcherDB(db_path=tmp_path / "local.db")

        coordinator.recover_local_db(local, "NODE-A")

        row = local.get_video(VID)
        assert row["state"] == "unresolvable"
        assert "NODE-B" in (row["error_message"] or "")

    def test_publish_with_staging_off_does_not_narrow_on_a_pose_in_posed(
            self, node, tmp_path, monkeypatch):
        from mousereach.watcher.db import WatcherDB
        declared = "DLC_resnet101_TestShuffle3_1000"
        empty_archive = tmp_path / "archive_empty"
        empty_archive.mkdir()
        monkeypatch.setattr(repose, "archive_folder", lambda vid: empty_archive)
        (node["DLC_STAGING"] / f"{VID}{declared}.h5").write_bytes(b"pose")
        db = WatcherDB(db_path=tmp_path / "w.db")
        db.register_video(video_id=VID, source_path="x", current_path="x")
        db.force_state(VID, "outdated", reason="test setup", reprocess_scope="full")

        off = repose.publish_pending(db, [db.get_video(VID)], hostname="NODE-A",
                                     staging_dir=False, repose_dir=node["REPOSE_QUEUE"],
                                     declared=declared)
        assert VID not in off["narrowed"] and VID not in off["staged"]

        on = repose.publish_pending(db, [db.get_video(VID)], hostname="NODE-A",
                                    repose_dir=node["REPOSE_QUEUE"], declared=declared)
        assert VID in on["staged"]              # the default still reads staging


class TestCoordinatorErrorsRaise:
    """An error is never turned into a claim decision."""

    def test_ensure_tables_raises_on_a_file_that_is_not_a_database(self, tmp_path):
        pytest.importorskip("sqlalchemy")
        from sqlalchemy.exc import DBAPIError
        from mousereach.watcher.coordination import PipelineCoordinator
        bad = tmp_path / "watcher_central.db"
        bad.write_bytes(b"this is not a sqlite database file" * 200)
        c = PipelineCoordinator(db_path=bad)
        with pytest.raises(DBAPIError):
            c.ensure_tables()
        assert c._tables_ensured is False
        with pytest.raises(DBAPIError):
            c.try_claim_collage(COLLAGE, "NODE-A")

    def test_claim_raises_when_the_table_is_broken(self, coordinator):
        from sqlalchemy import text
        from sqlalchemy.exc import DBAPIError
        with coordinator.engine.connect() as conn:
            conn.execute(text("DROP TABLE pipeline_collages"))
            conn.commit()
        with pytest.raises(DBAPIError):
            coordinator.try_claim_collage(COLLAGE, "NODE-A")
        with pytest.raises(DBAPIError):
            coordinator.release_collage_claim(COLLAGE, "NODE-A")

    def test_no_shared_root_raises(self, monkeypatch):
        pytest.importorskip("sqlalchemy")
        from mousereach.watcher import coordination
        monkeypatch.setattr(coordination, "_coordination_db_path", lambda: None)
        c = coordination.PipelineCoordinator()
        with pytest.raises(FileNotFoundError):
            c.try_claim_collage(COLLAGE, "NODE-A")
