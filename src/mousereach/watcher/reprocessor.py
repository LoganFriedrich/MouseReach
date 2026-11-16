"""
mousereach.watcher.reprocessor - Scan archived videos for outdated tool versions.

The ReprocessingScanner compares each archived video's _processing_manifest.json
against the current pipeline_versions.json to detect videos processed with
outdated tools. Outdated videos are marked in the watcher DB and re-enter
the processing pipeline automatically.

Usage:
    # Integrated into ProcessingOrchestrator poll loop (automatic)
    # Or run standalone:
    mousereach-version-check          Show version status of archived videos
    mousereach-version-check --mark   Also mark outdated videos for reprocessing
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# The ground-truth file's name ending (pipeline.manifest.GROUND_TRUTH_SUFFIX).
GROUND_TRUTH_SUFFIX = "_unified_ground_truth.json"

# Dependency-aware reprocessing: map each version-tracked component to its pipeline
# stage, in order. Given the stale components we re-run from the EARLIEST stale
# post-DLC stage and reuse the still-current upstream outputs.
#
# Every stage in pipeline.manifest.TRACKED_STAGES needs a line here (a test
# checks it): a component missing from this map is ignored, and a video whose
# only stale or unrecorded component is ignored is re-run from segmentation.
# Reach assignment has no start stage of its own -- the pipeline recomputes it
# whenever it recomputes outcomes -- so it starts at 'outcome'.
_STAGE_FOR_COMPONENT = {
    "segmenter": "segmentation",
    "reach_detector": "reach",
    "outcome_detector": "outcome",
    "assignment": "outcome",
    "kinematic_extractor": "kinematics",
}
_POST_DLC_ORDER = ["segmentation", "reach", "outcome", "kinematics"]


def earliest_stale_stage(stale_components) -> str:
    """The earliest post-DLC stage to re-run given the stale components (that
    stage + everything downstream re-runs; upstream is reused). Falls back to
    'segmentation' when the stale set is unknown."""
    stages = [_STAGE_FOR_COMPONENT[c] for c in stale_components if c in _STAGE_FOR_COMPONENT]
    if not stages:
        return "segmentation"
    return min(stages, key=_POST_DLC_ORDER.index)



def failed_rerun_stamp(current: dict, review_entry=None, gt_entry=None) -> str:
    """What a filed video's failed re-run was attempted against, as one short
    fingerprint: the declared versions (pipeline_versions.json 'versions' and
    'compatible_versions'), the saved review the scan would apply (path,
    whole-second modified time, size) and the ground truth filed beside it
    (name, whole-second modified time).

    WHY: a filed video whose re-run failed for good goes back to 'archived'
    (orchestrator.ProcessingOrchestrator._restore_filed_after_failed_rerun)
    with its out-of-date manifest. The version scan would mark it 'outdated'
    again on its very next pass, the re-run would fail the same way, and so on
    for ever -- each round costing a run and making the video drop off and
    reappear in the list a person reads. The row keeps this fingerprint
    (pipeline_versions_hash) from the moment of the failure; the scan offers
    the re-run again only once it differs: the declaration changed, or a
    review or ground truth was saved since. A person can always press Re-run.
    Compared for equality only, never ordered, so no two clocks are
    compared."""
    import hashlib
    current = current or {}

    def _entry(e, n):
        if not e:
            return None
        try:
            return [Path(e[0]).name, int(float(e[1]))] + ([e[2]] if n == 3 else [])
        except Exception:
            return [str(e)]
    payload = {
        "versions": current.get("versions") or {},
        "compatible_versions": current.get("compatible_versions") or {},
        "review": _entry(review_entry, 3),
        "ground_truth": _entry(gt_entry, 2),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()[:16]


def failed_rerun_stamp_for(video_id: str, nas_root=None) -> Optional[str]:
    """failed_rerun_stamp for one video, looked up the way the version scan
    sees it: the declaration on the share, the review from
    causal_review_io.review_path_index, the ground truth in the video's filed
    folder. None when it cannot be worked out (the scan then records it on
    its next pass). Never raises."""
    try:
        from mousereach.config import Paths
        from mousereach.pipeline.versions import get_current_versions
        current = get_current_versions(nas_root or Paths.NAS_ROOT)
        try:
            from mousereach.review.causal_review_io import review_path_index
            review = review_path_index([video_id]).get(video_id)
        except Exception:
            review = None
        gt = None
        try:
            from mousereach.archive.core import get_archive_destination
            p = Path(get_archive_destination(video_id)) / f"{video_id}{GROUND_TRUTH_SUFFIX}"
            if p.is_file():
                gt = (p, p.stat().st_mtime)
        except Exception:
            gt = None
        return failed_rerun_stamp(current, review, gt)
    except Exception as e:
        logger.warning("%s: could not fingerprint the failed re-run (%s)", video_id, e)
        return None


def pose_scorers_in_archive(archive_dir) -> Dict[str, set]:
    """{video_id: set of DLC scorers whose pose file is in the archive}.

    One walk of the archive tree, not one per video. Pose files do NOT live
    beside a video's results -- results are in Analyzed/{project}/{cohort}/ and
    pose is in Analyzed/{project}/DLC Model N/{cohort}/ -- so there is no cheap
    per-video path to check, and thousands of separate globs over the NAS is not a scan
    anyone would wait for.

    The walk never enters Analyzed/Archive/: a superseded pose there keeps its
    original name, and counting it would call the declared pose "already on
    disk" and schedule a re-run from segmentation on a pose that is not live
    (no GPU re-pose would ever be queued). DLC Model <N>/ folders ARE walked --
    they are where live pose is stored.
    """
    from mousereach.pipeline.analyzed_tree import iter_files
    from mousereach.pipeline.manifest import extract_dlc_model_info

    index: Dict[str, set] = {}
    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        return index
    for h5 in iter_files(archive_dir, "*DLC*.h5"):
        video_id = h5.name.split("DLC")[0].rstrip("_")
        scorer = extract_dlc_model_info(h5).get('dlc_scorer', '')
        if scorer:
            index.setdefault(video_id, set()).add(scorer)
    return index


def local_cache_path() -> Optional[Path]:
    """Where a watcher keeps its version-scan cache: this machine's own
    processing root (local disk, never the share -- each machine's scanner
    is its own). None when no processing root is configured."""
    from mousereach.config import Paths
    root = getattr(Paths, "PROCESSING_ROOT", None)
    return Path(root) / "version_scan_cache.json" if root else None


class ReprocessingScanner:
    """Scan archived videos for outdated tool versions."""

    def __init__(self, db, nas_root: Path, cache_path: Optional[Path] = None):
        r"""
        Args:
            db: WatcherDB instance
            nas_root: NAS root path (e.g. <NAS>\MouseReach_Pipeline)
            cache_path: Optional local file that keeps the caches below across
                restarts (see _load_caches). None keeps them in memory only.
        """
        self.db = db
        self.nas_root = Path(nas_root)
        self.archive_dir = self.nas_root / "Analyzed"
        # Manifest cache keyed by mtime: a steady-state scan re-read every
        # unchanged manifest over the NAS every pass (over half an hour) to report "0 outdated" every time. A stat is far cheaper
        # than a read; only changed manifests are re-parsed. The cache only
        # pays off on a PERSISTENT scanner instance -- both orchestrator call
        # sites hold one.
        self._manifest_cache: dict = {}
        # Review fingerprints keyed by (path, size, modified time): the
        # pending-review check compares a review's sha256 with the one the
        # archived kinematics applied, and re-hashing every review every scan
        # cost minutes a pass. Only a changed file is re-read.
        self._review_sha_cache: dict = {}
        # The in-file reviewed_at of reviews applied before the applied_review
        # stamp existed, keyed the same way. WHY: for those videos the check
        # opened the review on every pass to read one field -- about half of
        # a warm pass.
        self._reviewed_at_cache: dict = {}
        # Rebuilt at the start of every scan (see _deep_manifest_index): the
        # tree changes between passes, so this one is never persisted.
        self._deep_manifests: Optional[dict] = None
        self._cache_path = Path(cache_path) if cache_path else None
        self._load_caches()

    # The three caches above, kept on local disk between watcher runs. WHY: the
    # first scan after a start re-read every manifest and review over the share
    # -- many minutes, during which the work loop waits -- although
    # nearly all of them were unchanged since the last run. Every entry is
    # still checked against the file's current listing (modified time, and
    # path + size for reviews) before use, so a stale entry is never trusted;
    # a missing or unreadable cache file only means a cold first scan.
    _CACHE_FORMAT = 1

    def _load_caches(self) -> None:
        p = getattr(self, "_cache_path", None)
        if p is None or not p.is_file():
            return
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if doc.get("format") != self._CACHE_FORMAT:
                return
            self._manifest_cache = {v: (m, man) for v, m, man in doc["manifests"]}
            self._review_sha_cache = {(path, size, m): d
                                      for path, size, m, d in doc["review_sha"]}
            self._reviewed_at_cache = {(path, size, m): r
                                       for path, size, m, r in doc["reviewed_at"]}
            logger.info("Version scan: %d manifests and %d review entries kept "
                        "from the last run (%s)", len(self._manifest_cache),
                        len(self._review_sha_cache) + len(self._reviewed_at_cache), p)
        except Exception as e:
            self._manifest_cache, self._review_sha_cache, self._reviewed_at_cache = {}, {}, {}
            logger.info("Version scan: cache %s not used (%s); first scan reads "
                        "everything", p, e)

    def _save_caches(self, manifest_index: dict, review_index) -> None:
        """Write the caches, keeping only entries for files this scan saw (so
        the file does not grow with videos that left the archive). Atomic
        replace; never raises."""
        p = getattr(self, "_cache_path", None)
        if p is None:
            return
        try:
            live_reviews = None
            if review_index is not None:
                live_reviews = {(str(path), size, m)
                                for path, m, size in review_index.values()}

            def keep(key):
                return live_reviews is None or key in live_reviews

            doc = {
                "format": self._CACHE_FORMAT,
                "manifests": [[v, m, man] for v, (m, man) in self._manifest_cache.items()
                              if not manifest_index or v in manifest_index],
                "review_sha": [[*k, d] for k, d in self._review_sha_cache.items() if keep(k)],
                "reviewed_at": [[*k, r] for k, r in self._reviewed_at_cache.items() if keep(k)],
            }
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(doc), encoding="utf-8")
            os.replace(tmp, p)
        except Exception as e:
            logger.warning("Version scan: could not save cache %s: %s", p, e)

    def scan(self, mark_outdated: bool = True, adopt_orphans: bool = True,
             full_only: bool = False) -> dict:
        """Scan all archived videos, optionally mark outdated ones.

        Args:
            mark_outdated: If True, update DB state to 'outdated' for stale videos
            adopt_orphans: If True, register disk-archived videos that have no
                row on this node (see the reconcile block). A GPU node that
                also processes runs this scan for what IT archived; with three
                such nodes each adopting the whole archive, every video would
                be scanned and marked three times over, so they pass False.
            full_only: If True, mark only videos whose scope would be 'full'
                (they need a NEW pose). A node with no handler for narrower
                scopes must not mark them: an 'outdated' row nothing drains
                is not harmless bookkeeping (it syncs, and it hides the
                video from every other node's scan). Such videos are counted
                under 'outdated_partial_unmarked' instead.

        Returns:
            Summary dict with counts:
                scanned: total archived videos checked
                current: videos with up-to-date versions
                outdated: videos with stale versions
                crystallized_skipped: crystallized videos skipped
                no_manifest: videos with no manifest found
                errors: scan errors
        """
        from mousereach.pipeline.versions import (
            get_current_versions, compare_manifest_to_current
        )
        from mousereach.config import is_supported_tray_type

        summary = {
            'scanned': 0,
            'current': 0,
            'outdated': 0,
            'outdated_full': 0,    # Need a genuine re-pose (no current pose on disk)
            'outdated_partial': 0, # Only need seg/reach/outcomes rerun
            'pose_already_current': 0,  # manifest says an old model, but the
                                        # declared model's pose is already on
                                        # disk -- downstream re-run, no GPU
            'unsupported_tray': 0,      # Easy/Flat videos: not re-run for versions
                                        # (re-run once if they have no session file)
            'crystallized_skipped': 0,
            'no_manifest': 0,
            'review_triggered': 0,  # version-current but a newer human review to apply
            'no_manifest_review': 0,  # no manifest, but a saved review its
                                      # kinematics have not applied: re-run them
            'review_mislabel': 0,   # pending review declares a segment mislabel -> deep review
            'review_mislabel_videos': [],
            'errors': 0,
            'outdated_videos': [],
            # Filed videos whose re-run failed for good and that nothing has
            # changed for since (failed_rerun_stamp): not offered again.
            'failed_rerun_waiting': 0,
            'failed_rerun_waiting_videos': [],
            'unsupported_tray_videos': [],
        }

        self._deep_manifests = None     # one walk per scan, not per video
        # Load current versions
        current = get_current_versions(self.nas_root)
        # Say so every scan while the declaration disagrees with the code:
        # silent drift here has twice made whole version bumps invisible to
        # staleness scanning (segmenter 2.2, kinematic_extractor 2.1).
        try:
            from mousereach.pipeline.versions import declaration_drift
            for _stage, _decl, _code in declaration_drift(current):
                logger.warning("declaration drift: %s declared %s but the code stamps %s "
                               "-- staleness scanning is blind to this stage", _stage, _decl, _code)
        except Exception:
            pass
        if not current or not current.get('versions'):
            logger.warning("No pipeline_versions.json found or empty -- cannot scan")
            return summary

        # ONE enumeration for the whole scan instead of a per-video directory
        # walk. WHY: _load_manifest globs the tree per video, and the pending-
        # review check rglobbed the WHOLE archive per video; with a few
        # thousand rows a single scan issued thousands of directory walks over
        # the NAS and ran for upwards of half an hour -- and the orchestrator
        # processes nothing while a scan runs, so the stall was total.
        manifest_index: dict = {}
        manifest_mtime: dict = {}
        feats_mtime: dict = {}
        archive_listed = False
        try:
            manifest_index, manifest_mtime, feats_mtime = self._index_archive()
            archive_listed = True
        except Exception as e:
            manifest_index, manifest_mtime, feats_mtime = {}, {}, {}
            # A listing that stopped part-way may have seen a video's ground
            # truth but not its features file; read as "ground truth newer
            # than kinematics", that would mark a re-run it does not owe.
            # This pass looks at no filed ground truth at all.
            self._filed_gt = {}
            logger.warning(
                f"Archive index build failed ({e}); falling back to per-video walks")

        # RECONCILE DISK AGAINST DB before scanning: the archive on disk is
        # the ground truth for what exists; this db is one node's ledger.
        # Videos archived by ANOTHER node (also_process GPU machines) have no
        # row here, so no version scan anywhere ever covered them -- hundreds
        # accumulated exactly this way before being adopted by hand. Adopt them automatically so the
        # blind spot cannot return on the next version bump. Bounded per scan;
        # the remainder is adopted on subsequent passes, and freshly adopted
        # rows are version-checked on the NEXT pass (this pass's archived
        # list was already fetched).
        summary['adopted'] = 0
        summary['adopt_no_mp4'] = []
        try:
            conn = self.db._get_connection()
            try:
                with_rows = {r[0] for r in conn.execute("SELECT video_id FROM videos")}
            finally:
                conn.close()
        except Exception as e:
            with_rows = None
            logger.warning(f"Disk-vs-db reconcile skipped (could not list rows): {e}")
        if adopt_orphans and with_rows is not None and manifest_index:
            orphans = sorted(s for s in manifest_index if s not in with_rows)
            for stem in orphans[:200]:
                mp4 = manifest_index[stem].parent / f"{stem}.mp4"
                if not mp4.exists():
                    # Manifest with no video beside it: named, never silent.
                    summary['adopt_no_mp4'].append(stem)
                    continue
                try:
                    self.db.register_video(video_id=stem, source_path=str(mp4),
                                           current_path=str(mp4))
                    self.db.force_state(
                        stem, 'archived',
                        reason='reconcile: archived on disk with no row on this '
                               'node (archived by another machine); adopted so '
                               'version compliance covers it')
                    summary['adopted'] += 1
                except Exception as e:
                    summary['errors'] += 1
                    logger.error(f"Reconcile adopt failed for {stem}: {e}")
            if summary['adopted'] or summary['adopt_no_mp4']:
                logger.info(
                    f"Reconcile: adopted {summary['adopted']} disk-archived "
                    f"video(s) with no local row; "
                    f"{len(summary['adopt_no_mp4'])} manifest-without-mp4 stem(s)")

        # Get all archived videos from DB
        archived = self.db.get_videos_in_state('archived')
        logger.info(f"Scanning {len(archived)} archived videos for version compliance")

        # Every archived and outdated video's saved review, found once from a
        # few listings (causal_review_io.review_path_index), not looked up
        # per video. None -> the per-video lookup, as before.
        try:
            from mousereach.review.causal_review_io import review_path_index
            review_index = review_path_index(
                [v['video_id'] for v in archived]
                + [v['video_id'] for v in self.db.get_videos_in_state('outdated')])
        except Exception as e:
            review_index = None
            logger.warning(f"Review index build failed ({e}); looking reviews up per video")

        # Built on first need only: a scan where nothing is DLC-stale never walks
        # the archive tree at all.
        pose_index = None

        from mousereach.watcher.db import FAILED_RERUN_PREFIX
        for video in archived:
            video_id = video['video_id']
            summary['scanned'] += 1

            # A filed video whose re-run failed for good is not offered again
            # until something it was attempted against has changed
            # (failed_rerun_stamp says why: otherwise it loops for ever).
            if (video.get('error_message') or '').startswith(FAILED_RERUN_PREFIX):
                # Without this pass's review index or archive listing the
                # fingerprint cannot be compared fairly: wait.
                stamp_now = (failed_rerun_stamp(
                    current, review_index.get(video_id),
                    getattr(self, '_filed_gt', {}).get(video_id))
                    if (review_index is not None and archive_listed) else None)
                kept = video.get('pipeline_versions_hash')
                if not kept and stamp_now and mark_outdated:
                    # The failure could not record its fingerprint: record it
                    # now and wait, rather than re-run the same failure.
                    try:
                        self.db.set_fields(video_id, pipeline_versions_hash=stamp_now)
                    except Exception as e:
                        logger.warning(f"{video_id}: could not record the failed "
                                       f"re-run's fingerprint ({e})")
                    kept = stamp_now
                if stamp_now is None or not kept or kept == stamp_now:
                    summary['failed_rerun_waiting'] += 1
                    summary['failed_rerun_waiting_videos'].append(video_id)
                    continue
                logger.info(f"{video_id}: its re-run failed before, but the declared "
                            f"versions, its review or its ground truth have changed "
                            f"since -- checked again")

            # E (Easy) and F (Flat) trays are not re-run for version changes.
            # They are processed (last, never skipped), but
            # what they need is only the session's reaches and attention: they
            # are not segmented, and neither pellet outcomes nor kinematics are
            # made for them (config.is_outcome_free_tray). A
            # newer segmenter, outcome detector or extractor gives them nothing
            # they use, and re-posing them would spend GPU time on it. They are counted and named, not silently dropped.
            #
            # ONE exception: a filed one with no features file has never had
            # its session file (FeatureExtractor.extract_session_only) -- every
            # Easy/Flat video filed before 2026-10-23. It is re-run once from
            # segmentation, its pose kept: whole-video span, reaches, session
            # file. The work order runs Easy/Flat only when nothing else waits.
            if not is_supported_tray_type(f"{video_id}.mp4"):
                summary['unsupported_tray'] += 1
                summary['unsupported_tray_videos'].append(video_id)
                if manifest_index and video_id not in feats_mtime:
                    summary['outdated'] += 1
                    summary['outdated_partial'] += 1
                    summary['outdated_videos'].append({
                        'video_id': video_id, 'scope': 'segmentation',
                        'stale_components': ['outcome_free_session_file']})
                    if mark_outdated and not full_only:
                        self.db.force_state(video_id, 'outdated',
                                            reprocess_scope='segmentation',
                                            mark_reason=None)
                        logger.info(f"Marked {video_id} as outdated (scope=segmentation, "
                                    f"Easy/Flat with no session file)")
                continue

            try:
                # Load manifest
                manifest = self._load_manifest_indexed(video_id, manifest_index,
                                                       manifest_mtime)
                if not manifest:
                    # No manifest: no versions to compare. A saved human review
                    # its kinematics have not applied is still owed, though --
                    # the check below works without a manifest (it falls back
                    # to the features file's time, or to "no kinematics yet").
                    # WHY: this used to `continue` before the review check, so
                    # a review saved for such a video never reached its filed
                    # kinematics. The kinematics-only re-run writes a manifest,
                    # so the next scan compares its versions as for any video.
                    #
                    # Only when this pass listed the archive. WHY: if the
                    # listing failed (a share hiccup), "no manifest and no
                    # features file" is what EVERY filed video looks like, and
                    # each one with a saved review or ground truth would be
                    # marked for a re-run it does not owe. Unsure -> no change.
                    if not archive_listed:
                        summary['no_manifest'] += 1
                        continue
                    review_path = self._pending_review_path(
                        video_id, None, feats_mtime, review_index)
                    gt_pending = self._pending_ground_truth(video_id, None, feats_mtime)
                    if review_path is None and not gt_pending:
                        summary['no_manifest'] += 1
                        continue
                    if (review_path is not None
                            and self._review_declares_mislabel(review_path)
                            and not self._segments_human_fixed(video_id)):
                        summary['review_mislabel'] += 1
                        summary['review_mislabel_videos'].append(video_id)
                        if mark_outdated:
                            self._divert_mislabel_to_deep_review(video_id)
                        continue
                    summary['no_manifest_review'] += 1
                    owed = (['human_review'] if review_path is not None else [])
                    owed += (['ground_truth'] if gt_pending else [])
                    self._mark_kinematics_rerun(
                        video_id, owed + ['no manifest'], summary,
                        mark_outdated, full_only)
                    continue

                # Compare against current versions
                comparison = compare_manifest_to_current(manifest, current)
                # Human-authored segmentation is version-exempt (see helper).
                comparison = self._drop_human_seg_staleness(
                    video_id, manifest_index, comparison)
                # A human review the archived kinematics have not applied must
                # still be applied -- re-run so the reviewer's triage
                # resolution flows into features/DB (the extractor applies it;
                # see orchestrator + resolve_review_path).
                review_path = self._pending_review_path(
                    video_id, manifest, feats_mtime, review_index)
                review_pending = review_path is not None
                # Ground truth saved or changed after the kinematics were made
                # is owed the same kinematics-only re-run: the feature
                # extractor applies ground truth over the algorithm's answers.
                gt_pending = self._pending_ground_truth(video_id, manifest, feats_mtime)

                # EXCEPTION: a pending review that declares a segment mislabel
                # (true_segment_num set) must NOT drive a kinematics-only re-run
                # -- that would apply the reviewer's outcomes against the very
                # boundaries the reviewer said are wrong. The video needs manual
                # re-segmentation, so it goes to the DEEP_REVIEW queue instead,
                # mirroring the triage-return divert in review_return.py.
                if (review_pending
                        and self._review_declares_mislabel(review_path)
                        and not self._segments_human_fixed(video_id)):
                    summary['review_mislabel'] += 1
                    summary['review_mislabel_videos'].append(video_id)
                    if mark_outdated:
                        self._divert_mislabel_to_deep_review(video_id)
                    continue

                if comparison['is_current'] and not review_pending and not gt_pending:
                    summary['current'] += 1
                    # A failed re-run whose reason has since gone away (the
                    # declaration now counts the filed results current) is
                    # no longer owed: its message is cleared, so it leaves
                    # the list a person reads.
                    if mark_outdated and (video.get('error_message') or '').startswith(
                            FAILED_RERUN_PREFIX):
                        try:
                            self.db.set_fields(video_id, error_message=None)
                        except Exception as e:
                            logger.warning(f"{video_id}: could not clear its failed "
                                           f"re-run message ({e})")
                else:
                    summary['outdated'] += 1
                    if not comparison['is_current']:
                        if comparison['needs_full_reprocess']:
                            # The manifest names an older DLC model. That does
                            # NOT always mean the video has to go back on a GPU.
                            # After a bulk re-pose, nearly every video whose
                            # manifest names the older model has the declared
                            # model's pose sitting in the archive. What
                            # is stale about them is the ANALYSIS -- segments,
                            # reaches and outcomes were computed from the old
                            # pose -- so every post-DLC stage must re-run, but
                            # inference must not. At roughly 14 minutes of GPU
                            # each, re-posing them would burn about 288 GPU-hours
                            # to produce files that already exist.
                            #
                            # The manifest is not corrected here. It is telling
                            # the truth about what produced the current results;
                            # the reprocess run rewrites it with the pose that
                            # actually ran, which is the only honest way for it
                            # to change.
                            if pose_index is None:
                                pose_index = pose_scorers_in_archive(self.archive_dir)
                            declared = (current.get('versions') or {}).get('dlc_scorer', '')
                            if declared and declared in pose_index.get(video_id, ()):
                                scope = 'segmentation'
                                summary['pose_already_current'] += 1
                                logger.info(
                                    "%s: manifest records %s but the declared pose "
                                    "is already on disk -- re-running from "
                                    "segmentation, no re-pose",
                                    video_id,
                                    (manifest.get('dlc_model') or {}).get('dlc_scorer', '?'))
                            else:
                                scope = 'full'  # genuinely needs a new pose
                        else:
                            # A stage the manifest never recorded a version for
                            # counts like a stale one, at ITS stage. WHY: only
                            # the stale list used to be passed, so a manifest
                            # missing nothing but its kinematics version gave
                            # an empty list and a re-run from segmentation with
                            # every check, instead of kinematics only.
                            scope = earliest_stale_stage(
                                list(comparison['stale_components'])
                                + list(comparison.get('unrecorded_components') or []))
                        stale = list(comparison['stale_components']) + [
                            f"{c} (not recorded)"
                            for c in (comparison.get('unrecorded_components') or [])]
                        if review_pending:
                            stale.append('human_review')
                        if gt_pending:
                            stale.append('ground_truth')
                    else:
                        # version-current, only a newer human review or ground
                        # truth to apply -> re-run just kinematics (the
                        # extractor applies both).
                        scope = 'kinematics'
                        stale = ['human_review'] if review_pending else []
                        stale += ['ground_truth'] if gt_pending else []
                        summary['review_triggered'] += 1

                    if scope == 'full':
                        summary['outdated_full'] += 1
                    else:
                        summary['outdated_partial'] += 1

                    summary['outdated_videos'].append({
                        'video_id': video_id,
                        'scope': scope,
                        'stale_components': stale,
                    })

                    if full_only and scope != 'full':
                        summary['outdated_partial_unmarked'] = (
                            summary.get('outdated_partial_unmarked', 0) + 1)
                        summary['outdated'] -= 1          # counted above; not marked
                        continue
                    if mark_outdated:
                        self.db.force_state(
                            video_id, 'outdated',
                            reprocess_scope=scope,
                            mark_reason=None,  # scanner mark; hand-marks set one
                        )
                        logger.info(
                            f"Marked {video_id} as outdated "
                            f"(scope={scope}, stale: {stale})"
                        )

            except Exception as e:
                summary['errors'] += 1
                logger.error(f"Error scanning {video_id}: {e}")

        # TWO-WAY DOOR: rows already marked 'outdated' are re-checked against
        # the declaration and un-marked when they compare current. WHY: the
        # loop above only walks 'archived', so 'outdated' was a one-way door --
        # two declaration edits with no algorithm change behind them marked
        # a large part of the corpus, and no corrective declaration could ever
        # undo it.
        # With compatible_versions (versions.py) a bugfix bump can declare the
        # old version equivalent; this loop is what lets that declaration
        # actually heal the queue.
        summary['unmarked_current'] = 0
        for video in self.db.get_videos_in_state('outdated'):
            video_id = video['video_id']
            if not is_supported_tray_type(f"{video_id}.mp4"):
                continue  # Easy/Flat are not re-run for versions (see the mark side)
            if (video.get('mark_reason') or '').strip():
                # A HAND-MARK. A person said "re-run this", with a recorded
                # why -- version currency is irrelevant (the usual reason is
                # precisely that the current-version OUTPUTS are wrong). The
                # protection used to live only in the compat branch below, so
                # a version-current-outright row lost its hand-mark within one
                # scan: the dashboard Re-run button's promise self-cancelled
                # inside 30 minutes for every current video, and a repair
                # re-mark was wiped mid-repair. Never un-mark a
                # row carrying a reason; only the drain clears it.
                continue
            try:
                manifest = self._load_manifest_indexed(video_id, manifest_index,
                                                       manifest_mtime)
                if not manifest:
                    continue
                comparison = compare_manifest_to_current(manifest, current)
                # Human-authored segmentation is version-exempt here too --
                # the door must be able to HEAL rows the mark side created
                # before the exemption existed.
                comparison = self._drop_human_seg_staleness(
                    video_id, manifest_index, comparison)
                if not comparison['is_current']:
                    continue
                if comparison.get('compat_used') and (video.get('mark_reason') or '').strip():
                    # Current only via compatible_versions AND carrying an
                    # explicit mark_reason: this is a hand-marked targeted
                    # re-run -- exactly what compat's contract promises
                    # ("affected videos are re-marked by hand"). Never un-mark
                    # it; the drain re-runs it. Compat-current rows WITHOUT a
                    # reason are declaration-era phantoms (the scanner never
                    # writes a reason) and un-marking them is this door's
                    # purpose: a sizable share of outdated rows were phantoms
                    # frozen by the blanket compat-skip this replaces, while
                    # current-outright rows numbered ZERO -- the door served
                    # nothing.
                    continue
                # A saved human review the archived kinematics have not applied
                # still owes a re-run -- same identity-first check as the mark
                # side (mtimes lie across the NAS/local clock boundary).
                if self._pending_review_path(video_id, manifest,
                                             feats_mtime, review_index) is not None:
                    continue  # a re-run is still owed for the review
                if self._pending_ground_truth(video_id, manifest, feats_mtime):
                    continue  # a re-run is still owed for the ground truth
                if mark_outdated:
                    try:
                        self.db.update_state(video_id, 'archived',
                                             reprocess_scope=None,
                                             mark_reason=None)
                    except Exception:
                        self.db.force_state(
                            video_id, 'archived', reprocess_scope=None,
                            mark_reason=None,
                            reason='reprocess mark stale: versions compare current')
                    logger.info(
                        f"Un-marked {video_id}: versions compare current, "
                        f"'outdated' mark was stale")
                summary['unmarked_current'] += 1
            except Exception as e:
                summary['errors'] += 1
                logger.error(f"Error re-checking outdated {video_id}: {e}")

        # Also count crystallized (for reporting)
        crystallized = self.db.get_videos_in_state('crystallized')
        summary['crystallized_skipped'] = len(crystallized)

        logger.info(
            f"Scan complete: {summary['scanned']} checked, "
            f"{summary['current']} current, {summary['outdated']} outdated, "
            f"{summary['unmarked_current']} un-marked (were outdated, now current), "
            f"{summary['crystallized_skipped']} crystallized, "
            f"{summary.get('adopted', 0)} adopted from disk"
        )
        self._save_caches(manifest_index, review_index)

        return summary

    def _pending_ground_truth(self, video_id: str, manifest=None,
                              feats_mtime=None) -> bool:
        """True when ground truth filed beside this video was saved or changed
        after its kinematics were made (pipeline.manifest.ground_truth_applied,
        the rule shared with recording retirement). Only the ground truth in
        the video's own filed folder is looked at, from the scan's one listing
        of the archive (_index_archive). Never raises."""
        hit = getattr(self, "_filed_gt", {}).get(video_id)
        if hit is None:
            return False
        try:
            from mousereach.pipeline.manifest import ground_truth_applied
            gt_path, gt_mtime = hit
            ft = (feats_mtime or {}).get(video_id)
            return not ground_truth_applied(manifest, gt_path, gt_mtime=gt_mtime,
                                            features_mtime=ft,
                                            sha_cache=self._review_sha_cache)
        except Exception as err:
            logger.warning('%s: could not check its ground truth, so it counts as '
                           'applied: %s', video_id, err)
            return False

    def _mark_kinematics_rerun(self, video_id, stale, summary, mark_outdated,
                               full_only) -> None:
        """Count, and when marking, mark a filed video 'outdated' for a
        kinematics-only re-run (scope 'kinematics'): the feature extractor
        applies the saved human answers; no algorithm and no quality or
        hand-score check runs again (orchestrator.filed_rerun_reuses_analysis)."""
        summary['outdated'] += 1
        summary['outdated_partial'] += 1
        summary['review_triggered'] += 1
        summary['outdated_videos'].append({
            'video_id': video_id, 'scope': 'kinematics',
            'stale_components': list(stale)})
        if full_only:
            summary['outdated_partial_unmarked'] = (
                summary.get('outdated_partial_unmarked', 0) + 1)
            summary['outdated'] -= 1
            return
        if mark_outdated:
            self.db.force_state(video_id, 'outdated', reprocess_scope='kinematics',
                                mark_reason=None)
            logger.info(f"Marked {video_id} as outdated (scope=kinematics, "
                        f"stale: {list(stale)})")

    def _pending_review_path(self, video_id: str, manifest=None,
                             feats_mtime=None, review_index=None):
        """Path of a saved human review that the archived kinematics have NOT
        applied -- else None. Such videos are re-run (post_dlc scope); the
        feature extractor then substitutes the human calls. Never raises; any
        error yields None (no spurious reprocessing).

        Identity first, clocks last. The manifest's applied_review stamp
        (reviewed_at + sha256, written by record_kinematic_version the moment
        the extractor applies a review) is compared by CONTENT -- no clock can
        break it. Fallback for legacy manifests without a stamp: the review's
        in-file reviewed_at vs the manifest's created_at, both process-clock
        stamps. Only when either is missing does the old mtime comparison run.
        WHY the mtime comparison had to go: it mixed the NAS clock (SMB-written
        reviews) with the local clock (archived features keep it via copy2),
        and a NAS clock skew of minutes made every fresh re-run look staler
        than the review it had just applied -- reviewed videos reprocessed in
        a loop.

        ``feats_mtime`` is the scan's one-walk {video_id: mtime} index; pass it
        to avoid a per-video rglob of the whole archive. ``review_index`` is
        the scan's ``causal_review_io.review_path_index``; without it the
        review is looked up per video.

        SAME RULE in video_prep/core/collage_provenance.py ``review_applied``
        (recording retirement): change both together --
        tests/test_rerun_and_retirement_agree.py fails if they disagree."""
        try:
            key = None
            if review_index is not None:
                hit = review_index.get(video_id)
                if hit is None:
                    return None
                review, r_mtime, r_size = hit
                key = (str(review), r_size, r_mtime)
            else:
                from mousereach.review.causal_review_io import resolve_review_path
                review = resolve_review_path(video_id)
                if review is None:
                    return None
            stamp = (manifest or {}).get('applied_review') or {}
            if stamp.get('sha256'):
                from mousereach.pipeline.fsutil import sha256_file
                digest = self._review_sha_cache.get(key) if key else None
                if digest is None:
                    digest = sha256_file(review)
                    if key:
                        self._review_sha_cache[key] = digest
                return None if digest == stamp['sha256'] else review
            if feats_mtime is not None:
                ft = feats_mtime.get(video_id)
            else:
                # Never Analyzed/Archive/: a superseded features file keeps its
                # name, and its mtime would say the review was already applied.
                from mousereach.pipeline.analyzed_tree import first_file
                feats = first_file(self.archive_dir,
                                   f"{video_id}_features.json")
                ft = feats.stat().st_mtime if feats else None
            if ft is None:
                return review  # reviewed but no kinematics yet -> needs a run
            reviewed_at = None
            cache = getattr(self, '_reviewed_at_cache', None)
            if cache is None:
                cache = self._reviewed_at_cache = {}
            if key is not None and key in cache:
                reviewed_at = cache[key]
            else:
                try:
                    doc = json.loads(Path(review).read_text(encoding='utf-8'))
                    reviewed_at = doc.get('reviewed_at')
                    if key is not None:
                        cache[key] = reviewed_at      # a failed read is never cached
                except Exception:
                    pass
            created_at = (manifest or {}).get('created_at')
            if reviewed_at and created_at:
                # ISO strings from the same format sort lexicographically.
                return review if str(reviewed_at) > str(created_at) else None
            review_mtime = key[2] if key is not None else review.stat().st_mtime
            return review if review_mtime > ft else None
        except Exception as err:
            logger.warning('%s: could not check for a pending review, so it counts as having none: %s', video_id, err)
            return None

    def _drop_human_seg_staleness(self, video_id: str, manifest_index,
                                  comparison):
        """Remove 'segmenter' from a comparison's stale set when the archived
        segmentation is HUMAN-authored (boundary_source == 'human'). WHY: a
        human's cuts are facts about the video, not algorithm output -- no
        segmenter release can stale them, and the reprocess run preserves them
        rather than re-cutting. So the manifest keeps recording the segmenter
        version that originally cut the video, the comparison keeps calling it
        stale, and the scanner re-marked such videos after every single re-run
        (one video was swept into the versioned Archive dozens of times before
        this was added).

        SAME RULE in video_prep/core/collage_provenance.py
        ``_drop_human_seg_staleness`` (recording retirement): change both
        together -- tests/test_rerun_and_retirement_agree.py fails if they
        disagree."""
        if 'segmenter' not in (comparison.get('stale_components') or ()):
            return comparison
        try:
            mpath = manifest_index.get(video_id)
            if mpath is None:
                return comparison
            seg = Path(mpath).parent / f"{video_id}_segments.json"
            if not seg.exists():
                return comparison
            doc = json.loads(seg.read_text(encoding='utf-8'))
            if doc.get('boundary_source') != 'human':
                return comparison
        except Exception:
            return comparison
        comparison = dict(comparison)
        comparison['stale_components'] = [
            c for c in comparison['stale_components'] if c != 'segmenter']
        if (not comparison['stale_components']
                and not comparison.get('needs_full_reprocess')):
            comparison['is_current'] = True
        return comparison

    @staticmethod
    def _review_declares_mislabel(review_path) -> bool:
        """True if the review carries any segmentation_wrong record (a reviewer
        set true_segment_num). Never raises; unreadable review yields False."""
        try:
            import json
            from mousereach.review.triage_status import segmentation_corrected
            doc = json.loads(Path(review_path).read_text(encoding="utf-8"))
            return bool(segmentation_corrected(doc))
        except Exception as err:
            logger.warning('Could not read review %s, so it counts as declaring no mislabel: %s', review_path, err)
            return False

    def _segments_human_fixed(self, video_id: str) -> bool:
        """True if the archived segments file carries boundary_source == 'human'
        -- the fix-segmentation tool already corrected the boundaries, so an
        old segmentation_wrong record in the review describes a fixed problem
        and must not re-divert the video. Never raises; unreadable -> False."""
        try:
            # Never Analyzed/Archive/: an older generation's human segments
            # there would say the live boundaries were already fixed.
            from mousereach.pipeline.analyzed_tree import first_file
            seg = first_file(self.archive_dir, f"{video_id}_segments.json")
            if seg is None:
                return False
            return json.loads(seg.read_text(
                encoding="utf-8")).get("boundary_source") == "human"
        except Exception as err:
            logger.warning('%s: could not read its segments, so they count as not human-fixed: %s', video_id, err)
            return False

    def _divert_mislabel_to_deep_review(self, video_id: str) -> None:
        """Move the video's bundle out of Analyzed into the DEEP_REVIEW queue so
        a human re-segments it. Never raises; a failure is logged and the video
        stays where it is (it will be retried on the next scan)."""
        try:
            from mousereach.pipeline.analyzed_tree import first_file
            from mousereach.watcher.review_gate import route_deep_review
            from mousereach.watcher.db import crystallized_refusal
            # A crystallized (publication-locked) video is never moved out of
            # its filed folder (db.crystallized_refusal); route_deep_review
            # would refuse too, this says so before looking for its files.
            locked = crystallized_refusal(self.db, video_id, "diverted to deep review")
            if locked:
                logger.warning("%s", locked)
                return
            # Never Analyzed/Archive/: the hit's PARENT is what gets moved into
            # the deep review queue, so an archived copy would route an
            # archive folder (and strip superseded outputs out of the archive).
            outcomes = first_file(self.archive_dir,
                                  f"{video_id}_pellet_outcomes.json")
            if outcomes is None:
                logger.warning(
                    "%s: pending review declares a segment mislabel but no "
                    "pellet_outcomes.json found in the archive to route from",
                    video_id)
                return
            route_deep_review(
                video_id, outcomes.parent,
                reason="pending human review declares segment mislabel "
                       "(true_segment_num set) -- needs re-segmentation",
                db=self.db)
            logger.info(
                "%s: pending review declares a segment mislabel -- routed to "
                "deep review instead of a kinematics-only re-run", video_id)
        except Exception as e:
            logger.error("%s: could not divert to deep review: %s", video_id, e)

    def _index_archive(self):
        """``(manifests {stem: path}, manifest mtimes, features mtimes)`` for the
        live archive, from ONE listing of every folder under Analyzed.

        Every manifest and features file at any depth below a project folder,
        never inside a superseded folder (analyzed_tree says why) and never a
        dot-name or anything inside a dot-folder. The modified times come from
        the listing itself. WHY: separate network stats per file cost
        minutes a pass. WHY any depth: this used to list exactly
        Analyzed/<project>/<cohort>, and a project folder may now be two levels
        (group/project) -- a fixed depth would have dropped every video filed
        that way from the version scan without a word."""
        from mousereach.pipeline.analyzed_tree import iter_file_entries
        man, man_t, feat_t = {}, {}, {}
        self._filed_gt = {}
        root = self.archive_dir
        if not root.exists():
            return man, man_t, feat_t
        for e in iter_file_entries(root, "*.json"):
            rel = Path(e.path).relative_to(root).parts
            if len(rel) < 3 or any(p.startswith(".") for p in rel):
                continue
            n = e.name
            if n.endswith("_processing_manifest.json"):
                s = n[: -len("_processing_manifest.json")]
                man[s] = Path(e.path)
                man_t[s] = e.stat().st_mtime
            elif n.endswith("_features.json"):
                feat_t[n[: -len("_features.json")]] = e.stat().st_mtime
            elif n.endswith(GROUND_TRUTH_SUFFIX):
                # Ground truth filed beside a video, from the same listing
                # (no extra network call): _pending_ground_truth reads it.
                self._filed_gt[n[: -len(GROUND_TRUTH_SUFFIX)]] = (
                    Path(e.path), e.stat().st_mtime)
        return man, man_t, feat_t

    def _load_manifest_indexed(self, video_id: str, manifest_index: dict,
                               manifest_mtime: Optional[dict] = None):
        """Manifest via the scan's one-walk index; falls back to the per-video
        walk only for stems the index missed (e.g. archived mid-scan). The
        modified time comes from the index's listing when it has one."""
        p = manifest_index.get(video_id)
        if p is not None:
            try:
                mtime = (manifest_mtime or {}).get(video_id)
                if mtime is None:
                    mtime = p.stat().st_mtime
                hit = self._manifest_cache.get(video_id)
                if hit is not None and hit[0] == mtime:
                    return hit[1]
                with open(p) as f:
                    manifest = json.load(f)
                self._manifest_cache[video_id] = (mtime, manifest)
                return manifest
            except Exception:
                pass
        return self._load_manifest(video_id)

    def _deep_manifest_index(self) -> dict:
        """``{stem: path}`` for every manifest in the tree at ANY depth, from
        ONE walk, built the first time a stem misses the scan's index and kept
        for the rest of that scan.

        WHY THIS EXISTS. The fallback used to be a per-video
        ``iter_files(archive_dir, "<stem>_processing_manifest.json")`` -- a full
        recursive walk of Analyzed, per video, to look for ONE name. A video
        whose manifest the scan's index does not hold (archived somewhere the
        index does not cover, or simply with no manifest on disk) therefore cost
        a whole-tree walk, and a video with no manifest cost one on EVERY pass
        because nothing ever satisfies it. A few dozen such videos took most
        of a pass that ran over half an hour -- and the work loop does nothing while a
        scan runs. The same walk now serves all of them.

        Same tree, same exclusions and same first-hit-wins order as the old
        per-video walk (one ``iter_files`` in os.walk order), so the manifest
        found for any stem is unchanged."""
        if self._deep_manifests is None:
            from mousereach.pipeline.analyzed_tree import iter_files
            suffix = "_processing_manifest.json"
            found: dict = {}
            for path in iter_files(self.archive_dir, "*" + suffix):
                found.setdefault(path.name[: -len(suffix)], path)
            self._deep_manifests = found
            logger.info("Deep manifest index built from one walk: %d manifest(s) "
                        "(for stems the scan's index missed)", len(found))
        return self._deep_manifests

    def _load_manifest(self, video_id: str) -> Optional[dict]:
        """Find and load processing manifest for an archived video, or None.

        The folder the video is filed in first (the filing rule says where, and
        it is never a superseded folder); then the one-walk deep index for
        anything filed elsewhere. No walk is ever done for a single video.
        """
        if not self.archive_dir.exists():
            return None
        from mousereach.archive.core import filed_folder
        try:
            manifest_path = filed_folder(video_id, self.archive_dir) / f"{video_id}_processing_manifest.json"
        except Exception:
            manifest_path = None
        if manifest_path is None or not manifest_path.is_file():
            manifest_path = self._deep_manifest_index().get(video_id)
        if manifest_path is None:
            return None
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read manifest {manifest_path}: {e}")
            return None

    def get_version_report(self) -> str:
        """Generate a human-readable version compliance report."""
        summary = self.scan(mark_outdated=False)

        lines = []
        lines.append("=" * 70)
        lines.append("Pipeline Version Compliance Report")
        lines.append("=" * 70)
        lines.append("")

        # Current versions
        from mousereach.pipeline.versions import get_current_versions
        current = get_current_versions(self.nas_root)
        if current and current.get('versions'):
            lines.append("Current pipeline versions:")
            for key, value in current['versions'].items():
                lines.append(f"  {key:20s}: {value or '(not set)'}")
            lines.append(f"  Last updated: {current.get('updated_at', '?')}")
        else:
            lines.append("WARNING: No pipeline_versions.json found")
        lines.append("")

        # Summary
        lines.append("Archived video status:")
        lines.append(f"  Total archived:     {summary['scanned']}")
        lines.append(f"  Current (up-to-date): {summary['current']}")
        lines.append(f"  Outdated:           {summary['outdated']}")
        if summary['outdated'] > 0:
            lines.append(f"    Needs re-pose:    {summary['outdated_full']} (no current pose on disk)")
            lines.append(f"    Partial reprocess: {summary['outdated_partial']} (seg/reach/outcomes only)")
            lines.append(f"      of which pose was already current: {summary['pose_already_current']} "
                         f"(manifest named an old model; no GPU needed)")
        lines.append(f"  Crystallized:       {summary['crystallized_skipped']}")
        lines.append(f"  Unsupported tray:   {summary['unsupported_tray']} "
                     f"(Easy/Flat -- not re-run for versions: they need only the session's reaches)")
        lines.append(f"  No manifest:        {summary['no_manifest']}")
        lines.append(f"  Errors:             {summary['errors']}")
        lines.append("")

        # Outdated details
        if summary['outdated_videos']:
            lines.append("Outdated videos:")
            for item in summary['outdated_videos'][:20]:
                scope_label = "FULL" if item['scope'] == 'full' else "partial"
                stale = ', '.join(item['stale_components'])
                lines.append(f"  {item['video_id']:40s} [{scope_label}] stale: {stale}")
            if len(summary['outdated_videos']) > 20:
                lines.append(f"  ... and {len(summary['outdated_videos']) - 20} more")
        lines.append("")

        return '\n'.join(lines)
