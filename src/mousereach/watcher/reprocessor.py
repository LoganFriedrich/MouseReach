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
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Dependency-aware reprocessing: map each version-tracked component to its pipeline
# stage, in order. Given the stale components we re-run from the EARLIEST stale
# post-DLC stage and reuse the still-current upstream outputs.
_STAGE_FOR_COMPONENT = {
    "segmenter": "segmentation",
    "reach_detector": "reach",
    "outcome_detector": "outcome",
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


def pose_scorers_in_archive(archive_dir) -> Dict[str, set]:
    """{video_id: set of DLC scorers whose pose file is in the archive}.

    One walk of the archive tree, not one per video. Pose files do NOT live
    beside a video's results -- results are in Analyzed/{project}/{cohort}/ and
    pose is in Analyzed/{project}/DLC Model N/{cohort}/ -- so there is no cheap
    per-video path to check, and 2,600 separate globs over the NAS is not a scan
    anyone would wait for.
    """
    from mousereach.pipeline.manifest import extract_dlc_model_info

    index: Dict[str, set] = {}
    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        return index
    for h5 in archive_dir.rglob("*DLC*.h5"):
        video_id = h5.name.split("DLC")[0].rstrip("_")
        scorer = extract_dlc_model_info(h5).get('dlc_scorer', '')
        if scorer:
            index.setdefault(video_id, set()).add(scorer)
    return index


class ReprocessingScanner:
    """Scan archived videos for outdated tool versions."""

    def __init__(self, db, nas_root: Path):
        r"""
        Args:
            db: WatcherDB instance
            nas_root: NAS root path (e.g. Y:\LAB_ROOT\Behavior\MouseReach_Pipeline)
        """
        self.db = db
        self.nas_root = Path(nas_root)
        self.archive_dir = self.nas_root / "Analyzed"

    def scan(self, mark_outdated: bool = True) -> dict:
        """Scan all archived videos, optionally mark outdated ones.

        Args:
            mark_outdated: If True, update DB state to 'outdated' for stale videos

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
            'unsupported_tray': 0,      # E/F tray videos: not this pipeline's work
            'crystallized_skipped': 0,
            'no_manifest': 0,
            'review_triggered': 0,  # version-current but a newer human review to apply
            'review_mislabel': 0,   # pending review declares a segment mislabel -> deep review
            'review_mislabel_videos': [],
            'errors': 0,
            'outdated_videos': [],
            'unsupported_tray_videos': [],
        }

        # Load current versions
        current = get_current_versions(self.nas_root)
        if not current or not current.get('versions'):
            logger.warning("No pipeline_versions.json found or empty -- cannot scan")
            return summary

        # Get all archived videos from DB
        archived = self.db.get_videos_in_state('archived')
        logger.info(f"Scanning {len(archived)} archived videos for version compliance")

        # Built on first need only: a scan where nothing is DLC-stale never walks
        # the archive tree at all.
        pose_index = None

        for video in archived:
            video_id = video['video_id']
            summary['scanned'] += 1

            # E (Easy) and F (Flat) tray sessions do not belong in this pipeline
            # -- the algorithms are calibrated for the pillar tray, and the pellet
            # and tray landmarks are not reliably tracked on the others
            # (config.FilePatterns.UNSUPPORTED_TRAY_TYPES; router.SKIP_STEPS
            # already drops outcome detection for them). Scheduling reprocessing
            # work for them spends real machine time on sessions nobody is going
            # to analyse: all 31 videos in the Y: archive that would genuinely
            # need a new pose are E-tray. They are counted and named, not
            # silently dropped, because they should not be in the archive at all
            # and that is a separate decision.
            if not is_supported_tray_type(f"{video_id}.mp4"):
                summary['unsupported_tray'] += 1
                summary['unsupported_tray_videos'].append(video_id)
                continue

            try:
                # Load manifest
                manifest = self._load_manifest(video_id)
                if not manifest:
                    summary['no_manifest'] += 1
                    continue

                # Compare against current versions
                comparison = compare_manifest_to_current(manifest, current)
                # A freshly-saved human review that post-dates the archived
                # kinematics must also be applied -- re-run so the reviewer's
                # triage resolution flows into features/DB (the extractor applies
                # it; see orchestrator + resolve_review_path).
                review_path = self._pending_review_path(video_id)
                review_pending = review_path is not None

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

                if comparison['is_current'] and not review_pending:
                    summary['current'] += 1
                else:
                    summary['outdated'] += 1
                    if not comparison['is_current']:
                        if comparison['needs_full_reprocess']:
                            # The manifest names an older DLC model. That does
                            # NOT always mean the video has to go back on a GPU.
                            # The bulk re-pose already ran: 1,233 of the 1,264
                            # videos whose manifests say shuffle1 have the
                            # declared shuffle3 pose sitting in the archive. What
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
                            scope = earliest_stale_stage(comparison['stale_components'])
                        stale = list(comparison['stale_components'])
                        if review_pending:
                            stale.append('human_review')
                    else:
                        # version-current, only a newer human review to apply ->
                        # re-run just kinematics (the extractor applies the review).
                        scope = 'kinematics'
                        stale = ['human_review']
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

                    if mark_outdated:
                        self.db.force_state(
                            video_id, 'outdated',
                            reprocess_scope=scope,
                        )
                        logger.info(
                            f"Marked {video_id} as outdated "
                            f"(scope={scope}, stale: {stale})"
                        )

            except Exception as e:
                summary['errors'] += 1
                logger.error(f"Error scanning {video_id}: {e}")

        # Also count crystallized (for reporting)
        crystallized = self.db.get_videos_in_state('crystallized')
        summary['crystallized_skipped'] = len(crystallized)

        logger.info(
            f"Scan complete: {summary['scanned']} checked, "
            f"{summary['current']} current, {summary['outdated']} outdated, "
            f"{summary['crystallized_skipped']} crystallized"
        )

        return summary

    def _pending_review_path(self, video_id: str):
        """Path of a saved human review that is NEWER than the archived
        kinematics -- i.e. the reviewer's triage resolution has not yet been
        applied to the features/DB product -- else None. Such videos are re-run
        (post_dlc scope); the feature extractor then substitutes the human
        calls. Never raises; any error yields None (no spurious reprocessing)."""
        try:
            from mousereach.review.causal_review_io import resolve_review_path
            review = resolve_review_path(video_id)
            if review is None:
                return None
            feats = next(self.archive_dir.rglob(f"{video_id}_features.json"), None)
            if feats is None:
                return review  # reviewed but no kinematics yet -> needs a run
            return review if review.stat().st_mtime > feats.stat().st_mtime else None
        except Exception:
            return None

    @staticmethod
    def _review_declares_mislabel(review_path) -> bool:
        """True if the review carries any segmentation_wrong record (a reviewer
        set true_segment_num). Never raises; unreadable review yields False."""
        try:
            import json
            from mousereach.review.triage_status import segmentation_corrected
            doc = json.loads(Path(review_path).read_text(encoding="utf-8"))
            return bool(segmentation_corrected(doc))
        except Exception:
            return False

    def _segments_human_fixed(self, video_id: str) -> bool:
        """True if the archived segments file carries boundary_source == 'human'
        -- the fix-segmentation tool already corrected the boundaries, so an
        old segmentation_wrong record in the review describes a fixed problem
        and must not re-divert the video. Never raises; unreadable -> False."""
        try:
            seg = next(self.archive_dir.rglob(f"{video_id}_segments.json"), None)
            if seg is None:
                return False
            return json.loads(seg.read_text(
                encoding="utf-8")).get("boundary_source") == "human"
        except Exception:
            return False

    def _divert_mislabel_to_deep_review(self, video_id: str) -> None:
        """Move the video's bundle out of Analyzed into the DEEP_REVIEW queue so
        a human re-segments it. Never raises; a failure is logged and the video
        stays where it is (it will be retried on the next scan)."""
        try:
            from mousereach.watcher.review_gate import route_deep_review
            outcomes = next(
                self.archive_dir.rglob(f"{video_id}_pellet_outcomes.json"), None)
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

    def _load_manifest(self, video_id: str) -> Optional[dict]:
        """Find and load processing manifest for an archived video.

        Searches the archive directory tree for {video_id}_processing_manifest.json.
        """
        if not self.archive_dir.exists():
            return None

        # Try direct glob first (project/cohort/manifest)
        for manifest_path in self.archive_dir.glob(f"*/*/{video_id}_processing_manifest.json"):
            try:
                with open(manifest_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read manifest {manifest_path}: {e}")

        # Fall back to recursive search
        for manifest_path in self.archive_dir.rglob(f"{video_id}_processing_manifest.json"):
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
                     f"(E/F sessions -- not this pipeline's work)")
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
