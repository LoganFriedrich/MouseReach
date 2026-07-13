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


class ReprocessingScanner:
    """Scan archived videos for outdated tool versions."""

    def __init__(self, db, nas_root: Path):
        """
        Args:
            db: WatcherDB instance
            nas_root: NAS root path (e.g. Y:\2_Connectome\Behavior\MouseReach_Pipeline)
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

        summary = {
            'scanned': 0,
            'current': 0,
            'outdated': 0,
            'outdated_full': 0,    # Need full reprocess (DLC changed)
            'outdated_partial': 0, # Only need seg/reach/outcomes rerun
            'crystallized_skipped': 0,
            'no_manifest': 0,
            'review_triggered': 0,  # version-current but a newer human review to apply
            'errors': 0,
            'outdated_videos': [],
        }

        # Load current versions
        current = get_current_versions(self.nas_root)
        if not current or not current.get('versions'):
            logger.warning("No pipeline_versions.json found or empty -- cannot scan")
            return summary

        # Get all archived videos from DB
        archived = self.db.get_videos_in_state('archived')
        logger.info(f"Scanning {len(archived)} archived videos for version compliance")

        for video in archived:
            video_id = video['video_id']
            summary['scanned'] += 1

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
                review_pending = self._review_pending(video_id)

                if comparison['is_current'] and not review_pending:
                    summary['current'] += 1
                else:
                    summary['outdated'] += 1
                    if not comparison['is_current']:
                        scope = 'full' if comparison['needs_full_reprocess'] else 'post_dlc'
                        stale = list(comparison['stale_components'])
                        if review_pending:
                            stale.append('human_review')
                    else:
                        # version-current, but a newer human review needs applying
                        scope = 'post_dlc'
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

    def _review_pending(self, video_id: str) -> bool:
        """True if a saved human review exists and is NEWER than the archived
        kinematics -- i.e. the reviewer's triage resolution has not yet been
        applied to the features/DB product. Such videos are re-run (post_dlc
        scope); the feature extractor then substitutes the human calls. Never
        raises; any error yields False (no spurious reprocessing)."""
        try:
            from mousereach.review.causal_review_io import resolve_review_path
            review = resolve_review_path(video_id)
            if review is None:
                return False
            feats = next(self.archive_dir.rglob(f"{video_id}_features.json"), None)
            if feats is None:
                return True  # reviewed but no kinematics yet -> needs a run
            return review.stat().st_mtime > feats.stat().st_mtime
        except Exception:
            return False

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
            lines.append(f"    Full reprocess:   {summary['outdated_full']} (DLC model changed)")
            lines.append(f"    Partial reprocess: {summary['outdated_partial']} (seg/reach/outcomes only)")
        lines.append(f"  Crystallized:       {summary['crystallized_skipped']}")
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
