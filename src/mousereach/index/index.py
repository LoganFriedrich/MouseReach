#!/usr/bin/env python3
"""
Pipeline Index - Fast file tracking for MouseReach pipeline.

Instead of scanning folders on every startup (slow on network drives),
we maintain an index file that tracks:
- All file paths and their locations
- Parsed JSON metadata (validation status, versions)
- Folder modification times (for smart invalidation)

Read index once (~5ms) instead of scanning folders (~30s).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class PipelineIndex:
    """Fast pipeline file index with smart invalidation.

    Architecture (v2.0+):
        - Single "Processing" folder for all post-DLC files
        - Review status determined by validation_status in JSON metadata
        - Files stay co-located (video + DLC + segments + reaches + outcomes)
    """

    INDEX_FILE = "pipeline_index.json"
    VERSION = "2.0"  # Bumped to trigger rebuilds for new architecture

    # Pipeline folders - unified architecture
    STAGES = [
        "DLC_Queue",   # Videos waiting for DLC processing
        "Processing",  # All post-DLC files (status in JSON metadata)
        "Failed",      # Processing errors
    ]

    # Validation status values
    STATUS_NEEDS_REVIEW = "needs_review"
    STATUS_AUTO_APPROVED = "auto_approved"
    STATUS_VALIDATED = "validated"

    def __init__(self, processing_root: Path = None):
        """Initialize index.

        Args:
            processing_root: Pipeline root directory. If None, uses config default.
        """
        if processing_root is None:
            from mousereach.config import PROCESSING_ROOT
            processing_root = PROCESSING_ROOT

        self.root = Path(processing_root)
        self.index_path = self.root / self.INDEX_FILE
        self._data: Optional[Dict] = None
        self._loaded = False
        self._dirty = False

    # =========================================================================
    # INDEX FILE I/O
    # =========================================================================

    def _empty_index(self) -> Dict:
        """Create empty index structure."""
        return {
            "version": self.VERSION,
            "generated_at": datetime.now().isoformat(),
            "processing_root": str(self.root),
            "folder_mtimes": {},
            "videos": {},
        }

    def load(self) -> Dict:
        """Load index from disk. Fast: single file read.

        Returns:
            Index data dictionary.
        """
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    self._data = json.load(f)
                # Validate version
                if self._data.get("version") != self.VERSION:
                    print(f"[Index] Version mismatch, rebuilding...")
                    self._data = self._empty_index()
                    self._dirty = True
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Index] Corrupt index file, rebuilding: {e}")
                self._data = self._empty_index()
                self._dirty = True
        else:
            self._data = self._empty_index()
            self._dirty = True

        self._loaded = True
        return self._data

    def save(self):
        """Write index to disk (atomic write with temp file)."""
        if not self._loaded:
            return

        self._data["generated_at"] = datetime.now().isoformat()

        # Atomic write: write to temp file then rename
        temp_path = self.index_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(self._data, f, indent=2)
            temp_path.replace(self.index_path)
            self._dirty = False
        except Exception as e:
            print(f"[Index] Failed to save: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def _ensure_loaded(self):
        """Load index if not already loaded."""
        if not self._loaded:
            self.load()

    def _is_supported_video(self, video_id: str, data: Dict = None) -> bool:
        """Check if video uses a supported tray type (filters out Flat/Easy trays).

        Args:
            video_id: Video identifier
            data: Optional pre-fetched video data dict

        Returns:
            True if tray is supported (Pillar), False for Flat/Easy trays
        """
        if data is None:
            data = self._data.get("videos", {}).get(video_id, {})

        metadata = data.get("metadata", {})
        # tray_supported is set during scanning from parse_tray_type()
        return metadata.get("tray_supported", True)  # Default True for backwards compat

    # =========================================================================
    # READ OPERATIONS (FAST)
    # =========================================================================

    def get_all_videos(self) -> Dict[str, Dict]:
        """Get all indexed videos.

        Returns:
            Dict mapping video_id -> video data
        """
        self._ensure_loaded()
        return self._data.get("videos", {})

    def get_videos_in_stage(self, stage: str, include_unsupported: bool = False) -> List[str]:
        """Get all video_ids currently in a pipeline stage.

        Args:
            stage: Pipeline stage name (e.g., "Processing")
            include_unsupported: If False (default), excludes Flat/Easy tray videos

        Returns:
            List of video_ids in that stage
        """
        self._ensure_loaded()
        return [
            vid for vid, data in self._data.get("videos", {}).items()
            if data.get("current_stage") == stage
            and (include_unsupported or self._is_supported_video(vid, data))
        ]

    def get_video_data(self, video_id: str) -> Optional[Dict]:
        """Get all cached data for a video.

        Args:
            video_id: Video identifier (e.g., "20250704_CNT0101_P1")

        Returns:
            Video data dict or None if not found
        """
        self._ensure_loaded()
        return self._data.get("videos", {}).get(video_id)

    def get_video_metadata(self, video_id: str) -> Dict:
        """Get cached metadata for a video.

        Args:
            video_id: Video identifier

        Returns:
            Metadata dict (validation status, confidence, etc.)
        """
        self._ensure_loaded()
        video_data = self._data.get("videos", {}).get(video_id, {})
        return video_data.get("metadata", {})

    def get_video_files(self, video_id: str) -> Dict[str, List[str]]:
        """Get file paths for a video by stage.

        Args:
            video_id: Video identifier

        Returns:
            Dict mapping stage -> list of filenames
        """
        self._ensure_loaded()
        video_data = self._data.get("videos", {}).get(video_id, {})
        return video_data.get("files", {})

    def get_dlc_files(self, stage: str = None) -> List[Path]:
        """Get DLC file paths, optionally filtered by stage.

        Args:
            stage: Optional stage filter (e.g., "Processing")

        Returns:
            List of Path objects to DLC .h5 files
        """
        self._ensure_loaded()
        results = []
        for video_id, data in self._data.get("videos", {}).items():
            if stage and data.get("current_stage") != stage:
                continue
            files = data.get("files", {})
            for folder, filenames in files.items():
                for fname in filenames:
                    if "DLC" in fname and fname.endswith(".h5"):
                        results.append(self.root / folder / fname)
        return results

    def get_needs_review(self, step: str) -> List[str]:
        """Get videos needing review for a step (based on metadata, not folder).

        Args:
            step: "seg", "reach", or "outcome"

        Returns:
            List of video_ids needing review
        """
        self._ensure_loaded()

        # Use metadata-based detection
        if step == "seg":
            return self.get_needs_seg_review()
        elif step == "reach":
            return self.get_needs_reach_review()
        elif step == "outcome":
            return self.get_needs_outcome_review()
        return []

    # =========================================================================
    # STATE-BASED REVIEW DETECTION (NEW ARCHITECTURE)
    # =========================================================================

    def get_needs_seg_review(self, include_unsupported: bool = False,
                              include_auto_approved: bool = False) -> List[str]:
        """Get videos where segmentation needs review.

        A video needs seg review if:
        - Has _segments.json file (seg_validation in metadata)
        - seg_validation == "needs_review" (excludes auto_approved unless requested)
        - Uses a supported tray type (unless include_unsupported=True)
        - Does NOT have a complete GT file (all items human_verified)

        Args:
            include_unsupported: If True, includes Flat/Easy tray videos
            include_auto_approved: If True, also returns auto_approved videos

        Returns:
            List of video_ids needing segmentation review
        """
        self._ensure_loaded()
        result = []
        for video_id, data in self._data.get("videos", {}).items():
            # Skip unsupported tray types
            if not include_unsupported and not self._is_supported_video(video_id, data):
                continue

            metadata = data.get("metadata", {})
            seg_status = metadata.get("seg_validation")

            # Skip if validated
            if seg_status == self.STATUS_VALIDATED:
                continue

            # Skip if not started
            if not seg_status:
                continue

            # Only include needs_review (unless auto_approved requested)
            if seg_status == self.STATUS_AUTO_APPROVED and not include_auto_approved:
                continue

            # Skip if GT is complete (all items human_verified)
            if metadata.get("seg_gt_complete"):
                continue

            result.append(video_id)

        return result

    def get_needs_reach_review(self, include_unsupported: bool = False,
                               include_auto_approved: bool = False) -> List[str]:
        """Get videos where reach detection needs review.

        A video needs reach review if:
        - Has _reaches.json file (reach_validation in metadata)
        - reach_validation == "needs_review" (excludes auto_approved unless requested)
        - Uses a supported tray type (unless include_unsupported=True)
        - Does NOT have a complete GT file (all items human_verified)

        Args:
            include_unsupported: If True, includes Flat/Easy tray videos
            include_auto_approved: If True, also returns auto_approved videos

        Returns:
            List of video_ids needing reach review
        """
        self._ensure_loaded()
        result = []
        for video_id, data in self._data.get("videos", {}).items():
            # Skip unsupported tray types
            if not include_unsupported and not self._is_supported_video(video_id, data):
                continue

            metadata = data.get("metadata", {})
            reach_status = metadata.get("reach_validation")

            # Skip if validated
            if reach_status == self.STATUS_VALIDATED:
                continue

            # Skip if not started
            if not reach_status:
                continue

            # Only include needs_review (unless auto_approved requested)
            if reach_status == self.STATUS_AUTO_APPROVED and not include_auto_approved:
                continue

            # Skip if GT is complete (all items human_verified)
            if metadata.get("reach_gt_complete"):
                continue

            result.append(video_id)

        return result

    def get_needs_outcome_review(self, include_unsupported: bool = False,
                                  include_auto_approved: bool = False) -> List[str]:
        """Get videos where outcome detection needs review.

        A video needs outcome review if:
        - Has _pellet_outcomes.json file (outcome_validation in metadata)
        - outcome_validation == "needs_review" (excludes auto_approved unless requested)
        - Uses a supported tray type (unless include_unsupported=True)
        - Does NOT have a complete GT file (all items human_verified)

        Args:
            include_unsupported: If True, includes Flat/Easy tray videos
            include_auto_approved: If True, also returns auto_approved videos

        Returns:
            List of video_ids needing outcome review
        """
        self._ensure_loaded()
        result = []
        for video_id, data in self._data.get("videos", {}).items():
            # Skip unsupported tray types
            if not include_unsupported and not self._is_supported_video(video_id, data):
                continue

            metadata = data.get("metadata", {})
            outcome_status = metadata.get("outcome_validation")

            # Skip if validated
            if outcome_status == self.STATUS_VALIDATED:
                continue

            # Skip if not started
            if not outcome_status:
                continue

            # Only include needs_review (unless auto_approved requested)
            if outcome_status == self.STATUS_AUTO_APPROVED and not include_auto_approved:
                continue

            # Skip if GT is complete (all items human_verified)
            if metadata.get("outcome_gt_complete"):
                continue

            result.append(video_id)

        return result

    def get_ready_to_archive(self, include_unsupported: bool = False) -> List[str]:
        """Get videos ready to be archived (all stages validated).

        A video is ready for archive if ALL of:
        - seg_validation == "validated"
        - reach_validation == "validated"
        - outcome_validation == "validated"
        - Uses a supported tray type (unless include_unsupported=True)

        Returns:
            List of video_ids ready for archiving
        """
        self._ensure_loaded()
        result = []
        for video_id, data in self._data.get("videos", {}).items():
            # Skip unsupported tray types
            if not include_unsupported and not self._is_supported_video(video_id, data):
                continue

            metadata = data.get("metadata", {})

            seg_ok = metadata.get("seg_validation") == self.STATUS_VALIDATED
            reach_ok = metadata.get("reach_validation") == self.STATUS_VALIDATED
            outcome_ok = metadata.get("outcome_validation") == self.STATUS_VALIDATED

            if seg_ok and reach_ok and outcome_ok:
                result.append(video_id)

        return result

    def get_pipeline_status(self, video_id: str) -> Dict[str, str]:
        """Get validation status for all pipeline steps for a video.

        Returns:
            Dict with keys: seg, reach, outcome
            Values: "not_started", "needs_review", "auto_approved", "validated"
        """
        self._ensure_loaded()
        video_data = self._data.get("videos", {}).get(video_id, {})
        metadata = video_data.get("metadata", {})

        return {
            "seg": metadata.get("seg_validation", "not_started"),
            "reach": metadata.get("reach_validation", "not_started"),
            "outcome": metadata.get("outcome_validation", "not_started"),
        }

    def get_stage_counts(self, include_unsupported: bool = False) -> Dict[str, int]:
        """Get count of videos in each folder.

        Args:
            include_unsupported: If False (default), excludes Flat/Easy tray videos

        Returns:
            Dict mapping folder name -> video count
        """
        self._ensure_loaded()
        counts = {stage: 0 for stage in self.STAGES}
        for video_id, data in self._data.get("videos", {}).items():
            # Skip unsupported tray types
            if not include_unsupported and not self._is_supported_video(video_id, data):
                continue

            stage = data.get("current_stage")
            if stage in counts:
                counts[stage] += 1
        return counts

    def get_status_counts(self, include_unsupported: bool = False) -> Dict[str, Dict[str, int]]:
        """Get count of videos by validation status for each step.

        Args:
            include_unsupported: If False (default), excludes Flat/Easy tray videos

        Returns:
            Dict with keys: seg, reach, outcome
            Each value is a dict: {needs_review: N, auto_approved: N, validated: N, not_started: N}
        """
        self._ensure_loaded()

        counts = {
            "seg": {"not_started": 0, "needs_review": 0, "auto_approved": 0, "validated": 0},
            "reach": {"not_started": 0, "needs_review": 0, "auto_approved": 0, "validated": 0},
            "outcome": {"not_started": 0, "needs_review": 0, "auto_approved": 0, "validated": 0},
        }

        for video_id, data in self._data.get("videos", {}).items():
            # Skip unsupported tray types
            if not include_unsupported and not self._is_supported_video(video_id, data):
                continue

            metadata = data.get("metadata", {})

            for step in ["seg", "reach", "outcome"]:
                status = metadata.get(f"{step}_validation", "not_started")
                if status in counts[step]:
                    counts[step][status] += 1

        return counts

    # =========================================================================
    # WRITE OPERATIONS (UPDATE INDEX)
    # =========================================================================

    def record_video(self, video_id: str, stage: str, files: Dict[str, List[str]],
                     metadata: Dict = None):
        """Add or update a video entry in the index.

        Args:
            video_id: Video identifier
            stage: Current pipeline stage
            files: Dict mapping stage -> list of filenames
            metadata: Optional metadata dict
        """
        self._ensure_loaded()

        if video_id not in self._data["videos"]:
            self._data["videos"][video_id] = {
                "video_id": video_id,
                "current_stage": stage,
                "files": {},
                "metadata": {},
                "mtimes": {},
            }

        video_data = self._data["videos"][video_id]
        video_data["current_stage"] = stage

        # Merge files
        for folder, filenames in files.items():
            if folder not in video_data["files"]:
                video_data["files"][folder] = []
            for fname in filenames:
                if fname not in video_data["files"][folder]:
                    video_data["files"][folder].append(fname)

        # Update metadata
        if metadata:
            video_data["metadata"].update(metadata)

        self._dirty = True

    def record_file_created(self, path: Path, metadata: Dict = None):
        """Called when a new file is created (e.g., _segments.json).

        Automatically extracts video_id and updates the index.

        Args:
            path: Path to the created file
            metadata: Optional metadata to store
        """
        from mousereach.config import get_video_id

        video_id = get_video_id(path.name)
        stage = path.parent.name
        fname = path.name

        self._ensure_loaded()

        if video_id not in self._data["videos"]:
            self._data["videos"][video_id] = {
                "video_id": video_id,
                "current_stage": stage,
                "files": {},
                "metadata": {},
                "mtimes": {},
            }

        video_data = self._data["videos"][video_id]

        # Add file to appropriate stage
        if stage not in video_data["files"]:
            video_data["files"][stage] = []
        if fname not in video_data["files"][stage]:
            video_data["files"][stage].append(fname)

        # Update current stage
        video_data["current_stage"] = stage

        # Update metadata
        if metadata:
            video_data["metadata"].update(metadata)

        # Record mtime
        if path.exists():
            video_data["mtimes"][path.suffix.lstrip('.')] = path.stat().st_mtime

        self._dirty = True

    def record_files_moved(self, video_id: str, from_stage: str, to_stage: str,
                           files: List[str] = None):
        """Called when files move between folders (triage/advance).

        Args:
            video_id: Video identifier
            from_stage: Source stage
            to_stage: Destination stage
            files: Optional list of filenames that moved
        """
        self._ensure_loaded()

        video_data = self._data["videos"].get(video_id)
        if not video_data:
            return

        # Move files in index
        if files:
            if from_stage in video_data["files"]:
                for fname in files:
                    if fname in video_data["files"][from_stage]:
                        video_data["files"][from_stage].remove(fname)
                # Clean up empty list
                if not video_data["files"][from_stage]:
                    del video_data["files"][from_stage]

            if to_stage not in video_data["files"]:
                video_data["files"][to_stage] = []
            video_data["files"][to_stage].extend(files)

        # Update current stage
        video_data["current_stage"] = to_stage
        self._dirty = True

    def record_validation_changed(self, video_id: str, step: str, status: str,
                                   extra: Dict = None):
        """Called when validation status changes.

        Args:
            video_id: Video identifier
            step: "seg", "reach", or "outcome"
            status: Validation status ("validated", "needs_review", "auto_review")
            extra: Optional extra metadata
        """
        self._ensure_loaded()

        video_data = self._data["videos"].get(video_id)
        if not video_data:
            return

        video_data["metadata"][f"{step}_validation"] = status
        if extra:
            video_data["metadata"].update(extra)

        self._dirty = True

    def record_gt_created(self, video_id: str, gt_type: str, is_complete: bool = False):
        """Record that a ground truth file was created.

        Args:
            video_id: Video identifier
            gt_type: Type of GT ("seg", "reach", or "outcome")
            is_complete: True if ALL items in GT are human_verified
        """
        self._ensure_loaded()

        video_data = self._data["videos"].get(video_id)
        if not video_data:
            return

        # Track GT files in metadata
        gt_key = f"{gt_type}_gt"
        video_data["metadata"][gt_key] = True
        video_data["metadata"][f"{gt_key}_at"] = datetime.now().isoformat()
        video_data["metadata"][f"{gt_type}_gt_complete"] = is_complete

        self._dirty = True

    def record_gt_complete(self, video_id: str, gt_type: str, is_complete: bool):
        """Update whether a GT file is complete (all items human_verified).

        Args:
            video_id: Video identifier
            gt_type: Type of GT ("seg", "reach", or "outcome")
            is_complete: True if ALL items in GT are human_verified
        """
        self._ensure_loaded()

        video_data = self._data["videos"].get(video_id)
        if not video_data:
            return

        video_data["metadata"][f"{gt_type}_gt_complete"] = is_complete
        self._dirty = True

    def remove_video(self, video_id: str):
        """Remove a video from the index.

        Args:
            video_id: Video identifier to remove
        """
        self._ensure_loaded()
        if video_id in self._data["videos"]:
            del self._data["videos"][video_id]
            self._dirty = True

    # =========================================================================
    # SMART INVALIDATION
    # =========================================================================

    def check_stale_folders(self) -> List[str]:
        """Check which folders have changed since last scan.

        Returns:
            List of folder names that need refreshing
        """
        self._ensure_loaded()
        stale = []

        for stage in self.STAGES:
            folder_path = self.root / stage
            if not folder_path.exists():
                continue

            cached_mtime = self._data["folder_mtimes"].get(stage, 0)
            try:
                current_mtime = folder_path.stat().st_mtime
                if current_mtime > cached_mtime:
                    stale.append(stage)
            except OSError:
                pass

        return stale

    def update_folder_mtime(self, stage: str):
        """Update cached mtime for a folder.

        Args:
            stage: Folder/stage name
        """
        self._ensure_loaded()
        folder_path = self.root / stage
        if folder_path.exists():
            self._data["folder_mtimes"][stage] = folder_path.stat().st_mtime
            self._dirty = True

    def refresh_folder(self, stage: str, progress_callback=None):
        """Re-scan a single folder that changed.

        Args:
            stage: Folder/stage name to refresh
            progress_callback: Optional callback(current, total, message)
        """
        from .scanner import scan_folder

        self._ensure_loaded()

        folder_path = self.root / stage
        if not folder_path.exists():
            return

        # Remove old entries for this stage
        videos_to_update = []
        for video_id, data in list(self._data["videos"].items()):
            if data.get("current_stage") == stage:
                videos_to_update.append(video_id)

        # Scan folder and update index
        scan_folder(self, folder_path, stage, progress_callback)

        # Update folder mtime
        self.update_folder_mtime(stage)
        self._dirty = True

    # =========================================================================
    # STATUS / DIAGNOSTICS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get index status summary.

        Returns:
            Dict with status information
        """
        self._ensure_loaded()

        return {
            "index_path": str(self.index_path),
            "exists": self.index_path.exists(),
            "version": self._data.get("version"),
            "generated_at": self._data.get("generated_at"),
            "total_videos": len(self._data.get("videos", {})),
            "stage_counts": self.get_stage_counts(),
            "stale_folders": self.check_stale_folders(),
            "dirty": self._dirty,
        }

    def print_status(self):
        """Print index status to console."""
        status = self.get_status()

        # Count unsupported videos
        total_all = len(self._data.get("videos", {}))
        unsupported_count = sum(
            1 for vid, data in self._data.get("videos", {}).items()
            if not self._is_supported_video(vid, data)
        )
        supported_count = total_all - unsupported_count

        print("=" * 60)
        print("MouseReach Pipeline Index Status (v2.0 - Single Folder Architecture)")
        print("=" * 60)
        print(f"Index file: {status['index_path']}")
        print(f"Exists: {status['exists']}")
        print(f"Version: {status['version']}")
        print(f"Generated: {status['generated_at']}")
        print(f"Total videos: {supported_count}", end="")
        if unsupported_count > 0:
            print(f" ({unsupported_count} Flat/Easy tray videos hidden)")
        else:
            print()
        print()

        # Show folder counts
        print("Videos by folder:")
        for stage, count in status['stage_counts'].items():
            if count > 0:
                print(f"  {stage}: {count}")
        print()

        # Show validation status counts
        status_counts = self.get_status_counts()
        print("Validation status:")
        for step in ["seg", "reach", "outcome"]:
            counts = status_counts[step]
            needs = counts["needs_review"] + counts["auto_approved"]  # Both need potential review
            validated = counts["validated"]
            total = needs + validated + counts["not_started"]
            if total > 0:
                print(f"  {step.upper()}: {validated} validated, {needs} need review, {counts['not_started']} not started")

        # Show ready to archive
        ready = len(self.get_ready_to_archive())
        if ready > 0:
            print(f"\nReady to archive: {ready} videos")

        print()
        if status['stale_folders']:
            print(f"Stale folders (need refresh): {status['stale_folders']}")
        else:
            print("All folders up to date")
        print("=" * 60)
