"""Triage clearing tool: per-segment review of algo-flagged segments.

Wraps the GroundTruthWidget with a worklist-driven workflow:
  - Pulls one segment at a time from a TriageWorklist
  - Loads only that segment's frame range
  - Reviewer marks the causal reach (start, end, interaction frame,
    outcome, notes about why algo missed it)
  - Saves cleared state to the algo JSONs and unified GT JSON, then
    auto-advances to the next triaged segment
  - When all segments cleared: "all caught up" dialog and close

Distinct from the GT scoring tool: this tool's purpose is closing
algo-flagged triage, not generating ground truth from scratch. CLI
entry: ``mousereach-review-tool``.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)
from napari.utils.notifications import show_info, show_warning

from .ground_truth_widget import GroundTruthWidget
from .triage_queue import TriageEntry, TriageWorklist
from .unified_gt import get_timestamp, get_username, save_unified_gt


# Banner colors to make this visually distinct from the GT tool
BANNER_BG = "#FFF3CD"  # amber
BANNER_BORDER = "#FFC107"
BANNER_TEXT = "#856404"


class TriageClearingWidget(GroundTruthWidget):
    """GroundTruthWidget subclass that drives a TriageWorklist.

    Behavior differences vs the GT widget:
      - Loads only one triaged segment at a time (frame range = segment
        boundaries + padding)
      - Adds a top instruction banner with the algo's triage reason
      - Adds a notes QTextEdit for "why algo couldn't get this"
      - Adds a navigation row (Prev / Save&Next / Skip)
      - Overrides save to do per-segment writes that preserve other
        segments and set ``triage_cleared``, ``original_triage_reason``,
        ``human_verified`` fields

    Save flow:
      - ``Ctrl+S`` → ``_save_and_advance()`` → ``_save_current_triage_segment()``
        → ``_advance_worklist()``
    """

    def __init__(
        self,
        napari_viewer,
        worklist: TriageWorklist,
        pre_pad: int = 30,
        post_pad: int = 30,
    ):
        # IMPORTANT: super().__init__ calls _build_ui which sets up the
        # entire widget. We add our extra UI bits after that.
        super().__init__(napari_viewer, review_mode=True)
        self.worklist = worklist
        self.pre_pad = pre_pad
        self.post_pad = post_pad
        self._notes_edit: Optional[QTextEdit] = None
        self._banner_label: Optional[QLabel] = None
        self._progress_label: Optional[QLabel] = None
        self._build_triage_clearing_ui()

    # ------------------------------------------------------------------
    # Triage clearing UI
    # ------------------------------------------------------------------

    def _build_triage_clearing_ui(self) -> None:
        """Add the triage banner, notes box, and nav row to the existing UI."""
        # The parent's _build_ui has already populated self.layout(). We
        # prepend the banner at index 0 and append the nav row at the end.
        main_layout = self.layout()

        # Banner (instruction + algo triage reason)
        banner = QFrame()
        banner.setStyleSheet(
            f"QFrame {{ background: {BANNER_BG}; border: 2px solid {BANNER_BORDER}; "
            f"padding: 6px; }} "
            f"QLabel {{ color: {BANNER_TEXT}; font-weight: bold; }}"
        )
        banner_layout = QVBoxLayout()
        banner_layout.setContentsMargins(8, 6, 8, 6)
        banner.setLayout(banner_layout)
        title = QLabel("TRIAGE CLEARING — find the causal reach in this segment")
        title.setStyleSheet("font-size: 13px; font-weight: bold;")
        banner_layout.addWidget(title)
        self._banner_label = QLabel("(no segment loaded)")
        self._banner_label.setWordWrap(True)
        self._banner_label.setStyleSheet("font-size: 11px; font-weight: normal;")
        banner_layout.addWidget(self._banner_label)
        main_layout.insertWidget(0, banner)

        # Notes box
        notes_frame = QFrame()
        notes_layout = QVBoxLayout()
        notes_layout.setContentsMargins(5, 5, 5, 5)
        notes_frame.setLayout(notes_layout)
        notes_layout.addWidget(QLabel("<b>Reviewer notes (why algo couldn't get this):</b>"))
        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText(
            "Optional. e.g., 'pellet bounced off pillar then settled in SA, "
            "algo missed because…'"
        )
        self._notes_edit.setMaximumHeight(60)
        notes_layout.addWidget(self._notes_edit)
        main_layout.addWidget(notes_frame)

        # Navigation row
        nav = QFrame()
        nav.setStyleSheet(f"QFrame {{ background: #F5F5F5; padding: 4px; }}")
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(6, 4, 6, 4)
        nav.setLayout(nav_layout)
        prev_btn = QPushButton("← Previous")
        prev_btn.setToolTip("Go back to previous triaged segment (re-review)")
        prev_btn.clicked.connect(self._on_previous_clicked)
        skip_btn = QPushButton("Skip without saving →")
        skip_btn.setToolTip("Skip this segment without recording a call; advance to next")
        skip_btn.clicked.connect(self._on_skip_clicked)
        save_btn = QPushButton("✓ Save && Next")
        save_btn.setStyleSheet("font-weight: bold; background: #28A745; color: white;")
        save_btn.setToolTip("Save this segment's call as cleared, then advance (Ctrl+S)")
        save_btn.clicked.connect(self._on_save_and_next_clicked)
        nav_layout.addWidget(prev_btn)
        nav_layout.addStretch()
        self._progress_label = QLabel("(no worklist loaded)")
        self._progress_label.setStyleSheet("font-size: 11px;")
        nav_layout.addWidget(self._progress_label)
        nav_layout.addStretch()
        nav_layout.addWidget(skip_btn)
        nav_layout.addWidget(save_btn)
        main_layout.addWidget(nav)

    def _refresh_banner_and_progress(self) -> None:
        """Update the banner + progress label based on current worklist entry."""
        if not self.worklist:
            return
        entry = self.worklist.current
        if entry is None:
            if self._banner_label:
                self._banner_label.setText("All triaged segments reviewed.")
            if self._progress_label:
                self._progress_label.setText("0 remaining")
            return
        if self._banner_label:
            txt = (
                f"<b>Video:</b> {entry.video_name} &nbsp; "
                f"<b>Segment:</b> {entry.segment_num} "
                f"(frames {entry.start_frame}-{entry.end_frame})<br>"
                f"<b>Algo flag reason:</b> {entry.triage_reason}"
            )
            if entry.already_cleared:
                txt += "<br><i>(previously cleared — re-review will overwrite)</i>"
            self._banner_label.setText(txt)
        if self._progress_label:
            self._progress_label.setText(self.worklist.progress_string())

    # ------------------------------------------------------------------
    # Entry loading
    # ------------------------------------------------------------------

    def load_current_entry(self) -> bool:
        """Load the video + frame range for the worklist's current entry.

        Returns True if loaded successfully, False if no current entry
        (e.g., worklist exhausted) or video missing.
        """
        if not self.worklist:
            return False
        entry = self.worklist.current
        if entry is None:
            self._on_worklist_exhausted()
            return False
        video_path = self.worklist.video_path_for(entry)
        if video_path is None:
            show_warning(f"Video not found for {entry.video_name}; skipping.")
            self.worklist.advance()
            return self.load_current_entry()
        frame_range = (
            max(0, entry.start_frame - self.pre_pad),
            entry.end_frame + self.post_pad,
        )
        self._load_video(video_path, frame_range=frame_range)
        # Ensure an OutcomeGT row exists for this segment so the reviewer
        # can pick the outcome class and stamp the interaction / known
        # frame via the inherited GT-tool outcome row. Triaged segments
        # usually have no GT entry; without this placeholder the outcomes
        # sidebar is empty and there's nowhere to click.
        self._ensure_outcome_placeholder(entry.segment_num)
        self._refresh_outcomes_list()
        # Clear notes for the new segment
        if self._notes_edit is not None:
            self._notes_edit.setPlainText("")
        self._refresh_banner_and_progress()
        # If already cleared, pre-populate notes from prior call
        if entry.already_cleared:
            self._prefill_prior_call(entry)
        return True

    def _algo_files_dir(self) -> Path:
        """Triage tool runs against quarantines where the video and the
        algo JSONs live in sibling directories (``videos/`` vs
        ``algo_outputs_current/``). Point saves at the worklist's
        algo_dir instead of the video's parent."""
        if self.worklist is not None:
            return self.worklist.algo_dir
        return super()._algo_files_dir()

    def _save_to_algo_files(self):
        """Override the inherited Review-Tool "✓ Save & Continue" button so
        it does triage clearing (with proper markers) AND advances the
        worklist -- not a raw file dump that leaves the reviewer on the
        same segment. Single canonical save path regardless of which
        save button the user clicks."""
        self._on_save_and_next_clicked()

    def _add_reach(self):
        """Override the GT-tool's reach adder so newly-added reaches are
        tagged with the CURRENT TRIAGE ENTRY's segment_num, not whichever
        segment the napari slider happens to land in. The triage tool's
        whole purpose is to clear one specific segment; the reviewer
        adding a reach is by definition adding the causal reach for THAT
        segment, even if the displayed frame falls in a neighboring
        boundary-defined slot."""
        super()._add_reach()
        if not self.worklist or self.worklist.current is None:
            return
        target_seg = int(self.worklist.current.segment_num)
        # The reach we just added is the new self._selected_reach (set by
        # the parent's _add_reach).
        new_reach = getattr(self, "_selected_reach", None)
        if new_reach is None:
            return
        if int(getattr(new_reach, "segment_num", -1)) == target_seg:
            return
        new_reach.segment_num = target_seg
        # Re-render the sidebar so the row moves to the right segment header.
        self._refresh_reaches_list()

    def _ensure_outcome_placeholder(self, segment_num: int) -> None:
        """If self.gt.outcomes has no entry for ``segment_num``, append a
        placeholder so the GT-tool outcome row renders. The placeholder
        starts as ``outcome="unknown"`` / not-determined; the reviewer's
        dropdown + interaction-frame button then mutate it normally."""
        if not self.gt:
            return
        from mousereach.review.unified_gt import OutcomeGT
        for o in (self.gt.outcomes or []):
            if int(getattr(o, "segment_num", -1)) == int(segment_num):
                return
        placeholder = OutcomeGT(
            segment_num=int(segment_num),
            outcome="unknown",
            determined=False,
        )
        if self.gt.outcomes is None:
            self.gt.outcomes = [placeholder]
        else:
            self.gt.outcomes.append(placeholder)

    def _prefill_prior_call(self, entry: TriageEntry) -> None:
        """If the segment is already cleared, load the prior call into UI."""
        # The unified GT loader inside _load_video populates self.gt with
        # any existing entries; we just pull the reviewer's prior notes
        # off the causal reach.
        if not self.gt:
            return
        # Find the reach in this segment that has a note
        for r in (self.gt.reaches or []):
            if getattr(r, "segment_num", None) == entry.segment_num:
                # Pull any note that was previously saved on the reach
                note = getattr(r, "review_note", None)
                if note and self._notes_edit is not None:
                    self._notes_edit.setPlainText(str(note))
                    break

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_save_and_next_clicked(self) -> None:
        ok = self._save_current_triage_segment()
        if ok:
            self.worklist.advance()
            if self.worklist.is_at_end():
                self._on_worklist_exhausted()
            else:
                self.load_current_entry()

    def _on_previous_clicked(self) -> None:
        self.worklist.retreat()
        self.load_current_entry()

    def _on_skip_clicked(self) -> None:
        self.worklist.advance()
        if self.worklist.is_at_end():
            self._on_worklist_exhausted()
        else:
            self.load_current_entry()

    def _on_worklist_exhausted(self) -> None:
        QMessageBox.information(
            self,
            "All caught up",
            f"All {len(self.worklist)} triaged segments have been reviewed.",
        )

    # ------------------------------------------------------------------
    # Save (per-segment, preserves other segments and writes triage fields)
    # ------------------------------------------------------------------

    def _save_current_triage_segment(self) -> bool:
        """Save the reviewer's call for the current entry to algo + GT JSONs.

        Per-segment scope: only updates segments[i] where i==current entry's
        segment_num. Other segments are preserved unchanged.

        Schema additions on save:
          - segments[i].flagged_for_review = False
          - segments[i].triage_cleared = True
          - segments[i].original_triage_reason = <orig flag_reason>
          - segments[i].human_verified = True (on outcomes)
          - segments[i].cleared_by = <username>, cleared_at = <timestamp>
          - Causal reach: human_corrected=True, review_note=<notes>
          - Other auto-detected reaches in segment: exclude_from_analysis=True,
            exclude_reason="not causal per human review"
        """
        if not self.worklist or not self.gt or not self.video_path:
            show_warning("No segment loaded.")
            return False
        entry = self.worklist.current
        if entry is None:
            return False
        # Reviewer must have set at least the causal reach for this segment
        causal = self._find_causal_reach_for_segment(entry.segment_num)
        if causal is None or not causal.start_determined or not causal.end_determined:
            show_warning(
                "Mark the causal reach (S = start frame, E = end frame) before saving."
            )
            return False
        outcome_gt = self._find_outcome_for_segment(entry.segment_num)
        if outcome_gt is None or not outcome_gt.determined:
            show_warning(
                "Set the outcome category (and interaction frame if applicable) "
                "before saving."
            )
            return False

        username = get_username()
        timestamp = get_timestamp()
        notes = self._notes_edit.toPlainText().strip() if self._notes_edit else ""
        parent = self._algo_files_dir()
        video_stem = self.video_path.stem
        if "DLC_" in video_stem:
            video_stem = video_stem.split("DLC_")[0]

        # === Update _reaches.json (per-segment scope) ===
        reaches_path = parent / f"{video_stem}_reaches.json"
        if reaches_path.exists():
            try:
                with open(reaches_path, "r") as f:
                    reach_data = json.load(f)
                for seg in reach_data.get("segments", []):
                    if seg.get("segment_num") != entry.segment_num:
                        continue
                    orig_reason = seg.get("flag_reason")
                    seg["flagged_for_review"] = False
                    if orig_reason is not None:
                        seg["original_triage_reason"] = orig_reason
                    seg["triage_cleared"] = True
                    seg["cleared_by"] = username
                    seg["cleared_at"] = timestamp
                    # Mark all reaches in the segment as missed by default,
                    # then mark the reviewer's causal reach with the correction.
                    for reach in seg.get("reaches", []):
                        if reach.get("reach_id") == causal.reach_id:
                            reach["start_frame"] = causal.start_frame
                            reach["end_frame"] = causal.end_frame
                            reach["human_corrected"] = True
                            reach["review_note"] = notes
                            reach["exclude_from_analysis"] = False
                            reach["exclude_reason"] = None
                        else:
                            reach["exclude_from_analysis"] = True
                            reach["exclude_reason"] = "not causal per human review"
                            reach["human_corrected"] = True
                    break
                with open(reaches_path, "w") as f:
                    json.dump(reach_data, f, indent=2)
            except Exception as e:
                show_warning(f"Failed to save reaches: {e}")
                return False

        # === Update _pellet_outcomes.json (per-segment scope) ===
        outcomes_path = parent / f"{video_stem}_pellet_outcomes.json"
        if outcomes_path.exists():
            try:
                with open(outcomes_path, "r") as f:
                    outcome_data = json.load(f)
                for seg in outcome_data.get("segments", []):
                    if seg.get("segment_num") != entry.segment_num:
                        continue
                    orig_reason = seg.get("flag_reason")
                    orig_outcome = seg.get("outcome")
                    seg["flagged_for_review"] = False
                    seg["triage_cleared"] = True
                    seg["human_verified"] = True
                    seg["cleared_by"] = username
                    seg["cleared_at"] = timestamp
                    if orig_reason is not None:
                        seg["original_triage_reason"] = orig_reason
                    if orig_outcome and orig_outcome != outcome_gt.outcome:
                        seg["original_outcome"] = orig_outcome
                    seg["outcome"] = outcome_gt.outcome
                    if outcome_gt.interaction_frame is not None:
                        seg["interaction_frame"] = outcome_gt.interaction_frame
                    if outcome_gt.outcome_known_frame is not None:
                        seg["outcome_known_frame"] = outcome_gt.outcome_known_frame
                    seg["causal_reach_id"] = causal.reach_id
                    break
                with open(outcomes_path, "w") as f:
                    json.dump(outcome_data, f, indent=2)
            except Exception as e:
                show_warning(f"Failed to save outcomes: {e}")
                return False

        # === Update unified GT JSON ===
        try:
            save_unified_gt(self.gt, self._unified_gt_path_for_video())
        except Exception as e:
            show_warning(f"Failed to save unified GT: {e}")

        show_info(
            f"Cleared {entry.video_name} segment {entry.segment_num}: "
            f"{outcome_gt.outcome} (causal reach {causal.reach_id})"
        )
        # Reflect cleared state on the entry so the worklist doesn't re-flag it
        entry.already_cleared = True
        return True

    def _find_causal_reach_for_segment(self, segment_num: int):
        """Find the reach the reviewer marked for this segment.

        The reviewer marks one reach as the causal reach via S/E shortcuts.
        Convention: the most-recently-determined reach in this segment is
        the causal one.
        """
        if not self.gt or not self.gt.reaches:
            return None
        candidates = [
            r for r in self.gt.reaches
            if getattr(r, "segment_num", None) == segment_num
            and getattr(r, "start_determined", False)
            and getattr(r, "end_determined", False)
        ]
        if not candidates:
            return None
        # Pick by most-recent determined_at if available
        try:
            return max(candidates, key=lambda r: getattr(r, "start_determined_at", "") or "")
        except Exception:
            return candidates[0]

    def _find_outcome_for_segment(self, segment_num: int):
        if not self.gt or not self.gt.outcomes:
            return None
        for o in self.gt.outcomes:
            if getattr(o, "segment_num", None) == segment_num:
                return o
        return None

    def _unified_gt_path_for_video(self) -> Path:
        """Resolve unified GT path for the current video, matching the
        quarantine layout (sibling gt/) if present.
        """
        parent = self.video_path.parent
        gt_sibling = parent.parent / "gt"
        if gt_sibling.is_dir():
            return gt_sibling / f"{self.video_path.stem}_unified_ground_truth.json"
        return parent / f"{self.video_path.stem}_unified_ground_truth.json"


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------

def main():
    """Launch the triage clearing tool (CLI: ``mousereach-review-tool``).

    Default: scan latest quarantine for triaged segments, walk through them
    one at a time. ``--algo-dir`` overrides; ``--video-name`` restricts to
    one video; ``--include-cleared`` allows re-reviewing already-cleared
    segments.
    """
    import argparse
    import napari
    from .triage_queue import find_default_algo_dir

    parser = argparse.ArgumentParser(
        description=(
            "MouseReach Triage Clearing Tool — review and resolve algo-flagged "
            "segments one at a time."
        ),
    )
    parser.add_argument(
        "--algo-dir", type=Path, default=None,
        help="Directory containing *_reaches.json / *_pellet_outcomes.json. "
             "Defaults to the latest quarantine under "
             "CONNECTOME_ROOT/Behavior/MouseReach_Improvement/validation_runs/DLC_*/iterations/*/algo_outputs/.",
    )
    parser.add_argument(
        "--video-name", type=str, default=None,
        help="Restrict worklist to this video stem (e.g. 20250624_CNT0107_P3).",
    )
    parser.add_argument(
        "--include-cleared", action="store_true",
        help="Include already-cleared segments in the worklist (for re-review).",
    )
    parser.add_argument(
        "--pre-pad", type=int, default=30,
        help="Frames to load before each segment's start (context). Default 30.",
    )
    parser.add_argument(
        "--post-pad", type=int, default=30,
        help="Frames to load after each segment's end (context). Default 30.",
    )
    args = parser.parse_args()

    algo_dir = args.algo_dir or find_default_algo_dir()
    if algo_dir is None:
        print("No algo_dir specified and no default quarantine found.")
        print("Set MOUSEREACH_TRIAGE_ALGO_DIR or pass --algo-dir.")
        return 1
    if not algo_dir.is_dir():
        print(f"algo_dir not found: {algo_dir}")
        return 1

    video_filter = [args.video_name] if args.video_name else None
    worklist = TriageWorklist.from_algo_dir(
        algo_dir,
        video_filter=video_filter,
        include_cleared=args.include_cleared,
    )
    if len(worklist) == 0:
        print(f"No triaged segments pending review in {algo_dir}.")
        print("If you expect there to be some, try --include-cleared to re-review.")
        return 0

    print(
        f"Loaded triage worklist: {len(worklist)} segments across "
        f"{len(worklist.videos_covered())} videos."
    )
    print(f"First entry: {worklist.current.video_name} seg {worklist.current.segment_num}")

    viewer = napari.Viewer(title="MouseReach Triage Clearing")
    widget = TriageClearingWidget(
        viewer, worklist=worklist,
        pre_pad=args.pre_pad, post_pad=args.post_pad,
    )
    viewer.window.add_dock_widget(widget, name="Triage Clearing", area="right")
    widget.load_current_entry()
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
