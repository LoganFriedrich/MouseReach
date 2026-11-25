"""Fix a video's segment boundaries, and nothing else.

WHY THIS EXISTS
---------------
The segmenter always emits exactly 21 boundaries. That count is hard-coded and
forced by a safety net, so downstream sees a complete, plausible segmentation
whatever actually happened -- boundaries invented at the median cadence, real
detections discarded to fit the count, or an even grid when reference tracking
failed. Nothing about the output says which.

The damage is not that a segment is roughly wrong. It is that the NUMBERING
shifts: if the segmenter misses one tray advance, everything after it is called
by the wrong number, and the pellet a person scored as number 7 on the bench gets
compared against footage of pellet 8. Reviewers hitting this found the algorithm
right every time and the numbering wrong -- drifting by one, then two, through a
single video.

The segmenter already knows where the alternatives are. It proposes candidate
timepoints from four tray corners, merges them by agreement, then keeps 21 and
throws the rest away. Those discarded candidates are exactly what a person needs
to see, so they are now saved and this tool puts them in front of you.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No outcomes, no reaches, no causal attribution. Segmentation only. Getting the
cuts right is a different job from judging what happened between them, and mixing
them makes both slower. When you are done here the video goes back through the
pipeline and the other tools ask their own questions.

USAGE
-----
  mousereach-fix-segmentation
  mousereach-fix-segmentation --queue-dir &lt;dir&gt;   # default: the deep-review queue

ASCII-only console output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import logging
logger = logging.getLogger(__name__)

import numpy as np


# Archive-then-remove everything computed from the old cuts. It lives in the
# Qt-free hand_cuts module, which every cut editor saves through; imported here
# under its old name for the callers that import it from this module.
from .hand_cuts import (invalidate_stale_downstream, mark_as_hand_cuts,  # noqa: E402
                        commit_hand_cuts, HandCutsNotSaved, BOUNDARY_CALLS_KEY,
                        release_after_hand_cuts, asked_again_note)


from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QGroupBox, QMessageBox,
    QCheckBox, QSpinBox, QRadioButton, QButtonGroup, QGridLayout,
)

from .small_screen import (CollapsibleSection, fit_review_window,
                           scrollable_body, wrap_all_labels)

# Candidate chips per row in the guided walk (a grid, so five chips never
# make the panel wider than a small screen allows).
CHIP_COLUMNS = 3

from .keyboard import keep_typing_in_panel, review_keys_on_layer

# How close a proposed boundary has to be to an existing one to count as "the
# same boundary". The segmenter merges candidates within 30 frames, so anything
# inside that is the same tray advance seen twice.
SAME_BOUNDARY_FRAMES = 30


def read_segmentation(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def needs_fixing(seg: dict) -> List[str]:
    """The segmenter's own reasons this video wants a person, if any."""
    return list((seg or {}).get("needs_human") or [])


def segments_from(boundaries: List[int], total_frames: int) -> List[tuple]:
    """(segment_num, start, end) pairs, the way outcome detection reads them."""
    b = sorted(int(x) for x in boundaries)
    out = []
    for i in range(len(b) - 1):
        out.append((i + 1, b[i], b[i + 1] - 1))
    return out


# How far the algo's boundary may be from the human's judgment before the
# guided walk counts it as wrong and asks for the real frame.
GUIDED_TOLERANCE_FRAMES = 10


def move_segment_start(boundaries: List[int], seg_idx: int, frame: int) -> List[int]:
    """New boundary list with segment ``seg_idx`` (0-based) STARTING at
    ``frame``. A segment's start IS its opening boundary, so this moves
    boundary[seg_idx]. Pure -- returns a new sorted list."""
    b = sorted(int(x) for x in boundaries)
    if not (0 <= seg_idx < len(b) - 1):
        return b
    b[seg_idx] = int(frame)
    return sorted(b)


def move_segment_end(boundaries: List[int], seg_idx: int, frame: int) -> List[int]:
    """New boundary list with segment ``seg_idx`` (0-based) ENDING at
    ``frame``. A segment ends one frame before the NEXT boundary, so this
    moves boundary[seg_idx + 1] to frame + 1. Pure -- returns a new sorted
    list."""
    b = sorted(int(x) for x in boundaries)
    if not (0 <= seg_idx < len(b) - 1):
        return b
    b[seg_idx + 1] = int(frame) + 1
    return sorted(b)


class FixSegmentationWidget(QWidget):
    """Accept or reject the segmenter's candidate timepoints, and save the cuts."""

    def __init__(self, napari_viewer, queue_dir: Path):
        super().__init__()
        self.viewer = napari_viewer
        self.queue_dir = Path(queue_dir)

        self.video_stem: Optional[str] = None
        self.seg_path: Optional[Path] = None
        self.seg: dict = {}
        self.candidates: List[dict] = []
        self.boundaries: List[int] = []
        self.original_boundaries: List[int] = []
        self.n_frames: int = 0
        self._uncut: bool = False          # no segments file: cut from scratch
        self._pose_frames: Optional[int] = None  # frames in the pose file shown
        self._video_layer = None
        self._queue: List[Path] = []
        self._gw_idx: int = 0
        # What the person did with each cut, keyed by the cut's CURRENT frame
        # (never by its position in the list, which shifts when a cut moves
        # past another, is removed or is added). A cut is called once it has
        # an entry here; Save refuses while any cut has none (owner decision
        # MRR-08d, hand_cuts.BOUNDARY_CALLS_KEY).
        self._calls: Dict[int, dict] = {}
        self._removed_calls: List[dict] = []

        from qtpy.QtCore import QTimer
        self.is_playing = False
        self.playback_direction = 1
        self.playback_speed = 1.0
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)

        self._build_ui()
        self._load_queue()

    # ---------------------------------------------------------------- ui

    QUICK_GUIDE = """
This tool fixes ONE thing: where a video's segment boundaries are. A
boundary is the first frame after the scoring area jumps (the tray finishing
an advance). Boundary 1 ends the pre-roll (the pre-pellet setup footage) and
begins segment 1 / pellet 1; boundary N ends segment N-1 and begins segment
N. The DLC points are drawn on the video -- watch the scoring-area points
jump to see where a boundary really is. The WHOLE video is loaded, pre-roll
and post-roll included: step backwards from boundary 1 to watch the pre-roll
end.

THE QUICK WAY -- the box pinned at the top ("This boundary"):
it always shows the current boundary as "Boundary N of M at frame X. Is
this where the tray finished moving?" with three buttons:
* "Yes, correct" -- records Yes and loads the next boundary.
* "Move it to the current frame" -- step the video to the first frame
  after the tray finished moving, then press it: the boundary moves there
  and the next one loads.
* "No boundary here" -- the segmenter invented it: it is removed and later
  segments renumber.
"Save these cuts" is pinned at the bottom. Everything between scrolls.
The two optional sections ("Manual cut editing", "Segments these cuts
produce") start folded: click their titles to open them.

THE WORKFLOW IN DETAIL -- one question per boundary (the Guided walk box):

* The walk parks the video on each boundary in turn and asks: "Is this the
  boundary?" (is this frame the first one after the jump, within 10 frames).
  The header states explicitly which pellet is BEFORE this boundary and
  which is AFTER it -- check the footage on both sides matches that claim.
* Yes is pre-selected -- just press Confirm when it is right.
* If it is elsewhere: click a candidate chip (jumps there AND uses it as the
  answer), or scrub to the right frame and press "Use current frame", or
  type the frame number.
* If there is NO tray advance anywhere near here, press "No advance here --
  remove this boundary". Every later segment renumbers automatically.
* Press "Confirm -> next boundary". The walk goes on to the next boundary
  nobody has answered. Repeat to the end, then "Save these cuts". Saving
  stamps the boundaries as human-made; the pipeline keeps them on the re-run.

EVERY BOUNDARY MUST BE ANSWERED BEFORE SAVING: "Save these cuts" refuses,
and says which frames are left, until every boundary has been accepted
(Confirm), moved, removed, or added by you. A cut you add by hand counts as
answered. A video saved before this rule with boundaries left unanswered is
listed here again; the boundaries already answered keep their answer and
the walk starts at boundary 1. "Back to the algorithm's cuts" clears every
answer.

IF A BOUNDARY WAS MISSED (a segment looks double-length in the table, or
you saw an extra jump): add it with "Add a cut at the current frame" in the
manual controls below -- it counts as answered by you.

EVERYTHING BELOW THE QUESTION IS OPTIONAL. The candidate table and the
add/drop-cut buttons are manual controls; the segment table just shows what
the current boundaries produce -- red rows hint at a missing boundary. If
you never need them, Confirm at every boundary + Save is the whole job.

NAVIGATION (identical to the other review tools, plus segment jumps):
Play/Rev/Stop with speed buttons; step buttons for 1/10/100 frames; <seg and
seg> jump ONE SEGMENT LENGTH (this video's median boundary spacing) -- the
fast way to chase an offset that shifted every later segment; "Go to" jumps
to a typed frame. Keyboard: Space = play/pause, b = reverse, arrows = 1
frame, Shift-arrows = 10, Ctrl-arrows = 100, Alt-arrows = 1 segment,
keys 1-6 = speed. The keys work while the video has the keyboard -- click
the video first. Anything typed while you are working in this panel stays in
the panel and never runs a key.

WHICH VIDEOS ARE LISTED: every video in the queue that cannot leave until a
person sets or confirms its cuts, and nobody has since it arrived -- whatever
sent it (bad cuts, a flagged session, a quality hold), and also a video whose
cuts were fixed on an earlier visit and that was sent back. The orange line
under the video name says why it was sent and why it cannot leave yet. If
the cuts already look right, answer Yes at every boundary and Save: confirming
them is the step the queue is waiting for.

A VIDEO WITH NO CUTS AT ALL: when the segmenter failed with an error, the
video arrives with no segments file. It is listed after the others, with the
note "The segmenter could not run on this video". Add every cut by hand with
"Add a cut at the current frame" at each tray advance, then Save: that writes
its first segments file. If neither its pose file nor its video can be read,
its length is unknown and Save says so and writes nothing.

GROUND TRUTH DOES NOT COUNT HERE: a ground-truth file with every boundary
determined does not release a video or stand in for this step. Set or confirm
the cuts here even when the video has ground truth.

FINISHING: saving releases a video that was sent here for its cuts. Reach
detection, outcomes and reach assignment then run again on the new cuts. An
answer a person gave on the old cuts stands on a pellet whose frames did not
move; on a pellet whose frames moved it is dropped, and that pellet takes the
re-run algorithms' call -- it goes to triage only if they flag it, like any
pellet of a new video. A video
sent for another reason (a flagged session, a quality hold) is released once
its cuts are saved: by the "Release finished videos" button on the Review
Queues tab if every pellet is already answered, otherwise by answering its
open pellets in Re-segmentation Review (it releases on the save that answers
the last one, or with "Clear -> re-enter pipeline"). Everything after is
automatic.

"Skip this video" moves on without saving anything.
"""

    def _build_ui(self):
        # Small-screen layout (review/small_screen.py): everything but the
        # Save/Skip row and the status line sits in a scroll area, so the
        # panel never needs more height or width than the screen gives it;
        # the Save row stays pinned at the bottom, always in reach.
        layout = scrollable_body(self)
        outer = self.layout()
        self._scroll_body = layout.parentWidget()

        head_row = QHBoxLayout()
        self.header = QLabel("No video loaded")
        self.header.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.header.setWordWrap(True)
        head_row.addWidget(self.header, 1)
        from mousereach.review.help_button import attach_help
        attach_help(head_row, "Re-segmentation", self.QUICK_GUIDE, self)
        layout.addLayout(head_row)

        intro = QLabel(
            "Fix WHERE this video's segment cuts are -- nothing else. Answer "
            "the three questions per segment, Confirm, repeat, then 'Save "
            "these cuts'. Everything below the questions is optional manual "
            "control. Click the ? for the full guide.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #888;")
        layout.addWidget(intro)

        self.why = QLabel("")
        self.why.setWordWrap(True)
        self.why.setStyleSheet("color: #e08020;")
        layout.addWidget(self.why)

        # --- video navigation: SAME controls and keys as the causal/triage
        # review tool, so operators never re-learn navigation between tools.
        play_row = QHBoxLayout()
        self._play_rev_btn = QPushButton("Rev")
        self._play_rev_btn.setToolTip("Play backwards (keyboard: b)")
        self._play_rev_btn.setMaximumWidth(40)
        self._play_rev_btn.clicked.connect(self._play_reverse)
        play_row.addWidget(self._play_rev_btn)
        self._play_btn = QPushButton("Play")
        self._play_btn.setToolTip("Play forwards (keyboard: Space toggles)")
        self._play_btn.setMaximumWidth(40)
        self._play_btn.clicked.connect(self._play_forward)
        play_row.addWidget(self._play_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setToolTip("Stop playback")
        self._stop_btn.setMaximumWidth(40)
        self._stop_btn.clicked.connect(self._stop_play)
        play_row.addWidget(self._stop_btn)
        play_row.addStretch()
        play_row.addWidget(QLabel("Speed:"))
        self._speed_buttons = {}
        for speed in [0.25, 0.5, 1, 2, 4, 8]:
            label = ("%sx" % speed) if speed < 1 else ("%dx" % int(speed))
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMaximumWidth(35)
            b.setToolTip("Playback speed (keyboard: 1-6)")
            b.clicked.connect(lambda _=False, s=speed: self._set_speed(s))
            self._speed_buttons[speed] = b
            play_row.addWidget(b)
        self._speed_buttons[1].setChecked(True)
        layout.addLayout(play_row)

        step_row = QHBoxLayout()
        segb = QPushButton("<seg")
        segb.setMaximumWidth(42)
        segb.setToolTip("Jump ONE SEGMENT LENGTH backwards (the median "
                        "boundary-to-boundary interval of this video, "
                        "~30s of tray cadence). Keyboard: Alt-Left. For "
                        "chasing an offset that shifted every later segment.")
        segb.clicked.connect(lambda: self._jump_frames(-self._segment_length()))
        step_row.addWidget(segb)
        for delta, label in [(-100, "<<"), (-10, "<"), (-1, "-1"),
                             (1, "+1"), (10, ">"), (100, ">>")]:
            b = QPushButton(label)
            b.setMaximumWidth(35)
            b.setToolTip("Step %+d frame(s) (keyboard: arrows = 1, "
                         "Shift-arrows = 10, Ctrl-arrows = 100)" % delta)
            b.clicked.connect(lambda _=False, d=delta: self._jump_frames(d))
            step_row.addWidget(b)
        segf = QPushButton("seg>")
        segf.setMaximumWidth(42)
        segf.setToolTip("Jump ONE SEGMENT LENGTH forwards (the median "
                        "boundary-to-boundary interval of this video). "
                        "Keyboard: Alt-Right.")
        segf.clicked.connect(lambda: self._jump_frames(self._segment_length()))
        step_row.addWidget(segf)
        step_row.addStretch()
        layout.addLayout(step_row)
        # Frame/time readout and Go-to on their own row: on one row with the
        # step buttons they were wider than a small screen's panel.
        step_row = QHBoxLayout()
        self._frame_label = QLabel("Frame: -- / --")
        step_row.addWidget(self._frame_label)
        self._time_label = QLabel("Time: --:--")
        step_row.addWidget(self._time_label)
        step_row.addStretch()
        step_row.addWidget(QLabel("Go to:"))
        self._goto_spin = QSpinBox()
        self._goto_spin.setRange(0, 10_000_000)
        self._goto_spin.setToolTip("Type a frame number and press Go")
        step_row.addWidget(self._goto_spin)
        gb = QPushButton("Go")
        gb.setMaximumWidth(36)
        gb.setToolTip("Jump the video to the typed frame")
        gb.clicked.connect(lambda: self._goto(int(self._goto_spin.value())))
        step_row.addWidget(gb)
        layout.addLayout(step_row)


        # --- guided walk: one question per BOUNDARY ------------------------
        gw_box = QGroupBox("Guided walk (one boundary at a time)")
        gv = QVBoxLayout()
        gw_box.setLayout(gv)

        self.gw_header = QLabel("")
        self.gw_header.setStyleSheet("font-weight: bold;")
        self.gw_header.setWordWrap(True)
        gv.addWidget(self.gw_header)

        # The question on its own line, the answer controls under it (side
        # by side they did not fit a small screen's panel).
        self.gw_q = QLabel("")
        self.gw_q.setWordWrap(True)
        gv.addWidget(self.gw_q)
        r = QHBoxLayout()
        self.gw_yes = QRadioButton("Yes")
        self.gw_yes.setChecked(True)
        self.gw_yes.setToolTip("This frame IS the boundary (the first frame "
                               "after the jump, within 10 frames).")
        self.gw_no = QRadioButton("No, it is at frame:")
        self.gw_no.setToolTip(
            "The jump is elsewhere. Give the first frame AFTER it -- via a "
            "candidate chip, the playhead, or typing.")
        g = QButtonGroup(self)
        g.addButton(self.gw_yes); g.addButton(self.gw_no)
        r.addWidget(self.gw_yes)
        r.addWidget(self.gw_no)
        self.gw_spin = QSpinBox()
        self.gw_spin.setRange(0, 10_000_000)
        self.gw_spin.setEnabled(False)
        self.gw_no.toggled.connect(self.gw_spin.setEnabled)
        # Choosing "No" means a frame gets typed next: give the box the
        # keyboard so the number lands there, not on the video's keys
        # (review/keyboard.py). clicked runs after toggled has enabled the box.
        self.gw_no.clicked.connect(
            lambda: (self.gw_spin.setFocus(), self.gw_spin.selectAll()))
        r.addWidget(self.gw_spin)
        b = QPushButton("Use current frame")
        b.setToolTip("Copy the playhead's frame into the box. Pick the first "
                     "frame AFTER the scoring-area jump.")
        b.clicked.connect(lambda: self._gw_use_playhead(self.gw_spin, self.gw_no))
        r.addWidget(b)
        gv.addLayout(r)

        note = QLabel("A boundary is the FIRST frame AFTER the scoring-area "
                      "jump (the tray has finished advancing). Step backwards "
                      "to watch the jump happen.")
        note.setStyleSheet("color: #888;")
        note.setWordWrap(True)
        gv.addWidget(note)

        # Nearby candidate tray advances the segmenter itself proposed --
        # click one to jump the playhead there AND fill it in as the answer.
        # The chips sit in a grid (CHIP_COLUMNS across) under their label:
        # five in one row were wider than a small screen's panel.
        gv.addWidget(QLabel("Segmenter's nearby candidates:"))
        self.gw_cands = QGridLayout()
        self.gw_cands.setHorizontalSpacing(4)
        self.gw_cands.setVerticalSpacing(2)
        gv.addLayout(self.gw_cands)

        nav = QHBoxLayout()
        bb = QPushButton("< Back a boundary")
        bb.setToolTip("Revisit the previous boundary.")
        bb.clicked.connect(self._gw_back)
        nav.addWidget(bb)
        nav.addStretch()
        cb = QPushButton("Confirm -> next boundary")
        cb.setStyleSheet("font-weight: bold;")
        cb.setToolTip("Record the answer (moving the boundary if you said "
                      "No) and load the next one.")
        cb.clicked.connect(self._gw_confirm)
        nav.addWidget(cb)
        gv.addLayout(nav)
        # The remove button on its own row: with Back and Confirm beside it
        # the row was wider than a small screen's panel.
        nav = QHBoxLayout()
        rmb = QPushButton("No advance here -- remove this boundary")
        rmb.setStyleSheet("background: #5a1616; color: white;")
        rmb.setToolTip(
            "There is no tray advance anywhere near this frame: the "
            "segmenter invented this boundary. Removing it merges the two "
            "segments it separated, and every later segment renumbers.")
        rmb.clicked.connect(self._gw_remove_boundary)
        nav.addWidget(rmb)
        nav.addStretch()
        gv.addLayout(nav)

        self.gw_status = QLabel("")
        self.gw_status.setWordWrap(True)
        self.gw_status.setStyleSheet("color: #e08020;")
        layout.addWidget(gw_box)

        # --- the current boundary's question, PINNED at the top -----------
        # Outside the scroll area, so the question and its answer buttons are
        # always on screen together, whatever is scrolled below. The three
        # buttons are shortcuts over the guided walk's own answers (same
        # records, same moves); the full controls stay in the box below for
        # typing a frame or picking a candidate.
        quick = QGroupBox("This boundary")
        qv = QVBoxLayout()
        qv.setContentsMargins(6, 4, 6, 4)
        quick.setLayout(qv)
        self.gw_plain = QLabel("")
        self.gw_plain.setWordWrap(True)
        self.gw_plain.setStyleSheet("font-weight: bold; font-size: 13px;")
        qv.addWidget(self.gw_plain)
        qrow = QHBoxLayout()
        yb = QPushButton("Yes, correct")
        yb.setStyleSheet("font-weight: bold; background: #2d5016; color: white;")
        yb.setToolTip("The tray finished moving at this frame (within %d "
                      "frames). Records Yes and goes to the next boundary."
                      % GUIDED_TOLERANCE_FRAMES)
        yb.clicked.connect(self._gw_quick_yes)
        qrow.addWidget(yb)
        mb = QPushButton("Move it to the current frame")
        mb.setToolTip("Step the video to the first frame AFTER the tray "
                      "finished moving, then press this: the boundary moves "
                      "to that frame and the next boundary loads.")
        mb.clicked.connect(self._gw_quick_move)
        qrow.addWidget(mb)
        nb = QPushButton("No boundary here")
        nb.setStyleSheet("background: #5a1616; color: white;")
        nb.setToolTip(
            "There is no tray advance anywhere near this frame: the "
            "segmenter invented this boundary. Removing it merges the two "
            "segments it separated, and every later segment renumbers.")
        nb.clicked.connect(self._gw_remove_boundary)
        qrow.addWidget(nb)
        qrow.addStretch()
        qv.addLayout(qrow)
        qv.addWidget(self.gw_status)
        outer.insertWidget(0, quick)

        # --- candidates ---------------------------------------------------
        # Optional, so it folds away (closed by default): on a small screen the
        # guided walk and the video come first. Click its title to open it.
        cand_box = CollapsibleSection("Manual cut editing (OPTIONAL)")
        self.cand_section = cand_box
        cv = QVBoxLayout()
        cv.setContentsMargins(4, 0, 0, 0)
        cand_box.content.setLayout(cv)
        sub = QLabel("For fixes the questions above cannot express.")
        sub.setStyleSheet("color: #888;")
        cv.addWidget(sub)
        cv.addWidget(QLabel(
            "Every timepoint the corner trackers proposed; 'yes' rows are the "
            "current cuts. Click a row to jump the video there. Use this when "
            "a cut is MISSING or EXTRA (e.g. a segment-number offset) -- the "
            "questions above only move existing cuts."))
        self.cand_table = QTableWidget(0, 5)
        self.cand_table.setHorizontalHeaderLabels(
            ["cut?", "frame", "time", "corners agreeing", "agreement"])
        self.cand_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cand_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cand_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cand_table.itemSelectionChanged.connect(self._jump_to_selected)
        self.cand_table.setMinimumHeight(180)
        cv.addWidget(self.cand_table)

        # Two rows of buttons: one row was wider than a small screen's panel.
        row = QHBoxLayout()
        row2 = QHBoxLayout()
        for i, (text, slot, tip) in enumerate((
                ("Use this candidate", self._use_selected,
                 "Add the selected table row's timepoint as a new cut."),
                ("Drop this cut", self._drop_selected,
                 "Remove the cut at/near the selected table row's timepoint."),
                ("Add a cut at the current frame", self._add_here,
                 "Add a cut exactly where the video playhead is now -- for a "
                 "tray advance the segmenter never proposed."))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            (row if i < 2 else row2).addWidget(b)
        self.only_unused = QCheckBox("Hide candidates already used")
        self.only_unused.setToolTip(
            "Show only proposed timepoints that are NOT currently cuts.")
        self.only_unused.toggled.connect(lambda _: self._refresh_candidates())
        row2.addWidget(self.only_unused)
        row.addStretch()
        row2.addStretch()
        cv.addLayout(row)
        cv.addLayout(row2)
        layout.addWidget(cand_box)

        # --- resulting segments -------------------------------------------
        # Read-only, so it folds away too (closed by default).
        seg_box = CollapsibleSection("Segments these cuts produce (READ-ONLY)")
        self.seg_section = seg_box
        sv = QVBoxLayout()
        sv.setContentsMargins(4, 0, 0, 0)
        seg_box.content.setLayout(sv)
        sub = QLabel("A check -- nothing to fill in.")
        sub.setStyleSheet("color: #888;")
        sv.addWidget(sub)
        sv.addWidget(QLabel(
            "A tray advance comes about every 30 seconds. A segment far from "
            "that is where a cut is missing or wrong."))
        legend = QLabel(
            "What a boundary means: boundary 1 is the START of segment 1 (= "
            "pellet 1) and the end of the pre-pellet setup frames; boundary 2 "
            "is the end of segment 1 and the start of segment 2; and so on -- "
            "each cut sits on the first frame AFTER a scoring-area jump.")
        legend.setStyleSheet("color: #888;")
        legend.setWordWrap(True)
        sv.addWidget(legend)
        self.seg_table = QTableWidget(0, 4)
        self.seg_table.setHorizontalHeaderLabels(
            ["segment", "starts at frame", "length (s)", ""])
        self.seg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.seg_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.seg_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.seg_table.itemSelectionChanged.connect(self._jump_to_segment)
        self.seg_table.setMinimumHeight(180)
        sv.addWidget(self.seg_table)
        layout.addWidget(seg_box)
        layout.addStretch()

        # Every label in the scrolling part wraps to the panel's width.
        wrap_all_labels(self._scroll_body)

        # --- actions (pinned under the scroll area) -----------------------
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("font-weight: bold;")
        self.count_label.setWordWrap(True)
        outer.addWidget(self.count_label)
        act = QHBoxLayout()
        for text, slot, style, tip in (
                ("Back to the algorithm's cuts", self._reset, "",
                 "Throw away every change on this video and restore the "
                 "algorithm's original cuts."),
                ("Save these cuts", self._save, "font-weight: bold;",
                 "Write the cuts (stamped human-made, kept by the pipeline) "
                 "and load the next video. A video sent here for its cuts is "
                 "released by this save; one sent for another reason is "
                 "released by the Clear button in Re-segmentation Review."),
                ("Skip this video", self._next_video, "",
                 "Move on WITHOUT saving anything on this video; it stays in "
                 "the queue.")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            b.setStyleSheet(style)
            act.addWidget(b)
        act.addStretch()
        outer.addLayout(act)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

    # ------------------------------------------------------------- queue

    def _needs_reseg(self, bundle: Path, seg: dict) -> bool:
        """Does this bundle belong in the re-segmentation queue?

        Exactly when it cannot leave the queue until a person sets or confirms
        its cuts, and nobody has done so since it arrived
        (resegmented.needs_recut). That is the proof every way out of the
        queue demands of a pillar video, whatever brought it here: the
        segmenter's needs_human verdict, a reviewer's mislabel, the triage
        escalate button, a flagged session or a quality hold.

        WHY not the old three-ways-in test: it listed only bundles routed for
        a segmentation reason whose cuts had never been set by hand. A video
        whose cuts were hand-fixed on an earlier visit and then sent back, and
        a video sent here for a flagged session, were never listed, yet could
        not leave without new cuts -- no tool offered them any way out.
        ``seg`` is unused; kept so callers need not change."""
        from mousereach.review.resegmented import needs_recut
        return needs_recut(bundle, bundle.name)[0]

    @staticmethod
    def _uncut_bundle(bundle: Path) -> bool:
        """A queue folder that is a video sent here with no segments file: a
        Pillar tray (Easy and Flat are never segmented; Clear releases them),
        with a routing record or review manifest saying a route put it here,
        and not a hidden folder or a leftover kept for a person."""
        from mousereach.config import is_outcome_free_tray
        from mousereach.watcher.review_routing import LEFT_NOTE
        stem = bundle.name
        if stem.startswith(".") or is_outcome_free_tray(stem):
            return False
        if (bundle / LEFT_NOTE).exists():
            return False
        return any((bundle / n).is_file() for n in (
            "%s_routing.json" % stem, "%s_manifest.json" % stem, "manifest.json"))

    def _load_queue(self):
        """Videos that cannot leave the queue until a person sets or confirms
        their cuts (see _needs_reseg)."""
        self._queue = []
        if not self.queue_dir.is_dir():
            self.header.setText("Queue not found: %s" % self.queue_dir)
            return
        for bundle in sorted(self.queue_dir.iterdir()):
            if not bundle.is_dir():
                continue
            sp = bundle / ("%s_segments.json" % bundle.name)
            if not sp.is_file():
                # No segments file: the segmenter could not run on this video
                # and the watcher sent it here to be cut by hand. WHY listed:
                # every way out of the queue needs cuts a person saved here,
                # and this tool used to skip such a bundle, so no tool could
                # make its first cuts. It is marked from scratch, so last.
                if self._uncut_bundle(bundle) and self._needs_reseg(bundle, {}):
                    self._queue.append((bundle, 0))
                continue
            try:
                seg = read_segmentation(sp)
                if not self._needs_reseg(bundle, seg):
                    continue
                cands = seg.get("candidates") or []
                cuts = sorted(int(b) for b in (seg.get("boundaries") or []))
                spare = sum(
                    1 for c in cands
                    if not any(abs(int(c["frame"]) - b) <= SAME_BOUNDARY_FRAMES
                               for b in cuts))
                self._queue.append((bundle, spare))
            except Exception as err:
                logger.warning('Left %s out of the re-segmentation queue, because it could not be read: %s', bundle, err)
                continue
        # The lab's order decides which videos come first. Within one tier,
        # most alternatives first: a video where the segmenter bailed out has
        # no candidates at all, so it has to be marked from scratch -- that is
        # the slowest work and should not be what the tool opens on.
        try:
            from mousereach.watcher.work_priority import load_lab_policy
            _policy = load_lab_policy()
            self._queue.sort(
                key=lambda t: (_policy.rank({"video_id": t[0].name}), -t[1]))
        except Exception:
            self._queue.sort(key=lambda t: -t[1])
        n_blind = sum(1 for _, spare in self._queue if spare == 0)
        self._queue = [b for b, _ in self._queue]
        self.status.setText(
            "%d video(s) need their cuts checked. %d of them have no candidate "
            "timepoints at all (the segmenter bailed out) and have to be marked "
            "by hand -- those are last." % (len(self._queue), n_blind))
        if self._queue:
            self._load_bundle(self._queue[0])
        else:
            self.header.setText("Nothing in the queue needs its cuts checked.")

    def _next_video(self):
        if self.video_stem:
            self._queue = [b for b in self._queue if b.name != self.video_stem]
        if self._queue:
            self._load_bundle(self._queue[0])
        else:
            self.header.setText("Queue finished.")
            self.why.setText("")

    def _load_bundle(self, bundle: Path):
        stem = bundle.name
        # One review window per video (review_claims, MRR-02d): two people
        # re-cutting the same video at once would each save over the other's
        # cuts. The video left is let go; one another window holds is skipped.
        from . import review_claims
        prev = self.seg_path.parent if getattr(self, "seg_path", None) else None
        if prev is not None and Path(prev) != Path(bundle):
            review_claims.release(prev)
        if not review_claims.claim(bundle):
            self._queue = [b for b in self._queue if Path(b) != Path(bundle)]
            self.status.setText("Skipped %s: another review window has it open (%s)."
                                % (stem, review_claims.holder(bundle) or "unknown"))
            if self._queue:
                self._load_bundle(self._queue[0])
            else:
                self.header.setText("Queue finished (the rest is open in other "
                                    "review windows).")
            return
        self.video_stem = stem
        self.seg_path = bundle / ("%s_segments.json" % stem)
        # No segments file (the segmenter could not run): start from no cuts;
        # saving writes the first segments file, all of it the person's.
        self._uncut = not self.seg_path.is_file()
        self._pose_frames = None
        self.seg = {} if self._uncut else read_segmentation(self.seg_path)
        self.candidates = list(self.seg.get("candidates") or [])
        self.boundaries = sorted(int(b) for b in (self.seg.get("boundaries") or []))
        self.original_boundaries = list(self.boundaries)
        self.n_frames = int(self.seg.get("total_frames") or 0)

        reasons = needs_fixing(self.seg)
        self.header.setText("%s   --   %d frames" % (stem, self.n_frames))
        note = "The segmenter says: " + "; ".join(reasons) if reasons else ""
        # Why it is here and why it cannot leave yet: a video sent back after
        # an earlier hand fix, or sent for a flagged session, looks fine to the
        # segmenter, and the person must still see what is asked of them.
        try:
            from .release_cli import describe_routing_reason
            from .resegmented import resegmented_since_entry
            # For a quality-check hold this includes what the check found.
            sent_for = describe_routing_reason(bundle, stem)
            _, pending = resegmented_since_entry(bundle, stem)
            extra = (("Sent here: %s. " % sent_for) if sent_for else "") + pending
            note = (note + "   |   " + extra) if note else extra
        except Exception as err:
            logger.warning('Could not read why %s is in the queue: %s', stem, err)
        if self._uncut:
            try:
                from .release_cli import _routing_reason
                sent_for = _routing_reason(bundle, stem)
            except Exception:
                sent_for = ""
            note = ("The segmenter could not run on this video (%s): there are "
                    "no cuts yet. Mark every cut by hand from the video, then "
                    "save." % (sent_for or "no reason was recorded"))
        elif not self.candidates:
            note += ("   |   No candidate timepoints for this video: the "
                     "segmenter could not track the enclosure well enough to "
                     "propose any, so its cuts are an even grid that means "
                     "nothing. Mark the cuts by hand from the video.")
        self.why.setText(note)

        self._load_video(bundle, stem)
        self._refresh_candidates()
        self._refresh_segments()
        self._removed_calls = []
        # Calls a person already made on these very cuts on this visit (a save
        # that left boundaries unwalked) or that the video was sent back to
        # finish: kept, so only the remaining boundaries are walked
        # (resegmented.carried_calls).
        try:
            from .resegmented import carried_calls
            kept = {} if self._uncut else carried_calls(bundle, stem, self.seg)
        except Exception as err:
            logger.warning("Could not read the earlier boundary calls of %s: %s", stem, err)
            kept = {}
        self._calls = {f: r for f, r in kept.items() if f in self.boundaries}
        # The walk always starts at boundary 1 (owner rule): the person sees
        # every cut in order; ones already answered keep their answer.
        self._gw_idx = 0
        self._gw_load()
        if self._calls:
            self.gw_status.setText(
                "%d of %d boundaries were already accepted or moved by a person "
                "and keep that answer; the walk starts at boundary 1."
                % (len(self._calls), len(self.boundaries)))

    def closeEvent(self, event):
        """Closing the window lets its video go (review_claims)."""
        if getattr(self, "seg_path", None):
            from . import review_claims
            review_claims.release(Path(self.seg_path).parent)
        super().closeEvent(event)

    # ------------------------------------------------------------ guided walk

    def _call(self, frame: int, action: str, shown: Optional[int] = None) -> None:
        """Record a person's explicit call on the cut now at ``frame``."""
        rec = {"frame": int(frame), "action": action,
               "by": os.environ.get("USERNAME", os.environ.get("USER", "unknown")),
               "at": datetime.now().isoformat()}
        if shown is not None:
            rec["shown_frame"] = int(shown)
        self._calls[int(frame)] = rec

    def _uncall(self, frame: int, action: str = "removed") -> None:
        """The cut at ``frame`` is gone: drop its call, keep the removal."""
        self._calls.pop(int(frame), None)
        self._removed_calls.append({
            "frame": int(frame), "action": action,
            "by": os.environ.get("USERNAME", os.environ.get("USER", "unknown")),
            "at": datetime.now().isoformat()})

    def _uncalled(self) -> List[int]:
        """Cut frames nobody has accepted, moved or added yet, in order."""
        return [b for b in sorted(self.boundaries) if int(b) not in self._calls]

    def _first_uncalled_idx(self, after: Optional[int] = None) -> int:
        """Position of the first uncalled cut after position ``after`` (else
        the first anywhere); the last position when every cut is called."""
        b = sorted(self.boundaries)
        order = list(range(len(b)))
        if after is not None:
            order = order[after + 1:] + order[:after + 1]
        for i in order:
            if int(b[i]) not in self._calls:
                return i
        return max(0, len(b) - 1)

    def _gw_load(self):
        """Show the question for the current boundary and park the playhead
        ON it, so stepping backwards shows the jump (and, for boundary 1,
        the pre-roll) immediately."""
        b = sorted(self.boundaries)
        if not b:
            self.gw_header.setText("No boundaries to walk (no cuts yet) -- "
                                   "use the manual controls below.")
            self.gw_plain.setText("No boundaries yet -- add cuts with "
                                  "'Manual cut editing' below.")
            return
        self._gw_idx = max(0, min(self._gw_idx, len(b) - 1))
        i = self._gw_idx
        frame = int(b[i])
        num = i + 1
        n = len(b)
        left = "the PRE-ROLL (no pellet yet -- setup footage)" if i == 0 \
            else "pellet %d (segment %d)" % (num - 1, num - 1)
        right = "the POST-ROLL (no pellet -- session over)" if i == n - 1 \
            else "pellet %d (segment %d)" % (num, num)
        self.gw_header.setText(
            "Boundary %d of %d -- the algorithm put it at frame %d.\n"
            "BEFORE this boundary (earlier frames): %s.\n"
            "AFTER this boundary (later frames): %s."
            % (num, n, frame, left, right))
        self.gw_plain.setText(
            "Boundary %d of %d at frame %d. Is this where the tray finished "
            "moving?" % (num, n, frame))
        self.gw_q.setText(
            "Is this the boundary? (is frame %d the first frame after the "
            "scoring-area jump, within %d frames)"
            % (frame, GUIDED_TOLERANCE_FRAMES))
        done = self._calls.get(frame)
        if done:
            self.gw_q.setText(self.gw_q.text() + "\n(Already %s by %s -- confirming "
                              "again is fine.)" % (done.get("action"), done.get("by")))
        self.gw_yes.setChecked(True)
        self.gw_spin.setValue(frame)
        self.gw_status.setText("")
        self._gw_fill_candidate_row(self.gw_cands, frame,
                                    self.gw_spin, self.gw_no, is_end=False)
        self._goto(frame)

    def _gw_candidates_near(self, frame: int, limit: int = 5) -> List[dict]:
        """The segmenter's candidate tray advances nearest ``frame``, closest
        first. Excludes only candidates within the question's own tolerance
        of the boundary -- those are the same answer as 'yes'. Candidates
        just outside it (11-30 frames off) are exactly the alternatives the
        question exists to surface, so they stay."""
        cands = [c for c in self.candidates
                 if abs(int(c["frame"]) - frame) > GUIDED_TOLERANCE_FRAMES]
        cands.sort(key=lambda c: abs(int(c["frame"]) - frame))
        return cands[:limit]

    def _gw_fill_candidate_row(self, row: QGridLayout, boundary_frame: int,
                               spin: QSpinBox, checkbox: QCheckBox,
                               is_end: bool):
        """Rebuild one question's candidate chips. Clicking a chip jumps the
        playhead to the frame it implies for THIS question (cut frame for a
        start, cut-1 for an end) and fills it in as the answer -- same
        pick-from-what-the-algo-saw flow as the review tool's reach picker.

        The chips fill a grid CHIP_COLUMNS wide (its label sits above it), so
        the row never makes the panel wider than a small screen allows."""
        # drop previous chips
        while row.count():
            item = row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        cands = self._gw_candidates_near(boundary_frame)
        if not cands:
            lbl = QLabel("(none nearby)")
            lbl.setStyleSheet("color: #666;")
            row.addWidget(lbl, 0, 0)
            return
        for i, c in enumerate(cands):
            cut = int(c["frame"])
            answer = cut - 1 if is_end else cut
            btn = QPushButton("f=%d  (%d/4, %.2f)" % (
                answer, int(c.get("n_proposers") or 0),
                float(c.get("consensus_score") or 0.0)))
            btn.setToolTip(
                "Tray advance the segmenter saw at frame %d (proposed by %s). "
                "Click: jump there and use it as the answer." % (
                    cut, ", ".join(c.get("proposers") or []) or "?"))
            btn.clicked.connect(
                lambda _=False, f=answer: self._gw_pick_candidate(
                    f, spin, checkbox))
            row.addWidget(btn, i // CHIP_COLUMNS, i % CHIP_COLUMNS)

    def _gw_pick_candidate(self, frame: int, spin: QSpinBox, checkbox: QCheckBox):
        self._goto(frame)
        checkbox.setChecked(True)
        spin.setValue(int(frame))

    def _gw_use_playhead(self, spin, checkbox):
        try:
            f = int(self.viewer.dims.current_step[0])
        except Exception:
            return
        checkbox.setChecked(True)
        spin.setValue(f)

    def _gw_confirm(self):
        """Record this boundary's answer (moving it if denied), then go to the
        next boundary nobody has called. The answer is kept under the cut's
        frame (its frame after a move), never its position."""
        b = sorted(self.boundaries)
        if not b:
            return
        i = self._gw_idx
        frame = int(b[i])
        note = ""
        now_at = frame
        if self.gw_no.isChecked():
            f = int(self.gw_spin.value())
            if f != frame and f in b:
                self.gw_status.setText(
                    "Frame %d is already another boundary. Pick another frame, or "
                    "use 'No advance here -- remove this boundary'." % f)
                return
            self._calls.pop(frame, None)
            b[i] = f
            self.boundaries = sorted(b)
            self._call(f, "moved", shown=frame)
            now_at = f
            note = "boundary %d moved to frame %d." % (i + 1, f)
        else:
            self._call(frame, "accepted", shown=frame)
        self._refresh_candidates(); self._refresh_segments()
        left = self._uncalled()
        total = len(self.boundaries)
        if left:
            pos = sorted(self.boundaries).index(now_at)
            self._gw_idx = self._first_uncalled_idx(after=pos)
            self._gw_load()
            if note:
                self.gw_status.setText(note)
        else:
            self.gw_status.setText(
                ("%s  " % note if note else "")
                + "Walk finished (%d/%d boundaries accepted or moved). Check the "
                  "segment table for double-length rows (a missed boundary), "
                  "then 'Save these cuts'." % (total - len(left), total))

    def _gw_quick_yes(self):
        """Pinned 'Yes, correct': the guided walk's Yes + Confirm."""
        if not self.boundaries:
            return
        self.gw_yes.setChecked(True)
        self._gw_confirm()

    def _gw_quick_move(self):
        """Pinned 'Move it to the current frame': the guided walk's
        'Use current frame' + Confirm (records No, moved to the playhead)."""
        if not self.boundaries:
            return
        self._gw_use_playhead(self.gw_spin, self.gw_no)
        if not self.gw_no.isChecked():
            self.gw_status.setText("Could not read the current frame -- "
                                   "nothing was changed.")
            return
        self._gw_confirm()

    def _gw_remove_boundary(self):
        """The segmenter invented this boundary: remove the cut entirely."""
        b = sorted(self.boundaries)
        if not b:
            return
        i = self._gw_idx
        frame = int(b[i])
        self._uncall(frame)
        del b[i]
        self.boundaries = b
        self._refresh_candidates(); self._refresh_segments()
        self.gw_status.setText(
            "Boundary at frame %d removed -- the two segments it separated "
            "are now one, and later segments renumbered." % frame)
        if self.boundaries:
            self._gw_idx = self._first_uncalled_idx(after=i - 1 if i > 0 else None)
            self._gw_load()

    def _gw_back(self):
        if self._gw_idx > 0:
            self._gw_idx -= 1
            self._gw_load()

    def _load_video(self, bundle: Path, stem: str):
        """Decode-on-demand layer, added once (swapping layers crashes the
        renderer -- see the causal review tool, which learned this the hard way)."""
        import cv2
        from mousereach.review.causal_review_widget import _LazyVideo

        # Bundles are staged NOT self-contained: the mp4 normally stays in the
        # finished-work tree and the bundle carries only the small JSONs plus a
        # manifest naming where the real files are. Look in all three places, in
        # the order that costs least.
        mp4 = bundle / ("%s.mp4" % stem)
        if not mp4.is_file():
            try:
                man = json.loads((bundle / ("%s_manifest.json" % stem)).read_text())
                cand = man.get("canonical_video_path")
                if cand and Path(cand).is_file():
                    mp4 = Path(cand)
            except Exception:
                pass
        if not mp4.is_file():
            try:
                from mousereach.config import Paths
                root = getattr(Paths, "ANALYZED_OUTPUT", None)
                if root:
                    # WHY first_file, not rglob: a superseded copy under
                    # Analyzed/Archive/ keeps its name; re-segmenting against it
                    # would score cuts on the wrong file.
                    from mousereach.pipeline.analyzed_tree import first_file
                    hit = first_file(root, "%s.mp4" % stem)
                    if hit is not None:
                        mp4 = hit
            except Exception:
                pass
        if not mp4.is_file():
            self.status.setText("No video file found for %s -- cuts can still be "
                                "edited from the candidate list." % stem)
            return
        try:
            cap = cv2.VideoCapture(str(mp4))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            if self._video_layer is not None and self._video_layer in self.viewer.layers:
                self.viewer.layers.remove(self._video_layer)
            self._video_layer = self.viewer.add_image(
                _LazyVideo(mp4, n, h, w), name=stem, rgb=True)
            self.n_frames = self.n_frames or n
            self._goto_spin.setRange(0, max(0, self.n_frames - 1))
        except Exception as e:
            self.status.setText("Could not open the video (%s); the candidate "
                                "list still works." % e)
        self._add_dlc_overlay(bundle, stem, mp4)
        self._bind_nav_keys()
        try:
            self.viewer.dims.events.current_step.connect(self._on_frame_change)
        except Exception:
            pass

    def _add_dlc_overlay(self, bundle: Path, stem: str, mp4: Path):
        """Draw the DLC tracking points on the video -- the scoring-area
        points jumping IS how a human sees a segment boundary, so re-seg is
        blind without them. Pose found like the mp4: bundle -> manifest ->
        next to the canonical video."""
        import numpy as np
        try:
            import pandas as pd
            pose = next(iter(bundle.glob("%sDLC*.h5" % stem)), None)
            if pose is None:
                try:
                    man = json.loads(
                        (bundle / ("%s_manifest.json" % stem)).read_text())
                    cand = man.get("canonical_dlc_h5_path")
                    if cand and Path(cand).is_file():
                        pose = Path(cand)
                except Exception:
                    pass
            if pose is None:
                pose = next(iter(mp4.parent.glob("%sDLC*.h5" % stem)), None)
            if pose is None:
                self.status.setText(
                    "No DLC pose file found -- video loads without tracking "
                    "points (boundary checking is much harder without them).")
                return
            df = pd.read_hdf(pose)
            df.columns = ['_'.join(str(c) for c in col[1:]) for col in df.columns]
            n = len(df)
            self._pose_frames = n
            bodyparts = sorted({c[:-2] for c in df.columns if c.endswith('_x')})
            colors_base = [
                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [1, 0, 0.5]]
            frames_all = np.arange(n)
            pts, cols, bps = [], [], []
            for i, bp in enumerate(bodyparts):
                xc, yc, lc = bp + '_x', bp + '_y', bp + '_likelihood'
                if xc not in df.columns or yc not in df.columns:
                    continue
                xs = df[xc].to_numpy(dtype=float)
                ys = df[yc].to_numpy(dtype=float)
                lks = (df[lc].to_numpy(dtype=float)
                       if lc in df.columns else np.ones(n))
                valid = ~(np.isnan(xs) | np.isnan(ys))
                if not valid.any():
                    continue
                fv = frames_all[valid]
                lv = np.clip(lks[valid], 0.0, 1.0)
                alpha = np.where(lv < 0.5, 0.05,
                                 0.10 + 0.90 * (((lv - 0.5) / 0.5) ** 2))
                base = np.array(colors_base[i % len(colors_base)], dtype=float)
                pts.append(np.column_stack([fv, ys[valid], xs[valid]]))
                cols.append(np.column_stack([np.tile(base, (len(fv), 1)), alpha]))
                bps.extend([bp] * len(fv))
            if not pts:
                return
            if (getattr(self, "_points_layer", None) is not None
                    and self._points_layer in self.viewer.layers):
                self.viewer.layers.remove(self._points_layer)
            self._points_layer = self.viewer.add_points(
                np.vstack(pts), name='DLC Points', size=3,
                face_color=np.vstack(cols),
                features={'bp': bps},
                text={'string': '{bp}', 'size': 7, 'color': 'white',
                      'translation': [0, -7, 0]},
            )
        except Exception as e:
            self.status.setText("DLC points could not be drawn (%s); the "
                                "video still works." % e)

    # -- playback: byte-for-byte the causal review tool's machinery, so the
    # two tools can never feel different to drive.

    def _play_forward(self):
        self.playback_direction = 1
        self._start_playback()

    def _play_reverse(self):
        self.playback_direction = -1
        self._start_playback()

    def _stop_play(self):
        self.is_playing = False
        self.playback_timer.stop()
        self._play_btn.setText("Play")
        self._play_rev_btn.setText("Rev")

    def _start_playback(self):
        if self.is_playing:
            self._stop_play()
            return
        self.is_playing = True
        interval = max(1, int(1000 / (self._fps_play * self.playback_speed)))
        self.playback_timer.start(interval)
        if self.playback_direction == 1:
            self._play_btn.setText("||")
        else:
            self._play_rev_btn.setText("||")

    def _playback_step(self):
        current = self.viewer.dims.current_step[0]
        skip = max(1, int(self.playback_speed))
        new_frame = current + (skip * self.playback_direction)
        if 0 <= new_frame < self.n_frames:
            self.viewer.dims.set_current_step(0, new_frame)
        else:
            self._stop_play()

    def _set_speed(self, speed: float):
        self.playback_speed = speed
        for s, btn in self._speed_buttons.items():
            btn.setChecked(s == speed)
        if self.is_playing:
            interval = max(1, int(1000 / (self._fps_play * self.playback_speed)))
            self.playback_timer.stop()
            self.playback_timer.start(interval)

    @property
    def _fps_play(self) -> float:
        return float(self.seg.get("fps") or 60.0)

    def _segment_length(self) -> int:
        """This video's typical segment length in frames: the median interval
        between the current boundaries. Falls back to 30s of video when there
        are not enough boundaries to measure."""
        import numpy as np
        b = sorted(self.boundaries)
        if len(b) >= 3:
            return int(np.median(np.diff(b)))
        return int(30 * self._fps_play)

    def _jump_frames(self, delta: int):
        try:
            cur = int(self.viewer.dims.current_step[0])
        except Exception:
            return
        self._goto(cur + int(delta))

    def _on_frame_change(self, event=None):
        try:
            f = int(self.viewer.dims.current_step[0])
            self._frame_label.setText("Frame: %d / %d" % (f, self.n_frames))
            t = f / self._fps_play
            self._time_label.setText("Time: %d:%05.2f" % (int(t // 60), t % 60))
        except Exception:
            pass

    def _bind_nav_keys(self):
        """The causal review tool's exact keyboard scheme, bound through
        napari so it fires in a docked widget: Space play/pause, b reverse,
        arrows 1 frame, Shift-arrows 10, Ctrl-arrows 100, 1-6 speeds.

        Called after every video load: the keys go on the viewer once, and on
        the video and dots layers each time they are made, because napari looks
        a key up in the selected layer's shortcuts first (review/keyboard.py)."""
        if not getattr(self, "_nav_keys_bound", False):
            self._nav_keys_bound = True

            def _toggle_play(_):
                if self.is_playing:
                    self._stop_play()
                else:
                    self._play_forward()

            def _toggle_reverse(_):
                if self.is_playing:
                    self._stop_play()
                else:
                    self._play_reverse()

            keys = {'Space': _toggle_play, 'b': _toggle_reverse}
            for key, delta in [('Left', -1), ('Right', 1), ('Shift-Left', -10),
                               ('Shift-Right', 10), ('Control-Left', -100),
                               ('Control-Right', 100)]:
                keys[key] = (lambda d: (lambda _: self._jump_frames(d)))(delta)
            # One segment length at a time -- for chasing a shifted-by-one video.
            keys['Alt-Left'] = lambda _: self._jump_frames(-self._segment_length())
            keys['Alt-Right'] = lambda _: self._jump_frames(self._segment_length())
            for key, spd in zip('123456', [0.25, 0.5, 1, 2, 4, 8]):
                keys[key] = (lambda s: (lambda _: self._set_speed(s)))(spd)
            self._review_keys = keys
            for key, func in keys.items():
                self.viewer.bind_key(key, func, overwrite=True)
        for layer in (self._video_layer, getattr(self, "_points_layer", None)):
            if layer is not None and layer in self.viewer.layers:
                review_keys_on_layer(layer, self._review_keys)

    def showEvent(self, event):
        super().showEvent(event)
        # Docked by now: what is typed in this panel stays in it and never
        # runs the video's keys (review/keyboard.py).
        keep_typing_in_panel(self)

    # -------------------------------------------------------------- tables

    def _fps(self) -> float:
        return float(self.seg.get("fps") or 60.0)

    def _is_cut(self, frame: int) -> bool:
        return any(abs(frame - b) <= SAME_BOUNDARY_FRAMES for b in self.boundaries)

    def _refresh_candidates(self):
        rows = [c for c in self.candidates
                if not (self.only_unused.isChecked() and self._is_cut(int(c["frame"])))]
        rows.sort(key=lambda c: int(c["frame"]))
        self.cand_table.setRowCount(len(rows))
        self._cand_rows = rows
        fps = self._fps()
        for i, c in enumerate(rows):
            f = int(c["frame"])
            cut = self._is_cut(f)
            vals = ["yes" if cut else "",
                    str(f),
                    "%.1f s" % (f / fps),
                    "%d of 4  (%s)" % (int(c.get("n_proposers") or 0),
                                       ", ".join(c.get("proposers") or [])),
                    "%.2f" % float(c.get("consensus_score") or 0.0)]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if cut:
                    item.setBackground(Qt.darkGreen)
                self.cand_table.setItem(i, j, item)
        self.count_label.setText("%d cuts  (the algorithm made %d)"
                                 % (len(self.boundaries), len(self.original_boundaries)))

    def _refresh_segments(self):
        segs = segments_from(self.boundaries, self.n_frames)
        fps = self._fps()
        lengths = [(e - s + 1) / fps for _, s, e in segs]
        median = float(np.median(lengths)) if lengths else 0.0
        self.seg_table.setRowCount(len(segs))
        for i, (num, s, e) in enumerate(segs):
            secs = (e - s + 1) / fps
            odd = median > 0 and (secs > median * 1.5 or secs < median * 0.5)
            vals = [str(num), str(s), "%.1f" % secs,
                    "unusual length -- check for a missing cut" if odd else ""]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if odd:
                    item.setBackground(Qt.darkRed)
                self.seg_table.setItem(i, j, item)

    # ------------------------------------------------------------- actions

    def _selected_candidate(self) -> Optional[dict]:
        rows = self.cand_table.selectionModel().selectedRows()
        if not rows:
            return None
        i = rows[0].row()
        return self._cand_rows[i] if i < len(self._cand_rows) else None

    def _goto(self, frame: int):
        try:
            self.viewer.dims.set_current_step(
                0, max(0, min(int(frame), max(0, self.n_frames - 1))))
        except Exception:
            pass

    def _jump_to_selected(self):
        c = self._selected_candidate()
        if c:
            self._goto(int(c["frame"]))

    def _jump_to_segment(self):
        rows = self.seg_table.selectionModel().selectedRows()
        if not rows:
            return
        segs = segments_from(self.boundaries, self.n_frames)
        i = rows[0].row()
        if i < len(segs):
            self._goto(segs[i][1])

    def _use_selected(self):
        c = self._selected_candidate()
        if not c:
            return
        f = int(c["frame"])
        if self._is_cut(f):
            self.status.setText("Frame %d is already a cut." % f)
            return
        self.boundaries = sorted(self.boundaries + [f])
        self._call(f, "added")
        self.status.setText("Added a cut at frame %d." % f)
        self._refresh_candidates(); self._refresh_segments()

    def _drop_selected(self):
        c = self._selected_candidate()
        if not c:
            return
        f = int(c["frame"])
        near = [b for b in self.boundaries if abs(b - f) <= SAME_BOUNDARY_FRAMES]
        if not near:
            self.status.setText("Frame %d is not a cut." % f)
            return
        for b in near:
            self.boundaries.remove(b)
            self._uncall(b)
        self.status.setText("Removed the cut near frame %d." % f)
        self._refresh_candidates(); self._refresh_segments()

    def _add_here(self):
        try:
            f = int(self.viewer.dims.current_step[0])
        except Exception:
            return
        if self._is_cut(f):
            self.status.setText("There is already a cut at frame %d." % f)
            return
        self.boundaries = sorted(self.boundaries + [f])
        self._call(f, "added")
        self.status.setText("Added a cut at frame %d (not one the algorithm "
                            "proposed)." % f)
        self._refresh_candidates(); self._refresh_segments()

    def _reset(self):
        self.boundaries = list(self.original_boundaries)
        # Back to the algorithm's cuts: no cut is a person's call any more.
        self._calls = {}
        self._removed_calls = []
        self._gw_idx = 0
        self._gw_load()
        self.status.setText("Back to the algorithm's cuts. Walk every boundary again.")
        self._refresh_candidates(); self._refresh_segments()

    def _save(self):
        if not self.seg_path or not self.boundaries:
            return
        # The bundle may have left the queue while it was open here (the
        # return scan or another window moved it). Nothing is written then:
        # writing would recreate its folder holding only these cuts and a
        # clear marker -- a "cleared" video with no files.
        if not Path(self.seg_path).parent.is_dir():
            QMessageBox.warning(self, "This video has left the queue",
                                "%s is no longer in the re-segmentation queue (the "
                                "watcher or another window moved it while it was open "
                                "here). Nothing was saved. Load the next video."
                                % self.video_stem)
            return
        if len(self.boundaries) < 2:
            QMessageBox.warning(self, "Too few cuts",
                                "At least two cuts are needed to make a segment.")
            return
        # Every boundary must be explicitly accepted or moved first (owner
        # decision MRR-08d, 2026-11-19): saving stamps the cuts as a person's,
        # and a cut nobody looked at is not a person's call.
        left = self._uncalled()
        if left:
            shown = ", ".join(str(f) for f in left[:10]) + (" ..." if len(left) > 10 else "")
            QMessageBox.warning(
                self, "Not every boundary checked",
                "%d of %d boundaries have not been accepted or moved yet (frames %s).\n\n"
                "Walk each one: press Confirm if it is right, or move it, or remove "
                "it. Nothing was saved." % (len(left), len(self.boundaries), shown))
            self._gw_idx = self._first_uncalled_idx()
            self._gw_load()
            return

        uncut = not Path(self.seg_path).is_file()
        # A first segments file takes the video's length from its pose file,
        # as the segmenter does (len of the pose table); the video's own
        # frame count only when no pose was found.
        length = int(getattr(self, "_pose_frames", None) or self.n_frames or 0)
        if uncut and length <= 0:
            # A first segments file must say how long the video is; every
            # later step reads the last segment's end from it.
            QMessageBox.warning(self, "Video length unknown",
                                "This video has no segments file and its video "
                                "could not be opened, so its length is not "
                                "known. Nothing was saved.")
            return

        archive = (Path(self.seg_path).parents[3] / "_archived"
                   / ("segmentation_before_human_fix_%s"
                      % datetime.now().strftime("%Y%m%d_%H%M%S")))

        # The fields every later step reads; boundary_source "human" keeps
        # the segmenter from replacing the cuts on the re-run (hand_cuts says
        # why each is there).
        seg = mark_as_hand_cuts(
            self.seg, self.boundaries, self.original_boundaries,
            os.environ.get("USERNAME", os.environ.get("USER", "unknown")))
        if uncut:
            # The first segments file of a video the segmenter could not cut.
            # fps: the segmenter's own default (it never reads the video's).
            seg.update({"video_name": self.video_stem,
                        "total_frames": length,
                        "fps": 60.0,
                        "candidates": [],
                        "note": "No segmenter output: every cut set by hand."})
            try:
                from .release_cli import _routing_reason
                why = _routing_reason(Path(self.seg_path).parent, self.video_stem)
            except Exception:
                why = ""
            seg["needs_human_resolved"] = [why or "no segments file"]
        # What the person did with every cut, keyed by frame
        # (hand_cuts.BOUNDARY_CALLS_KEY): the proof, read by
        # resegmented_since_entry and release_after_hand_cuts, that every
        # boundary was explicitly accepted, moved or added. The older
        # position-keyed "guided_walk" record of an earlier save describes
        # other cuts; the archived copy keeps it.
        seg[BOUNDARY_CALLS_KEY] = ([self._calls[f] for f in sorted(self._calls)
                                    if f in self.boundaries]
                                   + list(self._removed_calls))
        seg.pop("guided_walk", None)

        # Archive the original, then archive and remove every output computed
        # from the old cuts (a boundary change voids them), then replace the
        # segments file in one step. Nothing is written when either of the
        # first two cannot be done.
        try:
            removed = commit_hand_cuts(self.seg_path, self.video_stem, seg, archive)
        except HandCutsNotSaved as e:
            QMessageBox.critical(self, ("Could not archive" if e.stage == "archive"
                                        else "Could not invalidate stale outputs"),
                                 str(e))
            return

        n_added = len([b for b in self.boundaries if b not in self.original_boundaries])
        n_removed = len([b for b in self.original_boundaries if b not in self.boundaries])
        self.status.setText(
            "Saved %d cuts for %s (%d added, %d removed). Original archived."
            % (len(self.boundaries), self.video_stem, n_added, n_removed)
            + ("  %d downstream file(s) computed from the old cuts were "
               "archived and removed; the pipeline recomputes them."
               % len(removed) if removed else ""))

        # Release the bundle if this fix answers why it was queued
        # (hand_cuts.release_after_hand_cuts gates it on the routing reason).
        self.status.setText(self.status.text()
                            + release_after_hand_cuts(Path(self.seg_path).parent,
                                                      self.video_stem))

        self._next_video()


def _asked_again_note(bundle: Path, stem: str) -> str:
    """hand_cuts.asked_again_note, under the name this module always had."""
    return asked_again_note(bundle, stem)


def main():
    """Launch the segmentation fixer over the re-segmentation queue."""
    import argparse
    import napari
    from mousereach.config import Paths

    ap = argparse.ArgumentParser(
        description="Correct a video's segment cuts, using the candidate tray "
                    "advances the segmenter found but did not use.")
    ap.add_argument("--queue-dir", type=Path, default=None,
                    help="Queue of bundles to work through "
                         "(default: the re-segmentation queue)")
    args = ap.parse_args()

    queue = args.queue_dir or getattr(Paths, "DEEP_REVIEW", None)
    if queue is None:
        print("[FAIL] no re-segmentation queue configured; pass --queue-dir")
        return 1

    print("Segmentation fixer over %s" % queue)
    viewer = napari.Viewer(title="MouseReach -- fix segmentation")
    widget = FixSegmentationWidget(viewer, queue)
    dock = viewer.window.add_dock_widget(widget, name="Fix segmentation",
                                         area="right")
    fit_review_window(viewer, widget, dock)
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
