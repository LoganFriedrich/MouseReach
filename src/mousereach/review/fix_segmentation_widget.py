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

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QGroupBox, QMessageBox,
    QCheckBox, QSplitter, QSpinBox, QRadioButton, QButtonGroup,
)

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
        self._video_layer = None
        self._queue: List[Path] = []
        self._gw_idx: int = 0
        self._gw_records: Dict[int, dict] = {}

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

THE WORKFLOW -- one question per boundary:

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
* Press "Confirm -> next boundary". Repeat to the end, then "Save these
  cuts". Saving stamps the boundaries as human-made; the pipeline keeps
  them on the re-run.

IF A BOUNDARY WAS MISSED (a segment looks double-length in the table, or
you saw an extra jump): add it with "Add a cut at the current frame" in the
manual controls below -- the walk picks it up immediately.

EVERYTHING BELOW THE QUESTION IS OPTIONAL. The candidate table and the
add/drop-cut buttons are manual controls; the segment table just shows what
the current boundaries produce -- red rows hint at a missing boundary. If
you never need them, Confirm + Save is the whole job.

NAVIGATION (identical to the other review tools): Play/Rev/Stop with speed
buttons; step buttons for 1/10/100 frames; "Go to" jumps to a typed frame.
Keyboard: Space = play/pause, b = reverse, arrows = 1 frame, Shift-arrows =
10, Ctrl-arrows = 100, keys 1-6 = speed.

FINISHING: saving cuts does NOT release the video from the deep-review
queue. When its cuts are right, open Deep Review on it and press
"Clear -> re-enter pipeline" -- that is the release; everything after is
automatic.

"Skip this video" moves on without saving anything.
"""

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

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
        for delta, label in [(-100, "<<"), (-10, "<"), (-1, "-1"),
                             (1, "+1"), (10, ">"), (100, ">>")]:
            b = QPushButton(label)
            b.setMaximumWidth(35)
            b.setToolTip("Step %+d frame(s) (keyboard: arrows = 1, "
                         "Shift-arrows = 10, Ctrl-arrows = 100)" % delta)
            b.clicked.connect(lambda _=False, d=delta: self._jump_frames(d))
            step_row.addWidget(b)
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

        split = QSplitter(Qt.Vertical)
        layout.addWidget(split, 1)

        # --- guided walk: one question per BOUNDARY ------------------------
        gw_box = QGroupBox("Guided walk (one boundary at a time)")
        gv = QVBoxLayout()
        gw_box.setLayout(gv)

        self.gw_header = QLabel("")
        self.gw_header.setStyleSheet("font-weight: bold;")
        self.gw_header.setWordWrap(True)
        gv.addWidget(self.gw_header)

        r = QHBoxLayout()
        self.gw_q = QLabel("")
        self.gw_q.setWordWrap(True)
        r.addWidget(self.gw_q, 1)
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
        self.gw_cands = QHBoxLayout()
        self.gw_cands.addWidget(QLabel("Segmenter's nearby candidates:"))
        self.gw_cands.addStretch()
        gv.addLayout(self.gw_cands)

        nav = QHBoxLayout()
        bb = QPushButton("< Back a boundary")
        bb.setToolTip("Revisit the previous boundary.")
        bb.clicked.connect(self._gw_back)
        nav.addWidget(bb)
        rmb = QPushButton("No advance here -- remove this boundary")
        rmb.setStyleSheet("background: #5a1616; color: white;")
        rmb.setToolTip(
            "There is no tray advance anywhere near this frame: the "
            "segmenter invented this boundary. Removing it merges the two "
            "segments it separated, and every later segment renumbers.")
        rmb.clicked.connect(self._gw_remove_boundary)
        nav.addWidget(rmb)
        nav.addStretch()
        cb = QPushButton("Confirm -> next boundary")
        cb.setStyleSheet("font-weight: bold;")
        cb.setToolTip("Record the answer (moving the boundary if you said "
                      "No) and load the next one.")
        cb.clicked.connect(self._gw_confirm)
        nav.addWidget(cb)
        gv.addLayout(nav)

        self.gw_status = QLabel("")
        self.gw_status.setWordWrap(True)
        self.gw_status.setStyleSheet("color: #e08020;")
        gv.addWidget(self.gw_status)
        split.addWidget(gw_box)

        # --- candidates ---------------------------------------------------
        cand_box = QGroupBox("Manual cut editing (OPTIONAL -- for fixes the "
                             "questions above cannot express)")
        cv = QVBoxLayout()
        cand_box.setLayout(cv)
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
        cv.addWidget(self.cand_table)

        row = QHBoxLayout()
        for text, slot, tip in (
                ("Use this candidate", self._use_selected,
                 "Add the selected table row's timepoint as a new cut."),
                ("Drop this cut", self._drop_selected,
                 "Remove the cut at/near the selected table row's timepoint."),
                ("Add a cut at the current frame", self._add_here,
                 "Add a cut exactly where the video playhead is now -- for a "
                 "tray advance the segmenter never proposed.")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        self.only_unused = QCheckBox("Hide candidates already used")
        self.only_unused.setToolTip(
            "Show only proposed timepoints that are NOT currently cuts.")
        self.only_unused.toggled.connect(lambda _: self._refresh_candidates())
        row.addWidget(self.only_unused)
        row.addStretch()
        cv.addLayout(row)
        split.addWidget(cand_box)

        # --- resulting segments -------------------------------------------
        seg_box = QGroupBox("Segments these cuts produce (READ-ONLY check -- "
                            "nothing to fill in)")
        sv = QVBoxLayout()
        seg_box.setLayout(sv)
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
        sv.addWidget(self.seg_table)
        split.addWidget(seg_box)

        # --- actions -------------------------------------------------------
        act = QHBoxLayout()
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("font-weight: bold;")
        act.addWidget(self.count_label)
        act.addStretch()
        for text, slot, style, tip in (
                ("Back to the algorithm's cuts", self._reset, "",
                 "Throw away every change on this video and restore the "
                 "algorithm's original cuts."),
                ("Save these cuts", self._save, "font-weight: bold;",
                 "Write the corrected cuts (stamped human-made, kept by the "
                 "pipeline) and load the next video. Does NOT release the "
                 "video from deep review -- that is Deep Review's Clear button."),
                ("Skip this video", self._next_video, "",
                 "Move on WITHOUT saving anything on this video; it stays in "
                 "the queue.")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            b.setStyleSheet(style)
            act.addWidget(b)
        layout.addLayout(act)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    # ------------------------------------------------------------- queue

    def _needs_reseg(self, bundle: Path, seg: dict) -> bool:
        """Does this bundle belong in the re-segmentation queue?

        Three ways in, one way out:
          * the segmenter's own needs_human verdict (the original criterion);
          * a reviewer declared a segment mislabel (true_segment_num in the
            travelling causal review);
          * the bundle was ROUTED here for a segmentation reason (the triage
            escalate button, or the watcher's mislabel diverts -- their
            routing reasons all name segmentation).
        The way out: boundary_source == "human" means the cuts were already
        hand-fixed, so nothing here is pending regardless of the above."""
        if seg.get("boundary_source") == "human":
            return False
        if needs_fixing(seg):
            return True
        stem = bundle.name
        try:
            from mousereach.review.triage_status import segmentation_corrected
            rp = bundle / ("%s_causal_review.json" % stem)
            if rp.is_file() and segmentation_corrected(
                    json.loads(rp.read_text(encoding="utf-8"))):
                return True
        except Exception:
            pass
        try:
            rj = bundle / ("%s_routing.json" % stem)
            if rj.is_file():
                reason = str(json.loads(rj.read_text(
                    encoding="utf-8")).get("routed_reason", "")).lower()
                if "segment" in reason or "re-seg" in reason:
                    return True
        except Exception:
            pass
        return False

    def _load_queue(self):
        """Videos needing a human's cuts: the segmenter's own needs_human
        verdict, a reviewer-declared segment mislabel, or a
        segmentation-reason routing (see _needs_reseg)."""
        self._queue = []
        if not self.queue_dir.is_dir():
            self.header.setText("Queue not found: %s" % self.queue_dir)
            return
        for bundle in sorted(self.queue_dir.iterdir()):
            if not bundle.is_dir():
                continue
            sp = bundle / ("%s_segments.json" % bundle.name)
            if not sp.is_file():
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
            except Exception:
                continue
        # Most alternatives first. A video where the segmenter bailed out has no
        # candidates at all, so it has to be marked from scratch -- that is the
        # slowest work and should not be what the tool opens on.
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
        self.video_stem = stem
        self.seg_path = bundle / ("%s_segments.json" % stem)
        self.seg = read_segmentation(self.seg_path)
        self.candidates = list(self.seg.get("candidates") or [])
        self.boundaries = sorted(int(b) for b in (self.seg.get("boundaries") or []))
        self.original_boundaries = list(self.boundaries)
        self.n_frames = int(self.seg.get("total_frames") or 0)

        reasons = needs_fixing(self.seg)
        self.header.setText("%s   --   %d frames" % (stem, self.n_frames))
        note = "The segmenter says: " + "; ".join(reasons) if reasons else ""
        if not self.candidates:
            note += ("   |   No candidate timepoints for this video: the "
                     "segmenter could not track the enclosure well enough to "
                     "propose any, so its cuts are an even grid that means "
                     "nothing. Mark the cuts by hand from the video.")
        self.why.setText(note)

        self._load_video(bundle, stem)
        self._refresh_candidates()
        self._refresh_segments()
        self._gw_idx = 0
        self._gw_records = {}
        self._gw_load()

    # ------------------------------------------------------------ guided walk

    def _gw_load(self):
        """Show the question for the current boundary and park the playhead
        ON it, so stepping backwards shows the jump (and, for boundary 1,
        the pre-roll) immediately."""
        b = sorted(self.boundaries)
        if not b:
            self.gw_header.setText("No boundaries to walk (no cuts yet) -- "
                                   "use the manual controls below.")
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
        self.gw_q.setText(
            "Is this the boundary? (is frame %d the first frame after the "
            "scoring-area jump, within %d frames)"
            % (frame, GUIDED_TOLERANCE_FRAMES))
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

    def _gw_fill_candidate_row(self, row: QHBoxLayout, boundary_frame: int,
                               spin: QSpinBox, checkbox: QCheckBox,
                               is_end: bool):
        """Rebuild one question's candidate chips. Clicking a chip jumps the
        playhead to the frame it implies for THIS question (cut frame for a
        start, cut-1 for an end) and fills it in as the answer -- same
        pick-from-what-the-algo-saw flow as the review tool's reach picker."""
        # drop previous chips (everything after the leading label, before stretch)
        while row.count() > 2:
            item = row.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        cands = self._gw_candidates_near(boundary_frame)
        if not cands:
            lbl = QLabel("(none nearby)")
            lbl.setStyleSheet("color: #666;")
            row.insertWidget(1, lbl)
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
            row.insertWidget(1 + i, btn)

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
        """Record this boundary's answer (moving it if denied), advance."""
        b = sorted(self.boundaries)
        if not b:
            return
        i = self._gw_idx
        frame = int(b[i])
        rec = {"boundary_num": i + 1, "algo_frame": frame,
               "confirmed": not self.gw_no.isChecked()}
        note = ""
        if self.gw_no.isChecked():
            f = int(self.gw_spin.value())
            rec["corrected_frame"] = f
            b[i] = f
            self.boundaries = sorted(b)
            note = "boundary %d moved to frame %d." % (i + 1, f)
        self._gw_records[i + 1] = rec
        self._refresh_candidates(); self._refresh_segments()
        total = len(self.boundaries)
        if self._gw_idx < total - 1:
            self._gw_idx += 1
            self._gw_load()
            if note:
                self.gw_status.setText(note)
        else:
            self.gw_status.setText(
                ("%s  " % note if note else "")
                + "Walk finished (%d/%d boundaries answered). Check the "
                  "segment table for double-length rows (a missed boundary), "
                  "then 'Save these cuts'." % (len(self._gw_records), total))

    def _gw_remove_boundary(self):
        """The segmenter invented this boundary: remove the cut entirely."""
        b = sorted(self.boundaries)
        if not b:
            return
        i = self._gw_idx
        frame = int(b[i])
        self._gw_records[i + 1] = {"boundary_num": i + 1, "algo_frame": frame,
                                   "confirmed": False, "removed": True}
        del b[i]
        self.boundaries = b
        self._refresh_candidates(); self._refresh_segments()
        self.gw_status.setText(
            "Boundary at frame %d removed -- the two segments it separated "
            "are now one, and later segments renumbered." % frame)
        self._gw_idx = min(self._gw_idx, len(self.boundaries) - 1)
        if self.boundaries:
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
                    hit = next(iter(Path(root).rglob("%s.mp4" % stem)), None)
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
        arrows 1 frame, Shift-arrows 10, Ctrl-arrows 100, 1-6 speeds."""
        if getattr(self, "_nav_keys_bound", False):
            return
        self._nav_keys_bound = True
        v = self.viewer

        @v.bind_key('Space', overwrite=True)
        def _toggle_play(viewer):
            if self.is_playing:
                self._stop_play()
            else:
                self._play_forward()

        @v.bind_key('b', overwrite=True)
        def _toggle_reverse(viewer):
            if self.is_playing:
                self._stop_play()
            else:
                self._play_reverse()

        for key, delta in [('Left', -1), ('Right', 1), ('Shift-Left', -10),
                           ('Shift-Right', 10), ('Control-Left', -100),
                           ('Control-Right', 100)]:
            v.bind_key(key, (lambda d: (lambda viewer: self._jump_frames(d)))(delta),
                       overwrite=True)
        for key, spd in zip('123456', [0.25, 0.5, 1, 2, 4, 8]):
            v.bind_key(key, (lambda s: (lambda viewer: self._set_speed(s)))(spd),
                       overwrite=True)

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
        self.status.setText("Added a cut at frame %d (not one the algorithm "
                            "proposed)." % f)
        self._refresh_candidates(); self._refresh_segments()

    def _reset(self):
        self.boundaries = list(self.original_boundaries)
        self.status.setText("Back to the algorithm's cuts.")
        self._refresh_candidates(); self._refresh_segments()

    def _save(self):
        if not self.seg_path or not self.boundaries:
            return
        if len(self.boundaries) < 2:
            QMessageBox.warning(self, "Too few cuts",
                                "At least two cuts are needed to make a segment.")
            return

        archive = (Path(self.seg_path).parents[3] / "_archived"
                   / ("segmentation_before_human_fix_%s"
                      % datetime.now().strftime("%Y%m%d_%H%M%S")))
        try:
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.seg_path, archive / self.seg_path.name)
        except Exception as e:
            QMessageBox.critical(self, "Could not archive",
                                 "Refusing to overwrite the original: %s" % e)
            return

        seg = dict(self.seg)
        seg["boundaries"] = [int(b) for b in self.boundaries]
        seg["algo_boundaries"] = [int(b) for b in self.original_boundaries]
        seg["boundary_source"] = "human"
        seg["corrected_by"] = os.environ.get("USERNAME", os.environ.get("USER", "unknown"))
        seg["corrected_at"] = datetime.now().isoformat()
        # The reason this video was queued is answered now. Leaving it set would
        # send the video straight back here on the next pass.
        seg["needs_human_resolved"] = needs_fixing(self.seg)
        seg["needs_human"] = []
        # The guided walk's per-segment answers: which identities/boundaries
        # the human confirmed vs corrected, with the algo's original numbers.
        if self._gw_records:
            seg["guided_walk"] = [self._gw_records[k]
                                  for k in sorted(self._gw_records)]

        tmp = self.seg_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(seg, indent=2))
        os.replace(tmp, self.seg_path)

        n_added = len([b for b in self.boundaries if b not in self.original_boundaries])
        n_removed = len([b for b in self.original_boundaries if b not in self.boundaries])
        self.status.setText(
            "Saved %d cuts for %s (%d added, %d removed). Original archived."
            % (len(self.boundaries), self.video_stem, n_added, n_removed))
        self._next_video()


def main():
    """Launch the segmentation fixer over the deep-review queue."""
    import argparse
    import napari
    from mousereach.config import Paths

    ap = argparse.ArgumentParser(
        description="Correct a video's segment cuts, using the candidate tray "
                    "advances the segmenter found but did not use.")
    ap.add_argument("--queue-dir", type=Path, default=None,
                    help="Queue of bundles to work through "
                         "(default: the deep-review queue)")
    args = ap.parse_args()

    queue = args.queue_dir or getattr(Paths, "DEEP_REVIEW", None)
    if queue is None:
        print("[FAIL] no deep-review queue configured; pass --queue-dir")
        return 1

    print("Segmentation fixer over %s" % queue)
    viewer = napari.Viewer(title="MouseReach -- fix segmentation")
    widget = FixSegmentationWidget(viewer, queue)
    viewer.window.add_dock_widget(widget, name="Fix segmentation", area="right")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
