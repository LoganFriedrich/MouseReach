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
    QCheckBox, QSplitter, QSpinBox,
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

        self._build_ui()
        self._load_queue()

    # ---------------------------------------------------------------- ui

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.header = QLabel("No video loaded")
        self.header.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.why = QLabel("")
        self.why.setWordWrap(True)
        self.why.setStyleSheet("color: #e08020;")
        layout.addWidget(self.why)

        split = QSplitter(Qt.Vertical)
        layout.addWidget(split, 1)

        # --- guided walk ---------------------------------------------------
        gw_box = QGroupBox("Guided walk (one segment at a time)")
        gv = QVBoxLayout()
        gw_box.setLayout(gv)

        self.gw_header = QLabel("")
        self.gw_header.setStyleSheet("font-weight: bold;")
        self.gw_header.setWordWrap(True)
        gv.addWidget(self.gw_header)

        # Q1: identity
        r1 = QHBoxLayout()
        self.gw_q1 = QLabel("")
        self.gw_q1.setWordWrap(True)
        r1.addWidget(self.gw_q1, 1)
        self.gw_id_no = QCheckBox("No, it is actually segment:")
        r1.addWidget(self.gw_id_no)
        self.gw_id_spin = QSpinBox()
        self.gw_id_spin.setRange(0, 60)
        self.gw_id_spin.setEnabled(False)
        self.gw_id_no.toggled.connect(self.gw_id_spin.setEnabled)
        r1.addWidget(self.gw_id_spin)
        gv.addLayout(r1)

        # Q2: start boundary
        r2 = QHBoxLayout()
        self.gw_q2 = QLabel("")
        self.gw_q2.setWordWrap(True)
        r2.addWidget(self.gw_q2, 1)
        self.gw_start_no = QCheckBox("No, it starts at frame:")
        r2.addWidget(self.gw_start_no)
        self.gw_start_spin = QSpinBox()
        self.gw_start_spin.setRange(0, 10_000_000)
        self.gw_start_spin.setEnabled(False)
        self.gw_start_no.toggled.connect(self.gw_start_spin.setEnabled)
        r2.addWidget(self.gw_start_spin)
        b = QPushButton("Use current frame")
        b.setToolTip("Copy the playhead's frame into the box. Pick the frame "
                     "AFTER the scoring-area jump.")
        b.clicked.connect(lambda: self._gw_use_playhead(self.gw_start_spin,
                                                        self.gw_start_no))
        r2.addWidget(b)
        gv.addLayout(r2)
        n2 = QLabel("Operator note: the start is the frame AFTER the "
                    "scoring-area jump (the tray has finished advancing).")
        n2.setStyleSheet("color: #888;")
        n2.setWordWrap(True)
        gv.addWidget(n2)

        # Q3: end boundary
        r3 = QHBoxLayout()
        self.gw_q3 = QLabel("")
        self.gw_q3.setWordWrap(True)
        r3.addWidget(self.gw_q3, 1)
        self.gw_end_no = QCheckBox("No, it ends at frame:")
        r3.addWidget(self.gw_end_no)
        self.gw_end_spin = QSpinBox()
        self.gw_end_spin.setRange(0, 10_000_000)
        self.gw_end_spin.setEnabled(False)
        self.gw_end_no.toggled.connect(self.gw_end_spin.setEnabled)
        r3.addWidget(self.gw_end_spin)
        b = QPushButton("Use current frame")
        b.setToolTip("Copy the playhead's frame into the box. Pick the frame "
                     "BEFORE the scoring-area jump.")
        b.clicked.connect(lambda: self._gw_use_playhead(self.gw_end_spin,
                                                        self.gw_end_no))
        r3.addWidget(b)
        gv.addLayout(r3)
        n3 = QLabel("Operator note: the end is the frame BEFORE the next "
                    "scoring-area jump (the tray has not started moving yet).")
        n3.setStyleSheet("color: #888;")
        n3.setWordWrap(True)
        gv.addWidget(n3)

        nav = QHBoxLayout()
        bb = QPushButton("< Back a segment")
        bb.clicked.connect(self._gw_back)
        nav.addWidget(bb)
        nav.addStretch()
        cb = QPushButton("Confirm answers -> next segment")
        cb.setStyleSheet("font-weight: bold;")
        cb.clicked.connect(self._gw_confirm)
        nav.addWidget(cb)
        gv.addLayout(nav)

        self.gw_status = QLabel("")
        self.gw_status.setWordWrap(True)
        self.gw_status.setStyleSheet("color: #e08020;")
        gv.addWidget(self.gw_status)
        split.addWidget(gw_box)

        # --- candidates ---------------------------------------------------
        cand_box = QGroupBox("Candidate tray advances the video suggests")
        cv = QVBoxLayout()
        cand_box.setLayout(cv)
        cv.addWidget(QLabel(
            "Every timepoint the corner trackers proposed. Ticked ones are cuts. "
            "Click a row to jump there."))
        self.cand_table = QTableWidget(0, 5)
        self.cand_table.setHorizontalHeaderLabels(
            ["cut?", "frame", "time", "corners agreeing", "agreement"])
        self.cand_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cand_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cand_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cand_table.itemSelectionChanged.connect(self._jump_to_selected)
        cv.addWidget(self.cand_table)

        row = QHBoxLayout()
        for text, slot in (("Use this candidate", self._use_selected),
                           ("Drop this cut", self._drop_selected),
                           ("Add a cut at the current frame", self._add_here)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        self.only_unused = QCheckBox("Hide candidates already used")
        self.only_unused.toggled.connect(lambda _: self._refresh_candidates())
        row.addWidget(self.only_unused)
        row.addStretch()
        cv.addLayout(row)
        split.addWidget(cand_box)

        # --- resulting segments -------------------------------------------
        seg_box = QGroupBox("Segments these cuts produce")
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
        for text, slot, style in (
                ("Back to the algorithm's cuts", self._reset, ""),
                ("Save these cuts", self._save, "font-weight: bold;"),
                ("Skip this video", self._next_video, "")):
            b = QPushButton(text)
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

    def _gw_segments(self) -> List[tuple]:
        return segments_from(self.boundaries, self.n_frames)

    def _gw_load(self):
        """Show the three questions for the current segment and park the
        playhead at its algo start so the operator is looking at the right
        footage immediately."""
        segs = self._gw_segments()
        if not segs:
            self.gw_header.setText("No segments to walk (no cuts yet) -- use "
                                   "the manual controls below.")
            return
        self._gw_idx = max(0, min(self._gw_idx, len(segs) - 1))
        num, s, e = segs[self._gw_idx]
        self.gw_header.setText(
            "Segment %d of %d  --  the algorithm thinks frames %d-%d contain "
            "pellet %d (and thus that this is segment %d)."
            % (num, len(segs), s, e, num, num))
        self.gw_q1.setText(
            "Q1: Is this segment number %d (= pellet number %d)?" % (num, num))
        self.gw_q2.setText(
            "Q2: Does this segment start within %d frames of frame %d?"
            % (GUIDED_TOLERANCE_FRAMES, s))
        self.gw_q3.setText(
            "Q3: Does this segment end within %d frames of frame %d?"
            % (GUIDED_TOLERANCE_FRAMES, e))
        self.gw_id_no.setChecked(False)
        self.gw_id_spin.setValue(num)
        self.gw_start_no.setChecked(False)
        self.gw_start_spin.setValue(s)
        self.gw_end_no.setChecked(False)
        self.gw_end_spin.setValue(e)
        self.gw_status.setText("")
        self._goto(s)

    def _gw_use_playhead(self, spin, checkbox):
        try:
            f = int(self.viewer.dims.current_step[0])
        except Exception:
            return
        checkbox.setChecked(True)
        spin.setValue(f)

    def _gw_confirm(self):
        """Apply this segment's answers to the cuts, record them, advance."""
        segs = self._gw_segments()
        if not segs:
            return
        num, s, e = segs[self._gw_idx]
        rec = {"segment_num": num, "algo_start": s, "algo_end": e,
               "identity_confirmed": not self.gw_id_no.isChecked(),
               "start_confirmed": not self.gw_start_no.isChecked(),
               "end_confirmed": not self.gw_end_no.isChecked()}
        notes = []
        if self.gw_id_no.isChecked():
            rec["true_segment_num"] = int(self.gw_id_spin.value())
            notes.append(
                "identity: you say this is really segment %d -- numbering is "
                "set only by the cuts, so add or remove the missing/extra cut "
                "EARLIER in the video (manual controls below) until the "
                "numbers line up." % rec["true_segment_num"])
        if self.gw_start_no.isChecked():
            f = int(self.gw_start_spin.value())
            rec["corrected_start"] = f
            self.boundaries = move_segment_start(self.boundaries, self._gw_idx, f)
            notes.append("start moved to frame %d." % f)
        if self.gw_end_no.isChecked():
            f = int(self.gw_end_spin.value())
            rec["corrected_end"] = f
            self.boundaries = move_segment_end(self.boundaries, self._gw_idx, f)
            notes.append("end moved to frame %d (cut at %d)." % (f, f + 1))
        self._gw_records[num] = rec
        self._refresh_candidates(); self._refresh_segments()
        done = len(self._gw_records)
        total = len(self._gw_segments())
        if self._gw_idx < total - 1:
            self._gw_idx += 1
            self._gw_load()
            if notes:
                self.gw_status.setText(" ".join(notes))
        else:
            self.gw_status.setText(
                ("%s  " % " ".join(notes) if notes else "")
                + "Walk finished (%d/%d segments answered). Check the segment "
                  "table, then 'Save these cuts'." % (done, total))

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
        except Exception as e:
            self.status.setText("Could not open the video (%s); the candidate "
                                "list still works." % e)

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
