# MouseReach Algorithm Reference

> Auto-generated reference for the MouseReach analysis pipeline algorithms.
> For the most current parameter values, see the source files referenced in each section.

## Pipeline Overview

The MouseReach pipeline analyzes high-speed video (60 fps) of mice performing a single-pellet reaching task. A DeepLabCut model tracks 18 body parts frame-by-frame, producing x/y coordinates and confidence scores for each point. Three sequential algorithms process this tracking data:

1. **Segmentation** identifies the 21 pellet presentation events per session
2. **Reach Detection** identifies individual reaching attempts within each presentation
3. **Outcome Classification** determines what happened to each pellet

Each algorithm is rule-based (no machine learning beyond the DLC pose estimation). All spatial measurements use a "ruler unit" defined as the distance between SABL and SABR (scoring area bottom-left and bottom-right), which corresponds to 9mm in physical space.

---

## 1. Segmentation (v2.1.0)

**Source:** `src/mousereach/segmentation/core/segmenter_robust.py`

### What It Detects

Each session presents 21 pellets via an automated dispensing tray. The tray advances forward to position each pellet at the pillar. The algorithm detects the 21 moments when the tray advances by tracking the motion of scoring area anchor points.

### Primary Detection Method: SABL-Centered Crossing

The SABL (Scoring Area Bottom Left) point is physically attached to the tray and moves with it. The algorithm:

1. Smooths the SABL x-position with a 5-frame median filter to remove tracking noise
2. Computes frame-to-frame velocity
3. Detects frames where SABL crosses the box center (midpoint of BOXL and BOXR) with positive velocity exceeding 0.03 ruler units/frame
4. Enforces a minimum interval of 300 frames (~5 seconds) between detections to prevent duplicates

The box center crossing is evaluated within a window of -5 to +10 pixels, and the velocity threshold starts at 1.2 but is lowered to 0.8 if fewer than 21 candidates are found.

### Secondary Validation

Each primary detection is validated by checking whether other scoring area points (SABR, SATL, SATR) show correlated motion within a 3-frame alignment window. Validated detections receive higher confidence scores.

### Fallback Detection: Motion Peak Finding

If the primary method fails to find 21 boundaries (due to poor SABL tracking), a fallback method activates:

1. Computes velocity peaks across the full SABL signal
2. Finds peaks with minimum spacing of 20 seconds and prominence above the 85th percentile of the velocity distribution
3. These detections receive a confidence score of 0.5 (vs. ~0.95 for primary detections)

### Grid Fitting

If the combined primary + fallback detections still don't yield exactly 21 boundaries, the algorithm fits the candidates to an expected interval grid (1839 frames / ~30.65 seconds between presentations). It interpolates missing boundaries and removes extras based on deviation from the grid.

### Quality Assessment

Reference point stability determines overall confidence:
- BOXL/BOXR standard deviation < 5 pixels: "good" quality
- < 15 pixels: "suspect" quality
- Above 15 pixels: "bad" quality (camera vibration or tracking failure)

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Velocity threshold | 0.03 ruler/frame | Minimum tray advance speed |
| Smoothing window | 5 frames | Median filter for noise |
| Minimum interval | 300 frames | ~5 sec, prevents duplicates |
| Maximum interval | 1200 frames | ~20 sec, flags gaps |
| Expected boundaries | 21 | Standard pellet count |
| Expected interval | 1839 frames | ~30.65 sec between pellets |
| Crossing range | -5 to +10 pixels | Window around box center |
| Alignment window | 3 frames | Multi-anchor validation |

---

## 2. Reach Detection (v3.5.0)

**Source:** `src/mousereach/reach/core/reach_detector.py`

### What It Detects

Within each pellet presentation segment, the mouse may make zero or more reaching attempts through the slit. Each reach is defined by a start frame (hand emerges through slit), an apex (maximum extension), and an end frame (hand retracts). The algorithm identifies these events by tracking hand visibility while the mouse's nose is engaged at the slit.

### Detection Logic: State Machine

The detector operates as a three-state machine for each segment:

**State 1: IDLE**
The mouse is not engaged. Transitions to ENGAGED when the nose comes within 25 pixels of the slit center (midpoint of BOXL and BOXR).

**State 2: ENGAGED**
The mouse's nose is at the slit. The detector watches for hand emergence. Transitions to REACHING when any of the four hand tracking points (RightHand, RHLeft, RHOut, RHRight) reaches a DLC confidence of 0.5 or higher. This is the reach start frame.

**State 3: REACHING**
A reach is in progress. The detector tracks hand position to determine the apex (frame of maximum extension from slit) and watches for the reach end. The reach ends when the first of three conditions is met:

1. **Hand disappearance:** All four hand points drop below 0.5 confidence for 2 or more consecutive frames. This catches the common case where the hand retracts back through the slit and DLC loses tracking.

2. **Hand retraction:** The hand moves back toward the slit by more than 40% of its maximum extension distance. This catches slow retractions where DLC maintains tracking.

3. **Hand return:** The hand position returns within 5 pixels of the slit center after having been extended more than 5 pixels beyond it.

After a reach ends, the state returns to IDLE (or ENGAGED if the nose is still at the slit).

### Post-Processing

After initial detection, three post-processing steps refine the results:

**Merging:** Reaches separated by 2 or fewer frames are merged into a single reach. This handles brief DLC tracking dropouts mid-reach that would otherwise split one reach into two.

**Splitting:** Reaches longer than 25 frames (the 95th percentile of ground truth reach duration) are examined for confidence valleys. If hand confidence drops from above 0.5 to below 0.35 then rises again, the reach is split at the valley. This handles cases where the algorithm concatenates two distinct reaches.

**Filtering:** Reaches with maximum extent below -15 pixels (hand behind the slit, toward the mouse) are removed. These are grooming or paw-adjustment movements, not reaching attempts. Reaches shorter than 4 frames are also removed (ground truth analysis showed 42% of false positives were 3 frames or shorter).

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Hand confidence threshold | 0.5 | Minimum DLC confidence for hand detection |
| Nose engagement distance | 25 pixels | Maximum nose-to-slit distance |
| Minimum reach duration | 4 frames | Eliminates brief false positives |
| Hand disappearance threshold | 2 frames | Consecutive frames below confidence |
| Retraction fraction | 0.40 | 40% of max extension triggers end |
| Hand return threshold | 5 pixels | Distance from slit for "returned" |
| Gap tolerance (merge) | 2 frames | Maximum gap to merge across |
| Split threshold | 25 frames | Duration above which splitting is attempted |
| Confidence valley | 0.35 | Low confidence that triggers split |
| Minimum extent | -15 pixels | Filters behind-slit detections |

### Hand Tracking Points

The four DLC-tracked hand points and their typical visibility:
- **RightHand** — dorsal surface of the paw, most reliably tracked
- **RHLeft** — left edge of paw (from camera perspective)
- **RHOut** — outer/distal edge of paw
- **RHRight** — right edge of paw (from camera perspective)

A reach starts when ANY of these four exceeds 0.5 confidence. The reach is active as long as at least one remains above threshold.

### Known Limitations

- **DLC tracking dropout:** When the hand moves quickly or is partially occluded, DLC confidence drops below 0.5 and the algorithm cannot see the reach. This is the primary source of missed reaches (~7% false negative rate).
- **End frame bias:** DLC confidence often dips momentarily 1-2 frames before the hand fully retracts, causing the algorithm to end reaches slightly early. This introduces a systematic ~1 frame bias in kinematic measurements.
- **Ambiguous paw-at-slit:** When the mouse rests its paw at the slit edge without extending through, all four hand points may briefly exceed 0.5 confidence. The minimum duration (4 frames) and minimum extent (-15 pixels) filters catch most but not all of these.

---

## 3. Outcome Classification (v2.4.4)

**Source:** `src/mousereach/outcomes/core/pellet_outcome.py`

### What It Classifies

For each pellet presentation segment, the algorithm determines what happened to the pellet:

| Outcome | Definition |
|---------|------------|
| **retrieved** | Mouse successfully grasped and consumed the pellet |
| **displaced_sa** | Mouse knocked the pellet off the pillar; pellet remains in the scoring area |
| **displaced_outside** | Mouse knocked the pellet outside the scoring area |
| **untouched** | Pellet remained on the pillar throughout the segment |
| **no_pellet** | No pellet was detected (operator error or dispensing failure) |
| **uncertain** | Tracking data is ambiguous; requires human review |

### Detection Logic: Four-Stage Progressive Validation

The classifier evaluates each segment through four stages of increasing specificity. Each stage can produce a final classification or pass to the next stage for further analysis.

**Stage 0: Early Retrieval Detection**

Checks whether the pellet disappeared while a paw was nearby:
- Pellet visibility (DLC confidence) must drop from above 0.8 to below 0.3
- At least one hand point must be within 30 pixels of the pellet position at the time of disappearance
- If the pellet was more than 0.4 ruler units from the pillar when grabbed, it's classified as `displaced_outside` rather than `retrieved`

**Stage 1: Initial Hypothesis**

Evaluates the overall pellet trajectory:
- **Pellet disappeared** (visibility below 10% of segment): Likely `retrieved`
- **Pellet never appeared** (no frames above 0.5 confidence): `no_pellet`
- **Pellet started far from pillar** (>0.3 ruler units at segment start): Flagged as potential operator error, classified as `untouched`
- **Otherwise:** Passes to Stage 2 for displacement analysis

**Stage 2: Feature Validation**

Two key checks:

*Sustained Displacement Detection:* The pellet's distance from the pillar must exceed 0.30 ruler units (~2.7mm) for at least 10 consecutive frames. Brief excursions (e.g., from tracking jitter) are ignored. The algorithm also identifies the displacement onset frame by looking backward for the moment when pellet confidence dropped (indicating paw occlusion).

*Paw Proximity Verification:* For any detected displacement, the algorithm looks back 30 frames from the displacement onset to confirm a hand point was within 30 pixels of the pellet. Displacement without paw proximity suggests tray wobble (the tray advancing caused apparent pellet motion) rather than a real interaction. If no paw was nearby and the pellet ends near the pillar, the classification is `untouched`.

**Stage 3: Temporal Consistency**

Resolves ambiguous cases from Stage 2:
- **Displaced then disappeared:** If the pellet was displaced but then its visibility drops to zero, it was likely retrieved after displacement. Checks for eating signature (see below).
- **Pellet moved with tray:** If the pellet's position relative to the scoring area anchor points remained constant while its absolute position changed, the motion was from tray advancement, not mouse interaction. Classified as `untouched`.
- **Pellet displaced then returned:** If the pellet moves away from pillar then returns, checks for eating signature before classifying.

*Eating Signature Detection:* A strong indicator of retrieval. The algorithm looks for sustained periods (30+ frames) where the nose is retracted from the slit (>30 pixels from BOXR) AND a hand point is near the nose/mouth (<30 pixels). This pattern indicates the mouse is holding and eating the pellet.

**Stage 4: Confidence and Interaction Frame**

Assigns a confidence score and identifies the interaction frame (the frame when the pellet was first affected by the mouse). The interaction frame is used to determine the causal reach — which reaching attempt caused the pellet outcome. The causal reach is the most recent reach whose temporal window contains the interaction frame.

### Spatial Reference: Pillar Geometry

The pellet sits on a pillar whose position is calculated geometrically from the scoring area points. The pillar is located at the apex of a 55-degree isosceles triangle whose base is the SABL-SABR line, at a perpendicular distance of 0.944 ruler units (~8.5mm) from that line. All "distance from pillar" measurements reference this calculated position.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pellet confidence threshold | 0.5 | Minimum DLC confidence for pellet tracking |
| SA point confidence threshold | 0.8 | Minimum for scoring area reference points |
| On-pillar threshold | 0.20 ruler | ~1.8mm, pellet considered on pillar |
| Displaced threshold | 0.25 ruler | ~2.25mm, minimum for displacement |
| Sustained displacement duration | 10 frames | Minimum for real displacement |
| Sustained displacement distance | 0.30 ruler | ~2.7mm, minimum distance |
| Paw proximity threshold | 30 pixels | Hand near pellet for grab detection |
| Eating distance threshold | 30 pixels | Hand-to-nose for eating signature |
| Eating duration | 30 frames | Minimum frames for eating behavior |
| Pillar perpendicular distance | 0.944 ruler | ~8.5mm, geometric constant |
| Skip start frames | 20 | Ignore first 20 frames of segment |
| Skip end frames | 30 | Ignore last 30 frames of segment |

### Known Limitations

- **Tray wobble false positives:** When the tray advances, the pellet moves in absolute coordinates even if it stays on the pillar. The tray-relative motion filter (Stage 3) catches most cases but subtle wobble during slow tray movement can trigger false `displaced_sa` classifications.
- **Pellet occlusion by paw:** When the mouse's paw covers the pellet, DLC confidence drops, which can look like pellet disappearance (retrieval). The paw proximity check (Stage 2) helps distinguish, but fast grabs where the paw is near the pellet for only 1-2 frames can be missed.
- **Multiple reaches per segment:** When several reaches occur before the outcome, attributing the correct causal reach is ambiguous. The algorithm assigns the most recent reach, which may not be correct if an earlier reach displaced the pellet.

---

## Physical Geometry Reference

All spatial measurements use ruler units (1 ruler = SABL-to-SABR distance = 9mm).

| Measurement | Ruler Units | Millimeters | Description |
|-------------|-------------|-------------|-------------|
| Ruler (SABL-SABR) | 1.000 | 9.0 | Reference distance |
| Pillar diameter | 0.458 | 4.125 | Pellet platform |
| Pellet diameter | 0.278 | ~2.5 | Sucrose pellet |
| Pillar to SA edge | 1.069 | 9.618 | Pillar center to scoring area |
| Pillar perpendicular | 0.944 | 8.5 | Pillar distance from SA baseline |
| SABR to SATR | 1.667 | 15.0 | Scoring area height |

---

## Current Performance (as of 2026-01-30, 19 GT videos)

| Metric | Value | Videos |
|--------|-------|--------|
| Segmentation boundary recall | 98.7% | 19 |
| Reach detection precision | 81.0% | 16 |
| Reach detection recall | 93.1% | 16 |
| Reach detection F1 | 85.2% | 16 |
| Outcome classification accuracy | 95.8% | 19 |

Run `python -m mousereach.eval.report_cli` to regenerate these numbers and produce plots.

---

## Version History

| Component | Version | Key Changes |
|-----------|---------|-------------|
| Segmentation | v2.1.0 | SABL-centered crossing with multi-strategy fallback |
| Reach Detection | v3.5.0 | Data-driven thresholds from 321 ground truth reaches |
| Outcome Classification | v2.4.4 | Improved pellet start/end position handling |

---

*This document describes the algorithms as implemented in code. For the source of each parameter value, see the files referenced at the top of each section.*