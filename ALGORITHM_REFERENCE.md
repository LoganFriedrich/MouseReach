# MouseReach Algorithm Reference

This page gives the overview of the four algorithms and the version each one has. The
detailed descriptions are in:

- `docs/ALGO_1_SEGMENTATION.md` -- segmentation
- `docs/ALGO_2_REACH_DETECTION.md` -- reach detection (where each reach starts and ends)
- `docs/ALGO_3_OUTCOME_CASCADE.md` -- outcome classification
- `docs/ALGO_4_REACH_ASSIGNMENT.md` -- which reach caused each pellet's outcome
- `docs/KINEMATICS_FIELDS.md` -- the kinematic values computed for each reach

## Pipeline Overview

The MouseReach pipeline analyzes video of mice performing a single-pellet reaching task. A DeepLabCut model tracks body parts frame by frame, producing x/y coordinates and a confidence score (likelihood) for each point. Four algorithms then run in order on this tracking data:

1. **Segmentation** finds the 21 cuts on a pillar tray; the 20 stretches between them are the 20 pellet presentations. Easy and Flat tray videos are not segmented: the whole video is one span.
2. **Reach Detection** finds every reach in the whole video. It is a trained classifier (a scikit-learn gradient-boosted model) that scores every frame; it is not a set of hand-written rules.
3. **Outcome Classification** decides what happened to each pellet (retrieved, displaced in the scoring area, or untouched). It is an ordered cascade of 33 rule-based stages; a pellet no stage can decide is sent to a person for review (triage).
4. **Reach Assignment** decides which reach caused each pellet's outcome.

Kinematics (speeds, distances, timing) are then computed for each reach. The kinematics read the frame rate from each video (or from the pose tool's record of it) and assume no fixed rate. (Segmentation still records a fixed 60 frames per second in its output file; it does not measure the rate.)

All spatial measurements use a "ruler unit" defined as the distance between SABL and SABR (scoring area bottom-left and bottom-right), which corresponds to 9mm in physical space.

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

## Evaluation Results

Accuracy figures measured against a lab's own hand-labeled videos describe that lab's
data, not the tool, so they are not kept in this repository. Keep your evaluation
results with your own data.

---

## Version Summary

Versions that run today (each is the version constant in the code):

| Component | Version | Where it is described |
|-----------|---------|-----------------------|
| Segmentation | 2.2.4 | `docs/ALGO_1_SEGMENTATION.md` |
| Reach Detection | 8.1.0 (v8 trained classifier) | `docs/ALGO_2_REACH_DETECTION.md` |
| Outcome Classification | 6.1.0 (33-stage v6 cascade) | `docs/ALGO_3_OUTCOME_CASCADE.md` |
| Reach Assignment | 2.1.0 | `docs/ALGO_4_REACH_ASSIGNMENT.md` |
| Kinematics extractor | 2.6.0 | `docs/KINEMATICS_FIELDS.md` |

