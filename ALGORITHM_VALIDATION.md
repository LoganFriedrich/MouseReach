# MouseReach Algorithm Validation

**Algorithm versions:** Reach detection v5.3.0 | Outcome classification v2.4.4 | Segmentation v2.1.0
**Evaluated:** 2026-02-16 against frame-by-frame human annotation

All numbers below describe agreement between algorithm output and trained human annotators on the same videos. "Within 2 frames" means within 200 ms at 60 fps recording rate. Human annotators themselves typically disagree by 1-2 frames, so errors beyond 2 frames are unambiguously the algorithm's fault.

---

## Reach Detection

**Ground truth corpus:** 2,608 human-annotated reaches across 23 videos.

| Question | Answer | Numbers |
|----------|--------|---------|
| What fraction of real reaches did the algorithm find? | 95.7% | Found 2,495 of 2,608 human-annotated reaches |
| What fraction of real reaches did the algorithm miss? | 4.3% | 113 real reaches invisible to the pipeline |
| How many spurious "reaches" did the algorithm invent? | 494 | Detections that do not correspond to any real reaching movement |
| When it found a reach, how often was the start frame within 2 frames of the human? | 99.7% | 2,487 of 2,495 detected reaches |
| When it found a reach, how often was the end frame within 2 frames of the human? | 99.0% | 2,470 of 2,495 detected reaches |
| Combining everything: what fraction of all real reaches were both found AND correctly bounded? | 94.5% | 2,465 of 2,608 |

**Not yet measured:** Exact (0-frame) start and end match rates for the current version. Only "within 2 frames" has been broken out.

**Important caveat:** The numbers above were measured on the same 23 videos used to train the reach boundary model. Performance on new videos from new animals may be lower.

---

## Pellet Outcome Classification

**Ground truth corpus:** 400 pellet presentations across 20 human-verified videos.

| Question | Answer | Numbers |
|----------|--------|---------|
| How often did the algorithm get the outcome right? | 98.5% | 394 of 400 correct |
| How often did it correctly identify a retrieval? | 92.3% | 48 of 52 retrievals; 4 were called "displaced" instead |
| How often did it correctly identify "displaced to scoring area"? | 99.4% | 169 of 170 |
| How often did it correctly identify "untouched"? | 99.4% | 176 of 177 |
| When the mouse touched the pellet, how close was the algorithm's interaction frame to the human's? | Within 5 frames 88.2% of the time; mean error 12.7 frames | 212 pellet presentations with timing data |

**The weak spot is retrievals.** The algorithm sees the pellet move before it disappears and calls it "displaced" instead of "retrieved."

---

## Trial Segmentation

**Ground truth corpus:** 25 videos.

| Question | Answer |
|----------|--------|
| How often did the algorithm find the correct pellet-presentation boundary? | 99.0% mean detection rate |
| Timing bias? | Slightly early (fractions of a frame on average) |

Segmentation is reliable.

---

## Causal Reach Identification

| Question | Answer |
|----------|--------|
| Which reach caused the pellet outcome -- how often does the algorithm get this right? | **Unknown. Not yet measurable.** |

No review tool exposes this for human verification. Until a human can see and correct the algorithm's causal reach assignment, this link between kinematics and outcomes is unvalidated.

---

## What Each Error Means for the Science

| Error type | Consequence |
|------------|-------------|
| **Missed reach** | That reach produces no kinematics. It is absent from the dataset. If short or unusual reaches are missed more often, the dataset is biased toward the kinds of reaches the algorithm can see. |
| **Spurious "reach"** | Kinematics computed over frames with no reaching movement. Contaminates any analysis that treats all detections as real. |
| **Wrong start or end frame** | Velocity, trajectory, peak extension, duration -- all computed over the wrong temporal window. Even 1 frame of error changes the numbers. |
| **Wrong pellet outcome** | A retrieval counted as a displacement (or vice versa) corrupts success-rate calculations and any kinematic comparison between outcome types. |
| **Wrong interaction frame** | The wrong reach gets credited with causing the outcome. The kinematic profile linked to an outcome belongs to a different reach entirely. |
| **Unknown causal reach** | Cannot ask "what were the kinematics of the reach that produced this outcome?" with any certainty. |
