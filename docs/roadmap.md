# Roadmap: from letter recognizer to fingerspelling tutor

## Where the project is now

The realtime pipeline works end to end: webcam → MediaPipe HandLandmarker →
skeleton render → CNN → smoothed letter prediction, with a sidebar UI showing
the detected letter, confidence, and the exact model input.
(This document replaces the original `realtime-plan.md`, whose goals are all
complete.)

## Direction decision (2026-07-09)

Evolve the recognizer into an **ASL fingerspelling learner/tutor**, not a
dynamic word/sentence recognizer.

**Why not word/sentence recognition:** real ASL words are dynamic, two-handed,
and involve motion — they need temporal models (LSTM/Transformer over landmark
sequences) and much harder datasets (WLASL/MS-ASL). That would discard the
current pipeline and is out of scope for the course timeline. It stays here as
future work.

**Why the tutor:** it builds directly on the existing classifier and turns it
into a product. The "spell-a-word" mode gives an honest word-level story while
staying on static fingerspelling.

## Scoring design (decided)

Two complementary signals:

```
pass  = cnn_pred == target AND cnn_conf > threshold      # gate (existing CNN)
bar   = 100 * (1 - clamp(mean_landmark_dist / D_max))    # graded feedback
```

- The **CNN** (existing `best_model.h5`) decides pass/fail.
- **Landmark distance** to a stored per-letter reference pose gives a graded
  0–100 % closeness bar, plus per-joint feedback (color the worst-off joints
  red) so the learner sees *which fingers* are wrong.
- Reference poses are computed, not hand-crafted: average the normalized 21
  landmarks per letter over the training data (wrist-centered, scale-normalized
  so the comparison is translation/scale invariant) → `data/reference_poses.json`.

## Planned modules

```
src/tutor/
├── reference.py   # build/load reference poses; similarity(live, target)
├── modes.py       # state machine: Teach, Practice, TimedQuiz, SpellWord
└── session.py     # scores, streaks, timers, results log
```

The tutor is a thin state/mode layer over the existing detect → render →
classify loop in `src/live_inference.py`; the sidebar gains a mode header,
target prompt, ghost skeleton overlay, and the graded bar.

## Build order (each step demoable on its own)

1. **Reference poses + similarity metric** — foundation; also a nice
   standalone visualization.
2. **Teach mode** — target letter + ghost skeleton + live per-joint feedback.
3. **Practice mode** — free signing with live pass/fail + graded bar.
4. **Timed quiz** — prompt, timer, streak, end screen.
5. **Spell-a-word** — hold-to-commit per letter (dwell timer), build a word,
   celebrate on match.

## Portfolio extras

- Short write-up on the two-signal scoring design (why CNN + geometric
  distance beats either alone).
- Metrics view: per-letter accuracy, which letters learners struggle with,
  improvement over a session.

## Future work (explicitly out of scope for now)

- Dynamic signs (J, Z) via landmark sequences.
- True word-level recognition (WLASL/MS-ASL, temporal models).
- Browser-based demo.
