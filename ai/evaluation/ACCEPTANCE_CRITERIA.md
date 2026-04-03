# Phase 1 evaluation — acceptance criteria

This document defines **proposed** pass/fail thresholds for Phase 1 model quality on the held-out test set (or agreed primary evaluation split). **These thresholds must be reviewed and agreed with the team** before any model is labeled **Phase 1 complete**. Until then, treat them as a working baseline, not a binding gate.

## Required metrics (reported by `ai/evaluation/evaluate.py`)

Evaluation must report at minimum: overall accuracy, per-class precision/recall/F1 for all seven classes, macro-averaged F1, weighted F1, and a normalized 7×7 confusion matrix (see `metrics.json` and `confusion_matrix.png`).

## Phase 1 pass thresholds (minimum)

| Criterion | Threshold | Notes |
|-----------|-----------|--------|
| Macro-averaged F1 | ≥ **0.75** | Training logs may use `val_macro_f1`; `evaluate.py` writes test **`f1_macro`** in `metrics.json` — same numerical bar on the agreed holdout |
| Overall accuracy | ≥ **0.70** | Headline quality bar |
| Per-class recall (all 7 classes) | ≥ **0.55** each | No class may fall below this floor |

If any of the above fails, the run does **not** meet Phase 1 acceptance under this rubric (pending team sign-off).

## Repetition subclasses — data caveat

The three repetition-related classes — **part_word_repetition**, **phrase_repetition**, **word_repetition** — may underperform early on because of **limited and/or imbalanced data** relative to other classes.

- **Team agreement**: If recall for any of these three drops **below 0.50**, `evaluate.py` records a structured **`data_warning`** in `metrics.json`. That flag is informational; it does **not** replace the global Phase 1 thresholds above unless the team explicitly relaxes or amends criteria for Phase 1.
- **Action**: Review data collection, labeling, and class balance for these subclasses before changing the model or declaring Phase 1 complete.

## Sign-off

Record team agreement (date, owners) in your experiment log or release notes when locking Phase 1 criteria for production or stakeholder demos.
