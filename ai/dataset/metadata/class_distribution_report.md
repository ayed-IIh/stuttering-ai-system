# Class Distribution Report

- **Script v1.0.0** | Generated: 2026-03-25T08:44:55Z
- **Input**: `ai/dataset/metadata/dataset_inventory.csv`
- **Total usable samples**: 199
- **Total audio duration**: 27.7 min (1659 s)

## Class Distribution

| Class | Count | Share % | Total Duration (s) | Mean Duration (s) | Median Duration (s) | Repetition? | Below MIN_VIABLE? |
|---|---|---|---|---|---|---|---|
| `fluent` | 38 | 19.1% | 361.1 | 9.50 | 10.00 | No | YES |
| `blocks` | 37 | 18.59% | 333.6 | 9.02 | 10.00 | No | YES |
| `interjections` | 12 | 6.03% | 81.3 | 6.77 | 6.85 | No | YES |
| `prolongations` | 31 | 15.58% | 251.3 | 8.11 | 10.00 | No | YES |
| `part_word_repetition` | 61 | 30.65% | 482.1 | 7.90 | 10.00 | Yes | — |
| `phrase_repetition` | 2 | 1.01% | 16.2 | 8.12 | 8.12 | Yes | YES |
| `word_repetition` | 18 | 9.05% | 133.6 | 7.42 | 8.07 | Yes | YES |

## Imbalance Analysis

| Metric | Value |
|---|---|
| Majority class | `part_word_repetition` (61 samples) |
| Minority class | `phrase_repetition` (2 samples) |
| **Imbalance ratio** | **30.5:1** |
| MIN_VIABLE threshold | 50 samples/class |
| RECOMMENDED threshold | 100 samples/class |

## Repetition Subclass Status

> **Note:** The client has confirmed additional data will be provided for repetition subclasses. Current counts are documented below. Do not initiate training on `phrase_repetition` until the minimum viable threshold is reached.

| Class | Current Count | MIN_VIABLE | Gap to Fill | Status |
|---|---|---|---|---|
| `part_word_repetition` | 61 | 50 | 0 | BELOW RECOMMENDED |
| `phrase_repetition` | 2 | 50 | 48 | CRITICAL |
| `word_repetition` | 18 | 50 | 32 | BELOW MIN |

## Per-Class Recommendations

### `fluent`

**WEIGHTED SAMPLING** — 38 samples is below RECOMMENDED (100). Use class-weighted loss during training. Monitor per-class F1 closely.

### `blocks`

**WEIGHTED SAMPLING** — 37 samples is below RECOMMENDED (100). Use class-weighted loss during training. Monitor per-class F1 closely.

### `interjections`

**AUGMENT REQUIRED** — 12 samples is below MIN_VIABLE (50). Apply offline augmentation (time-stretch, pitch-shift, noise injection) to reach ≥50 before training. Gap to MIN_VIABLE: **38 samples**.

### `prolongations`

**WEIGHTED SAMPLING** — 31 samples is below RECOMMENDED (100). Use class-weighted loss during training. Monitor per-class F1 closely.

### `part_word_repetition`

**WEIGHTED SAMPLING + MONITOR** — 61 samples is above MIN_VIABLE but below RECOMMENDED (100). Apply class-weighted loss. Incorporate client data when available to reach ≥100.

### `phrase_repetition`

**BLOCKED** — 2 samples is critically insufficient. Do not include in training until client-provided data raises count to ≥50. Current gap: **48 samples needed**.

### `word_repetition`

**AUGMENT + AWAIT DATA** — 18 samples is below the minimum viable threshold (50). Apply offline augmentation (time-stretch, pitch-shift, noise injection) and incorporate client-provided samples before training. Gap to MIN_VIABLE: **32 samples**.

## Training Readiness Summary

**TRAINING BLOCKED** — 1 class(es) have critically insufficient samples: `phrase_repetition`. Await client-provided data before starting any training run.

## Plot References

| Plot | File |
|---|---|
| Per-class sample counts (bar chart) | `plots/class_sample_counts.png` |
| Duration distributions (box plot)   | `plots/duration_distributions.png` |
| Class share (pie chart)             | `plots/class_share_pie.png` |
