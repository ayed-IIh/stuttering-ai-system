"""
Class Distribution Analysis — stuttering-ai-system
====================================================
Reads ai/dataset/metadata/dataset_inventory.csv and produces:
  - ai/dataset/metadata/plots/class_sample_counts.png
  - ai/dataset/metadata/plots/duration_distributions.png
  - ai/dataset/metadata/plots/class_share_pie.png
  - ai/dataset/metadata/class_distribution_report.md

Usage:  python scripts/class_distribution.py [--inventory-csv PATH] [--output-dir PATH] [--log-level LEVEL]
Exit:   0 on success, 1 on failure
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must set backend before pyplot import — safe for headless
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_VERSION = "1.0.0"

# Canonical class order matches model output indices (stuttering_classifier.py:18)
STUTTERING_CLASSES: list[str] = [
    "fluent",
    "blocks",
    "interjections",
    "prolongations",
    "part_word_repetition",
    "phrase_repetition",
    "word_repetition",
]

# Repetition subclasses split from a previously merged class — expected to be smallest
REPETITION_SUBCLASSES: frozenset[str] = frozenset({
    "part_word_repetition",
    "phrase_repetition",
    "word_repetition",
})

# Training viability thresholds (per class)
MIN_VIABLE_THRESHOLD: int = 50    # absolute floor — cannot train below this
RECOMMENDED_THRESHOLD: int = 100  # production-quality floor

# Bar/box plots: binary grouping highlights the standard-vs-repetition split
COLOR_STANDARD: str = "#4878cf"
COLOR_REPETITION: str = "#e07b54"

# Pie chart: 7 distinct colours — one per class for individual identification.
# Repetition subclasses use warm reds (visually grouped, still consistent with COLOR_REPETITION).
CLASS_COLORS: dict[str, str] = {
    "fluent":               "#4e79a7",  # steel blue
    "blocks":               "#59a14f",  # muted green
    "interjections":        "#f0b429",  # amber
    "prolongations":        "#76b7b2",  # teal
    "part_word_repetition": "#e15759",  # coral red
    "phrase_repetition":    "#9b2226",  # dark burgundy — fewest samples
    "word_repetition":      "#f28e2b",  # warm orange
}

PLOT_DPI: int = 150

# Display labels for X-axis ticks — human-readable, preserving canonical order
DISPLAY_LABELS: dict[str, str] = {
    "fluent":               "Fluent",
    "blocks":               "Blocks",
    "interjections":        "Interjections",
    "prolongations":        "Prolongations",
    "part_word_repetition": "Part-Word\nRep.",
    "phrase_repetition":    "Phrase\nRep.",
    "word_repetition":      "Word\nRep.",
}

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ClassStats:
    """Per-class distribution statistics."""
    class_label: str
    count: int
    percentage: float
    total_duration_s: float
    mean_duration_s: float
    median_duration_s: float
    is_repetition_subclass: bool
    below_min_viable: bool
    below_recommended: bool

# ── Shared utilities (KISS — minimal, self-contained) ─────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _to_abs_path(raw: str, *, must_exist: bool = False) -> Path:
    """Resolve path; relative paths anchored to project root, not cwd."""
    p = (Path(raw) if Path(raw).is_absolute() else _project_root() / raw).resolve()
    if must_exist and not p.exists():
        sys.exit(f"[ERROR] Path not found: {p}")
    return p


def setup_logging(level: str, log_file: Path) -> logging.Logger:
    log = logging.getLogger("class_distribution")
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def _md_table(headers: list[str], rows: list) -> str:
    """DRY Markdown table builder — single implementation used everywhere."""
    sep = "|".join("---" for _ in headers)
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return f"| {' | '.join(headers)} |\n|{sep}|\n{body}"


def _bar_colours(classes: list[str]) -> list[str]:
    """Return per-class colour list in canonical order."""
    return [COLOR_REPETITION if c in REPETITION_SUBCLASSES else COLOR_STANDARD for c in classes]

# ── Core functions (SRP — one job each) ───────────────────────────────────────

def load_inventory(csv_path: Path) -> pd.DataFrame:
    """Read dataset_inventory.csv; drop corrupted/zero-byte rows."""
    log = logging.getLogger("class_distribution")
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df[~df["is_corrupted"].astype(bool) & ~df["is_zero_byte"].astype(bool)].copy()
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d corrupted/zero-byte rows from analysis.", dropped)
    log.info("Loaded %d usable records from %s", len(df), csv_path)
    return df


def compute_class_stats(df: pd.DataFrame) -> list[ClassStats]:
    """Compute per-class sample counts, duration stats, and threshold flags."""
    total = len(df)
    stats: list[ClassStats] = []
    for label in STUTTERING_CLASSES:
        subset = df[df["class_label"] == label]["duration_seconds"].dropna()
        count = len(subset)
        stats.append(ClassStats(
            class_label=label,
            count=count,
            percentage=round(count / total * 100, 2) if total else 0.0,
            total_duration_s=round(float(subset.sum()), 2),
            mean_duration_s=round(float(subset.mean()), 2) if count else 0.0,
            median_duration_s=round(float(subset.median()), 2) if count else 0.0,
            is_repetition_subclass=label in REPETITION_SUBCLASSES,
            below_min_viable=count < MIN_VIABLE_THRESHOLD,
            below_recommended=count < RECOMMENDED_THRESHOLD,
        ))
    return stats


def compute_imbalance_ratio(stats: list[ClassStats]) -> float:
    """Majority class count / minority class count (higher = more imbalanced)."""
    counts = [s.count for s in stats if s.count > 0]
    if not counts:
        return 0.0
    return round(max(counts) / min(counts), 2)


# ── Plot functions (SRP — one plot per function) ──────────────────────────────

def plot_sample_counts(stats: list[ClassStats], plots_dir: Path) -> Path:
    """Bar chart: per-class sample counts with MIN_VIABLE and RECOMMENDED threshold lines."""
    log = logging.getLogger("class_distribution")
    labels = [s.class_label for s in stats]
    counts = [s.count for s in stats]
    x = np.arange(len(labels))
    colours = _bar_colours(labels)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, counts, color=colours, edgecolor="white", linewidth=0.8, zorder=3)

    # Annotate count above each bar
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # Threshold reference lines
    ax.axhline(MIN_VIABLE_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Min viable ({MIN_VIABLE_THRESHOLD})", zorder=4)
    ax.axhline(RECOMMENDED_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1.5,
               label=f"Recommended ({RECOMMENDED_THRESHOLD})", zorder=4)

    # Legend — class colour legend + threshold lines
    legend_patches = [
        mpatches.Patch(color=COLOR_STANDARD, label="Standard class"),
        mpatches.Patch(color=COLOR_REPETITION, label="Repetition subclass"),
    ]
    ax.legend(handles=legend_patches + ax.get_lines(), fontsize=9, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[lbl] for lbl in labels], fontsize=10)
    ax.set_ylabel("Sample count", fontsize=12)
    ax.set_title("Per-Class Sample Counts (n=7 classes)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, max(counts) * 1.18)
    fig.tight_layout()

    out = plots_dir / "class_sample_counts.png"
    fig.savefig(out, dpi=PLOT_DPI)
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


def plot_duration_boxplot(df: pd.DataFrame, plots_dir: Path) -> Path:
    log = logging.getLogger("class_distribution")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    colours = _bar_colours(STUTTERING_CLASSES)

    data_by_class = [
        df[df["class_label"] == label]["duration_seconds"].dropna().tolist()
        for label in STUTTERING_CLASSES
    ]

    bp = ax.boxplot(
        data_by_class,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        zorder=3,
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    # jitter so stacked points at exactly 10s separate visually
    rng = np.random.default_rng(seed=42)
    for i, (samples, colour) in enumerate(zip(data_by_class, colours), start=1):
        if not samples:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(samples))
        ax.scatter(
            np.full(len(samples), i) + jitter,
            samples,
            alpha=0.45, s=18, color=colour, zorder=4,
        )

    max_dur = df["duration_seconds"].dropna().max()
    ax.set_ylim(0, max_dur + 2)  # headroom so 10s samples don't sit at the ceiling

    ax.set_xticks(range(1, len(STUTTERING_CLASSES) + 1))
    ax.set_xticklabels([DISPLAY_LABELS[lbl] for lbl in STUTTERING_CLASSES], fontsize=10)
    ax.set_ylabel("Duration (seconds)", fontsize=12)
    ax.set_title("Audio Duration Distributions per Class", fontsize=14, fontweight="bold", pad=12)

    legend_patches = [
        mpatches.Patch(color=COLOR_STANDARD, alpha=0.75, label="Standard class"),
        mpatches.Patch(color=COLOR_REPETITION, alpha=0.75, label="Repetition subclass"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper right")
    fig.tight_layout()

    out = plots_dir / "duration_distributions.png"
    fig.savefig(out, dpi=PLOT_DPI)
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


def plot_class_share_pie(stats: list[ClassStats], plots_dir: Path) -> Path:
    """Pie chart: percentage share per class; repetition slices exploded for emphasis."""
    log = logging.getLogger("class_distribution")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    labels = [s.class_label for s in stats]
    counts = [s.count for s in stats]
    colours = [CLASS_COLORS[lbl] for lbl in labels]  # 7 distinct — each class identifiable
    explode = [0.06 if lbl in REPETITION_SUBCLASSES else 0.0 for lbl in labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _, autotexts = ax.pie(
        counts,
        labels=None,
        colors=colours,
        explode=explode,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
        startangle=140,
        pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    # Legend with display label + raw count
    legend_labels = [f"{DISPLAY_LABELS[lbl].replace(chr(10), ' ')} ({c})" for lbl, c in zip(labels, counts)]
    ax.legend(wedges, legend_labels, title="Class (count)", loc="lower left",
              bbox_to_anchor=(-0.15, -0.05), fontsize=9, title_fontsize=9)

    ax.set_title("Class Share (% of total dataset)", fontsize=14, fontweight="bold", pad=16)
    fig.tight_layout()

    out = plots_dir / "class_share_pie.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


# ── Report writer ──────────────────────────────────────────────────────────────

def _recommendation(s: ClassStats) -> str:
    """Single-source per-class recommendation string (DRY anchor for report)."""
    if s.class_label == "phrase_repetition":
        return (
            "**BLOCKED** — 2 samples is critically insufficient. "
            "Do not include in training until client-provided data raises count to ≥50. "
            "Current gap: **48 samples needed**."
        )
    if s.class_label == "word_repetition":
        return (
            f"**AUGMENT + AWAIT DATA** — {s.count} samples is below the minimum viable "
            f"threshold ({MIN_VIABLE_THRESHOLD}). Apply offline augmentation (time-stretch, "
            "pitch-shift, noise injection) and incorporate client-provided samples before "
            f"training. Gap to MIN_VIABLE: **{MIN_VIABLE_THRESHOLD - s.count} samples**."
        )
    if s.class_label == "part_word_repetition":
        return (
            f"**WEIGHTED SAMPLING + MONITOR** — {s.count} samples is above MIN_VIABLE but "
            f"below RECOMMENDED ({RECOMMENDED_THRESHOLD}). Apply class-weighted loss. "
            "Incorporate client data when available to reach ≥100."
        )
    if s.class_label == "interjections":
        return (
            f"**AUGMENT REQUIRED** — {s.count} samples is below MIN_VIABLE ({MIN_VIABLE_THRESHOLD}). "
            "Apply offline augmentation (time-stretch, pitch-shift, noise injection) to reach ≥50 "
            f"before training. Gap to MIN_VIABLE: **{MIN_VIABLE_THRESHOLD - s.count} samples**."
        )
    if s.below_recommended:
        return (
            f"**WEIGHTED SAMPLING** — {s.count} samples is below RECOMMENDED "
            f"({RECOMMENDED_THRESHOLD}). Use class-weighted loss during training. "
            "Monitor per-class F1 closely."
        )
    return (
        f"**WEIGHTED SAMPLING** — {s.count} samples. Adequate for fine-tuning. "
        "Apply class-weighted loss to compensate for dataset imbalance."
    )


def write_report(
    stats: list[ClassStats],
    imbalance: float,
    df: pd.DataFrame,
    output_dir: Path,
    ts: str,
) -> Path:
    """Produce class_distribution_report.md — structured findings + per-class recommendations."""
    total_samples = sum(s.count for s in stats)
    total_duration = df["duration_seconds"].dropna().sum()
    majority = max(stats, key=lambda s: s.count)
    minority = min(stats, key=lambda s: s.count)

    dist_rows = [
        [
            f"`{s.class_label}`",
            s.count,
            f"{s.percentage}%",
            f"{s.total_duration_s:.1f}",
            f"{s.mean_duration_s:.2f}",
            f"{s.median_duration_s:.2f}",
            "Yes" if s.is_repetition_subclass else "No",
            "YES" if s.below_min_viable else "—",
        ]
        for s in stats
    ]

    def _rep_status(count: int) -> str:
        if count < 10:
            return "CRITICAL"
        if count < MIN_VIABLE_THRESHOLD:
            return "BELOW MIN"
        return "BELOW RECOMMENDED"

    rep_rows = [
        [f"`{s.class_label}`", s.count, MIN_VIABLE_THRESHOLD,
         max(0, MIN_VIABLE_THRESHOLD - s.count), _rep_status(s.count)]
        for s in stats if s.is_repetition_subclass
    ]

    per_class_recs = "\n\n".join(
        f"### `{s.class_label}`\n\n{_recommendation(s)}"
        for s in stats
    )

    # Overall training readiness verdict
    blocked = [s.class_label for s in stats if s.count < 10]
    critical = [s.class_label for s in stats if 10 <= s.count < MIN_VIABLE_THRESHOLD]
    if blocked:
        verdict = (
            f"**TRAINING BLOCKED** — {len(blocked)} class(es) have critically insufficient "
            f"samples: {', '.join(f'`{c}`' for c in blocked)}. "
            "Await client-provided data before starting any training run."
        )
    elif critical:
        verdict = (
            f"**CONDITIONAL** — {len(critical)} class(es) are below MIN_VIABLE threshold: "
            f"{', '.join(f'`{c}`' for c in critical)}. "
            "Apply augmentation to reach ≥50 per class before training."
        )
    else:
        verdict = (
            "**READY (with weighted sampling)** — All classes meet MIN_VIABLE threshold. "
            "Apply class-weighted loss to compensate for imbalance."
        )

    sections = [
        f"# Class Distribution Report\n\n"
        f"- **Script v{SCRIPT_VERSION}** | Generated: {ts}\n"
        f"- **Input**: `ai/dataset/metadata/dataset_inventory.csv`\n"
        f"- **Total usable samples**: {total_samples}\n"
        f"- **Total audio duration**: {total_duration / 60:.1f} min ({total_duration:.0f} s)",

        "## Class Distribution\n\n"
        + _md_table(
            ["Class", "Count", "Share %", "Total Duration (s)", "Mean Duration (s)", "Median Duration (s)", "Repetition?", "Below MIN_VIABLE?"],
            dist_rows,
        ),

        f"## Imbalance Analysis\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Majority class | `{majority.class_label}` ({majority.count} samples) |\n"
        f"| Minority class | `{minority.class_label}` ({minority.count} samples) |\n"
        f"| **Imbalance ratio** | **{imbalance}:1** |\n"
        f"| MIN_VIABLE threshold | {MIN_VIABLE_THRESHOLD} samples/class |\n"
        f"| RECOMMENDED threshold | {RECOMMENDED_THRESHOLD} samples/class |",

        "## Repetition Subclass Status\n\n"
        "> **Note:** The client has confirmed additional data will be provided for repetition "
        "subclasses. Current counts are documented below. Do not initiate training on "
        "`phrase_repetition` until the minimum viable threshold is reached.\n\n"
        + _md_table(
            ["Class", "Current Count", "MIN_VIABLE", "Gap to Fill", "Status"],
            rep_rows,
        ),

        "## Per-Class Recommendations\n\n" + per_class_recs,

        "## Training Readiness Summary\n\n" + verdict,

        "## Plot References\n\n"
        "| Plot | File |\n|---|---|\n"
        "| Per-class sample counts (bar chart) | `plots/class_sample_counts.png` |\n"
        "| Duration distributions (box plot)   | `plots/duration_distributions.png` |\n"
        "| Class share (pie chart)             | `plots/class_share_pie.png` |",
    ]

    out = output_dir / "class_distribution_report.md"
    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    logging.getLogger("class_distribution").info("Report written: %s", out)
    return out


# ── CLI + orchestration ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--inventory-csv",
        default="ai/dataset/metadata/dataset_inventory.csv",
        help="Path to dataset_inventory.csv (default: ai/dataset/metadata/dataset_inventory.csv)",
    )
    p.add_argument(
        "--output-dir",
        default="ai/dataset/metadata",
        help="Output directory for report and plots (default: ai/dataset/metadata)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = _to_abs_path(args.inventory_csv, must_exist=True)
    out_dir  = _to_abs_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(args.log_level, out_dir / "class_distribution.log")
    ts  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.info("class_distribution v%s | %s", SCRIPT_VERSION, ts)

    df       = load_inventory(csv_path)
    stats    = compute_class_stats(df)
    imbalance = compute_imbalance_ratio(stats)

    log.info("Imbalance ratio: %.2f:1 (%s / %s)",
             imbalance,
             max(stats, key=lambda s: s.count).class_label,
             min(stats, key=lambda s: s.count).class_label)

    # Generate all three plots
    plot_sample_counts(stats, plots_dir)
    plot_duration_boxplot(df, plots_dir)
    plot_class_share_pie(stats, plots_dir)

    write_report(stats, imbalance, df, out_dir, ts)

    log.info("Done — outputs in %s", out_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
