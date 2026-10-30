"""
Draws the graphs from data/experiments.json (run experiments.py first).

  graphs/learning_curve.png   test accuracy while training, with and without augmentation
  graphs/per_digit.png        accuracy for each digit, with 95% confidence intervals
  graphs/confusion.png        which digit each mistake was mistaken for
  graphs/summary.txt          the numbers behind the graphs

Run:  python3 graphs.py
Needs numpy and matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(HERE, "..", "data", "experiments.json")
GRAPHS_DIR = os.path.join(HERE, "..", "graphs")

BACKGROUND = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_LIGHT = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # with augmentation
ORANGE = "#eb6834"    # without augmentation
BLUE_SCALE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ONE_PASS = 937 * 64   # training images in one pass through MNIST

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": TEXT,
    "axes.edgecolor": TEXT_LIGHT, "axes.labelcolor": TEXT, "axes.facecolor": BACKGROUND,
    "figure.facecolor": BACKGROUND, "xtick.color": TEXT_LIGHT, "ytick.color": TEXT_LIGHT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def save(fig, filename):
    fig.savefig(os.path.join(GRAPHS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def learning_curve(with_aug, without_aug):
    """Average of the 5 seeds as a line, shaded band from the lowest to the highest seed."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for runs, color, label in ((without_aug, ORANGE, "Without augmentation"),
                               (with_aug, BLUE, "With augmentation")):
        seen = np.array([point[0] for point in runs[0]["curve"]], float)
        acc = np.array([[point[1] for point in run["curve"]] for run in runs]) * 100
        seen[0] = 32    # a log axis can't show 0, so "before training" sits at 32
        mean = acc.mean(0)
        ax.fill_between(seen, acc.min(0), acc.max(0), color=color, alpha=0.18, linewidth=0)
        ax.plot(seen, mean, color=color, linewidth=2, label=label)
        label_offset = 7 if runs is without_aug else -7
        ax.annotate(f"{mean[-1]:.1f}%", (seen[-1], mean[-1]), xytext=(6, label_offset),
                    textcoords="offset points", va="center", color=TEXT, fontsize=11)

    ax.axvline(ONE_PASS, color=TEXT_LIGHT, linewidth=1, linestyle="--")
    ax.text(ONE_PASS * 1.08, 14, "augmentation starts\n(pass 2)", color=TEXT_LIGHT, fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([32, 100, 1000, 10000, 100000, 479744])
    ax.set_xticklabels(["0", "100", "1,000", "10,000", "100,000", "479,744"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Training images processed (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy during training (mean of 5 seeds; band = lowest to highest seed)",
                 loc="left", fontsize=13)
    ax.text(40, 40, "Both versions are trained the same way\nuntil pass 2, so the lines overlap there.",
            color=TEXT_LIGHT, fontsize=10)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "learning_curve.png")


def per_digit(with_aug, without_aug, images_per_digit):
    """Dots with 95% confidence intervals. Dots instead of bars because the axis
    doesn't start at 0, and bars would make the gaps look bigger than they are."""
    summary = []
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bar_gap = 0.3
    groups = ((with_aug, BLUE, "With augmentation"), (without_aug, ORANGE, "Without augmentation"))
    for k, (runs, color, label) in enumerate(groups):
        acc = np.array([run["perDigit"] for run in runs]).mean(0)
        ci = 1.96 * np.sqrt(acc * (1 - acc) / images_per_digit)
        x = np.arange(10) + (k - 0.5) * bar_gap
        ax.errorbar(x, acc * 100, yerr=ci * 100, fmt="o", color=color, markersize=8, ecolor=color,
                    elinewidth=2, capsize=4, label=label, zorder=3)
        digits = ", ".join(f"{d}: {a * 100:.1f}% ±{c * 100:.1f}" for d, (a, c) in enumerate(zip(acc, ci)))
        summary.append(f"{label} per digit (mean of 5 seeds): {digits}")

    ax.set_xticks(range(10))
    ax.set_ylim(88, 100)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy for each digit (mean of 5 seeds; lines = 95% confidence interval)",
                 loc="left", fontsize=13, pad=30)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=2)
    save(fig, "per_digit.png")
    return summary


def confusion_matrix(with_aug):
    """Average mistakes over the 5 augmented runs. Correct answers are greyed out
    so the colors only show the mistakes."""
    counts = np.array([run["confusion"] for run in with_aug]).mean(0)
    mistakes = counts.copy()
    np.fill_diagonal(mistakes, np.nan)
    most = np.nanmax(mistakes)

    fig, ax = plt.subplots(figsize=(7.6, 7))
    ax.grid(False)
    ax.imshow(mistakes, cmap=LinearSegmentedColormap.from_list("blues", BLUE_SCALE), vmin=0, vmax=most)
    for true in range(10):
        for guess in range(10):
            n = counts[true, guess]
            if true == guess:
                ax.add_patch(plt.Rectangle((guess - .5, true - .5), 1, 1, color="#ecebe7"))
                ax.text(guess, true, f"{n:.0f}", ha="center", va="center", fontsize=9, color=TEXT_LIGHT)
            elif n >= 0.5:
                on_dark_square = n > most * 0.55
                ax.text(guess, true, f"{n:.0f}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if on_dark_square else TEXT)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Network's guess")
    ax.set_ylabel("True digit")
    ax.set_title("Mistakes on the 10,000 test images\n(with augmentation, average of 5 seeds; grey = correct)",
                 loc="left", fontsize=12)
    save(fig, "confusion.png")

    worst = sorted(((counts[t, g], t, g) for t in range(10) for g in range(10) if t != g), reverse=True)[:8]
    return "Most common mistakes (true -> guess, average count): " + \
        ", ".join(f"{t}->{g}: {n:.1f}" for n, t, g in worst)


def seed_comparison(seeds, with_aug, without_aug):
    """Paired t-test: each seed is trained both ways, so compare them seed by seed."""
    a = np.array([run["accuracy"] for run in with_aug]) * 100
    b = np.array([run["accuracy"] for run in without_aug]) * 100
    diff = a - b
    std_error = diff.std(ddof=1) / np.sqrt(len(diff))
    t = diff.mean() / std_error
    half_width = 2.776 * diff.std(ddof=1) / np.sqrt(len(diff))    # 2.776 = t critical value for 4 degrees of freedom
    low, high = diff.mean() - half_width, diff.mean() + half_width
    return [
        "Seeds: " + ", ".join(map(str, seeds)),
        "With augmentation:    " + ", ".join(f"{v:.2f}%" for v in a) + f"   mean {a.mean():.2f}%, SD {a.std(ddof=1):.2f}",
        "Without augmentation: " + ", ".join(f"{v:.2f}%" for v in b) + f"   mean {b.mean():.2f}%, SD {b.std(ddof=1):.2f}",
        f"Difference (with - without), same seeds: mean {diff.mean():+.2f} points, "
        f"95% CI {low:+.2f} to {high:+.2f}, paired t(4) = {t:.2f}",
    ]


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    with_aug = [run for run in results["runs"] if run["augmented"]]
    without_aug = [run for run in results["runs"] if not run["augmented"]]
    images_per_digit = np.array(with_aug[0]["confusion"]).sum(1)

    learning_curve(with_aug, without_aug)
    summary = per_digit(with_aug, without_aug, images_per_digit)
    summary.append(confusion_matrix(with_aug))
    summary = seed_comparison(results["seeds"], with_aug, without_aug) + summary

    with open(os.path.join(GRAPHS_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
