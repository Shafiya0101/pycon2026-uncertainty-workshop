"""Large-font, colour-independent plots (readable from the back row)."""
import matplotlib.pyplot as plt
import numpy as np
from .policy import risk_coverage_curve


def set_style():
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
                         "legend.fontsize": 14, "figure.figsize": (9, 5.5),
                         "axes.grid": True, "grid.alpha": 0.3})


def plot_region_grid(summary, title="Error rate per fingerprint region"):
    """2x2 grid. Each cell is labelled with text (not colour only)."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    cells = {"Confident": (0, 0), "Shallow": (0, 1), "Surface-varying": (1, 0), "Gap-like": (1, 1)}
    for name, (r, c) in cells.items():
        n, err = int(summary.loc[name, "n"]), summary.loc[name, "error_rate"]
        shade = 0.9 if n == 0 or np.isnan(err) else 1 - 0.7 * err
        ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, color=str(shade), ec="black", lw=2))
        txt = "empty" if n == 0 else f"{err:.0%} wrong\n(n={n})"
        ax.text(c + .5, 1 - r + .5, f"{name}\n{txt}", ha="center", va="center", fontsize=16)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_xticks([.5, 1.5]); ax.set_yticks([.5, 1.5])
    ax.set_xticklabels(["low SE", "high SE"]); ax.set_yticklabels(["high SC", "low SC"])
    ax.grid(False); ax.set_title(title)
    return fig


def plot_risk_coverage(df, label_col, scores, extra_points=None):
    """Risk-coverage curves; every line has a distinct style AND a direct label."""
    set_style()
    fig, ax = plt.subplots()
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (1, 1))]
    for s, ls in zip(scores, styles):
        cov, acc = risk_coverage_curve(df[s], df[label_col])
        ax.plot(cov, acc, ls=ls, lw=2.5, label=s)
        ax.text(cov[len(cov) // 2], acc[len(acc) // 2], s, fontsize=11)
    for (cov, acc, name, marker) in (extra_points or []):
        ax.scatter([cov], [acc], s=160, marker=marker, color="black", zorder=5, label=name)
    ax.set_xlabel("Coverage (fraction answered)"); ax.set_ylabel("Accuracy on answered")
    ax.legend(); ax.set_title("Accuracy vs coverage")
    return fig
