"""Create a publication-quality overview of the ResiliRoad method."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def box(ax, xy, width, height, title, text, color, title_color="white"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.7, edgecolor=color, facecolor="white", zorder=2,
    )
    ax.add_patch(patch)
    band = FancyBboxPatch(
        (x, y + height - 0.12), width, 0.12,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0, facecolor=color, zorder=3,
    )
    ax.add_patch(band)
    ax.text(x + width / 2, y + height - 0.06, title, ha="center", va="center",
            fontsize=10.2, fontweight="bold", color=title_color, zorder=4)
    ax.text(x + width / 2, y + (height - 0.12) / 2, text, ha="center", va="center",
            fontsize=8.8, color="#172033", linespacing=1.25, zorder=4)
    return patch


def arrow(ax, a, b, color="#46546a", label=None, bend=0.0):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                        linewidth=1.6, color=color,
                        connectionstyle=f"arc3,rad={bend}", zorder=1)
    ax.add_patch(p)
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my + 0.025, label, ha="center", va="bottom", fontsize=8,
                color=color, bbox=dict(fc="white", ec="none", pad=0.5))


fig, ax = plt.subplots(figsize=(14, 5.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

blue = "#155e75"
orange = "#c2410c"
green = "#15803d"
purple = "#6d28d9"
navy = "#172554"

box(ax, (0.025, 0.34), 0.14, 0.36, "INPUT", "Road graph  G\nedge weights + coordinates", blue)
box(ax, (0.205, 0.55), 0.16, 0.32, "BASE SPECTRUM", "Laplacian L\n$\\lambda_2(G)$ and Fiedler $u_2$", purple)
box(ax, (0.205, 0.12), 0.16, 0.32, "DISRUPTION", "Failed set S\nindependent or spatial cluster", orange)
box(ax, (0.405, 0.58), 0.17, 0.28, "FAST PRIOR", "$\\hat y_{spec}$ from summed\nFiedler edge sensitivities", purple)
box(ax, (0.405, 0.17), 0.17, 0.28, "DAMAGED GRAPH", "$G-S$ adjacency\nnode and disruption features", orange)
box(ax, (0.625, 0.32), 0.16, 0.38, "GRAPH LEARNER", "3-layer GCN + pooling\nlearned correction $r_\\theta$\n\nAblations: Fiedler / coords", green)
box(ax, (0.825, 0.51), 0.15, 0.32, "PREDICTION", "$\\hat y=\\mathrm{clip}(\\hat y_{spec}$\n$+\\tanh r_\\theta,0,1)$", navy)
box(ax, (0.825, 0.10), 0.15, 0.26, "EVALUATION", "Exact $y$ via $\\lambda_2(G-S)$\nMAE, rank, CI, runtime", blue)

arrow(ax, (0.165, 0.56), (0.205, 0.70))
arrow(ax, (0.165, 0.47), (0.205, 0.30))
arrow(ax, (0.365, 0.70), (0.405, 0.72))
arrow(ax, (0.365, 0.28), (0.405, 0.30))
arrow(ax, (0.285, 0.55), (0.47, 0.45), bend=0.14)
arrow(ax, (0.575, 0.72), (0.625, 0.57), label="prior")
arrow(ax, (0.575, 0.31), (0.625, 0.45), label="features")
arrow(ax, (0.785, 0.54), (0.825, 0.67))
arrow(ax, (0.90, 0.51), (0.90, 0.36), label="compare")

ax.text(0.5, 0.965, "ResiliRoad: analytical sensitivity as a prior, graph learning as a correction",
        ha="center", va="top", fontsize=14.5, fontweight="bold", color="#111827")
ax.text(0.5, 0.025, "Training uses graph-disjoint synthetic splits; evaluation includes five zero-shot OSM networks and paired seed-level uncertainty.",
        ha="center", va="bottom", fontsize=9, color="#475569")

fig.tight_layout(pad=0.25)
for suffix in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"method_overview.{suffix}", dpi=300, bbox_inches="tight",
                facecolor="white")
plt.close(fig)
