"""Regenerate the manuscript design figure from the published group split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("configs/fixed_split_groups.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = json.loads(args.split.read_text(encoding="utf-8"))
    membership = {group: split for split in ("train", "val", "test") for group in groups[split]}
    ordered = sorted(membership, key=lambda name: (int(name.split("F")[0][1:]), int(name.split("F")[1])))
    colors = {"train": "#a8bfe5", "val": "#ffe463", "test": "#d772b9"}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(11.7, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.95, 1.05), height_ratios=(1.0, 0.95), wspace=0.13, hspace=0.13)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(grid[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))

    # a: acquisition geometry (schematic, not a scale drawing)
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    ax_a.set_xlim(-1.45, 1.8)
    ax_a.set_ylim(-1.27, 1.34)
    ax_a.add_patch(Circle((0, 0), 0.9, facecolor="#e7eef9", edgecolor="#385d8a", linewidth=1.8))
    for theta in np.deg2rad(np.arange(0, 360, 5)):
        ax_a.plot([0.88*np.cos(theta), 0.93*np.cos(theta)], [0.88*np.sin(theta), 0.93*np.sin(theta)], color="#647c9c", lw=0.45)
    ax_a.text(0, 0, "phantom\nvolume", ha="center", va="center", fontsize=12)
    ax_a.plot(1.22, 0, "o", ms=13, markerfacecolor="#f2a070", markeredgecolor="#87482b", markeredgewidth=1.5)
    ax_a.add_patch(FancyArrowPatch((1.18, 0), (0.30, 0), arrowstyle="-|>", mutation_scale=20, lw=2.3, color="#87482b"))
    ax_a.text(1.36, 0.02, "S11", color="#87482b", va="center", fontweight="bold", fontsize=11)
    ax_a.plot(0.57, 1.05, "o", ms=13, markerfacecolor="#9fcf72", markeredgecolor="#3d6d1d", markeredgewidth=1.5)
    ax_a.add_patch(FancyArrowPatch((0.52, 1.00), (0.12, 0.21), arrowstyle="-|>", mutation_scale=20, lw=2.3, color="#3d6d1d"))
    ax_a.text(0.64, 1.14, "S21", color="#3d6d1d", fontweight="bold", fontsize=11)
    ax_a.text(0, -1.13, "1–8 GHz · 1,001 frequencies · 72 angular positions", ha="center", color="#687789", fontsize=9)
    ax_a.text(-0.06, 1.04, "a", transform=ax_a.transAxes, fontsize=16, fontweight="bold")

    # b: exact phantom-group partition used in the fixed experiment
    y = np.arange(len(ordered))
    ax_b.barh(y, np.full(len(y), 10), color=[colors[membership[group]] for group in ordered], edgecolor="#4c5260", linewidth=0.7)
    ax_b.set_yticks(y, ordered)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 10.5)
    ax_b.set_xticks(np.arange(0, 11, 2))
    ax_b.grid(axis="x", color="#d7dce3", lw=0.8)
    ax_b.set_axisbelow(True)
    ax_b.set_xlabel("Samples per phantom")
    ax_b.set_title("Phantom-group split prevents leakage", loc="left", fontweight="bold")
    ax_b.legend(handles=[Patch(facecolor=colors[key], edgecolor="#4c5260", label=key.title()) for key in ("train", "val", "test")],
                frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax_b.text(-0.06, 1.04, "b", transform=ax_b.transAxes, fontsize=16, fontweight="bold")

    # c: feature and model path. The final label names the actual L2 model.
    ax_c.axis("off")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    labels = [
        ("Complex S-parameters", "#e7eef9", "#385d8a"),
        ("Inverse Fourier transform", "#fff3c8", "#97812c"),
        ("Early-time range gate", "#fae8d9", "#9b5a35"),
        ("Angular-harmonic magnitudes", "#dff2cf", "#5c8c32"),
        ("L2-regularized linear readout", "#f7dce1", "#9e5277"),
    ]
    for idx, (label, fill, edge) in enumerate(labels):
        yy = 0.87 - idx * 0.18
        ax_c.add_patch(FancyBboxPatch((0.08, yy-0.065), 0.78, 0.105, boxstyle="round,pad=0.01,rounding_size=0.008",
                                     facecolor=fill, edgecolor=edge, linewidth=1.5))
        ax_c.text(0.47, yy-0.012, label, ha="center", va="center", fontsize=10)
        if idx < len(labels)-1:
            ax_c.add_patch(FancyArrowPatch((0.47, yy-0.073), (0.47, yy-0.151), arrowstyle="-|>",
                                           mutation_scale=13, linewidth=1.2, color="#7b8797"))
    ax_c.text(-0.02, 1.00, "c", transform=ax_c.transAxes, fontsize=16, fontweight="bold")

    # d: lesion label balance in the published split
    splits = ("Train", "Validation", "Test")
    no_lesion = np.array([70, 10, 20])
    lesion = np.array([70, 10, 20])
    x = np.arange(3)
    ax_d.bar(x, no_lesion, width=0.62, color="#c9d9f4", edgecolor="#385d8a", label="No lesion")
    ax_d.bar(x, lesion, bottom=no_lesion, width=0.62, color="#f2a070", edgecolor="#9b5a35", label="Lesion")
    for pos, total, count in zip(x, no_lesion+lesion, (14, 2, 4)):
        ax_d.text(pos, total+4, f"{count} groups", ha="center", color="#647083", fontsize=10)
    ax_d.set_xticks(x, splits)
    ax_d.set_ylim(0, 160)
    ax_d.set_ylabel("Samples")
    ax_d.set_title("Balanced labels within each split", loc="left", fontweight="bold")
    ax_d.legend(frameon=False, ncol=2, loc="upper right")
    ax_d.text(-0.06, 1.04, "d", transform=ax_d.transAxes, fontsize=16, fontweight="bold")

    for ax in (ax_b, ax_d):
        ax.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
