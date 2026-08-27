#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a clean two-panel framework figure for the manuscript.

Corrections:
- panels (a) and (b) are stacked vertically;
- no text inside boxes is bold;
- all text fits inside boxes;
- no thin vertical border lines at the left/right edges;
- panel (b) explicitly states that the information-footprint diagnostic uses
  RAW DSMC bulk fields, not surrogate-reconstructed fields;
- the surrogate-reconstruction track and the DSMC observability track are shown
  as complementary but logically distinct analyses.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch


def box(ax, xy, wh, text, fc, fontsize=11.2, lw=1.4):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.015",
                       edgecolor="#3a3a3a", facecolor=fc, linewidth=lw)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="normal", linespacing=1.12)
    return p


def arrow(ax, p0, p1, lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color="#444444",
                                 shrinkA=1.5, shrinkB=1.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig02_framework_clean_v2")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.5, 13.0), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.02], hspace=0.18,
                          left=0.055, right=0.965, top=0.965, bottom=0.055)

    # ------------------ panel (a) ------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.97, "(a) Geometry-consistent surrogate reconstruction",
            ha="left", va="top", fontsize=15.5, fontweight="semibold")

    box(ax, (0.03, 0.54), (0.25, 0.18),
        "DSMC database\n57 physical conditions", "#dceaf4", fontsize=11.5)
    box(ax, (0.03, 0.22), (0.25, 0.18),
        "Corrected geometry,\ncoordinates, and masks", "#e9eef2", fontsize=10.8)

    box(ax, (0.36, 0.66), (0.30, 0.14),
        "Smooth-field surrogate\n$u,\\,v,\\,T$", "#e2f0dd", fontsize=11.2)
    box(ax, (0.36, 0.45), (0.30, 0.14),
        "Pressure-focused surrogate\n$P$", "#fff1c9", fontsize=11.2)
    box(ax, (0.36, 0.24), (0.30, 0.14),
        "Face-aware wall surrogate\n$C_p,\\,C_q,\\,|\\tau|$", "#f5dfe6", fontsize=11.0)
    box(ax, (0.74, 0.43), (0.22, 0.22),
        "Reconstructed fields\nand wall-load profiles", "#e8e0f1", fontsize=11.0)

    arrow(ax, (0.155, 0.54), (0.155, 0.40))
    ax.plot([0.28, 0.335], [0.31, 0.31], color="#444444", lw=1.4)
    ax.plot([0.335, 0.335], [0.31, 0.73], color="#444444", lw=1.4)
    for yy in [0.73, 0.52, 0.31]:
        arrow(ax, (0.335, yy), (0.36, yy))
    arrow(ax, (0.66, 0.73), (0.74, 0.58))
    arrow(ax, (0.66, 0.52), (0.74, 0.54))
    arrow(ax, (0.66, 0.31), (0.74, 0.48))

    ax.text(0.50, 0.10,
            "Supported interpolation and independent physical-condition holdouts",
            ha="center", va="center", fontsize=10.8)

    # ------------------ panel (b) ------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.97, "(b) Nonlocal bulk-to-wall observability from raw DSMC fields",
            ha="left", va="top", fontsize=15.5, fontweight="semibold")

    # nested neighborhoods and protrusion sketch
    cx, cy = 0.20, 0.49
    radii = [0.055, 0.092, 0.135]
    colors = ["#cfe3ef", "#dfead7", "#f6ead7"]
    for rr, cc in reversed(list(zip(radii, colors))):
        ax.add_patch(Circle((cx, cy), rr, facecolor=cc, edgecolor="#98a3a8",
                            linewidth=0.8, alpha=0.78))
    tri = np.array([[0.10, 0.27], [0.30, 0.27], [0.20, 0.67]])
    ax.add_patch(Polygon(tri, closed=False, fill=False, edgecolor="#777777", linewidth=2.0))
    ax.plot([0.075, 0.325], [0.27, 0.27], color="#444444", lw=1.6)
    ax.add_patch(Circle((cx, cy), 0.018, facecolor="#c73e78", edgecolor="#333333", linewidth=1.0))
    ax.text(cx, 0.14, "Wall point $s$ with nested\nraw-DSMC neighborhoods $\\Omega_R(s)$",
            ha="center", va="center", fontsize=10.5)

    box(ax, (0.40, 0.55), (0.27, 0.17),
        "Cumulative raw-DSMC patch\n$\\{u,v,T,\\log P\\}_{\\Omega_R(s)}$",
        "#dceaf4", fontsize=10.8)
    box(ax, (0.72, 0.55), (0.24, 0.17),
        "Fixed region-to-point\ndiagnostic regressor", "#dfeff0", fontsize=10.8)
    box(ax, (0.72, 0.28), (0.24, 0.17),
        "Information horizon\n$R_{95}$", "#ffe9ae", fontsize=11.2)

    arrow(ax, (0.30, 0.57), (0.40, 0.63))
    arrow(ax, (0.67, 0.635), (0.72, 0.635))
    arrow(ax, (0.84, 0.55), (0.84, 0.45))

    ax.text(0.66, 0.14,
            "Leave-one-physical-case-out validation across\n"
            "$Ma\\times Kn\\times$ orientation; DSMC wall loads are the targets",
            ha="center", va="center", fontsize=10.5)

    fig.savefig(str(out) + ".png", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", facecolor="white")
    print(str(out) + ".png")
    print(str(out) + ".pdf")


if __name__ == "__main__":
    main()
