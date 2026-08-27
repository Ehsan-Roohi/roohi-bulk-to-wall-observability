#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_make_fig09_surface_profiles_publication.py

Publication-quality replacement for manuscript Fig. 9:
Representative leave-one-physical-case-out protrusion-wall profiles.

Fixes:
- No raw case-title text inside panels.
- Row labels are clean: Backward-facing, Forward-facing, Symmetric.
- Column labels are clean: Cp, Cq, |tau|.
- Legend is outside the plotting panels, large enough to read, and does not cover curves.
- Axis labels and tick labels are journal-readable.
- Supports both dense-radius output folders and previous nonlocal output folders.

Expected input folder contains:
    loocv_predictions_surface.csv

Example:
python3 21_make_fig09_surface_profiles_publication.py \
  --in ./protrusion_16_nonlocal_bulk_to_wall_v02_denseR \
  --out ./fig09_surface_profiles_publication \
  --base-font 18
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGETS = ["Cp", "Cq", "tau_abs"]
TARGET_LABEL = {"Cp": r"$C_p$", "Cq": r"$C_q$", "tau_abs": r"$|\tau|$"}
GEOM_LABEL = {"BWD": "Backward-facing", "FWD": "Forward-facing", "ISO": "Symmetric"}


def parse_radius_from_config(config: str) -> Optional[float]:
    if not isinstance(config, str) or not config.startswith("R"):
        return None
    try:
        return float(config[1:].replace("p", "."))
    except Exception:
        return None


def clean_config_label(config: str) -> str:
    if config == "parameter_only":
        return "Parameters only"
    if config == "local":
        return "Local"
    if config == "full":
        return "Full domain"
    r = parse_radius_from_config(config)
    if r is not None:
        return rf"$R/h_s={r:g}$"
    return str(config)


def choose_configs(preds: pd.DataFrame) -> List[str]:
    available = set(preds["config"].unique())
    preferred = [
        "parameter_only",
        "local",
        "R0p20",
        "R0p50",
        "R1",
        "R1p50",
        "R3",
        "full",
    ]
    chosen = [c for c in preferred if c in available]
    if len(chosen) < 4:
        # Fallback: choose parameter/local, three representative radii, full.
        configs = list(available)
        radii = sorted([(parse_radius_from_config(c), c) for c in configs if parse_radius_from_config(c) is not None])
        chosen = []
        for c in ["parameter_only", "local"]:
            if c in available:
                chosen.append(c)
        if radii:
            selected = [radii[0][1], radii[len(radii)//2][1], radii[-1][1]]
            for c in selected:
                if c not in chosen:
                    chosen.append(c)
        if "full" in available and "full" not in chosen:
            chosen.append("full")
    return chosen


def case_geom(case_id: str) -> str:
    if "_BWD_" in case_id:
        return "BWD"
    if "_FWD_" in case_id:
        return "FWD"
    return "ISO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Folder containing loocv_predictions_surface.csv")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder")
    ap.add_argument("--base-font", type=int, default=18)
    ap.add_argument("--dpi", type=int, default=450)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = in_dir / "loocv_predictions_surface.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    preds = pd.read_csv(csv_path)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": args.base_font,
        "axes.labelsize": args.base_font + 1,
        "axes.titlesize": args.base_font + 2,
        "xtick.labelsize": args.base_font - 2,
        "ytick.labelsize": args.base_font - 2,
        "legend.fontsize": args.base_font - 5,
        "axes.linewidth": 1.4,
        "lines.linewidth": 2.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    preferred_cases = [
        "PHASE1__Ma6_Kn0.33_BWD_hphs1.5_Tw1",
        "PHASE1__Ma6_Kn0.33_FWD_hphs1.5_Tw1",
        "PHASE1__Ma6_Kn0.33_ISO_hphs1.5_Tw1",
    ]
    cases = [c for c in preferred_cases if c in set(preds["case_id"])]
    if len(cases) < 3:
        # Fallback: choose Ma6-Kn0.33 cases if naming differs.
        tmp = preds[preds["case_id"].astype(str).str.contains("Ma6_Kn0.33", regex=False)]
        cases = list(pd.unique(tmp["case_id"]))[:3]
    if len(cases) < 3:
        cases = list(pd.unique(preds["case_id"]))[:3]

    configs = choose_configs(preds)

    # Larger top margin to keep legend outside panels.
    fig, axes = plt.subplots(3, 3, figsize=(18.6, 13.4), constrained_layout=False)
    plt.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.83, wspace=0.26, hspace=0.36)

    # Consistent colors; DSMC emphasized.
    color_map = {
        "DSMC": "black",
        "parameter_only": "#d62728",
        "local": "#ff7f0e",
        "R0p20": "#2ca02c",
        "R0p50": "#9467bd",
        "R1": "#8c564b",
        "R1p50": "#e377c2",
        "R3": "#7f7f7f",
        "full": "#1f77b4",
    }

    for i, case_id in enumerate(cases):
        g = case_geom(case_id)
        case_df = preds[preds["case_id"] == case_id]
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            true_df = case_df[case_df["config"] == "full"].sort_values("s01")
            if true_df.empty:
                true_df = case_df[case_df["config"] == case_df["config"].iloc[0]].sort_values("s01")

            # DSMC reference
            ax.plot(
                true_df["s01"],
                true_df[f"true_{target}"],
                lw=3.2,
                color=color_map["DSMC"],
                label="DSMC",
                zorder=5,
            )

            # Predictions
            for cfg in configs:
                sub = case_df[case_df["config"] == cfg].sort_values("s01")
                if sub.empty:
                    continue
                ax.plot(
                    sub["s01"],
                    sub[f"pred_{target}"],
                    lw=1.9 if cfg != "full" else 2.4,
                    color=color_map.get(cfg, None),
                    label=clean_config_label(cfg),
                    alpha=0.96,
                )

            if i == 0:
                ax.set_title(TARGET_LABEL[target], pad=8)
            if j == 0:
                ax.text(
                    -0.24, 0.5, GEOM_LABEL[g],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=args.base_font,
                    fontweight="bold",
                )
            ax.set_xlabel("Normalized surface coordinate")
            ax.set_ylabel(TARGET_LABEL[target])
            ax.grid(True, alpha=0.25)
            ax.set_xlim(-0.02, 1.02)

    # Figure-level legend above panels, outside panel area.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.3,
        handletextpad=0.55,
        fontsize=args.base_font - 5,
    )

    fig.suptitle(
        r"Representative surface-load profiles ($Ma=6$, $Kn=0.33$, $h_p/h_s=1.5$, $T_w/T_\infty=1$)",
        y=0.985,
        fontsize=args.base_font + 5,
    )

    out_png = out_dir / "fig09_surface_profiles_publication.png"
    out_pdf = out_dir / "fig09_surface_profiles_publication.pdf"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[DONE] wrote")
    print(out_png)
    print(out_pdf)
    print("[INFO] configs shown:", ", ".join(configs))


if __name__ == "__main__":
    main()
