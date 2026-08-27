#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised final post-processing for the nonlocal bulk-to-wall study.
Main changes relative to v01:
- cleaner journal-style figure labels
- smaller/moved legends so they do not cover curves
- no heatmap figure
- no bar-chart figures
- cleaner surface-profile figure without raw case IDs in each panel
- x-axis labels for error-vs-radius shortened to avoid overlap
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator

TARGETS = ["Cp", "Cq", "tau_abs"]
TARGET_LABEL = {
    "Cp": r"$C_p$",
    "Cq": r"$C_q$",
    "tau_abs": "$|\\tau|$",
}
GEOMS = ["BWD", "FWD", "ISO"]
GEOM_LABEL = {
    "BWD": "Backward-facing",
    "FWD": "Forward-facing",
    "ISO": "Symmetric",
}


def set_publication_style(base_font: int = 18):
    plt.rcParams.update({
        "font.size": base_font,
        "axes.labelsize": base_font + 2,
        "axes.titlesize": base_font + 1,
        "xtick.labelsize": base_font - 2,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 5,
        "figure.titlesize": base_font + 4,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
        "grid.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 350,
        "font.family": "DejaVu Sans",
    })


def savefig(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=350)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_radius_from_config(config: str) -> Optional[float]:
    if not isinstance(config, str) or not config.startswith("R"):
        return None
    s = config[1:].replace("p", ".")
    try:
        return float(s)
    except Exception:
        return None


def config_order(configs: List[str]) -> List[str]:
    out = []
    if "parameter_only" in configs:
        out.append("parameter_only")
    if "local" in configs:
        out.append("local")
    radii = sorted([(parse_radius_from_config(c), c) for c in configs if parse_radius_from_config(c) is not None])
    out.extend([c for _, c in radii])
    if "full" in configs:
        out.append("full")
    return out


def get_max_radius(configs: List[str]) -> float:
    vals = [parse_radius_from_config(c) for c in configs]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(max(vals)) if vals else np.nan


def clean_config_label(config: str, multiline: bool = False) -> str:
    if config == "parameter_only":
        return "param\nonly" if multiline else "parameters only"
    if config == "local":
        return "local"
    if config == "full":
        return "full\ndomain" if multiline else "full domain"
    r = parse_radius_from_config(config)
    if r is not None:
        return f"{r:g}"
    return str(config)


def add_censor_columns(horizon: pd.DataFrame, rmax: float) -> pd.DataFrame:
    h = horizon.copy()
    h["is_censored"] = h["R95_label"].astype(str).eq("not_reached") | ~np.isfinite(h["R95_over_hs"].astype(float))
    h["R95_lower_bound_hs"] = h["R95_over_hs"].astype(float)
    h.loc[h["is_censored"], "R95_lower_bound_hs"] = rmax
    h["R95_lower_bound_lambda"] = h["R95_lower_bound_hs"] / (h["Kn"].astype(float) * h["hphs"].astype(float))
    h["R95_display"] = [f">{rmax:g}" if c else f"{v:g}" for c, v in zip(h["is_censored"], h["R95_over_hs"])]
    return h


def require_file(in_dir: Path, name: str) -> Path:
    p = in_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return p


def rel_reduction(a: float, b: float) -> float:
    if not np.isfinite(a) or abs(a) < 1e-15:
        return np.nan
    return 100.0 * (a - b) / a


def make_error_reduction_table(err: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    configs = list(err["config"].unique())
    for target in TARGETS:
        sub = err[err["target"] == target]
        def get(cfg):
            v = sub[sub["config"] == cfg]["relL2"]
            return float(v.iloc[0]) if len(v) else np.nan
        parameter = get("parameter_only")
        local = get("local")
        full = get("full")
        finite_configs = [c for c in configs if parse_radius_from_config(c) is not None]
        finite_sub = sub[sub["config"].isin(finite_configs)].copy()
        if len(finite_sub):
            best_row = finite_sub.loc[finite_sub["relL2"].idxmin()]
            best_finite_cfg = str(best_row["config"])
            best_finite = float(best_row["relL2"])
        else:
            best_finite_cfg, best_finite = "", np.nan
        rows.append({
            "target": target,
            "target_label": TARGET_LABEL[target],
            "parameter_only_relL2_pct": 100.0 * parameter,
            "local_relL2_pct": 100.0 * local,
            "best_finite_radius_config": best_finite_cfg,
            "best_finite_radius_relL2_pct": 100.0 * best_finite,
            "full_domain_relL2_pct": 100.0 * full,
            "reduction_parameter_to_full_pct": rel_reduction(parameter, full),
            "reduction_parameter_to_best_finite_pct": rel_reduction(parameter, best_finite),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "table01_error_reduction_parameter_vs_bulk.csv", index=False)
    return tab


def make_R95_summary_tables(h: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (target, geom), g in h.groupby(["target", "geom"]):
        n = len(g)
        n_c = int(g["is_censored"].sum())
        n_r = n - n_c
        rows.append({
            "target": target,
            "geom": geom,
            "n_cases": n,
            "n_reached": n_r,
            "n_censored": n_c,
            "reached_fraction": n_r / n if n else np.nan,
            "R95_reached_mean_hs": float(g.loc[~g["is_censored"], "R95_over_hs"].mean()) if n_r else np.nan,
            "R95_lower_bound_mean_hs": float(g["R95_lower_bound_hs"].mean()),
            "R95_lower_bound_median_hs": float(g["R95_lower_bound_hs"].median()),
            "R95_lower_bound_mean_lambda": float(g["R95_lower_bound_lambda"].mean()),
            "full_relL2_mean_pct": 100.0 * float(g["full_relL2"].mean()),
        })
    s1 = pd.DataFrame(rows).sort_values(["target", "geom"])
    s1.to_csv(out_dir / "table02_R95_censored_summary_by_target_geom.csv", index=False)

    rows = []
    for (target, geom, Kn), g in h.groupby(["target", "geom", "Kn"]):
        n = len(g)
        n_c = int(g["is_censored"].sum())
        n_r = n - n_c
        rows.append({
            "target": target,
            "geom": geom,
            "Kn": Kn,
            "n_cases": n,
            "n_reached": n_r,
            "n_censored": n_c,
            "R95_reached_mean_hs": float(g.loc[~g["is_censored"], "R95_over_hs"].mean()) if n_r else np.nan,
            "R95_lower_bound_mean_hs": float(g["R95_lower_bound_hs"].mean()),
            "R95_lower_bound_mean_lambda": float(g["R95_lower_bound_lambda"].mean()),
            "full_relL2_mean_pct": 100.0 * float(g["full_relL2"].mean()),
        })
    s2 = pd.DataFrame(rows).sort_values(["target", "geom", "Kn"])
    s2.to_csv(out_dir / "table03_R95_censored_summary_by_target_geom_Kn.csv", index=False)

    keep = [
        "case_id", "target", "Ma", "Kn", "geom", "hphs",
        "full_relL2", "best_config", "best_relL2",
        "is_censored", "R95_display", "R95_over_hs",
        "R95_lower_bound_hs", "R95_lower_bound_lambda",
    ]
    h[keep].to_csv(out_dir / "table04_casewise_R95_with_censoring.csv", index=False)
    return s1, s2


def write_key_numbers(err_tab: pd.DataFrame, r95_tab: pd.DataFrame, h: pd.DataFrame, out_dir: Path, rmax: float):
    lines = []
    lines.append("Key numbers for manuscript")
    lines.append("==========================")
    lines.append(f"Largest finite radius tested: R/h_s = {rmax:g}")
    lines.append("")
    for _, r in err_tab.iterrows():
        lines.append(
            f"{r['target_label']}: parameter-only {r['parameter_only_relL2_pct']:.2f}% -> full-domain {r['full_domain_relL2_pct']:.2f}% "
            f"(reduction {r['reduction_parameter_to_full_pct']:.1f}%)."
        )
    lines.append("")
    for _, r in r95_tab.iterrows():
        lines.append(
            f"{TARGET_LABEL.get(r['target'], r['target'])}, {r['geom']}: reached {int(r['n_reached'])}/{int(r['n_cases'])}, "
            f"lower-bound mean R95/h_s={r['R95_lower_bound_mean_hs']:.3g}."
        )
    (out_dir / "manuscript_key_numbers_and_wording.txt").write_text("\n".join(lines))


def plot_error_vs_radius(err: pd.DataFrame, out_dir: Path):
    configs = config_order(err["config"].unique().tolist())
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 6.2), constrained_layout=True)
    short_labels = []
    for c in configs:
        if c == "parameter_only":
            short_labels.append("Param.\nonly")
        elif c == "local":
            short_labels.append("Local")
        elif c == "full":
            short_labels.append("Full\ndomain")
        else:
            r = parse_radius_from_config(c)
            short_labels.append(f"$R/h_s={r:g}$" if r is not None else str(c))
    for ax, target in zip(axes, TARGETS):
        sub = err[err["target"] == target].set_index("config")
        y = [100.0 * float(sub.loc[c, "relL2"]) if c in sub.index else np.nan for c in configs]
        x = np.arange(len(configs))
        ax.plot(x, y, marker="o", linewidth=2.8, markersize=8, color="#2f6db0")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=28, ha="right")
        ax.tick_params(axis="x", labelsize=11, pad=4)
        ax.set_xlabel(r"Observed bulk-field radius")
        ax.set_ylabel("LOOCV relative error (%)")
        ax.set_title(TARGET_LABEL[target], pad=8)
        ax.grid(True, axis="y", alpha=0.35)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.suptitle("Bulk-to-wall surface-load error versus observed bulk-field radius", y=1.03)
    savefig(fig, out_dir, "fig01_error_vs_radius_largefonts")


def plot_R95_vs_Kn_censored(h: pd.DataFrame, out_dir: Path, rmax: float):
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.6), constrained_layout=True)
    markers = {"BWD": "o", "FWD": "s", "ISO": "^"}
    xticks = [0.10, 0.33, 0.80]
    xticklabels = ["0.10", "0.33", "0.80"]
    for ax, target in zip(axes, TARGETS):
        sub = h[h["target"] == target]
        for geom in GEOMS:
            gg = sub[sub["geom"] == geom]
            if gg.empty:
                continue
            sm = gg.groupby("Kn")["R95_lower_bound_hs"].mean().reset_index().sort_values("Kn")
            ax.plot(sm["Kn"], sm["R95_lower_bound_hs"], marker=markers[geom], linewidth=2.6, markersize=8, label=GEOM_LABEL[geom])
            cen = gg.groupby("Kn")["is_censored"].sum().reset_index()
            cen = cen[cen["is_censored"] > 0]
            for _, row in cen.iterrows():
                yy = float(sm.loc[sm["Kn"] == row["Kn"], "R95_lower_bound_hs"].iloc[0])
                ax.scatter([row["Kn"]], [yy], s=90, facecolors="white", edgecolors="black", linewidths=1.4, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(0.09, 0.9)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelsize=13)
        ax.set_xlabel("Knudsen number")
        ax.set_ylabel(r"Lower-bound $R_{95}/h_s$")
        ax.set_title(TARGET_LABEL[target], pad=8)
        ax.grid(True, which="major", alpha=0.35)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=3, frameon=False, fontsize=12, title="Orientation")
    fig.suptitle(r"Knudsen-dependent information horizon (open symbols indicate right-censoring)", y=1.16)
    savefig(fig, out_dir, "fig03_R95_vs_Kn_censored_lowerbound")


def select_profile_configs(preds: pd.DataFrame) -> List[str]:
    available = set(preds["config"].unique())
    choices = ["parameter_only", "local", "R0p20", "R0p50", "R1", "R1p50", "R3", "full"]
    return [c for c in choices if c in available]


def plot_surface_profiles(preds: pd.DataFrame, out_dir: Path):
    if preds.empty:
        return
    preferred = [
        "PHASE1__Ma6_Kn0.33_BWD_hphs1.5_Tw1",
        "PHASE1__Ma6_Kn0.33_FWD_hphs1.5_Tw1",
        "PHASE1__Ma6_Kn0.33_ISO_hphs1.5_Tw1",
    ]
    cases = [c for c in preferred if c in set(preds["case_id"])]
    if len(cases) < 3:
        cases = list(pd.unique(preds["case_id"]))[:3]
    configs = select_profile_configs(preds)
    fig, axes = plt.subplots(len(cases), 3, figsize=(18.0, 12.8), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, case_id in enumerate(cases):
        case_df = preds[preds["case_id"] == case_id]
        geom = "BWD" if "_BWD_" in case_id else ("FWD" if "_FWD_" in case_id else "ISO")
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            true_df = case_df[case_df["config"] == "full"].sort_values("s01")
            if true_df.empty:
                true_df = case_df[case_df["config"] == case_df["config"].iloc[0]].sort_values("s01")
            ax.plot(true_df["s01"], true_df[f"true_{target}"], lw=2.8, label="DSMC", color="#1f77b4")
            for cfg in configs:
                sub = case_df[case_df["config"] == cfg].sort_values("s01")
                if sub.empty:
                    continue
                if cfg == "parameter_only":
                    lab = "Parameters only"
                elif cfg == "full":
                    lab = "Full domain"
                elif cfg == "local":
                    lab = "Local"
                else:
                    lab = rf"$R/h_s={parse_radius_from_config(cfg):g}$"
                ax.plot(sub["s01"], sub[f"pred_{target}"], lw=1.8, label=lab)
            ax.set_xlabel("Normalized surface coordinate")
            ax.set_ylabel(TARGET_LABEL[target])
            ax.grid(True, alpha=0.30)
            if i == 0:
                ax.set_title(TARGET_LABEL[target], pad=8)
            if j == 0:
                row_label = {
                    "BWD": "Backward-facing",
                    "FWD": "Forward-facing",
                    "ISO": "Symmetric",
                }[geom]
                ax.text(-0.28, 0.5, row_label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=15, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    hh, ll = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            hh.append(h)
            ll.append(l)
            seen.add(l)
    fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=4, frameon=False, fontsize=10, columnspacing=1.1, handlelength=1.8)
    fig.suptitle(r"Representative surface-load profiles ($Ma=6$, $Kn=0.33$, $h_p/h_s=1.5$, $T_w/T_\infty=1$)", y=1.12)
    savefig(fig, out_dir, "fig07_surface_profiles_clean")


def plot_Cq_focus(h: pd.DataFrame, out_dir: Path):
    sub = h[h["target"] == "Cq"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9.4, 6.2), constrained_layout=True)
    markers = {"BWD": "o", "FWD": "s", "ISO": "^"}
    for geom in GEOMS:
        gg = sub[sub["geom"] == geom]
        sm = gg.groupby("Kn")["R95_lower_bound_hs"].mean().reset_index().sort_values("Kn")
        ax.plot(sm["Kn"], sm["R95_lower_bound_hs"], marker=markers[geom], linewidth=2.6, markersize=8, label=GEOM_LABEL[geom])
        cen = gg.groupby("Kn")["is_censored"].sum().reset_index()
        cen = cen[cen["is_censored"] > 0]
        for _, row in cen.iterrows():
            yy = float(sm.loc[sm["Kn"] == row["Kn"], "R95_lower_bound_hs"].iloc[0])
            ax.scatter([row["Kn"]], [yy], s=90, facecolors="white", edgecolors="black", linewidths=1.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Knudsen number")
    ax.set_ylabel(r"Lower-bound $R_{95}/h_s$ for $C_q$")
    ax.set_title(r"Heat-transfer information horizon", pad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=12, borderaxespad=0.0)
    ax.grid(True, which="both", alpha=0.35)
    ax.set_ylim(bottom=0)
    savefig(fig, out_dir, "fig08_Cq_focus_Kn_dependent_nonlocality")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--base-font", type=int, default=18)
    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_publication_style(args.base_font)

    err = pd.read_csv(require_file(in_dir, "error_summary_by_config_target.csv"))
    horizon = pd.read_csv(require_file(in_dir, "information_horizon_R95.csv"))
    preds_path = in_dir / "loocv_predictions_surface.csv"
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()

    configs = err["config"].unique().tolist()
    rmax = get_max_radius(configs)
    if not np.isfinite(rmax):
        raise RuntimeError("Could not infer finite radii from config labels.")

    h = add_censor_columns(horizon, rmax)
    h.to_csv(out_dir / "information_horizon_R95_censored_casewise.csv", index=False)

    err_tab = make_error_reduction_table(err, out_dir)
    r95_target_geom, _ = make_R95_summary_tables(h, out_dir)
    write_key_numbers(err_tab, r95_target_geom, h, out_dir, rmax)

    target_rows = []
    for target, g in h.groupby("target"):
        target_rows.append({
            "target": target,
            "n_cases": len(g),
            "n_reached": int((~g["is_censored"]).sum()),
            "n_censored": int(g["is_censored"].sum()),
            "R95_lower_bound_mean_hs": float(g["R95_lower_bound_hs"].mean()),
            "R95_lower_bound_median_hs": float(g["R95_lower_bound_hs"].median()),
            "R95_lower_bound_mean_lambda": float(g["R95_lower_bound_lambda"].mean()),
            "full_relL2_mean_pct": 100.0 * float(g["full_relL2"].mean()),
        })
    pd.DataFrame(target_rows).to_csv(out_dir / "table00_target_level_R95_summary.csv", index=False)

    plot_error_vs_radius(err, out_dir)
    plot_R95_vs_Kn_censored(h, out_dir, rmax)
    if not preds.empty:
        plot_surface_profiles(preds, out_dir)
    plot_Cq_focus(h, out_dir)

    print("[DONE] Revised final post-processing outputs written to:")
    print(out_dir)
    print("\nKey figure files:")
    for name in [
        "fig01_error_vs_radius_largefonts.pdf",
        "fig03_R95_vs_Kn_censored_lowerbound.pdf",
        "fig07_surface_profiles_clean.pdf",
        "fig08_Cq_focus_Kn_dependent_nonlocality.pdf",
    ]:
        p = out_dir / name
        if p.exists():
            print("  -", p)

if __name__ == "__main__":
    main()
