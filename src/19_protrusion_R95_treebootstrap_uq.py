#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_protrusion_R95_treebootstrap_uq.py

Lightweight uncertainty analysis for the model-conditional information horizon R95.

The script repeats the Phase-1 leave-one-physical-case-out diagnostic using a native
multi-output ExtraTrees forest. For each fitted forest, the individual decision trees
are retained. Bootstrap resampling of trees generates an ensemble distribution for the
surface-profile prediction error at every tested radius. These error distributions are
then propagated through the same R95 definition used in the manuscript.

What this uncertainty represents
--------------------------------
- finite diagnostic-regressor ensemble uncertainty;
- sensitivity of R95 and right-censoring to the fitted tree ensemble.

What it does NOT represent
--------------------------
- independent DSMC-realization/sampling uncertainty;
- uncertainty in the underlying DSMC wall profiles;
- causal-information uncertainty.

Independent DSMC realizations would still be required for full DSMC noise calibration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from sklearn.ensemble import ExtraTreesRegressor

TARGETS_ALL = ["Cp", "Cq", "tau_abs"]
TARGET_LABEL = {"Cp": r"$C_p$", "Cq": r"$C_q$", "tau_abs": r"$|\tau|$"}
GEOMS = ["BWD", "FWD", "ISO"]
GEOM_LABEL = {"BWD": "Backward-facing", "FWD": "Forward-facing", "ISO": "Symmetric"}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def import_module(path: str, name: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_radii(text: str) -> List[float]:
    return sorted({float(x.strip()) for x in str(text).split(",") if x.strip()})


def prepare_train_test(train_df, test_df, targets):
    ignore = set(["case_id", "surface_i", "geom", "config"] + TARGETS_ALL)
    cols = [c for c in train_df.columns if c not in ignore and np.issubdtype(train_df[c].dtype, np.number)]
    for c in cols:
        if c not in test_df.columns:
            test_df[c] = np.nan
    Xtr = train_df[cols].to_numpy(dtype=float)
    Xte = test_df[cols].to_numpy(dtype=float)
    Xtr[~np.isfinite(Xtr)] = np.nan
    Xte[~np.isfinite(Xte)] = np.nan
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for X in (Xtr, Xte):
        ind = np.where(~np.isfinite(X))
        X[ind] = np.take(med, ind[1])
    ytr = train_df[targets].to_numpy(dtype=float)
    yte = test_df[targets].to_numpy(dtype=float)
    return Xtr, Xte, ytr, yte


def tree_bootstrap_predictions(model: ExtraTreesRegressor, Xtest: np.ndarray,
                               n_boot: int, rng: np.random.Generator):
    # Native multi-output ExtraTrees trees return [n_points, n_targets].
    preds = []
    for tree in model.estimators_:
        pp = np.asarray(tree.predict(Xtest))
        if pp.ndim == 1:
            pp = pp[:, None]
        preds.append(pp)
    tree_pred = np.stack(preds, axis=0)
    ntree = tree_pred.shape[0]
    out = np.empty((n_boot, tree_pred.shape[1], tree_pred.shape[2]), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, ntree, size=ntree)
        out[b] = tree_pred[idx].mean(axis=0)
    return out


def rel_l2(pred, true, eps=1e-12):
    m = np.isfinite(pred) & np.isfinite(true)
    if m.sum() < 2:
        return np.nan
    return float(np.linalg.norm(pred[m] - true[m]) / (np.linalg.norm(true[m]) + eps))


def quant(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else np.nan


def compute_bootstrap_R95(err_df: pd.DataFrame, radii: Sequence[float], nl, rmax: float):
    rows = []
    labels = [nl.fmt_r_label(R) for R in radii]
    for (boot, case_id, target), g in err_df.groupby(["bootstrap", "case_id", "target"]):
        gd = {str(r["config"]): r for _, r in g.iterrows()}
        if "full" not in gd:
            continue
        full_err = float(gd["full"]["relL2"])
        threshold = 1.05 * full_err
        R95 = np.nan
        cfg_hit = "not_reached"
        for R, cfg in zip(radii, labels):
            if cfg in gd and np.isfinite(float(gd[cfg]["relL2"])) and float(gd[cfg]["relL2"]) <= threshold:
                R95 = float(R); cfg_hit = cfg; break
        m = g.iloc[0]
        cens = not np.isfinite(R95)
        rows.append({
            "bootstrap": int(boot), "case_id": case_id, "target": target,
            "Ma": float(m["Ma"]), "Kn": float(m["Kn"]), "geom": str(m["geom"]),
            "hphs": float(m["hphs"]), "full_relL2": full_err,
            "R95_over_hs": R95, "R95_label": cfg_hit, "is_censored": cens,
            "R95_lower_bound_hs": rmax if cens else R95,
        })
    return pd.DataFrame(rows)


def summarize_r95(r95: pd.DataFrame, out_dir: Path):
    case_rows = []
    for (case_id, target), g in r95.groupby(["case_id", "target"]):
        x = g["R95_lower_bound_hs"].to_numpy(dtype=float)
        case_rows.append({
            "case_id": case_id, "target": target,
            "Ma": g.iloc[0]["Ma"], "Kn": g.iloc[0]["Kn"], "geom": g.iloc[0]["geom"],
            "R95_lower_bound_median_hs": quant(x, .5),
            "R95_lower_bound_CI2p5_hs": quant(x, .025),
            "R95_lower_bound_CI97p5_hs": quant(x, .975),
            "censor_probability": float(g["is_censored"].mean()),
            "full_relL2_median_pct": 100.0 * quant(g["full_relL2"], .5),
            "full_relL2_CI2p5_pct": 100.0 * quant(g["full_relL2"], .025),
            "full_relL2_CI97p5_pct": 100.0 * quant(g["full_relL2"], .975),
        })
    case_tab = pd.DataFrame(case_rows)
    case_tab.to_csv(out_dir / "R95_treebootstrap_casewise_summary.csv", index=False)

    group_rows = []
    for (target, geom, Kn), g in r95.groupby(["target", "geom", "Kn"]):
        # For each bootstrap first average over Mach, then summarize across bootstrap.
        bmeans = g.groupby("bootstrap")["R95_lower_bound_hs"].mean().to_numpy(dtype=float)
        bcens = g.groupby("bootstrap")["is_censored"].mean().to_numpy(dtype=float)
        group_rows.append({
            "target": target, "geom": geom, "Kn": Kn,
            "R95_lower_bound_mean_median_hs": quant(bmeans, .5),
            "R95_lower_bound_mean_CI2p5_hs": quant(bmeans, .025),
            "R95_lower_bound_mean_CI97p5_hs": quant(bmeans, .975),
            "mean_censored_fraction": float(np.mean(bcens)),
        })
    group_tab = pd.DataFrame(group_rows)
    group_tab.to_csv(out_dir / "R95_treebootstrap_summary_by_target_geom_Kn.csv", index=False)
    return case_tab, group_tab


def plot_uq(group_tab: pd.DataFrame, out_dir: Path):
    targets = [t for t in TARGETS_ALL if t in set(group_tab["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(6.0*len(targets), 5.3), squeeze=False, constrained_layout=True)
    axes = axes.ravel()
    markers = {"BWD": "o", "FWD": "s", "ISO": "^"}
    xticks = [0.10, 0.33, 0.80]
    for ax, target in zip(axes, targets):
        sub = group_tab[group_tab["target"] == target]
        for geom in GEOMS:
            g = sub[sub["geom"] == geom].sort_values("Kn")
            if g.empty: continue
            x = g["Kn"].to_numpy(dtype=float)
            y = g["R95_lower_bound_mean_median_hs"].to_numpy(dtype=float)
            lo = g["R95_lower_bound_mean_CI2p5_hs"].to_numpy(dtype=float)
            hi = g["R95_lower_bound_mean_CI97p5_hs"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=[y-lo, hi-y], marker=markers[geom], capsize=4, lw=2.2, ms=7, label=GEOM_LABEL[geom])
        ax.set_xscale("log")
        ax.set_xlim(.09, .9)
        ax.set_xticks(xticks); ax.set_xticklabels(["0.10", "0.33", "0.80"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Knudsen number")
        ax.set_ylabel(r"Lower-bound $R_{95}/h_s$")
        ax.set_title(TARGET_LABEL[target])
        ax.grid(True, alpha=.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.suptitle("Diagnostic-regressor tree-bootstrap uncertainty", y=1.10)
    fig.savefig(out_dir / "fig_R95_treebootstrap_UQ.png", dpi=350, bbox_inches="tight")
    fig.savefig(out_dir / "fig_R95_treebootstrap_UQ.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nonlocal-script", required=True)
    ap.add_argument("--field-base-script", required=True)
    ap.add_argument("--surface-base-script", required=True)
    ap.add_argument("--targets", default="Cp,Cq,tau_abs")
    ap.add_argument("--radii", default="0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3")
    ap.add_argument("--max-gas-points", type=int, default=60000)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--bootstraps", type=int, default=300)
    ap.add_argument("--min-leaf", type=int, default=2)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    targets = parse_list(args.targets)
    bad = set(targets) - set(TARGETS_ALL)
    if bad: raise ValueError(f"Unknown targets: {bad}")
    radii = parse_radii(args.radii)
    rmax = max(radii)
    out_dir = ensure_dir(Path(args.out).resolve())
    with open(out_dir / "run_config.json", "w") as f: json.dump(vars(args), f, indent=2)

    nl = import_module(args.nonlocal_script, "nonlocal_base_uq")
    field_base = import_module(args.field_base_script, "field_base_uq")
    surface_base = import_module(args.surface_base_script, "surface_base_uq")

    m = nl.get_phase1_cases(field_base.load_manifest(Path(args.audit_dir)))
    configs: List[Tuple[str, Optional[float]]] = [("parameter_only", None), ("local", None)]
    configs += [("R", R) for R in radii]
    configs += [("full", None)]

    rng = np.random.default_rng(args.seed)
    parts: Dict[str, List[pd.DataFrame]] = {}
    case_rows = []
    for _, row in m.iterrows():
        c = nl.read_case_data(field_base, surface_base, row, args, rng)
        if c is None: continue
        fd = nl.make_features_for_case(c, configs)
        for cfg, df in fd.items(): parts.setdefault(cfg, []).append(df)
        case_rows.append({"case_id": c.case_id, "Ma": c.Ma, "Kn": c.Kn, "geom": c.geom, "hphs": c.hphs})
        print(f"[INFO] features {len(case_rows):02d}/{len(m)} {c.case_id}", flush=True)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    case_table = pd.DataFrame(case_rows)

    boot_error_rows = []
    for cidx, (cfg, df) in enumerate(tables.items()):
        print(f"[INFO] config {cidx+1}/{len(tables)} {cfg}", flush=True)
        for fold, case_id in enumerate(case_table["case_id"]):
            tr = df[df["case_id"] != case_id].copy()
            te = df[df["case_id"] == case_id].copy()
            Xtr, Xte, ytr, yte = prepare_train_test(tr, te, targets)
            model = ExtraTreesRegressor(
                n_estimators=args.trees, min_samples_leaf=args.min_leaf,
                max_features=args.max_features, random_state=args.seed + 1000*cidx + fold,
                n_jobs=args.jobs,
            )
            model.fit(Xtr, ytr)
            brng = np.random.default_rng(args.seed + 100000*cidx + 1000*fold)
            pred_boot = tree_bootstrap_predictions(model, Xte, args.bootstraps, brng)
            meta = case_table[case_table["case_id"] == case_id].iloc[0]
            for b in range(args.bootstraps):
                for j, target in enumerate(targets):
                    boot_error_rows.append({
                        "bootstrap": b, "case_id": case_id, "config": cfg, "target": target,
                        "Ma": meta.Ma, "Kn": meta.Kn, "geom": meta.geom, "hphs": meta.hphs,
                        "relL2": rel_l2(pred_boot[b, :, j], yte[:, j]),
                    })
    err = pd.DataFrame(boot_error_rows)
    err.to_csv(out_dir / "R95_treebootstrap_errors.csv", index=False)
    r95 = compute_bootstrap_R95(err, radii, nl, rmax)
    r95.to_csv(out_dir / "R95_treebootstrap_samples.csv", index=False)
    case_tab, group_tab = summarize_r95(r95, out_dir)
    if args.make_plots:
        plot_uq(group_tab, out_dir)

    with open(out_dir / "final_summary.txt", "w") as f:
        f.write("Tree-bootstrap UQ for model-conditional R95\n")
        f.write("==========================================\n\n")
        f.write("This analysis quantifies diagnostic-regressor ensemble uncertainty only.\n")
        f.write("It does not replace independent DSMC-realization uncertainty.\n\n")
        f.write(group_tab.to_string(index=False))
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
