#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_protrusion_R95_robustness_controls.py

Runs the three controls requested for the PoF paper:

1. Second-regressor robustness:
   compare the original ExtraTrees diagnostic with a structurally different
   Ridge diagnostic, and optionally RandomForest or MLP.

2. Threshold sensitivity:
   recompute R_epsilon for epsilon = 0.025, 0.05, 0.10.

3. Full-domain reference control:
   compare the primary full-domain ring-statistics reference against the
   largest cumulative patch, usually R/h_s = 3.

The script imports the existing 16_protrusion_nonlocal_bulk_to_wall_footprint.py
and reuses its case reading and feature extraction functions.

Main outputs:
  error_metrics_by_model_config_case.csv
  error_summary_by_model_config_target.csv
  R_epsilon_by_model_threshold_reference.csv
  R_epsilon_summary_by_model_threshold_reference.csv
  censoring_summary_by_model_threshold_reference.csv
  robustness_key_findings.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["Cp", "Cq", "tau_abs"]
TARGET_LABEL = {"Cp": r"$C_p$", "Cq": r"$C_q$", "tau_abs": r"$|\tau|$"}


def import_module_from_path(path: str, name: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {p}")
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_list_float(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_list_str(s: str) -> List[str]:
    return [x.strip().lower() for x in str(s).split(",") if x.strip()]


def parse_r_from_label(label: str) -> Optional[float]:
    if not isinstance(label, str) or not label.startswith("R"):
        return None
    try:
        return float(label[1:].replace("p", "."))
    except Exception:
        return None


def config_order(configs: List[str]) -> List[str]:
    out = []
    for c in ["parameter_only", "local"]:
        if c in configs:
            out.append(c)
    radii = sorted([(parse_r_from_label(c), c) for c in configs if parse_r_from_label(c) is not None])
    out.extend([c for _, c in radii])
    if "full" in configs:
        out.append("full")
    return out


def clean_config_label(c: str) -> str:
    if c == "parameter_only":
        return "Param."
    if c == "local":
        return "Local"
    if c == "full":
        return "Full"
    r = parse_r_from_label(c)
    return f"{r:g}" if r is not None else str(c)


def rel_l2(pred, true, eps=1e-12) -> float:
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    m = np.isfinite(pred) & np.isfinite(true)
    if m.sum() < 2:
        return np.nan
    return float(np.linalg.norm(pred[m] - true[m]) / (np.linalg.norm(true[m]) + eps))


def build_model(model_name: str, args, seed: int):
    if model_name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=int(args.trees),
            min_samples_leaf=int(args.min_leaf),
            max_features=args.max_features,
            random_state=seed,
            n_jobs=int(args.jobs),
            bootstrap=False,
        )
    if model_name == "randomforest":
        return RandomForestRegressor(
            n_estimators=int(args.trees),
            min_samples_leaf=int(args.min_leaf),
            max_features=args.max_features,
            random_state=seed,
            n_jobs=int(args.jobs),
            bootstrap=True,
        )
    if model_name == "ridge":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(args.ridge_alpha), random_state=seed),
        )
    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(int(x) for x in str(args.mlp_widths).split(",")),
                activation="relu",
                solver="adam",
                alpha=float(args.mlp_alpha),
                learning_rate_init=float(args.mlp_lr),
                batch_size=int(args.mlp_batch_size),
                max_iter=int(args.mlp_max_iter),
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
                verbose=False,
            ),
        )
    raise ValueError(f"Unknown model: {model_name}")


def align_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    ignore = set(["case_id", "surface_i", "geom", "config"] + TARGETS)
    feature_cols = [
        c for c in train_df.columns
        if c not in ignore and np.issubdtype(train_df[c].dtype, np.number)
    ]

    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = np.nan

    Xtr = train_df[feature_cols].to_numpy(dtype=float)
    Xte = test_df[feature_cols].to_numpy(dtype=float)
    Xtr[~np.isfinite(Xtr)] = np.nan
    Xte[~np.isfinite(Xte)] = np.nan

    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    inds = np.where(~np.isfinite(Xtr))
    Xtr[inds] = np.take(med, inds[1])
    inds = np.where(~np.isfinite(Xte))
    Xte[inds] = np.take(med, inds[1])

    return Xtr, Xte, feature_cols


def evaluate_case(model_name, case_id, config, y_true, y_pred, meta):
    rows = []
    for j, target in enumerate(TARGETS):
        row = {
            "model": model_name,
            "case_id": case_id,
            "config": config,
            "target": target,
            "relL2": rel_l2(y_pred[:, j], y_true[:, j]),
        }
        row.update(meta)
        rows.append(row)
    return rows


def compute_R_eps(metrics: pd.DataFrame, eps_list: List[float], rmax_label: str) -> pd.DataFrame:
    rows = []
    radius_labels = [c for c in config_order(metrics["config"].unique().tolist()) if parse_r_from_label(c) is not None]
    radius_labels = sorted(radius_labels, key=lambda c: parse_r_from_label(c))
    rmax = float(parse_r_from_label(rmax_label))

    for (model, case_id, target), g in metrics.groupby(["model", "case_id", "target"]):
        md = {r["config"]: r for _, r in g.iterrows()}
        meta0 = g.iloc[0].to_dict()

        for eps in eps_list:
            for ref_mode, ref_config in [
                ("full_domain_ring", "full"),
                ("max_cumulative_patch", rmax_label),
            ]:
                if ref_config not in md:
                    continue

                ref_err = float(md[ref_config]["relL2"])
                threshold = (1.0 + float(eps)) * ref_err

                reached = False
                Rval = np.nan
                Rlabel = "not_reached"
                err_at_R = np.nan

                for lab in radius_labels:
                    if lab not in md:
                        continue
                    e = float(md[lab]["relL2"])
                    if np.isfinite(e) and e <= threshold:
                        reached = True
                        Rval = float(parse_r_from_label(lab))
                        Rlabel = lab
                        err_at_R = e
                        break

                rows.append({
                    "model": model,
                    "case_id": case_id,
                    "target": target,
                    "epsilon": float(eps),
                    "reference_mode": ref_mode,
                    "reference_config": ref_config,
                    "reference_error": ref_err,
                    "threshold_error": threshold,
                    "reached": reached,
                    "is_censored": not reached,
                    "R_epsilon_label": Rlabel,
                    "R_epsilon_over_hs": Rval,
                    "R_epsilon_lower_bound_hs": Rval if reached else rmax,
                    "error_at_R_epsilon": err_at_R,
                    "Ma": meta0.get("Ma", np.nan),
                    "Kn": meta0.get("Kn", np.nan),
                    "geom": meta0.get("geom", ""),
                    "hphs": meta0.get("hphs", np.nan),
                    "TwTinf": meta0.get("TwTinf", np.nan),
                })

    return pd.DataFrame(rows)


def write_summaries(metrics: pd.DataFrame, Reps: pd.DataFrame, out_dir: Path):
    err_summary = (
        metrics.groupby(["model", "config", "target"])
        .agg(relL2_mean=("relL2", "mean"), relL2_median=("relL2", "median"))
        .reset_index()
        .sort_values(["model", "target", "config"])
    )
    err_summary.to_csv(out_dir / "error_summary_by_model_config_target.csv", index=False)

    Rsummary = (
        Reps.groupby(["model", "target", "epsilon", "reference_mode"])
        .agg(
            n=("case_id", "count"),
            n_reached=("reached", "sum"),
            censored_fraction=("is_censored", "mean"),
            R_lower_mean=("R_epsilon_lower_bound_hs", "mean"),
            R_lower_median=("R_epsilon_lower_bound_hs", "median"),
            ref_error_mean=("reference_error", "mean"),
        )
        .reset_index()
        .sort_values(["model", "target", "epsilon", "reference_mode"])
    )
    Rsummary.to_csv(out_dir / "R_epsilon_summary_by_model_threshold_reference.csv", index=False)

    censor_by_geom = (
        Reps.groupby(["model", "target", "epsilon", "reference_mode", "geom", "Kn"])
        .agg(
            n=("case_id", "count"),
            n_reached=("reached", "sum"),
            censored_fraction=("is_censored", "mean"),
            R_lower_mean=("R_epsilon_lower_bound_hs", "mean"),
        )
        .reset_index()
        .sort_values(["model", "target", "epsilon", "reference_mode", "geom", "Kn"])
    )
    censor_by_geom.to_csv(out_dir / "censoring_summary_by_model_threshold_reference.csv", index=False)

    lines = []
    lines.append("R95 robustness controls")
    lines.append("=======================")
    lines.append("")
    lines.append("Second-regressor error reductions:")
    for model in sorted(metrics["model"].unique()):
        sub = err_summary[(err_summary["model"] == model) & (err_summary["config"].isin(["parameter_only", "full"]))]
        if sub.empty:
            continue
        lines.append(f"Model: {model}")
        for target in TARGETS:
            p = sub[(sub["target"] == target) & (sub["config"] == "parameter_only")]["relL2_mean"]
            f = sub[(sub["target"] == target) & (sub["config"] == "full")]["relL2_mean"]
            if len(p) and len(f):
                lines.append(f"  {target}: parameter-only {100*p.iloc[0]:.2f}% -> full {100*f.iloc[0]:.2f}%")
        lines.append("")
    lines.append("Threshold/reference summary:")
    lines.append(Rsummary.to_string(index=False))
    (out_dir / "robustness_key_findings.txt").write_text("\n".join(lines))

    return err_summary, Rsummary, censor_by_geom


def plot_error_vs_radius(err_summary: pd.DataFrame, out_dir: Path):
    configs = config_order(err_summary["config"].unique().tolist())
    for model in sorted(err_summary["model"].unique()):
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), constrained_layout=True)
        for ax, target in zip(axes, TARGETS):
            sub = err_summary[(err_summary["model"] == model) & (err_summary["target"] == target)].set_index("config")
            y = [100.0 * float(sub.loc[c, "relL2_mean"]) if c in sub.index else np.nan for c in configs]
            x = np.arange(len(configs))
            ax.plot(x, y, marker="o", lw=2.2)
            ax.set_xticks(x)
            ax.set_xticklabels([clean_config_label(c) for c in configs], fontsize=9)
            ax.set_ylabel("Mean rel. error (%)")
            ax.set_title(TARGET_LABEL[target])
            ax.grid(True, axis="y", alpha=0.3)
        axes[-1].set_xlabel(r"Model input / observed radius $R/h_s$")
        fig.suptitle(f"Error-versus-radius control, model = {model}")
        fig.savefig(out_dir / f"fig_control_error_vs_radius_{model}.png", dpi=320, bbox_inches="tight")
        fig.savefig(out_dir / f"fig_control_error_vs_radius_{model}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_reference_comparison(Reps: pd.DataFrame, out_dir: Path):
    eps_values = sorted(Reps["epsilon"].unique())
    eps = 0.05 if 0.05 in eps_values else eps_values[0]
    sub0 = Reps[np.isclose(Reps["epsilon"], eps)]

    for model in sorted(sub0["model"].unique()):
        for target in TARGETS:
            sub = sub0[(sub0["model"] == model) & (sub0["target"] == target)]
            if sub.empty:
                continue
            data = []
            labels = []
            for mode, label in [("full_domain_ring", "full ring"), ("max_cumulative_patch", r"max cumulative $R/h_s=3$")]:
                vals = sub[sub["reference_mode"] == mode]["R_epsilon_lower_bound_hs"].to_numpy(float)
                if len(vals):
                    data.append(vals)
                    labels.append(label)
            fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
            ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_ylabel(r"Lower-bound $R_{\epsilon}/h_s$")
            ax.set_title(f"{TARGET_LABEL[target]}, model={model}, epsilon={eps:g}")
            ax.grid(True, axis="y", alpha=0.3)
            fig.savefig(out_dir / f"fig_control_R_eps_reference_comparison_{model}_{target}.png", dpi=320, bbox_inches="tight")
            fig.savefig(out_dir / f"fig_control_R_eps_reference_comparison_{model}_{target}.pdf", bbox_inches="tight")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nonlocal-script", required=True)
    ap.add_argument("--field-base-script", required=True)
    ap.add_argument("--surface-base-script", required=True)
    ap.add_argument("--radii", default="0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3")
    ap.add_argument("--models", default="extratrees,ridge")
    ap.add_argument("--thresholds", default="0.025,0.05,0.10")
    ap.add_argument("--max-gas-points", type=int, default=60000)
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--min-leaf", type=int, default=2)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--mlp-widths", default="128,64")
    ap.add_argument("--mlp-alpha", type=float, default=1e-4)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-batch-size", type=int, default=256)
    ap.add_argument("--mlp-max-iter", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--save-feature-tables", action="store_true")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    nl = import_module_from_path(args.nonlocal_script, "nonlocal_base_controls")
    field_base = import_module_from_path(args.field_base_script, "field_base_controls")
    surface_base = import_module_from_path(args.surface_base_script, "surface_base_controls")

    rng = np.random.default_rng(args.seed)
    radii = parse_list_float(args.radii)
    eps_list = parse_list_float(args.thresholds)
    models = parse_list_str(args.models)

    m = field_base.load_manifest(Path(args.audit_dir))
    cases = nl.get_phase1_cases(m)
    if cases.empty:
        raise RuntimeError("No Phase-1 cases found.")

    configs = [("parameter_only", None), ("local", None)] + [("R", r) for r in radii] + [("full", None)]
    label_for_config = [nl.fmt_r_label(r) if c == "R" else c for c, r in configs]
    rmax_label = nl.fmt_r_label(max(radii))

    print(f"[INFO] Phase-1 cases: {len(cases)}")
    print(f"[INFO] radii: {radii}; rmax_label={rmax_label}")
    print(f"[INFO] models: {models}")
    print(f"[INFO] thresholds: {eps_list}")

    feature_tables_parts = {lab: [] for lab in label_for_config}
    case_meta_rows = []

    for i, row in cases.iterrows():
        cd = nl.read_case_data(field_base, surface_base, row, args, rng)
        if cd is None:
            continue
        print(f"[INFO] feature extraction {i+1:02d}/{len(cases)} {cd.case_id}", flush=True)
        fdict = nl.make_features_for_case(cd, configs)
        for lab, df in fdict.items():
            feature_tables_parts[lab].append(df)
        case_meta_rows.append({
            "case_id": cd.case_id,
            "Ma": cd.Ma,
            "Kn": cd.Kn,
            "geom": cd.geom,
            "hphs": cd.hphs,
            "TwTinf": cd.TwTinf,
        })

    case_table = pd.DataFrame(case_meta_rows)
    case_table.to_csv(out_dir / "case_table_phase1.csv", index=False)

    feature_tables = {}
    for lab, parts in feature_tables_parts.items():
        if parts:
            df = pd.concat(parts, ignore_index=True)
            feature_tables[lab] = df
            if args.save_feature_tables:
                df.to_csv(out_dir / f"feature_table_{lab}.csv", index=False)

    case_ids = list(case_table["case_id"])
    metric_rows = []

    for model_i, model_name in enumerate(models):
        for cfg_i, config in enumerate(config_order(list(feature_tables.keys()))):
            df = feature_tables[config]
            print(f"[INFO] LOOCV model={model_name} config={config} samples={len(df)}", flush=True)

            for fold_i, holdout in enumerate(case_ids):
                train_df = df[df["case_id"] != holdout].copy()
                test_df = df[df["case_id"] == holdout].copy()
                if train_df.empty or test_df.empty:
                    continue

                Xtr, Xte, _ = align_feature_columns(train_df, test_df)
                ytr = train_df[TARGETS].to_numpy(float)
                yte = test_df[TARGETS].to_numpy(float)

                seed = int(args.seed + 10000 * model_i + 100 * cfg_i + fold_i)
                model = build_model(model_name, args, seed)
                model.fit(Xtr, ytr)
                ypr = model.predict(Xte)

                meta = case_table[case_table["case_id"] == holdout].iloc[0].to_dict()
                metric_rows.extend(evaluate_case(model_name, holdout, config, yte, ypr, meta))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "error_metrics_by_model_config_case.csv", index=False)

    Reps = compute_R_eps(metrics, eps_list, rmax_label)
    Reps.to_csv(out_dir / "R_epsilon_by_model_threshold_reference.csv", index=False)

    err_summary, _, _ = write_summaries(metrics, Reps, out_dir)

    if args.make_plots:
        plot_error_vs_radius(err_summary, out_dir)
        plot_reference_comparison(Reps, out_dir)

    print("[DONE] outputs written to", out_dir)
    for fn in [
        "error_summary_by_model_config_target.csv",
        "R_epsilon_by_model_threshold_reference.csv",
        "R_epsilon_summary_by_model_threshold_reference.csv",
        "censoring_summary_by_model_threshold_reference.csv",
        "robustness_key_findings.txt",
    ]:
        print("  -", out_dir / fn)


if __name__ == "__main__":
    main()
