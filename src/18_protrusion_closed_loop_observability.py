#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_protrusion_closed_loop_observability.py

Close the logical loop between the trained coordinate-conditioned field surrogates
and the nonlocal bulk-to-wall observability diagnostic.

Scientific question
-------------------
Does the information-horizon geometry inferred from the raw DSMC bulk fields remain
stable when the same held-out physical cases are represented by the reconstructed
surrogate fields?

The diagnostic regressor is ALWAYS trained on raw DSMC field descriptors from the
remaining physical cases. For each independent Phase-1 operator holdout, the exact
same fitted diagnostic is then evaluated twice:

  (A) raw DSMC patch descriptors  -> DSMC wall loads
  (B) surrogate patch descriptors -> DSMC wall loads

Thus the comparison isolates the effect of replacing the held-out bulk field with
its surrogate reconstruction. The wall targets remain the raw DSMC Cp, Cq, |tau|.

By default the independent Phase-1 holdouts are the three Ma=6, Kn=0.33 cases
(BWD/FWD/ISO), which were not used to train the field and pressure checkpoints.

Outputs include paired error curves, paired R95 values, censoring agreement, and
source-wise surface predictions for multiple diagnostic-regressor random seeds.

This is not a causal-information analysis and it is not DSMC-realization UQ.
It is a closed-loop preservation test conditional on the trained surrogate and
on the ExtraTrees diagnostic class.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from sklearn.ensemble import ExtraTreesRegressor


TARGETS = ["Cp", "Cq", "tau_abs"]
TARGET_LABEL = {"Cp": r"$C_p$", "Cq": r"$C_q$", "tau_abs": r"$|\tau|$"}
SOURCE_LABEL = {"dsmc": "Raw DSMC field", "surrogate": "Reconstructed surrogate field"}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def import_module(path: str, name: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {p}")
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_radii(text: str) -> List[float]:
    return sorted({float(x.strip()) for x in str(text).split(",") if x.strip()})


def load_checkpoint_model(base, checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    norm = ckpt["normalizers"]
    a = ckpt.get("args", {})
    model = base.SmoothFieldOperator(
        case_dim=len(norm["cmu"]),
        point_dim=len(norm["pmu"]),
        out_dim=len(norm["ymu"]),
        latent=int(a.get("latent", 160)),
        hidden=int(a.get("hidden", 224)),
        depth=int(a.get("depth", 4)),
        dropout=float(a.get("dropout", 0.03)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    data = {
        "cmu": np.asarray(norm["cmu"]), "csd": np.asarray(norm["csd"]),
        "pmu": np.asarray(norm["pmu"]), "psd": np.asarray(norm["psd"]),
        "ymu": np.asarray(norm["ymu"]), "ysd": np.asarray(norm["ysd"]),
    }
    return model, data, ckpt


def predict_surrogate_bulk(field_base, pressure_base, field_model, field_data,
                           pressure_model, pressure_data, device, row, xy, batch):
    x = xy[:, 0]
    y = xy[:, 1]

    Xp_f = field_base.smooth_point_features(x, y, row)
    Xc_f = np.repeat(field_base.case_features(row)[None, :], len(x), axis=0)
    Yf_t = field_base.predict_points(field_model, device, Xc_f, Xp_f, field_data, batch=batch)
    Yf = field_base.inverse_target_matrix(Yf_t)

    Xp_p = pressure_base.smooth_point_features(x, y, row)
    Xc_p = np.repeat(pressure_base.case_features(row)[None, :], len(x), axis=0)
    Yp_t = pressure_base.predict_points(pressure_model, device, Xc_p, Xp_p, pressure_data, batch=batch)
    P = pressure_base.inverse_target_matrix(Yp_t)[:, 0]

    u = np.asarray(Yf[:, 0], dtype=float)
    v = np.asarray(Yf[:, 1], dtype=float)
    T = np.asarray(Yf[:, 2], dtype=float)
    logP = np.log(np.maximum(np.asarray(P, dtype=float), 1e-30))
    return np.column_stack([u, v, T, logP])


def prepare_train_and_tests(nonlocal_base, train_df, raw_test, surrogate_test):
    ignore = set(["case_id", "surface_i", "geom", "config"] + TARGETS)
    feature_cols = [
        c for c in train_df.columns
        if c not in ignore and np.issubdtype(train_df[c].dtype, np.number)
    ]
    for c in feature_cols:
        if c not in raw_test.columns:
            raw_test[c] = np.nan
        if c not in surrogate_test.columns:
            surrogate_test[c] = np.nan

    Xtr = train_df[feature_cols].to_numpy(dtype=float)
    Xraw = raw_test[feature_cols].to_numpy(dtype=float)
    Xsur = surrogate_test[feature_cols].to_numpy(dtype=float)

    Xtr[~np.isfinite(Xtr)] = np.nan
    Xraw[~np.isfinite(Xraw)] = np.nan
    Xsur[~np.isfinite(Xsur)] = np.nan
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for X in (Xtr, Xraw, Xsur):
        ind = np.where(~np.isfinite(X))
        X[ind] = np.take(med, ind[1])
    return Xtr, Xraw, Xsur, feature_cols


def evaluate_curve(nonlocal_base, case_id, config, source, seed, y_true, y_pred, meta):
    rows = []
    for j, target in enumerate(TARGETS):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        row = {
            "case_id": case_id,
            "config": config,
            "source": source,
            "diagnostic_seed": seed,
            "target": target,
            "relL2": nonlocal_base.rel_l2(yp, yt),
            "rangeMAE_pct": nonlocal_base.range_mae_pct(yp, yt),
            "true_peak": float(np.nanmax(yt)),
            "pred_peak": float(np.nanmax(yp)),
            "peak_abs_err": float(abs(np.nanmax(yp) - np.nanmax(yt))),
        }
        row.update(meta)
        rows.append(row)
    return rows


def compute_r95(nonlocal_base, metrics: pd.DataFrame, radii: Sequence[float], rmax: float):
    radius_labels = [nonlocal_base.fmt_r_label(R) for R in radii]
    rows = []
    keys = ["diagnostic_seed", "source", "case_id", "target"]
    for key, g in metrics.groupby(keys):
        seed, source, case_id, target = key
        gd = {str(r["config"]): r for _, r in g.iterrows()}
        if "full" not in gd:
            continue
        full_err = float(gd["full"]["relL2"])
        threshold = 1.05 * full_err
        R95 = np.nan
        label = "not_reached"
        for R, cfg in zip(radii, radius_labels):
            if cfg in gd and np.isfinite(float(gd[cfg]["relL2"])) and float(gd[cfg]["relL2"]) <= threshold:
                R95 = float(R)
                label = cfg
                break
        m = g.iloc[0]
        censored = not np.isfinite(R95)
        rows.append({
            "diagnostic_seed": int(seed),
            "source": source,
            "case_id": case_id,
            "target": target,
            "Ma": float(m["Ma"]),
            "Kn": float(m["Kn"]),
            "geom": str(m["geom"]),
            "hphs": float(m["hphs"]),
            "full_relL2": full_err,
            "R95_over_hs": R95,
            "R95_label": label,
            "is_censored": censored,
            "R95_lower_bound_hs": rmax if censored else R95,
        })
    return pd.DataFrame(rows)


def quantile_summary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(x)), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def make_preservation_table(r95: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path):
    rows = []
    for (case_id, target), g in r95.groupby(["case_id", "target"]):
        a = g[g["source"] == "dsmc"].set_index("diagnostic_seed")
        b = g[g["source"] == "surrogate"].set_index("diagnostic_seed")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        a = a.loc[common]
        b = b.loc[common]
        censor_agree = float(np.mean(a["is_censored"].to_numpy() == b["is_censored"].to_numpy()))
        lower_diff = b["R95_lower_bound_hs"].to_numpy() - a["R95_lower_bound_hs"].to_numpy()
        md, lo, hi = quantile_summary(lower_diff)

        mf = metrics[(metrics["case_id"] == case_id) & (metrics["target"] == target) & (metrics["config"] == "full")]
        ma = mf[mf["source"] == "dsmc"].set_index("diagnostic_seed")
        mb = mf[mf["source"] == "surrogate"].set_index("diagnostic_seed")
        cc = ma.index.intersection(mb.index)
        inflation = 100.0 * (mb.loc[cc, "relL2"].to_numpy() - ma.loc[cc, "relL2"].to_numpy()) if len(cc) else np.array([])
        im, il, ih = quantile_summary(inflation)

        meta = g.iloc[0]
        rows.append({
            "case_id": case_id,
            "target": target,
            "Ma": meta["Ma"], "Kn": meta["Kn"], "geom": meta["geom"],
            "n_seeds": len(common),
            "censoring_agreement_fraction": censor_agree,
            "surrogate_minus_DSMC_R95_lower_bound_median_hs": md,
            "R95_difference_CI2p5_hs": lo,
            "R95_difference_CI97p5_hs": hi,
            "full_error_inflation_median_percentage_points": im,
            "full_error_inflation_CI2p5": il,
            "full_error_inflation_CI97p5": ih,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "closed_loop_preservation_summary.csv", index=False)
    return tab


def plot_error_curves(metrics, radii, out_dir):
    radius_cfgs = [("R" + f"{r:.2f}".replace(".", "p")).replace("p00", "") for r in radii]
    order = ["parameter_only", "local"] + radius_cfgs + ["full"]
    present = [c for c in order if c in set(metrics["config"])]
    labels = []
    for c in present:
        if c == "parameter_only": labels.append("Param.\nonly")
        elif c == "local": labels.append("Local")
        elif c == "full": labels.append("Full\ndomain")
        else:
            r = float(c[1:].replace("p", "."))
            labels.append(f"{r:g}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    colors = {"dsmc": "#1f77b4", "surrogate": "#d62728"}
    for ax, target in zip(axes, TARGETS):
        for source in ["dsmc", "surrogate"]:
            sub = metrics[(metrics["target"] == target) & (metrics["source"] == source)]
            means, los, his = [], [], []
            for cfg in present:
                vals = 100.0 * sub[sub["config"] == cfg]["relL2"].to_numpy(dtype=float)
                med, lo, hi = quantile_summary(vals)
                means.append(med); los.append(lo); his.append(hi)
            x = np.arange(len(present))
            ax.plot(x, means, marker="o", lw=2.4, ms=7, color=colors[source], label=SOURCE_LABEL[source])
            ax.fill_between(x, los, his, color=colors[source], alpha=0.18)
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
        ax.set_xlabel(r"Observed radius $R/h_s$")
        ax.set_ylabel("Relative error (%)")
        ax.set_title(TARGET_LABEL[target])
        ax.grid(True, axis="y", alpha=0.3)
    handles, labels0 = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels0, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Closed-loop preservation of bulk-to-wall error curves", y=1.10)
    fig.savefig(out_dir / "fig_closed_loop_error_curves.png", dpi=350, bbox_inches="tight")
    fig.savefig(out_dir / "fig_closed_loop_error_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_r95_pair(r95, out_dir, rmax):
    cases = list(pd.unique(r95["case_id"]))
    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 3.8 * len(cases)), squeeze=False, constrained_layout=True)
    colors = {"dsmc": "#1f77b4", "surrogate": "#d62728"}
    for i, case_id in enumerate(cases):
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            g = r95[(r95["case_id"] == case_id) & (r95["target"] == target)]
            for k, source in enumerate(["dsmc", "surrogate"]):
                vals = g[g["source"] == source]["R95_lower_bound_hs"].to_numpy(dtype=float)
                med, lo, hi = quantile_summary(vals)
                ax.errorbar(k, med, yerr=[[med-lo], [hi-med]], fmt="o", ms=9, capsize=5, color=colors[source])
                cens = g[g["source"] == source]["is_censored"].mean()
                if cens > 0:
                    ax.scatter([k], [med], s=130, facecolors="none", edgecolors="black", zorder=4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["DSMC", "Surrogate"])
            ax.set_ylim(0, rmax * 1.15)
            ax.set_ylabel(r"Lower-bound $R_{95}/h_s$")
            ax.set_title(TARGET_LABEL[target])
            ax.grid(True, axis="y", alpha=0.3)
            if j == 0:
                short = case_id.replace("PHASE1__", "").replace("_hphs1.5_Tw1", "")
                ax.text(-0.35, 0.5, short, transform=ax.transAxes, rotation=90, va="center", fontsize=10)
    fig.suptitle("Raw-DSMC versus surrogate-field information horizons\n(open overlays indicate at least partial right-censoring)")
    fig.savefig(out_dir / "fig_closed_loop_R95_pair.png", dpi=350, bbox_inches="tight")
    fig.savefig(out_dir / "fig_closed_loop_R95_pair.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nonlocal-script", required=True)
    ap.add_argument("--field-base-script", required=True)
    ap.add_argument("--field-checkpoint", required=True)
    ap.add_argument("--pressure-base-script", required=True)
    ap.add_argument("--pressure-checkpoint", required=True)
    ap.add_argument("--surface-base-script", required=True)
    ap.add_argument("--radii", default="0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3")
    ap.add_argument("--max-gas-points", type=int, default=60000)
    ap.add_argument("--predict-batch-size", type=int, default=65536)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--min-leaf", type=int, default=2)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--diagnostic-seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out).resolve())
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] device =", device, flush=True)

    nl = import_module(args.nonlocal_script, "nonlocal_base_closedloop")
    field_base = import_module(args.field_base_script, "field_base_closedloop")
    pressure_base = import_module(args.pressure_base_script, "pressure_base_closedloop")
    surface_base = import_module(args.surface_base_script, "surface_base_closedloop")

    field_model, field_data, _ = load_checkpoint_model(field_base, Path(args.field_checkpoint), device)
    pressure_model, pressure_data, _ = load_checkpoint_model(pressure_base, Path(args.pressure_checkpoint), device)

    m_all = field_base.load_manifest(Path(args.audit_dir))
    train_mask, val_mask, labels = field_base.make_split(m_all)
    labels_arr = np.asarray(labels, dtype=object)
    split_map = {str(m_all.iloc[i]["case_id"]): str(labels_arr[i]) for i in range(len(m_all))}
    phase1 = nl.get_phase1_cases(m_all)
    holdout_ids = [cid for cid in phase1["case_id"].astype(str) if "val_phase1" in split_map.get(cid, "")]
    if not holdout_ids:
        holdout_ids = phase1[np.isclose(phase1["Ma"], 6.0) & np.isclose(phase1["Kn"], 0.33)]["case_id"].astype(str).tolist()
    print("[INFO] independent Phase-1 operator holdouts:")
    for cid in holdout_ids: print("  -", cid)

    radii = parse_radii(args.radii)
    configs: List[Tuple[str, Optional[float]]] = [("parameter_only", None), ("local", None)]
    configs += [("R", R) for R in radii]
    configs += [("full", None)]
    rmax = max(radii)

    rng = np.random.default_rng(args.seed)
    raw_parts: Dict[str, List[pd.DataFrame]] = {}
    sur_parts: Dict[str, List[pd.DataFrame]] = {}
    case_rows = []

    for ii, row in phase1.iterrows():
        raw_case = nl.read_case_data(field_base, surface_base, row, args, rng)
        if raw_case is None:
            continue
        gas_z_sur = predict_surrogate_bulk(
            field_base, pressure_base, field_model, field_data,
            pressure_model, pressure_data, device, row,
            raw_case.gas_xy, args.predict_batch_size,
        )
        sur_case = replace(raw_case, gas_z=gas_z_sur)
        raw_feat = nl.make_features_for_case(raw_case, configs)
        sur_feat = nl.make_features_for_case(sur_case, configs)
        for cfg, df in raw_feat.items(): raw_parts.setdefault(cfg, []).append(df)
        for cfg, df in sur_feat.items(): sur_parts.setdefault(cfg, []).append(df)
        case_rows.append({
            "case_id": raw_case.case_id, "Ma": raw_case.Ma, "Kn": raw_case.Kn,
            "geom": raw_case.geom, "hphs": raw_case.hphs, "TwTinf": raw_case.TwTinf,
            "is_independent_operator_holdout": raw_case.case_id in holdout_ids,
        })
        print(f"[INFO] features {len(case_rows):02d}/{len(phase1)} {raw_case.case_id}", flush=True)

    raw_tables = {k: pd.concat(v, ignore_index=True) for k, v in raw_parts.items()}
    sur_tables = {k: pd.concat(v, ignore_index=True) for k, v in sur_parts.items()}
    case_table = pd.DataFrame(case_rows)
    case_table.to_csv(out_dir / "closed_loop_case_table.csv", index=False)

    metric_rows = []
    pred_rows = []
    config_names = list(raw_tables.keys())
    for sidx in range(args.diagnostic_seeds):
        seed = args.seed + 10000 + sidx
        print(f"[INFO] diagnostic seed {sidx+1}/{args.diagnostic_seeds}", flush=True)
        for cidx, cfg in enumerate(config_names):
            rdf = raw_tables[cfg]
            sdf = sur_tables[cfg]
            for fold, holdout in enumerate(holdout_ids):
                train_df = rdf[rdf["case_id"] != holdout].copy()
                raw_test = rdf[rdf["case_id"] == holdout].copy()
                sur_test = sdf[sdf["case_id"] == holdout].copy()
                if train_df.empty or raw_test.empty or sur_test.empty:
                    continue
                Xtr, Xraw, Xsur, _ = prepare_train_and_tests(nl, train_df, raw_test, sur_test)
                ytr = train_df[TARGETS].to_numpy(dtype=float)
                yte = raw_test[TARGETS].to_numpy(dtype=float)
                model = ExtraTreesRegressor(
                    n_estimators=args.trees,
                    min_samples_leaf=args.min_leaf,
                    max_features=args.max_features,
                    random_state=seed + 100*cidx + fold,
                    n_jobs=args.jobs,
                )
                model.fit(Xtr, ytr)
                pr_raw = model.predict(Xraw)
                pr_sur = model.predict(Xsur)
                cr = case_table[case_table["case_id"] == holdout].iloc[0]
                meta = {"Ma": cr.Ma, "Kn": cr.Kn, "geom": cr.geom, "hphs": cr.hphs, "TwTinf": cr.TwTinf}
                metric_rows += evaluate_curve(nl, holdout, cfg, "dsmc", sidx, yte, pr_raw, meta)
                metric_rows += evaluate_curve(nl, holdout, cfg, "surrogate", sidx, yte, pr_sur, meta)
                for source, pp in [("dsmc", pr_raw), ("surrogate", pr_sur)]:
                    for kk, (_, rr) in enumerate(raw_test.iterrows()):
                        rec = {"case_id": holdout, "config": cfg, "source": source,
                               "diagnostic_seed": sidx, "surface_i": int(rr["surface_i"]), "s01": float(rr["s01"])}
                        for j, t in enumerate(TARGETS):
                            rec[f"true_{t}"] = float(yte[kk, j])
                            rec[f"pred_{t}"] = float(pp[kk, j])
                        pred_rows.append(rec)

    metrics = pd.DataFrame(metric_rows)
    preds = pd.DataFrame(pred_rows)
    metrics.to_csv(out_dir / "closed_loop_metrics_by_case_seed.csv", index=False)
    preds.to_csv(out_dir / "closed_loop_surface_predictions.csv", index=False)

    r95 = compute_r95(nl, metrics, radii, rmax)
    r95.to_csv(out_dir / "closed_loop_R95_by_case_seed.csv", index=False)
    preservation = make_preservation_table(r95, metrics, out_dir)

    # Compact source-wise summary.
    summary_rows = []
    for (source, target, cfg), g in metrics.groupby(["source", "target", "config"]):
        med, lo, hi = quantile_summary(100.0 * g["relL2"].to_numpy(dtype=float))
        summary_rows.append({"source": source, "target": target, "config": cfg,
                             "median_relL2_pct": med, "CI2p5_pct": lo, "CI97p5_pct": hi})
    pd.DataFrame(summary_rows).to_csv(out_dir / "closed_loop_error_curve_summary.csv", index=False)

    if args.make_plots:
        plot_error_curves(metrics, radii, out_dir)
        plot_r95_pair(r95, out_dir, rmax)

    with open(out_dir / "final_summary.txt", "w") as f:
        f.write("Closed-loop observability preservation test\n")
        f.write("=========================================\n\n")
        f.write("The diagnostic was trained on raw DSMC descriptors from the remaining Phase-1 cases.\n")
        f.write("Each independent operator holdout was tested with raw DSMC descriptors and with reconstructed surrogate descriptors using the same diagnostic model.\n\n")
        f.write("Independent holdouts:\n")
        for cid in holdout_ids: f.write(f"  - {cid}\n")
        f.write("\nPreservation summary:\n")
        f.write(preservation.to_string(index=False))
        f.write("\n\nInterpretation: this tests preservation of the model-conditional predictive footprint; it does not establish causality or DSMC realization uncertainty.\n")

    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
