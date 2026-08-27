#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_protrusion_nonlocal_bulk_to_wall_footprint.py

Pilot code for the new Physics-of-Fluids idea:

    "What region of the bulk DSMC field is needed to determine wall pressure,
     heat flux, and shear stress on a triangular protrusion, and how does this
     information length depend on Kn, Mach number, and protrusion orientation?"

The script builds a nonlocal bulk-to-wall regression experiment from the existing
DSMC audit data:

    bulk field patch around each protrusion-wall point
        -> Cp(s), Cq(s), |tau|(s)

For every surface point and every physical case, the script extracts patch
statistics from the gas field variables

    z = {u, v, T, log(P)}

inside progressively larger neighborhoods of radius R around that surface point,

    R/h_s = 0.1, 0.25, 0.5, 1.0, 2.0, full-domain.

It then trains models with leave-one-physical-case-out cross validation on the
complete Phase-1 block:

    Ma = {4, 6, 8}, Kn = {0.1, 0.33, 0.8},
    geometry = {ISO, FWD, BWD}, h_p/h_s = 1.5, T_w/T_inf = 1.

The result is an "information horizon"

    R95_alpha = min R such that E_alpha(R) <= 1.05 E_alpha(full),

for alpha in {Cp, Cq, tau_abs}.  This is an empirical observability length:
the smallest bulk-region radius that recovers nearly all of the predictive
information available from the full field.

Important interpretation
------------------------
This is a pilot/diagnostic tool. It does not prove causality. It measures
predictive information in the available DSMC fields under the chosen model
class. It is designed to test whether the bulk-to-wall mapping becomes more
nonlocal as Kn increases and whether FWD/BWD/ISO geometries have different
information footprints.

Inputs expected
---------------
- audit_dir from 01_protrusion_dataset_audit_extract_v2.py
- field_base_script: usually 10_protrusion_train_smooth_field_operator_v4_geomfix.py
- surface_base_script: usually 06_protrusion_train_unified_operator_v8_geomfix.py

Example command
---------------
cd .

python3 16_protrusion_nonlocal_bulk_to_wall_footprint.py \
  --audit-dir ./protrusion_01_audit_v2 \
  --out ./protrusion_16_nonlocal_bulk_to_wall_v01 \
  --field-base-script ./10_protrusion_train_smooth_field_operator_v4_geomfix.py \
  --surface-base-script ./06_protrusion_train_unified_operator_v8_geomfix.py \
  --phase-filter phase1 \
  --radii 0.1,0.25,0.5,1.0,2.0 \
  --max-gas-points 60000 \
  --trees 350 \
  --min-leaf 2 \
  --jobs -1 \
  --make-plots

Outputs
-------
case_table_phase1.csv
surface_patch_dataset_<config>.csv
loocv_metrics_by_case.csv
loocv_predictions_surface.csv
information_horizon_R95.csv
information_horizon_summary.csv
fig_error_vs_radius.png/pdf
fig_R95_vs_Kn.png/pdf
fig_R95_vs_Ma.png/pdf
fig_R95_heatmaps.png/pdf
fig_example_surface_profiles.png/pdf
final_summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score


try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    cKDTree = None
    SCIPY_AVAILABLE = False


TARGETS = ["Cp", "Cq", "tau_abs"]
FIELD_VARS = ["u", "v", "T", "logP"]
GEOM_ORDER = ["BWD", "FWD", "ISO"]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def import_module_from_path(path: str, name: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {p}")
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_radii(s: str) -> List[float]:
    vals = []
    for token in str(s).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    vals = sorted(set(vals))
    return vals


def fmt_r_label(R: float) -> str:
    return ("R" + f"{R:.2f}".replace(".", "p")).replace("p00", "")


def rel_l2(pred, true, eps=1e-12) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    m = np.isfinite(pred) & np.isfinite(true)
    if m.sum() < 2:
        return np.nan
    return float(np.linalg.norm(pred[m] - true[m]) / (np.linalg.norm(true[m]) + eps))


def range_mae_pct(pred, true, qlo=1.0, qhi=99.0, eps=1e-12) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    m = np.isfinite(pred) & np.isfinite(true)
    if m.sum() < 3:
        return np.nan
    lo, hi = np.nanpercentile(true[m], [qlo, qhi])
    den = max(float(hi - lo), eps)
    return float(100.0 * np.nanmean(np.abs(pred[m] - true[m])) / den)


def safe_logp(P):
    P = np.asarray(P, dtype=float)
    finite = np.isfinite(P) & (P > 0)
    floor = np.nanpercentile(P[finite], 0.1) if finite.any() else 1e-12
    floor = max(float(floor), 1e-12)
    return np.log(np.maximum(P, floor))


def numeric_case_id(row) -> str:
    return str(row.get("case_id", row.get("physical_key", "")))


def get_phase1_cases(m: pd.DataFrame) -> pd.DataFrame:
    phase = m["phase"].astype(str).str.upper()
    out = m[
        (phase == "PHASE1")
        & np.isclose(m["hphs"].astype(float), 1.5)
        & np.isclose(m["TwTinf"].astype(float), 1.0)
    ].copy()
    out = out.sort_values(["Ma", "Kn", "geom", "case_id"]).reset_index(drop=True)
    return out


def one_hot_geom(geom: str) -> Dict[str, float]:
    return {f"geom_{g}": 1.0 if str(geom).upper() == g else 0.0 for g in GEOM_ORDER}


def contraction_label(name: str) -> str:
    return name.replace("tau_abs", "|tau|")


# ---------------------------------------------------------------------
# Surface and field containers
# ---------------------------------------------------------------------

@dataclass
class SurfaceGeomData:
    xmid: np.ndarray
    ymid: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    seg_len: np.ndarray
    s01: np.ndarray
    dist_apex: np.ndarray
    face_left: np.ndarray
    face_right: np.ndarray
    xnorm: np.ndarray
    ynorm: np.ndarray


@dataclass
class CaseData:
    case_id: str
    row: pd.Series
    geom: str
    Ma: float
    Kn: float
    hphs: float
    TwTinf: float
    hs: float
    hp: float
    apex_x: float
    apex_y: float
    gas_xy: np.ndarray
    gas_z: np.ndarray
    tree: Optional[object]
    surface: pd.DataFrame
    surf_geom: SurfaceGeomData
    targets: np.ndarray


def compute_surface_geometry(surface_base, row: pd.Series, surface: pd.DataFrame) -> SurfaceGeomData:
    geom = surface_base.geometry_from_row(row)
    hs = max(float(geom.hs), 1e-12)

    s01 = surface["s01"].to_numpy(dtype=float)
    n = len(surface)

    if all(c in surface.columns for c in ["v1x", "v1y", "v2x", "v2y"]):
        v1x = surface["v1x"].to_numpy(dtype=float)
        v1y = surface["v1y"].to_numpy(dtype=float)
        v2x = surface["v2x"].to_numpy(dtype=float)
        v2y = surface["v2y"].to_numpy(dtype=float)
        xmid = 0.5 * (v1x + v2x)
        ymid = 0.5 * (v1y + v2y)
        dx = v2x - v1x
        dy = v2y - v1y
        seg_len_phys = np.sqrt(dx * dx + dy * dy)
        seg_len_phys = np.where(seg_len_phys > 1e-14, seg_len_phys, 1.0)
        tx = dx / seg_len_phys
        ty = dy / seg_len_phys
        nx = -ty
        ny = tx
        seg_len = seg_len_phys / hs
    else:
        # Fallback: synthetic coordinates on the two triangle sides.
        verts = geom.vertices
        left = verts[0]
        right = verts[1]
        apex = verts[2]
        xmid = np.empty(n, dtype=float)
        ymid = np.empty(n, dtype=float)
        tx = np.empty(n, dtype=float)
        ty = np.empty(n, dtype=float)
        nx = np.empty(n, dtype=float)
        ny = np.empty(n, dtype=float)
        seg_len = np.ones(n, dtype=float)
        for i, s in enumerate(s01):
            if s <= 0.5:
                t = s / 0.5
                p0 = left
                p1 = apex
            else:
                t = (s - 0.5) / 0.5
                p0 = apex
                p1 = right
            p = (1 - t) * p0 + t * p1
            xmid[i], ymid[i] = p
            vec = p1 - p0
            L = max(float(np.linalg.norm(vec)), 1e-12)
            tx[i], ty[i] = vec / L
            nx[i], ny[i] = -ty[i], tx[i]
            seg_len[i] = L / hs

    d_ap = np.sqrt((xmid - geom.apex_x) ** 2 + (ymid - geom.apex_y) ** 2)
    if np.isfinite(d_ap).any():
        apex_idx = int(np.nanargmin(d_ap))
        apex_s01 = float(s01[apex_idx])
    else:
        apex_s01 = 0.5

    face_left = (s01 <= apex_s01).astype(float)
    face_right = 1.0 - face_left

    xnorm = (xmid - geom.apex_x) / hs
    ynorm = (ymid - geom.y_min) / hs
    dist_apex = d_ap / hs

    return SurfaceGeomData(
        xmid=xmid, ymid=ymid,
        tx=tx, ty=ty, nx=nx, ny=ny,
        seg_len=seg_len, s01=s01,
        dist_apex=dist_apex,
        face_left=face_left, face_right=face_right,
        xnorm=xnorm, ynorm=ynorm,
    )


def read_case_data(field_base, surface_base, row: pd.Series, args, rng: np.random.Generator) -> Optional[CaseData]:
    case_id = numeric_case_id(row)
    geom_obj = surface_base.geometry_from_row(row)
    geom = str(row.get("geom", geom_obj.geom)).upper()

    field = field_base.read_field(row)
    gas_mask = field_base.gas_mask(field, row)
    idx_all = np.flatnonzero(gas_mask)
    if len(idx_all) == 0:
        print(f"[WARN] no gas cells for {case_id}")
        return None

    if args.max_gas_points > 0 and len(idx_all) > args.max_gas_points:
        idx = rng.choice(idx_all, size=int(args.max_gas_points), replace=False)
    else:
        idx = idx_all

    x = field["x"][idx].astype(float)
    y = field["y"][idx].astype(float)
    trans = field_base.target_matrix(field)[idx]
    phys = field_base.inverse_target_matrix(trans).astype(float)  # [u,v,T,P]
    u = phys[:, 0]
    v = phys[:, 1]
    T = phys[:, 2]
    P = phys[:, 3]
    logP = safe_logp(P)
    gas_z = np.column_stack([u, v, T, logP]).astype(float)
    gas_xy = np.column_stack([x, y]).astype(float)

    if SCIPY_AVAILABLE:
        tree = cKDTree(gas_xy)
    else:
        tree = None

    surface = surface_base.read_prot_surface(str(row["zip_path"]), str(row["xlsx_inner"]))
    if surface is None or surface.empty:
        print(f"[WARN] no protrusion surface for {case_id}")
        return None

    needed = ["Cp", "Cq", "tau_abs"]
    if any(c not in surface.columns for c in needed):
        print(f"[WARN] missing surface target columns for {case_id}")
        return None

    surf_geom = compute_surface_geometry(surface_base, row, surface)
    targets = surface[needed].to_numpy(dtype=float)

    return CaseData(
        case_id=case_id,
        row=row,
        geom=geom,
        Ma=float(row["Ma"]),
        Kn=float(row["Kn"]),
        hphs=float(row["hphs"]),
        TwTinf=float(row["TwTinf"]),
        hs=float(geom_obj.hs),
        hp=float(geom_obj.hp),
        apex_x=float(geom_obj.apex_x),
        apex_y=float(geom_obj.apex_y),
        gas_xy=gas_xy,
        gas_z=gas_z,
        tree=tree,
        surface=surface,
        surf_geom=surf_geom,
        targets=targets,
    )


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def base_surface_feature_dict(case: CaseData, i: int) -> Dict[str, float]:
    sg = case.surf_geom
    d = {
        "Ma": case.Ma,
        "log10Kn": math.log10(case.Kn),
        "Kn": case.Kn,
        "hphs": case.hphs,
        "TwTinf": case.TwTinf,
        "s01": float(sg.s01[i]),
        "xsurf_norm": float(sg.xnorm[i]),
        "ysurf_norm": float(sg.ynorm[i]),
        "tx": float(sg.tx[i]),
        "ty": float(sg.ty[i]),
        "nx": float(sg.nx[i]),
        "ny": float(sg.ny[i]),
        "seg_len_hs": float(sg.seg_len[i]),
        "dist_apex_hs": float(sg.dist_apex[i]),
        "face_left": float(sg.face_left[i]),
        "face_right": float(sg.face_right[i]),
    }
    d.update(one_hot_geom(case.geom))
    return d


def _stats_for_indices(
    case: CaseData,
    i_surf: int,
    idx: np.ndarray,
    prefix: str,
    min_points: int = 5,
) -> Dict[str, float]:
    sg = case.surf_geom
    hs = max(case.hs, 1e-12)

    p = np.array([sg.xmid[i_surf], sg.ymid[i_surf]], dtype=float)
    tx = np.array([sg.tx[i_surf], sg.ty[i_surf]], dtype=float)
    nx = np.array([sg.nx[i_surf], sg.ny[i_surf]], dtype=float)

    # If too few points, fall back to nearest neighbors.
    if idx is None or len(idx) < min_points:
        idx = nearest_indices(case, p, max(min_points, 12))

    pts = case.gas_xy[idx]
    z = case.gas_z[idx]
    rel = (pts - p[None, :]) / hs
    rt = rel @ tx
    rn = rel @ nx
    rr = np.sqrt(np.sum(rel * rel, axis=1))

    # Smooth distance weights. Use a local scale based on the selected cloud.
    scale = np.nanpercentile(rr, 75) if len(rr) > 4 else np.nanmax(rr)
    scale = max(float(scale), 1e-6)
    w = np.exp(-0.5 * (rr / scale) ** 2)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(rr)
    w = w / np.sum(w)

    out: Dict[str, float] = {}
    out[f"{prefix}_npts"] = float(len(idx))
    out[f"{prefix}_mean_r_hs"] = float(np.sum(w * rr))
    out[f"{prefix}_std_r_hs"] = float(np.sqrt(np.sum(w * (rr - out[f'{prefix}_mean_r_hs']) ** 2)))
    out[f"{prefix}_mean_rt"] = float(np.sum(w * rt))
    out[f"{prefix}_mean_rn"] = float(np.sum(w * rn))

    # Design matrix for local gradient in tangent/normal coordinates.
    X = np.column_stack([np.ones_like(rt), rt, rn])
    sw = np.sqrt(np.maximum(w, 1e-12))
    Xw = X * sw[:, None]

    for j, name in enumerate(FIELD_VARS):
        vals = z[:, j]
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            for key in ["mean", "std", "min", "max", "nearest", "grad_t", "grad_n"]:
                out[f"{prefix}_{name}_{key}"] = np.nan
            continue

        vf = vals.copy()
        med = np.nanmedian(vf[finite])
        vf[~finite] = med

        mu = float(np.sum(w * vf))
        var = float(np.sum(w * (vf - mu) ** 2))
        out[f"{prefix}_{name}_mean"] = mu
        out[f"{prefix}_{name}_std"] = float(np.sqrt(max(var, 0.0)))
        out[f"{prefix}_{name}_min"] = float(np.nanmin(vf))
        out[f"{prefix}_{name}_max"] = float(np.nanmax(vf))

        k_near = int(np.nanargmin(rr))
        out[f"{prefix}_{name}_nearest"] = float(vf[k_near])

        # Weighted linear fit z = a + b_t * rt + b_n * rn.
        try:
            yw = vf * sw
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            out[f"{prefix}_{name}_grad_t"] = float(beta[1])
            out[f"{prefix}_{name}_grad_n"] = float(beta[2])
        except Exception:
            out[f"{prefix}_{name}_grad_t"] = 0.0
            out[f"{prefix}_{name}_grad_n"] = 0.0

    return out


def nearest_indices(case: CaseData, p: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(case.gas_xy))
    if SCIPY_AVAILABLE and case.tree is not None:
        _d, idx = case.tree.query(p, k=k)
        return np.atleast_1d(idx).astype(int)
    d2 = np.sum((case.gas_xy - p[None, :]) ** 2, axis=1)
    return np.argsort(d2)[:k].astype(int)


def radius_indices(case: CaseData, p: np.ndarray, R_phys: float) -> np.ndarray:
    if SCIPY_AVAILABLE and case.tree is not None:
        idx = case.tree.query_ball_point(p, r=float(R_phys))
        return np.asarray(idx, dtype=int)
    d2 = np.sum((case.gas_xy - p[None, :]) ** 2, axis=1)
    return np.flatnonzero(d2 <= R_phys * R_phys).astype(int)


def ring_indices(case: CaseData, p: np.ndarray, r0_hs: float, r1_hs: Optional[float]) -> np.ndarray:
    hs = max(case.hs, 1e-12)
    rel = (case.gas_xy - p[None, :]) / hs
    rr = np.sqrt(np.sum(rel * rel, axis=1))
    if r1_hs is None:
        return np.flatnonzero(rr > r0_hs).astype(int)
    return np.flatnonzero((rr > r0_hs) & (rr <= r1_hs)).astype(int)


def feature_dict_for_config(case: CaseData, i_surf: int, config: str, R: Optional[float]) -> Dict[str, float]:
    out = base_surface_feature_dict(case, i_surf)
    p = np.array([case.surf_geom.xmid[i_surf], case.surf_geom.ymid[i_surf]], dtype=float)

    if config == "parameter_only":
        return out

    if config == "local":
        idx = nearest_indices(case, p, 16)
        out.update(_stats_for_indices(case, i_surf, idx, prefix="local"))
        return out

    if config == "full":
        # Multi-ring information footprint. This is stronger than a single
        # global average because it allows the model to use how information is
        # distributed with distance from the wall point.
        bins = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.0), (1.0, 2.0), (2.0, None)]
        for r0, r1 in bins:
            label = f"ring_{str(r0).replace('.','p')}_{'inf' if r1 is None else str(r1).replace('.','p')}"
            idx = ring_indices(case, p, r0, r1)
            out.update(_stats_for_indices(case, i_surf, idx, prefix=label, min_points=5))
        return out

    if config.startswith("R"):
        if R is None:
            raise ValueError("R config needs R value")
        R_phys = float(R) * case.hs
        idx = radius_indices(case, p, R_phys)
        out.update(_stats_for_indices(case, i_surf, idx, prefix=fmt_r_label(R), min_points=5))
        return out

    raise ValueError(f"Unknown config: {config}")


def make_features_for_case(case: CaseData, configs: Sequence[Tuple[str, Optional[float]]]) -> Dict[str, pd.DataFrame]:
    outputs = {}
    n = len(case.surface)
    for config, R in configs:
        label = config if config in ["parameter_only", "local", "full"] else fmt_r_label(float(R))
        rows = []
        for i in range(n):
            d = {
                "case_id": case.case_id,
                "surface_i": i,
                "geom": case.geom,
                "Ma": case.Ma,
                "Kn": case.Kn,
                "hphs": case.hphs,
                "TwTinf": case.TwTinf,
                "config": label,
            }
            feats = feature_dict_for_config(case, i, config, R)
            d.update(feats)
            for j, target in enumerate(TARGETS):
                d[target] = float(case.targets[i, j])
            rows.append(d)
        outputs[label] = pd.DataFrame(rows)
    return outputs


def align_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ignore = set(["case_id", "surface_i", "geom", "config"] + TARGETS)
    feature_cols = [c for c in train_df.columns if c not in ignore and np.issubdtype(train_df[c].dtype, np.number)]
    # Ensure columns in test.
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = np.nan
    Xtr = train_df[feature_cols].to_numpy(dtype=float)
    Xte = test_df[feature_cols].to_numpy(dtype=float)
    # Replace NaNs/Infs with training medians.
    Xtr[~np.isfinite(Xtr)] = np.nan
    Xte[~np.isfinite(Xte)] = np.nan
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(Xtr))
    Xtr[inds] = np.take(med, inds[1])
    inds = np.where(~np.isfinite(Xte))
    Xte[inds] = np.take(med, inds[1])
    return Xtr, Xte, feature_cols


# ---------------------------------------------------------------------
# Modeling and evaluation
# ---------------------------------------------------------------------

def build_model(args, seed: int):
    base = ExtraTreesRegressor(
        n_estimators=int(args.trees),
        max_depth=None if args.max_depth <= 0 else int(args.max_depth),
        min_samples_leaf=int(args.min_leaf),
        max_features=args.max_features,
        bootstrap=False,
        random_state=int(seed),
        n_jobs=int(args.jobs),
    )
    return MultiOutputRegressor(base, n_jobs=1)


def evaluate_prediction(case_id, config, y_true, y_pred, case_meta) -> List[Dict[str, float]]:
    rows = []
    for j, target in enumerate(TARGETS):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        k_true = int(np.nanargmax(yt)) if np.isfinite(yt).any() else -1
        k_pred = int(np.nanargmax(yp)) if np.isfinite(yp).any() else -1
        row = {
            "case_id": case_id,
            "config": config,
            "target": target,
            "relL2": rel_l2(yp, yt),
            "rangeMAE_pct": range_mae_pct(yp, yt),
            "true_peak": float(np.nanmax(yt)),
            "pred_peak": float(np.nanmax(yp)),
            "peak_abs_err": float(abs(np.nanmax(yp) - np.nanmax(yt))),
            "true_peak_surf_i": k_true,
            "pred_peak_surf_i": k_pred,
            "peak_index_error": float(abs(k_pred - k_true)) if k_true >= 0 and k_pred >= 0 else np.nan,
        }
        row.update(case_meta)
        rows.append(row)
    return rows


def run_loocv(feature_tables: Dict[str, pd.DataFrame], case_table: pd.DataFrame, args, out_dir: Path):
    configs = list(feature_tables.keys())
    case_ids = case_table["case_id"].tolist()

    metrics_rows = []
    pred_rows = []

    for cfg_i, config in enumerate(configs, start=1):
        df = feature_tables[config]
        print(f"[INFO] LOOCV config {cfg_i}/{len(configs)}: {config}  samples={len(df)}", flush=True)

        for fold_i, holdout_case in enumerate(case_ids, start=1):
            train_df = df[df["case_id"] != holdout_case].copy()
            test_df = df[df["case_id"] == holdout_case].copy()
            if train_df.empty or test_df.empty:
                continue

            Xtr, Xte, feature_cols = align_feature_columns(train_df, test_df)
            ytr = train_df[TARGETS].to_numpy(dtype=float)
            yte = test_df[TARGETS].to_numpy(dtype=float)

            model = build_model(args, seed=args.seed + 1000 * cfg_i + fold_i)
            model.fit(Xtr, ytr)
            ypr = model.predict(Xte)

            meta_row = case_table[case_table["case_id"] == holdout_case].iloc[0].to_dict()
            meta = {
                "Ma": float(meta_row["Ma"]),
                "Kn": float(meta_row["Kn"]),
                "hphs": float(meta_row["hphs"]),
                "TwTinf": float(meta_row["TwTinf"]),
                "geom": str(meta_row["geom"]),
                "hs": float(meta_row["hs"]),
                "hp": float(meta_row["hp"]),
            }

            metrics_rows.extend(evaluate_prediction(holdout_case, config, yte, ypr, meta))

            # Store surface-wise predictions for later profile plotting.
            for row_i, (_, sr) in enumerate(test_df.iterrows()):
                out = {
                    "case_id": holdout_case,
                    "config": config,
                    "surface_i": int(sr["surface_i"]),
                    "s01": float(sr["s01"]),
                    "Ma": meta["Ma"],
                    "Kn": meta["Kn"],
                    "geom": meta["geom"],
                    "hphs": meta["hphs"],
                    "TwTinf": meta["TwTinf"],
                }
                for j, target in enumerate(TARGETS):
                    out[f"true_{target}"] = float(yte[row_i, j])
                    out[f"pred_{target}"] = float(ypr[row_i, j])
                pred_rows.append(out)

    metrics = pd.DataFrame(metrics_rows)
    preds = pd.DataFrame(pred_rows)

    metrics.to_csv(out_dir / "loocv_metrics_by_case.csv", index=False)
    preds.to_csv(out_dir / "loocv_predictions_surface.csv", index=False)

    return metrics, preds


def compute_information_horizon(metrics: pd.DataFrame, radii: Sequence[float], out_dir: Path) -> pd.DataFrame:
    radius_labels = [fmt_r_label(R) for R in radii]
    rows = []

    for (case_id, target), g in metrics.groupby(["case_id", "target"]):
        gd = {r["config"]: r for _, r in g.iterrows()}
        if "full" not in gd:
            continue
        full_err = float(gd["full"]["relL2"])
        if not np.isfinite(full_err):
            continue
        threshold = 1.05 * full_err

        best_cfg = None
        best_err = np.inf
        for _, r in g.iterrows():
            e = float(r["relL2"])
            if np.isfinite(e) and e < best_err:
                best_err = e
                best_cfg = str(r["config"])

        R95 = np.nan
        R95_label = "not_reached"
        for R, label in zip(radii, radius_labels):
            if label in gd:
                e = float(gd[label]["relL2"])
                if np.isfinite(e) and e <= threshold:
                    R95 = float(R)
                    R95_label = label
                    break

        meta = g.iloc[0].to_dict()
        Kn = float(meta["Kn"])
        hphs = float(meta["hphs"])
        # In the Sabouri-Lekzian setup, Kn is defined based on protrusion height hp.
        # Since R is normalized by hs, lambda/hs = Kn * hp/hs = Kn*hphs.
        R95_over_lambda = R95 / (Kn * hphs) if np.isfinite(R95) and Kn > 0 and hphs > 0 else np.nan

        rows.append({
            "case_id": case_id,
            "target": target,
            "Ma": float(meta["Ma"]),
            "Kn": Kn,
            "geom": str(meta["geom"]),
            "hphs": hphs,
            "TwTinf": float(meta["TwTinf"]),
            "full_relL2": full_err,
            "threshold_1p05_full": threshold,
            "R95_over_hs": R95,
            "R95_label": R95_label,
            "R95_over_lambda_assuming_Kn_based_on_hp": R95_over_lambda,
            "best_config": best_cfg,
            "best_relL2": best_err,
        })

    horizon = pd.DataFrame(rows)
    horizon.to_csv(out_dir / "information_horizon_R95.csv", index=False)

    summary = horizon.groupby(["target", "geom", "Kn"])[["R95_over_hs", "R95_over_lambda_assuming_Kn_based_on_hp", "full_relL2", "best_relL2"]].mean().reset_index()
    summary.to_csv(out_dir / "information_horizon_summary.csv", index=False)
    return horizon


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_error_vs_radius(metrics: pd.DataFrame, radii: Sequence[float], out_dir: Path):
    order = ["parameter_only", "local"] + [fmt_r_label(R) for R in radii] + ["full"]
    available = [c for c in order if c in set(metrics["config"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        vals = []
        for cfg in available:
            sub = metrics[(metrics["target"] == target) & (metrics["config"] == cfg)]
            vals.append(100.0 * sub["relL2"].mean())
        ax.plot(np.arange(len(available)), vals, marker="o")
        ax.set_xticks(np.arange(len(available)))
        ax.set_xticklabels(available, rotation=40, ha="right")
        ax.set_ylabel("LOOCV relative L2 (%)")
        ax.set_title(contraction_label(target))
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Bulk-to-wall surface-load error versus observed bulk-field radius")
    fig.savefig(out_dir / "fig_error_vs_radius.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_error_vs_radius.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_R95_vs_Kn(horizon: pd.DataFrame, out_dir: Path):
    if horizon.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    markers = {"BWD": "o", "FWD": "s", "ISO": "^"}
    for ax, target in zip(axes, TARGETS):
        sub = horizon[horizon["target"] == target].copy()
        for geom, gg in sub.groupby("geom"):
            gsum = gg.groupby("Kn")["R95_over_hs"].mean().reset_index().sort_values("Kn")
            ax.plot(gsum["Kn"], gsum["R95_over_hs"], marker=markers.get(geom, "o"), label=geom)
        ax.set_xscale("log")
        ax.set_xlabel("Kn")
        ax.set_ylabel(r"$R_{95}/h_s$")
        ax.set_title(contraction_label(target))
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.suptitle("Knudsen-dependent information horizon")
    fig.savefig(out_dir / "fig_R95_vs_Kn.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_R95_vs_Kn.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_R95_vs_Ma(horizon: pd.DataFrame, out_dir: Path):
    if horizon.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    markers = {"BWD": "o", "FWD": "s", "ISO": "^"}
    for ax, target in zip(axes, TARGETS):
        sub = horizon[horizon["target"] == target].copy()
        for geom, gg in sub.groupby("geom"):
            gsum = gg.groupby("Ma")["R95_over_hs"].mean().reset_index().sort_values("Ma")
            ax.plot(gsum["Ma"], gsum["R95_over_hs"], marker=markers.get(geom, "o"), label=geom)
        ax.set_xlabel("Ma")
        ax.set_ylabel(r"$R_{95}/h_s$")
        ax.set_title(contraction_label(target))
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle("Mach-number dependence of information horizon")
    fig.savefig(out_dir / "fig_R95_vs_Ma.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_R95_vs_Ma.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_R95_heatmaps(horizon: pd.DataFrame, out_dir: Path):
    if horizon.empty:
        return
    Mas = sorted(horizon["Ma"].unique())
    Kns = sorted(horizon["Kn"].unique())
    geoms = [g for g in GEOM_ORDER if g in set(horizon["geom"])]
    fig, axes = plt.subplots(len(TARGETS), len(geoms), figsize=(4.2 * len(geoms), 3.6 * len(TARGETS)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, target in enumerate(TARGETS):
        for j, geom in enumerate(geoms):
            ax = axes[i, j]
            mat = np.full((len(Kns), len(Mas)), np.nan)
            sub = horizon[(horizon["target"] == target) & (horizon["geom"] == geom)]
            for _, r in sub.iterrows():
                iy = Kns.index(r["Kn"])
                ix = Mas.index(r["Ma"])
                mat[iy, ix] = r["R95_over_hs"]
            im = ax.imshow(mat, aspect="auto", origin="lower")
            ax.set_xticks(np.arange(len(Mas)))
            ax.set_xticklabels([str(int(m)) if float(m).is_integer() else str(m) for m in Mas])
            ax.set_yticks(np.arange(len(Kns)))
            ax.set_yticklabels([str(k) for k in Kns])
            ax.set_xlabel("Ma")
            ax.set_ylabel("Kn")
            ax.set_title(f"{contraction_label(target)} / {geom}")
            for iy in range(len(Kns)):
                for ix in range(len(Mas)):
                    if np.isfinite(mat[iy, ix]):
                        ax.text(ix, iy, f"{mat[iy, ix]:.2g}", ha="center", va="center", fontsize=8, color="white" if mat[iy, ix] > np.nanmax(mat)*0.5 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(r"Information horizon $R_{95}/h_s$ across Phase-1 physical conditions")
    fig.savefig(out_dir / "fig_R95_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_R95_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_example_surface_profiles(preds: pd.DataFrame, horizon: pd.DataFrame, out_dir: Path):
    if preds.empty:
        return
    # Pick representative center cases if available, otherwise first three.
    preferred = [
        "PHASE1__Ma6_Kn0.33_BWD_hphs1.5_Tw1",
        "PHASE1__Ma6_Kn0.33_FWD_hphs1.5_Tw1",
        "PHASE1__Ma6_Kn0.33_ISO_hphs1.5_Tw1",
    ]
    available_cases = list(pd.unique(preds["case_id"]))
    cases = [c for c in preferred if c in available_cases]
    if len(cases) < 3:
        cases = available_cases[:3]

    # Use configs that are most interpretable.
    configs = [c for c in ["parameter_only", "local", "R0p25", "R0p50", "R1", "R2", "full"] if c in set(preds["config"])]

    fig, axes = plt.subplots(len(cases), len(TARGETS), figsize=(5.0 * len(TARGETS), 3.6 * len(cases)), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, case_id in enumerate(cases):
        case_df = preds[preds["case_id"] == case_id]
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            # True from full config or first config.
            true_df = case_df[case_df["config"] == configs[-1]].sort_values("s01")
            if true_df.empty:
                true_df = case_df[case_df["config"] == case_df["config"].iloc[0]].sort_values("s01")
            ax.plot(true_df["s01"], true_df[f"true_{target}"], color="black", lw=2.0, label="DSMC")
            for cfg in configs:
                if cfg == "parameter_only":
                    lw = 0.9
                    alpha = 0.6
                elif cfg == "full":
                    lw = 1.8
                    alpha = 0.9
                else:
                    lw = 1.1
                    alpha = 0.75
                sub = case_df[case_df["config"] == cfg].sort_values("s01")
                if sub.empty:
                    continue
                ax.plot(sub["s01"], sub[f"pred_{target}"], lw=lw, alpha=alpha, label=cfg)
            ax.set_title(f"{case_id}\n{contraction_label(target)}", fontsize=8)
            ax.set_xlabel("normalized protrusion coordinate")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Representative surface profiles from nonlocal bulk-to-wall models")
    fig.savefig(out_dir / "fig_example_surface_profiles.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_example_surface_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field-base-script", required=True)
    ap.add_argument("--surface-base-script", required=True)
    ap.add_argument("--phase-filter", default="phase1", choices=["phase1"])
    ap.add_argument("--radii", default="0.1,0.25,0.5,1.0,2.0")
    ap.add_argument("--max-gas-points", type=int, default=60000, help="Random gas cells kept per case for feature extraction. <=0 keeps all.")
    ap.add_argument("--trees", type=int, default=350)
    ap.add_argument("--max-depth", type=int, default=0, help="0 means no max depth.")
    ap.add_argument("--min-leaf", type=int, default=2)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-feature-tables", action="store_true")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    rng = np.random.default_rng(args.seed)

    field_base = import_module_from_path(args.field_base_script, "field_base_nonlocal")
    surface_base = import_module_from_path(args.surface_base_script, "surface_base_nonlocal")

    m = field_base.load_manifest(Path(args.audit_dir))
    m = get_phase1_cases(m)
    if m.empty:
        raise RuntimeError("No Phase-1 cases found. Check audit directory and manifest.")

    print(f"[INFO] Phase-1 unique cases selected: {len(m)}", flush=True)
    if len(m) != 27:
        print(f"[WARN] Expected 27 Phase-1 cases, found {len(m)}. Continuing.", flush=True)

    radii = parse_radii(args.radii)
    configs: List[Tuple[str, Optional[float]]] = [("parameter_only", None), ("local", None)]
    configs += [("R", R) for R in radii]
    configs += [("full", None)]

    # Read cases and build feature tables.
    case_datas: List[CaseData] = []
    case_rows = []
    feature_tables: Dict[str, List[pd.DataFrame]] = {("parameter_only" if c[0] == "parameter_only" else "local" if c[0] == "local" else "full" if c[0] == "full" else fmt_r_label(c[1])): [] for c in configs}

    for i, row in m.iterrows():
        cd = read_case_data(field_base, surface_base, row, args, rng)
        if cd is None:
            continue
        case_datas.append(cd)
        case_rows.append({
            "case_id": cd.case_id,
            "Ma": cd.Ma,
            "Kn": cd.Kn,
            "geom": cd.geom,
            "hphs": cd.hphs,
            "TwTinf": cd.TwTinf,
            "hs": cd.hs,
            "hp": cd.hp,
            "n_gas": len(cd.gas_xy),
            "n_surface": len(cd.surface),
            "lambda_over_hs_assuming_Kn_based_on_hp": cd.Kn * cd.hphs,
        })
        print(f"[INFO] building features {len(case_datas):02d}: {cd.case_id} gas={len(cd.gas_xy)} surface={len(cd.surface)}", flush=True)
        fdict = make_features_for_case(cd, configs)
        for label, df in fdict.items():
            feature_tables[label].append(df)

    case_table = pd.DataFrame(case_rows)
    case_table.to_csv(out_dir / "case_table_phase1.csv", index=False)

    feature_tables_df: Dict[str, pd.DataFrame] = {}
    for label, parts in feature_tables.items():
        if parts:
            df = pd.concat(parts, ignore_index=True)
            feature_tables_df[label] = df
            if args.save_feature_tables:
                df.to_csv(out_dir / f"surface_patch_dataset_{label}.csv", index=False)

    # Always save a compact metadata table for all samples for traceability.
    compact = feature_tables_df[list(feature_tables_df.keys())[0]][["case_id", "surface_i", "geom", "Ma", "Kn", "hphs", "TwTinf", "s01"] + TARGETS].copy()
    compact.to_csv(out_dir / "surface_samples_metadata.csv", index=False)

    metrics, preds = run_loocv(feature_tables_df, case_table, args, out_dir)
    horizon = compute_information_horizon(metrics, radii, out_dir)

    # Summary tables
    summary = metrics.groupby(["config", "target"])[["relL2", "rangeMAE_pct", "peak_abs_err", "peak_index_error"]].mean().reset_index()
    summary.to_csv(out_dir / "error_summary_by_config_target.csv", index=False)

    by_phys = metrics.groupby(["config", "target", "Ma", "Kn", "geom"])[["relL2", "rangeMAE_pct"]].mean().reset_index()
    by_phys.to_csv(out_dir / "error_summary_by_physical_condition.csv", index=False)

    if args.make_plots:
        plot_error_vs_radius(metrics, radii, out_dir)
        plot_R95_vs_Kn(horizon, out_dir)
        plot_R95_vs_Ma(horizon, out_dir)
        plot_R95_heatmaps(horizon, out_dir)
        plot_example_surface_profiles(preds, horizon, out_dir)

    # Final text summary
    with open(out_dir / "final_summary.txt", "w") as f:
        f.write("Nonlocal bulk-to-wall information footprint study\n")
        f.write("=================================================\n\n")
        f.write("Physical question:\n")
        f.write("  What radius of the bulk DSMC field is needed to predict Cp, Cq, and |tau| on the protrusion wall?\n\n")
        f.write("Phase-1 cases:\n")
        f.write(f"  {len(case_table)} physical cases; Ma={sorted(case_table['Ma'].unique())}; Kn={sorted(case_table['Kn'].unique())}; geom={sorted(case_table['geom'].unique())}\n\n")
        f.write("Radii tested R/h_s:\n")
        f.write("  " + ", ".join(str(r) for r in radii) + ", full\n\n")
        f.write("Mean LOOCV errors by model configuration and target:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nInformation-horizon summary, averaged by target and geometry:\n")
        if not horizon.empty:
            f.write(horizon.groupby(["target", "geom"])[["R95_over_hs", "R95_over_lambda_assuming_Kn_based_on_hp", "full_relL2"]].mean().reset_index().to_string(index=False))
        f.write("\n\nInterpretation notes:\n")
        f.write("  R95/h_s is the minimum cumulative bulk-field radius whose error is within 5% of the full-domain model.\n")
        f.write("  R95/lambda assumes the original DSMC Kn is based on protrusion height hp, so lambda/h_s = Kn*(hp/h_s).\n")
        f.write("  This is a predictive-information diagnostic, not a causality proof.\n")

    print("[DONE] outputs under:", out_dir, flush=True)
    print("Key outputs:")
    for name in [
        "final_summary.txt",
        "error_summary_by_config_target.csv",
        "information_horizon_R95.csv",
        "information_horizon_summary.csv",
        "fig_error_vs_radius.png",
        "fig_R95_vs_Kn.png",
        "fig_R95_heatmaps.png",
        "fig_example_surface_profiles.png",
    ]:
        print("  -", out_dir / name)


if __name__ == "__main__":
    main()
