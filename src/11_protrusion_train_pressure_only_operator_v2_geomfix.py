#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_protrusion_train_pressure_only_operator.py

Field-only smooth geometry operator for DSMC protrusion flows.

Why this version:
  The previous unified model produced nonphysical block seams in 2D predicted contours.
  Those seams were caused by hard spatial/zone features in the trunk representation.
  This script removes those hard spatial indicators and trains a FIELD-ONLY operator
  using only smooth geometry distances and Fourier coordinates.

Targets:
  log(P) only, converted to physical P for metrics and plots.

This is a pressure-specific model to improve the P contours after the
multi-output smooth field model still had higher pressure error.

Plotting:
  Uses Tecplot FEQUADRILATERAL cell connectivity and PolyCollection cell coloring.
  No rectangular interpolation, no scatter, no Delaunay.

Example on Vast:
  cd .

  python3 11_protrusion_train_pressure_only_operator.py \
    --audit-dir ./protrusion_01_audit_v2 \
    --out ./protrusion_10_smooth_field_v01 \
    --epochs 350 \
    --points-per-case 18000 \
    --val-points-per-case 26000 \
    --batch-size 16384 \
    --hidden 224 \
    --latent 160 \
    --depth 4 \
    --lr 7e-4 \
    --device auto \
    --make-plots \
    --max-val-plots 15
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath


VAR_RE = re.compile(r'"([^"]+)"')
N_RE = re.compile(r'\bN\s*=\s*(\d+)', re.IGNORECASE)
E_RE = re.compile(r'\bE\s*=\s*(\d+)', re.IGNORECASE)


# -----------------------------
# Tecplot reader
# -----------------------------

def read_text_from_zip(zip_path: str, inner: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read(inner)
    return raw.decode("utf-8", errors="ignore")


def parse_tecplot_fequad(text: str) -> Dict[str, np.ndarray]:
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("Tecplot text too short")

    variables = VAR_RE.findall(lines[0])
    if not variables:
        variables = ["X", "Y", "Z", "u", "v", "n", "rho", "P", "T"]

    header2 = lines[1]
    nm = N_RE.search(header2)
    em = E_RE.search(header2)
    if not nm or not em:
        raise ValueError("Could not parse N/E from Tecplot ZONE line")
    N = int(nm.group(1))
    E = int(em.group(1))

    arr = np.fromstring("\n".join(lines[2:]), sep=" ")
    expected = 3 * N + 6 * E + 4 * E
    if arr.size < expected:
        raise ValueError(f"numeric data too short: got {arr.size}, expected {expected}")

    idx = 0
    Xn = arr[idx:idx+N].astype(np.float64); idx += N
    Yn = arr[idx:idx+N].astype(np.float64); idx += N
    Zn = arr[idx:idx+N].astype(np.float64); idx += N

    out = {}
    for name in variables[3:9]:
        out[name.strip()] = arr[idx:idx+E].astype(np.float64)
        idx += E

    conn = arr[idx:idx+4*E].astype(np.int64).reshape(E, 4) - 1
    conn = np.clip(conn, 0, N-1)

    x = Xn[conn].mean(axis=1)
    y = Yn[conn].mean(axis=1)

    out.update({"Xn": Xn, "Yn": Yn, "Zn": Zn, "conn": conn, "x": x, "y": y})
    return out


def read_field(row: pd.Series) -> Dict[str, np.ndarray]:
    return parse_tecplot_fequad(read_text_from_zip(str(row["zip_path"]), str(row["plt_inner"])))


# -----------------------------
# Dataset manifest and split
# -----------------------------

def load_manifest(audit_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(audit_dir / "manifest_raw.csv")
    geom = pd.read_csv(audit_dir / "geometry_diagnostics.csv")

    # Keep first raw occurrence for each unique physical condition.
    # IMPORTANT: different audit versions do not always repeat hphs/TwTinf
    # in geometry_diagnostics.csv, so merge on physical_key only and then
    # keep the parameter columns from manifest_raw.
    raw_first = raw.drop_duplicates("physical_key", keep="first").copy()
    geom_first = geom.drop_duplicates("physical_key", keep="first").copy()

    m = raw_first.merge(
        geom_first,
        on="physical_key",
        how="left",
        suffixes=("", "_geomdiag"),
    )

    # If merge produced duplicate metadata columns, prefer manifest_raw values.
    for col in ["case_id", "phase", "Ma", "Kn", "geom", "hphs", "TwTinf"]:
        alt = f"{col}_geomdiag"
        if col not in m.columns and alt in m.columns:
            m[col] = m[alt]
        # If both exist, keep m[col] from raw_first.

    # Make sure required columns exist and are numeric where needed.
    if "Tw" not in m.columns:
        m["Tw"] = m["TwTinf"]

    for col in ["Ma", "Kn", "hphs", "TwTinf", "Tw"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    # Fill geometry helper columns if some audit columns are missing.
    if "geom_apex_x" not in m.columns:
        m["geom_apex_x"] = np.nan
    if "geom_apex_y" not in m.columns:
        m["geom_apex_y"] = np.nan
    if "geom_x_min" not in m.columns:
        m["geom_x_min"] = np.nan
    if "geom_x_max" not in m.columns:
        m["geom_x_max"] = np.nan
    if "geom_y_min" not in m.columns:
        m["geom_y_min"] = 0.0
    if "geom_y_max" not in m.columns:
        m["geom_y_max"] = np.nan

    required = ["physical_key", "case_id", "phase", "Ma", "Kn", "geom", "hphs", "TwTinf", "zip_path", "plt_inner"]
    missing = [c for c in required if c not in m.columns]
    if missing:
        raise KeyError(f"Missing required manifest columns after merge: {missing}")

    m = m.sort_values(["phase", "Ma", "Kn", "geom", "hphs", "TwTinf"]).reset_index(drop=True)
    return m


def make_split(m: pd.DataFrame):
    labels = pd.Series("train", index=m.index, dtype=object)

    # Phase 1 holdout: Ma=6, Kn=0.33, baseline height and Tw.
    mask1 = (
        (m["phase"] == "PHASE1") &
        (np.isclose(m["Ma"], 6.0)) &
        (np.isclose(m["Kn"], 0.33)) &
        (np.isclose(m["hphs"], 1.5)) &
        (np.isclose(m["TwTinf"], 1.0))
    )
    labels.loc[mask1] = "val_phase1_Ma6_Kn033"

    # Phase 2 holdout: h/h_s=1.25 at Ma=6,Tw=1 for Kn=0.1,0.33 all geometries.
    mask2 = (
        (m["phase"] == "PHASE2") &
        (np.isclose(m["Ma"], 6.0)) &
        (np.isclose(m["hphs"], 1.25)) &
        (np.isclose(m["TwTinf"], 1.0))
    )
    labels.loc[mask2] = "val_phase2_h125"

    # Phase 3 holdout: Tw=2 at Ma=6,h=1.5 for Kn=0.1,0.33 all geometries.
    mask3 = (
        (m["phase"] == "PHASE3") &
        (np.isclose(m["Ma"], 6.0)) &
        (np.isclose(m["hphs"], 1.5)) &
        (np.isclose(m["TwTinf"], 2.0))
    )
    labels.loc[mask3] = "val_phase3_Tw2"

    train = labels == "train"
    val = ~train
    return train.to_numpy(), val.to_numpy(), labels


# -----------------------------
# Geometry and smooth features
# -----------------------------

def triangle_vertices(row: pd.Series) -> np.ndarray:
    # Correct DSMC protrusion vertices from surface geometry diagnostics.
    # base_left/base_right are the wall-foot endpoints. For FWD/BWD the apex
    # lies outside this base interval; using geom_x_min/geom_x_max as the base
    # creates a false right triangle.
    x_min = float(row.get("geom_x_min", np.nan))
    x_max = float(row.get("geom_x_max", np.nan))
    y_min = float(row.get("geom_y_min", 0.0))
    ax = float(row.get("geom_apex_x", np.nan))
    ay = float(row.get("geom_apex_y", np.nan))
    bl = float(row.get("geom_base_left_x", np.nan))
    br = float(row.get("geom_base_right_x", np.nan))

    if not np.isfinite(bl) or not np.isfinite(br):
        bl, br = x_min, x_max

    if not np.isfinite(ax) or not np.isfinite(ay) or not np.isfinite(bl) or not np.isfinite(br):
        # fallback with hs=0.02, hphs
        hs = 0.02
        hp = float(row["hphs"]) * hs
        geom = str(row["geom"]).upper()
        y_min = 0.0
        ay = hp
        if geom == "BWD":
            bl = 0.22; br = 0.24; ax = br + (hp - (br - bl))  # fallback only
        elif geom == "FWD":
            bl = 0.22; br = 0.24; ax = bl - (hp - (br - bl))  # fallback only
        else:
            bl = 0.22; br = 0.24; ax = 0.5 * (bl + br)
    return np.array([[bl, y_min], [br, y_min], [ax, ay]], dtype=np.float64)


def geom_scales(row: pd.Series):
    verts = triangle_vertices(row)
    yb = float(np.nanmin(verts[:, 1]))
    hp = float(np.nanmax(verts[:, 1]) - yb)
    hphs = float(row["hphs"])
    hs = hp / max(hphs, 1e-12)
    apex_idx = int(np.nanargmax(verts[:, 1]))
    apex = verts[apex_idx]
    return verts, apex, hs, hp, yb


def point_segment_dist(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    c2 = vx * vx + vy * vy + 1e-30
    t = np.clip((wx * vx + wy * vy) / c2, 0.0, 1.0)
    qx = ax + t * vx
    qy = ay + t * vy
    return np.hypot(px - qx, py - qy)


def inside_triangle(x, y, verts):
    path = MplPath(np.vstack([verts, verts[0]]))
    return path.contains_points(np.column_stack([x, y]))


def smooth_point_features(x: np.ndarray, y: np.ndarray, row: pd.Series) -> np.ndarray:
    verts, apex, hs, hp, yb = geom_scales(row)
    hs = max(hs, 1e-12)
    xn = (x - apex[0]) / hs
    yn = (y - yb) / hs

    # Distances normalized by hs, all continuous.
    d_wall = np.maximum(yn, 0.0)
    d_apex = np.hypot(x - apex[0], y - apex[1]) / hs

    # Distances to triangle edges.
    d_edges = []
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        d = point_segment_dist(x, y, verts[a, 0], verts[a, 1], verts[b, 0], verts[b, 1]) / hs
        d_edges.append(d)
    d0, d1, d2 = d_edges
    d_solid = np.minimum(np.minimum(d0, d1), d2)

    # Smooth compressed distances.
    def log1p_clip(z):
        return np.log1p(np.clip(z, 0, 20))

    # Fourier coordinates: smooth, no hard zones.
    freqs = [0.5, 1.0, 2.0]
    feats = [
        xn, yn,
        log1p_clip(np.abs(xn)), log1p_clip(np.abs(yn)),
        d_wall, d_solid, d0, d1, d2, d_apex,
        np.exp(-0.5 * (d_apex / 0.75)**2),
        np.exp(-0.5 * (d_solid / 0.50)**2),
        np.exp(-0.5 * (d_wall / 0.75)**2),
    ]
    for f in freqs:
        feats.extend([
            np.sin(np.pi * f * xn), np.cos(np.pi * f * xn),
            np.sin(np.pi * f * yn), np.cos(np.pi * f * yn),
        ])
    return np.column_stack(feats).astype(np.float32)


def case_features(row: pd.Series) -> np.ndarray:
    geom = str(row["geom"]).upper()
    g = np.array([geom == "BWD", geom == "FWD", geom == "ISO"], dtype=np.float32)
    return np.r_[
        float(row["Ma"]),
        np.log10(float(row["Kn"])),
        float(row["hphs"]),
        float(row["TwTinf"]),
        g
    ].astype(np.float32)


def target_matrix(field: Dict[str, np.ndarray]) -> np.ndarray:
    """Pressure-only transformed target: log(P)."""
    P = np.maximum(np.asarray(field["P"], dtype=np.float64), 1e-30)
    return np.log(P).reshape(-1, 1).astype(np.float32)


def inverse_target_matrix(Y: np.ndarray) -> np.ndarray:
    """Convert transformed target [logP] back to physical [P]."""
    Y = np.asarray(Y, dtype=np.float64)
    out = np.exp(np.clip(Y[:, 0], -80.0, 80.0)).reshape(-1, 1)
    return out.astype(np.float32)


def pressure_sample_weights(field: Dict[str, np.ndarray], row: pd.Series, idx: np.ndarray, args) -> np.ndarray:
    """Point weights for pressure: emphasize compression/high-P, near-wall, and apex regions."""
    P = np.asarray(field["P"], dtype=np.float64)[idx]
    x = np.asarray(field["x"], dtype=np.float64)[idx]
    y = np.asarray(field["y"], dtype=np.float64)[idx]
    verts, apex, hs, hp, yb = geom_scales(row)
    hs = max(hs, 1e-12)

    p95 = np.nanpercentile(P[np.isfinite(P)], 95) if np.isfinite(P).any() else 1.0
    pnorm = np.clip(P / max(p95, 1e-30), 0.0, 3.0)
    w = np.ones_like(P, dtype=np.float64)
    w += float(args.pressure_peak_weight) * pnorm**2

    # distance to solid boundary
    dists = []
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        dists.append(point_segment_dist(x, y, verts[a, 0], verts[a, 1], verts[b, 0], verts[b, 1]) / hs)
    dsolid = np.minimum(np.minimum(dists[0], dists[1]), dists[2])
    dapex = np.hypot(x - apex[0], y - apex[1]) / hs
    dwall = np.maximum((y - yb) / hs, 0.0)

    sig_s = max(float(args.pressure_wall_sigma), 1e-6)
    sig_a = max(float(args.pressure_apex_sigma), 1e-6)
    w += float(args.pressure_wall_weight) * np.exp(-0.5 * (dsolid / sig_s)**2)
    w += 0.5 * float(args.pressure_wall_weight) * np.exp(-0.5 * (dwall / sig_s)**2)
    w += float(args.pressure_apex_weight) * np.exp(-0.5 * (dapex / sig_a)**2)
    w = np.clip(w, 1.0, float(args.pressure_weight_clip))
    return w.reshape(-1, 1).astype(np.float32)


def gas_mask(field, row):
    x = field["x"]; y = field["y"]
    verts, apex, hs, hp, yb = geom_scales(row)
    ins = inside_triangle(x, y, verts)
    Y = target_matrix(field)
    return (
        np.isfinite(x) & np.isfinite(y) &
        (~ins) &
        (y >= yb - 1e-12) &
        np.all(np.isfinite(Y), axis=1)
    )


# -----------------------------
# Normalization
# -----------------------------

def mean_std(x, eps=1e-12):
    mu = np.nanmean(x, axis=0).astype(np.float32)
    sd = np.nanstd(x, axis=0).astype(np.float32)
    sd = np.where(sd < eps, 1.0, sd).astype(np.float32)
    return mu, sd


def zscore(x, mu, sd):
    return ((x - mu) / sd).astype(np.float32)


def unzscore(x, mu, sd):
    return (x * sd + mu).astype(np.float32)


# -----------------------------
# Model
# -----------------------------

class MLP(nn.Module):
    def __init__(self, din, dout, hidden=192, depth=3, dropout=0.0):
        super().__init__()
        layers = []
        d = din
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, dout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmoothFieldOperator(nn.Module):
    def __init__(self, case_dim, point_dim, out_dim=1, latent=160, hidden=224, depth=4, dropout=0.03):
        super().__init__()
        self.branch = MLP(case_dim, latent, hidden, depth, dropout)
        self.trunk = MLP(point_dim, latent, hidden, depth, dropout)
        self.decoder = MLP(latent, out_dim, hidden, max(2, depth-1), dropout)

    def forward(self, c, p):
        z = self.branch(c) * self.trunk(p)
        return self.decoder(z)


# -----------------------------
# Dataset
# -----------------------------

def build_dataset(m, train_mask, val_mask, args):
    rng = np.random.default_rng(args.seed)
    Xc_tr, Xp_tr, Y_tr, W_tr = [], [], [], []
    Xc_va, Xp_va, Y_va, W_va, case_ids_va, split_va = [], [], [], [], [], []

    fields_cache = {}
    for i, row in m.iterrows():
        field = read_field(row)
        fields_cache[i] = field
        mask = gas_mask(field, row)
        idx_all = np.flatnonzero(mask)
        if len(idx_all) == 0:
            print("[WARN] no valid cells:", row["case_id"], flush=True)
            continue
        nsel = args.points_per_case if train_mask[i] else args.val_points_per_case
        replace = len(idx_all) < nsel
        idx = rng.choice(idx_all, size=nsel, replace=replace)
        x = field["x"][idx]
        y = field["y"][idx]
        xp = smooth_point_features(x, y, row)
        cf = np.repeat(case_features(row)[None, :], len(idx), axis=0)
        yy = target_matrix(field)[idx]
        ww = pressure_sample_weights(field, row, idx, args)

        if train_mask[i]:
            Xc_tr.append(cf); Xp_tr.append(xp); Y_tr.append(yy); W_tr.append(ww)
        elif val_mask[i]:
            Xc_va.append(cf); Xp_va.append(xp); Y_va.append(yy); W_va.append(ww)
            case_ids_va.extend([i] * len(idx))
            split_va.extend([row["split_label"]] * len(idx))

    Xc_tr = np.vstack(Xc_tr); Xp_tr = np.vstack(Xp_tr); Y_tr = np.vstack(Y_tr); W_tr = np.vstack(W_tr)
    Xc_va = np.vstack(Xc_va); Xp_va = np.vstack(Xp_va); Y_va = np.vstack(Y_va); W_va = np.vstack(W_va)
    case_ids_va = np.array(case_ids_va, dtype=np.int64)
    split_va = np.array(split_va, dtype=object)

    cmu, csd = mean_std(Xc_tr)
    pmu, psd = mean_std(Xp_tr)
    ymu, ysd = mean_std(Y_tr)

    data = dict(
        Xc_tr=zscore(Xc_tr, cmu, csd), Xp_tr=zscore(Xp_tr, pmu, psd), Y_tr=zscore(Y_tr, ymu, ysd), W_tr=W_tr,
        Xc_va=zscore(Xc_va, cmu, csd), Xp_va=zscore(Xp_va, pmu, psd), Y_va=zscore(Y_va, ymu, ysd), W_va=W_va,
        Y_va_raw=Y_va,
        case_ids_va=case_ids_va,
        split_va=split_va,
        cmu=cmu, csd=csd, pmu=pmu, psd=psd, ymu=ymu, ysd=ysd,
        fields_cache=fields_cache,
    )
    return data


def rel_l2(pred, true, eps=1e-12):
    m = np.isfinite(pred) & np.isfinite(true)
    if not m.any(): return np.nan
    return float(np.linalg.norm(pred[m] - true[m]) / (np.linalg.norm(true[m]) + eps))


def train_model(data, args, out_dir):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] device =", device, flush=True)

    train_ds = TensorDataset(
        torch.from_numpy(data["Xc_tr"]),
        torch.from_numpy(data["Xp_tr"]),
        torch.from_numpy(data["Y_tr"]),
        torch.from_numpy(data["W_tr"]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(data["Xc_va"]),
        torch.from_numpy(data["Xp_va"]),
        torch.from_numpy(data["Y_va"]),
        torch.from_numpy(data["W_va"]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SmoothFieldOperator(
        case_dim=data["Xc_tr"].shape[1],
        point_dim=data["Xp_tr"].shape[1],
        latent=args.latent,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=25, min_lr=1e-6)

    best = np.inf
    patience = 0
    best_path = out_dir / "best_smooth_field.pt"
    logs = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = []
        for cx, px, yy, ww in train_loader:
            cx = cx.to(device); px = px.to(device); yy = yy.to(device); ww = ww.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(cx, px)
            loss_elem = F.huber_loss(pred, yy, delta=args.huber_delta, reduction="none")
            loss = (loss_elem * ww).mean()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tl.append(float(loss.detach().cpu()))
        model.eval()
        vl = []
        with torch.no_grad():
            for cx, px, yy, ww in val_loader:
                cx = cx.to(device); px = px.to(device); yy = yy.to(device); ww = ww.to(device)
                pred = model(cx, px)
                loss_elem = F.huber_loss(pred, yy, delta=args.huber_delta, reduction="none")
                vl.append(float((loss_elem * ww).mean().cpu()))
        tr = float(np.mean(tl)); va = float(np.mean(vl))
        sched.step(va)
        lr = opt.param_groups[0]["lr"]
        logs.append({"epoch": ep, "train_loss": tr, "val_loss": va, "lr": lr})
        if va < best:
            best = va
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "normalizers": {
                    "cmu": data["cmu"], "csd": data["csd"],
                    "pmu": data["pmu"], "psd": data["psd"],
                    "ymu": data["ymu"], "ysd": data["ysd"],
                },
            }, best_path)
        else:
            patience += 1
        if ep == 1 or ep % args.print_every == 0:
            print(f"[INFO] epoch {ep:04d} train={tr:.5e} val={va:.5e} lr={lr:.2e}", flush=True)
        if patience >= args.early_patience:
            print(f"[INFO] early stopping at epoch {ep}", flush=True)
            break
    pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def predict_points(model, device, Xc, Xp, data, batch=65536):
    Xc_z = zscore(Xc.astype(np.float32), data["cmu"], data["csd"])
    Xp_z = zscore(Xp.astype(np.float32), data["pmu"], data["psd"])
    out = []
    with torch.no_grad():
        for s in range(0, len(Xc_z), batch):
            e = min(len(Xc_z), s + batch)
            cx = torch.from_numpy(Xc_z[s:e]).to(device)
            px = torch.from_numpy(Xp_z[s:e]).to(device)
            out.append(model(cx, px).cpu().numpy())
    return unzscore(np.vstack(out), data["ymu"], data["ysd"])


# -----------------------------
# Cell polygon plots
# -----------------------------

def normalized_polygons(field, row, idx):
    verts, apex, hs, hp, yb = geom_scales(row)
    Xn = field["Xn"]; Yn = field["Yn"]; conn = field["conn"][idx]
    xv = (Xn[conn] - apex[0]) / hs
    yv = (Yn[conn] - yb) / hs
    return np.stack([xv, yv], axis=2)


def draw_triangle(ax, row):
    verts, apex, hs, hp, yb = geom_scales(row)
    vv = verts.copy()
    vv[:, 0] = (vv[:, 0] - apex[0]) / hs
    vv[:, 1] = (vv[:, 1] - yb) / hs
    ax.add_patch(Polygon(vv, closed=True, facecolor="white", edgecolor="black", lw=1.2, zorder=20))


def polygon_quality(polys, max_edge, max_area):
    x = polys[:, :, 0]; y = polys[:, :, 1]
    emax = np.zeros(len(polys))
    for a, b in [(0,1),(1,2),(2,3),(3,0)]:
        emax = np.maximum(emax, np.hypot(x[:,a]-x[:,b], y[:,a]-y[:,b]))
    area = 0.5*np.abs(np.sum(x*np.roll(y,-1,axis=1)-y*np.roll(x,-1,axis=1), axis=1))
    return np.isfinite(emax) & (emax <= max_edge) & np.isfinite(area) & (area > 1e-12) & (area <= max_area)


def robust_range(v, lo=2, hi=98):
    v = np.asarray(v); v = v[np.isfinite(v)]
    if len(v)==0: return 0.0, 1.0
    a,b=np.nanpercentile(v,[lo,hi])
    if abs(b-a)<1e-30: b=a+1.0
    return float(a), float(b)


def signed_error(pred, true, kind):
    err=pred-true
    if kind=="signed_range_percent":
        den=max(robust_range(true,1,99)[1]-robust_range(true,1,99)[0],1e-30)
        return 100*err/den, "Error (% robust range)", r"$100(\hat q-q)/(q_{99}-q_1)$ (%)"
    elif kind=="safe_relative_percent":
        _, q95 = robust_range(np.abs(true), 0, 95)
        den=np.maximum(np.abs(true),0.05*max(q95,1.0))
        return 100*err/den, "Safe relative error (%)", "Safe relative error (%)"
    else:
        return err, "Error", r"$\hat q-q$"


def poly_panel(ax, polys, vals, title, cmap, vmin, vmax):
    pc=PolyCollection(polys,array=vals,cmap=cmap,edgecolors="none",linewidths=0,antialiaseds=False,rasterized=True)
    pc.set_clim(vmin,vmax)
    ax.add_collection(pc)
    ax.set_title(title)
    return pc


def make_plots(m, train_mask, val_mask, labels, model, device, data, args, out_dir):
    plot_dir=out_dir/"validation_cell_polygons"
    plot_dir.mkdir(parents=True, exist_ok=True)
    val_indices=np.flatnonzero(val_mask)
    chosen=[]
    for lab in sorted(pd.Series(labels[val_mask]).unique()):
        idxs=[i for i in val_indices if labels[i]==lab]
        chosen.extend(idxs[:max(1,args.max_val_plots//3)])
    chosen=chosen[:args.max_val_plots]

    metrics=[]
    names=["P"]
    for i in chosen:
        row=m.iloc[i]
        print("[INFO] plotting",row["case_id"],flush=True)
        field=read_field(row)
        mask=gas_mask(field,row)
        idx=np.flatnonzero(mask)
        x=field["x"][idx]; y=field["y"][idx]
        Xp=smooth_point_features(x,y,row)
        Xc=np.repeat(case_features(row)[None,:],len(idx),axis=0)
        pred_trans=predict_points(model,device,Xc,Xp,data,batch=args.predict_batch_size)
        true_trans=target_matrix(field)[idx]
        pred=inverse_target_matrix(pred_trans)
        true=inverse_target_matrix(true_trans)

        verts,apex,hs,hp,yb=geom_scales(row)
        xloc=(x-apex[0])/hs; yloc=(y-yb)/hs
        dists=[]
        for a,b in [(0,1),(1,2),(2,0)]:
            dists.append(point_segment_dist(x,y,verts[a,0],verts[a,1],verts[b,0],verts[b,1])/hs)
        dist=np.minimum(np.minimum(dists[0],dists[1]),dists[2])
        common=(xloc>=args.xmin)&(xloc<=args.xmax)&(yloc>=args.ymin)&(yloc<=args.ymax)&(yloc>=args.wall_buffer)&(dist>=args.solid_buffer)
        polys_all=normalized_polygons(field,row,idx)
        common &= polygon_quality(polys_all,args.max_edge,args.max_area)
        for j,nm in enumerate(names):
            metrics.append({"case_id":row["case_id"],"split_label":labels[i],"variable":nm,"relL2":rel_l2(pred[common,j],true[common,j])})
            if nm not in args.plot_variables:
                continue
            qtrue=true[common,j]; qpred=pred[common,j]; polys=polys_all[common]
            err,etitle,elabel=signed_error(qpred,qtrue,args.error_kind)
            vmin,vmax=robust_range(np.r_[qtrue,qpred],2,98)
            emax=min(args.error_cap,float(np.nanpercentile(np.abs(err[np.isfinite(err)]),args.error_percentile)))
            emax=max(emax,1e-12)
            fig,axes=plt.subplots(1,3,figsize=(12,3.25),sharex=True,sharey=True)
            panels=[(qtrue,f"DSMC {nm}","viridis",vmin,vmax,nm),(qpred,f"Predicted {nm}","viridis",vmin,vmax,nm),(err,etitle,"coolwarm",-emax,emax,elabel)]
            for ax,(vals,title,cmap,a,b,cblab) in zip(axes,panels):
                pc=poly_panel(ax,polys,vals,title,cmap,a,b)
                draw_triangle(ax,row)
                ax.set_xlim(args.xmin,args.xmax); ax.set_ylim(args.ymin,args.ymax)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(r"$(x-x_{apex})/h_s$")
                cb=fig.colorbar(pc,ax=ax,shrink=0.82); cb.set_label(cblab,fontsize=8)
            axes[0].set_ylabel(r"$(y-y_b)/h_s$")
            fig.suptitle(f"Smooth field operator cell-polygon validation: {row['case_id']} ({labels[i]})",fontsize=10)
            fig.tight_layout()
            fig.savefig(plot_dir/f"smooth_cellpoly_{nm}_{row['case_id']}.png",dpi=args.dpi)
            plt.close(fig)
    met=pd.DataFrame(metrics)
    met.to_csv(out_dir/"validation_field_metrics_by_case.csv",index=False)
    if not met.empty:
        summ=met.groupby(["split_label","variable"])["relL2"].mean().unstack()
        summ["mean_P"]=summ[["P"]].mean(axis=1)
        summ.to_csv(out_dir/"validation_field_metrics_summary.csv")
        print("[VALIDATION SUMMARY]")
        print(summ.to_string())


def plot_training(out_dir):
    p=out_dir/"training_log.csv"
    if p.exists():
        df=pd.read_csv(p)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.semilogy(df["epoch"],df["train_loss"],label="train")
        ax.semilogy(df["epoch"],df["val_loss"],label="val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Huber loss")
        ax.grid(True,alpha=0.25); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir/"fig_training_loss.png",dpi=220); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit-dir",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=350)
    ap.add_argument("--points-per-case",type=int,default=18000)
    ap.add_argument("--val-points-per-case",type=int,default=26000)
    ap.add_argument("--batch-size",type=int,default=16384)
    ap.add_argument("--hidden",type=int,default=224)
    ap.add_argument("--latent",type=int,default=160)
    ap.add_argument("--depth",type=int,default=4)
    ap.add_argument("--dropout",type=float,default=0.03)
    ap.add_argument("--lr",type=float,default=7e-4)
    ap.add_argument("--weight-decay",type=float,default=1e-5)
    ap.add_argument("--huber-delta",type=float,default=1.0)
    ap.add_argument("--pressure-peak-weight", type=float, default=2.0)
    ap.add_argument("--pressure-wall-weight", type=float, default=1.0)
    ap.add_argument("--pressure-apex-weight", type=float, default=1.5)
    ap.add_argument("--pressure-wall-sigma", type=float, default=0.35)
    ap.add_argument("--pressure-apex-sigma", type=float, default=0.75)
    ap.add_argument("--pressure-weight-clip", type=float, default=10.0)
    ap.add_argument("--grad-clip",type=float,default=1.0)
    ap.add_argument("--early-patience",type=int,default=100)
    ap.add_argument("--print-every",type=int,default=10)
    ap.add_argument("--device",default="auto")
    ap.add_argument("--seed",type=int,default=1234)
    ap.add_argument("--make-plots",action="store_true")
    ap.add_argument("--max-val-plots",type=int,default=15)
    ap.add_argument("--plot-variables",nargs="+",default=["P"])
    ap.add_argument("--predict-batch-size",type=int,default=65536)
    ap.add_argument("--error-kind",default="signed_range_percent",choices=["signed_range_percent","safe_relative_percent","absolute"])
    ap.add_argument("--error-percentile",type=float,default=98)
    ap.add_argument("--error-cap",type=float,default=25)
    ap.add_argument("--solid-buffer",type=float,default=0.04)
    ap.add_argument("--wall-buffer",type=float,default=0.04)
    ap.add_argument("--max-edge",type=float,default=0.35)
    ap.add_argument("--max-area",type=float,default=0.05)
    ap.add_argument("--xmin",type=float,default=-4.5)
    ap.add_argument("--xmax",type=float,default=4.5)
    ap.add_argument("--ymin",type=float,default=0)
    ap.add_argument("--ymax",type=float,default=5)
    ap.add_argument("--dpi",type=int,default=260)
    args=ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir=Path(args.out); out_dir.mkdir(parents=True,exist_ok=True)
    m=load_manifest(Path(args.audit_dir))
    train_mask,val_mask,labels=make_split(m)
    m["split_label"]=labels
    print(f"[INFO] unique cases={len(m)} train={train_mask.sum()} val={val_mask.sum()}",flush=True)
    data=build_dataset(m,train_mask,val_mask,args)
    print(f"[INFO] train points={len(data['Y_tr'])} val points={len(data['Y_va'])}",flush=True)
    model,device=train_model(data,args,out_dir)
    if args.make_plots:
        make_plots(m,train_mask,val_mask,labels.to_numpy(),model,device,data,args,out_dir)
        plot_training(out_dir)
    with open(out_dir/"final_summary.txt","w") as f:
        f.write("Pressure-only smooth protrusion operator\n")
        f.write("Training target: log(P) only; reported/plots use physical P=exp(logP). High-pressure/near-wall/apex weighted loss.\n")
        f.write(f"Train cases: {train_mask.sum()} Validation cases: {val_mask.sum()}\n")
        f.write(f"Train points: {len(data['Y_tr'])} Validation points: {len(data['Y_va'])}\n")
    print("[DONE] outputs under:",out_dir)


if __name__=="__main__":
    main()
