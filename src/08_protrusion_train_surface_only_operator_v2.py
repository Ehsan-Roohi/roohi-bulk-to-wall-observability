#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_protrusion_train_surface_only_operator.py

Surface-only face-split operator for protrusion-wall properties:
  Cp(s), Cq(s), |tau|(s)

This script intentionally separates the surface model from the full-field model.
Reason: the field has ~5e5 training points, while the protrusion surface has only
~2520 training points; a unified loss tends to compromise and smooth surface peaks.

Requires 06_protrusion_train_unified_operator_v7.py for dataset readers/features.

Example on Vast:
  cd .
  python3 08_protrusion_train_surface_only_operator.py \
    --audit-dir ./protrusion_01_audit_v2 \
    --out ./protrusion_08_surface_only_v01 \
    --base-script ./06_protrusion_train_unified_operator_v7.py \
    --epochs 1500 --hidden 192 --latent 128 --lr 8e-4 --make-plots
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def import_base(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Base script not found: {p}")
    spec = importlib.util.spec_from_file_location("prot_base", str(p))
    mod = importlib.util.module_from_spec(spec)
    # Required for @dataclass in dynamically imported modules on Python 3.12+
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=3, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SurfaceOnlyOperator(nn.Module):
    def __init__(self, case_dim, surf_dim, latent=128, hidden=192, depth=3, dropout=0.03, full_residual_scale=0.05):
        super().__init__()
        self.full_residual_scale = float(full_residual_scale)
        self.base_branch = MLP(5, latent, hidden, depth, dropout)
        self.h_branch = MLP(4, latent, hidden, depth, dropout)
        self.tw_branch = MLP(4, latent, hidden, depth, dropout)
        self.full_branch = MLP(case_dim, latent, hidden, max(1, depth-1), dropout)

        self.trunk = MLP(surf_dim, latent, hidden, depth, dropout)
        self.wind_decoder = MLP(latent, 3, hidden, depth, dropout)
        self.lee_decoder = MLP(latent, 3, hidden, depth, dropout)
        self.gate = MLP(surf_dim, 1, max(32, hidden // 2), 2, 0.0)

    def branch_embedding(self, case_x):
        # same convention as v7 after normalization
        Ma_log_geom = torch.cat([case_x[:, 0:2], case_x[:, 4:7]], dim=1)
        log_geom = torch.cat([case_x[:, 1:2], case_x[:, 4:7]], dim=1)
        h_delta = case_x[:, 2:3]
        tw_delta = case_x[:, 3:4]
        z = self.base_branch(Ma_log_geom)
        z = z + h_delta * self.h_branch(log_geom)
        z = z + tw_delta * self.tw_branch(log_geom)
        if self.full_residual_scale > 0:
            z = z + self.full_residual_scale * self.full_branch(case_x)
        return z

    def forward(self, case_x, surf_x):
        b = self.branch_embedding(case_x)
        t = self.trunk(surf_x)
        z = b * t
        pw = self.wind_decoder(z)
        pl = self.lee_decoder(z)
        g = torch.sigmoid(self.gate(surf_x))
        return g * pw + (1.0 - g) * pl


def finite_mean_std(x, eps=1e-12):
    mu = np.nanmean(x, axis=0).astype(np.float32)
    sd = np.nanstd(x, axis=0).astype(np.float32)
    sd = np.where(sd < eps, 1.0, sd).astype(np.float32)
    return mu, sd


def zscore(x, mu, sd):
    return ((x - mu) / sd).astype(np.float32)


def unzscore(x, mu, sd):
    return x * sd + mu


def rel_l2(pred, true, eps=1e-12):
    m = np.isfinite(pred) & np.isfinite(true)
    if not m.any():
        return np.nan
    return float(np.linalg.norm(pred[m] - true[m]) / (np.linalg.norm(true[m]) + eps))


def make_peak_weights(surf: pd.DataFrame, y_raw: np.ndarray, base, args):
    # Use base v7 weighting, then add channel-specific robust high-weight near peaks.
    dummy = SimpleNamespace(
        surface_apex_weight=args.apex_weight,
        surface_peak_weight=args.peak_weight,
        surface_peak_sigma=args.peak_sigma,
        surface_weight_clip=args.weight_clip,
    )
    w = base.compute_surface_weights(surf, y_raw, dummy).reshape(-1)
    s = surf["s01"].to_numpy(float)
    for j in range(y_raw.shape[1]):
        v = y_raw[:, j]
        if np.isfinite(v).any():
            im = int(np.nanargmax(np.abs(v)))
            sp = s[im]
            w += args.peak_weight * np.exp(-0.5 * ((s - sp) / max(args.peak_sigma, 1e-6)) ** 2)
    return np.clip(w, 1.0, args.weight_clip).astype(np.float32)[:, None]


def load_surface_dataset(base, audit_dir: Path, args):
    m = base.load_unique_manifest(audit_dir)
    m = base.merge_diagnostics(m, audit_dir)
    if "Tw" not in m.columns:
        m["Tw"] = m["TwTinf"]
    train_mask, val_mask, split_label = base.make_split(m, args.split)

    Xc, Xs, Y, W, case_idx, is_val, rows = [], [], [], [], [], [], []
    for i, row in m.iterrows():
        if not (train_mask[i] or val_mask[i]):
            continue
        try:
            surf = base.read_prot_surface(str(row["zip_path"]), str(row["xlsx_inner"]))
        except Exception as exc:
            print("[WARN] skip surface", row["case_id"], exc, flush=True)
            continue
        if surf is None or surf.empty:
            continue
        geom = base.geometry_from_row(row)
        sx = base.build_surface_point_features(surf, row=row, geom=geom)
        y = surf[["Cp", "Cq", "tau_abs"]].to_numpy(dtype=np.float32)
        ok = np.all(np.isfinite(sx), axis=1) & np.all(np.isfinite(y), axis=1)
        if ok.sum() < 5:
            continue
        sx = sx[ok]
        y = y[ok]
        surf_ok = surf.loc[ok].reset_index(drop=True)
        w = make_peak_weights(surf_ok, y, base, args)
        cf = np.repeat(base.case_features(row)[None, :], len(y), axis=0)

        Xc.append(cf.astype(np.float32))
        Xs.append(sx.astype(np.float32))
        Y.append(y.astype(np.float32))
        W.append(w.astype(np.float32))
        case_idx.append(np.full(len(y), i, dtype=np.int64))
        is_val.append(np.full(len(y), bool(val_mask[i]), dtype=bool))
        rows.append(row)

    Xc = np.vstack(Xc)
    Xs = np.vstack(Xs)
    Y = np.vstack(Y)
    W = np.vstack(W)
    case_idx = np.concatenate(case_idx)
    is_val = np.concatenate(is_val)

    tr = ~is_val
    va = is_val
    case_mu, case_sd = finite_mean_std(Xc[tr])
    surf_mu, surf_sd = finite_mean_std(Xs[tr])
    y_mu, y_sd = finite_mean_std(Y[tr])

    data = {
        "m": m,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "split_label": split_label,
        "Xc": zscore(Xc, case_mu, case_sd),
        "Xs": zscore(Xs, surf_mu, surf_sd),
        "Y": zscore(Y, y_mu, y_sd),
        "Y_raw": Y,
        "W": W,
        "case_idx": case_idx,
        "is_val": is_val,
        "case_mu": case_mu,
        "case_sd": case_sd,
        "surf_mu": surf_mu,
        "surf_sd": surf_sd,
        "y_mu": y_mu,
        "y_sd": y_sd,
    }
    return data


def train_model(data, args, out_dir: Path):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] device =", device, flush=True)

    tr = ~data["is_val"]
    va = data["is_val"]
    train_ds = TensorDataset(
        torch.from_numpy(data["Xc"][tr]),
        torch.from_numpy(data["Xs"][tr]),
        torch.from_numpy(data["Y"][tr]),
        torch.from_numpy(data["W"][tr]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(data["Xc"][va]),
        torch.from_numpy(data["Xs"][va]),
        torch.from_numpy(data["Y"][va]),
        torch.from_numpy(data["W"][va]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SurfaceOnlyOperator(
        case_dim=data["Xc"].shape[1],
        surf_dim=data["Xs"].shape[1],
        latent=args.latent,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        full_residual_scale=args.full_residual_scale,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=35, min_lr=1e-6)

    best = np.inf
    best_path = out_dir / "best_surface_only.pt"
    patience = 0
    logs = []
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for cx, sx, yy, ww in train_loader:
            cx = cx.to(device); sx = sx.to(device); yy = yy.to(device); ww = ww.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(cx, sx)
            loss_elem = F.huber_loss(pred, yy, delta=args.huber_delta, reduction="none").mean(dim=1, keepdim=True)
            loss = (loss_elem * ww).mean()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vlosses = []
        with torch.no_grad():
            for cx, sx, yy, ww in val_loader:
                cx = cx.to(device); sx = sx.to(device); yy = yy.to(device); ww = ww.to(device)
                pred = model(cx, sx)
                loss_elem = F.huber_loss(pred, yy, delta=args.huber_delta, reduction="none").mean(dim=1, keepdim=True)
                vloss = (loss_elem * ww).mean()
                vlosses.append(float(vloss.cpu()))
        tr_loss = float(np.mean(losses))
        va_loss = float(np.mean(vlosses))
        sched.step(va_loss)
        lr = opt.param_groups[0]["lr"]
        logs.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "lr": lr})
        if va_loss < best:
            best = va_loss
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "normalizers": {
                    "case_mu": data["case_mu"], "case_sd": data["case_sd"],
                    "surf_mu": data["surf_mu"], "surf_sd": data["surf_sd"],
                    "y_mu": data["y_mu"], "y_sd": data["y_sd"],
                },
            }, best_path)
        else:
            patience += 1
        if ep == 1 or ep % args.print_every == 0:
            print(f"[INFO] epoch {ep:04d} train={tr_loss:.5e} val={va_loss:.5e} lr={lr:.2e}", flush=True)
        if patience >= args.early_patience:
            print(f"[INFO] early stopping at epoch {ep}", flush=True)
            break

    pd.DataFrame(logs).to_csv(out_dir / "surface_only_training_log.csv", index=False)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def predict_all(model, data, device, args):
    pred = []
    with torch.no_grad():
        for start in range(0, len(data["Y"]), args.predict_batch_size):
            end = min(len(data["Y"]), start + args.predict_batch_size)
            cx = torch.from_numpy(data["Xc"][start:end]).to(device)
            sx = torch.from_numpy(data["Xs"][start:end]).to(device)
            pred.append(model(cx, sx).cpu().numpy())
    pred = unzscore(np.vstack(pred), data["y_mu"], data["y_sd"])
    return pred


def compute_metrics_and_plots(base, model, data, device, args, out_dir: Path):
    pred = predict_all(model, data, device, args)
    true = data["Y_raw"]
    names = ["Cp", "Cq", "tau_abs"]
    recs = []
    m = data["m"]
    split_label = data["split_label"]
    for ci in np.unique(data["case_idx"][data["is_val"]]):
        mask = (data["case_idx"] == ci) & data["is_val"]
        row = m.iloc[int(ci)]
        rec = {
            "case_id": row["case_id"],
            "split_label": split_label.iloc[int(ci)],
            "phase": row["phase"], "Ma": row["Ma"], "Kn": row["Kn"], "geom": row["geom"],
            "hphs": row["hphs"], "TwTinf": row["TwTinf"],
            "n_points": int(mask.sum()),
        }
        for j, n in enumerate(names):
            rec[f"relL2_{n}"] = rel_l2(pred[mask, j], true[mask, j])
            # peak magnitude and location from actual surface coordinate
            try:
                surf = base.read_prot_surface(str(row["zip_path"]), str(row["xlsx_inner"]))
                yy = surf[n].to_numpy(float)
                ss = surf["s01"].to_numpy(float)
                valid = np.isfinite(yy) & np.isfinite(ss)
                if valid.any():
                    im_true = int(np.nanargmax(np.abs(yy[valid])))
                    # align with mask order, using raw arrays length should match
                    yy_pred = pred[mask, j]
                    im_pred = int(np.nanargmax(np.abs(yy_pred)))
                    vt = yy[valid][im_true]
                    vp = yy_pred[im_pred]
                    st = ss[valid][im_true]
                    sp = ss[valid][im_pred] if im_pred < valid.sum() else np.nan
                    rec[f"peak_relerr_{n}"] = float(abs(vp - vt) / (abs(vt) + 1e-12))
                    rec[f"peak_s_error_{n}"] = float(abs(sp - st)) if np.isfinite(sp) else np.nan
            except Exception:
                pass
        rec["relL2_mean_surface"] = float(np.nanmean([rec[f"relL2_{n}"] for n in names]))
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out_dir / "surface_only_validation_metrics_by_case.csv", index=False)
    if not df.empty:
        summ = df.groupby("split_label")[[f"relL2_{n}" for n in names] + ["relL2_mean_surface"]].mean()
        summ.to_csv(out_dir / "surface_only_validation_metrics_summary.csv")
        print("[SURFACE-ONLY VALIDATION SUMMARY]")
        print(summ.to_string())
    if args.make_plots:
        plot_surface_cases(base, model, data, pred, device, args, out_dir)
        plot_training_and_bars(out_dir)
    return df


def plot_training_and_bars(out_dir: Path):
    logp = out_dir / "surface_only_training_log.csv"
    if logp.exists():
        df = pd.read_csv(logp)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.semilogy(df["epoch"], df["train_loss"], label="train")
        ax.semilogy(df["epoch"], df["val_loss"], label="validation")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted Huber loss")
        ax.legend(); ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_surface_only_training_loss.png", dpi=220)
        plt.close(fig)
    mp = out_dir / "surface_only_validation_metrics_by_case.csv"
    if mp.exists():
        df = pd.read_csv(mp)
        if not df.empty:
            cols = ["relL2_Cp", "relL2_Cq", "relL2_tau_abs"]
            g = df.groupby("split_label")[cols].mean() * 100
            fig, ax = plt.subplots(figsize=(8,4))
            g.plot(kind="bar", ax=ax)
            ax.set_ylabel("Mean relative L2 error (%)")
            ax.set_title("Surface-only operator: protrusion-wall validation")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_surface_only_validation_errors.png", dpi=220)
            plt.close(fig)


def plot_surface_cases(base, model, data, pred, device, args, out_dir: Path):
    plot_dir = out_dir / "surface_only_profiles"
    plot_dir.mkdir(parents=True, exist_ok=True)
    m = data["m"]
    split_label = data["split_label"]
    chosen = list(np.unique(data["case_idx"][data["is_val"]]))[:args.max_val_plots]
    names = ["Cp", "Cq", "tau_abs"]
    for ci in chosen:
        row = m.iloc[int(ci)]
        mask = (data["case_idx"] == ci) & data["is_val"]
        try:
            surf = base.read_prot_surface(str(row["zip_path"]), str(row["xlsx_inner"]))
        except Exception:
            continue
        if surf is None or surf.empty:
            continue
        true = surf[names].to_numpy(float)
        if true.shape[0] != mask.sum():
            # If filtering differed, use just finite rows.
            ok = np.all(np.isfinite(true), axis=1)
            surf = surf.loc[ok].reset_index(drop=True)
            true = surf[names].to_numpy(float)
        pr = pred[mask]
        nmin = min(len(pr), len(true), len(surf))
        pr = pr[:nmin]; true = true[:nmin]; surf = surf.iloc[:nmin].reset_index(drop=True)
        s = surf["s01"].to_numpy(float)

        if all(c in surf.columns for c in ["v1x","v1y","v2x","v2y"]):
            ymid = 0.5*(surf["v1y"].to_numpy(float)+surf["v2y"].to_numpy(float))
            iap = int(np.nanargmax(ymid)) if np.isfinite(ymid).any() else len(s)//2
            apex_s = float(s[iap])
        else:
            apex_s = 0.5
        wind = s <= apex_s + 1e-10
        lee = s >= apex_s - 1e-10

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
        for j, name in enumerate(names):
            ax = axes[j]
            for side_mask, lab in [(wind, "windward"), (lee, "leeward")]:
                if side_mask.any():
                    order = np.argsort(s[side_mask])
                    ss = s[side_mask][order]
                    ax.plot(ss, true[side_mask, j][order], "k-", lw=1.6, label="DSMC" if (j==0 and lab=="windward") else None)
                    ax.plot(ss, pr[side_mask, j][order], "r--", lw=1.6, label="NN" if (j==0 and lab=="windward") else None)
            ax.axvline(apex_s, color="0.55", ls=":", lw=0.9, label="apex" if j==0 else None)
            ax.set_title(name)
            ax.set_xlabel("normalized protrusion coordinate")
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Surface-only validation: {row['case_id']} ({split_label.iloc[int(ci)]})", fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_dir / f"surface_only_{row['case_id']}.png", dpi=220)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-script", default="06_protrusion_train_unified_operator_v7.py")
    ap.add_argument("--split", default="supported", choices=["supported","phase1_only"])
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.03)
    ap.add_argument("--full-residual-scale", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--early-patience", type=int, default=160)
    ap.add_argument("--print-every", type=int, default=25)
    ap.add_argument("--peak-weight", type=float, default=2.0)
    ap.add_argument("--apex-weight", type=float, default=2.0)
    ap.add_argument("--peak-sigma", type=float, default=0.06)
    ap.add_argument("--weight-clip", type=float, default=12.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--predict-batch-size", type=int, default=65536)
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--max-val-plots", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = import_base(args.base_script)

    data = load_surface_dataset(base, Path(args.audit_dir), args)
    print(f"[INFO] surface points: train={(~data['is_val']).sum()} val={data['is_val'].sum()}", flush=True)
    print(f"[INFO] feature dims: case={data['Xc'].shape[1]} surface={data['Xs'].shape[1]}", flush=True)

    model, device = train_model(data, args, out_dir)
    compute_metrics_and_plots(base, model, data, device, args, out_dir)

    with open(out_dir / "final_summary.txt", "w") as f:
        f.write("Surface-only face-split protrusion-wall operator\n")
        f.write("================================================\n\n")
        f.write("Targets: Cp(s), Cq(s), tau_abs(s) on the protrusion wall only.\n")
        f.write("This model is intentionally separate from the full-field operator.\n")
        f.write(f"Train surface points: {int((~data['is_val']).sum())}\n")
        f.write(f"Validation surface points: {int(data['is_val'].sum())}\n")
        mp = out_dir / "surface_only_validation_metrics_summary.csv"
        if mp.exists():
            f.write("\nValidation summary:\n")
            f.write(mp.read_text())
    print("[DONE] outputs under:", out_dir)


if __name__ == "__main__":
    main()
