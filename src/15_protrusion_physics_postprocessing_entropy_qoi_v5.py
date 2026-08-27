#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_protrusion_physics_postprocessing_entropy_qoi_v5.py

Fixed downloadable wrapper for entropy/QoI postprocessing.

Fixes compared with v4
----------------------
1) The entropy error panel is now shown as a PERCENT error, not absolute entropy difference.
2) The percent error is normalized by the robust DSMC entropy range:
       error_percent = 100 * (S_ML - S_DSMC) / (S_DSMC,99 - S_DSMC,1)
   This avoids singular relative errors where entropy is close to zero.
3) The solid triangular protrusion is masked and overlaid.
4) The white holes/speckle are removed by robust binned-grid filling.

This script imports 15_protrusion_physics_postprocessing_entropy_qoi_v2.py from the same folder,
patches only the entropy plotting routine, and then runs the original v2 main().

Required files in the same folder
---------------------------------
15_protrusion_physics_postprocessing_entropy_qoi_v2.py
15_protrusion_physics_postprocessing_entropy_qoi_v5.py

Recommended command
-------------------
cd .

python3 15_protrusion_physics_postprocessing_entropy_qoi_v5.py \
  --audit-dir ./protrusion_01_audit_v2 \
  --out ./protrusion_15_physics_entropy_qoi_v05 \
  --field-base-script ./10_protrusion_train_smooth_field_operator_v4_geomfix.py \
  --field-checkpoint ./protrusion_10_smooth_field_v04_geomfix/best_smooth_field.pt \
  --pressure-base-script ./11_protrusion_train_pressure_only_operator_v2_geomfix.py \
  --pressure-checkpoint ./protrusion_11_pressure_only_v02_geomfix/best_smooth_field.pt \
  --surface-script ./08_protrusion_train_surface_only_operator_v2.py \
  --surface-base-script ./06_protrusion_train_unified_operator_v8_geomfix.py \
  --surface-checkpoint ./protrusion_08_surface_only_v02_geomfix/best_surface_only.pt \
  --sample-points 100000 \
  --plot-points 90000 \
  --cases val \
  --make-plots
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon


def import_module_from_path(path: str, name: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {p}")
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_case_geometry(case_id: str):
    """
    Infer normalized protrusion polygon from case_id.

    Normalized coordinates:
        xloc = (x - x_apex) / h_s
        yloc = (y - y_b) / h_s

    Correct normalized solid vertices:
      BWD : [(-1.5, 0), (-0.5, 0), (0, hphs)]
      FWD : [( 0.5, 0), ( 1.5, 0), (0, hphs)]
      ISO : [(-0.5, 0), ( 0.5, 0), (0, hphs)]
    """
    cid = str(case_id)

    gm = re.search(r"_(BWD|FWD|ISO)_", cid)
    hm = re.search(r"_hphs([0-9]+(?:\.[0-9]+)?)_", cid)

    geom = gm.group(1) if gm else "ISO"
    hphs = float(hm.group(1)) if hm else 1.5

    if geom == "BWD":
        verts = np.array([[-1.5, 0.0], [-0.5, 0.0], [0.0, hphs]], dtype=float)
    elif geom == "FWD":
        verts = np.array([[0.5, 0.0], [1.5, 0.0], [0.0, hphs]], dtype=float)
    else:
        verts = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, hphs]], dtype=float)

    return geom, hphs, verts


def _entropy_fill_grid(
    x,
    y,
    z,
    xlim=(-4.5, 4.5),
    ylim=(0.0, 5.0),
    nx=280,
    ny=190,
):
    """
    Robust binned-grid renderer with iterative neighbor filling.

    This fills plotting holes in gas region only. The solid mask is applied after gridding.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[m]
    y = y[m]
    z = z[m]

    if x.size < 10:
        raise ValueError("Not enough finite points for entropy plot")

    xmin, xmax = xlim
    ymin, ymax = ylim

    xe = np.linspace(xmin, xmax, nx + 1)
    ye = np.linspace(ymin, ymax, ny + 1)

    ix = np.clip(np.digitize(x, xe) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, ny - 1)

    sums = np.zeros((ny, nx), dtype=float)
    cnts = np.zeros((ny, nx), dtype=float)
    np.add.at(sums, (iy, ix), z)
    np.add.at(cnts, (iy, ix), 1.0)

    grid = np.divide(
        sums,
        np.maximum(cnts, 1.0),
        out=np.full((ny, nx), np.nan),
        where=cnts > 0,
    )

    for _ in range(40):
        miss = ~np.isfinite(grid)
        if not miss.any():
            break

        acc = np.zeros_like(grid)
        w = np.zeros_like(grid)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                shifted = np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
                valid = np.isfinite(shifted)

                # Prevent wrap-around pollution.
                if dy == -1:
                    valid[-1, :] = False
                elif dy == 1:
                    valid[0, :] = False
                if dx == -1:
                    valid[:, -1] = False
                elif dx == 1:
                    valid[:, 0] = False

                mask = miss & valid
                acc[mask] += shifted[mask]
                w[mask] += 1.0

        fillable = miss & (w > 0)
        if not fillable.any():
            break

        grid[fillable] = acc[fillable] / w[fillable]

    if np.any(~np.isfinite(grid)):
        med = np.nanmedian(z)
        if not np.isfinite(med):
            med = 0.0
        grid[~np.isfinite(grid)] = med

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(xc, yc)

    return XX, YY, grid


def _mask_solid_region(XX, YY, ZZ, verts):
    """Mask the triangular protrusion solid region."""
    poly = MplPath(verts)
    pts = np.c_[XX.ravel(), YY.ravel()]
    inside = poly.contains_points(pts, radius=-1e-12).reshape(XX.shape)
    Zm = np.array(ZZ, copy=True)
    Zm[inside] = np.nan
    return Zm


def _draw_geometry(ax, verts):
    patch = Polygon(
        verts,
        closed=True,
        facecolor="white",
        edgecolor="black",
        linewidth=1.1,
        zorder=20,
    )
    ax.add_patch(patch)


def _plot_entropy_panel(ax, x, y, z, title, cmap, vmin, vmax, verts, cbar_label=None):
    XX, YY, ZZ = _entropy_fill_grid(x, y, z)
    ZM = _mask_solid_region(XX, YY, ZZ, verts)

    im = ax.pcolormesh(
        XX,
        YY,
        ZM,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    _draw_geometry(ax, verts)

    ax.set_title(title)
    ax.set_xlabel(r"$(x-x_{apex})/h_s$")
    ax.set_ylabel(r"$(y-y_b)/h_s$")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0.0, 5.0)
    ax.grid(True, linewidth=0.25, alpha=0.20)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)

    return im


def _robust_entropy_percent_error(S_pred, S_true, qlo=1.0, qhi=99.0):
    """
    Error in percent of the robust DSMC entropy range.

    This is the safest percent form for entropy because S can be near zero:
        100 * (S_pred - S_true) / (S_q99 - S_q1)
    """
    S_pred = np.asarray(S_pred, dtype=float)
    S_true = np.asarray(S_true, dtype=float)
    m = np.isfinite(S_pred) & np.isfinite(S_true)
    if m.sum() < 10:
        return np.full_like(S_true, np.nan, dtype=float), np.nan

    lo, hi = np.nanpercentile(S_true[m], [qlo, qhi])
    denom = float(hi - lo)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        denom = np.nanstd(S_true[m])
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        denom = 1.0

    err_pct = 100.0 * (S_pred - S_true) / denom
    return err_pct, denom


def fig_entropy_examples(plot_payloads, out_dir):
    """
    Replacement plotting routine.

    Fixes:
    - solid region is masked
    - white speckles are removed
    - entropy error is shown in percent robust range
    """
    out_dir = Path(out_dir)

    if not plot_payloads:
        print("[WARN] No entropy plot payloads available.")
        return

    n = len(plot_payloads)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(13.5, 3.65 * n),
        squeeze=False,
        constrained_layout=True,
    )

    for r, payload in enumerate(plot_payloads):
        case_id = str(payload.get("case_id", f"case_{r+1}"))
        split_label = str(payload.get("split_label", ""))
        _geom, _hphs, verts = _parse_case_geometry(case_id)

        x = np.asarray(payload["xloc"], dtype=float)
        y = np.asarray(payload["yloc"], dtype=float)
        st = np.asarray(payload["S_true"], dtype=float)
        sp = np.asarray(payload["S_pred"], dtype=float)

        mf = np.isfinite(x) & np.isfinite(y) & np.isfinite(st) & np.isfinite(sp)
        x = x[mf]
        y = y[mf]
        st = st[mf]
        sp = sp[mf]

        err_pct, denom = _robust_entropy_percent_error(sp, st)

        joint = np.r_[st[np.isfinite(st)], sp[np.isfinite(sp)]]
        if joint.size == 0:
            continue

        vmin12, vmax12 = np.nanpercentile(joint, [2, 98])

        if np.isfinite(err_pct).any():
            err_lim = float(np.nanpercentile(np.abs(err_pct[np.isfinite(err_pct)]), 98))
        else:
            err_lim = 10.0
        err_lim = max(err_lim, 1.0)

        # Optional cap so one local artifact does not destroy contrast.
        err_lim = min(err_lim, 40.0)

        panels = [
            ("DSMC entropy proxy", st, "viridis", vmin12, vmax12, r"$\Delta s_{eq}/R$"),
            ("ML entropy proxy", sp, "viridis", vmin12, vmax12, r"$\Delta s_{eq}/R$"),
            ("Error (% robust range)", err_pct, "coolwarm", -err_lim, err_lim, "%"),
        ]

        for c, (title, val, cmap, vmin, vmax, label) in enumerate(panels):
            _plot_entropy_panel(
                ax=axes[r, c],
                x=x,
                y=y,
                z=val,
                title=title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                verts=verts,
                cbar_label=label,
            )

        text = case_id if split_label == "" else f"{case_id}\n{split_label}"
        axes[r, 0].text(
            0.02,
            0.95,
            text,
            transform=axes[r, 0].transAxes,
            va="top",
            fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="none"),
        )

    fig.suptitle("Local-equilibrium entropy-rise proxy validation", fontsize=15)

    png_path = out_dir / "fig_01_entropy_validation_examples.png"
    pdf_path = out_dir / "fig_01_entropy_validation_examples.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] wrote {png_path}")
    print(f"[DONE] wrote {pdf_path}")


def main():
    here = Path(__file__).resolve().parent
    base_path = here / "15_protrusion_physics_postprocessing_entropy_qoi_v2.py"

    if not base_path.exists():
        raise FileNotFoundError(
            "Base script not found.\n"
            f"Expected:\n  {base_path}\n\n"
            "Please keep 15_protrusion_physics_postprocessing_entropy_qoi_v2.py "
            "in the same folder as this v5 script."
        )

    base = import_module_from_path(str(base_path), "entropy_qoi_v2_base")

    # Replace only the entropy plotting routine.
    base.fig_entropy_examples = fig_entropy_examples

    # Run original CLI/main.
    base.main()


if __name__ == "__main__":
    main()
