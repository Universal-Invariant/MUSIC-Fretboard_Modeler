"""Regenerate the simulation-calibrated pluck-envelope lookup table.

Directly integrates the 1-D wave equation (leap-frog finite differences, CFL
number 0.5) for a triangular initial displacement with zero initial velocity,
and records the max-over-time excursion profile G(u; p).  Stored as shape ratio
R(u; p) = G(u; p) / max_u G(u; p), so R.max() == 1 exactly and the runtime
model amp(u) = A·R(u;p) reproduces the true physics: a pluck of height A
released from rest reaches a global peak excursion of exactly A at the pluck
point (d'Alembert solution; verified against this same simulation).

Usage::

    python -m fretlab.tools.calibrate_pluck [--out fretlab/data/pluck_envelope_table.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def sim_profile(p: float, N: int = 240, T: float = 80.0, c: float = 1.0):
    """Max-over-time |y| profile of a plucked string on u∈[0,1]."""
    dx = 1.0 / N
    dt = 0.5 * dx / c
    x = np.linspace(0.0, 1.0, N + 1)
    y0 = np.where(x <= p, x / p, (1.0 - x) / (1.0 - p))
    y0[0] = y0[-1] = 0.0
    r2 = (c * dt / dx) ** 2
    yp = y0.copy()          # zero initial velocity ⇒ y(-dt) == y(0)
    yc = y0.copy()
    peak = np.abs(yc).copy()
    for _ in range(int(T / dt)):
        yn = np.zeros_like(yc)
        yn[1:-1] = (2 * yc[1:-1] - yp[1:-1]
                    + r2 * (yc[2:] - 2 * yc[1:-1] + yc[:-2]))
        yp, yc = yn, yc
        np.maximum(peak, np.abs(yc), out=peak)
    return x, peak


def build_table(nu: int = 65,
                p_grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_grid = np.linspace(0.0, 1.0, nu)
    if p_grid is None:
        p_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    rows = []
    for p in p_grid:
        x, pk = sim_profile(float(p))
        gmax = float(pk.max())
        if abs(gmax - 1.0) > 0.02:      # apex height must be ~1
            raise RuntimeError(f"simulation diverged at p={p}: peak {gmax}")
        prof = np.interp(u_grid, x, pk)
        rows.append(prof / prof.max())   # global-peak normalization: R.max()==1
    return u_grid, np.asarray(p_grid, dtype=float), np.asarray(rows)


def main(out_path: str | None = None) -> None:
    out = Path(out_path or Path(__file__).resolve().parents[1]
               / "data" / "pluck_envelope_table.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    u, p, R = build_table()
    np.savez(out, u=u, p=p, R=R)
    print(f"wrote {out}  shape={R.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output .npz path")
    args = ap.parse_args()
    main(args.out)
