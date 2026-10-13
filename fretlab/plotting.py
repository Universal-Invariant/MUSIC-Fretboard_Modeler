"""Matplotlib side-profile renderer.

Primary view (v0.1): 2-D nut→saddle profile showing
  * fingerboard surface (bowed) — filled "wood" band
  * fret crowns as wires sitting on the board
  * open string centerline + ±peak envelope (shaded)
  * optional fretted-at-k overlay
  * clearance callouts at flagged frets

A 3-D switchable view is planned (v0.4, plotly); the API keeps a ``mode`` kwarg
reserved for it already.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe default; callers may override before import
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .analysis import StringAnalysis

STATUS_COLORS = {"OK": "#2e7d32", "MARGINAL": "#f9a825", "BUZZ": "#c62828", "n/a": "#9e9e9e"}


def plot_string_profile(
    res: StringAnalysis,
    path: str | None = "profile.png",
    fretted_at: int | None = None,
    show_advisories: bool = True,
    dpi: int = 150,
    mode: str = "2d",
    instrument=None,
    x_limit_mm: float | None = None,
):
    """Render one string's profile. Returns the matplotlib Figure."""
    if mode != "2d":
        raise NotImplementedError("3-D rendering arrives in v0.4 (plotly)")

    inst = instrument or getattr(res, "_instrument", None)
    if inst is None:
        raise ValueError("plot_string_profile needs the Instrument (pass instrument=...)")

    cfg = inst.cfg
    x_end = x_limit_mm or (cfg.contact_points.saddle_position_mm + 15.0)
    xs = np.linspace(0.0, x_end, 1200)

    board_z = inst.board_z(xs)
    crown_z = inst.crown_z(xs)
    thickness = cfg.fretboard.thickness_mm

    fig, ax = plt.subplots(figsize=(12, 4.2))

    # wood band
    ax.fill_between(xs, board_z - thickness, board_z, color="#8d6e63", alpha=0.85,
                    label="fingerboard")
    ax.plot(xs, board_z, color="#5d4037", lw=1.0)

    # fret wires
    for c in res.checks[1:]:
        cz = float(inst.crown_z(np.array([c.x_mm]))[0])
        col = STATUS_COLORS.get(_status_for(cfg, c), "#9e9e9e")
        board_at_fret = float(board_z[np.argmin(np.abs(xs - c.x_mm))])
        h_wire = max(cz - board_at_fret, 0.05)
        ax.add_patch(Rectangle((c.x_mm - 0.6, board_at_fret), 1.2, h_wire + 0.1,
                               facecolor="#b0bec5", edgecolor="#37474f", lw=0.6))
        if _flagged(cfg, c):
            ax.annotate(f"{c.fret}", (c.x_mm, cz + 0.35), fontsize=7,
                        ha="center", color=col, weight="bold")

    # open string + envelope
    op = res.open_path
    m = op.inside(xs)
    xm = xs[m]
    center = op.line(xm)
    env_local = np.interp(xm - op.x0_mm, res.open_env.s_mm, res.open_env.amp_mm)
    ax.fill_between(xm, center - env_local, center + env_local,
                    color="#1e88e5", alpha=0.18)
    ax.plot(xm, center - env_local, color="#1e88e5", lw=0.8, ls="--")
    ax.plot(xm, center + env_local, color="#1e88e5", lw=0.8, ls="--",
            label=f"open envelope ({res.physics.name})")
    ax.plot(xm, center, color="#0d47a1", lw=1.6, label="string centerline")

    # optional fretted overlay
    if fretted_at is not None and fretted_at in res.fretted_paths:
        fp, fe = res.fretted_paths[fretted_at]
        mf = fp.inside(xs)
        xfm = xs[mf]
        cf = fp.line(xfm)
        ef = np.interp(xfm - fp.x0_mm, fe.s_mm, fe.amp_mm)
        ax.fill_between(xfm, cf - ef, cf + ef, color="#43a047", alpha=0.18)
        ax.plot(xfm, cf, color="#2e7d32", lw=1.2, ls="-.",
                label=f"fretted @{fretted_at}")

    # nut & saddle markers
    nx, nz = inst.nut_anchor()
    sx, sz = inst.saddle_anchor()
    ax.plot([nx], [nz], marker="v", color="#6a1b9a", ms=8, label="nut")
    ax.plot([sx], [sz], marker="^", color="#e65100", ms=8, label="saddle")

    ax.set_xlabel("distance from nut (mm)")
    ax.set_ylabel("height above nut plane (mm)")
    title = f"{cfg.instrument.name} — {res.physics.name} " \
            f"(f={res.physics.frequency_hz:.1f} Hz, T={res.physics.tension_N:.0f} N)"
    ax.set_title(title, fontsize=11)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-5, x_end)

    if show_advisories and res.advisories:
        txt = "\n".join("• " + a for a in res.advisories)
        ax.text(0.99, 0.02, txt, transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round", fc="#fff8e1", ec="#f9a825", alpha=0.9))

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi)
    return fig


def _status_for(cfg, c) -> str:
    vals = [v for v in (c.open_clearance_mm, c.fretted_clearance_mm) if v is not None]
    if not vals:
        return "n/a"
    m = min(vals)
    if m < cfg.buzz.buzz_clearance_mm:
        return "BUZZ"
    if m < cfg.buzz.warn_clearance_mm:
        return "MARGINAL"
    return "OK"


def _flagged(cfg, c) -> bool:
    return _status_for(cfg, c) in ("BUZZ", "MARGINAL")
