"""Neck relief (bow) models and the linearized truss-rod solver.

Sign convention: **relief is positive** — the board surface dips *away* from the
strings at mid-neck (concave up when viewed with strings above).  Back-bow is
negative.  Bow is measured relative to the straight board plane, anchored zero at
both ends of the supported span (like a feeler gauge under a straightedge resting
on the nut and last fret).

v0.1 shape models
-----------------
flat                 : no bow
cubic                : asymmetric bump peaking near ~25% of the span (classic relief look)
constant_curvature   : parabolic arc from κ = 1/R (signed radius), zero at both ends
circular_arc         : true circular segment through the endpoints (sagitta or radius)
data                 : measured points (CSV / inline), cubic-spline'd
custom               : user expression in x (mm from nut); negative dz == relief dip

Truss model (linearized engineering beam)
-----------------------------------------
Net peak relief at operating tension::

    relief_eff = relief_base + turns · truss_stiffness
                 + (T_total − T_tune) · string_load_coefficient

``truss_stiffness`` is the (signed) relief change per full turn (mm/turn; tightening
reduces relief ⇒ negative), ``string_load_coefficient`` the relief added per newton
of tension **above the tuned baseline** (mm/N).  Both are *calibrated constants*,
not E·I derivations — they absorb section geometry, rod leverage and thread
compliance.  Calibrate them from a before/after measurement pair (see
:func:`calibrate_truss`).

Baseline convention: the configured bow shape normally describes the neck
**at tune** (that's what you measure with a feeler gauge on a strung guitar).
The tension term therefore measures deviation from ``truss.tune_tension_N``,
not absolute load: retuning down a whole step reduces relief slightly; an
untensioned (bare-neck) baseline model should set ``tune_tension_N = 0`` so
full string load adds relief as expected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .config import BowConfig, FretLabConfig
from .functions import compile_expr


# --------------------------------------------------------------------------- #
# Shape functions: return dz(x); dz < 0 means the surface dipped (relief > 0)
# --------------------------------------------------------------------------- #

def cubic_relief_shape(x: np.ndarray, L: float, peak_frac: float = 0.25) -> np.ndarray:
    """Unit-amplitude classic relief bump: min = -1 at ``peak_frac·L``, 0 at ends.

    Construction: a skewed sin² whose *support* is [0, 2·peak_frac] (the surface
    returns to the chord line at x = 2·peak_frac·L and stays flat past it — like
    the real straightedge-over-neck profile), times a half-sine so both endpoints
    are exactly zero.  The maximum sits precisely at peak_frac·L by symmetry of
    the support window.
    """
    t = np.clip(np.asarray(x, dtype=float) / L, 0.0, 1.0)
    p = min(max(peak_frac, 1e-3), 0.499)
    span = 2.0 * p
    w = np.zeros_like(t)
    inside = t < span
    # smooth bump on [0, span], peak at span/2 == p
    u = t[inside] / span                      # 0..1 across the support
    bump = np.sin(math.pi * u) ** 2           # symmetric → max at u=0.5
    taper = np.sin(math.pi * t[inside])       # keeps endpoints pinned at 0
    taper_max = np.sin(math.pi * p)
    w[inside] = bump * (taper / taper_max)
    m = w.max() if len(w) else 0.0
    if m > 0:
        w = w / m
    return -w


def circular_segment(x: np.ndarray, L: float, sagitta_mm: float) -> np.ndarray:
    """Circular arc through (0,0) & (L,0); sagitta>0 dips down by that amount at center."""
    s = max(abs(sagitta_mm), 1e-9)
    R = (L * L / 4.0 + s * s) / (2.0 * s)
    xs = np.asarray(x, dtype=float)
    inside = np.clip(R * R - (xs - L / 2.0) ** 2, 0.0, None)
    # At center: (R - s) - R = -s (dip). At ends: (R - s) - (R - s) = 0. ✓
    z = (R - s) - np.sqrt(inside)
    return z if sagitta_mm >= 0 else -z


def constant_curvature_bow(x: np.ndarray, L: float, radius_mm: float) -> np.ndarray:
    """Parabola from signed curvature κ=1/R, zero at both ends. R>0 ⇒ relief (dip)."""
    k = 1.0 / radius_mm
    xs = np.asarray(x, dtype=float)
    return -(k / 2.0) * xs * (L - xs)  # at center: -kL²/8 → negative dz = relief ✓


def _load_points(bow: BowConfig) -> tuple[np.ndarray, np.ndarray]:
    # Support both CurveSource object and bare list of [x,y] pairs
    data = bow.data
    if isinstance(data, list):
        # Bare list: [[0,0],[324,-0.1],[648,0]]
        if len(data) < 2:
            raise ValueError("bow data needs at least 2 points")
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("bow data must be a list of [x,y] pairs")
        order = np.argsort(arr[:, 0])
        return arr[order, 0], arr[order, 1]
    
    # CurveSource object with .csv or .points attributes
    if hasattr(data, 'csv') and data.csv:
        df = pd.read_csv(Path(data.csv))
        if "x_mm" not in df.columns or "deflection_mm" not in df.columns:
            raise ValueError("bow CSV needs columns x_mm, deflection_mm")
        order = np.argsort(df["x_mm"].to_numpy(float))
        xs = df["x_mm"].to_numpy(float)[order]
        ys = df["deflection_mm"].to_numpy(float)[order]
    elif hasattr(data, 'points') and data.points:
        arr = np.asarray(data.points, dtype=float)
        xs, ys = arr[:, 0], arr[:, 1]
    else:
        raise ValueError("bow.model='data' requires csv or points")
    if len(xs) < 2:
        raise ValueError("bow data needs at least 2 points")
    return xs, ys


class BowFunction:
    """Callable neck-deflection profile over [0, span]; dz<0 == relief dip."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], model: str, span_mm: float):
        self._fn = fn
        self.model = model
        self.span_mm = span_mm

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        return np.asarray(self._fn(np.asarray(x, dtype=float)), dtype=float)

    def peak_relief_mm(self, samples: int = 512) -> float:
        """Signed peak: positive = relief depth, negative = back-bow height."""
        xs = np.linspace(0.0, self.span_mm, samples)
        zs = self(xs)
        if not len(zs):
            return 0.0
        deepest_dip = -zs.min()   # >0 if there is a dip
        highest_rise = zs.max()   # >0 if back-bow (surface above chord)
        return float(deepest_dip) if deepest_dip >= highest_rise else float(-highest_rise)

    def scaled_to(self, target_relief_mm: float) -> "BowFunction":
        """Return same-shape function whose peak relief equals the target.

        For shapes that cannot be rescaled (``data`` measurements, ``custom``
        expressions) the original curve is returned unchanged — those models
        ARE the measurement, and the truss solver reports its residual instead
        of silently warping real data.
        """
        if self.model in ("data", "custom"):
            return self
        base = self.peak_relief_mm()
        if abs(base) < 1e-12:
            if abs(target_relief_mm) < 1e-12:
                return self
            # degenerate flat → fall back to cubic shape scaled to target
            L = self.span_mm
            return BowFunction(
                lambda x, A=target_relief_mm, L=L: A * cubic_relief_shape(x, L),
                model="truss-scaled(cubic)", span_mm=L,
            )
        scale = target_relief_mm / base
        return BowFunction(lambda x, f=self._fn, s=scale: f(x) * s,
                           model=self.model + "+scaled", span_mm=self.span_mm)


def make_bow_fn(bow: BowConfig, L: float) -> BowFunction:
    """Build a callable x→dz from configuration (dz<0 == relief dip).

    For ``circular_arc`` the config field is interpreted as:
      * ``radius_mm`` set            → true circular arc of that radius (sagitta
        derived; positive radius ⇒ relief dip)
      * only ``max_relief_mm`` set   → sagitta (dip depth at mid-span) directly
    """
    if bow.model == "flat":
        return BowFunction(lambda x: np.zeros_like(x), model="flat", span_mm=L)
    if bow.model == "cubic":
        A = bow.max_relief_mm
        return BowFunction(lambda x, A=A: A * cubic_relief_shape(x, L),
                           model="cubic", span_mm=L)
    if bow.model == "constant_curvature":
        if bow.radius_mm is None:
            raise ValueError("constant_curvature bow needs radius_mm")
        R = bow.radius_mm
        return BowFunction(lambda x, R=R: constant_curvature_bow(x, L, R),
                           model="constant_curvature", span_mm=L)
    if bow.model == "circular_arc":
        if bow.radius_mm is not None:
            # sagitta of a circular segment: s = R - sqrt(R² - (L/2)²)
            R = abs(bow.radius_mm)
            half = L / 2.0
            if R < half:
                raise ValueError("bow radius smaller than half span — impossible arc")
            sag = R - math.sqrt(R * R - half * half)
            sign = 1.0 if bow.radius_mm > 0 else -1.0
        elif bow.max_relief_mm:
            sag = bow.max_relief_mm
            sign = 1.0 if sag >= 0 else -1.0
        else:
            raise ValueError("circular_arc bow needs radius_mm or max_relief_mm")

        def arc(x, s=sag):
            z = circular_segment(np.clip(x, 0.0, L), L, abs(s))
            return z if sign >= 0 else -z

        return BowFunction(arc, model="circular_arc", span_mm=L)
    if bow.model == "data":
        xs, ys = _load_points(bow)
        cs = CubicSpline(xs, ys)
        return BowFunction(lambda x: cs(np.clip(x, xs[0], xs[-1])),
                           model="data", span_mm=L)
    if bow.model == "custom":
        f = compile_expr(bow.expr)

        def custom(x, f=f):
            vals = np.asarray(f(x), dtype=float)
            return np.broadcast_to(vals, x.shape).astype(float)

        return BowFunction(custom, model="custom", span_mm=L)
    raise ValueError(f"unknown bow model {bow.model}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Truss-rod linearized solver
# --------------------------------------------------------------------------- #

@dataclass
class TrussState:
    effective_relief_mm: float
    net_curvature_per_mm: float
    turns_applied: float
    note: str = ""
    #: True when the configured bow model is a measurement (data/custom) and the
    #: truss/tension correction could NOT be applied to the shape itself.
    unscaled_measurement: bool = False


def _tension_delta(cfg: FretLabConfig, total_tension_N: float) -> float:
    """Tension above the tuned baseline [N].

    The configured bow shape normally describes the neck **at tune**, so the
    string-load term measures deviation from ``truss.tune_tension_N``.  Set
    ``tune_tension_N = 0`` for a bare-neck (untensioned) baseline model.
    """
    return total_tension_N - cfg.truss.tune_tension_N


def resolve_truss(cfg: FretLabConfig, total_tension_N: float) -> TrussState:
    """Resolve configured truss state + string load into an effective peak relief.

    Linearized superposition around the configured bow amplitude::

        relief_eff = relief_base + turns·truss_stiffness
                     + ΔT·string_load_coefficient        (ΔT = T − T_tune)

    For radius-driven models (constant_curvature / circular_arc with radius_mm)
    the *base* is the physical sagitta of that radius — NOT the nominal
    ``max_relief_mm`` field — so the linear correction starts from reality.
    """
    L = cfg.instrument.scale_length_mm
    base_fn = make_bow_fn(cfg.fretboard.bow, L)
    base_relief = base_fn.peak_relief_mm()

    tension_relief = _tension_delta(cfg, total_tension_N) \
        * cfg.truss.string_load_coefficient
    # More tension than the baseline state ⇒ MORE relief (neck pulled into
    # concave-up bow); less tension (retuned down) ⇒ relief drops.
    d_relief = cfg.truss.state_turns * cfg.truss.truss_stiffness + tension_relief
    eff_relief = base_relief + d_relief
    turns = cfg.truss.state_turns
    note = "manual truss state"

    if cfg.truss.solve_mode == "target_relief" and cfg.truss.target_relief_mm is not None:
        target = cfg.truss.target_relief_mm
        if abs(cfg.truss.truss_stiffness) > 1e-12:
            turns = (target - base_relief - tension_relief) / cfg.truss.truss_stiffness
            eff_relief = target
            note = f"solved {turns:+.2f} turns to reach {target:.2f} mm relief"
        else:
            note = "truss_stiffness too small to solve for target relief"

    kappa = (8.0 * eff_relief / (L * L)) if L else 0.0
    unscaled = base_fn.model in ("data", "custom")
    if unscaled:
        note += "; measurement shape kept as-is (relief delta reported only)"
    return TrussState(effective_relief_mm=eff_relief, net_curvature_per_mm=kappa,
                      turns_applied=turns, note=note, unscaled_measurement=unscaled)


def resolved_bow(cfg: FretLabConfig, total_tension_N: float) -> BowFunction:
    """Selected bow *shape*, amplitude-scaled to the truss/tension-resolved relief.

    Shape models (flat/cubic/data/custom) are rescaled so their peak equals the
    resolved relief; radius-driven models (constant_curvature / circular_arc)
    are rebuilt with the net radius implied by the resolved curvature
    (κ_eff = 8·relief/L² ⇒ R_net = 1/κ_eff), keeping the geometry self-consistent.
    Measurement models (data/custom) are never warped — :class:`TrussState`
    reports the residual instead.
    """
    L = cfg.instrument.scale_length_mm
    raw = make_bow_fn(cfg.fretboard.bow, L)
    state = resolve_truss(cfg, total_tension_N)

    model = cfg.fretboard.bow.model
    if model in ("constant_curvature", "circular_arc") and L > 0 \
            and abs(state.effective_relief_mm) > 1e-9:
        kappa = 8.0 * state.effective_relief_mm / (L * L)   # signed, + = relief
        R_net = 1.0 / kappa
        sign = 1.0 if state.effective_relief_mm >= 0 else -1.0
        if model == "constant_curvature":
            fn = lambda x, R=R_net: constant_curvature_bow(x, L, R)
            return BowFunction(fn, model="constant_curvature+truss", span_mm=L)
        sag = (R_net - math.sqrt(R_net * R_net - (L / 2.0) ** 2)) \
            if abs(R_net) >= L / 2.0 else abs(state.effective_relief_mm)

        def arc(x, s=sag):
            z = circular_segment(np.clip(x, 0.0, L), L, s)
            return z if sign >= 0 else -z

        return BowFunction(arc, model="circular_arc+truss", span_mm=L)

    return raw.scaled_to(state.effective_relief_mm)


# --------------------------------------------------------------------------- #
# Calibration helper (v0.2 feature, shipped early because it's cheap)
# --------------------------------------------------------------------------- #

@dataclass
class TrussCalibration:
    truss_stiffness_mm_per_turn: float
    string_load_coefficient_mm_per_N: float
    residuals_mm: list[float]
    n_samples: int


def calibrate_truss(samples: list[tuple[float, float, float]],
                    fit_string_load: bool = False) -> TrussCalibration:
    """Least-squares fit of the linearized model from measured pairs.

    Each sample is ``(turns_delta, tension_delta_N, measured_relief_change_mm)``
    taken between two setups of the SAME instrument — i.e. differences relative
    to a reference setup, so no unknown constant offset pollutes the fit.  With
    one sample we solve stiffness directly (assuming the tension term is
    negligible or known); with ≥2 samples spanning different tensions we can
    least-squares fit both terms when ``fit_string_load=True``::

        d = turns·k + ΔT·c      (c = string_load_coefficient, mm/N)
    """
    if not samples:
        raise ValueError("need at least one calibration sample")
    rows = np.array([[t, T, d] for t, T, d in samples], dtype=float)
    if fit_string_load and len(rows) >= 2:
        A = np.column_stack([rows[:, 0], rows[:, 1]])
        coef, *_ = np.linalg.lstsq(A, rows[:, 2], rcond=None)
        k, c = float(coef[0]), float(coef[1])
        resid = rows[:, 2] - A @ np.array([k, c])
    else:
        denom = np.sum(rows[:, 0] ** 2)
        if denom < 1e-12:
            raise ValueError("all samples have zero turn delta; move the rod!")
        k = float(np.sum(rows[:, 0] * rows[:, 2]) / denom)
        c = 0.0
        resid = rows[:, 2] - rows[:, 0] * k
    return TrussCalibration(truss_stiffness_mm_per_turn=k,
                            string_load_coefficient_mm_per_N=c,
                            residuals_mm=[round(float(r), 4) for r in resid],
                            n_samples=len(rows))
