"""Fretboard geometry: fret positions, board plane, neck fall, fret heights.

Coordinate convention (side view, single string):
  * ``x`` — distance in mm from the **nut** toward the saddle (increases down-neck).
  * ``z`` — height in mm above the reference plane through the nut, positive up.

The "board line" is the machined fingerboard surface ignoring bow::

    z_board_plane(x) = tan(board_angle) * x  -  neck_fall ramp past the last fret

Positive ``board_angle_deg`` raises the saddle-side end relative to the nut (i.e.
more headstock pitch on an acoustic → higher action).  Neck fall subtracts a
smooth ramp starting at the last fret (heel carve / overstand drop).

The fret crown line adds wire height on top of the (bowed) board surface.  All
heights refer to the **crown top**; the string centerline sits half a gauge above
whichever surface it rests on (see :mod:`fretlab.strings`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .config import FretHeightConfig, FretLabConfig, InstrumentConfig
from .functions import compile_expr


# --------------------------------------------------------------------------- #
# Fret positions
# --------------------------------------------------------------------------- #

def equal_temperament_positions(scale_mm: float, num_frets: int,
                                octave_fret: int = 12) -> np.ndarray:
    """Generic equal-temperament layout: ``octave_fret`` divides the string in half.

    d_n = S * (1 - 2**(-n/octave_fret))  for n = 1..num_frets.
    The classic "rule of 17" is this with octave_fret = 12; mandolins etc. can
    use 15ths-based layouts via custom tables instead.
    """
    n = np.arange(1, num_frets + 1, dtype=float)
    return scale_mm * (1.0 - 2.0 ** (-n / float(octave_fret)))


def rule_of_17_positions(scale_mm: float, num_frets: int) -> np.ndarray:
    """Classic ET wire placement (octave at the 12th). Kept as a friendly alias."""
    return equal_temperament_positions(scale_mm, num_frets, octave_fret=12)


def true_temperament_positions(scale_mm: float, num_frets: int) -> np.ndarray:
    """Positions using just ratios for a historically-informed 'TT' layout.

    A common 5-limit ratio set per semitone; provided so luthiers can compare
    layouts visually.  Octaves repeat every 13 steps in this table.
    """
    tt_ratios = [
        1 / 1, 25 / 24, 10 / 9, 8 / 7, 6 / 5, 5 / 4, 4 / 3, 45 / 32, 3 / 2,
        8 / 5, 5 / 3, 9 / 5, 15 / 8,
    ]
    pos = []
    for i in range(num_frets):
        oct_i, semi = divmod(i, 13)
        ratio = tt_ratios[semi] * (2.0 ** oct_i)
        pos.append(scale_mm * (1.0 - 1.0 / ratio))
    return np.asarray(pos)


def fret_positions(cfg: InstrumentConfig) -> np.ndarray:
    model = cfg.fret_position_model
    if model == "rule_of_17":
        return equal_temperament_positions(
            cfg.scale_length_mm, cfg.num_frets, cfg.octave_fret)
    if model == "true_temperament":
        return true_temperament_positions(cfg.scale_length_mm, cfg.num_frets)
    if model == "custom_positions":
        return np.asarray(cfg.custom_fret_positions_mm, dtype=float)
    if model == "data":
        if not (cfg.fret_data.csv or cfg.fret_data.points):
            raise ValueError("fret_position_model='data' needs fret_data.csv/points")
        if cfg.fret_data.csv:
            xs = _load_curve_csv(Path(cfg.fret_data.csv), "dist_mm", "height_mm")[0]
        else:
            xs = np.asarray(cfg.fret_data.points, dtype=float)[:, 0]
        return xs.astype(float)
    raise ValueError(f"unknown fret position model {model!r}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Board plane + neck fall
# --------------------------------------------------------------------------- #

@dataclass
class BoardPlane:
    """Un-bowed fingerboard surface profile z(x)."""

    angle_deg: float = 0.0
    neck_fall_mm: float = 0.0
    last_fret_x_mm: float = 0.0
    board_end_x_mm: float = 0.0
    _fall_span_mm: float = field(default=40.0, repr=False)

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = math.tan(math.radians(self.angle_deg)) * x
        if self.neck_fall_mm and self.last_fret_x_mm > 0:
            # smoothstep ramp from 0 at last fret to full fall over `_fall_span`
            t = np.clip((x - self.last_fret_x_mm) / self._fall_span_mm, 0.0, 1.0)
            z = z - self.neck_fall_mm * (t * t * (3.0 - 2.0 * t))
        return z

    def slope(self, x: np.ndarray | float) -> np.ndarray:
        """dz/dx of the plane part only (used for crown-normal lift offsets)."""
        x = np.asarray(x, dtype=float)
        return np.full_like(x, math.tan(math.radians(self.angle_deg)))


def make_board_plane(cfg: FretLabConfig) -> BoardPlane:
    frets = fret_positions(cfg.instrument)
    last = float(frets[-1]) if len(frets) else cfg.instrument.scale_length_mm
    end = cfg.fretboard.length_mm or (cfg.contact_points.saddle_position_mm + 20.0)
    return BoardPlane(
        angle_deg=cfg.instrument.board_angle_deg,
        neck_fall_mm=cfg.instrument.neck_fall_mm,
        last_fret_x_mm=last,
        board_end_x_mm=end,
    )


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #

def _load_curve_csv(path: Path, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"{path} must have columns '{x_col}' and '{y_col}'")
    order = np.argsort(df[x_col].to_numpy(dtype=float))
    return df[x_col].to_numpy(float)[order], df[y_col].to_numpy(float)[order]


# --------------------------------------------------------------------------- #
# Fret height profiles
# --------------------------------------------------------------------------- #

class FretHeights:
    """Per-fret crown heights above the local board surface.

    Heights are *piecewise-constant* along the neck: a fret wire has essentially
    uniform height across its width, so querying any x returns the height of the
    **nearest fret** (before the first fret → the nut-region height; after the
    last → the last fret's height).  Linear interpolation between frets would
    produce phantom low spots mid-span and wrong crown tops inside the ±window
    used by the clearance analysis, so it is deliberately avoided.

    ``at_frets()`` returns the configured heights index-aligned with the fret
    positions (exact, independent of the nearest-fret mapping).
    """

    def __init__(self, x: np.ndarray, h: np.ndarray, mode: str = "",
                 fret_x: np.ndarray | None = None):
        self.x = np.asarray(x, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.mode = mode
        if fret_x is not None:
            self.fret_x = np.asarray(fret_x, dtype=float)
            # exact per-fret values (taper/custom/data evaluated at the wires)
            self._fret_h = np.interp(self.fret_x, self.x, self.h) \
                if len(self.x) >= 2 else np.full_like(self.fret_x, 
                    float(self.h[0]) if len(self.h) else 0.0)
        else:
            self.fret_x = None
            self._fret_h = self.h.copy()

    def at_frets(self) -> np.ndarray:
        """Heights evaluated at each fret position (index-aligned)."""
        return self._fret_h.astype(float)

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.fret_x is None or len(self.fret_x) == 0:
            # no wires to key off: fall back to piecewise-linear on samples
            if len(self.x) >= 2:
                return np.interp(x, self.x, self.h)
            c = float(self.h[0]) if len(self.h) else 0.0
            return np.full_like(x, c)
        idx = np.searchsorted(self.fret_x, x)
        idx = np.clip(idx, 0, len(self.fret_x) - 1)
        # nearest wire (edges clamp automatically via clip + comparison below)
        left = np.maximum(idx - 1, 0)
        d_l = np.abs(x - self.fret_x[left])
        d_r = np.abs(x - self.fret_x[idx])
        pick = np.where(d_l <= d_r, left, idx)
        return self._fret_h[pick]


def make_fret_height_profile(fc: FretHeightConfig, fret_x: np.ndarray,
                             scale_mm: float) -> FretHeights:
    """Return an object giving crown height above the local board at any x."""
    fret_x = np.asarray(fret_x, dtype=float)
    if fc.model == "uniform":
        return FretHeights(np.array([0.0, max(scale_mm, 1.0)]),
                           np.array([fc.height_mm, fc.height_mm]),
                           mode="uniform", fret_x=fret_x)
    if fc.model == "taper":
        h = np.interp(fret_x, [0.0, scale_mm],
                      [fc.height_at_nut_mm, fc.height_at_last_fret_mm])
        return FretHeights(fret_x, h, mode="taper", fret_x=fret_x)
    if fc.model == "data":
        if fc.data.csv:
            xs, hs = _load_curve_csv(Path(fc.data.csv), "dist_mm", "height_mm")
        elif fc.data.points:
            arr = np.asarray(fc.data.points, dtype=float)
            xs, hs = arr[:, 0], arr[:, 1]
        else:
            raise ValueError("fret_heights.model='data' requires csv or points")
        order = np.argsort(xs)
        return FretHeights(xs[order], hs[order], mode="data", fret_x=fret_x)
    if fc.model == "custom":
        f = compile_expr(fc.expr)
        h = np.broadcast_to(np.asarray(f(fret_x), dtype=float), fret_x.shape)
        return FretHeights(fret_x, h.astype(float), mode="custom", fret_x=fret_x)
    raise ValueError(f"unknown fret_height model {fc.model}")  # pragma: no cover
