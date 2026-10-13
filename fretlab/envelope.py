"""Peak-displacement vibration envelopes for an ideal plucked string.

We deliberately avoid time-domain simulation *at analysis runtime*: what a setup
needs is the *maximum* excursion of the string at every point along its length —
the envelope that the string sweeps out after a pluck.  If that envelope clears
every fret crown, the note will not buzz (all else equal).

Models
------
fundamental
    Single half-sine hump: E(x) = A·sin(πx/L). Conservative & simple.

pluck_spectrum
    Physically calibrated pluck envelope.  For a triangular pluck of height A
    released from rest, the exact d'Alembert solution reaches a max-over-time
    excursion of exactly A at the pluck point (not the smaller mid-string
    swing); the profile is asymmetric for off-center plucks.  We interpolate a
    lookup table generated offline by direct finite-difference simulation of
    the wave equation (``fretlab/data/pluck_envelope_table.npz``; regenerate
    with ``python -m fretlab.tools.calibrate_pluck``).  The table is normalized
    so its global peak is 1, hence amp(u) = A·R(u;p) has peak exactly A —
    verified against an independent high-resolution simulation.

hi_harmonic
    Worst-case ring-out test: pure harmonic m (default 4th partial ⇒ node at L/4,
    antinodes at L/8, 3L/8...), amplitude scaled by ``pluck_amplitude_mm``.

worst_case_modal (via :func:`worst_case_modal_envelope`)
    Paranoid alternative: phase-aligned truncated modal sum.  Not selected by
    default; useful as a conservative upper bound.

All envelopes are returned as *positive amplitudes* around the static centerline;
the analysis layer subtracts them (string swings both ways symmetrically about
the static line in this peak-envelope abstraction).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExcitationConfig

_TABLE_PATH = Path(__file__).parent / "data" / "pluck_envelope_table.npz"


@dataclass
class Envelope:
    """Peak vertical excursion (±mm) sampled on a grid over speaking length."""

    s_mm: np.ndarray      # position along speaking segment [0..L]
    amp_mm: np.ndarray    # peak amplitude at each s (>= 0)
    energy_fraction: float = 1.0   # pluck-model only: share of ideal energy captured

    @property
    def peak_mm(self) -> float:
        return float(self.amp_mm.max()) if len(self.amp_mm) else 0.0


def _triad_coeffs(L: float, x0: float, n_terms: int) -> np.ndarray:
    """Fourier amplitudes of a triangular initial shape (pluck at x0, height 1)."""
    n = np.arange(1, n_terms + 1, dtype=float)
    b = (2.0 / (n * math.pi)) ** 2 * np.abs(np.sin(n * math.pi * x0 / L))
    return b


def _modal_energy_fraction(L: float, x0_ratio: float, n_terms: int) -> float:
    """Fraction of ideal-pluck energy held in modes 1..n_terms (triangular pluck).

    Modal energy ∝ ω_n²·b_n² ∝ (n·b_n)² for a plucked string.  Used to report how
    much of the modeled motion the truncated spectrum actually represents.
    """
    n_all = np.arange(1, 64, dtype=float)
    b_all = (2.0 / (n_all * math.pi)) ** 2 * np.abs(
        np.sin(n_all * math.pi * x0_ratio))
    e_all = (n_all * b_all) ** 2
    k = max(int(n_terms), 1)
    return float(e_all[:k].sum() / e_all.sum()) if e_all.sum() > 0 else 1.0


@functools.lru_cache(maxsize=1)
def _load_pluck_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(_TABLE_PATH)
    return d["u"], d["p"], d["R"]


def _sim_shape_ratio(u: np.ndarray, p: float) -> np.ndarray:
    """Bilinear interpolation of the simulated shape ratio R(u; p) = G(u)/G_max.

    Table rows are normalized so max_u R == 1 (global peak at the pluck point).
    Table covers p ∈ [0.05, 0.5]; outside that we mirror p → 1−p (the pluck
    problem is symmetric about mid-string).  Clamped at the table edges.
    """
    u_grid, p_grid, R = _load_pluck_table()
    pm = p if p <= 0.5 else 1.0 - p
    pm = float(np.clip(pm, p_grid[0], p_grid[-1]))
    um = np.clip(u, u_grid[0], u_grid[-1])
    i = int(np.clip(np.searchsorted(p_grid, pm) - 1, 0, len(p_grid) - 2))
    w = (pm - p_grid[i]) / (p_grid[i + 1] - p_grid[i])
    row = (1.0 - w) * R[i] + w * R[i + 1]
    return np.interp(um, u_grid, row)


def worst_case_modal_envelope(exc: ExcitationConfig, speaking_length_mm: float,
                              samples: int = 512) -> Envelope:
    """Conservative alternative envelope: phase-aligned worst-case modal sum.

    E(x) = Σₙ |bₙ|·|sin(nπx/L)| with triangular-pluck coefficients bₙ, rescaled
    so the *global* peak equals the analytic midpoint excursion A·min(p, 1−p).
    Because the truncated modal sum has Gibbs-like ripple, its local excursions
    in the tails (near nut and saddle) exceed the smooth Gibbs solution — up to
    ~1.3× around the 4th-partial lobes — which is exactly where fretted notes
    buzz against low crowns.  Use this as a "paranoid" check; not used by default.
    """
    L = speaking_length_mm
    s = np.linspace(0.0, L, samples)
    A = exc.pluck_amplitude_mm
    p = exc.pluck_position_ratio
    b = _triad_coeffs(L, p * L, max(int(exc.harmonics), 1))
    raw = np.zeros_like(s)
    for k, bk in enumerate(b, start=1):
        raw += bk * np.abs(np.sin(k * math.pi * s / L))
    amp = A * min(p, 1.0 - p) * raw / max(float(raw.max()), 1e-12)
    amp[0] = 0.0
    amp[-1] = 0.0
    return Envelope(s_mm=s, amp_mm=amp,
                    energy_fraction=_modal_energy_fraction(L, p, exc.harmonics))


def make_envelope(exc: ExcitationConfig, speaking_length_mm: float,
                  samples: int = 512) -> Envelope:
    L = speaking_length_mm
    s = np.linspace(0.0, L, samples)
    A = exc.pluck_amplitude_mm

    if exc.model == "fundamental":
        amp = A * np.sin(math.pi * s / L)
        energy_fraction = 1.0

    elif exc.model == "pluck_spectrum":
        p = exc.pluck_position_ratio
        u = s / L
        # Simulation-calibrated shape: amp(u) = A·R(u;p), where the table is
        # normalized so the GLOBAL peak R.max() == 1 (verified against a direct
        # finite-difference wave simulation: for a triangular pluck of height A
        # released from rest, d'Alembert's solution reaches max-over-time
        # excursion exactly A at the pluck point; mid-string swing is lower,
        # e.g. ≈0.61A at p=0.18).  R encodes that true profile shape.
        R = _sim_shape_ratio(u, p)
        amp = A * R
        # enforce exact endpoint nodes (table interp can leave tiny residue)
        amp[0] = 0.0
        amp[-1] = 0.0
        energy_fraction = _modal_energy_fraction(L, p, exc.harmonics)

    elif exc.model == "hi_harmonic":
        m = exc.harmonic_number
        amp = A * np.abs(np.sin(m * math.pi * s / L))
        energy_fraction = 1.0

    else:  # pragma: no cover - validated by pydantic
        raise ValueError(f"unknown excitation model {exc.model}")

    # exact zeros at the segment endpoints (nodes at nut/fret & saddle)
    amp[0] = 0.0
    amp[-1] = 0.0
    return Envelope(s_mm=s, amp_mm=amp, energy_fraction=energy_fraction)
