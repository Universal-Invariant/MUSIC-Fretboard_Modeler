"""Clearance & buzz analysis: the payoff of the whole model.

For one string we evaluate two classic shop tests at every fret:

1. **Open-string test** — envelope of the open string (nut→saddle) vs. every crown:
   ``clearance = min over x near fret f of [center(x) − amp(x) − r − crown(x)]``
   where ``r`` is the string radius (buzz happens underside-first).

2. **Fretted-at-f test** — vibrating segment starts at the crown of fret *f* and
   runs to the saddle; we measure its envelope against the *next* crown (f+1).
   This is exactly what happens when you fret a note and check for buzz at the
   following wire.

The minimum over all frets is the string's ``min_clearance_mm``; compared against
the configured thresholds it yields OK / MARGINAL / BUZZ.  Heuristic advisories
translate a shortfall into "raise saddle by ~Δ" or "add relief" suggestions;
a real inverse-solving optimizer lives in :mod:`fretlab.optimizer`.

Fretless instruments (violin family): there are no crowns; the obstacle is the
board surface itself, checked in the open state only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FretLabConfig
from .envelope import Envelope, make_envelope
from .strings import StringPath, StringPhysics


@dataclass
class FretCheck:
    fret: int                 # 0 == nut row, 1..N wires
    x_mm: float
    crown_z_mm: float         # crown top height (absolute z)
    open_clearance_mm: float | None
    fretted_clearance_mm: float | None   # this fret vs. NEXT fret when fretting here
    status: str

    @property
    def clearance_mm(self) -> float | None:
        vals = [v for v in (self.open_clearance_mm, self.fretted_clearance_mm)
                if v is not None]
        return min(vals) if vals else None


@dataclass
class StringAnalysis:
    physics: StringPhysics
    checks: list[FretCheck]
    open_path: StringPath
    open_env: Envelope
    fretted_paths: dict[int, tuple[StringPath, Envelope]] = field(default_factory=dict)
    action_at_12th_mm: float = float("nan")
    action_last_fret_mm: float = float("nan")
    advisories: list[str] = field(default_factory=list)
    fretless: bool = False
    board_min_gap_mm: float | None = None   # fretless: envelope vs board surface
    model_notes: list[str] = field(default_factory=list)  # e.g. spectrum truncation

    # -- summaries -----------------------------------------------------------

    @property
    def min_clearance_mm(self) -> float:
        vals = [c.clearance_mm for c in self.checks if c.clearance_mm is not None]
        if self.board_min_gap_mm is not None:
            vals.append(self.board_min_gap_mm)
        return min(vals) if vals else float("inf")

    @property
    def worst_fret(self) -> int:
        scored = [c for c in self.checks if c.clearance_mm is not None]
        if not scored:
            return -1
        return min(scored, key=lambda c: c.clearance_mm).fret

    @property
    def status(self) -> str:
        m = self.min_clearance_mm
        if m < getattr(self, "_buzz_mm", 0.12):
            return "BUZZ"
        if m < getattr(self, "_warn_mm", 0.25):
            return "MARGINAL"
        return "OK"

    def configure_thresholds(self, buzz_mm: float, warn_mm: float) -> None:
        self._buzz_mm, self._warn_mm = buzz_mm, warn_mm

    def buzz_table(self) -> pd.DataFrame:
        rows = []
        for c in self.checks:
            rows.append({
                "fret": c.fret,
                "x_mm": round(c.x_mm, 2),
                "crown_z_mm": round(c.crown_z_mm, 3),
                "open_clear_mm": None if c.open_clearance_mm is None
                                 else round(c.open_clearance_mm, 3),
                "fretted_clear_mm": None if c.fretted_clearance_mm is None
                                    else round(c.fretted_clearance_mm, 3),
                "status": c.status,
            })
        return pd.DataFrame(rows)

    def plot(self, path: str = "profile.png", **kw):
        from .plotting import plot_string_profile

        return plot_string_profile(self, path=path, **kw)


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #

def _classify(cfg: FretLabConfig, clearance: float | None) -> str:
    if clearance is None:
        return "n/a"
    if clearance < cfg.buzz.buzz_clearance_mm:
        return "BUZZ"
    if clearance < cfg.buzz.warn_clearance_mm:
        return "MARGINAL"
    return "OK"


def gap_along(path: StringPath, env: Envelope, instrument, radius_mm: float,
              obstacle="crown", fret_x: np.ndarray | None = None):
    """Return (xs, gaps) where gaps = string underside lowest swing − obstacle z.

    When checking against fret crowns we evaluate the crown line on a grid that
    **includes every wire position** inside the segment; otherwise a coarse
    envelope sampling can step right over a tall crown between two samples and
    report a phantom clearance (or vice versa).
    """
    xs = path.x0_mm + env.s_mm
    center = path.line(xs)
    if obstacle == "crown":
        obs_x = xs
        if fret_x is not None and len(fret_x):
            inside = fret_x[(fret_x >= xs.min() - 1e-9) & (fret_x <= xs.max() + 1e-9)]
            if len(inside):
                obs_x = np.sort(np.concatenate([xs, inside]))
        obs = instrument.crown_z(obs_x)
        if obs_x is not xs:
            # re-evaluate string geometry on the denser grid for a pointwise gap
            center = path.line(obs_x)
            amp = np.interp(obs_x - path.x0_mm, env.s_mm, env.amp_mm)
            xs, env_amp, obs = obs_x, amp, obs
            return xs, center - env_amp - radius_mm - obs
    else:
        obs = instrument.board_z(xs)
    gaps = center - env.amp_mm - radius_mm - obs
    return xs, gaps


def analyze_string(
    cfg: FretLabConfig,
    instrument,                      # Instrument (duck-typed to avoid circular import)
    phys: StringPhysics,
    open_path: StringPath,
    samples: int = 512,
) -> StringAnalysis:
    """Run open + fretted envelope-vs-crown checks for one string."""
    fret_x = instrument.fret_x
    n_frets = len(fret_x)
    exc = cfg.excitation
    r = phys.radius_mm

    # ---- open string envelope ----------------------------------------------
    open_env = make_envelope(exc, open_path.speaking_length_mm, samples=samples)
    xs, gap = gap_along(open_path, open_env, instrument, r, fret_x=fret_x)

    def local_min_gap(x_target: float, half_width: float) -> float | None:
        m = np.abs(xs - x_target) <= max(half_width, 1.0)
        if not m.any():
            return None
        v = float(gap[m].min())
        return v if np.isfinite(v) else None

    checks: list[FretCheck] = []

    # nut row (fret 0) — anchor info only
    nut_x, nut_z = instrument.nut_anchor()
    checks.append(FretCheck(
        fret=0, x_mm=nut_x, crown_z_mm=nut_z,
        open_clearance_mm=None, fretted_clearance_mm=None, status="n/a",
    ))

    hw = 2.0  # mm evaluation window around each crown (half a typical fret width)
    fretted_paths: dict[int, tuple[StringPath, Envelope]] = {}
    for i, fx in enumerate(fret_x, start=1):
        cz = float(instrument.crown_z(np.array([fx]))[0])
        oc = local_min_gap(float(fx), hw)

        fc: float | None = None
        if i < n_frets:
            # fretted AT fret i → vibrating from crown i (+radius) to saddle;
            # check crown i+1
            nxt_fx = float(fret_x[i])
            fp = _fretted_path(open_path, float(fx), cz + r, fret_index=i)
            fe = make_envelope(exc, fp.speaking_length_mm, samples=samples)
            fretted_paths[i] = (fp, fe)
            fxs, fgap = gap_along(fp, fe, instrument, r, fret_x=fret_x)
            m = np.abs(fxs - nxt_fx) <= hw
            if m.any():
                v = float(fgap[m].min())
                fc = v if np.isfinite(v) else None

        checks.append(FretCheck(
            fret=i, x_mm=float(fx), crown_z_mm=cz,
            open_clearance_mm=oc,
            fretted_clearance_mm=fc,
            status=_classify(cfg, _min_or_none(oc, fc)),
        ))

    # ---- actions (centerline − crown top, shop convention) -------------------
    def action_at(fx: float) -> float:
        cz = float(instrument.crown_z(np.array([fx]))[0])
        return float(open_path.line(np.array([fx]))[0] - cz)

    octave_x = cfg.instrument.scale_length_mm / 2.0
    action12 = action_at(octave_x) if n_frets >= 12 else float("nan")
    action_last = action_at(float(fret_x[-1])) if n_frets else float("nan")

    # ---- fretless fallback: envelope vs. board surface -----------------------
    board_gap: float | None = None
    if n_frets == 0:
        bxs, bgap = gap_along(open_path, open_env, instrument, r, obstacle="board")
        board_gap = float(bgap.min()) if len(bgap) else None

    result = StringAnalysis(
        physics=phys, checks=checks, open_path=open_path, open_env=open_env,
        fretted_paths=fretted_paths,
        action_at_12th_mm=action12, action_last_fret_mm=action_last,
        fretless=(n_frets == 0), board_min_gap_mm=board_gap,
    )
    if exc.model == "pluck_spectrum" and open_env.energy_fraction < 0.95:
        result.model_notes.append(
            f"pluck_spectrum truncated at {exc.harmonics} harmonics captures only "
            f"{open_env.energy_fraction:.0%} of ideal pluck energy — raise "
            "'harmonics' for sharper-pluck accuracy (peak envelope near the "
            "fretted/plucked region may be underestimated)")
    result._instrument = instrument  # convenience for .plot() without extra args
    result.configure_thresholds(cfg.buzz.buzz_clearance_mm, cfg.buzz.warn_clearance_mm)
    result.advisories = _advisories(cfg, result)
    return result


def _min_or_none(a, b):
    vals = [v for v in (a, b) if v is not None and np.isfinite(v)]
    return min(vals) if vals else None


def _fretted_path(open_path: StringPath, fret_x: float, crown_center_z: float,
                  fret_index: int, fret_lift_mm: float = 0.0) -> StringPath:
    """Vibrating segment from the crown of ``fret_x`` to the saddle.

    ``crown_center_z`` is already crown-top + string radius; ``fret_lift_mm`` is
    the future stretch-lift hook (zero today).
    """
    return StringPath(
        x0_mm=fret_x, z0_mm=crown_center_z + fret_lift_mm,
        x1_mm=open_path.x1_mm, z1_mm=open_path.z1_mm,
        speaking_length_mm=open_path.x1_mm - fret_x,
        fret_index=fret_index,
        nut_side_z_mm=open_path.z0_mm,
    )


def _advisories(cfg: FretLabConfig, res: StringAnalysis) -> list[str]:
    """Heuristic shop advice from the numbers (exact solver: fretlab.optimizer)."""
    adv: list[str] = []
    m = res.min_clearance_mm
    target = cfg.buzz.warn_clearance_mm
    if m < target:
        deficit = target - m
        L = cfg.instrument.scale_length_mm
        sad = cfg.contact_points.saddle_position_mm or (L * 1.015)
        x_worst = next((c.x_mm for c in res.checks if c.fret == res.worst_fret), L / 2)
        gain = (sad - x_worst) / max(sad - cfg.contact_points.nut_position_mm, 1e-9)
        if gain > 0.05:
            d_saddle = deficit / gain
            adv.append(f"raise saddle ≈{d_saddle:.2f} mm to clear worst fret "
                       f"(#{res.worst_fret}) by {target:.2f} mm")
        k = cfg.truss.truss_stiffness          # signed mm of relief per turn
        d_relief = deficit * 0.5               # relief buys ~half its depth in gap
        if abs(k) > 1e-9:
            turns = -d_relief / k              # + turns tighten ⇒ negative to loosen
            adv.append(f"or add ≈{d_relief:.2f} mm relief "
                       f"(≈{abs(turns):.2f} turn{'s' if abs(turns) != 1 else ''} "
                       f"{'loose' if turns < 0 else 'tight'})")
        else:
            adv.append(f"or add ≈{d_relief:.2f} mm relief via truss "
                       "(calibrate truss_stiffness [mm/turn] to get turns)")
    a12 = res.action_at_12th_mm
    if np.isfinite(a12):
        fam = cfg.instrument.family
        hi = {"guitar": 2.4, "bass": 2.8}.get(fam, 3.0)
        if a12 > hi:
            adv.append(f"action @12 = {a12:.2f} mm is high for {fam}; "
                       "consider lowering after fixing buzz")
        if a12 < 1.0 and fam in ("guitar", "bass"):
            adv.append(f"action @12 = {a12:.2f} mm is very low; "
                       "expect buzz on hard picking")
    if res.fretless and res.board_min_gap_mm is not None \
            and res.board_min_gap_mm < cfg.buzz.buzz_clearance_mm:
        adv.append("fretless: string envelope intersects the board — raise bridge "
                   "side or reduce pluck amplitude assumption")
    return adv
