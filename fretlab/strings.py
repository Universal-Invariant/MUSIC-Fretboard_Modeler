"""String specs, linear density, pitch, and static nut→fret→saddle geometry.

The static path of a string is a polyline in the (x, z) side-view plane:

  open position : nut contact ──────────────────────────── saddle contact
  fretted at k  : nut ──(straight over board, static)──► crown of fret k ──► saddle

Fretting at fret *k* shortens the speaking length and moves the forward anchor to
the crown of fret *k*; only the segment fret-k → saddle vibrates.

Centerline vs. underside
------------------------
Configured heights (nut slot, saddle) place the **string centerline** above the
surface it rests on.  Buzz happens when the string *underside* touches a crown,
so every clearance computation uses::

    string_bottom(x) = centerline(x) − gauge/2

:class:`StringPath` stores centerlines; the analysis layer applies the radius
offset via :attr:`StringPhysics.radius_mm`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .config import StringSpec

STEEL_DENSITY = 7_850.0   # kg/m³, plain steel core default
NIICKEL_DENSITY = 8_900.0  # winding material (info only)


@dataclass
class StringPhysics:
    spec: StringSpec
    linear_density_kg_m: float
    frequency_hz: float          # open fundamental on nominal scale length
    omega: float                 # rad/s
    wave_speed_m_s: float        # sqrt(T/μ) — carried for later dispersion work

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def tension_N(self) -> float:
        return self.spec.tension_N

    @property
    def gauge_mm(self) -> float:
        return self.spec.gauge_mm

    @property
    def radius_mm(self) -> float:
        return self.spec.gauge_mm / 2.0


def estimate_linear_density(spec: StringSpec) -> float:
    """kg/m. Direct unit weight wins; else solid-cylinder approximation.

    Wound strings pack lighter than solid metal of the same outer diameter —
    provide ``unit_weight_g_per_m`` (best) or an effective ``density_kg_m3``.
    """
    if spec.unit_weight_g_per_m is not None:
        return spec.unit_weight_g_per_m / 1000.0
    rho = spec.density_kg_m3 or STEEL_DENSITY
    r = spec.gauge_mm / 2000.0  # mm dia → m radius
    return rho * math.pi * r * r


def make_physics(spec: StringSpec, scale_mm: float) -> StringPhysics:
    mu = estimate_linear_density(spec)
    L = scale_mm / 1000.0
    v = math.sqrt(spec.tension_N / mu)
    f = v / (2.0 * L)
    return StringPhysics(
        spec=spec,
        linear_density_kg_m=mu,
        frequency_hz=f,
        omega=2.0 * math.pi * f,
        wave_speed_m_s=v,
    )


# --------------------------------------------------------------------------- #
# Static polyline geometry
# --------------------------------------------------------------------------- #

@dataclass
class StringPath:
    """Endpoints of the vibrating segment plus the static anchor heights.

    ``z0/z1`` are **centerline** heights [mm above the nut reference plane].
    ``fret_index`` identifies the fret wire used as forward anchor (None = open).
    """

    x0_mm: float
    z0_mm: float
    x1_mm: float
    z1_mm: float
    speaking_length_mm: float
    fret_index: int | None = None      # None == open string
    nut_side_z_mm: float | None = None  # centerline height back at the nut (static part)

    def line(self, x: np.ndarray | float) -> np.ndarray:
        """Centerline height z(x) of the straight (un-vibrated) string."""
        x = np.asarray(x, dtype=float)
        t = (x - self.x0_mm) / max(self.x1_mm - self.x0_mm, 1e-9)
        return self.z0_mm + t * (self.z1_mm - self.z0_mm)

    def inside(self, x: np.ndarray) -> np.ndarray:
        return (x >= self.x0_mm - 1e-9) & (x <= self.x1_mm + 1e-9)

    def shifted(self, dz: float) -> "StringPath":
        """Parallel copy raised/lowered by ``dz`` (used by the optimizer sweeps)."""
        return replace(self, z0_mm=self.z0_mm + dz, z1_mm=self.z1_mm + dz,
                       nut_side_z_mm=None if self.nut_side_z_mm is None
                       else self.nut_side_z_mm + dz)


def build_string_path(*, nut_x: float, nut_z: float, saddle_x: float, saddle_z: float,
                      fret_x: float | None = None, fret_crown_z: float | None = None,
                      fret_index: int | None = None,
                      fret_lift_mm: float = 0.0) -> StringPath:
    """Assemble the vibrating-segment polyline for open or fretted state.

    ``fret_lift_mm`` models the small extra stretch when fretting hard behind a
    tall crown (the note goes sharp); zero in the default config, hook kept so
    v0.2 can derive it from crown height & distance-behind-the-wire.
    The returned z0 is the CENTERLINE: crown top + half gauge + lift.
    """
    if fret_x is None:
        return StringPath(x0_mm=nut_x, z0_mm=nut_z, x1_mm=saddle_x, z1_mm=saddle_z,
                          speaking_length_mm=saddle_x - nut_x, fret_index=None,
                          nut_side_z_mm=nut_z)
    z0 = (fret_crown_z or 0.0) + fret_lift_mm
    return StringPath(x0_mm=fret_x, z0_mm=z0, x1_mm=saddle_x, z1_mm=saddle_z,
                      speaking_length_mm=saddle_x - fret_x, fret_index=fret_index,
                      nut_side_z_mm=nut_z)
