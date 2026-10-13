"""Board / neck material presets.

Informational in v0.1 (used for the linearized truss stiffness estimate and for
reporting).  Values are typical, not measured — users may override ``youngs_modulus``
and ``density`` per instrument in their config.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    youngs_modulus_mpa: float  # along the grain
    density_kg_m3: float
    notes: str = ""


MATERIALS: dict[str, Material] = {
    "ebony": Material("ebony", 13_500, 1_050, "Dense; classic fingerboard."),
    "rosewood": Material("rosewood", 11_000, 830, "Indian/Brazilian fretboard wood."),
    "pau-ferro": Material("pau-ferro", 11_500, 870, "Modern rosewood substitute."),
    "maple": Material("maple", 10_000, 705, "One-piece necks / veneered boards."),
    "mahogany": Material("mahogany", 9_000, 550, "Neck blanks."),
    "walnut": Material("walnut", 10_500, 660, ""),
    "ekang": Material("ekang", 12_000, 900, "Bass boards."),
    "corian": Material("corian", 4_500, 1_350, "Synthetic boards."),
    "carbon-fiber": Material("carbon-fiber", 45_000, 1_550, "Composite necks (Rickenbacker-style)."),
    "aluminum": Material("aluminum", 69_000, 2_700, "Dobro/neck-resonator."),
    "steel-truss": Material("steel-truss", 200_000, 7_850, "Truss rod material."),
}


def get_material(name: str) -> Material:
    key = name.lower().strip()
    if key not in MATERIALS:
        raise KeyError(
            f"Unknown material '{name}'. Known: {', '.join(sorted(MATERIALS))}. "
            "You can also specify youngs_modulus_mpa/density_kg_m3 directly."
        )
    return MATERIALS[key]
