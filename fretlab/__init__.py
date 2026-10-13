"""FretLab — virtual luthier fretboard modeler.

Quick start::

    from fretlab import Instrument

    inst = Instrument.from_yaml("config.yaml")
    result = inst.analyze(string="E1")
    print(result.buzz_table())
    result.plot("profile.png")

See README.md for the physics model and configuration schema.
"""
from .analysis import FretCheck, StringAnalysis
from .bow import BowFunction, TrussState, make_bow_fn, resolved_bow, resolve_truss
from .config import FretLabConfig
from .envelope import Envelope, make_envelope
from .geometry import BoardPlane, FretHeights, fret_positions, rule_of_17_positions
from .instrument import Instrument
from .materials import MATERIALS, Material, get_material
from .strings import StringPath, StringPhysics, estimate_linear_density, make_physics
from .units import inch_to_mm, mm_to_inch, mils_to_mm, mm_to_mils

__version__ = "0.1.0"

__all__ = [
    "Instrument",
    "FretLabConfig",
    "StringAnalysis",
    "FretCheck",
    "StringPhysics",
    "StringPath",
    "make_physics",
    "estimate_linear_density",
    "Envelope",
    "make_envelope",
    "BowFunction",
    "TrussState",
    "make_bow_fn",
    "resolved_bow",
    "resolve_truss",
    "BoardPlane",
    "FretHeights",
    "fret_positions",
    "rule_of_17_positions",
    "Material",
    "MATERIALS",
    "get_material",
    "inch_to_mm",
    "mm_to_inch",
    "mils_to_mm",
    "mm_to_mils",
    "__version__",
]
