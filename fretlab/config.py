"""Validated configuration schemas for FretLab.

All geometry is millimeters, angles in degrees, tension in newtons.  Configs load
from YAML or JSON and are validated with pydantic v2 before any modeling runs —
a bad setup file should fail loudly with a readable message, never silently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

BowModel = Literal["flat", "cubic", "constant_curvature", "circular_arc", "data", "custom"]
FretPosModel = Literal["rule_of_17", "true_temperament", "custom_positions", "data"]
FretHeightModel = Literal["uniform", "taper", "data", "custom"]
ExcitationModel = Literal["fundamental", "pluck_spectrum", "hi_harmonic"]
InstrumentFamily = Literal["guitar", "bass", "violin", "viola", "cello", "double_bass",
                          "mandolin", "ukulele", "fretless", "custom"]


class CurveSource(BaseModel):
    """Common fields for 'data' driven curves (CSV paths)."""

    csv: Optional[Path] = None
    # inline fallback data so configs stay self-contained: [(x_mm, y_mm), ...]
    points: list[tuple[float, float]] = Field(default_factory=list)


class BowConfig(BaseModel):
    model: BowModel = "cubic"
    max_relief_mm: float = Field(0.25, ge=-2.0, le=5.0,
                                 description="Signed peak relief (+ = concave up/relief).")
    radius_mm: Optional[float] = Field(None, description="For constant_curvature / circular_arc.")
    expr: Optional[str] = Field(None, description="Custom safe-eval expression in x (mm from nut).")
    data: CurveSource = Field(default_factory=CurveSource)

    @model_validator(mode="after")
    def _check(self) -> "BowConfig":
        if self.model == "custom" and not self.expr:
            raise ValueError("bow.model='custom' requires bow.expr")
        if self.model == "data":
            data = self.data
            ok = False
            if isinstance(data, list):
                ok = len(data) >= 2
            elif data is not None:
                ok = bool(getattr(data, "csv", None)) or bool(getattr(data, "points", None))
            if not ok:
                raise ValueError(
                    "bow.model='data' requires bow.data.csv, bow.data.points, "
                    "or a bare list of [x_mm, dz_mm] pairs"
                )
        if self.model in ("constant_curvature", "circular_arc") and self.radius_mm is None \
                and self.max_relief_mm is None:
            raise ValueError(f"bow.model='{self.model}' needs radius_mm")
        return self


class FretHeightConfig(BaseModel):
    model: FretHeightModel = "uniform"
    height_mm: float = Field(1.15, ge=0.0, le=3.0, description="Uniform crown height above board.")
    height_at_nut_mm: Optional[float] = None
    height_at_last_fret_mm: Optional[float] = None  # taper ramp endpoints
    expr: Optional[str] = None
    data: CurveSource = Field(default_factory=CurveSource)

    @model_validator(mode="after")
    def _check(self) -> "FretHeightConfig":
        if self.model == "taper" and (
            self.height_at_nut_mm is None or self.height_at_last_fret_mm is None
        ):
            raise ValueError("fret_heights.model='taper' needs both endpoint heights")
        if self.model == "custom" and not self.expr:
            raise ValueError("fret_heights.model='custom' requires expr")
        if self.model == "data" and not (self.data.csv or self.data.points):
            raise ValueError("fret_heights.model='data' needs data.csv or data.points")
        return self


class InstrumentConfig(BaseModel):
    name: str = "Untitled instrument"
    family: InstrumentFamily = "guitar"
    scale_length_mm: float = Field(..., gt=100.0, lt=2000.0)
    num_frets: int = Field(22, ge=0, le=36)
    board_angle_deg: float = Field(
        0.0, ge=-5.0, le=5.0,
        description="Angle of the fingerboard plane; + tips the headstock side up "
                    "(equivalent to more neck pitch on an acoustic).",
    )
    neck_fall_mm: float = Field(
        0.0, ge=-2.0, le=5.0,
        description="Depth the board drops past the last fret (heel carve / overstand fall).",
    )
    custom_fret_positions_mm: list[float] = Field(default_factory=list)
    fret_position_model: FretPosModel = "rule_of_17"
    octave_fret: int = Field(
        12, ge=6, le=24,
        description="Fret number that sits at half the scale length (12 for guitars; "
                    "some historical/fanned layouts differ).",
    )
    fret_data: CurveSource = Field(
        default_factory=CurveSource,
        description="CSV/points for fret_position_model='data' (columns dist_mm,height_mm).",
    )

    @model_validator(mode="after")
    def _check(self) -> "InstrumentConfig":
        if self.fret_position_model == "custom_positions" and len(
            self.custom_fret_positions_mm
        ) != self.num_frets:
            raise ValueError(
                "custom_fret_positions_mm must have exactly num_frets entries"
            )
        if self.fret_position_model == "data" and not (
            self.fret_data.csv or self.fret_data.points
        ):
            raise ValueError("fret_position_model='data' needs fret_data.csv or points")
        return self


class FretboardConfig(BaseModel):
    length_mm: Optional[float] = Field(
        None, description="Physical board length; defaults to scale + saddle extension."
    )
    width_at_nut_mm: float = Field(42.8, gt=10.0, le=120.0)
    width_at_last_fret_mm: float = Field(56.0, gt=10.0, le=140.0)
    thickness_mm: float = Field(6.0, gt=1.0, le=30.0, description="Board + veneer build-up.")
    material: str = "rosewood"
    youngs_modulus_mpa: Optional[float] = None  # override preset
    moment_of_inertia_mm4: Optional[float] = None  # override beam-section estimate
    bow: BowConfig = Field(default_factory=BowConfig)
    fret_heights: FretHeightConfig = Field(default_factory=FretHeightConfig)


class TrussConfig(BaseModel):
    state_turns: float = Field(
        0.0, ge=-10.0, le=10.0,
        description="Zero = calibrated reference position. + turns tighten (reduce relief).",
    )
    truss_stiffness: float = Field(
        -0.10,
        description="Relief change per full turn [mm/turn]. Negative = tightening "
                    "reduces relief (typical ≈ −0.05…−0.15 mm/turn).",
    )
    string_load_coefficient: float = Field(
        0.0, ge=0.0,
        description="Extra relief added per newton of tension *above the tuned "
                    "baseline* ``tune_tension_N`` [mm/N]. Keep 0 when "
                    "``fretboard.bow`` already describes the neck **at tune** "
                    "(the usual case); set >0 only for untensioned baseline models.",
    )
    tune_tension_N: float = Field(
        0.0, ge=0.0,
        description="Total string tension [N] that the configured bow shape "
                    "corresponds to (default 0 ⇒ the shape is read as the "
                    "at-tune state and the load term stays inactive unless you "
                    "also set string_load_coefficient with a nonzero baseline).",
    )
    solve_mode: Literal["manual", "target_relief"] = "manual"
    target_relief_mm: Optional[float] = Field(
        None, description="When solve_mode='target_relief', turns are solved to hit this."
    )


class StringSpec(BaseModel):
    name: str
    gauge_mm: float = Field(..., gt=0.02, lt=3.0, description="Plain/wound outer diameter.")
    tension_N: float = Field(..., gt=1.0, lt=400.0)
    lateral_mm: float = Field(0.0, description="Offset from board centerline (fan/twist later).")
    density_kg_m3: Optional[float] = Field(
        None, description="Override for linear density estimate (wound strings pack lighter)."
    )
    unit_weight_g_per_m: Optional[float] = Field(
        None, description="Direct linear density input if known (takes precedence)."
    )
    note: str = ""

    @field_validator("name")
    @classmethod
    def _uniq_chars(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("string name must be non-empty")
        return v.strip()


class ContactPointConfig(BaseModel):
    nut_height_mm: float = Field(
        1.3, ge=0.0, le=10.0,
        description="String centerline height above the LOCAL board surface at the nut "
                    "(before slotting; subtract slot depth for finished value).",
    )
    nut_position_mm: float = Field(0.0, ge=-5.0, le=20.0)
    saddle_height_mm: float = Field(
        3.2, ge=0.0, le=25.0,
        description="String centerline height above the board line at the saddle.",
    )
    saddle_position_mm: Optional[float] = Field(
        None, description="Default: scale_length + typical compensation."
    )
    saddle_compensation_mm: dict[str, float] = Field(
        default_factory=dict,
        description="Per-string forward offset of the saddle contact point beyond "
                    "``saddle_position_mm`` (wound strings intonate further back).",
    )
    nut_slot_depth_mm: Optional[float] = Field(
        None, description="Optional convenience: if set, nut_height is measured from the "
                          "slot floor (finished slot height above board = this value).",
    )
    action_at_12th_mm: Optional[float] = Field(
        None, description="Convenience target: if set and saddle_height omitted-ish, "
                          "saddle height is back-solved to produce this open-string action."
    )


class ExcitationConfig(BaseModel):
    model: ExcitationModel = "pluck_spectrum"
    pluck_amplitude_mm: float = Field(
        1.2, gt=0.0, le=15.0,
        description="Pluck deflection A (mm): the peak of the vibration "
                    "envelope equals A for pluck_spectrum/fundamental models "
                    "(d'Alembert); total travel is ~2A peak-to-peak.")
    pluck_position_ratio: float = Field(0.18, gt=0.0, lt=1.0)
    harmonics: int = Field(12, ge=1, le=64)
    harmonic_number: int = Field(
        4, ge=1, le=64, description="For hi_harmonic buzz test (default 4th ring-out)."
    )


class BuzzConfig(BaseModel):
    warn_clearance_mm: float = Field(0.25, ge=0.0)
    buzz_clearance_mm: float = Field(0.12, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "BuzzConfig":
        if self.buzz_clearance_mm > self.warn_clearance_mm:
            raise ValueError("buzz_clearance_mm must be <= warn_clearance_mm")
        return self


class FretLabConfig(BaseModel):
    """Top-level document."""

    version: str = "0.1"
    instrument: InstrumentConfig
    fretboard: FretboardConfig = Field(default_factory=FretboardConfig)
    truss: TrussConfig = Field(default_factory=TrussConfig)
    strings: list[StringSpec] = Field(default_factory=list)
    contact_points: ContactPointConfig = Field(default_factory=ContactPointConfig)
    excitation: ExcitationConfig = Field(default_factory=ExcitationConfig)
    buzz: BuzzConfig = Field(default_factory=BuzzConfig)

    @model_validator(mode="after")
    def _defaults(self) -> "FretLabConfig":
        if self.contact_points.saddle_position_mm is None:
            # light compensation default; real compensation is per-string in v0.2
            self.contact_points.saddle_position_mm = self.instrument.scale_length_mm * 1.015
        if not self.strings:
            raise ValueError("at least one string must be configured")
        names = [s.name for s in self.strings]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate string names: {names}")
        return self

    # -- IO helpers -----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FretLabConfig":
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "FretLabConfig":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load(cls, path: str | Path) -> "FretLabConfig":
        p = Path(path)
        return cls.from_json(p) if p.suffix.lower() in (".json",) else cls.from_yaml(p)

    def to_yaml(self, path: str | Path) -> None:
        data = self.model_dump(mode="json")
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
