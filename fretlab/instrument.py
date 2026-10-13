"""Top-level instrument model: assembles geometry + bow + strings + analysis.

This is the main library entry point::

    inst = Instrument.from_yaml("config.yaml")
    result = inst.analyze(string="E1")      # StringAnalysis for one string
    report = inst.report()                   # all strings → tidy DataFrame

``Instrument`` owns nothing but configuration and derived callables; every method
is pure, so what-if sweeps (truss turns, saddle height...) are just config copies.
The resolved bow profile is computed **once** per instance (per string would be
pointless — neck deflection uses total tension) and cached.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import StringAnalysis, analyze_string
from .bow import BowFunction, TrussState, resolved_bow, resolve_truss
from .config import FretLabConfig
from .geometry import (
    BoardPlane,
    FretHeights,
    fret_positions,
    make_board_plane,
    make_fret_height_profile,
)
from .strings import StringPath, StringPhysics, build_string_path, make_physics


@dataclass
class Instrument:
    cfg: FretLabConfig
    fret_x: np.ndarray = field(init=False)
    board: BoardPlane = field(init=False)
    heights: FretHeights = field(init=False)
    physics: dict[str, StringPhysics] = field(init=False)
    bow: BowFunction = field(init=False)
    truss_state: TrussState = field(init=False)

    def __post_init__(self) -> None:
        self.fret_x = fret_positions(self.cfg.instrument)
        self.board = make_board_plane(self.cfg)
        self.heights = make_fret_height_profile(
            self.cfg.fretboard.fret_heights, self.fret_x,
            self.cfg.instrument.scale_length_mm,
        )
        self.physics = {
            s.name: make_physics(s, self.cfg.instrument.scale_length_mm)
            for s in self.cfg.strings
        }
        self.truss_state = resolve_truss(self.cfg, self.total_tension_N)
        self.bow = resolved_bow(self.cfg, self.total_tension_N)

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Instrument":
        return cls(FretLabConfig.from_yaml(path))

    @classmethod
    def from_config(cls, cfg: FretLabConfig) -> "Instrument":
        return cls(cfg)

    def with_mods(self, **dotted_value) -> "Instrument":
        """Copy with changed fields, e.g. ``inst.with_mods(**{"truss.state_turns": 0.5})``."""
        cfg = copy.deepcopy(self.cfg)
        for dotted, value in dotted_value.items():
            obj = cfg
            parts = dotted.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], value)
        return Instrument(FretLabConfig.model_validate(cfg.model_dump()))

    # -- surfaces -------------------------------------------------------------

    @property
    def total_tension_N(self) -> float:
        return sum(s.tension_N for s in self.cfg.strings)

    def crown_z(self, x: np.ndarray | float) -> np.ndarray:
        """Fret-crown line height z(x) [mm above nut reference plane]."""
        x = np.asarray(x, dtype=float)
        return self.board(x) + self.bow(x) + self.heights(x)

    def board_z(self, x: np.ndarray | float) -> np.ndarray:
        """Actual (bowed) fingerboard surface height."""
        x = np.asarray(x, dtype=float)
        return self.board(x) + self.bow(x)

    def nut_anchor(self) -> tuple[float, float]:
        """(x, centerline z) of the string at the nut. Height is measured from
        the LOCAL board surface, so it follows angle/bow automatically."""
        cp = self.cfg.contact_points
        x = cp.nut_position_mm
        z = float(self.board_z(np.array([x]))[0])
        return x, z + cp.nut_height_mm

    def saddle_anchor(self, string_name: str | None = None) -> tuple[float, float]:
        """(x, centerline z) at the saddle, honoring per-string compensation."""
        cp = self.cfg.contact_points
        x = cp.saddle_position_mm + (
            cp.saddle_compensation_mm.get(string_name, 0.0) if string_name else 0.0)
        z = float(self.board_z(np.array([x]))[0])
        return x, z + cp.saddle_height_mm

    def open_path(self, name: str) -> StringPath:
        phys = self.physics[name]
        nut_x, nut_z = self.nut_anchor()
        sad_x, sad_z = self.saddle_anchor(name)
        # nut/saddle configured heights are centerline-above-surface already
        return build_string_path(nut_x=nut_x, nut_z=nut_z,
                                 saddle_x=sad_x, saddle_z=sad_z)

    # -- analysis ---------------------------------------------------------------

    def analyze(self, string: str | None = None, samples: int = 512) -> StringAnalysis:
        spec_name = string or self.cfg.strings[0].name
        phys = self.physics[spec_name]
        return analyze_string(self.cfg, self, phys, self.open_path(spec_name),
                              samples=samples)

    def report(self, samples: int = 256) -> pd.DataFrame:
        rows = []
        for name in self.physics:
            a = self.analyze(name, samples=samples)
            rows.append({
                "string": name,
                "freq_Hz": round(a.physics.frequency_hz, 2),
                "tension_N": a.physics.tension_N,
                "gauge_mm": a.physics.gauge_mm,
                "action_12th_mm": _round(a.action_at_12th_mm, 3),
                "action_last_fret_mm": _round(a.action_last_fret_mm, 3),
                "min_clearance_mm": round(a.min_clearance_mm, 3),
                "worst_fret": a.worst_fret,
                "status": a.status,
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        c = self.cfg
        return {
            "name": c.instrument.name,
            "family": c.instrument.family,
            "scale_mm": c.instrument.scale_length_mm,
            "frets": c.instrument.num_frets,
            "total_tension_N": round(self.total_tension_N, 1),
            "effective_relief_mm": round(self.truss_state.effective_relief_mm, 3),
            "truss_note": self.truss_state.note,
            "bow_model": self.bow.model,
        }


def _round(v: float, n: int):
    return None if not np.isfinite(v) else round(float(v), n)
