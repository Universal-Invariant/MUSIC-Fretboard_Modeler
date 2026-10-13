"""Unit helpers for FretLab.

The whole tool is **millimeter-first**: every public field ends in ``_mm``,
``_deg``, or ``_N``.  These helpers exist for import/export convenience when
users measure with rule/masters (inches, mils, thousandths).
"""
from __future__ import annotations

MM_PER_INCH = 25.4
MILS_PER_MM = 1000.0 / MM_PER_INCH  # 39.3701 "thou" per mm


def inch_to_mm(value: float) -> float:
    return value * MM_PER_INCH


def mm_to_inch(value: float) -> float:
    return value / MM_PER_INCH


def mils_to_mm(value: float) -> float:
    """Thousandths of an inch (feeler gauges, feeler 'mils') to mm."""
    return value * MM_PER_INCH / 1000.0


def mm_to_mils(value: float) -> float:
    return value / MM_PER_INCH * 1000.0


def cm_to_mm(value: float) -> float:
    return value * 10.0


def mm_to_cm(value: float) -> float:
    return value / 10.0


def deg_to_rad(value: float) -> float:
    from math import radians

    return radians(value)


def inches_str_to_mm(s: str) -> float:
    """Parse strings like ``'0.010\"'``, ``'1/32'``, ``'2-1/8'`` into mm.

    Handy for pasting gauge-chart values.
    """
    from fractions import Fraction

    s = s.strip().rstrip('"').replace("in", "").strip()
    total = 0.0
    for token in s.replace("-", " ").split():
        try:
            total += float(token)
        except ValueError:
            total += float(Fraction(token))
    return inch_to_mm(total)
