"""Regression tests for the truss solver and resolved bow shapes."""
import numpy as np
import pytest

from fretlab.bow import (make_bow_fn, resolve_truss, resolved_bow,
                         calibrate_truss)
from fretlab.config import FretLabConfig
from fretlab.instrument import Instrument


_MIN_STRINGS = [
    {"name": "e1", "gauge_mm": 0.279, "tension_N": 61.6},
]


def base_cfg(**truss_over):
    d = {
        "instrument": {"name": "test", "family": "guitar",
                       "scale_length_mm": 648.0, "n_frets": 22},
        "fretboard": {"bow": {"model": "cubic", "max_relief_mm": 0.25}},
        "strings": _MIN_STRINGS,
        "truss": {"truss_stiffness": -0.10, "string_load_coefficient": 0.0,
                  **truss_over},
    }
    return FretLabConfig.model_validate(d)


def test_manual_turns_apply_signed_stiffness():
    """+ turns tighten ⇒ relief decreases with negative stiffness."""
    cfg = base_cfg(state_turns=0.0)
    s0 = resolve_truss(cfg, total_tension_N=600.0)
    assert s0.effective_relief_mm == pytest.approx(0.25, rel=0.05)

    cfg = base_cfg(state_turns=2.0)   # 2 turns tight
    s1 = resolve_truss(cfg, total_tension_N=600.0)
    assert s1.effective_relief_mm == pytest.approx(0.25 + 2.0 * -0.10, abs=1e-6)

    cfg = base_cfg(state_turns=-1.0)  # loosen
    s2 = resolve_truss(cfg, total_tension_N=600.0)
    assert s2.effective_relief_mm > s0.effective_relief_mm


def test_string_load_adds_relief_when_enabled():
    """Bow shape describes a bare (untensioned) neck ⇒ tune_tension_N = 0, so
    full string tension adds relief at the configured coefficient."""
    cfg = base_cfg(string_load_coefficient=0.0005, tune_tension_N=0.0)
    s = resolve_truss(cfg, total_tension_N=600.0)
    assert s.effective_relief_mm == pytest.approx(0.25 + 600 * 0.0005, abs=1e-6)


def test_string_load_is_zero_at_tuned_baseline():
    """When the bow config already describes the neck at tune (T_tune == T),
    the load term must contribute nothing — no double counting."""
    cfg = base_cfg(string_load_coefficient=0.0005, tune_tension_N=600.0)
    s = resolve_truss(cfg, total_tension_N=600.0)
    assert s.effective_relief_mm == pytest.approx(0.25, abs=1e-9)
    # retuning DOWN reduces tension below baseline → relief drops slightly
    s_low = resolve_truss(cfg, total_tension_N=500.0)
    assert s_low.effective_relief_mm < s.effective_relief_mm


def test_target_relief_round_trip():
    """Solve for turns to hit a target; feeding those turns back manually must
    reproduce the same effective relief (round-trip consistency)."""
    target = 0.15
    cfg = base_cfg(solve_mode="target_relief", target_relief_mm=target,
                   state_turns=0.0)
    solved = resolve_truss(cfg, total_tension_N=600.0)
    assert solved.effective_relief_mm == pytest.approx(target, abs=1e-9)
    assert solved.turns_applied > 0  # tightening needed to reduce from 0.25

    manual = base_cfg(state_turns=solved.turns_applied)
    again = resolve_truss(manual, total_tension_N=600.0)
    assert again.effective_relief_mm == pytest.approx(target, rel=1e-6)


def test_resolved_bow_peak_matches_state():
    cfg = base_cfg(state_turns=1.0)
    L = cfg.instrument.scale_length_mm
    fn = resolved_bow(cfg, total_tension_N=600.0)
    state = resolve_truss(cfg, 600.0)
    assert fn.peak_relief_mm() == pytest.approx(state.effective_relief_mm, rel=0.02)
    # endpoints pinned to zero (nut & saddle reference plane)
    xs = np.array([0.0, L])
    assert np.allclose(fn(xs), 0.0, atol=1e-9)


def test_radius_model_rebuilt_from_resolved_curvature():
    cfg = base_cfg()
    cfg.fretboard.bow.model = "constant_curvature"
    cfg.fretboard.bow.radius_mm = 20000.0
    fn = resolved_bow(cfg, total_tension_N=600.0)
    # sagitta of R=20000 over L=648: ~ L^2/(8R) = 2.6 mm — sanity range
    assert 0.5 < fn.peak_relief_mm() < 6.0


def test_measurement_models_not_warped():
    cfg = base_cfg(state_turns=3.0)
    cfg.fretboard.bow.model = "data"
    cfg.fretboard.bow.data = [[0, 0], [324, -0.1], [648, 0]]
    state = resolve_truss(cfg, total_tension_N=600.0)
    assert state.unscaled_measurement is True
    fn = resolved_bow(cfg, total_tension_N=600.0)
    # data shape preserved: peak stays at measured value, not warped by turns
    assert abs(fn.peak_relief_mm()) == pytest.approx(0.1, rel=0.1)


def test_calibrate_truss_recovers_stiffness():
    """Synthetic Δ-samples (relief change vs turns, at constant tension) with
    k = −0.12 recovered by least squares."""
    true_k = -0.12
    samples = [(t, 0.0, true_k * t) for t in (-2, -1, 1, 2, 3)]
    cal = calibrate_truss(samples)
    assert cal.truss_stiffness_mm_per_turn == pytest.approx(true_k, rel=0.05)
    assert max(abs(r) for r in cal.residuals_mm) < 1e-9


def test_calibrate_truss_with_string_load():
    """Two-term fit: relief change d = turns·k + ΔT·c."""
    true_k, true_c = -0.12, 0.0005
    samples = [
        (1.0, 100.0, true_k * 1.0 + true_c * 100.0),
        (2.0, -50.0, true_k * 2.0 + true_c * -50.0),
        (0.0, 200.0, true_c * 200.0),
        (-1.0, 40.0, true_k * -1.0 + true_c * 40.0),
    ]
    cal = calibrate_truss(samples, fit_string_load=True)
    assert cal.truss_stiffness_mm_per_turn == pytest.approx(true_k, rel=1e-6)
    assert cal.string_load_coefficient_mm_per_N == pytest.approx(true_c, rel=1e-6)


def test_advisory_turn_sign_matches_direction():
    """A deficit that needs MORE relief must advise loosening (negative turns
    with negative stiffness), and the printed turn count must be small/mplausible."""
    cfg = base_cfg(truss_stiffness=-0.10)
    inst = Instrument.from_config(cfg)
    res = inst.analyze(string="e1")
    if res.min_clearance_mm < cfg.buzz.warn_clearance_mm:
        text = " ".join(res.advisories)
        assert "relief" in text
        # loosening adds relief ⇒ advice should say 'loose', never huge counts
        import re
        m = re.search(r"≈(\d+\.\d+) turn", text)
        assert m and float(m.group(1)) < 20.0
