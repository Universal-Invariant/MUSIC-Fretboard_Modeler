"""Regression tests for the peak-displacement envelopes."""
import numpy as np
import pytest

from fretlab.config import ExcitationConfig
from fretlab.envelope import make_envelope, worst_case_modal_envelope


def test_fundamental_peak_and_nodes():
    exc = ExcitationConfig(model="fundamental", pluck_amplitude_mm=2.0)
    env = make_envelope(exc, 648.0, samples=1001)
    assert env.peak_mm == pytest.approx(2.0, rel=1e-3)
    # antinode at mid-length
    i = int(np.argmax(env.amp_mm))
    assert env.s_mm[i] == pytest.approx(324.0, abs=1.0)
    assert env.amp_mm[0] == 0.0 and env.amp_mm[-1] == 0.0


@pytest.mark.parametrize("p", [0.18, 0.33, 0.5, 0.75])
def test_pluck_spectrum_global_peak_equals_A(p):
    """Physics (exact d'Alembert solution, verified by direct wave simulation):
    a triangular pluck of height A released from rest reaches max-over-time
    excursion exactly A at the pluck point.  The calibrated envelope must have
    global peak == A for every pluck position."""
    A = 1.7
    exc = ExcitationConfig(model="pluck_spectrum", pluck_amplitude_mm=A,
                           pluck_position_ratio=p, harmonics=12)
    env = make_envelope(exc, 648.0, samples=2049)
    # rel=1e-2: the coarse p-grid (0.05 steps) means bilinear interpolation
    # slightly attenuates the sharp apex for plucks between grid rows.
    assert env.peak_mm == pytest.approx(A, rel=1e-2)
    assert env.amp_mm[0] == 0.0 and env.amp_mm[-1] == 0.0
    assert (env.amp_mm >= 0).all()


@pytest.mark.parametrize("p,A,mid_rel", [
    (0.18, 1.0, 0.61),   # mid-string swing < A when plucked off-center
    (0.33, 1.0, 0.75),
    (0.50, 1.0, 1.00),   # centered pluck: peak IS at midpoint
])
def test_pluck_spectrum_midpoint_swing_matches_simulation(p, A, mid_rel):
    """Mid-string excursion must match the simulated ratio R(0.5; p) ≈ known
    values (0.61A at p=0.18 etc.), guarding table/model mismatches."""
    exc = ExcitationConfig(model="pluck_spectrum", pluck_amplitude_mm=A,
                           pluck_position_ratio=p)
    env = make_envelope(exc, 648.0, samples=2049)
    mid = float(np.interp(324.0, env.s_mm, env.amp_mm))
    assert mid == pytest.approx(A * mid_rel, rel=0.02)


@pytest.mark.parametrize("p", [0.10, 0.25, 0.5])
def test_pluck_spectrum_table_normalized_to_global_peak(p):
    """The shipped table rows must be normalized so max_u R(u;p) == 1; the
    envelope peak is then exactly A.  Recomputed from the table so a silent
    regeneration with wrong normalization fails loudly."""
    from fretlab.envelope import _sim_shape_ratio
    exc = ExcitationConfig(model="pluck_spectrum", pluck_amplitude_mm=1.2,
                           pluck_position_ratio=p)
    env = make_envelope(exc, 648.0, samples=2049)
    u = env.s_mm / 648.0
    expected = 1.2 * float(_sim_shape_ratio(u, p).max())
    assert env.peak_mm == pytest.approx(expected, rel=1e-3)
    assert float(_sim_shape_ratio(u, p).max()) == pytest.approx(1.0, rel=1e-3)


def test_pluck_spectrum_apex_near_pluck_point_for_low_p():
    """Physics (validated by direct wave simulation): for plucks near the nut
    the max excursion sits at the pluck point, NOT mid-string, and exceeds the
    midpoint swing by ~1.5–1.8×.  Guards against regressing to the old
    'always peak at L/2' approximation."""
    exc = ExcitationConfig(model="pluck_spectrum", pluck_amplitude_mm=2.0,
                           pluck_position_ratio=0.15)
    env = make_envelope(exc, 648.0, samples=2049)
    i = int(np.argmax(env.amp_mm))
    assert env.s_mm[i] == pytest.approx(0.15 * 648.0, rel=0.05)
    mid = float(np.interp(324.0, env.s_mm, env.amp_mm))
    assert env.peak_mm > 1.4 * mid


def test_pluck_spectrum_centered_peaks_at_midpoint():
    for p in (0.5,):
        exc = ExcitationConfig(model="pluck_spectrum", pluck_amplitude_mm=2.0,
                               pluck_position_ratio=p)
        env = make_envelope(exc, 648.0, samples=2049)
        i = int(np.argmax(env.amp_mm))
        assert env.s_mm[i] == pytest.approx(324.0, rel=0.01)


def test_pluck_spectrum_symmetry_in_p():
    """Plucking at p and 1−p produce identical envelopes."""
    e1 = make_envelope(ExcitationConfig(model="pluck_spectrum",
                                        pluck_position_ratio=0.2), 648.0, 1025)
    e2 = make_envelope(ExcitationConfig(model="pluck_spectrum",
                                        pluck_position_ratio=0.8), 648.0, 1025)
    assert np.allclose(e1.amp_mm, e2.amp_mm)


def test_pluck_spectrum_no_phantom_overshoot():
    """Guard against worst-case overshoot creeping into the default model:
    the envelope peak must be exactly A (d'Alembert), never larger."""
    exc = ExcitationConfig(model="pluck_spectrum", pluck_amplitude_mm=3.0,
                           pluck_position_ratio=0.3)
    env = make_envelope(exc, 648.0)
    assert env.peak_mm == pytest.approx(3.0, rel=1e-3)


def test_hi_harmonic_nodes():
    exc = ExcitationConfig(model="hi_harmonic", pluck_amplitude_mm=1.0,
                           harmonic_number=4)
    env = make_envelope(exc, 648.0, samples=1297)
    assert env.peak_mm == pytest.approx(1.0, rel=1e-3)
    # node at L/4 for the 4th partial
    q = np.interp(648.0 / 4, env.s_mm, env.amp_mm)
    assert q == pytest.approx(0.0, abs=0.01)


def test_worst_case_modal_envelope_conservative():
    """Opt-in paranoid envelope: same global peak as the analytic midpoint
    excursion A·min(p,1−p), but modal ripple means local excursions exceed
    the smooth Gibbs solution in the tails."""
    exc = ExcitationConfig(model="pluck_spectrum", pluck_amplitude_mm=2.0,
                           pluck_position_ratio=0.5, harmonics=16)
    rough = worst_case_modal_envelope(exc, 648.0, samples=2049)
    assert rough.peak_mm == pytest.approx(2.0 * 0.5, rel=1e-3)
    # compare against a clean half-sine of the same midpoint height
    smooth_amp = 1.0 * np.sin(np.pi * rough.s_mm / 648.0)
    tail = (rough.s_mm > 20) & (rough.s_mm < 0.25 * 648.0)
    ratio = rough.amp_mm[tail] / np.maximum(smooth_amp[tail], 1e-9)
    assert ratio.max() > 1.1


def test_energy_fraction_reported():
    exc = ExcitationConfig(model="pluck_spectrum", harmonics=3)
    env = make_envelope(exc, 648.0)
    assert 0.0 < env.energy_fraction < 1.0
    exc2 = ExcitationConfig(model="pluck_spectrum", harmonics=64)
    env2 = make_envelope(exc2, 648.0)
    assert env2.energy_fraction > env.energy_fraction
