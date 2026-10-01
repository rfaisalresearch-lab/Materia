"""Powder diffraction, attenuation and edges from XrayDB data."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import xray as X

pytest.importorskip("xraydb", reason="BLOCKED: xraydb is not installed")
FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def silicon(a=5.431, shift=(0, 0, 0)):
    frac = np.vstack([FCC, FCC + 0.25]) + np.asarray(shift)
    return Structure([14] * 8, frac * a, Cell(np.eye(3) * a, (True,) * 3))


def test_silicon_reflections_follow_bragg_and_diamond_rules():
    peaks = X.powder_pattern(silicon(), "CuKa1", (20, 100)).value
    hkl = [tuple(p["hkl"]) for p in peaks]
    assert hkl[:6] == [(1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (3, 3, 1), (4, 2, 2)]
    assert (2, 0, 0) not in hkl and (2, 2, 2) not in hkl
    assert [p["multiplicity"] for p in peaks[:6]] == [8, 12, 24, 6, 24, 24]
    for p in peaks:
        d = 5.431 / np.sqrt(sum(v * v for v in p["hkl"]))
        assert p["d_A"] == pytest.approx(d, rel=1e-9)
        assert np.sin(np.radians(p["two_theta_deg"] / 2)) == pytest.approx(
            1.540593 / (2 * d), rel=1e-9)
    assert max(p["relative_intensity"] for p in peaks) == pytest.approx(100.0)


def test_pattern_is_invariant_to_origin_and_supercell():
    reference = X.powder_pattern(silicon(), "CuKa1", (20, 100)).value
    shifted = X.powder_pattern(silicon(shift=(0.13, 0.31, 0.07)), "CuKa1", (20, 100)).value
    base = silicon()
    double = Structure(list(base.numbers) * 2,
                       np.vstack([base.positions, np.asarray(base.positions) + [5.431, 0, 0]]),
                       Cell(np.diag([10.862, 5.431, 5.431]), (True,) * 3))
    doubled = X.powder_pattern(double, "CuKa1", (20, 100)).value
    for other in (shifted, doubled):
        assert [round(p["two_theta_deg"], 6) for p in other] == \
            [round(p["two_theta_deg"], 6) for p in reference]
        assert np.allclose([p["relative_intensity"] for p in other],
                           [p["relative_intensity"] for p in reference], rtol=1e-6)


def test_debye_waller_damps_high_angles():
    cold = X.powder_pattern(silicon(), "CuKa1", (20, 100)).value
    warm = X.powder_pattern(silicon(), "CuKa1", (20, 100), debye_waller_A2={"Si": 0.5}).value
    assert warm[-1]["relative_intensity"] < cold[-1]["relative_intensity"]
    s = np.sin(np.radians(cold[-1]["two_theta_deg"] / 2)) / 1.540593
    s0 = np.sin(np.radians(cold[0]["two_theta_deg"] / 2)) / 1.540593
    ratio = (warm[-1]["relative_intensity"] / cold[-1]["relative_intensity"])
    assert ratio == pytest.approx(np.exp(-2 * 0.5 * (s ** 2 - s0 ** 2)), rel=1e-6)


def test_rock_salt_odd_reflections_are_weak():
    a = 5.6402
    nacl = Structure([11] * 4 + [17] * 4, np.vstack([FCC, FCC + [0.5, 0, 0]]) * a,
                     Cell(np.eye(3) * a, (True,) * 3))
    peaks = {tuple(p["hkl"]): p for p in X.powder_pattern(nacl, "CuKa1", (20, 90)).value}
    assert peaks[(2, 0, 0)]["relative_intensity"] == pytest.approx(100.0)
    assert peaks[(1, 1, 1)]["relative_intensity"] < 15
    assert peaks[(3, 1, 1)]["relative_intensity"] < 5


def test_attenuation_matches_nist_xcom():
    water = X.attenuation("H2O", [1e5], density=1.0).value
    assert water["mass_attenuation_cm2_g"][0] == pytest.approx(0.1707, rel=0.01)
    lead = X.attenuation("Pb", [1e5], density=11.35).value
    assert lead["mass_attenuation_cm2_g"][0] == pytest.approx(5.549, rel=0.01)
    si = X.attenuation(silicon(), [8047.8], thickness_um=10).value
    assert si["density_g_cm3"] == pytest.approx(2.329, rel=1e-3)
    assert si["transmission"][0] == pytest.approx(
        np.exp(-si["linear_attenuation_per_cm"][0] * 1e-3))


def test_edges_and_lines():
    copper = X.edges_and_lines("Cu")
    assert copper["edges"]["K"]["energy_eV"] == pytest.approx(8979, abs=2)
    assert copper["lines"]["Ka1"]["energy_eV"] == pytest.approx(8047, abs=2)


def test_refusals():
    molecule = silicon()
    molecule.cell = Cell(np.eye(3) * 5.431, (True, True, False))
    with pytest.raises(X.XrayError, match="periodic"):
        X.powder_pattern(molecule)
    with pytest.raises(X.XrayError, match="Unknown source"):
        X.powder_pattern(silicon(), "FeKa9")
    with pytest.raises(X.XrayError, match="density"):
        X.attenuation("H2O", [1e4])
    with pytest.raises(X.XrayError, match="between 100 eV"):
        X.attenuation("H2O", [1e7], density=1.0)
