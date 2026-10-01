"""Dimensional analysis and sparse candidate-law discovery."""

from __future__ import annotations

import numpy as np
import pytest

from materia.analysis import discovery as D
from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics.potentials import LennardJones

FORCE = {"M": 1, "L": 1, "T": -2}


def test_pendulum_and_drag_groups():
    assert D.pi_groups({"T": {"T": 1}, "L": {"L": 1}, "g": {"L": 1, "T": -2},
                        "m": {"M": 1}}) == [{"T": 2, "L": -1, "g": 1}]
    groups = D.pi_groups({"F": FORCE, "rho": {"M": 1, "L": -3}, "v": {"L": 1, "T": -1},
                          "D": {"L": 1}, "mu": {"M": 1, "L": -1, "T": -1}})
    assert len(groups) == 2
    dims = {"F": FORCE, "rho": {"M": 1, "L": -3}, "v": {"L": 1, "T": -1}, "D": {"L": 1},
            "mu": {"M": 1, "L": -1, "T": -1}}
    for group in groups:
        total = [sum(e * float(D.dimension_vector(dims[n])[i]) for n, e in group.items())
                 for i in range(len(D.BASE))]
        assert total == [0.0] * len(D.BASE)


@pytest.fixture(scope="module")
def lj_forces():
    eps, sig = 0.0103, 3.40
    lj = LennardJones({"Ar": (eps, sig)}, cutoff_A=50.0)
    r = np.linspace(3.2, 8.0, 120)
    forces = np.array([lj.energy_and_forces(Structure(
        [18, 18], np.array([[10, 10, 10], [10, 10, 10 + d]]),
        Cell(np.eye(3) * 40, (False, False, False))))[1][1, 2] for d in r])
    terms = {f"r^-{n}": (lambda x, n=n: x ** (-n)) for n in range(1, 15)}
    dims = {name: FORCE for name in terms}
    return r, forces, terms, dims, eps, sig


def test_lennard_jones_force_law_is_recovered(lj_forces):
    r, forces, terms, dims, eps, sig = lj_forces
    result = D.discover(r, forces, terms, dims, FORCE, threshold=1e-4)
    assert sorted(result.value["selected"]) == ["r^-13", "r^-7"]
    coefficients = result.value["coefficients"]
    assert coefficients["r^-13"] == pytest.approx(48 * eps * sig ** 12, rel=1e-8)
    assert coefficients["r^-7"] == pytest.approx(-24 * eps * sig ** 6, rel=1e-8)
    assert result.extra["relative_rms_held_out"] < 1e-8
    assert result.extra["term_stability"]["r^-13"] == 1.0
    assert result.provenance.origin.value == "estimated"


def test_noisy_data_still_selects_the_right_terms(lj_forces):
    r, forces, terms, dims, eps, sig = lj_forces
    noisy = forces + np.random.default_rng(2).normal(0, 1e-3 * np.abs(forces).max(), len(forces))
    small = {n: terms[n] for n in ("r^-6", "r^-7", "r^-8", "r^-12", "r^-13", "r^-14")}
    result = D.discover(r, noisy, small, {n: FORCE for n in small}, FORCE)
    assert set(result.value["selected"]) == {"r^-7", "r^-13"}
    assert result.extra["relative_rms_held_out"] < 0.01
    assert result.extra["term_stability"]["r^-13"] >= 0.8
    assert result.provenance.parameters["method"] == "best-subset"


def test_refusals(lj_forces):
    r, forces, terms, dims, *_ = lj_forces
    with pytest.raises(D.DiscoveryError, match="dimensions"):
        D.discover(r, forces, {"a": lambda x: x}, {"a": {"L": 1}}, FORCE)
    with pytest.raises(D.DiscoveryError, match="Unknown base"):
        D.dimension_vector({"Q": 1})
