"""Quasi-harmonic thermodynamics: harmonic limits and refusals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.core_model.units import BOLTZMANN_EV_K
from materia.physics import eam
from materia.physics import quasiharmonic as Q


def copper(a=3.615, pbc=(True, True, True)):
    cell = Cell(np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a, pbc)
    return Structure([29], [[0, 0, 0]], cell)


def test_harmonic_free_energy_limits():
    nu = np.array([2.0, 5.0, 8.0])
    w = np.full(3, 1.0)
    zero_point = 0.5 * Q.PLANCK_EV_S * nu.sum() * 1e12
    f0, s0, c0 = Q.harmonic_free_energy(nu, w, 0.0)
    assert f0 == pytest.approx(zero_point) and s0 == 0 and c0 == 0
    t = 5000.0
    kt = BOLTZMANN_EV_K * t
    f, s, c = Q.harmonic_free_energy(nu, w, t)
    classical = kt * sum(math.log(Q.PLANCK_EV_S * v * 1e12 / kt) for v in nu)
    assert f == pytest.approx(classical, abs=1e-3 * kt)
    assert c == pytest.approx(3 * BOLTZMANN_EV_K, rel=1e-3)
    df = Q.harmonic_free_energy(nu, w, t + 1)[0] - Q.harmonic_free_energy(nu, w, t - 1)[0]
    assert -df / 2 == pytest.approx(s, rel=1e-5)


def test_refusals():
    pot = eam.load_shipped("Cu-Zhou04")
    temps = [0, 100, 200]
    with pytest.raises(Q.QuasiHarmonicError, match="five volumes"):
        Q.quasiharmonic(copper(), pot, [0.98, 1.0, 1.02], temps, (6, 6, 6), (4, 4, 4))
    with pytest.raises(Q.QuasiHarmonicError, match="periodic"):
        Q.quasiharmonic(copper(pbc=(True, True, False)), pot, np.linspace(0.97, 1.05, 5),
                        temps, (6, 6, 6), (4, 4, 4))
    with pytest.raises(Q.QuasiHarmonicError, match="increasing"):
        Q.quasiharmonic(copper(), pot, np.linspace(0.97, 1.05, 5), [0, 200, 100],
                        (6, 6, 6), (4, 4, 4))
    with pytest.raises(Q.QuasiHarmonicError, match="dynamically unstable"):
        Q.quasiharmonic(copper(), pot, np.linspace(0.97, 1.12, 6), temps, (6, 6, 6),
                        (16, 16, 16))
    with pytest.raises(Q.QuasiHarmonicError, match="outside the sampled volumes"):
        Q.quasiharmonic(copper(), pot, np.linspace(0.97, 1.03, 5), [0, 300, 600],
                        (6, 6, 6), (8, 8, 8))
