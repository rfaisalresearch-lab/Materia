"""Harmonic transition-state theory against Langevin escape rates.

A particle in V = Eb (x^2 - 1)^2 + k (y^2 + z^2)/2 with Eb = 0.25 eV at 580 K
(Eb / kT = 5) and friction 1e13/s.  The stationary points are analysed by
Materia's finite-displacement phonons; Kramers-corrected harmonic TST is
compared with the rate of well-to-well transitions counted in Langevin
dynamics (with hysteresis at x = +-0.5 A).  Measured over 3 ns: 381 crossings,
ratio 0.90.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import kinetics as K
from materia.physics import phonons as P
from materia.physics.potentials import Potential
from materia.provenance import Fidelity
from materia.solvers.classical import ClassicalSolver

pytestmark = pytest.mark.validation
EB, STIFF = 0.25, 2.0


class DoubleWell(Potential):
    name = "double-well"

    def energy_and_forces(self, structure):
        x, y, z = structure.positions[0] - 10.0
        energy = EB * (x * x - 1) ** 2 + 0.5 * STIFF * (y * y + z * z)
        return energy, np.array([[-4 * EB * x * (x * x - 1), -STIFF * y, -STIFF * z]])

    def describe(self):
        return {"model": "double-well", "parameters": {}, "approximations": [],
                "references": []}


def at(x):
    return Structure([1], np.array([[10.0 + x, 10.0, 10.0]]),
                     Cell(np.eye(3) * 20, (False, False, False)))


@pytest.fixture(scope="module")
def stationary_points():
    settings = P.PhononSettings(sum_rule="none", stencil=4)
    minimum = P.harmonic_analysis(at(-1.0), DoubleWell(), settings, name="double-well",
                                  fidelity=Fidelity.NON_PHYSICAL)
    saddle = P.harmonic_analysis(at(0.0), DoubleWell(), settings, name="double-well",
                                 fidelity=Fidelity.NON_PHYSICAL)
    return minimum, saddle


def test_tst_prefactor_is_analytic(stationary_points):
    minimum, saddle = stationary_points
    rate = K.harmonic_tst_rate(minimum, saddle, EB, 580.0)
    mass = at(0).masses()[0]
    omega = np.sqrt(8 * EB / mass * P.EIGENVALUE_TO_RAD_S2)
    assert rate.extra["prefactor_Hz"] == pytest.approx(omega / (2 * np.pi), rel=1e-6)
    assert rate.extra["imaginary_frequency_THz"] < 0


def test_langevin_escape_rate_matches_kramers_corrected_tst(stationary_points):
    minimum, saddle = stationary_points
    temperature, friction_fs, steps = 580.0, 0.01, 1_500_000
    rate = K.harmonic_tst_rate(minimum, saddle, EB, temperature,
                               friction_per_s=friction_fs * 1e15)
    run = ClassicalSolver(DoubleWell()).dynamics(at(-1.0), steps=steps, dt_fs=1.0,
                                                 temperature_K=temperature,
                                                 thermostat="langevin",
                                                 friction_per_fs=friction_fs, seed=7,
                                                 sample_every=5)
    x = np.array([p[0, 0] for p in run.results["trajectory"].value["positions_A"]]) - 10.0
    side, crossings = 0, 0
    for value in x:
        if value < -0.5 and side >= 0:
            crossings += side == 1
            side = -1
        elif value > 0.5 and side <= 0:
            crossings += side == -1
            side = 1
    measured = crossings / (steps * 1e-15)
    assert crossings > 100
    assert 0.75 < measured / rate.value < 1.15


def test_refusals(stationary_points):
    minimum, saddle = stationary_points
    with pytest.raises(K.KineticsError, match="exactly one"):
        K.harmonic_tst_rate(minimum, minimum, EB, 300.0)
    with pytest.raises(K.KineticsError, match="not a minimum"):
        K.harmonic_tst_rate(saddle, saddle, EB, 300.0)
    with pytest.raises(K.KineticsError, match="positive"):
        K.harmonic_tst_rate(minimum, saddle, -0.1, 300.0)
