"""Rigid-ion model against closed forms and physical invariants.

* A Born-Mayer rocksalt crystal (charges +1 and -1, repulsion only between
  unlike nearest neighbours) has the lattice energy
  ``E = -M k_e / r + 6 (phi(r) - phi(r_c))`` per ion pair with the Madelung
  constant ``M = 1.747564594633...`` (Borwein et al., Lattice Sums Then and
  Now, 2013), and its equilibrium distance solves
  ``M k_e / r^2 = 6 A b exp(-b r)``.  The parameters are illustrative, not a
  fit to NaCl.
* BKS alpha-quartz relaxed at the experimental cell keeps the three Si and six
  O sites equivalent, and its Si-O bonds stay within 0.02 A of the measured
  ones.  That is a plausibility check of a fixed-cell relaxation, not a
  validation of the BKS fit.
* Velocity-Verlet dynamics of BKS quartz conserves energy to second order in
  the time step.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.core_model.units import COULOMB_K_EV_A
from materia.materials import default_library
from materia.physics import rigid_ion as R
from materia.physics.neighbors import neighbor_list
from materia.solvers.classical import ClassicalSolver
from materia.structure_builder.lattice import bulk

pytestmark = pytest.mark.validation

MADELUNG_ROCKSALT = 1.747564594633182
A_EV, B_PER_A, CUTOFF_A = 1200.0, 3.125, 3.4
FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def rocksalt(r):
    a = 2.0 * r
    return Structure([11] * 4 + [17] * 4,
                     np.vstack([FCC * a, (FCC + [0.5, 0, 0]) * a]), Cell.cubic(a))


def born_mayer():
    return R.RigidIon(R.RigidIonParameters(
        "born-mayer-test", "illustrative Born-Mayer rocksalt", {"Na": 1.0, "Cl": -1.0},
        {("Na", "Cl"): R.BuckinghamPair(A_EV, B_PER_A, 0.0)},
        source="illustrative parameters for a closed-form check", cutoff_A=CUTOFF_A))


def closed_form(r):
    shift = A_EV * math.exp(-B_PER_A * CUTOFF_A)
    return 4 * (-MADELUNG_ROCKSALT * COULOMB_K_EV_A / r
                + 6 * (A_EV * math.exp(-B_PER_A * r) - shift))


@pytest.mark.parametrize("r", [2.6, 2.8, 3.0])
def test_rocksalt_lattice_energy(r):
    energy, forces = born_mayer().energy_and_forces(rocksalt(r))
    assert energy == pytest.approx(closed_form(r), rel=1e-9)
    assert np.abs(forces).max() < 1e-9


def test_rocksalt_equilibrium_distance():
    potential = born_mayer()
    expected = brentq(lambda r: MADELUNG_ROCKSALT * COULOMB_K_EV_A / r ** 2
                      - 6 * A_EV * B_PER_A * math.exp(-B_PER_A * r), 2.0, 3.2, xtol=1e-14)
    found = minimize_scalar(lambda r: potential.energy(rocksalt(r)), bounds=(2.6, 3.0),
                            method="bounded", options={"xatol": 1e-9})
    assert 2.6 < expected < 3.0
    assert found.x == pytest.approx(expected, abs=1e-5)


def quartz(repeat=(1, 1, 1)):
    return bulk(default_library().get("silicon_dioxide"), repeat)


def environments(structure, cutoff=3.2):
    nl = neighbor_list(structure.positions, structure.cell, cutoff)
    return [np.sort(nl.d[nl.i == i]) for i in range(len(structure))]


def test_relaxed_quartz_keeps_its_symmetry_and_plausible_bonds():
    start = quartz()
    out = ClassicalSolver(R.RigidIon()).relax(start, fmax_eV_A=1e-4, max_steps=3000)
    assert out.convergence.converged
    relaxed = out.structure
    shells = environments(relaxed)
    for element in (14, 8):
        sites = [shells[i] for i in np.flatnonzero(relaxed.numbers == element)]
        for other in sites[1:]:
            assert np.allclose(other, sites[0], atol=1e-4)
    si = shells[int(np.flatnonzero(relaxed.numbers == 14)[0])]
    bonds = si[:4]
    assert si[4] > 2.5
    measured = environments(start)[int(np.flatnonzero(start.numbers == 14)[0])][:4]
    assert np.allclose(bonds, measured, atol=0.02)
    assert out.results["energy"].extra["potential_checks"]["ewald_converged"]


def test_dynamics_conserves_energy_to_second_order():
    solver = ClassicalSolver(R.RigidIon())
    start = solver.relax(quartz((2, 2, 2)), fmax_eV_A=1e-3, max_steps=3000).structure
    spreads = []
    for dt in (1.0, 0.5):
        run = solver.dynamics(start, steps=int(round(200 / dt)), dt_fs=dt,
                              temperature_K=300.0, seed=1)
        total = np.array(run.results["trajectory"].value["total_eV"])
        spreads.append(float(total.max() - total.min()))
    assert spreads[0] / len(start) < 2e-4
    assert 2.5 < spreads[0] / spreads[1] < 6.0
