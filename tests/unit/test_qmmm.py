"""QM/MM electrostatic embedding: water dimer interaction and analytic forces."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics.molecular import KCAL_MOL_EV, PySCFPotential
from materia.physics.potentials import UnsupportedSystem
from materia.physics.qmmm import QMMMPotential

pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")


@pytest.fixture(scope="module")
def dimer():
    angle, r = np.radians(104.52), 0.9572
    donor = np.array([[0, 0, 0], [r, 0, 0], [r * np.cos(angle), r * np.sin(angle), 0]])
    oxygen = np.array([r + 1.95, 0, 0])
    acceptor = np.array([oxygen, oxygen + [0.586, 0, 0.757], oxygen + [0.586, 0, -0.757]])
    return Structure([8, 1, 1, 8, 1, 1], np.vstack([donor, acceptor]) + 8.0,
                     Cell(np.eye(3) * 20, (False, False, False)))


def test_interaction_energy_and_forces(dimer):
    qm = PySCFPotential("hf", basis="6-31g*")
    qmmm = QMMMPotential([0, 1, 2], [[3, 4, 5]], qm=qm)
    energy, forces = qmmm.energy_and_forces(dimer)
    monomer = Structure([8, 1, 1], dimer.positions[:3], dimer.cell)
    interaction = (energy - qm.energy(monomer)) / KCAL_MOL_EV
    assert -8.0 < interaction < -3.5
    h = 1e-4
    for atom in (0, 4):
        plus, minus = dimer.copy(), dimer.copy()
        plus.positions[atom, 0] += h
        minus.positions[atom, 0] -= h
        fd = -(qmmm.energy(plus) - qmmm.energy(minus)) / (2 * h)
        assert fd == pytest.approx(forces[atom, 0], abs=1e-4)
    assert qmmm.describe()["parameters"]["mm_model"] == "TIP3P"


def test_refusals(dimer):
    with pytest.raises(ValueError, match="both"):
        QMMMPotential([0, 1, 2, 3], [[3, 4, 5]])
    incomplete = QMMMPotential([0, 1], [[3, 4, 5]])
    ok, why = incomplete.supports(dimer)
    assert not ok and "exactly one" in why
    periodic = dimer.copy()
    periodic.cell = Cell(np.eye(3) * 20, (True, True, True))
    ok, why = QMMMPotential([0, 1, 2], [[3, 4, 5]]).supports(periodic)
    assert not ok and "isolated" in why
    with pytest.raises(ValueError, match="implicit"):
        QMMMPotential([0, 1, 2], [[3, 4, 5]], qm=PySCFPotential("hf", dielectric=78.4))
