"""Vacancy and surface energetics: size invariants and refusals with the shipped Cu EAM."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.materials.loader import load
from materia.physics import eam
from materia.physics import energetics as E
from materia.structure_builder.lattice import bulk


@pytest.fixture(scope="module")
def copper():
    return load("copper"), eam.load_shipped("Cu-Zhou04")


def test_unrelaxed_values_do_not_depend_on_size(copper):
    material, pot = copper
    vacancy = E.vacancy_formation(bulk(material), pot, supercells=((3, 3, 3), (4, 4, 4)))
    unrelaxed = [r["unrelaxed_formation_energy_eV"] for r in vacancy.extra["series"]]
    assert unrelaxed[0] == pytest.approx(unrelaxed[1], abs=1e-8)
    assert all(r["formation_energy_eV"] < r["unrelaxed_formation_energy_eV"]
               for r in vacancy.extra["series"])
    assert vacancy.convergence.converged
    surface = E.surface_energy(material, pot, (1, 0, 0), layers=(6, 8))
    unrelaxed = [r["unrelaxed_surface_energy_J_m2"] for r in surface.extra["series"]]
    assert unrelaxed[0] == pytest.approx(unrelaxed[1], abs=1e-8)
    assert surface.value <= unrelaxed[-1] + 1e-9


def test_equilibrium_bulk_has_zero_pressure():
    from materia.physics.potentials import LennardJones

    lj = LennardJones({"Ar": (0.0103, 3.40)}, cutoff_A=8.0)
    frac = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    argon = Structure([18] * 4, frac * 5.3, Cell(np.eye(3) * 5.3, (True,) * 3))
    ref = E.equilibrium_bulk(argon, lj)
    plus = lj.energy_and_forces(E._scaled(ref["structure"], 1.0005))[0]
    minus = lj.energy_and_forces(E._scaled(ref["structure"], 0.9995))[0]
    centre = lj.energy_and_forces(ref["structure"])[0]
    assert abs(plus - minus) < 1e-2 * (plus + minus - 2 * centre)


def test_refusals(copper):
    material, pot = copper
    slab = bulk(material)
    slab.cell = Cell(slab.cell.matrix, (True, True, False))
    with pytest.raises(E.EnergeticsError, match="periodic"):
        E.equilibrium_bulk(slab, pot)
    rattled = bulk(material)
    rattled.positions = np.asarray(rattled.positions) + np.array([[0.05, 0, 0]] + [[0, 0, 0]] * 3)
    with pytest.raises(E.EnergeticsError, match="feel forces"):
        E.equilibrium_bulk(rattled, pot)
    far = Structure(bulk(material).numbers, np.asarray(bulk(material).positions) * 1.3,
                    Cell(bulk(material).cell.matrix * 1.3, (True,) * 3))
    with pytest.raises(E.EnergeticsError, match="edge"):
        E.equilibrium_bulk(far, pot)
