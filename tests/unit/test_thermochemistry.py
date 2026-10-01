"""Ideal-gas thermochemistry: parity with ASE IdealGasThermo and refusals."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import thermochemistry as T

pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")
CM1_TO_EV = 1.239841984e-4


def molecule(numbers, positions):
    return Structure(numbers, positions, Cell(np.eye(3) * 20, (False, False, False)))


WATER = molecule([8, 1, 1], [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
CO2 = molecule([6, 8, 8], [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
METHANE = molecule([6, 1, 1, 1, 1], np.array([[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1],
                                              [-1, -1, 1]]) * 0.629)


@pytest.mark.parametrize("structure,frequencies,group,sigma", [
    (WATER, [1595.0, 3657.0, 3756.0], "C2v", 2),
    (CO2, [667.4, 667.4, 1333.0, 2349.0], "Dooh", 2),
    (METHANE, [1306.0] * 3 + [1534.0] * 2 + [2917.0] + [3019.0] * 3, "Td", 12),
])
def test_matches_ase_ideal_gas_thermo(structure, frequencies, group, sigma):
    from ase import Atoms
    from ase.thermochemistry import IdealGasThermo

    out = T.ideal_gas(structure, -10.0, frequencies, temperature_K=500.0, pressure_Pa=2e5,
                      spin_multiplicity=1).value
    assert out["point_group"] == group and out["symmetry_number"] == sigma
    atoms = Atoms(numbers=structure.numbers, positions=structure.positions,
                  masses=structure.masses())
    reference = IdealGasThermo(vib_energies=np.array(frequencies) * CM1_TO_EV,
                               geometry=out["geometry"], potentialenergy=-10.0, atoms=atoms,
                               symmetrynumber=sigma, spin=0)
    assert out["entropy_eV_K"] == pytest.approx(
        reference.get_entropy(500.0, 2e5, verbose=False), rel=1e-6)
    assert out["enthalpy_eV"] == pytest.approx(reference.get_enthalpy(500.0, verbose=False),
                                               abs=1e-6)
    assert out["gibbs_energy_eV"] == pytest.approx(
        reference.get_gibbs_energy(500.0, 2e5, verbose=False), abs=1e-6)


def test_spin_and_heat_capacity_limits():
    singlet = T.ideal_gas(WATER, 0.0, [1595.0, 3657.0, 3756.0]).value
    triplet = T.ideal_gas(WATER, 0.0, [1595.0, 3657.0, 3756.0], spin_multiplicity=3).value
    k = 8.617333262e-5
    assert triplet["entropy_eV_K"] - singlet["entropy_eV_K"] == pytest.approx(k * np.log(3))
    cold = T.ideal_gas(WATER, 0.0, [1595.0, 3657.0, 3756.0], temperature_K=50.0).value
    assert cold["heat_capacity_p_eV_K"] == pytest.approx(4 * k, rel=1e-6)


def test_refusals():
    with pytest.raises(T.ThermochemistryError, match="not a minimum"):
        T.ideal_gas(WATER, 0.0, [-300.0, 3657.0, 3756.0])
    with pytest.raises(T.ThermochemistryError, match="has 4 vibrations"):
        T.ideal_gas(CO2, 0.0, [667.4, 1333.0, 2349.0])
    periodic = WATER.copy()
    periodic.cell = Cell(np.eye(3) * 10, (True, True, True))
    with pytest.raises(T.ThermochemistryError, match="periodic"):
        T.ideal_gas(periodic, 0.0, [1595.0, 3657.0, 3756.0])
