"""Molecular IR intensities against ASE's independent finite-difference Infrared module."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import molecular as M

pytestmark = pytest.mark.validation
pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")
pytest.importorskip("rdkit", reason="BLOCKED: rdkit is not installed")
D_A2_AMU_TO_KM_MOL = 42.2561


def test_water_ir_matches_ase_infrared(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.vibrations import Infrared

    from materia.solvers.classical import ClassicalSolver

    pot = M.PySCFPotential("hf", basis="def2-svp")
    water = ClassicalSolver(pot).relax(M.molecule_from_smiles("O"), fmax_eV_A=1e-4,
                                       max_steps=300).structure

    class PySCFDipole(Calculator):
        implemented_properties = ["energy", "forces", "dipole"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            s = Structure(atoms.numbers, atoms.positions, Cell(np.eye(3) * 20, (False,) * 3))
            energy, forces = pot.energy_and_forces(s)
            _, dipole = M._mean_field_dipole(pot, pot.molecule(s))
            self.results = {"energy": energy, "forces": forces, "dipole": dipole * M.BOHR_A}

    atoms = Atoms(numbers=water.numbers, positions=water.positions, masses=water.masses())
    atoms.calc = PySCFDipole()
    monkeypatch.chdir(tmp_path)
    ase_ir = Infrared(atoms, delta=0.005, nfree=4)
    ase_ir.run()
    out = M.vibrational_spectra(water, "hf", "def2-svp")
    assert np.allclose(out["wavenumbers_cm1"], ase_ir.get_frequencies()[-3:].real, atol=0.1)
    assert np.allclose(out["ir_intensity_km_mol"],
                       ase_ir.intensities[-3:] * D_A2_AMU_TO_KM_MOL, rtol=1e-3)
    assert out["apt_sum_rule_error_au"] < 1e-4
