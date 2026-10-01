"""Standard entropies and enthalpy corrections from B3LYP against NIST-JANAF.

S(298.15 K, 1 bar) in J/mol/K: H2O 188.835, CO2 213.785, CH4 186.251, and
H(298.15) - H(0) in kJ/mol: 9.905, 9.364, 10.016 (M. W. Chase, NIST-JANAF
Thermochemical Tables, 4th ed., J. Phys. Chem. Ref. Data Monograph 9, 1998).
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.physics import molecular as M
from materia.physics import thermochemistry as T

pytestmark = pytest.mark.validation
pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")
pytest.importorskip("rdkit", reason="BLOCKED: rdkit is not installed")


@pytest.mark.parametrize("smiles,entropy,correction", [
    ("O", 188.835, 9.905), ("O=C=O", 213.785, 9.364), ("C", 186.251, 10.016)])
def test_b3lyp_rrho_matches_janaf(smiles, entropy, correction):
    from materia.solvers.classical import ClassicalSolver

    pot = M.PySCFPotential("dft", basis="def2-tzvp", xc="b3lyp")
    s = ClassicalSolver(pot).relax(M.molecule_from_smiles(smiles), fmax_eV_A=1e-3,
                                   max_steps=300).structure
    spectra = M.vibrational_spectra(s, "dft", "def2-tzvp", xc="b3lyp")
    out = T.ideal_gas(s, pot.energy_and_forces(s)[0], spectra["wavenumbers_cm1"],
                      pressure_Pa=1e5, model=pot.name).value
    assert out["entropy_J_mol_K"] == pytest.approx(entropy, abs=0.8)
    thermal = (out["thermal_enthalpy_correction_eV"] - out["zero_point_energy_eV"]) * 96.4853
    assert thermal == pytest.approx(correction, abs=0.06)
