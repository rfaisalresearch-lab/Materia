"""XPS core binding energies by Delta-SCF against gas-phase experiment.

Experimental 1s binding energies (eV): CH4 290.84, H2O 539.90, CO (C) 296.23,
C2H2 291.14, compiled by W. L. Jolly, K. D. Bomben and C. J. Eyermann, At. Data
Nucl. Data Tables 31 (1984) 433.  The calculation is nonrelativistic, which
lowers it by about 0.1 eV for carbon and 0.35 eV for oxygen.
"""

from __future__ import annotations

import pytest

from materia.physics import molecular as M
from materia.physics.potentials import UnsupportedSystem

pytestmark = pytest.mark.validation
pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")
pytest.importorskip("rdkit", reason="BLOCKED: rdkit is not installed")


def relaxed(smiles):
    from materia.solvers.classical import ClassicalSolver
    pot = M.PySCFPotential("dft", basis="def2-tzvp", xc="tpss")
    out = ClassicalSolver(pot).relax(M.molecule_from_smiles(smiles, force_field="uff"),
                                     fmax_eV_A=2e-3, max_steps=200)
    assert out.convergence.converged
    return out.structure


@pytest.fixture(scope="module")
def molecules():
    return {name: relaxed(smiles) for name, smiles in
            (("CH4", "C"), ("H2O", "O"), ("CO", "[C-]#[O+]"), ("C2H2", "C#C"))}


def test_binding_energies_match_experiment(molecules):
    methane = M.core_binding_energy(molecules["CH4"], 0)
    water = M.core_binding_energy(molecules["H2O"], 0)
    carbon_monoxide = M.core_binding_energy(molecules["CO"], 0)
    assert methane["binding_energy_eV"] + 0.1 == pytest.approx(290.84, abs=0.3)
    assert water["binding_energy_eV"] + 0.35 == pytest.approx(539.90, abs=0.4)
    shift = carbon_monoxide["binding_energy_eV"] - methane["binding_energy_eV"]
    assert shift == pytest.approx(296.23 - 290.84, abs=0.2)
    for out in (methane, water, carbon_monoxide):
        assert out["core_population"] > 0.99
        assert 0.74 < out["cation_s2"] < 0.8
        assert out["frozen_orbital_estimate_eV"] < out["binding_energy_eV"] - 10


def test_equivalent_atoms_give_equal_localised_holes(molecules):
    first = M.core_binding_energy(molecules["C2H2"], 0)
    second = M.core_binding_energy(molecules["C2H2"], 1)
    assert first["core_population"] > 0.99
    assert first["binding_energy_eV"] == pytest.approx(second["binding_energy_eV"], abs=1e-3)
    assert first["binding_energy_eV"] + 0.1 == pytest.approx(291.14, abs=0.3)


def test_refusals(molecules):
    with pytest.raises(UnsupportedSystem, match="no core electrons"):
        M.core_binding_energy(molecules["CH4"], 1)
    with pytest.raises(UnsupportedSystem, match="no atom"):
        M.core_binding_energy(molecules["CH4"], 9)
