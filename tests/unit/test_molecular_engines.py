"""Molecular engines: parity with the native packages, gradients and refusals."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.core_model.units import BOHR_A, HARTREE_EV
from materia.physics import molecular as M
from materia.physics.potentials import UnsupportedSystem
from materia.provenance import Fidelity
from materia.solvers import registry
from materia.solvers.molecular import create

rdkit = pytest.importorskip("rdkit", reason="BLOCKED: rdkit is not installed")
tblite = pytest.importorskip("tblite", reason="BLOCKED: tblite is not installed")
pyscf = pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")


@pytest.fixture(scope="module")
def ethanol():
    return M.molecule_from_smiles("CCO", seed=1)


def gradient_error(potential, structure, h=1e-4, atoms=2):
    _, forces = potential.energy_and_forces(structure)
    worst = 0.0
    for i in range(atoms):
        for c in range(3):
            plus, minus = structure.copy(), structure.copy()
            plus.positions[i, c] += h
            minus.positions[i, c] -= h
            fd = -(potential.energy(plus) - potential.energy(minus)) / (2 * h)
            worst = max(worst, abs(fd - forces[i, c]))
    return worst


def test_smiles_builder_records_topology(ethanol):
    record = ethanol.info["molecule"]
    assert record["smiles"] == "CCO" and record["charge"] == 0
    assert len(ethanol) == 9 and not any(ethanol.cell.pbc)
    from rdkit import Chem
    block = Chem.MolFromMolBlock(record["molblock"], removeHs=False)
    assert np.allclose(block.GetConformer().GetPositions(), ethanol.positions, atol=1e-4)
    with pytest.raises(UnsupportedSystem, match="could not parse"):
        M.molecule_from_smiles("C1CC")


@pytest.mark.parametrize("field", ["mmff94", "uff"])
def test_rdkit_force_fields_match_rdkit(ethanol, field):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromMolBlock(ethanol.info["molecule"]["molblock"], removeHs=False)
    native = (AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
              if field == "mmff94" else AllChem.UFFGetMoleculeForceField(mol))
    potential = M.RDKitForceField(field)
    flat = [float(v) for v in ethanol.positions.ravel()]
    assert potential.energy(ethanol) == pytest.approx(native.CalcEnergy(flat) * M.KCAL_MOL_EV,
                                                      abs=1e-10)
    assert gradient_error(potential, ethanol) < 1e-6


def test_xtb_matches_tblite(ethanol):
    from tblite.interface import Calculator
    calc = Calculator("GFN2-xTB", ethanol.numbers, ethanol.positions / BOHR_A)
    calc.set("verbosity", 0)
    native = calc.singlepoint()
    energy, forces = M.XTBPotential().energy_and_forces(ethanol)
    assert energy == pytest.approx(native.get("energy") * HARTREE_EV, abs=1e-9)
    assert np.allclose(forces, -native.get("gradient") * HARTREE_EV / BOHR_A, atol=1e-9)
    assert gradient_error(M.XTBPotential(), ethanol) < 2e-4


def test_pyscf_hydrogen_reference_and_parity():
    h2 = Structure([1, 1], np.array([[5, 5, 5], [5, 5, 5 + 1.4 * BOHR_A]]),
                   Cell(np.eye(3) * 10, (False, False, False)))
    energy = M.PySCFPotential("hf", basis="sto-3g").energy(h2) / HARTREE_EV
    assert energy == pytest.approx(-1.1167, abs=1e-4)
    water = M.molecule_from_smiles("O")
    from pyscf import dft, gto
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip("OHH", water.positions)],
                basis="def2-svp", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-10
    native = mf.kernel() * HARTREE_EV
    assert M.PySCFPotential("dft", basis="def2-svp", xc="b3lyp").energy(water) == \
        pytest.approx(native, abs=1e-7)
    assert gradient_error(M.PySCFPotential("mp2", basis="cc-pvdz"), water, h=1e-3,
                          atoms=1) < 1e-4


def test_pyscf_coupled_cluster_properties():
    out = M.pyscf_single_point(M.molecule_from_smiles("O"), "ccsd(t)", "cc-pvdz")
    assert out["triples_Eh"] < 0 and out["ccsd_correlation_Eh"] < 0
    assert out["energy_Eh"] == pytest.approx(out["reference_energy_Eh"]
                                             + out["ccsd_correlation_Eh"] + out["triples_Eh"])
    assert sum(out["mulliken_charges_e"]) == pytest.approx(0.0, abs=1e-8)
    assert out["homo_lumo_gap_eV"] > 0


def test_refusals():
    crystal = Structure([1, 1], np.array([[0, 0, 0], [0, 0, 0.74]]),
                        Cell(np.eye(3) * 5, (True, True, True)))
    for potential in (M.PySCFPotential("hf"), M.RDKitForceField()):
        ok, why = potential.supports(crystal)
        assert not ok and "isolated" in why
    radical = Structure([1], np.zeros((1, 3)), Cell(np.eye(3) * 8, (False,) * 3))
    ok, why = M.PySCFPotential("hf").supports(radical)
    assert not ok and "spin" in why
    assert M.PySCFPotential("hf", spin=1).supports(radical)[0]
    ok, why = M.XTBPotential().supports(radical)
    assert not ok and "unpaired" in why


def test_missing_engine_is_refused(monkeypatch):
    def missing(name):
        raise ImportError(name)
    monkeypatch.setattr(M.importlib, "import_module", missing)
    h2 = Structure([1, 1], np.array([[0, 0, 0], [0, 0, 0.74]]),
                   Cell(np.eye(3) * 8, (False, False, False)))
    ok, why = M.XTBPotential().supports(h2)
    assert not ok and "pip install tblite" in why


def test_solvers_name_their_engine(ethanol):
    for name in ("rdkit/mmff94", "xtb/gfn2", "pyscf/b3lyp"):
        assert name in registry.available()
    solver = create("xtb/gfn2")
    assert solver.name == "xtb/gfn2" and solver.fidelity is Fidelity.TIER2_SEMI_EMPIRICAL
    out = solver.single_point(ethanol)
    prov = out["energy"].provenance
    assert prov.model == "xtb/gfn2" and prov.parameters["engine"] == "tblite"
    assert not any("no electronic structure" in a for a in prov.approximations)
    props = solver.properties(ethanol).value
    assert abs(sum(props["charges_e"])) < 1e-6 and props["homo_lumo_gap_eV"] > 0
    assert create("pyscf/hf", basis="sto-3g").potential.basis == "sto-3g"
    refused = create("rdkit/mmff94").properties(ethanol)
    assert not refused.supported
