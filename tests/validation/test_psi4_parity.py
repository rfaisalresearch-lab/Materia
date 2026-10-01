"""Psi4 driven by Materia against PySCF driven by Materia: two independent codes.

Skipped with BLOCKED when no Python environment has Psi4.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.physics import molecular as M
from materia.physics.potentials import UnsupportedSystem
from materia.solvers import psi4_driver as P

pytestmark = pytest.mark.validation
pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")
pytest.importorskip("rdkit", reason="BLOCKED: rdkit is not installed")


@pytest.fixture(scope="module")
def water():
    if P.find_python() is None:
        pytest.skip("BLOCKED: no Python environment with Psi4")
    return M.molecule_from_smiles("O")


def test_hartree_fock_energy_and_forces(water):
    e1, f1 = P.Psi4Potential("hf", "cc-pvdz").energy_and_forces(water)
    e2, f2 = M.PySCFPotential("hf", basis="cc-pvdz").energy_and_forces(water)
    assert e1 == pytest.approx(e2, abs=1e-7)
    assert np.abs(f1 - f2).max() < 2e-5


def test_mp2_and_coupled_cluster(water):
    e1, f1 = P.Psi4Potential("mp2", "cc-pvdz").energy_and_forces(water)
    e2, f2 = M.PySCFPotential("mp2", basis="cc-pvdz").energy_and_forces(water)
    assert e1 == pytest.approx(e2, abs=1e-6)
    assert np.abs(f1 - f2).max() < 5e-4
    cc = P.Psi4Potential("hf", "cc-pvdz").single_point(water, "ccsd(t)")
    reference = M.pyscf_single_point(water, "ccsd(t)", basis="cc-pvdz")
    assert cc["energy_eV"] == pytest.approx(reference["energy_eV"], abs=1e-5)


def test_b3lyp_on_fine_grids(water):
    from pyscf import dft, gto

    e1, _ = P.Psi4Potential("b3lyp", "cc-pvdz", grid=(99, 590)).energy_and_forces(water)
    mol = gto.M(atom=[("OHH"[i], tuple(p)) for i, p in enumerate(water.positions)],
                basis="cc-pvdz", unit="Angstrom", verbose=0)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level, mf.conv_tol = "b3lyp", 9, 1e-11
    assert e1 == pytest.approx(mf.kernel() * M.HARTREE_EV, abs=2e-6)


def test_both_engines_relax_to_the_same_geometry(water):
    from materia.solvers import molecular

    bonds = []
    for name in ("psi4/hf", "pyscf/hf"):
        solver = molecular.create(name, basis="cc-pvdz")
        out = solver.relax(water, fmax_eV_A=1e-3, max_steps=200)
        assert out.convergence.converged
        p = np.asarray(out.structure.positions)
        bonds.append([np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])])
        assert solver.name.startswith(name.split("/")[0])
    assert np.allclose(bonds[0], bonds[1], atol=2e-4)


def test_refusals(water, monkeypatch):
    periodic = water.copy()
    periodic.cell = Cell(np.eye(3) * 10, (True, True, True))
    with pytest.raises(UnsupportedSystem, match="periodic"):
        P.Psi4Potential().energy_and_forces(periodic)
    with pytest.raises(UnsupportedSystem, match="multiplicity"):
        P.Psi4Potential(multiplicity=2).energy_and_forces(water)
    monkeypatch.setenv(P.PYTHON_ENV, "/nonexistent/python")
    P.find_python.cache_clear()
    try:
        with pytest.raises(P.Psi4Missing):
            P.Psi4Potential().energy_and_forces(water)
    finally:
        monkeypatch.delenv(P.PYTHON_ENV)
        P.find_python.cache_clear()


@pytest.mark.parametrize("tda", [False, True])
def test_tddft_matches_pyscf(water, tda):
    from pyscf import dft, tddft

    formaldehyde = M.molecule_from_smiles("C=O")
    mine = P.Psi4Potential("b3lyp", "def2-svp", grid=(99, 590)).excited_states(
        formaldehyde, 5, tda=tda)
    mol = M.PySCFPotential("dft", basis="def2-svp").molecule(formaldehyde)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level, mf.conv_tol = "b3lyp", 9, 1e-11
    mf.kernel()
    solver = (tddft.TDA if tda else tddft.TDDFT)(mf)
    solver.nstates, solver.conv_tol = 5, 1e-9
    solver.kernel()
    assert np.allclose(mine["excitation_energies_eV"], solver.e * M.HARTREE_EV, atol=2e-4)
    assert np.allclose(mine["oscillator_strengths"], solver.oscillator_strength(), atol=5e-5)
