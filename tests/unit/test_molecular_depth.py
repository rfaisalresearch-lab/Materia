"""Implicit and explicit solvation, excited states and conformer search."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from materia.core_model.units import BOHR_A, HARTREE_EV
from materia.physics import molecular as M
from materia.physics.potentials import UnsupportedSystem

pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")
pytest.importorskip("tblite", reason="BLOCKED: tblite is not installed")
pytest.importorskip("rdkit", reason="BLOCKED: rdkit is not installed")
TRP_CAGE = Path(__file__).resolve().parents[1] / "data" / "1l2y_model1.pdb"
KCAL = 0.0433641043


@pytest.fixture(scope="module")
def water():
    return M.molecule_from_smiles("O")


def test_xtb_implicit_solvation_matches_tblite(water):
    from tblite.interface import Calculator
    for model in ("alpb", "gbsa"):
        calc = Calculator("GFN2-xTB", water.numbers, water.positions / BOHR_A)
        calc.set("verbosity", 0)
        calc.add(f"{model}-solvation", "water")
        solvated = M.XTBPotential(solvent="water", solvation_model=model)
        assert solvated.energy(water) == pytest.approx(
            calc.singlepoint().get("energy") * HARTREE_EV, abs=1e-9)
        shift = (solvated.energy(water) - M.XTBPotential().energy(water)) / KCAL
        assert -12 < shift < -3
        assert solvated.name == f"xtb/gfn2+{model}(water)"
    with pytest.raises(UnsupportedSystem, match="no alpb parameters"):
        M.XTBPotential(solvent="unobtainium").energy(water)


def test_pyscf_pcm_energy_and_gradient(water):
    solvated = M.PySCFPotential("hf", basis="6-31g*", dielectric=78.3553)
    energy, forces = solvated.energy_and_forces(water)
    shift = (energy - M.PySCFPotential("hf", basis="6-31g*").energy(water)) / KCAL
    assert -10 < shift < -4
    h = 1e-3
    plus, minus = water.copy(), water.copy()
    plus.positions[0, 2] += h
    minus.positions[0, 2] -= h
    fd = -(solvated.energy(plus) - solvated.energy(minus)) / (2 * h)
    assert fd == pytest.approx(forces[0, 2], abs=1e-6)
    with pytest.raises(ValueError, match="MP2"):
        M.PySCFPotential("mp2", dielectric=78.4)


def test_excited_states_match_pyscf(water):
    out = M.excited_states(water, 3, "tda", "6-31g*", xc="b3lyp")
    from pyscf import dft, gto, tddft
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip("OHH", water.positions)],
                basis="6-31g*", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-10
    mf.kernel()
    native = tddft.TDA(mf)
    native.nstates = 3
    native.kernel()
    assert np.allclose(out["excitation_energies_eV"], native.e * HARTREE_EV, atol=1e-5)
    assert np.allclose(np.array(out["wavelengths_nm"]) * np.array(out["excitation_energies_eV"]),
                       1239.841984)
    assert all(f >= 0 for f in out["oscillator_strengths"])
    cis = M.excited_states(water, 2, "cis", "6-31g*")
    assert cis["xc"] is None and cis["excitation_energies_eV"][0] > 0
    with pytest.raises(ValueError):
        M.excited_states(water, 0)


def test_conformers_are_distinct_and_ranked():
    found = M.conformers("CCCCO", count=20, seed=2)
    assert len(found) >= 3
    energies = [c["relative_energy_eV"] for c in found]
    assert energies[0] == 0.0 and energies == sorted(energies)
    potential = M.RDKitForceField("mmff94")
    for c in found[:3]:
        assert potential.energy(c["structure"]) == pytest.approx(c["energy_eV"], abs=1e-4)


def test_explicit_solvation_and_npt():
    pytest.importorskip("openmm", reason="BLOCKED: openmm is not installed")
    from materia.physics import biomolecular as B
    import openmm as mm
    from openmm import app, unit
    protein = B.load_pdb(str(TRP_CAGE))
    boxed = B.solvate(protein, padding_A=8.0)
    record = boxed.info[B.INFO_KEY]["solvation"]
    assert all(boxed.cell.pbc) and record["waters"] > 500
    assert record["sodium"] + record["chloride"] >= 1
    energy = B.OpenMMPotential("amber14", platform="CPU").energy(boxed)
    pdb = app.PDBFile(io.StringIO(boxed.info[B.INFO_KEY]["pdb"]))
    pdb.topology.setPeriodicBoxVectors(boxed.cell.matrix * 0.1)
    system = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml").createSystem(
        pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None, rigidWater=False)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CPU"))
    context.setPositions(boxed.positions * 0.1)
    native = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole) * B.KJ_MOL_EV
    assert energy == pytest.approx(native, abs=1e-4)
    run = B.simulate(boxed, force_field="amber14", steps=2000, report_every=1000, seed=1,
                     pressure_bar=1.0)
    assert len(run["trajectory"].value["density_g_cm3"]) == 2
    with pytest.raises(UnsupportedSystem, match="periodic"):
        B.simulate(protein, steps=10, pressure_bar=1.0)
    with pytest.raises(UnsupportedSystem, match="explicit water"):
        B.solvate(protein, force_field="amber14-gbn2")


def test_delta_scf_formaldehyde_n_to_pi_star():
    molecule = M.molecule_from_smiles("C=O")
    out = M.delta_scf(molecule, basis="def2-svp", xc="b3lyp")
    tddft = M.excited_states(molecule, 1, "tddft", "def2-svp", xc="b3lyp")
    assert 0 < out["triplet_eV"] < out["singlet_eV"]
    assert out["triplet_s2"] == pytest.approx(2.0, abs=0.05)
    assert abs(out["singlet_eV"] - tddft["excitation_energies_eV"][0]) < 0.6
    with pytest.raises(UnsupportedSystem, match="outside the basis"):
        M.delta_scf(molecule, hole=40)


def test_finite_field_dipole_and_polarizability(water):
    out = M.field_response(water, "hf", "aug-cc-pvdz")
    from pyscf import gto, scf
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip("OHH", water.positions)],
                basis="aug-cc-pvdz", verbose=0)
    analytic = scf.RHF(mol).run(conv_tol=1e-12).dip_moment(unit="au", verbose=0)
    assert np.allclose(out["dipole_au"], analytic, atol=1e-5)
    assert np.allclose(out["dipole_from_energy_au"], analytic, atol=1e-5)
    alpha = np.array(out["polarizability_au"])
    assert np.allclose(np.diag(alpha), out["polarizability_diagonal_from_energy_au"], atol=1e-3)
    assert np.allclose(alpha, alpha.T, atol=1e-3)
    assert 7.0 < out["isotropic_polarizability_au"] < 10.0
    with pytest.raises(ValueError, match="field_au"):
        M.field_response(water, field_au=0.1)


def relaxed(smiles, basis="sto-3g"):
    from materia.solvers.classical import ClassicalSolver
    pot = M.PySCFPotential("hf", basis=basis)
    out = ClassicalSolver(pot).relax(M.molecule_from_smiles(smiles), fmax_eV_A=1e-4,
                                     max_steps=400)
    assert out.convergence.converged
    return out.structure


def test_carbon_dioxide_obeys_mutual_exclusion():
    co2 = relaxed("O=C=O")
    out = M.vibrational_spectra(co2, "hf", "sto-3g", raman=True)
    assert out["rigid_modes_removed"] == 5
    nu = np.array(out["wavenumbers_cm1"])
    ir = np.array(out["ir_intensity_km_mol"])
    raman = np.array(out["raman_activity_A4_amu"])
    assert len(nu) == 4 and np.all(nu > 0)
    assert nu[0] == pytest.approx(nu[1], abs=0.5)
    symmetric = int(np.argmax(raman))
    assert ir[symmetric] < 1e-3
    others = [k for k in range(4) if k != symmetric]
    assert np.all(ir[others] > 1.0)
    assert np.all(raman[others] < 1e-3 * raman[symmetric])
    assert out["apt_sum_rule_error_au"] < 1e-3


def test_water_spectra_are_rotation_invariant():
    water = relaxed("O")
    first = M.vibrational_spectra(water, "hf", "sto-3g", raman=True)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]]) @ np.array([[1, 0, 0], [0, np.cos(1.1), -np.sin(1.1)],
                                                 [0, np.sin(1.1), np.cos(1.1)]])
    turned = water.copy()
    turned.positions = np.asarray(water.positions) @ rotation.T
    second = M.vibrational_spectra(turned, "hf", "sto-3g", raman=True)
    for key, tol in (("wavenumbers_cm1", 1e-3), ("ir_intensity_km_mol", 1e-3),
                     ("raman_activity_A4_amu", 1e-3), ("depolarization_ratio", 1e-3)):
        assert np.allclose(first[key], second[key], rtol=tol, atol=tol), key
    assert first["depolarization_ratio"][2] == pytest.approx(0.75, abs=1e-3)
    assert max(first["depolarization_ratio"][:2]) < 0.75


def test_spectra_refuse_unrelaxed_geometry(water):
    stretched = water.copy()
    stretched.positions = np.asarray(water.positions) * 1.1
    with pytest.raises(UnsupportedSystem, match="relax the molecule"):
        M.vibrational_spectra(stretched, "hf", "sto-3g")


def test_absorption_spectrum_integrates_to_oscillator_strength():
    excitations = {"excitation_energies_eV": [4.0, 6.0], "oscillator_strengths": [0.1, 0.5]}
    grid = np.linspace(1.0, 9.0, 4001)
    out = M.absorption_spectrum(excitations, grid, fwhm_eV=0.3)
    epsilon = np.array(out["molar_absorption_L_mol_cm"])
    wavenumber = grid * 8065.543937
    assert np.trapezoid(epsilon, wavenumber) == pytest.approx(M.EPSILON_PER_OSCILLATOR * 0.6,
                                                              rel=1e-6)
    peak = grid[np.argmax(epsilon)]
    assert peak == pytest.approx(6.0, abs=0.01)
    with pytest.raises(ValueError, match="fwhm"):
        M.absorption_spectrum(excitations, grid, fwhm_eV=5.0)
