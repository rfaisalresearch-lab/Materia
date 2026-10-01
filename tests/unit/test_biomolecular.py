"""OpenMM biomolecular force fields: parity, gradients, native dynamics, refusals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import biomolecular as B
from materia.physics.potentials import UnsupportedSystem
from materia.solvers import registry
from materia.solvers.molecular import create

pytest.importorskip("openmm", reason="BLOCKED: openmm is not installed")
TRP_CAGE = Path(__file__).resolve().parents[1] / "data" / "1l2y_model1.pdb"


@pytest.fixture(scope="module")
def protein():
    return B.load_pdb(str(TRP_CAGE))


def test_load_records_the_biomolecule(protein):
    record = protein.info[B.INFO_KEY]
    assert len(protein) == 304 and record["residues"] == 20
    assert record["sequence"].startswith("ASN LEU TYR ILE GLN TRP")
    assert not any(protein.cell.pbc) and "placeholder" in record["box_note"]


def test_energy_matches_openmm_directly(protein):
    import openmm as mm
    from openmm import app, unit
    pdb = app.PDBFile(str(TRP_CAGE))
    system = app.ForceField("amber14-all.xml", "implicit/gbn2.xml").createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    context = mm.Context(system, mm.VerletIntegrator(0.001),
                         mm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    native = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole) * B.KJ_MOL_EV
    assert B.OpenMMPotential("amber14-gbn2").energy(protein) == pytest.approx(native, abs=1e-8)


@pytest.mark.parametrize("field", ["amber14-gbn2", "charmm36", "amber99sbildn"])
def test_forces_are_energy_gradients(protein, field):
    potential = B.OpenMMPotential(field)
    _, forces = potential.energy_and_forces(protein)
    h = 1e-4
    for atom in (0, 120):
        for axis in range(3):
            plus, minus = protein.copy(), protein.copy()
            plus.positions[atom, axis] += h
            minus.positions[atom, axis] -= h
            fd = -(potential.energy(plus) - potential.energy(minus)) / (2 * h)
            assert fd == pytest.approx(forces[atom, axis], abs=1e-5)


def test_native_langevin_dynamics(protein):
    out = B.simulate(protein, steps=2000, report_every=500, seed=3)
    assert out["minimised_energy"].value < out["initial_energy"].value
    temperatures = out["trajectory"].value["temperature_K"]
    assert 200 < np.mean(temperatures) < 400
    assert out["trajectory"].provenance.model == "openmm/langevin-middle[openmm/amber14-gbn2]"
    assert np.array_equal(protein.positions, B.load_pdb(str(TRP_CAGE)).positions)


def test_refusals(protein):
    plain = Structure([6, 8], np.array([[0, 0, 0], [0, 0, 1.2]]),
                      Cell(np.eye(3) * 10, (False, False, False)))
    ok, why = B.OpenMMPotential().supports(plain)
    assert not ok and "load_pdb" in why
    boxed = protein.copy()
    boxed.cell = Cell(np.eye(3) * 60, (True, True, True))
    ok, why = B.OpenMMPotential("amber14-gbn2").supports(boxed)
    assert not ok and "implicit solvent" in why
    edited = protein.copy()
    edited.numbers[0] = 8
    ok, why = B.OpenMMPotential().supports(edited)
    assert not ok and "no longer matches" in why


def test_registered_solver(protein):
    assert "openmm/charmm36" in registry.available()
    solver = create("openmm/amber14-gbn2")
    out = solver.single_point(protein)
    assert out["energy"].provenance.model == "openmm/amber14-gbn2"
    assert out["energy"].provenance.parameters["engine"] == "openmm"
