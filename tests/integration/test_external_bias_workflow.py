import numpy as np
import pytest

from materia.physics.external_bias import BIAS_KEY
from materia.physics.potentials import Harmonic
from materia.python_api.api import Lab
from materia.solvers.classical import ClassicalSolver


def test_surface_force_participates_in_solver_and_undo():
    lab = Lab()
    surface = lab.materials.load("silicon").bulk()
    atom_id = int(surface.structure.ids[0])
    record = surface.apply_force(atom_id, [0.5, 0.0, 0.0])
    solver = ClassicalSolver(Harmonic(surface.structure.positions.copy(), 1.0))
    result = solver.single_point(surface.structure)
    assert record["atom_id"] == atom_id
    assert result.results["forces"].value[0].tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert "external_biases" in result.results["energy"].provenance.parameters
    lab.undo()
    assert BIAS_KEY not in surface.structure.info
    lab.redo()
    assert surface.biases()["constant_forces"][str(atom_id)] == record


def test_restraint_changes_relaxation_target():
    lab = Lab()
    surface = lab.materials.load("silicon").bulk()
    atom_id = int(surface.structure.ids[0])
    reference = surface.structure.positions.copy()
    target = reference[0] + np.array([0.2, 0.0, 0.0])
    surface.restrain(atom_id, 3.0, target)
    solver = ClassicalSolver(Harmonic(reference, 1.0))
    result = solver.relax(surface.structure, fmax_eV_A=1e-4, max_steps=1000)
    expected = reference[0] + 0.75 * (target - reference[0])
    assert result.structure.positions[0].tolist() == pytest.approx(expected.tolist(), abs=2e-3)


def test_detached_handle_changes_bias_without_writing_project_history():
    lab = Lab()
    active = lab.materials.load("silicon").bulk()
    detached = lab.materials.load("silicon").bulk(activate=False)
    before = len(lab.project.history.log)
    detached.apply_force(int(detached.structure.ids[0]), [1.0, 0.0, 0.0])
    assert len(lab.project.history.log) == before
    assert BIAS_KEY in detached.structure.info
    assert BIAS_KEY not in active.structure.info


def test_clear_biases_is_undoable_as_one_operation():
    lab = Lab()
    surface = lab.materials.load("silicon").bulk()
    atom_id = int(surface.structure.ids[0])
    surface.apply_force(atom_id, [1.0, 0.0, 0.0])
    surface.restrain(atom_id, 2.0)
    assert surface.clear_biases() == 2
    assert BIAS_KEY not in surface.structure.info
    lab.undo()
    assert len(surface.biases()["constant_forces"]) == 1
    assert len(surface.biases()["restraints"]) == 1
