import numpy as np

from materia.physics.structure_search import BasinHoppingSettings
from materia.python_api.api import Lab


def small_settings():
    return BasinHoppingSettings(
        trials=1,
        displacement_A=0.03,
        temperature_eV=0.05,
        fmax_eV_A=0.2,
        relaxation_steps=20,
        keep=2,
        seed=12,
        require_converged=False,
    )


def test_python_workflow_persists_and_retrieves_search_products():
    lab = Lab()
    surface = lab.materials.load("silicon").bulk()
    original = surface.structure.positions.copy()
    result = surface.search_minima(settings=small_settings())
    assert np.array_equal(surface.structure.positions, original)
    run_id = result.diagnostics["run_id"]
    assert len(lab.search.runs()) == 1
    stored_run = lab.search.runs()[0]
    assert run_id == stored_run
    assert lab.search.result(stored_run, "best_energy").extra["run_id"] == stored_run
    assert lab.search.array(stored_run, "candidate_positions").data.shape[1:] == (
        len(surface.structure), 3)


def test_activate_best_adds_one_undoable_structure_without_mutating_source():
    lab = Lab()
    surface = lab.materials.load("silicon").bulk()
    source_key = lab.project.active_structure_key
    original = surface.structure.positions.copy()
    result = lab.search.basin_hopping(
        surface, settings=small_settings(), activate_best=True)
    best_key = lab.project.active_structure_key
    assert best_key != source_key
    assert np.array_equal(surface.structure.positions, original)
    assert np.array_equal(lab.project.structure.positions, result.best.structure.positions)
    lab.undo()
    assert lab.project.active_structure_key == source_key
