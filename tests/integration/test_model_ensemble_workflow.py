from materia.python_api.api import Lab


def test_force_ensemble_persists_public_results_and_arrays():
    lab = Lab()
    surface = lab.materials.load("silicon").bulk()
    analysis = lab.ensembles.forces(
        surface, ["stillinger-weber-si", "lennard-jones"])
    run_id = analysis.metrics["run_id"]
    assert lab.ensembles.runs() == [run_id]
    assert lab.ensembles.result(run_id).extra["ensemble_kind"] == "forces"
    assert lab.ensembles.array(run_id).data.shape == (2, len(surface.structure), 3)


def test_energy_change_ensemble_keeps_input_structures_unchanged():
    lab = Lab()
    before = lab.materials.load("silicon").bulk()
    after = before.copy()
    after.structure.positions[0, 0] += 0.02
    before_positions = before.structure.positions.copy()
    after_positions = after.structure.positions.copy()
    analysis = lab.ensembles.energy_change(
        before, after, ["stillinger-weber-si", "lennard-jones"])
    run_id = analysis.metrics["run_id"]
    assert lab.ensembles.result(run_id, "energy_change").value == analysis.mean_change_eV
    assert (before.structure.positions == before_positions).all()
    assert (after.structure.positions == after_positions).all()
