import numpy as np

from materia.python_api.api import Lab


def test_surface_comparison_persists_results_and_arrays():
    lab = Lab()
    before = lab.materials.load("silicon").bulk()
    after = before.copy()
    after.structure.positions[0, 0] += 0.1
    comparison = before.compare_to(after, compare_bonds=False)
    run_id = comparison.metrics["run_id"]
    assert lab.compare.runs() == [run_id]
    assert lab.compare.result(run_id).value > 0.0
    stored = lab.compare.array(run_id, "displacements")
    assert stored.data.shape == (len(before.structure), 3)
    assert stored.meta["run_id"] == run_id


def test_comparison_does_not_mutate_or_add_structures():
    lab = Lab()
    before = lab.materials.load("silicon").bulk()
    after = before.copy()
    after.structure.translate([0.2, 0.0, 0.0])
    before_positions = before.structure.positions.copy()
    after_positions = after.structure.positions.copy()
    keys = list(lab.project.structures)
    lab.compare.structures(
        before, after, mode="cartesian", remove_translation=True,
        compare_bonds=False)
    assert np.array_equal(before.structure.positions, before_positions)
    assert np.array_equal(after.structure.positions, after_positions)
    assert list(lab.project.structures) == keys
