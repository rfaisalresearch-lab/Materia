import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics.structure_comparison import (
    StructureComparisonError,
    compare_structures,
)


def test_minimum_image_displacement_crosses_periodic_boundary():
    cell = Cell.cubic(10.0)
    before = Structure([14], np.array([[9.8, 0.0, 0.0]]), cell, ids=[7])
    after = Structure([14], np.array([[0.2, 0.0, 0.0]]), cell, ids=[7])
    comparison = compare_structures(
        before, after, mode="minimum-image", compare_bonds=False)
    assert comparison.displacements_A[0].tolist() == pytest.approx([0.4, 0.0, 0.0])
    assert comparison.metrics["rms_displacement_A"] == pytest.approx(0.4)


def test_fractional_mode_separates_cell_deformation_from_atomic_motion():
    before = Structure(
        [14], np.array([[1.0, 1.0, 1.0]]), Cell.cubic(4.0), ids=[3])
    after = Structure(
        [14], np.array([[2.0, 2.0, 2.0]]), Cell.cubic(8.0), ids=[3])
    comparison = compare_structures(before, after, compare_bonds=False)
    assert comparison.metrics["mode"] == "fractional"
    assert np.allclose(comparison.displacements_A, 0.0)
    assert np.allclose(comparison.deformation_gradient, 2.0 * np.eye(3))
    assert np.allclose(comparison.green_lagrange_strain, 1.5 * np.eye(3))
    assert comparison.metrics["relative_volume_change"] == pytest.approx(7.0)


def test_rigid_alignment_removes_cluster_rotation_and_translation():
    before_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    after_positions = before_positions @ rotation + np.array([4.0, -3.0, 2.0])
    before = Structure([6, 6, 6], before_positions, ids=[1, 2, 3])
    after = Structure([6, 6, 6], after_positions, ids=[1, 2, 3])
    comparison = compare_structures(before, after, compare_bonds=False)
    assert comparison.metrics["mode"] == "rigid-align"
    assert comparison.metrics["rms_displacement_A"] == pytest.approx(0.0, abs=1e-12)


def test_added_removed_and_substituted_atoms_are_reported_by_id():
    before = Structure([14, 14], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), ids=[4, 5])
    after = Structure([6, 14], np.array([[0.1, 0.0, 0.0], [3.0, 0.0, 0.0]]), ids=[4, 6])
    comparison = compare_structures(before, after, mode="cartesian", compare_bonds=False)
    assert comparison.atom_ids.tolist() == [4]
    assert comparison.added_atom_ids == [6]
    assert comparison.removed_atom_ids == [5]
    assert comparison.element_changes == [{
        "atom_id": 4,
        "before_atomic_number": 14,
        "after_atomic_number": 6,
    }]


def test_heuristic_bond_changes_are_explicit():
    before = Structure([1, 1], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), ids=[1, 2])
    after = Structure([1, 1], np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]), ids=[1, 2])
    comparison = compare_structures(before, after, mode="cartesian")
    assert [(bond["a"], bond["b"]) for bond in comparison.bonds_formed] == [(1, 2)]
    assert comparison.bonds_broken == []
    assert "heuristic" in comparison.provenance.approximations[0]


def test_comparison_refuses_no_common_ids():
    before = Structure([1], np.zeros((1, 3)), ids=[1])
    after = Structure([1], np.zeros((1, 3)), ids=[2])
    with pytest.raises(StructureComparisonError, match="no stable atom ids"):
        compare_structures(before, after)


def test_minimum_image_refuses_changed_cell():
    before = Structure([1], np.zeros((1, 3)), Cell.cubic(4.0), ids=[1])
    after = Structure([1], np.zeros((1, 3)), Cell.cubic(5.0), ids=[1])
    with pytest.raises(StructureComparisonError, match="identical cells"):
        compare_structures(before, after, mode="minimum-image")
