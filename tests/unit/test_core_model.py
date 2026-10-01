"""Cells, structures, atoms and selections."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model import Cell, Selection, Structure, StructureError
from materia.core_model.selection import (
    by_element,
    by_role,
    in_box,
    near_plane,
    neighbors_within,
    within_radius,
)


def test_cell_from_parameters_round_trips():
    cell = Cell.from_parameters(3.19, 3.19, 5.19, 90, 90, 120)
    assert cell.lengths == pytest.approx([3.19, 3.19, 5.19], rel=1e-9)
    assert cell.angles_deg == pytest.approx([90, 90, 120], abs=1e-9)
    assert cell.volume == pytest.approx(3.19 * 3.19 * 5.19 * np.sin(np.radians(120)))


def test_reciprocal_lattice_is_dual():
    cell = Cell.from_parameters(4.0, 5.0, 6.0, 80, 95, 110)
    product = cell.reciprocal @ cell.matrix.T / (2 * np.pi)
    assert product == pytest.approx(np.eye(3), abs=1e-12)


def test_fractional_and_cartesian_are_inverse():
    cell = Cell.from_parameters(4.0, 5.0, 6.0, 80, 95, 110)
    points = np.random.default_rng(0).random((20, 3)) * 4
    assert cell.to_cartesian(cell.to_fractional(points)) == pytest.approx(points)


def test_wrap_only_affects_periodic_axes():
    cell = Cell(np.diag([5.0, 5.0, 20.0]), (True, True, False))
    wrapped = cell.wrap(np.array([[6.0, -1.0, 25.0]]))
    assert wrapped[0][0] == pytest.approx(1.0)
    assert wrapped[0][1] == pytest.approx(4.0)
    assert wrapped[0][2] == pytest.approx(25.0)


def test_minimum_image_convention():
    cell = Cell.cubic(5.0)
    assert cell.minimum_image(np.array([[4.0, 0, 0]]))[0][0] == pytest.approx(-1.0)


def test_atom_ids_are_stable_across_edits():
    s = Structure([14, 14, 14], [[0, 0, 0], [2, 0, 0], [4, 0, 0]], Cell.cubic(20))
    original = list(s.ids)
    s.remove_atoms([original[1]])
    assert list(s.ids) == [original[0], original[2]]
    new_id = s.add_atom("P", [6, 0, 0])
    assert new_id not in original
    assert s.atom(original[2]).position[0] == pytest.approx(4.0)


def test_removing_an_unknown_id_is_rejected():
    s = Structure([14], [[0, 0, 0]], Cell.cubic(10))
    with pytest.raises(StructureError):
        s.remove_atoms([999])


def test_substitution_preserves_identity_and_position():
    s = Structure([14, 14], [[0, 0, 0], [2.35, 0, 0]], Cell.cubic(10))
    target = int(s.ids[1])
    s.substitute(target, "P")
    assert s.atom(target).symbol == "P"
    assert s.atom(target).position[0] == pytest.approx(2.35)
    assert s.formula() == "PSi"


def test_charge_and_electron_bookkeeping():
    s = Structure([14, 8], [[0, 0, 0], [2, 0, 0]], Cell.cubic(10))
    assert s.total_electrons() == pytest.approx(22)
    s.atom(int(s.ids[0])).charge = 2
    assert s.total_charge() == pytest.approx(2.0)
    assert s.total_electrons() == pytest.approx(20)


def test_isotope_mass_is_used_when_assigned():
    s = Structure([14], [[0, 0, 0]], Cell.cubic(10))
    assert s.masses()[0] == pytest.approx(28.085, rel=1e-6)
    s.mass_numbers[0] = 30
    assert s.masses()[0] == pytest.approx(29.9737701, rel=1e-7)


def test_geometry_measurements():
    s = Structure([14, 14, 14], [[0, 0, 0], [1, 0, 0], [1, 1, 0]], Cell.none())
    a, b, c = (int(i) for i in s.ids)
    assert s.distance(a, b) == pytest.approx(1.0)
    assert s.angle(a, b, c) == pytest.approx(90.0)


def test_distance_uses_minimum_image_for_periodic_cells():
    s = Structure([14, 14], [[0.2, 0, 0], [4.8, 0, 0]], Cell.cubic(5.0))
    ids = [int(i) for i in s.ids]
    assert s.distance(*ids) == pytest.approx(0.4)
    assert s.distance(*ids, mic=False) == pytest.approx(4.6)


def test_repeat_scales_cell_and_atom_count():
    s = Structure([14], [[0, 0, 0]], Cell.cubic(5.0))
    big = s.repeat(2, 3, 4)
    assert len(big) == 24
    assert big.cell.lengths == pytest.approx([10, 15, 20])
    assert len(set(int(i) for i in big.ids)) == 24


def test_serialisation_round_trip_preserves_everything():
    s = Structure([14, 15], [[0, 0, 0], [2.3, 0, 0]], Cell.cubic(10),
                  roles=["surface", "dopant"], labels=["Si(a)", "P@Si"])
    s.formal_charges[1] = -1.0
    s.magnetic_moments[0] = 1.0
    s.mass_numbers[0] = 29
    s.atom_meta[int(s.ids[0])] = {"note": "kept"}
    restored = Structure.from_dict(s.as_dict())
    assert list(restored.ids) == list(s.ids)
    assert restored.positions == pytest.approx(s.positions)
    assert list(restored.roles) == list(s.roles)
    assert restored.formal_charges == pytest.approx(s.formal_charges)
    assert restored.mass_numbers[0] == 29
    assert restored.atom_meta[int(s.ids[0])]["note"] == "kept"


def test_selection_set_algebra():
    s = Structure([14] * 4 + [15], [[i, 0, 0] for i in range(5)], Cell.cubic(20))
    silicon = by_element(s, "Si")
    phosphorus = by_element(s, "P")
    assert len(silicon) == 4 and len(phosphorus) == 1
    assert len(silicon.union(phosphorus)) == 5
    assert len(silicon.intersection(phosphorus)) == 0
    assert len(silicon.invert(s)) == 1
    assert "element" in silicon.query


def test_selection_prunes_deleted_atoms():
    s = Structure([14, 14], [[0, 0, 0], [2, 0, 0]], Cell.cubic(10))
    selection = Selection([int(i) for i in s.ids], "all")
    s.remove_atoms([int(s.ids[0])])
    assert len(selection.prune(s)) == 1


def test_selected_one_requires_a_singleton():
    s = Structure([14, 14], [[0, 0, 0], [2, 0, 0]], Cell.cubic(10))
    with pytest.raises(StructureError):
        Selection([int(i) for i in s.ids]).one(s)


def test_spatial_selectors():
    positions = [[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]]
    s = Structure([14] * 4, positions, Cell.cubic(30), roles=["surface"] * 2 + ["bulk"] * 2)
    assert len(within_radius(s, [0, 0, 0], 3.5)) == 4
    assert len(within_radius(s, [0, 0, 0], 1.0)) == 1
    assert len(in_box(s, [-1, -1, -1], [1, 1, 1])) == 1
    assert len(by_role(s, "surface")) == 2
    assert len(neighbors_within(s, int(s.ids[0]), 3.5)) == 3


def test_plane_selection_uses_reciprocal_lattice():
    s = Structure([14, 14], [[0, 0, 0], [0, 0, 2.5]], Cell.cubic(5.0))
    selection = near_plane(s, (0, 0, 1), 0.0, 0.2)
    assert len(selection) == 1
