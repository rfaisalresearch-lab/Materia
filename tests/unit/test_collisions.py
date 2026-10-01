"""Classical atomistic collision preparation and fragment analysis."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model import Structure
from materia.core_model.units import U_A2_FS2_TO_EV
from materia.elements import periodic_table as pt
from materia.experiments.collisions import CollisionError, fragments, prepare
from materia.python_api.api import ApiError, Lab


def atom(symbol: str) -> Structure:
    return Structure([pt.element(symbol).number], [[0.0, 0.0, 0.0]])


def test_prepare_places_exact_gap_and_conserves_momentum():
    projectile = Structure([29, 29], [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = Structure([29, 29], [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    speed = 0.08
    collision = prepare(projectile, target, speed, gap_A=3.25, impact_parameter_A=1.5)
    left = collision.positions[:2, 0]
    right = collision.positions[2:, 0]
    assert right.min() - left.max() == pytest.approx(3.25)
    momentum = (collision.masses()[:, None] * collision.velocities).sum(axis=0)
    np.testing.assert_allclose(momentum, 0.0, atol=1e-13)
    mass = projectile.masses().sum()
    expected = 0.5 * (mass / 2.0) * speed ** 2 * U_A2_FS2_TO_EV
    assert collision.info["collision"]["centre_of_mass_energy_eV"] == pytest.approx(expected)
    assert collision.roles.tolist() == ["projectile", "projectile", "target", "target"]
    assert collision.cell.pbc == (False, False, False)


def test_prepare_refuses_invalid_or_empty_inputs():
    with pytest.raises(CollisionError, match="at least one"):
        prepare(Structure(), atom("Cu"), 0.1)
    with pytest.raises(CollisionError, match="positive"):
        prepare(atom("Cu"), atom("Cu"), 0.0)
    with pytest.raises(CollisionError, match="finite"):
        prepare(atom("Cu"), atom("Cu"), float("nan"))


def test_fragments_are_sorted_and_report_formula_and_labels():
    structure = Structure(
        [29, 29, 79, 79, 79],
        [[0, 0, 0], [2.2, 0, 0], [10, 0, 0], [12.2, 0, 0], [14.4, 0, 0]])
    labels, pieces = fragments(structure, structure.positions)
    assert labels.tolist() == [1, 1, 0, 0, 0]
    assert [(piece["formula"], piece["atoms"]) for piece in pieces] == [
        ("Au3", 3), ("Cu2", 2)]


def test_python_namespace_prepares_owned_collision_and_refuses_unmarked_run():
    lab = Lab()
    collision = lab.collisions.prepare(atom("Cu"), atom("Cu"), 0.02)
    assert lab.project.structure is collision
    assert lab.project.history.log[-1].label == "Prepare atomistic collision"
    with pytest.raises(ApiError, match="no collision setup"):
        lab.collisions.run(atom("Cu"), steps=1)
