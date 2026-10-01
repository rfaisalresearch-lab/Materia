"""Validation and deterministic numerical helpers for periodic phonons."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import phonon_dispersion as P


def settings(**changes):
    values = {
        "supercell": (3, 3, 3),
        "q_path": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        "dos_mesh": (4, 4, 4),
    }
    values.update(changes)
    return P.PeriodicPhononSettings(**values)


@pytest.mark.parametrize("change, fragment", [
    ({"supercell": (0, 2, 2)}, "supercell"),
    ({"supercell": (True, 2, 2)}, "supercell"),
    ({"q_path": ()}, "q_path"),
    ({"q_path": ((0.0, float("nan"), 0.0),)}, "finite"),
    ({"q_labels": ("G",)}, "q_labels"),
    ({"dos_mesh": (1, -1, 1)}, "dos_mesh"),
    ({"displacement_A": 0.5}, "displacement_A"),
    ({"asr_tolerance": 1.0}, "asr_tolerance"),
    ({"symmetrize_iterations": 0}, "symmetrize_iterations"),
    ({"symmetrize_iterations": 21}, "symmetrize_iterations"),
    ({"dos_points": 50}, "dos_points"),
    ({"dos_width_THz": 0.0}, "dos_width_THz"),
])
def test_settings_reject_invalid_values(change, fragment):
    with pytest.raises(P.PhononError, match=fragment):
        settings(**change)


def test_settings_are_canonical_and_immutable():
    configured = P.PeriodicPhononSettings(
        supercell=[3, 4, 5], q_path=[[0, 0, 0], [0.5, 0, 0]],
        dos_mesh=[2, 3, 4], q_labels=["G", "X"])
    assert configured.supercell == (3, 4, 5)
    assert configured.q_path == ((0, 0, 0), (0.5, 0, 0))
    assert configured.dos_mesh == (2, 3, 4)
    assert configured.q_labels == ("G", "X")
    with pytest.raises(AttributeError):
        configured.dos_points = 1001


def test_hard_limits_are_enforced_before_force_evaluation():
    cell = Cell.cubic(5.0)
    too_many = Structure(
        [14] * (P.MAX_PRIMITIVE_ATOMS + 1),
        np.zeros((P.MAX_PRIMITIVE_ATOMS + 1, 3)), cell)
    with pytest.raises(P.PhononRefused, match="primitive cell"):
        P._validate_structure_and_size(too_many, settings())
    one = Structure([14], [[0, 0, 0]], cell)
    huge = settings(supercell=(P.MAX_SUPERCELL_ATOMS + 1, 1, 1))
    with pytest.raises(P.PhononRefused, match="supercell"):
        P._validate_structure_and_size(one, huge)


def test_only_unconstrained_three_dimensional_periodic_cells_are_supported():
    slab = Structure([14], [[0, 0, 0]], Cell.cubic(5.0, pbc=(True, True, False)))
    with pytest.raises(P.PhononRefused, match="three-dimensional"):
        P._validate_structure_and_size(slab, settings())
    bulk = Structure([14], [[0, 0, 0]], Cell.cubic(5.0))
    bulk.fixed[0] = True
    with pytest.raises(P.PhononRefused, match="fixed atoms"):
        P._validate_structure_and_size(bulk, settings())


def test_finite_range_supercell_must_resolve_the_cutoff():
    structure = Structure([14], [[0, 0, 0]], Cell.cubic(2.0))
    with pytest.raises(P.PhononRefused, match="alias"):
        P._validate_cutoff(structure, (2, 2, 2), 2.1)
    P._validate_cutoff(structure, (3, 3, 3), 2.1)


def test_signed_dos_is_normalised_to_three_n_states():
    frequencies = np.array([[-1.0, 0.0, 2.0], [-0.5, 1.0, 3.0]])
    grid, density, weights = P._dos(frequencies, 801, 0.05)
    assert P._trapezoid(density, grid) == pytest.approx(3.0, abs=1e-12)
    assert weights.sum() == pytest.approx(3.0)
    assert weights[frequencies < 0].sum() == pytest.approx(1.0)
    assert grid[0] < -1.0 and grid[-1] > 3.0


def test_plain_position_callback_is_refused():
    structure = Structure([14], [[0, 0, 0]], Cell.cubic(5.0))
    with pytest.raises(P.PhononError, match="Materia potential or an ASE calculator"):
        P._source(lambda positions: np.zeros_like(positions), structure, "plain", None)
