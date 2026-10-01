"""Periodic neighbour lists: translation invariance, images and coincidence."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.materials import load
from materia.physics.neighbors import (
    COLLISION_TOLERANCE_A, CoincidentAtoms, neighbor_list,
)
from materia.physics.potentials import LennardJones, OverlappingAtoms, StillingerWeber
from materia.solvers.classical import ClassicalSolver
from materia.structure_builder import bulk

CELLS = {
    "cubic": Cell.cubic(5.43),
    "triclinic": Cell.from_parameters(4.1, 4.7, 5.2, 78.0, 96.0, 113.0),
    "hexagonal": Cell.from_parameters(3.2, 3.2, 5.1, 90.0, 90.0, 120.0),
    "slab": Cell(np.array([[4.0, 0.0, 0.0], [1.3, 4.2, 0.0], [0.0, 0.0, 30.0]]),
                 (True, True, False)),
    "thin": Cell(np.array([[1.9, 0.0, 0.0], [0.4, 2.1, 0.0], [0.2, 0.3, 2.3]])),
}


def random_positions(cell: Cell, n: int = 7, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frac = rng.uniform(0.0, 1.0, (n, 3))
    if not cell.pbc[2]:
        frac[:, 2] = rng.uniform(0.3, 0.5, n)
    return frac @ cell.matrix


def signature(nl) -> list:
    return sorted((int(i), int(j), round(float(d), 9)) for i, j, d in zip(nl.i, nl.j, nl.d))


def brute_force(pos: np.ndarray, cell: Cell, cutoff: float) -> list:
    reach = [4 if p else 0 for p in cell.pbc]
    out = []
    for i, j in itertools.product(range(len(pos)), repeat=2):
        for shift in itertools.product(*[range(-r, r + 1) for r in reach]):
            if i == j and not any(shift):
                continue
            d = np.linalg.norm(pos[j] + np.array(shift) @ cell.matrix - pos[i])
            if d < cutoff:
                out.append((i, j, round(float(d), 9)))
    return sorted(out)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_list_matches_brute_force_including_thin_cells(name):
    cell = CELLS[name]
    pos = random_positions(cell)
    assert signature(neighbor_list(pos, cell, 4.5)) == brute_force(pos, cell, 4.5)


@pytest.mark.parametrize("name", sorted(CELLS))
@pytest.mark.parametrize("translation", [(1, 0, 0), (2, 0, 0), (0, -3, 0), (-2, 1, 0),
                                         (3, -2, 0), (0, 0, 2), (-4, 5, -3)])
def test_integer_translations_change_nothing(name, translation):
    cell = CELLS[name]
    shift = np.array([t if cell.pbc[a] else 0 for a, t in enumerate(translation)])
    pos = random_positions(cell)
    ref = neighbor_list(pos, cell, 4.5)
    moved = pos.copy()
    moved[2] += shift @ cell.matrix
    moved[5] -= 2 * shift @ cell.matrix
    out = neighbor_list(moved, cell, 4.5)
    assert signature(out) == signature(ref)
    np.testing.assert_allclose(
        moved[out.j] + out.shifts.astype(float) @ cell.matrix - moved[out.i], out.D,
        atol=1e-12)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_no_duplicate_pairs_and_consistent_half_list(name):
    cell = CELLS[name]
    pos = random_positions(cell)
    pos[1] += np.array([3, -2, 0]) @ cell.matrix * np.array(cell.pbc, dtype=float)[:, None].T[0]
    full = neighbor_list(pos, cell, 4.5)
    keys = list(zip(full.i.tolist(), full.j.tolist(), map(tuple, full.shifts.tolist())))
    assert len(keys) == len(set(keys))
    half = neighbor_list(pos, cell, 4.5, half=True)
    assert 2 * len(half) == len(full)
    unordered = {(min(i, j), max(i, j), round(float(d), 9))
                 for i, j, d in zip(half.i, half.j, half.d)}
    assert len(unordered) <= len(half)


def test_true_self_pair_is_excluded_by_index_and_image():
    cell = Cell.cubic(2.0)
    nl = neighbor_list(np.zeros((1, 3)), cell, 2.1)
    assert len(nl) == 6 and np.all(nl.i == 0) and np.all(nl.j == 0)
    assert not np.any(np.all(nl.shifts == 0, axis=1))


@pytest.mark.parametrize("separation", [0.0, 1e-10, 1e-8, COLLISION_TOLERANCE_A])
def test_distinct_coincident_atoms_are_refused_or_excluded_on_request(separation):
    pos = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(CoincidentAtoms):
        neighbor_list(pos, Cell.none(), 3.0)
    kept = neighbor_list(pos, Cell.none(), 3.0, coincident="exclude")
    assert not np.any(((kept.i == 0) & (kept.j == 1)) | ((kept.i == 1) & (kept.j == 0)))
    ok = neighbor_list(np.array([[0.0, 0, 0], [2 * COLLISION_TOLERANCE_A, 0, 0]]),
                       Cell.none(), 3.0)
    assert len(ok) == 2


def test_periodic_image_coincidence_is_refused():
    cell = Cell.cubic(4.0)
    with pytest.raises(CoincidentAtoms) as caught:
        neighbor_list(np.array([[0.0, 0, 0], [8.0, 0, 0]]), cell, 3.0)
    assert caught.value.lattice_shift == (-2, 0, 0)


@pytest.fixture(scope="module")
def silicon64():
    return bulk(load("silicon")).repeat(2, 2, 2)


@pytest.mark.parametrize("potential", [StillingerWeber("Si"),
                                       LennardJones({"Si": (0.5378, 2.1571)}, cutoff_A=6.0)])
@pytest.mark.parametrize("n", [1, 2, 3, -1, -2, -3])
def test_potentials_are_invariant_under_lattice_translations(silicon64, potential, n):
    assert len(silicon64) == 64
    e0, f0 = potential.energy_and_forces(silicon64)
    for axis in range(3):
        t = silicon64.copy()
        q = t.positions.copy()
        q[3] += n * silicon64.cell.matrix[axis]
        q[17] -= n * silicon64.cell.matrix[(axis + 1) % 3]
        t.positions = q
        e, f = potential.energy_and_forces(t)
        assert e == pytest.approx(e0, abs=1e-9)
        np.testing.assert_allclose(f, f0, atol=1e-9)


@pytest.mark.parametrize("potential", [StillingerWeber("Si"),
                                       LennardJones({"Si": (0.5378, 2.1571)})])
def test_potentials_refuse_coincident_atoms(silicon64, potential):
    t = silicon64.copy()
    q = t.positions.copy()
    q[5] = q[4] + silicon64.cell.matrix[0]
    t.positions = q
    ok, why = potential.supports(t)
    assert not ok and "collision tolerance" in why and "periodic image" in why
    with pytest.raises(OverlappingAtoms):
        potential.energy_and_forces(t)
    solver = ClassicalSolver(potential)
    assert not solver.single_point(t)["energy"].supported
    assert not solver.relax(t)["relaxed_structure"].supported


def test_dynamics_across_periodic_boundaries_has_no_jumps(silicon64):
    """Every atom drifts through the cell several times; stored coordinates are
    never wrapped, so they end several lattice vectors outside it.

    The bounds allow ordinary velocity-Verlet fluctuation at 1 fs, a few meV.
    A neighbour lost to coordinate drift changes the energy by electronvolts.
    """
    potential = StillingerWeber("Si")
    s = silicon64.copy()
    rng = np.random.default_rng(1)
    s.velocities = rng.normal(0.0, 0.004, s.positions.shape) + np.array([0.05, -0.04, 0.03])
    out = ClassicalSolver(potential).dynamics(s, steps=1200, dt_fs=1.0, thermostat="none",
                                              initialise_velocities=False, sample_every=10)
    final = out.results["final_structure"].value
    frac = final.cell.to_fractional(final.positions)
    assert np.abs(frac).max() > 3.0
    total = np.array(out.results["trajectory"].value["total_eV"])
    assert np.abs(np.diff(total)).max() < 0.02
    assert np.ptp(total) < 0.02
    wrapped = final.copy()
    wrapped.positions = final.cell.wrap(final.positions)
    e_stored, f_stored = potential.energy_and_forces(final)
    e_wrapped, f_wrapped = potential.energy_and_forces(wrapped)
    assert e_stored == pytest.approx(e_wrapped, abs=1e-9)
    np.testing.assert_allclose(f_stored, f_wrapped, atol=1e-9)
