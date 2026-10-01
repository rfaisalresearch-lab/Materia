"""Neighbour lists, bond perception and interatomic potentials."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.materials import load
from materia.physics.bonds import perceive_bonds
from materia.physics.neighbors import neighbor_list, radial_distribution
from materia.physics.potentials import (
    FCC_ECOH_OVER_EPS,
    FCC_R0_OVER_SIGMA,
    Harmonic,
    LennardJones,
    StillingerWeber,
    UnsupportedSystem,
)
from materia.physics.strain import local_strain
from materia.structure_builder import bulk, make_surface


def test_neighbour_list_is_symmetric_and_complete():
    silicon = bulk(load("silicon"), (2, 2, 2))
    nl = neighbor_list(silicon.positions, silicon.cell, 2.6)
    assert len(nl) == 2 * len(nl.half())
    assert set(nl.counts()) == {4}
    forward = {(int(i), int(j)) for i, j in zip(nl.i, nl.j)}
    for i, j in forward:
        assert (j, i) in forward


def test_neighbour_list_handles_cells_thinner_than_twice_the_cutoff():
    thin = Structure([14], [[0, 0, 0]], Cell.cubic(2.0))
    nl = neighbor_list(thin.positions, thin.cell, 5.0)
    assert len(nl) > 0
    assert float(nl.d.min()) == pytest.approx(2.0)


def test_neighbour_list_respects_non_periodic_axes():
    slab = Structure([14, 14], [[0, 0, 0], [0, 0, 4.0]],
                     Cell(np.diag([3.0, 3.0, 5.0]), (True, True, False)))
    nl = neighbor_list(slab.positions, slab.cell, 4.5)
    distances = sorted(set(np.round(nl.d, 6)))
    assert 4.0 in distances
    assert 1.0 not in distances


def test_empty_structure_gives_an_empty_neighbour_list():
    nl = neighbor_list(np.zeros((0, 3)), Cell.none(), 3.0)
    assert len(nl) == 0


def test_radial_distribution_peaks_at_the_bond_length():
    silicon = bulk(load("silicon"), (3, 3, 3))
    nl = neighbor_list(silicon.positions, silicon.cell, 6.0)
    r, g = radial_distribution(nl, len(silicon), silicon.cell.volume, 6.0, 300)
    window = (r > 1.5) & (r < 3.0)
    first_peak = r[window][int(np.argmax(g[window]))]
    assert first_peak == pytest.approx(2.3517, abs=0.06)
    assert g[window].max() > 5.0


def test_bond_perception_matches_diamond_coordination():
    silicon = bulk(load("silicon"), (2, 2, 2))
    bonds = perceive_bonds(silicon)
    assert len(bonds) == 2 * len(silicon)
    assert all(b.origin.startswith("distance-heuristic") for b in bonds)
    assert all(b.order == 1.0 for b in bonds)


def test_stillinger_weber_reproduces_its_fitted_cohesive_energy():
    silicon = bulk(load("silicon"), (2, 2, 2))
    potential = StillingerWeber("Si")
    energy, forces = potential.energy_and_forces(silicon)
    assert energy / len(silicon) == pytest.approx(-2 * 2.1683, rel=1e-7)
    assert np.abs(forces).max() < 1e-10


def test_stillinger_weber_forces_match_numerical_gradients():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 2), vacuum_A=10.0)
    rng = np.random.default_rng(11)
    slab.positions = slab.positions + rng.normal(0, 0.08, slab.positions.shape)
    potential = StillingerWeber("Si")
    _, analytic = potential.energy_and_forces(slab)
    step = 1e-5
    for atom in (0, 3, len(slab) - 1):
        for axis in range(3):
            plus = slab.copy()
            plus.positions[atom, axis] += step
            minus = slab.copy()
            minus.positions[atom, axis] -= step
            numerical = -(potential.energy(plus) - potential.energy(minus)) / (2 * step)
            assert numerical == pytest.approx(analytic[atom, axis], abs=1e-6)


def test_stillinger_weber_rejects_foreign_species():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2))
    slab.substitute(int(slab.ids[0]), "P")
    ok, reason = StillingerWeber("Si").supports(slab)
    assert not ok
    assert "P" in reason and "impurities" in reason


def test_stillinger_weber_with_impurity_is_still_exact_for_the_pure_host():
    silicon = bulk(load("silicon"), (2, 2, 2))
    potential = StillingerWeber("Si", impurities=["P"])
    energy, _ = potential.energy_and_forces(silicon)
    assert energy / len(silicon) == pytest.approx(-2 * 2.1683, rel=1e-7)


def test_impurity_forces_match_numerical_gradients():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 2), vacuum_A=10.0)
    slab.substitute(int(slab.ids[len(slab) // 2]), "P")
    rng = np.random.default_rng(5)
    slab.positions = slab.positions + rng.normal(0, 0.05, slab.positions.shape)
    potential = StillingerWeber("Si", impurities=["P"])
    _, analytic = potential.energy_and_forces(slab)
    step = 1e-5
    for atom in (0, len(slab) // 2, len(slab) - 1):
        for axis in range(3):
            plus = slab.copy()
            plus.positions[atom, axis] += step
            minus = slab.copy()
            minus.positions[atom, axis] -= step
            numerical = -(potential.energy(plus) - potential.energy(minus)) / (2 * step)
            assert numerical == pytest.approx(analytic[atom, axis], abs=1e-6)


def test_impurity_approximation_is_declared_in_the_description():
    description = StillingerWeber("Si", impurities=["P"]).describe()
    text = " ".join(description["approximations"])
    assert "geometric approximation" in text
    assert "no electronic content" in text


def test_lennard_jones_parameters_are_derived_from_tabulated_data():
    gold = load("gold")
    potential = LennardJones.from_material(gold)
    epsilon, sigma = potential.params["Au"]
    assert "fcc" in potential.describe()["approximations"][-1]
    a = gold.lattice.parameters()[0]
    crystal = bulk(gold, (3, 3, 3))
    per_atom = potential.energy(crystal) / len(crystal)
    assert per_atom == pytest.approx(-3.81, abs=1e-9)
    step = 1e-4

    def at(scale):
        scaled = crystal.copy()
        scaled.positions = scaled.positions * scale
        scaled.cell = type(crystal.cell)(crystal.cell.matrix * scale, crystal.cell.pbc)
        return potential.energy(scaled) / len(crystal)

    slope = (at(1 + step / a) - at(1 - step / a)) / (2 * step)
    assert abs(slope) < 1e-6
    infinite = LennardJones({"Au": (3.81 / FCC_ECOH_OVER_EPS,
                                    a / np.sqrt(2.0) / FCC_R0_OVER_SIGMA)}, cutoff_A=1e3)
    assert infinite.params["Au"][1] == pytest.approx(sigma, rel=0.01)


def test_lennard_jones_forces_match_numerical_gradients():
    gold = bulk(load("gold"), (2, 2, 2))
    rng = np.random.default_rng(3)
    gold.positions = gold.positions + rng.normal(0, 0.05, gold.positions.shape)
    potential = LennardJones.from_material(load("gold"), cutoff_A=8.0)
    _, analytic = potential.energy_and_forces(gold)
    step = 1e-6
    for atom in (0, 5):
        for axis in range(3):
            plus = gold.copy()
            plus.positions[atom, axis] += step
            minus = gold.copy()
            minus.positions[atom, axis] -= step
            numerical = -(potential.energy(plus) - potential.energy(minus)) / (2 * step)
            assert numerical == pytest.approx(analytic[atom, axis], rel=1e-3, abs=1e-4)


def test_lennard_jones_reports_missing_parameters_by_element():
    gold = bulk(load("gold"))
    gold.substitute(int(gold.ids[0]), "Pt")
    ok, reason = LennardJones({"Au": (0.4, 2.6)}).supports(gold)
    assert not ok and "Pt" in reason


def test_harmonic_potential_is_analytic():
    reference = np.zeros((2, 3))
    structure = Structure([14, 14], [[0.1, 0, 0], [0, 0.2, 0]], Cell.none())
    potential = Harmonic(reference, k_eV_A2=3.0)
    energy, forces = potential.energy_and_forces(structure)
    assert energy == pytest.approx(0.5 * 3.0 * (0.01 + 0.04))
    assert forces[0] == pytest.approx([-0.3, 0, 0])
    assert potential.describe()["fidelity"] == "non-physical"


def test_local_strain_needs_a_reference():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2))
    result = local_strain(slab, None)
    assert not result.supported
    assert "reference" in result.unsupported_reason


def test_local_strain_recovers_an_applied_deformation():
    from materia.structure_builder.lattice import apply_strain

    slab = make_surface(load("silicon"), (1, 1, 1), size=(3, 3, 3), vacuum_A=10.0)
    strained = apply_strain(slab, [0.01, 0.01, 0.0])
    result = local_strain(strained, slab, cutoff_A=3.0)
    assert result.supported
    tensor = result.value["tensor"]
    interior = ~np.isnan(tensor[:, 0, 0])
    assert interior.sum() > 0
    assert np.nanmean(tensor[interior, 0, 0]) == pytest.approx(0.01005, abs=2e-3)
    assert np.nanmean(tensor[interior, 2, 2]) == pytest.approx(0.0, abs=2e-3)


def _coordination(structure, bonds):
    index = {int(v): k for k, v in enumerate(structure.ids)}
    counts = np.zeros(len(structure), dtype=int)
    for bond in bonds:
        counts[index[bond.a]] += 1
        counts[index[bond.b]] += 1
    return counts


@pytest.mark.parametrize("material_id, expected", [
    ("sapphire", {13: 6, 8: 4}), ("molybdenum_disulfide", {42: 6, 16: 3}),
    ("tungsten", {74: 8}), ("titanium", {22: 12}), ("silicon_carbide_4h", {14: 4, 6: 4}),
])
def test_screened_contacts_are_not_bonds(material_id, expected):
    crystal = bulk(load(material_id), (3, 3, 3))
    bonds = perceive_bonds(crystal)
    counts = _coordination(crystal, bonds)
    for z, value in expected.items():
        assert set(counts[crystal.numbers == z].tolist()) == {value}
    assert "Voronoi" in bonds[0].origin
    legacy = perceive_bonds(crystal, method="distance")
    assert len(legacy) >= len(bonds) and "Voronoi" not in legacy[0].origin


def test_molecules_keep_every_bond_and_degenerate_cases_say_so():
    ethanol = Structure(np.array([6, 6, 8, 1, 1, 1, 1, 1, 1]), np.array([
        [1.1680, -0.4007, 0.0], [0.0, 0.5530, 0.0], [-1.1854, -0.2340, 0.0],
        [-1.9303, 0.3834, 0.0], [2.1225, 0.1319, 0.0], [1.1297, -1.0480, 0.8836],
        [1.1297, -1.0480, -0.8836], [0.0398, 1.2029, 0.8819],
        [0.0398, 1.2029, -0.8819]]) + 6.0, Cell(np.eye(3) * 12.0, (False, False, False)))
    bonds = perceive_bonds(ethanol)
    assert len(bonds) == 8 and "Voronoi" in bonds[0].origin
    water = Structure(np.array([8, 1, 1]), np.array([[0, 0, 0.12], [0, 0.76, -0.47],
                                                     [0, -0.76, -0.47]]) + 5.0,
                      Cell(np.eye(3) * 10.0, (False, False, False)))
    plain = perceive_bonds(water)
    assert len(plain) == 2 and "Voronoi" not in plain[0].origin


def test_voronoi_filter_is_skipped_for_very_large_structures():
    crystal = bulk(load("tungsten"), (3, 3, 3))
    skipped = perceive_bonds(crystal, voronoi_max_atoms=10)
    assert "skipped above 10 atoms" in skipped[0].origin
    assert len(skipped) == len(perceive_bonds(crystal, method="distance"))
    with pytest.raises(ValueError):
        perceive_bonds(crystal, method="guess")
