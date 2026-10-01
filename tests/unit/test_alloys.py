"""Exact-composition alloys and finite partial-occupancy samples."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from materia.materials import default_library
from materia.python_api import Lab
from materia.structure_builder.alloys import (
    AlloyError,
    exact_counts,
    random_substitutional,
    sample_partial_occupancy,
)
from materia.structure_builder.lattice import bulk


def test_hamilton_counts_are_exact_and_stable():
    assert exact_counts(10, [0.33, 0.33, 0.34]).tolist() == [3, 3, 4]
    assert exact_counts(7, [0.5, 0.5]).tolist() == [4, 3]
    with pytest.raises(AlloyError, match="sums to 1"):
        exact_counts(10, [0.2, 0.2])


def test_random_alloy_is_seeded_exact_and_non_mutating():
    source = bulk(default_library().get("silicon"), (2, 2, 2))
    before = source.numbers.copy()
    first, record = random_substitutional(
        source, {"Si": 0.75, "Ge": 0.25}, seed=17)
    second, _ = random_substitutional(
        source, {"Si": 0.75, "Ge": 0.25}, seed=17)
    other, _ = random_substitutional(
        source, {"Si": 0.75, "Ge": 0.25}, seed=18)
    assert np.array_equal(source.numbers, before)
    assert np.array_equal(first.numbers, second.numbers)
    assert not np.array_equal(first.numbers, other.numbers)
    symbols, counts = np.unique(first.numbers, return_counts=True)
    measured = dict(zip(symbols.tolist(), counts.tolist()))
    assert measured == {14: 48, 32: 16}
    assert record["realized_counts"] == {"Si": 48, "Ge": 16}
    assert record["changed_sites"] == 16
    assert record["input_digest"]
    assert first.info["alloy"] == record


def test_selected_sublattice_and_charged_site_refusal():
    source = bulk(default_library().get("silicon"), (2, 1, 1))
    chosen = source.ids[:4]
    alloy, record = random_substitutional(
        source, {"Si": 0.5, "P": 0.5}, atom_ids=chosen, seed=2)
    assert record["site_count"] == 4
    assert np.count_nonzero(alloy.numbers[:4] == 15) == 2
    assert np.array_equal(alloy.numbers[4:], source.numbers[4:])
    source.formal_charges[source.indices_of(chosen)] = 1.0
    with pytest.raises(AlloyError, match="charge model"):
        random_substitutional(source, {"Si": 0.5, "P": 0.5}, atom_ids=chosen)


@pytest.mark.parametrize("composition, fragment", [
    ({}, "at least one"),
    ({"Si": 0.4, "Ge": 0.4}, "sum to 1"),
    ({"Si": 1.1, "Ge": -0.1}, "non-negative"),
])
def test_bad_compositions_are_refused(composition, fragment):
    with pytest.raises(AlloyError, match=fragment):
        random_substitutional(
            bulk(default_library().get("silicon")), composition)


def test_partial_occupancy_sampling_is_exact_for_representable_fraction():
    material = copy.deepcopy(default_library().get("silicon"))
    for site in material.basis:
        site.occupancy = 0.5
    sampled, record = sample_partial_occupancy(material, (2, 2, 2), seed=9)
    assert len(sampled) == 32
    assert record["removed_sites"] == 32
    assert all(group["occupied_sites"] == 4 for group in record["groups"])
    again, _ = sample_partial_occupancy(material, (2, 2, 2), seed=9)
    assert np.array_equal(sampled.ids, again.ids)
    assert np.array_equal(sampled.positions, again.positions)


def test_python_api_registers_alloy_and_requires_explicit_occupancy_seed():
    lab = Lab()
    silicon = lab.materials.load("silicon").bulk(repeat=(2, 2, 2))
    alloy = silicon.alloy({"Si": 0.75, "Ge": 0.25}, seed=4)
    assert lab.project.structure is alloy.structure
    assert len(lab.project.structures) == 2
    assert alloy.structure.info["alloy"]["realized_counts"] == {"Si": 48, "Ge": 16}
    material = copy.deepcopy(default_library().get("silicon"))
    material.id = "partial-test"
    material.basis[0].occupancy = 0.5
    sampled = lab.alloys.sample_occupancy(material, (2, 1, 1), seed=5)
    assert "partial_occupancy" in sampled.structure.info
