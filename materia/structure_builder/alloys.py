"""Deterministic substitutional alloys and sampled partial occupancy."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..materials.schema import MaterialDefinition
from ..provenance import digest
from .lattice import bulk


class AlloyError(ValueError):
    """An alloy or occupancy request is invalid."""


def _composition(composition: Mapping[str, float]) -> Tuple[Sequence[str], np.ndarray]:
    if not composition:
        raise AlloyError("composition must name at least one element.")
    symbols = []
    fractions = []
    for symbol, fraction in composition.items():
        canonical = pt.symbol(pt.atomic_number(str(symbol)))
        if canonical in symbols:
            raise AlloyError(f"Element {canonical} appears more than once in composition.")
        value = float(fraction)
        if not math.isfinite(value) or value < 0:
            raise AlloyError(f"The fraction for {canonical} must be finite and non-negative.")
        symbols.append(canonical)
        fractions.append(value)
    values = np.asarray(fractions, dtype=float)
    if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise AlloyError(f"Composition fractions must sum to 1, not {values.sum():.12g}.")
    if np.count_nonzero(values) == 0:
        raise AlloyError("At least one composition fraction must be positive.")
    return symbols, values


def exact_counts(size: int, fractions: Sequence[float]) -> np.ndarray:
    """Hamilton apportionment of ``size`` sites with a stable tie break."""
    if type(size) is not int or size < 1:
        raise AlloyError("The number of sites must be a positive integer.")
    values = np.asarray(fractions, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(
            values < 0) or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise AlloyError("fractions must be a finite non-negative vector that sums to 1.")
    raw = size * values
    counts = np.floor(raw).astype(int)
    remaining = size - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remaining]] += 1
    return counts


def random_substitutional(
        structure: Structure, composition: Mapping[str, float], *,
        atom_ids: Optional[Iterable[int]] = None, seed: int = 0) -> Tuple[Structure, dict]:
    """Return a copy with selected sites assigned to an exact random composition."""
    symbols, fractions = _composition(composition)
    selected_ids = ([int(value) for value in atom_ids]
                    if atom_ids is not None else [int(value) for value in structure.ids])
    if not selected_ids:
        raise AlloyError("The alloy site selection is empty.")
    if len(set(selected_ids)) != len(selected_ids):
        raise AlloyError("The alloy site selection contains duplicate atom ids.")
    try:
        indices = structure.indices_of(selected_ids)
    except Exception as exc:
        raise AlloyError(str(exc)) from None
    if np.any(np.abs(structure.formal_charges[indices]) > 1e-12) or np.any(
            np.abs(structure.partial_charges[indices]) > 1e-12):
        raise AlloyError(
            "Charged sites cannot be randomly substituted without an explicit charge model.")
    counts = exact_counts(len(indices), fractions)
    assignments = np.concatenate([
        np.full(count, pt.atomic_number(symbol), dtype=np.int32)
        for symbol, count in zip(symbols, counts) if count])
    rng = np.random.default_rng(int(seed))
    rng.shuffle(assignments)
    output = structure.copy()
    original = output.numbers[indices].copy()
    output.numbers[indices] = assignments
    changed = original != assignments
    output.roles[indices[changed]] = "dopant"
    output.labels[indices] = np.asarray(
        [f"alloy-{pt.symbol(int(number))}" for number in assignments], dtype=object)
    output.invalidate_bonds()
    realized = {symbol: int(count) for symbol, count in zip(symbols, counts)}
    record = {
        "method": "seeded random substitution with exact Hamilton-apportioned counts",
        "seed": int(seed),
        "site_ids": selected_ids,
        "site_count": len(selected_ids),
        "target_fractions": {symbol: float(value)
                             for symbol, value in zip(symbols, fractions)},
        "realized_counts": realized,
        "realized_fractions": {symbol: count / len(selected_ids)
                               for symbol, count in realized.items()},
        "changed_sites": int(changed.sum()),
        "input_digest": digest({
            "numbers": structure.numbers.tolist(),
            "positions": np.round(structure.positions, 10).tolist(),
            "cell": structure.cell.matrix.tolist(), "pbc": list(structure.cell.pbc),
            "site_ids": selected_ids,
            "composition": {symbol: float(value)
                            for symbol, value in zip(symbols, fractions)},
            "seed": int(seed),
        }),
        "limitations": [
            "Random substitution is not a special quasirandom structure optimisation.",
            "No local relaxation, charge redistribution or magnetic-moment model is applied.",
        ],
    }
    output.info["alloy"] = record
    provenance = output.info.get("provenance")
    if isinstance(provenance, dict):
        provenance["model"] = "structure-builder/random-substitutional-alloy"
        provenance["origin"] = "estimated"
        provenance.setdefault("approximations", []).extend(record["limitations"])
        provenance.setdefault("parameters", {})["alloy"] = {
            "seed": int(seed), "target_fractions": record["target_fractions"],
            "realized_counts": realized, "site_ids": selected_ids}
        provenance["inputs_digest"] = record["input_digest"]
    return output, record


def sample_partial_occupancy(
        material: MaterialDefinition, repeat: Sequence[int], *, seed: int = 0
        ) -> Tuple[Structure, dict]:
    """Sample scalar basis-site occupancies over a repeated conventional cell."""
    raw_repeat = tuple(repeat)
    if len(raw_repeat) != 3 or any(int(value) != value or int(value) < 1
                                   for value in raw_repeat):
        raise AlloyError("repeat must contain three positive integers.")
    repeat_tuple = tuple(int(value) for value in raw_repeat)
    occupancies = np.asarray([float(site.occupancy) for site in material.basis])
    if not np.all(np.isfinite(occupancies)) or np.any(occupancies < 0) or np.any(
            occupancies > 1):
        raise AlloyError("Every basis occupancy must be finite and lie between 0 and 1.")
    structure = bulk(material, repeat_tuple)
    cells = int(np.prod(repeat_tuple))
    basis_size = len(material.basis)
    rng = np.random.default_rng(int(seed))
    remove_ids = []
    groups = []
    for basis_index, site in enumerate(material.basis):
        indices = np.arange(basis_index, len(structure), basis_size)
        expected = cells * occupancies[basis_index]
        keep_count = int(math.floor(expected))
        if rng.random() < expected - keep_count:
            keep_count += 1
        order = rng.permutation(indices)
        removed = order[keep_count:]
        remove_ids.extend(int(structure.ids[index]) for index in removed)
        groups.append({
            "basis_index": basis_index,
            "label": site.label or site.element,
            "element": site.element,
            "target_occupancy": float(occupancies[basis_index]),
            "available_sites": cells,
            "occupied_sites": keep_count,
            "realized_occupancy": keep_count / cells,
        })
    if len(remove_ids) == len(structure):
        raise AlloyError("The occupancy sample removed every atom.")
    if remove_ids:
        structure.remove_atoms(remove_ids)
    record = {
        "method": "seeded finite-supercell sampling of independent crystallographic sites",
        "seed": int(seed), "repeat": list(repeat_tuple), "groups": groups,
        "removed_sites": len(remove_ids), "remaining_atoms": len(structure),
        "input_digest": digest({
            "material": material.id, "repeat": list(repeat_tuple),
            "occupancies": occupancies.tolist(), "seed": int(seed),
        }),
        "limitations": [
            "Each crystallographic occupancy is sampled independently.",
            "No short-range occupational correlations or charge-balance constraints are inferred.",
            "A finite supercell can only realize rational occupancies at its site count.",
        ],
    }
    structure.info["partial_occupancy"] = record
    provenance = structure.info.get("provenance")
    if isinstance(provenance, dict):
        provenance["model"] = "structure-builder/partial-occupancy-sample"
        provenance["origin"] = "estimated"
        provenance.setdefault("approximations", []).extend(record["limitations"])
        provenance.setdefault("parameters", {})["occupancy_sample"] = {
            "seed": int(seed), "groups": groups}
        provenance["inputs_digest"] = record["input_digest"]
    return structure, record
