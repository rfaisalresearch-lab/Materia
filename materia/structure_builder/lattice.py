"""Bulk crystal generation from material definitions."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..materials.schema import MaterialDefinition
from ..provenance import Fidelity, Origin, Provenance


def conventional_cell(material: MaterialDefinition) -> Cell:
    a, b, c, alpha, beta, gamma = material.lattice.parameters()
    return Cell.from_parameters(a, b, c, alpha, beta, gamma)


def bulk(
    material: MaterialDefinition,
    repeat: Sequence[int] = (1, 1, 1),
    *,
    strain: Optional[np.ndarray] = None,
) -> Structure:
    """Build the conventional bulk cell of a material, optionally repeated.

    ``strain`` is an optional 3x3 engineering strain tensor applied to the
    cell (and affinely to the atoms).
    """
    cell = conventional_cell(material)
    frac = np.array([s.fractional for s in material.basis], dtype=float)
    numbers = []
    labels = []
    from ..elements import periodic_table as pt
    for s in material.basis:
        numbers.append(pt.atomic_number(s.element))
        labels.append(s.label or s.element)
    positions = cell.to_cartesian(frac)

    st = Structure(numbers, positions, cell, labels=labels,
                   roles=["bulk"] * len(numbers))
    st.info["material_id"] = material.id
    st.info["material_name"] = material.name
    st.info["prototype"] = material.prototype
    st.info["space_group"] = material.space_group_symbol
    st.info["provenance"] = Provenance(
        model="structure-builder/bulk",
        fidelity=Fidelity.TIER0_STRUCTURAL,
        origin=Origin.REFERENCE,
        approximations=["Ideal, defect-free lattice at the tabulated lattice parameter."],
        boundary_conditions="3D periodic",
        parameters={"material": material.id, "repeat": list(repeat)},
        references=list(material.references),
        dataset=material.source_path,
        dataset_license=material.license,
    ).as_dict()

    nx, ny, nz = (int(v) for v in repeat)
    if (nx, ny, nz) != (1, 1, 1):
        st = st.repeat(nx, ny, nz)
        st.info["provenance"]["parameters"]["repeat"] = [nx, ny, nz]

    if strain is not None:
        st = apply_strain(st, strain)
    return st


def apply_strain(structure: Structure, strain: np.ndarray) -> Structure:
    """Apply a homogeneous engineering strain tensor to cell and atoms."""
    e = np.asarray(strain, dtype=float)
    if e.shape == (3,):
        e = np.diag(e)
    if e.shape != (3, 3):
        raise ValueError("strain must be a 3-vector of normal strains or a 3x3 tensor")
    F = np.eye(3) + e
    out = structure.copy()
    out.cell = Cell(structure.cell.matrix @ F.T, structure.cell.pbc)
    out.positions = structure.positions @ F.T
    out.info.setdefault("applied_strain", []).append(e.tolist())
    return out


def lattice_planes(material: MaterialDefinition, miller: Sequence[int]) -> float:
    """Interplanar spacing d_hkl in angstrom for the conventional cell."""
    cell = conventional_cell(material)
    g = np.asarray(miller, dtype=float) @ cell.reciprocal
    n = np.linalg.norm(g)
    if n == 0:
        raise ValueError("(000) has no interplanar spacing")
    return float(2.0 * np.pi / n)
