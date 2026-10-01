"""Local strain from neighbour-shell deformation.

The local strain of atom *i* is obtained by least-squares fitting the affine
transform ``F`` that maps the reference bond vectors of *i* onto the current
ones, then reporting the Green-Lagrange strain ``E = (F^T F - I)/2``.  This is
the standard atomistic "local deformation gradient" measure.

It requires a reference configuration.  Without one, the function returns an
unsupported result rather than inventing a reference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..provenance import Fidelity, Origin, Provenance, Result, unsupported
from .neighbors import neighbor_list


def local_strain(
    current: Structure,
    reference: Optional[Structure],
    cutoff_A: float = 3.2,
    min_neighbours: int = 3,
) -> Result:
    """Per-atom Green-Lagrange strain tensor and von Mises scalar."""
    if reference is None:
        return unsupported(
            "local_strain", "physics/local-strain",
            "Local strain is defined relative to a reference configuration; none was given.",
            suggested_models=["Save a checkpoint before deforming, then pass it as the reference."],
            unit="dimensionless",
        )
    if len(reference) != len(current):
        return unsupported(
            "local_strain", "physics/local-strain",
            f"Reference has {len(reference)} atoms but the current structure has "
            f"{len(current)}; atom-by-atom strain is undefined after adding or removing atoms.",
            unit="dimensionless",
        )

    ref_nl = neighbor_list(reference.positions, reference.cell, cutoff_A,
                           coincident="exclude")
    n = len(current)
    strain = np.full((n, 3, 3), np.nan)
    von_mises = np.full(n, np.nan)
    order = np.argsort(ref_nl.i, kind="stable")
    i_s = ref_nl.i[order]
    counts = np.bincount(i_s, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    Dref = ref_nl.D[order]
    j_s = ref_nl.j[order]
    shifts = ref_nl.shifts[order]

    cur_cell = current.cell.matrix
    for a in range(n):
        s, e = starts[a], starts[a + 1]
        if e - s < min_neighbours:
            continue
        dref = Dref[s:e]
        dcur = (current.positions[j_s[s:e]] + shifts[s:e].astype(float) @ cur_cell
                - current.positions[a])
        F, *_ = np.linalg.lstsq(dref, dcur, rcond=None)
        F = F.T
        E = 0.5 * (F.T @ F - np.eye(3))
        strain[a] = E
        dev = E - np.trace(E) / 3.0 * np.eye(3)
        von_mises[a] = np.sqrt(2.0 / 3.0 * np.sum(dev * dev))

    return Result(
        name="local_strain",
        value={"tensor": strain, "von_mises": von_mises},
        unit="dimensionless",
        provenance=Provenance(
            model="physics/local-strain",
            fidelity=Fidelity.TIER0_STRUCTURAL,
            origin=Origin.CALCULATED,
            approximations=[
                "Affine fit of the first neighbour shell; not valid where the "
                "neighbour topology changes (bond breaking, large shear).",
                f"Neighbour shell defined by a {cutoff_A} A cutoff in the reference.",
            ],
            boundary_conditions=str(current.cell.pbc),
            parameters={"cutoff_A": cutoff_A, "min_neighbours": min_neighbours},
            references=["M. L. Falk and J. S. Langer, Phys. Rev. E 57 (1998) 7192"],
        ),
        extra={"atoms_without_strain": int(np.isnan(von_mises).sum())},
    )
