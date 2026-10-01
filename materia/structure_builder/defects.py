"""Point-defect construction.

Every operation here records what it did in ``structure.info['defects']`` so
that the project history and provenance panel can replay it.  None of these
functions computes a formation energy: that requires a solver, and the
defect-energy workflow in :mod:`materia.solvers` reports it with its own
provenance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..core_model.structure import Structure, StructureError
from ..elements import periodic_table as pt


def _log(structure: Structure, entry: dict) -> None:
    structure.info.setdefault("defects", []).append(entry)


def nearest_site(structure: Structure, point: Sequence[float],
                 element: Optional[str] = None) -> int:
    """Id of the atom closest to ``point`` (optionally of a given element)."""
    p = np.asarray(point, dtype=float)
    d = structure.positions - p
    if structure.cell.is_periodic:
        d = structure.cell.minimum_image(d)
    r = np.linalg.norm(d, axis=1)
    if element is not None:
        z = pt.atomic_number(element)
        mask = structure.numbers == z
        if not mask.any():
            raise StructureError(f"No {element} atoms in this structure")
        r = np.where(mask, r, np.inf)
    return int(structure.ids[int(np.argmin(r))])


def create_vacancy(structure: Structure, atom_id: int, *, mark_neighbours: bool = True) -> dict:
    """Remove an atom, creating a vacancy.  Returns a record of what was removed."""
    k = structure.index_of(atom_id)
    record = {
        "type": "vacancy",
        "removed_id": int(atom_id),
        "element": pt.symbol(int(structure.numbers[k])),
        "position_A": structure.positions[k].tolist(),
        "label": str(structure.labels[k]),
        "role": str(structure.roles[k]),
    }
    if mark_neighbours:
        from ..physics.bonds import perceive_bonds
        if structure.bonds is None:
            structure.set_bonds(perceive_bonds(structure))
        record["neighbour_ids"] = [int(i) for i in structure.neighbors_of(atom_id)]
    structure.remove_atoms([atom_id])
    if mark_neighbours:
        for nid in record.get("neighbour_ids", []):
            if structure.has_id(nid):
                structure.roles[structure.index_of(nid)] = "vacancy-neighbour"
    _log(structure, record)
    return record


def substitute_atom(structure: Structure, atom_id: int, element: str, *,
                    role: str = "dopant", mass_number: int = 0) -> dict:
    """Replace the element at a lattice site, keeping its id and position."""
    k = structure.index_of(atom_id)
    old = pt.symbol(int(structure.numbers[k]))
    structure.substitute(atom_id, element, role=role, mass_number=mass_number)
    structure.labels[k] = f"{element}@{old}"
    record = {
        "type": "substitution",
        "atom_id": int(atom_id),
        "from": old,
        "to": pt.symbol(pt.atomic_number(element)),
        "position_A": structure.positions[k].tolist(),
        "isotope": mass_number or None,
    }
    _log(structure, record)
    return record


def add_dopant(structure: Structure, element: str, site: Optional[int] = None,
               near: Optional[Sequence[float]] = None, host_element: Optional[str] = None) -> dict:
    """Substitutional dopant at ``site`` or at the lattice site nearest ``near``."""
    if site is None:
        if near is None:
            raise ValueError("add_dopant needs either site=<atom id> or near=(x,y,z)")
        site = nearest_site(structure, near, element=host_element)
    rec = substitute_atom(structure, site, element, role="dopant")
    rec["type"] = "dopant"
    return rec


def add_interstitial(structure: Structure, element: str, position: Sequence[float],
                     *, min_separation_A: float = 0.8) -> dict:
    """Insert an interstitial atom at a cartesian position.

    Raises if the requested position overlaps an existing atom, which would be
    an unphysical configuration rather than a defect.
    """
    p = np.asarray(position, dtype=float)
    d = structure.positions - p
    if structure.cell.is_periodic:
        d = structure.cell.minimum_image(d)
    r = np.linalg.norm(d, axis=1)
    if len(r) and r.min() < min_separation_A:
        raise StructureError(
            f"Interstitial at {p.tolist()} is {r.min():.3f} A from atom "
            f"#{int(structure.ids[int(np.argmin(r))])}; minimum separation is "
            f"{min_separation_A} A. Move it, or lower min_separation_A explicitly "
            "if you intend a hypothetical overlapping configuration."
        )
    new_id = structure.add_atom(element, p, role="interstitial",
                                label=f"{element}-interstitial")
    rec = {"type": "interstitial", "atom_id": new_id, "element": element,
           "position_A": p.tolist()}
    _log(structure, rec)
    return rec


def add_adatom(structure: Structure, element: str, xy: Sequence[float],
               height_A: Optional[float] = None) -> dict:
    """Place an adatom above the surface at ``xy``.

    With ``height_A`` the adatom sits that far above the topmost atom within
    3 A laterally.  Without it the adatom is lowered until its distance to
    some surface atom equals the sum of their covalent radii, and no surface
    atom is closer than its own covalent contact: the geometry of a single
    covalent bond to the nearest surface atom, a starting point to relax.
    """
    x, y = float(xy[0]), float(xy[1])
    pos = structure.positions
    if len(pos) == 0:
        raise StructureError("Cannot add an adatom to an empty structure")
    lateral = np.linalg.norm(pos[:, :2] - np.array([x, y]), axis=1)
    near = lateral < 3.0
    z_ref = pos[near, 2].max() if near.any() else pos[:, 2].max()
    rec: dict = {"type": "adatom", "element": element}
    if height_A is not None:
        z = z_ref + float(height_A)
        rec["placement"] = "explicit height above the local surface"
    else:
        r_new = pt.covalent_radius(element)
        radii = np.array([pt.covalent_radius(int(n)) or np.nan for n in structure.numbers])
        if r_new is None or np.isnan(radii).any():
            raise StructureError(f"No covalent radius is tabulated for {element} or a surface "
                                 "element, so the bond-length placement is unavailable. Give "
                                 "height_A explicitly.")
        contact = r_new + radii
        reach = lateral < contact
        if not reach.any():
            nearest = int(np.argmin(lateral - contact))
            z = float(pos[nearest, 2] + contact[nearest])
            partner = nearest
            rec["placement"] = ("no surface atom within covalent reach laterally; placed one "
                                "covalent contact above the nearest one")
        else:
            heights = pos[reach, 2] + np.sqrt(contact[reach] ** 2 - lateral[reach] ** 2)
            partner = int(np.nonzero(reach)[0][int(np.argmax(heights))])
            z = float(heights.max())
        rec.setdefault("placement", "covalent contact with the nearest surface atom: sum "
                                    "of covalent radii, no surface atom closer than its own "
                                    "contact")
        rec["bonded_to_atom_id"] = int(structure.ids[partner])
        rec["bond_length_A"] = float(contact[partner])
    p = np.array([x, y, z])
    new_id = structure.add_atom(element, p, role="adatom", label=f"{element}-adatom")
    rec.update({"atom_id": new_id, "position_A": p.tolist(),
                "height_above_surface_A": float(z - z_ref)})
    _log(structure, rec)
    return rec


def create_divacancy(structure: Structure, atom_id: int) -> List[dict]:
    """Remove an atom and its nearest neighbour."""
    from ..physics.bonds import perceive_bonds
    if structure.bonds is None:
        structure.set_bonds(perceive_bonds(structure))
    nbrs = structure.neighbors_of(atom_id)
    if not nbrs:
        raise StructureError(f"Atom #{atom_id} has no perceived bonds; cannot form a divacancy")
    first = create_vacancy(structure, atom_id)
    second = create_vacancy(structure, nbrs[0])
    return [first, second]
