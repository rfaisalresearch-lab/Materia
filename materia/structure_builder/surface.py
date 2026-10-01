"""Miller-indexed surface slab construction.

Algorithm
---------
Miller indices are given in the *conventional* cell of the material, but the
enumeration is carried out on the real translation lattice (see
:mod:`materia.structure_builder.primitive`), so centred lattices yield their
true primitive surface cells rather than the conventional multiple.

1. The plane normal ``n`` is obtained from the conventional reciprocal
   lattice: ``n ~ (h,k,l) . B``.
2. Short integer vectors of the primitive lattice are enumerated and their
   projections on ``n`` computed.  These projections form a one-dimensional
   lattice whose spacing ``d_min`` is the physical interplanar spacing of the
   crystal for that orientation.
3. Vectors with zero projection span the surface plane; the pair with the
   smallest area after Lagrange-Gauss reduction becomes the in-plane cell.
   The shortest vector with projection ``d_min`` is the stacking vector.
4. The resulting integer matrix defines a supercell, which is filled,
   repeated ``nz`` times, rotated so the normal is ``+z``, cut open in ``z``
   and padded with vacuum.

The construction is implemented directly so the computational core needs no
external build dependency.  ``tests/validation`` cross-checks the generated
cells against analytic values and, when ASE is installed, against
``ase.build.surface``.
"""

from __future__ import annotations

from math import gcd
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..materials.schema import MaterialDefinition
from ..provenance import Fidelity, Origin, Provenance
from .lattice import bulk, conventional_cell
from .primitive import primitive_structure

TOL = 1e-7


class SurfaceError(ValueError):
    pass


def _reduce_miller(miller: Sequence[int]) -> Tuple[int, int, int]:
    h, k, l = (int(round(v)) for v in miller)
    if (h, k, l) == (0, 0, 0):
        raise SurfaceError("(000) is not a valid surface orientation")
    g = gcd(gcd(abs(h), abs(k)), abs(l))
    return (h // g, k // g, l // g)


def _gauss_reduce(v1: np.ndarray, v2: np.ndarray, m1: np.ndarray, m2: np.ndarray):
    """Lagrange-Gauss 2D lattice reduction on cartesian vectors, carrying the
    integer coefficients along."""
    for _ in range(64):
        if v2 @ v2 < v1 @ v1:
            v1, v2, m1, m2 = v2, v1, m2, m1
        mu = round((v1 @ v2) / (v1 @ v1)) if v1 @ v1 > TOL else 0
        if mu == 0:
            break
        v2 = v2 - mu * v1
        m2 = m2 - mu * m1
    return v1, v2, m1, m2


def surface_basis(cell: Cell, miller: Sequence[int], max_index: int = 6,
                  lattice_matrix: Optional[np.ndarray] = None):
    """Integer matrix whose rows span the surface supercell.

    Parameters
    ----------
    cell
        The conventional cell, which defines the meaning of ``(hkl)``.
    lattice_matrix
        Rows of the true translation lattice (the primitive cell matrix).
        Defaults to the conventional cell, which is only correct for
        primitive lattices.

    Returns
    -------
    (M, normal, d_min)
        ``M`` is a 3x3 integer matrix in ``lattice_matrix`` coordinates,
        ``normal`` the unit surface normal, ``d_min`` the interplanar spacing
        of the real lattice for this orientation, in angstrom.
    """
    h, k, l = _reduce_miller(miller)
    P = cell.matrix if lattice_matrix is None else np.asarray(lattice_matrix, dtype=float)
    normal = np.array([h, k, l], dtype=float) @ cell.reciprocal
    normal = normal / np.linalg.norm(normal)

    rng = range(-max_index, max_index + 1)
    vecs = np.array([(u, v, w) for u in rng for v in rng for w in rng
                     if (u, v, w) != (0, 0, 0)], dtype=int)
    cart = vecs @ P
    proj = cart @ normal

    scale = float(np.abs(P).max())
    ptol = 1e-6 * max(1.0, scale)
    positive = proj[proj > ptol]
    if positive.size == 0:
        raise SurfaceError(
            f"No lattice vector with a positive component along ({h}{k}{l}) within "
            f"|index| <= {max_index}"
        )
    d_min = float(positive.min())

    in_plane_mask = np.abs(proj) < ptol
    if in_plane_mask.sum() < 2:
        raise SurfaceError(f"Could not find in-plane lattice vectors for ({h}{k}{l})")
    stack_mask = np.abs(proj - d_min) < 1e-6 * max(1.0, d_min)

    ip = vecs[in_plane_mask]
    ip_cart = cart[in_plane_mask]
    order = np.argsort(np.linalg.norm(ip_cart, axis=1))
    ip, ip_cart = ip[order], ip_cart[order]

    c1, x1 = ip[0], ip_cart[0]
    c2 = None
    for idx in range(1, len(ip)):
        if np.linalg.norm(np.cross(x1, ip_cart[idx])) > TOL * max(
                1.0, np.linalg.norm(x1) * np.linalg.norm(ip_cart[idx])):
            c2, x2 = ip[idx], ip_cart[idx]
            break
    if c2 is None:
        raise SurfaceError(f"Degenerate in-plane basis for ({h}{k}{l})")

    x1, x2, c1, c2 = _gauss_reduce(x1, x2, c1, c2)

    st = vecs[stack_mask]
    st_cart = cart[stack_mask]
    par = np.linalg.norm(st_cart - np.outer(st_cart @ normal, normal), axis=1)
    c3 = st[int(np.lexsort((np.linalg.norm(st_cart, axis=1), np.round(par, 6)))[0])]

    M = np.array([c1, c2, c3], dtype=int)
    if np.linalg.det(M.astype(float) @ P) < 0:
        M = np.array([c2, c1, c3], dtype=int)
    if abs(np.linalg.det(M)) < 0.5:
        raise SurfaceError(f"Singular surface basis for ({h}{k}{l})")
    return M, normal, d_min


def _fill_supercell(base: Structure, M: np.ndarray) -> Structure:
    """Fill a supercell defined by integer matrix ``M`` (rows in lattice units)."""
    m = base.cell.matrix
    super_matrix = M.astype(float) @ m
    inv_super = np.linalg.inv(super_matrix)
    n_cells = int(round(abs(np.linalg.det(M))))

    span = int(np.abs(M).max()) + 1
    numbers, positions, labels, roles = [], [], [], []
    base_pos = base.positions
    base_num = base.numbers
    base_lab = base.labels
    found = 0
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            for kk in range(-span, span + 1):
                shift = np.array([i, j, kk], dtype=float) @ m
                cand = base_pos + shift
                frac = cand @ inv_super
                inside = np.all((frac > -TOL) & (frac < 1.0 - TOL), axis=1)
                if not inside.any():
                    continue
                for idx in np.nonzero(inside)[0]:
                    numbers.append(int(base_num[idx]))
                    positions.append(cand[idx])
                    labels.append(str(base_lab[idx]))
                    roles.append("bulk")
                    found += 1

    expected = n_cells * len(base)
    if found != expected:
        eps = np.array([1e-4, 2e-4, 3e-4]) @ m
        numbers, positions, labels, roles = [], [], [], []
        for i in range(-span, span + 1):
            for j in range(-span, span + 1):
                for kk in range(-span, span + 1):
                    shift = np.array([i, j, kk], dtype=float) @ m
                    cand = base_pos + shift + eps
                    frac = cand @ inv_super
                    inside = np.all((frac > -TOL) & (frac < 1.0 - TOL), axis=1)
                    for idx in np.nonzero(inside)[0]:
                        numbers.append(int(base_num[idx]))
                        positions.append(cand[idx] - eps)
                        labels.append(str(base_lab[idx]))
                        roles.append("bulk")
        if len(numbers) != expected:
            raise SurfaceError(
                f"Supercell fill produced {len(numbers)} atoms, expected {expected}. "
                "This indicates a degenerate cell or a numerical tolerance problem."
            )

    out = Structure(numbers, np.array(positions), Cell(super_matrix, (True, True, True)),
                    labels=labels, roles=roles)
    out.info = dict(base.info)
    return out


def _rotate_to_z(structure: Structure, normal: np.ndarray) -> Structure:
    """Rotate so a1 is along +x, a2 lies in the xy plane and the normal is +z."""
    a1, a2, a3 = structure.cell.matrix
    e1 = a1 / np.linalg.norm(a1)
    n = np.cross(a1, a2)
    e3 = n / np.linalg.norm(n)
    if e3 @ normal < 0:
        e3 = -e3
    e2 = np.cross(e3, e1)
    R = np.array([e1, e2, e3])
    out = structure.copy()
    out.cell = Cell(structure.cell.matrix @ R.T, structure.cell.pbc)
    out.positions = structure.positions @ R.T
    return out


def _broken_bonds(structure: Structure, scale: float = 1.15) -> Tuple[int, int]:
    """Count broken bonds relative to the modal bulk coordination.

    Returns ``(total_broken, broken_on_top_layer)``.
    """
    from ..physics.bonds import perceive_bonds

    bonds = perceive_bonds(structure, scale=scale)
    counts = np.zeros(len(structure), dtype=int)
    index = {int(i): k for k, i in enumerate(structure.ids)}
    for b in bonds:
        counts[index[b.a]] += 1
        counts[index[b.b]] += 1
    if len(counts) == 0:
        return 0, 0
    missing = np.zeros(len(counts), dtype=int)
    for z in np.unique(structure.numbers):
        mask = structure.numbers == z
        reference = int(counts[mask].max())
        missing[mask] = np.maximum(reference - counts[mask], 0)
    z = structure.positions[:, 2]
    top = z >= z.max() - 0.6
    return int(missing.sum()), int(missing[top].sum())


def stable_termination_shift(
    material: MaterialDefinition,
    miller: Sequence[int],
    size: Tuple[int, int, int] = (1, 1, 3),
    n_candidates: int = 12,
) -> Tuple[float, List[dict]]:
    """Pick the cut through the stacking vector that breaks the fewest bonds.

    Ideal-truncation slabs can be cut at several inequivalent planes.  Cutting
    through the short, strong bonds of a bilayer leaves three dangling bonds
    per surface atom and is not the surface a real crystal cleaves along.  This
    routine tries every distinct cut, counts broken bonds with the standard
    covalent-radius criterion, and returns the best one along with the full
    table so the choice is inspectable rather than hidden.
    """
    candidates = []
    best = (None, None)
    for idx in range(n_candidates):
        shift = idx / n_candidates
        try:
            slab = make_surface(material, miller, size=size, vacuum_A=8.0,
                                shift=shift, termination_mode="as-cut")
        except SurfaceError:
            continue
        total, top = _broken_bonds(slab)
        candidates.append({"shift": shift, "broken_bonds": total,
                           "broken_on_top_layer": top, "n_atoms": len(slab)})
        key = (total, top, shift)
        if best[0] is None or key < best[0]:
            best = (key, shift)
    if best[1] is None:
        return 0.0, candidates
    return float(best[1]), candidates


def make_surface(
    material: MaterialDefinition,
    miller: Sequence[int] = (1, 1, 1),
    *,
    size: Tuple[int, int, int] = (1, 1, 4),
    vacuum_A: float = 12.0,
    termination: Optional[str] = None,
    shift: float = 0.0,
    fix_bottom_layers: int = 0,
    center: bool = True,
    termination_mode: str = "auto-stable",
    reconstruction: Optional[str] = None,
    reconstruction_options: Optional[dict] = None,
    relax_reconstruction: bool = True,
    reconstruction_model: str = "recommended",
    reconstruction_fmax_eV_A: float = 0.005,
    reconstruction_max_steps: int = 800,
) -> Structure:
    """Build a slab of ``material`` exposing the ``(hkl)`` surface.

    Parameters
    ----------
    size
        ``(nx, ny, nz)``: in-plane repetitions of the primitive surface cell
        and number of stacking repeats along the normal.
    vacuum_A
        Vacuum thickness added above the slab (and below, if ``center``).
    shift
        Fractional shift (0..1) of the cut plane along the stacking vector.
        Use it to select between terminations of a polar surface.
    fix_bottom_layers
        Mark the lowest ``n`` atomic planes as fixed, the usual way to emulate
        a semi-infinite substrate during relaxation.
    termination_mode
        ``"auto-stable"`` (default) searches the cuts through the stacking
        vector and keeps the one that breaks the fewest bonds, which is the
        plane a real crystal cleaves along.  ``"as-cut"`` uses ``shift``
        verbatim.  The chosen shift and the full candidate table are recorded
        in ``structure.info['surface']``.
    reconstruction
        Id of a reconstruction declared by the material, applied to the upper
        face after the slab is built.  ``None`` leaves the ideal truncation.
        A declared but unavailable reconstruction raises
        :class:`~materia.structure_builder.reconstruction.ReconstructionNotImplemented`
        rather than returning an approximation.
    relax_reconstruction
        Relax the reconstructed slab with the material's recommended
        interatomic model.  Switching it off returns the generator's topology
        with an unrelaxed, explicitly estimated geometry.
    reconstruction_model, reconstruction_fmax_eV_A, reconstruction_max_steps
        Which interatomic model relaxes the reconstruction and how hard it
        tries.  A run that stops before reaching the force target is recorded
        as not converged and its geometry is not offered for comparison.
    """
    h, k, l = _reduce_miller(miller)
    if termination_mode not in ("auto-stable", "as-cut"):
        raise SurfaceError(
            f"Unknown termination_mode {termination_mode!r}; use 'auto-stable' or 'as-cut'."
        )
    candidates: List[dict] = []
    if termination_mode == "auto-stable" and not shift:
        shift, candidates = stable_termination_shift(material, (h, k, l))
    cell = conventional_cell(material)
    conventional = bulk(material)
    base = primitive_structure(conventional)
    M, normal, d_min = surface_basis(cell, (h, k, l), lattice_matrix=base.cell.matrix)

    if shift:
        stack_cart = M[2].astype(float) @ base.cell.matrix
        base = base.copy()
        base.positions = base.positions + float(shift) * stack_cart
        base.wrap()

    slab = _fill_supercell(base, M)
    nx, ny, nz = (int(v) for v in size)
    if min(nx, ny, nz) < 1:
        raise SurfaceError("size components must be >= 1")
    slab = slab.repeat(nx, ny, nz)
    slab = _rotate_to_z(slab, normal)

    pos = slab.positions.copy()
    a1, a2, a3 = slab.cell.matrix
    thickness = float(a3[2])
    if thickness <= 0:
        raise SurfaceError("Slab has non-positive thickness along the surface normal")

    A2 = np.array([[a1[0], a1[1]], [a2[0], a2[1]]], dtype=float)
    frac_xy = pos[:, :2] @ np.linalg.inv(A2)
    frac_xy -= np.floor(frac_xy + 1e-9)
    pos[:, :2] = frac_xy @ A2

    zmin, zmax = pos[:, 2].min(), pos[:, 2].max()
    slab_height = zmax - zmin
    total_z = slab_height + (2.0 * vacuum_A if center else vacuum_A)
    pos[:, 2] -= zmin
    pos[:, 2] += vacuum_A if center else 0.0
    new_matrix = np.array([a1, a2, [0.0, 0.0, total_z]])
    out = Structure(
        slab.numbers, pos, Cell(new_matrix, (True, True, False)),
        labels=list(slab.labels), roles=list(slab.roles),
    )
    out.info = dict(base.info)

    _assign_surface_roles(out)
    if fix_bottom_layers > 0:
        _fix_bottom(out, fix_bottom_layers)

    d_hkl = float(2.0 * np.pi / np.linalg.norm(np.array([h, k, l], float) @ cell.reciprocal))
    out.info["surface"] = {
        "material_id": material.id,
        "miller": [h, k, l],
        "termination": termination or "ideal-bulk",
        "termination_mode": termination_mode,
        "termination_candidates": candidates,
        "shift": float(shift),
        "size": [nx, ny, nz],
        "vacuum_A": float(vacuum_A),
        "in_plane_a_A": float(np.linalg.norm(a1)),
        "in_plane_b_A": float(np.linalg.norm(a2)),
        "in_plane_angle_deg": float(np.degrees(np.arccos(
            np.clip(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2)), -1, 1)))),
        "interplanar_spacing_A": d_min,
        "conventional_d_hkl_A": d_hkl,
        "atoms_per_layer": int(len(base) * abs(int(round(np.linalg.det(M))))
                               // max(1, len(np.unique(np.round(pos[:, 2], 3))))) if len(pos) else 0,
        "slab_thickness_A": float(slab_height),
        "n_atomic_planes": int(len(np.unique(np.round(pos[:, 2], 3)))),
        "supercell_matrix": M.tolist(),
    }
    out.info["provenance"] = Provenance(
        model="structure-builder/surface",
        fidelity=Fidelity.TIER0_STRUCTURAL,
        origin=Origin.CALCULATED,
        approximations=[
            "Ideal bulk truncation: no surface reconstruction and no relaxation applied.",
            "Bulk lattice parameter used throughout the slab.",
        ],
        boundary_conditions="2D periodic in x,y; vacuum-terminated in z",
        parameters={
            "material": material.id, "miller": [h, k, l], "size": [nx, ny, nz],
            "vacuum_A": vacuum_A, "shift": shift,
        },
        references=list(material.references),
        dataset=material.source_path,
        dataset_license=material.license,
        notes=(
            "Surface roles are assigned geometrically from z-coordinates. "
            "Any reconstruction listed for this material but not applied here is "
            "reported as not implemented."
        ),
    ).as_dict()
    if reconstruction:
        from .reconstruction import apply_reconstruction

        apply_reconstruction(out, material, reconstruction,
                             relax=relax_reconstruction,
                             model=reconstruction_model,
                             fmax_eV_A=reconstruction_fmax_eV_A,
                             max_steps=reconstruction_max_steps,
                             options=reconstruction_options)
    return out


def _assign_surface_roles(structure: Structure, tolerance_A: float = 0.6) -> None:
    z = structure.positions[:, 2]
    if len(z) == 0:
        return
    top, bottom = z.max(), z.min()
    for i in range(len(structure)):
        if z[i] >= top - tolerance_A:
            structure.roles[i] = "surface"
        elif z[i] <= bottom + tolerance_A:
            structure.roles[i] = "surface"
        elif z[i] >= top - 2.5 * tolerance_A:
            structure.roles[i] = "subsurface"
        else:
            structure.roles[i] = "bulk"


def _fix_bottom(structure: Structure, n_layers: int) -> None:
    z = np.round(structure.positions[:, 2], 3)
    planes = np.unique(z)
    if n_layers >= len(planes):
        raise SurfaceError(
            f"fix_bottom_layers={n_layers} but the slab has only {len(planes)} atomic planes"
        )
    cutoff = planes[n_layers - 1]
    structure.fixed[:] = z <= cutoff + 1e-6


def passivate(structure: Structure, element: str = "H", bond_length_A: Optional[float] = None,
              side: str = "bottom") -> int:
    """Cap dangling bonds on one or both slab faces with a terminating species.

    A simple geometric capping: each under-coordinated surface atom gets one
    terminator placed along its missing tetrahedral direction, estimated as the
    negative sum of its existing bond vectors.  This is a geometric heuristic,
    not an optimised structure: relax afterwards.
    """
    from ..physics.bonds import perceive_bonds
    from ..elements import periodic_table as pt

    bonds = perceive_bonds(structure)
    structure.set_bonds(bonds)
    z = structure.positions[:, 2]
    top, bottom = z.max(), z.min()
    added = 0
    targets = []
    for i in range(len(structure)):
        aid = int(structure.ids[i])
        if side in ("bottom", "both") and z[i] <= bottom + 0.6:
            targets.append(aid)
        if side in ("top", "both") and z[i] >= top - 0.6:
            targets.append(aid)

    counts = np.array([len(structure.bonds_of(int(a))) for a in structure.ids])
    bulk_coord = int(np.bincount(counts).argmax()) if len(counts) else 4

    for aid in targets:
        nbrs = structure.neighbors_of(aid)
        missing = bulk_coord - len(nbrs)
        if missing <= 0:
            continue
        p0 = structure.positions[structure.index_of(aid)]
        if nbrs:
            v = np.zeros(3)
            for nb in nbrs:
                d = structure.positions[structure.index_of(nb)] - p0
                if structure.cell.is_periodic:
                    d = structure.cell.minimum_image(d[None, :])[0]
                v += d / np.linalg.norm(d)
            direction = -v
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([0.0, 0.0, 1.0 if p0[2] >= (top + bottom) / 2 else -1.0])
        else:
            direction = np.array([0.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        z0 = int(structure.numbers[structure.index_of(aid)])
        r = bond_length_A
        if r is None:
            ra = pt.covalent_radius(z0) or 1.1
            rb = pt.covalent_radius(element) or 0.31
            r = ra + rb
        structure.add_atom(element, p0 + r * direction, role="adsorbate",
                           label=f"{element}-passivation")
        added += 1
    structure.invalidate_bonds()
    structure.info.setdefault("passivation", []).append(
        {"element": element, "side": side, "added": added,
         "method": "geometric dangling-bond capping (heuristic)"}
    )
    return added
