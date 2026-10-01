"""Extended defects: stacking faults and dislocations as atomistic objects.

Stacking faults (Materia-native)
--------------------------------
An fcc crystal is oriented with x = [1-10], y = [11-2] and z = [111], and its
third cell vector is tilted by a fraction ``s`` of the Shockley partial
Burgers vector ``b_p = a/6 [11-2]``.  Periodicity then places one generalised
stacking fault per cell, with no free surfaces.  The fault energy per area is
``gamma(s) = (E(s) - E(0)) / A``; at ``s = 1`` the fault is the intrinsic
stacking fault, and the maximum along ``0 <= s <= 1`` is the unstable
stacking-fault energy.  With ``relax`` the atoms relax along z only, as is
conventional (Vitek 1968), so the imposed shear is kept.

Dislocations (built by matscipy, relaxed by Materia)
----------------------------------------------------
Straight dislocations are generated with matscipy (LGPL-2.1) from the
anisotropic elastic displacement field for the crystal's own elastic
constants, in a cylinder periodic along the line whose outer shell is held
fixed at the elastic solution.  Materia relaxes the core with the chosen
potential.  The centrosymmetry parameter (Kelchner, Plimpton and Hamilton
1998) marks atoms that are not in a perfect fcc environment, from which the
width of a dissociated core is measured.

References
----------
V. Vitek, Philos. Mag. 18 (1968) 773.
C. L. Kelchner, S. J. Plimpton and J. C. Hamilton, Phys. Rev. B 58 (1998) 11085.
J. P. Hirth and J. Lothe, Theory of Dislocations, 2nd ed. (Wiley, 1982).
P. Grigorev et al., J. Open Source Softw. 9 (2024) 5668 (matscipy).
"""

from __future__ import annotations

import importlib
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..provenance import Fidelity, Origin, Provenance, Result

EV_A2_MJ_M2 = 16021.766208


class DefectError(ValueError):
    pass


def fcc_111_cell(symbol: str, lattice_constant_A: float, layers: int = 12,
                 repeat_xy: Sequence[int] = (1, 1)) -> Structure:
    """fcc oriented x = [1-10], y = [11-2], z = [111]; ``layers`` must be a multiple of 3."""
    if layers % 3 or layers < 6:
        raise DefectError("layers must be a multiple of 3 and at least 6 (ABC stacking).")
    from ase.lattice.cubic import FaceCenteredCubic

    atoms = FaceCenteredCubic(directions=[[1, -1, 0], [1, 1, -2], [1, 1, 1]],
                              size=(repeat_xy[0], repeat_xy[1], layers // 3),
                              symbol=symbol, latticeconstant=lattice_constant_A, pbc=True)
    return Structure(atoms.numbers, atoms.positions, Cell(np.asarray(atoms.cell),
                                                          (True, True, True)))


def generalized_stacking_fault(potential, symbol: str, lattice_constant_A: float,
                               fractions: Sequence[float] = tuple(np.linspace(0, 1, 11)),
                               layers: int = 12, relax: bool = True,
                               fmax_eV_A: float = 1e-4) -> Result:
    """gamma(s) along [11-2] in mJ/m^2, with the intrinsic and unstable fault energies."""
    if 0.0 not in fractions or 1.0 not in fractions:
        raise DefectError("fractions must include 0 and 1.")
    base = fcc_111_cell(symbol, lattice_constant_A, layers)
    partial = lattice_constant_A / math.sqrt(6.0)
    area = float(np.linalg.norm(np.cross(base.cell.matrix[0], base.cell.matrix[1])))
    energies = []
    for s in fractions:
        cell = np.asarray(base.cell.matrix).copy()
        cell[2] = cell[2] + np.array([0.0, s * partial, 0.0])
        faulted = Structure(base.numbers, base.positions, Cell(cell, (True, True, True)))
        energies.append(_relaxed_normal_energy(potential, faulted, relax, fmax_eV_A))
    energies = np.array(energies)
    gamma = (energies - energies[list(fractions).index(0.0)]) / area * EV_A2_MJ_M2
    fractions = [float(f) for f in fractions]
    isf = float(gamma[fractions.index(1.0)])
    usf = float(gamma.max())
    label = getattr(potential, "model_label", lambda: potential.name)()
    prov = Provenance(
        model=f"defects/generalized-stacking-fault[{label}]",
        fidelity=getattr(potential, "fidelity", Fidelity.TIER1_CLASSICAL),
        origin=Origin.CALCULATED,
        approximations=["Tilted periodic cell, one fault per cell, no surfaces; "
                        f"{layers} (111) layers; static, zero temperature.",
                        "Atoms relaxed along [111] only." if relax else
                        "Rigid shift, no relaxation."],
        parameters={"symbol": symbol, "lattice_constant_A": lattice_constant_A,
                    "layers": layers, "relax": relax, "area_A2": area},
        references=["V. Vitek, Philos. Mag. 18 (1968) 773"])
    return Result("generalized_stacking_fault", {"fraction": fractions,
                                                 "gamma_mJ_m2": gamma.tolist()},
                  "mJ/m^2", prov, extra={"intrinsic_mJ_m2": isf, "unstable_mJ_m2": usf,
                                         "partial_burgers_A": partial})


def _relaxed_normal_energy(potential, structure: Structure, relax: bool, fmax: float) -> float:
    if not relax:
        return float(potential.energy_and_forces(structure)[0])
    from ase.constraints import FixCartesian
    from ase.optimize import BFGS

    from ..physics.neb import _ase_atoms, _calculator

    atoms = _ase_atoms(structure)
    atoms.calc = _calculator(potential, structure)
    atoms.set_constraint(FixCartesian(range(len(atoms)), mask=(True, True, False)))
    optimiser = BFGS(atoms, logfile=None)
    if not optimiser.run(fmax=fmax, steps=2000):
        raise DefectError("Relaxation normal to the fault plane did not converge.")
    return float(atoms.get_potential_energy())


def centrosymmetry(structure: Structure, neighbours: int = 12) -> np.ndarray:
    """Kelchner centrosymmetry parameter in A^2 for every atom (fcc: 12 neighbours)."""
    from ..physics.neighbors import neighbor_list

    positions = np.asarray(structure.positions)
    cutoff = 1.0
    while True:
        nl = neighbor_list(positions, structure.cell, cutoff)
        if np.bincount(nl.i, minlength=len(structure)).min() >= neighbours:
            break
        cutoff *= 1.2
    out = np.zeros(len(structure))
    for atom in range(len(structure)):
        mask = nl.i == atom
        vectors = nl.D[mask][np.argsort(nl.d[mask])[:neighbours]]
        sums = np.linalg.norm(vectors[:, None, :] + vectors[None, :, :], axis=2) ** 2
        np.fill_diagonal(sums, np.inf)
        total, used = 0.0, set()
        for i, j in sorted(((i, j) for i in range(neighbours) for j in range(i + 1, neighbours)),
                           key=lambda p: sums[p]):
            if i in used or j in used:
                continue
            used.update((i, j))
            total += sums[i, j]
        out[atom] = total
    return out


DISLOCATIONS = {
    "fcc-edge": "FCCEdge110Dislocation",
    "fcc-screw": "FCCScrew110Dislocation",
    "bcc-screw": "BCCScrew111Dislocation",
    "bcc-edge": "BCCEdge111Dislocation",
}


def dislocation(potential, kind: str, symbol: str, lattice_constant_A: float,
                c11_GPa: float, c12_GPa: float, c44_GPa: float, radius_A: float = 40.0,
                relax: bool = True, fmax_eV_A: float = 5e-3, max_steps: int = 5000) -> Result:
    """A straight dislocation in a cylinder, built by matscipy and relaxed with the potential."""
    if kind not in DISLOCATIONS:
        raise DefectError(f"kind must be one of {sorted(DISLOCATIONS)}.")
    try:
        module = importlib.import_module("matscipy.dislocation")
    except ImportError:
        raise DefectError("matscipy is not installed: pip install matscipy") from None
    builder = getattr(module, DISLOCATIONS[kind])(lattice_constant_A, C11=c11_GPa, C12=c12_GPa,
                                                  C44=c44_GPa, symbol=symbol)
    _, atoms = builder.build_cylinder(radius_A, method="adsl", verbose=False)
    fixed = np.asarray(atoms.arrays["fix_mask"], dtype=bool)
    structure = Structure(atoms.numbers, atoms.positions,
                          Cell(np.asarray(atoms.cell), (False, False, True)))
    structure.fixed[:] = fixed
    energy = None
    converged = None
    if relax:
        from ..solvers.classical import ClassicalSolver

        out = ClassicalSolver(potential).relax(structure, fmax_eV_A=fmax_eV_A,
                                               max_steps=max_steps)
        converged = bool(out.convergence.converged)
        structure = out.structure
        energy = float(out.results["energy"].value)
    csp = centrosymmetry(structure) if kind.startswith("fcc") else None
    label = getattr(potential, "model_label", lambda: potential.name)()
    prov = Provenance(
        model=f"defects/dislocation[{label}]",
        fidelity=getattr(potential, "fidelity", Fidelity.TIER1_CLASSICAL),
        origin=Origin.CALCULATED if (not relax or converged) else Origin.ESTIMATED,
        approximations=["Built by matscipy from the anisotropic elastic displacement field "
                        "for the given elastic constants; outer shell fixed at the elastic "
                        "solution; periodic along the line.",
                        "Core relaxed with the potential at zero temperature." if relax else
                        "Unrelaxed elastic solution."],
        parameters={"kind": kind, "symbol": symbol, "lattice_constant_A": lattice_constant_A,
                    "elastic_constants_GPa": [c11_GPa, c12_GPa, c44_GPa],
                    "radius_A": radius_A, "burgers_vector_A": np.asarray(builder.burgers).tolist(),
                    "fixed_atoms": int(fixed.sum())},
        references=["P. Grigorev et al., J. Open Source Softw. 9 (2024) 5668 (matscipy)",
                    "J. P. Hirth and J. Lothe, Theory of Dislocations (1982)"])
    return Result("dislocation", structure, "", prov,
                  extra={"energy_eV": energy, "relaxation_converged": converged,
                         "centrosymmetry_A2": None if csp is None else csp.tolist(),
                         "burgers_vector_A": np.asarray(builder.burgers).tolist(),
                         "axes": np.asarray(builder.axes).tolist()})


def dissociation_width(structure: Structure, centrosymmetry_A2: Sequence[float],
                       lattice_constant_A: float, glide_axis: int = 0,
                       normal_axis: int = 1) -> float:
    """Separation of Shockley partials, from the extent of the stacking-fault ribbon.

    Atoms whose centrosymmetry lies between 0.5 and 4 times ``(a/sqrt6)^2``
    (Kelchner et al. 1998 bands for stacking faults) near the glide plane are
    taken as the ribbon; the width is the distance between its outermost atoms.
    """
    csp = np.asarray(centrosymmetry_A2)
    partial2 = (lattice_constant_A / math.sqrt(6.0)) ** 2
    positions = np.asarray(structure.positions)
    centre = positions[~np.asarray(structure.fixed)].mean(axis=0)
    near = np.abs(positions[:, normal_axis] - centre[normal_axis]) < lattice_constant_A
    ribbon = near & (csp > 0.5 * partial2) & (csp < 4.0 * partial2)
    if ribbon.sum() < 2:
        raise DefectError("No stacking-fault ribbon was found; the core did not dissociate.")
    along = positions[ribbon, glide_axis]
    return float(along.max() - along.min())


def isotropic_partial_separation(shear_GPa: float, poisson: float, partial_A: float,
                                 gamma_mJ_m2: float, character_deg: float) -> float:
    """Equilibrium partial separation from isotropic elasticity (Hirth and Lothe 1982).

    ``d = (G b_p^2 / 8 pi gamma) ((2 - nu)/(1 - nu)) (1 - 2 nu cos(2 beta)/(2 - nu))``
    with ``beta`` the angle between the total Burgers vector and the line.
    """
    gamma = gamma_mJ_m2 / EV_A2_MJ_M2
    shear = shear_GPa / 160.21766208
    beta = math.radians(character_deg)
    return (shear * partial_A ** 2 / (8 * math.pi * gamma) * (2 - poisson) / (1 - poisson)
            * (1 - 2 * poisson * math.cos(2 * beta) / (2 - poisson)))


def _oriented_fcc(symbol: str, a: float, directions, size):
    from ase.lattice.cubic import FaceCenteredCubic

    return FaceCenteredCubic(directions=directions, size=size, symbol=symbol,
                             latticeconstant=a, pbc=True)


def symmetric_tilt_boundary(potential, symbol: str, lattice_constant_A: float,
                            plane: Sequence[int] = (3, 1, 0), grain_repeats: int = 4,
                            line_repeats: int = 2, translations: int = 6,
                            overlap_A: float = 1.6, fmax_eV_A: float = 1e-3,
                            max_steps: int = 5000) -> Result:
    """A symmetric [001] tilt boundary in fcc and its energy in mJ/m^2.

    Two grains are mirror images across the (h k 0) plane: grain 1 has the
    boundary normal along [h k 0] and grain 2 along [h -k 0], both periodic
    along [-k h 0] and [001].  The periodic cell holds two identical
    boundaries.  Rigid translations of grain 2 within the boundary plane are
    sampled on a ``translations x translations`` grid; atoms closer than
    ``overlap_A`` across a boundary are merged; every candidate is relaxed and
    the lowest energy is kept.  ``E_gb = (E - N e_bulk) / (2 A)``.
    """
    h, k, l = (int(v) for v in plane)
    if l != 0 or h <= 0 or k <= 0:
        raise DefectError("plane must be (h k 0) with h, k > 0 for a [001] tilt boundary.")
    sigma = h * h + k * k
    if sigma % 2 == 0:
        sigma //= 2
    from ..solvers.classical import ClassicalSolver

    grain1 = _oriented_fcc(symbol, lattice_constant_A, [[-k, h, 0], [0, 0, 1], [h, k, 0]],
                           (1, line_repeats, grain_repeats))
    grain2 = _oriented_fcc(symbol, lattice_constant_A, [[k, h, 0], [0, 0, 1], [h, -k, 0]],
                           (1, line_repeats, grain_repeats))
    length = float(grain1.cell[2, 2])
    if abs(grain1.cell[0, 0] - grain2.cell[0, 0]) > 1e-8:
        raise DefectError("The two grains are not commensurate along the boundary.")
    reference = _oriented_fcc(symbol, lattice_constant_A, [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              (1, 1, 1))
    bulk_cell = Structure(reference.numbers, reference.positions,
                          Cell(np.asarray(reference.cell), (True, True, True)))
    e_bulk = float(potential.energy_and_forces(bulk_cell)[0]) / len(bulk_cell)
    best = None
    tx, ty = grain1.cell[0, 0], grain1.cell[1, 1]
    for i in range(translations):
        for j in range(translations):
            shift = np.array([i * tx / translations, j * ty / translations, length])
            positions = np.vstack([grain1.positions, grain2.positions + shift])
            cell = np.diag([tx, ty, 2 * length])
            numbers = np.concatenate([grain1.numbers, grain2.numbers])
            candidate = Structure(numbers, positions, Cell(cell, (True, True, True)))
            candidate = _merge_close(candidate, overlap_A)
            relaxed = ClassicalSolver(potential).relax(candidate, fmax_eV_A=fmax_eV_A,
                                                       max_steps=max_steps)
            energy = float(relaxed.results["energy"].value)
            area = tx * ty
            gb = (energy - len(relaxed.structure) * e_bulk) / (2 * area) * EV_A2_MJ_M2
            if best is None or gb < best[0]:
                best = (gb, relaxed.structure, (i / translations, j / translations),
                        bool(relaxed.convergence.converged), len(relaxed.structure))
    gb, structure, fraction, converged, atoms = best
    misorientation = 2 * math.degrees(math.atan2(k, h))
    label = getattr(potential, "model_label", lambda: potential.name)()
    prov = Provenance(
        model=f"defects/symmetric-tilt-boundary[{label}]",
        fidelity=getattr(potential, "fidelity", Fidelity.TIER1_CLASSICAL),
        origin=Origin.CALCULATED if converged else Origin.ESTIMATED,
        approximations=["Two identical boundaries in a periodic bicrystal of fixed cell; "
                        f"rigid translations sampled on a {translations} x {translations} grid; "
                        f"atoms closer than {overlap_A:g} A merged; zero temperature.",
                        "A local search: lower-energy boundary structures with other atom "
                        "densities or larger periodicity are not explored."],
        parameters={"plane": [h, k, 0], "sigma": sigma, "misorientation_deg": misorientation,
                    "grain_repeats": grain_repeats, "line_repeats": line_repeats,
                    "best_translation_fraction": list(fraction), "atoms": atoms},
        references=["A. P. Sutton and R. W. Balluffi, Interfaces in Crystalline Materials "
                    "(Oxford, 1995)"])
    return Result("grain_boundary_energy", gb, "mJ/m^2", prov,
                  extra={"structure": structure, "sigma": sigma,
                         "misorientation_deg": misorientation, "relaxed": converged})


def _merge_close(structure: Structure, distance_A: float) -> Structure:
    """Remove one atom of every pair closer than ``distance_A`` (orthorhombic cell, minimum image)."""
    box = np.diag(np.asarray(structure.cell.matrix))
    positions = np.asarray(structure.positions)
    keep = np.ones(len(structure), dtype=bool)
    for i in range(len(structure)):
        if not keep[i]:
            continue
        delta = positions[i + 1:] - positions[i]
        delta -= box * np.round(delta / box)
        close = np.flatnonzero(np.linalg.norm(delta, axis=1) < distance_A) + i + 1
        keep[close] = False
    return Structure(structure.numbers[keep], positions[keep], structure.cell)
