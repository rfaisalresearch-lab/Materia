"""Defect and surface energetics for any Materia potential.

* Equilibrium bulk: the isotropic scale of a crystal that minimises the
  energy (cubic and other crystals whose shape is fixed by symmetry), with the
  cohesive energy per atom relative to isolated atoms at zero energy, which is
  how EAM and most classical potentials are referenced.
* Vacancy formation energy at constant volume,
  ``E_f = E(N - 1) - (N - 1) / N E(N)``, with ionic relaxation, in a sequence of
  supercells so that the size convergence is visible.
* Surface energy ``gamma = (E_slab - N e_bulk) / 2A`` for symmetric slabs of
  increasing thickness built by :func:`materia.structure_builder.surface.make_surface`
  and scaled to the potential's own lattice constant, with ionic relaxation.

Each result reports every size computed and whether the last two agree within
the stated tolerance; an unconverged series is marked as such, not hidden.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..provenance import Convergence, Origin, Provenance, Result

EV_A2_TO_J_M2 = 16.02176634


class EnergeticsError(ValueError):
    pass


def _scaled(structure: Structure, factor: float) -> Structure:
    out = structure.copy()
    out.cell = Cell(np.asarray(structure.cell.matrix) * factor, structure.cell.pbc)
    out.positions = np.asarray(structure.positions) * factor
    return out


def _label(potential: Any) -> str:
    return potential.model_label() if hasattr(potential, "model_label") else potential.name


def _relaxed_energy(potential: Any, structure: Structure, fmax: float,
                    max_steps: int = 5000) -> Tuple[float, Structure]:
    from ..solvers.classical import ClassicalSolver

    out = ClassicalSolver(potential).relax(structure, fmax_eV_A=fmax, max_steps=max_steps)
    if not out.convergence.converged:
        raise EnergeticsError(f"Ionic relaxation did not reach {fmax:g} eV/A in {max_steps} "
                              "steps.")
    return float(out.results["energy"].value), out.structure


def equilibrium_bulk(bulk: Structure, potential: Any, tolerance: float = 1e-7) -> dict:
    """Isotropic scale of ``bulk`` that minimises the energy, and the energy per atom."""
    from scipy.optimize import minimize_scalar

    if not all(bulk.cell.pbc):
        raise EnergeticsError("The bulk reference must be periodic in three directions.")
    forces = np.asarray(potential.energy_and_forces(bulk)[1])
    if np.abs(forces).max() > 1e-3:
        raise EnergeticsError("Atoms in the bulk reference feel forces; it is not a "
                              "symmetric crystal whose shape is fixed by isotropic scaling.")
    n = len(bulk)
    result = minimize_scalar(lambda f: potential.energy_and_forces(_scaled(bulk, f))[0] / n,
                             bounds=(0.9, 1.1), method="bounded",
                             options={"xatol": tolerance})
    if not 0.901 < result.x < 1.099:
        raise EnergeticsError("The energy minimum lies at the edge of a 10% scale range; the "
                              "input is far from this potential's equilibrium.")
    scaled = _scaled(bulk, float(result.x))
    stress_free = potential.energy_and_forces(scaled)
    return {"structure": scaled, "scale": float(result.x),
            "energy_per_atom_eV": float(stress_free[0]) / n,
            "volume_per_atom_A3": scaled.cell.volume / n}


def vacancy_formation(bulk: Structure, potential: Any,
                      supercells: Sequence[Tuple[int, int, int]] = ((3, 3, 3), (4, 4, 4)),
                      site: int = 0, relax: bool = True, fmax_eV_A: float = 1e-3,
                      tolerance_eV: float = 0.01) -> Result:
    """Constant-volume vacancy formation energy in each supercell."""
    reference = equilibrium_bulk(bulk, potential)
    rows = []
    for size in supercells:
        perfect = reference["structure"].repeat(*size)
        n = len(perfect)
        e_perfect = float(potential.energy_and_forces(perfect)[0])
        defective = perfect.copy()
        defective.remove_atoms([int(defective.ids[site])])
        if relax:
            e_defect, _ = _relaxed_energy(potential, defective, fmax_eV_A)
        else:
            e_defect = float(potential.energy_and_forces(defective)[0])
        unrelaxed = float(potential.energy_and_forces(defective)[0]) - (n - 1) / n * e_perfect
        rows.append({"supercell": list(size), "atoms": n,
                     "formation_energy_eV": e_defect - (n - 1) / n * e_perfect,
                     "unrelaxed_formation_energy_eV": unrelaxed})
    converged = len(rows) > 1 and abs(rows[-1]["formation_energy_eV"]
                                      - rows[-2]["formation_energy_eV"]) <= tolerance_eV
    prov = Provenance(
        model=f"energetics/vacancy[{_label(potential)}]",
        fidelity=getattr(potential, "fidelity", None), origin=Origin.CALCULATED,
        approximations=["Constant-volume supercell at the potential's equilibrium lattice "
                        "constant; ionic relaxation " + ("included." if relax else "omitted."),
                        "Neutral vacancy; no charge states or electronic corrections."],
        parameters={"supercells": [list(s) for s in supercells], "site": site,
                    "fmax_eV_A": fmax_eV_A, "lattice_scale": reference["scale"]},
        references=[])
    convergence = Convergence(converged=converged, iterations=len(rows),
                              residual=abs(rows[-1]["formation_energy_eV"]
                                           - rows[-2]["formation_energy_eV"])
                              if len(rows) > 1 else float("nan"),
                              residual_metric="change between the last two supercells (eV)",
                              tolerance=tolerance_eV,
                              message="Converged in supercell size." if converged else
                              "Not shown to be converged in supercell size.")
    return Result("vacancy_formation_energy", rows[-1]["formation_energy_eV"], "eV", prov,
                  convergence=convergence,
                  extra={"series": rows, "bulk_energy_per_atom_eV":
                         reference["energy_per_atom_eV"]})


def surface_energy(material: Any, potential: Any, miller: Sequence[int],
                   layers: Sequence[int] = (6, 8, 10), in_plane: Tuple[int, int] = (1, 1),
                   vacuum_A: float = 15.0, relax: bool = True, fmax_eV_A: float = 1e-3,
                   tolerance_J_m2: float = 0.005) -> Result:
    """Surface energy in J/m^2 from slabs of increasing thickness."""
    from ..structure_builder.lattice import bulk as build_bulk
    from ..structure_builder.surface import make_surface

    reference = equilibrium_bulk(build_bulk(material), potential)
    e_bulk = reference["energy_per_atom_eV"]
    factor = reference["scale"]
    rows = []
    for n_layers in layers:
        slab = make_surface(material, tuple(miller), size=(in_plane[0], in_plane[1],
                                                           int(n_layers)),
                            vacuum_A=vacuum_A / factor)
        slab = _scaled(slab, factor)
        area = float(np.linalg.norm(np.cross(slab.cell.matrix[0], slab.cell.matrix[1])))
        energy = (_relaxed_energy(potential, slab, fmax_eV_A)[0] if relax
                  else float(potential.energy_and_forces(slab)[0]))
        unrelaxed = float(potential.energy_and_forces(slab)[0])
        rows.append({"repeats": int(n_layers), "atoms": len(slab), "area_A2": area,
                     "surface_energy_J_m2": (energy - len(slab) * e_bulk) / (2 * area)
                     * EV_A2_TO_J_M2,
                     "unrelaxed_surface_energy_J_m2": (unrelaxed - len(slab) * e_bulk)
                     / (2 * area) * EV_A2_TO_J_M2})
    change = (abs(rows[-1]["surface_energy_J_m2"] - rows[-2]["surface_energy_J_m2"])
              if len(rows) > 1 else float("nan"))
    converged = len(rows) > 1 and change <= tolerance_J_m2
    prov = Provenance(
        model=f"energetics/surface[{_label(potential)}]",
        fidelity=getattr(potential, "fidelity", None), origin=Origin.CALCULATED,
        approximations=["Symmetric slab with two equivalent surfaces, scaled to the "
                        "potential's equilibrium lattice constant; ionic relaxation "
                        + ("included." if relax else "omitted."),
                        "Bulk reference energy per atom from the same potential."],
        parameters={"miller": list(miller), "layers": list(layers),
                    "in_plane": list(in_plane), "vacuum_A": vacuum_A,
                    "fmax_eV_A": fmax_eV_A, "lattice_scale": factor},
        references=[])
    convergence = Convergence(converged=converged, iterations=len(rows), residual=change,
                              residual_metric="change between the two thickest slabs (J/m^2)",
                              tolerance=tolerance_J_m2,
                              message="Converged in slab thickness." if converged else
                              "Not shown to be converged in slab thickness.")
    return Result("surface_energy", rows[-1]["surface_energy_J_m2"], "J/m^2", prov,
                  convergence=convergence,
                  extra={"series": rows, "bulk_energy_per_atom_eV": e_bulk,
                         "lattice_scale": factor})
