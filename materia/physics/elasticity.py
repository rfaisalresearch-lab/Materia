"""Elastic constants from the energy of strained cells, for any Materia potential.

``C_ij = (1/V0) d^2 E / d eps_i d eps_j`` in Voigt notation, by central
differences of the energy over homogeneous strains of a periodic cell,
``h -> (1 + eps) h``.  Diagonal terms use ``E(+e) - 2 E(0) + E(-e)``; off-
diagonal terms the four-point mixed difference.  Shear strains are
engineering strains (``eps_4 = 2 e_yz``).

The cell should be at zero stress: away from it, energy derivatives are not
the elastic constants that govern stress-strain response (they differ by
stress terms), and the result is refused unless ``allow_stress`` is set.
Internal coordinates are relaxed at every strain when ``relax_internal`` is
set, which matters for crystals with more than one atom per primitive cell
(diamond, wurtzite); the relaxed and clamped-ion tensors are both
physically defined and the record says which was computed.

From the tensor: Voigt, Reuss and Hill bulk and shear moduli, and Born
mechanical stability (all eigenvalues of C positive).
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..provenance import Fidelity, Origin, Provenance, Result

EV_A3_GPA = 160.21766208
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


class ElasticityError(ValueError):
    pass


def strain_matrix(voigt: np.ndarray) -> np.ndarray:
    e = np.zeros((3, 3))
    for k, (i, j) in enumerate(VOIGT):
        value = voigt[k] if k < 3 else voigt[k] / 2.0
        e[i, j] = value
        e[j, i] = value
    return e


def _strained(structure: Structure, voigt: np.ndarray) -> Structure:
    deformation = np.eye(3) + strain_matrix(voigt)
    out = structure.copy()
    out.cell = Cell(np.asarray(structure.cell.matrix) @ deformation.T, structure.cell.pbc)
    out.positions = np.asarray(structure.positions) @ deformation.T
    return out


def _energy(potential, structure: Structure, relax: bool, fmax: float) -> float:
    if not relax:
        return float(potential.energy_and_forces(structure)[0])
    from ..solvers.classical import ClassicalSolver

    out = ClassicalSolver(potential).relax(structure, fmax_eV_A=fmax, max_steps=5000)
    if not out.convergence.converged:
        raise ElasticityError("Internal relaxation did not converge at a strained cell.")
    return float(out.results["energy"].value)


def stress_by_differences(potential, structure: Structure, step: float = 1e-4) -> np.ndarray:
    """Voigt stress in GPa from first strain derivatives of the energy."""
    volume = structure.cell.volume
    out = np.zeros(6)
    for k in range(6):
        e = np.zeros(6)
        e[k] = step
        plus = _energy(potential, _strained(structure, e), False, 0)
        minus = _energy(potential, _strained(structure, -e), False, 0)
        out[k] = (plus - minus) / (2 * step) / volume * EV_A3_GPA
    return out


def elastic_tensor(potential, structure: Structure, strain: float = 5e-3,
                   relax_internal: bool = False, relax_fmax_eV_A: float = 1e-6,
                   allow_stress: bool = False, stress_tolerance_GPa: float = 0.1,
                   convergence_tolerance: float = 0.02) -> Result:
    """The 6x6 elastic tensor in GPa with moduli, stability and provenance.

    The tensor is computed at ``strain`` and again at twice it.  If any
    component larger than 1 percent of the largest changes by more than
    ``convergence_tolerance`` (relative), the result is labelled estimated:
    tabulated potentials such as EAM setfl files have kinks in their second
    derivatives, and too small a strain samples them as noise.
    """
    first = _tensor(potential, structure, strain, relax_internal, relax_fmax_eV_A,
                    allow_stress, stress_tolerance_GPa)
    second = _tensor(potential, structure, 2 * strain, relax_internal, relax_fmax_eV_A,
                     True, stress_tolerance_GPa)
    scale = np.abs(first.value).max()
    significant = np.abs(first.value) > 0.01 * scale
    change = float(np.max(np.abs(first.value - second.value)[significant]
                          / np.abs(first.value)[significant]))
    converged = change <= convergence_tolerance
    first.extra["strain_check"] = {"second_strain": 2 * strain,
                                   "max_relative_change": change,
                                   "tolerance": convergence_tolerance, "converged": converged}
    first.uncertainty = float(np.abs(first.value - second.value).max())
    first.uncertainty_kind = "model-spread"
    if not converged:
        first.provenance.origin = Origin.ESTIMATED
        first.provenance.notes = (
            f"Elastic constants changed by {change:.1%} when the strain was doubled to "
            f"{2 * strain:g}; increase the strain or use a smoother potential.")
    return first


def _tensor(potential, structure: Structure, strain: float, relax_internal: bool,
            relax_fmax_eV_A: float, allow_stress: bool, stress_tolerance_GPa: float) -> Result:
    t0 = time.perf_counter()
    if not all(structure.cell.pbc):
        raise ElasticityError("Elastic constants need a three-dimensional periodic cell.")
    if not 1e-5 <= strain <= 2e-2:
        raise ElasticityError("strain must lie between 1e-5 and 2e-2.")
    ok, why = potential.supports(structure)
    if not ok:
        raise ElasticityError(why)
    stress = stress_by_differences(potential, structure)
    if np.abs(stress).max() > stress_tolerance_GPa and not allow_stress:
        raise ElasticityError(
            f"The cell carries stress up to {np.abs(stress).max():.3g} GPa. Elastic constants "
            "are defined at zero stress: relax the lattice first or set allow_stress.")
    volume = structure.cell.volume
    reference = _energy(potential, structure, relax_internal, relax_fmax_eV_A)
    energy_cache = {}

    def energy(voigt: Tuple[float, ...]) -> float:
        key = tuple(np.round(voigt, 12))
        if key not in energy_cache:
            energy_cache[key] = _energy(potential, _strained(structure, np.array(voigt)),
                                        relax_internal, relax_fmax_eV_A)
        return energy_cache[key]

    c = np.zeros((6, 6))
    h = float(strain)
    for i in range(6):
        e = np.zeros(6)
        e[i] = h
        c[i, i] = (energy(tuple(e)) - 2 * reference + energy(tuple(-e))) / h ** 2
        for j in range(i + 1, 6):
            pp, pm, mp, mm = (np.zeros(6) for _ in range(4))
            pp[[i, j]] = (h, h)
            pm[[i, j]] = (h, -h)
            mp[[i, j]] = (-h, h)
            mm[[i, j]] = (-h, -h)
            c[i, j] = c[j, i] = (energy(tuple(pp)) - energy(tuple(pm)) - energy(tuple(mp))
                                 + energy(tuple(mm))) / (4 * h * h)
    c = c / volume * EV_A3_GPA
    s = np.linalg.inv(c)
    bulk_voigt = (c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[1, 2] + c[0, 2])) / 9
    shear_voigt = ((c[0, 0] + c[1, 1] + c[2, 2]) - (c[0, 1] + c[1, 2] + c[0, 2])
                   + 3 * (c[3, 3] + c[4, 4] + c[5, 5])) / 15
    bulk_reuss = 1 / (s[0, 0] + s[1, 1] + s[2, 2] + 2 * (s[0, 1] + s[1, 2] + s[0, 2]))
    shear_reuss = 15 / (4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[1, 2] + s[0, 2])
                        + 3 * (s[3, 3] + s[4, 4] + s[5, 5]))
    eigenvalues = np.linalg.eigvalsh(0.5 * (c + c.T))
    name = getattr(potential, "model_label", lambda: f"classical/{potential.name}")()
    prov = Provenance(
        model=f"elasticity/energy-strain[{name}]",
        fidelity=getattr(potential, "fidelity", Fidelity.TIER1_CLASSICAL),
        origin=Origin.CALCULATED,
        approximations=[
            "Second derivatives of the energy by central differences of homogeneous strains "
            f"of size {h:g}; zero temperature, static lattice.",
            "Internal coordinates relaxed at each strain." if relax_internal else
            "Clamped ions: internal coordinates follow the homogeneous strain.",
        ],
        parameters={"strain": h, "relax_internal": relax_internal,
                    "energy_evaluations": len(energy_cache) + 13,
                    "residual_stress_GPa": stress.tolist()},
        references=["J. F. Nye, Physical Properties of Crystals (Oxford, 1957)",
                    "R. Hill, Proc. Phys. Soc. A 65 (1952) 349"])
    return Result("elastic_tensor", c, "GPa", prov,
                  extra={"bulk_modulus_GPa": {"voigt": bulk_voigt, "reuss": bulk_reuss,
                                              "hill": 0.5 * (bulk_voigt + bulk_reuss)},
                         "shear_modulus_GPa": {"voigt": shear_voigt, "reuss": shear_reuss,
                                               "hill": 0.5 * (shear_voigt + shear_reuss)},
                         "born_stable": bool(eigenvalues.min() > 0),
                         "eigenvalues_GPa": eigenvalues.tolist(),
                         "residual_stress_GPa": stress.tolist(),
                         "wall_time_s": time.perf_counter() - t0})
