"""Deterministic fixed-cell basin hopping for classical atomistic structures."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core_model.structure import Structure
from ..project_format.arrays import StoredArray
from ..provenance import Convergence, Origin, Provenance, Result, digest
from .external_bias import has_external_bias

Progress = Callable[[float, str], Optional[bool]]


class StructureSearchError(ValueError):
    """Invalid structure-search input or result."""


class StructureSearchRefused(StructureSearchError):
    """A scientifically unsupported structure-search request."""


class StructureSearchCancelled(StructureSearchError):
    """The caller cancelled a structure search."""


@dataclass(frozen=True)
class BasinHoppingSettings:
    trials: int = 20
    displacement_A: float = 0.20
    temperature_eV: float = 0.10
    fmax_eV_A: float = 0.05
    relaxation_steps: int = 300
    keep: int = 10
    seed: int = 0
    energy_tolerance_eV: float = 1e-5
    rmsd_tolerance_A: float = 1e-3
    require_converged: bool = True

    def validate(self) -> "BasinHoppingSettings":
        for name, value, lower, upper in (
            ("trials", self.trials, 1, 500),
            ("relaxation_steps", self.relaxation_steps, 1, 100000),
            ("keep", self.keep, 1, 1000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise StructureSearchError(
                    f"{name} must be an integer from {lower} through {upper}.")
        if self.keep > self.trials + 1:
            raise StructureSearchError("keep cannot exceed trials + 1.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise StructureSearchError("seed must be an integer.")
        for name, value, lower, upper, inclusive in (
            ("displacement_A", self.displacement_A, 1e-6, 10.0, True),
            ("temperature_eV", self.temperature_eV, 0.0, 100.0, True),
            ("fmax_eV_A", self.fmax_eV_A, 1e-8, 100.0, True),
            ("energy_tolerance_eV", self.energy_tolerance_eV, 0.0, 10.0, False),
            ("rmsd_tolerance_A", self.rmsd_tolerance_A, 0.0, 10.0, False),
        ):
            if isinstance(value, bool):
                raise StructureSearchError(f"{name} must be a finite number.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise StructureSearchError(f"{name} must be a finite number.") from exc
            valid_lower = number >= lower if inclusive else number > lower
            if not math.isfinite(number) or not valid_lower or number > upper:
                relation = "at least" if inclusive else "greater than"
                raise StructureSearchError(
                    f"{name} must be {relation} {lower:g} and no more than {upper:g}.")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchMinimum:
    rank: int
    structure: Structure
    energy_eV: float
    trial: int
    fmax_eV_A: float
    converged: bool


@dataclass
class BasinHoppingResult:
    minima: List[SearchMinimum]
    history: Dict[str, np.ndarray]
    provenance: Provenance
    convergence: Convergence
    diagnostics: Dict[str, Any]

    @property
    def best(self) -> SearchMinimum:
        if not self.minima:
            raise StructureSearchError("The search did not retain any minima.")
        return self.minima[0]

    def results(self) -> Dict[str, Result]:
        summary = {
            "retained_minima": len(self.minima),
            "trials_attempted": int(self.diagnostics["trials_attempted"]),
            "accepted_moves": int(self.diagnostics["accepted_moves"]),
            "converged_relaxations": int(self.diagnostics["converged_relaxations"]),
            "candidate_energies_eV": [minimum.energy_eV for minimum in self.minima],
            "candidate_trials": [minimum.trial for minimum in self.minima],
        }
        return {
            "best_structure": Result(
                "structure_search_best", self.best.structure.copy(), "",
                self.provenance, convergence=self.convergence,
                extra={**summary, "rank": 1}),
            "best_energy": Result(
                "structure_search_best_energy", self.best.energy_eV, "eV",
                self.provenance, convergence=self.convergence, extra=summary),
            "candidate_energies": Result(
                "structure_search_candidate_energies",
                np.asarray([minimum.energy_eV for minimum in self.minima]),
                "eV", self.provenance, convergence=self.convergence,
                extra={"trials": [minimum.trial for minimum in self.minima]}),
            "search_history": Result(
                "structure_search_history",
                {key: value.tolist() for key, value in self.history.items()},
                "mixed", self.provenance, convergence=self.convergence,
                extra=dict(self.diagnostics)),
        }

    def stored_arrays(self) -> Dict[str, StoredArray]:
        meta = {
            "model": self.provenance.model,
            "inputs_digest": self.provenance.inputs_digest,
            "atom_ids": self.best.structure.ids.tolist(),
        }
        positions = np.stack([minimum.structure.positions for minimum in self.minima], axis=0)
        return {
            "candidate_positions": StoredArray(
                positions, "A", "Ranked fixed-cell structure-search minima", "trajectory",
                {**meta, "energies_eV": [minimum.energy_eV for minimum in self.minima]}),
            "candidate_energies": StoredArray(
                np.asarray([minimum.energy_eV for minimum in self.minima]), "eV",
                "Ranked structure-search energies", "table", meta),
            "trial_energy": StoredArray(
                self.history["energy_eV"], "eV", "Relaxed energy for every trial", "table",
                meta),
            "trial_fmax": StoredArray(
                self.history["fmax_eV_A"], "eV/A",
                "Residual force for every trial relaxation", "table", meta),
            "trial_accepted": StoredArray(
                self.history["accepted"], "boolean",
                "Metropolis acceptance decision for every trial", "table", meta),
            "trial_converged": StoredArray(
                self.history["converged"], "boolean",
                "Relaxation convergence for every trial", "table", meta),
        }


def _report(progress: Optional[Progress], cancelled, fraction: float, message: str) -> None:
    if cancelled is not None and cancelled():
        raise StructureSearchCancelled("Structure search cancelled by caller.")
    if progress is not None and progress(float(fraction), message) is False:
        raise StructureSearchCancelled("Structure search cancelled by progress callback.")


def _rmsd(first: Structure, second: Structure) -> float:
    delta = first.cell.minimum_image(second.positions - first.positions)
    free = ~(first.fixed | second.fixed)
    if np.any(free):
        delta = delta.copy()
        delta[free] -= np.mean(delta[free], axis=0)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _unique(candidate: SearchMinimum, minima: Sequence[SearchMinimum],
            settings: BasinHoppingSettings) -> bool:
    for existing in minima:
        if abs(candidate.energy_eV - existing.energy_eV) <= settings.energy_tolerance_eV:
            if _rmsd(candidate.structure, existing.structure) <= settings.rmsd_tolerance_A:
                return False
    return True


def basin_hopping(
    structure: Structure,
    potential,
    settings: Optional[BasinHoppingSettings] = None,
    *,
    progress: Optional[Progress] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> BasinHoppingResult:
    """Search fixed-cell local minima without changing the input structure."""
    started = time.perf_counter()
    settings = (settings or BasinHoppingSettings()).validate()
    if len(structure) == 0:
        raise StructureSearchRefused("Structure search requires at least one atom.")
    if not np.any(~structure.fixed):
        raise StructureSearchRefused("Structure search requires at least one movable atom.")
    if has_external_bias(structure):
        raise StructureSearchRefused(
            "Structure search refuses external forces and restraints because they change "
            "the energy landscape being ranked. Clear biases before searching.")
    supported, reason = potential.supports(structure)
    if not supported:
        raise StructureSearchRefused(f"The force model does not support this structure: {reason}")

    from ..solvers.classical import ClassicalSolver

    solver = ClassicalSolver(potential)
    rng = np.random.default_rng(settings.seed)
    free = ~structure.fixed

    def relax(candidate: Structure):
        result = solver.relax(
            candidate, fmax_eV_A=settings.fmax_eV_A,
            max_steps=settings.relaxation_steps, in_place=False)
        energy_result = result.results["energy"]
        forces_result = result.results["forces"]
        force_array = np.asarray(forces_result.value, dtype=float)
        fmax = float(np.linalg.norm(force_array[free], axis=1).max())
        converged = bool(result.convergence and result.convergence.converged)
        return result.structure.copy(), float(energy_result.value), fmax, converged

    _report(progress, cancelled, 0.0, "relaxing starting structure")
    current, current_energy, current_fmax, current_converged = relax(structure.copy())
    if settings.require_converged and not current_converged:
        raise StructureSearchRefused(
            "The starting structure did not converge within the search relaxation budget. "
            "Relax it first or set require_converged=False for an exploratory search.")
    minima = [SearchMinimum(1, current.copy(), current_energy, 0,
                            current_fmax, current_converged)]
    trial_values = []
    accepted_moves = 0
    converged_relaxations = int(current_converged)

    for trial in range(1, settings.trials + 1):
        _report(progress, cancelled, (trial - 1) / settings.trials,
                f"relaxing trial {trial} of {settings.trials}")
        proposal = current.copy()
        displacement = rng.normal(0.0, settings.displacement_A, proposal.positions.shape)
        displacement[~free] = 0.0
        if int(free.sum()) > 1:
            displacement[free] -= np.mean(displacement[free], axis=0)
        proposal.positions = proposal.positions + displacement
        if any(proposal.cell.pbc):
            proposal.wrap()
        relaxed, energy, fmax, converged = relax(proposal)
        converged_relaxations += int(converged)
        delta = energy - current_energy
        if not converged and settings.require_converged:
            accepted = False
        elif delta <= 0:
            accepted = True
        elif settings.temperature_eV == 0:
            accepted = False
        else:
            accepted = bool(rng.random() < math.exp(-delta / settings.temperature_eV))
        if accepted:
            current = relaxed.copy()
            current_energy = energy
            accepted_moves += 1
        candidate = SearchMinimum(0, relaxed.copy(), energy, trial, fmax, converged)
        if (converged or not settings.require_converged) and _unique(candidate, minima, settings):
            minima.append(candidate)
            minima.sort(key=lambda item: (item.energy_eV, item.trial))
            minima = minima[:settings.keep]
        trial_values.append((trial, energy, fmax, converged, accepted, delta))

    _report(progress, cancelled, 1.0, "ranking minima")
    for rank, minimum in enumerate(minima, 1):
        minimum.rank = rank
    history = {
        "trial": np.asarray([row[0] for row in trial_values], dtype=np.int64),
        "energy_eV": np.asarray([row[1] for row in trial_values], dtype=float),
        "fmax_eV_A": np.asarray([row[2] for row in trial_values], dtype=float),
        "converged": np.asarray([row[3] for row in trial_values], dtype=bool),
        "accepted": np.asarray([row[4] for row in trial_values], dtype=bool),
        "delta_energy_eV": np.asarray([row[5] for row in trial_values], dtype=float),
    }
    described = potential.describe()
    parameters = {
        "settings": settings.as_dict(),
        "potential": described.get("parameters", {}),
        "algorithm": "Gaussian-displacement basin hopping with FIRE local relaxation",
    }
    approximations = list(described.get("approximations", [])) + [
        "Fixed composition and fixed simulation cell.",
        "Basin hopping is stochastic global exploration and does not prove the global minimum.",
        "Duplicate detection compares ordered atom coordinates after translation removal and "
        "does not identify every permutation or crystallographic symmetry equivalence.",
    ]
    provenance = Provenance(
        model=f"structure-search/{potential.model_label()}",
        fidelity=potential.fidelity,
        origin=Origin.CALCULATED,
        parameters=parameters,
        approximations=approximations,
        references=list(described.get("references", [])) + [
            "D. J. Wales and J. P. K. Doye, J. Phys. Chem. A 101 (1997) 5111"
        ],
        boundary_conditions=("3D periodic fixed cell" if all(structure.cell.pbc)
                             else f"fixed cell, pbc={structure.cell.pbc}"),
        seed=settings.seed,
        inputs_digest=digest({"structure": structure.as_dict(), "settings": settings.as_dict()}),
    )
    convergence = Convergence(
        converged=converged_relaxations == settings.trials + 1,
        iterations=settings.trials,
        residual=float(minima[0].fmax_eV_A),
        residual_metric="best retained minimum max |F|",
        tolerance=settings.fmax_eV_A,
        message=(f"{converged_relaxations} of {settings.trials + 1} local relaxations "
                 "converged; global optimality is not a convergence claim."),
    )
    diagnostics = {
        "trials_attempted": settings.trials,
        "accepted_moves": accepted_moves,
        "converged_relaxations": converged_relaxations,
        "retained_minima": len(minima),
        "wall_time_s": time.perf_counter() - started,
        "global_minimum_proven": False,
    }
    return BasinHoppingResult(minima, history, provenance, convergence, diagnostics)
