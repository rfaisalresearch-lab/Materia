"""Cross-model disagreement for classical energies and forces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..project_format.arrays import StoredArray
from ..provenance import Fidelity, Origin, Provenance, Result, digest
from .external_bias import has_external_bias


class ModelEnsembleError(ValueError):
    """An invalid or unsupported model-ensemble request."""


@dataclass
class ForceEnsembleResult:
    labels: Tuple[str, ...]
    energies_eV: np.ndarray
    forces_eV_A: np.ndarray
    mean_forces_eV_A: np.ndarray
    component_std_eV_A: np.ndarray
    force_disagreement_eV_A: np.ndarray
    pairwise_force_rms_eV_A: np.ndarray
    metrics: Dict[str, Any]
    provenance: Provenance

    def results(self) -> Dict[str, Result]:
        extra = {**self.metrics, "models": list(self.labels)}
        return {
            "mean_forces": Result(
                "ensemble_mean_forces", self.mean_forces_eV_A.copy(), "eV/A",
                self.provenance,
                uncertainty=float(np.max(self.component_std_eV_A)),
                uncertainty_kind="maximum component model standard deviation",
                extra=extra),
            "force_disagreement": Result(
                "ensemble_force_disagreement", self.force_disagreement_eV_A.copy(),
                "eV/A", self.provenance, extra=extra),
            "raw_model_energies": Result(
                "ensemble_raw_model_energies", self.energies_eV.copy(), "eV",
                self.provenance, extra={
                    **extra,
                    "warning": (
                        "Absolute energies from different potentials can have different "
                        "reference zeros. Their spread is not an energy uncertainty."),
                }),
        }

    def stored_arrays(self) -> Dict[str, StoredArray]:
        meta = {
            "models": list(self.labels),
            "inputs_digest": self.provenance.inputs_digest,
            "atom_ids": self.metrics["atom_ids"],
        }
        return {
            "model_forces": StoredArray(
                self.forces_eV_A, "eV/A", "Forces from every ensemble model",
                "vectors", meta),
            "mean_forces": StoredArray(
                self.mean_forces_eV_A, "eV/A", "Cross-model mean atomic forces",
                "vectors", meta),
            "force_component_std": StoredArray(
                self.component_std_eV_A, "eV/A",
                "Sample standard deviation across models for each force component",
                "vectors", meta),
            "force_disagreement": StoredArray(
                self.force_disagreement_eV_A, "eV/A",
                "Per-atom RMS vector disagreement around the ensemble mean", "table", meta),
            "pairwise_force_rms": StoredArray(
                self.pairwise_force_rms_eV_A, "eV/A",
                "Pairwise whole-structure force RMS difference", "matrix", meta),
            "raw_model_energies": StoredArray(
                self.energies_eV, "eV",
                "Absolute model energies with model-dependent reference zeros", "table", meta),
        }


@dataclass
class EnergyChangeEnsembleResult:
    labels: Tuple[str, ...]
    before_energies_eV: np.ndarray
    after_energies_eV: np.ndarray
    changes_eV: np.ndarray
    mean_change_eV: float
    std_change_eV: float
    metrics: Dict[str, Any]
    provenance: Provenance

    def results(self) -> Dict[str, Result]:
        extra = {**self.metrics, "models": list(self.labels),
                 "changes_eV": self.changes_eV.tolist()}
        return {
            "energy_change": Result(
                "ensemble_energy_change", self.mean_change_eV, "eV",
                self.provenance, uncertainty=self.std_change_eV,
                uncertainty_kind="model standard deviation", extra=extra),
            "model_energy_changes": Result(
                "model_energy_changes", self.changes_eV.copy(), "eV",
                self.provenance, extra=extra),
        }

    def stored_arrays(self) -> Dict[str, StoredArray]:
        meta = {"models": list(self.labels), "inputs_digest": self.provenance.inputs_digest}
        return {
            "before_energies": StoredArray(
                self.before_energies_eV, "eV", "Reference-state energy by model", "table", meta),
            "after_energies": StoredArray(
                self.after_energies_eV, "eV", "Target-state energy by model", "table", meta),
            "energy_changes": StoredArray(
                self.changes_eV, "eV",
                "Within-model target minus reference energy", "table", meta),
        }


def _sources(potentials: Sequence[Any], labels: Optional[Sequence[str]]) -> List[Tuple[str, Any]]:
    if len(potentials) < 2:
        raise ModelEnsembleError("A model ensemble requires at least two potentials.")
    if labels is not None and len(labels) != len(potentials):
        raise ModelEnsembleError("labels must contain one name for every potential.")
    chosen = ([str(value) for value in labels] if labels is not None
              else [str(getattr(potential, "name", type(potential).__name__))
                    for potential in potentials])
    if any(not value.strip() for value in chosen):
        raise ModelEnsembleError("Model labels must be non-empty.")
    if len(set(chosen)) != len(chosen):
        raise ModelEnsembleError(
            "Model labels must be unique so stored ensemble axes are unambiguous.")
    return list(zip(chosen, potentials))


def _check_structure(structure: Structure) -> None:
    if len(structure) == 0:
        raise ModelEnsembleError("Model ensembles require at least one atom.")
    if has_external_bias(structure):
        raise ModelEnsembleError(
            "Model ensembles refuse external mechanical biases because the requested "
            "comparison is intrinsic model disagreement.")


def _evaluate(structure: Structure, sources: Sequence[Tuple[str, Any]]):
    energies = []
    forces = []
    descriptions = []
    fidelities = []
    for label, potential in sources:
        supported, reason = potential.supports(structure)
        if not supported:
            raise ModelEnsembleError(f"Model {label!r} does not support the structure: {reason}")
        energy, force = potential.energy_and_forces(structure.copy())
        force = np.asarray(force, dtype=float)
        if not np.isfinite(float(energy)) or force.shape != (len(structure), 3) or not np.all(
                np.isfinite(force)):
            raise ModelEnsembleError(f"Model {label!r} returned non-finite or malformed values.")
        energies.append(float(energy))
        forces.append(force)
        descriptions.append(dict(potential.describe()))
        fidelities.append(getattr(potential, "fidelity", Fidelity.TIER1_CLASSICAL))
    return np.asarray(energies), np.stack(forces), descriptions, fidelities


def _provenance(model: str, structures: dict, sources, descriptions, fidelities,
                notes: str) -> Provenance:
    fidelity = (Fidelity.NON_PHYSICAL if Fidelity.NON_PHYSICAL in fidelities
                else Fidelity.TIER1_CLASSICAL)
    parameters = {
        "models": [
            {"label": label, "name": getattr(potential, "name", type(potential).__name__),
             "parameters": description.get("parameters", {})}
            for (label, potential), description in zip(sources, descriptions)
        ],
        "ensemble_size": len(sources),
    }
    references = []
    approximations = [
        "Model spread measures disagreement among the selected models, not calibrated "
        "predictive uncertainty and not experimental error."
    ]
    for description in descriptions:
        references.extend(str(value) for value in description.get("references", []))
        approximations.extend(str(value) for value in description.get("approximations", []))
    references = list(dict.fromkeys(references))
    return Provenance(
        model=model,
        fidelity=fidelity,
        origin=Origin.CALCULATED,
        parameters=parameters,
        approximations=approximations,
        references=references,
        boundary_conditions="; ".join(
            f"{name}: pbc={structure.cell.pbc}" for name, structure in structures.items()),
        inputs_digest=digest({
            "structures": {name: structure.as_dict() for name, structure in structures.items()},
            "models": parameters["models"],
        }),
        notes=notes,
    )


def force_ensemble(structure: Structure, potentials: Sequence[Any], *,
                   labels: Optional[Sequence[str]] = None) -> ForceEnsembleResult:
    """Evaluate per-atom force disagreement across two or more potentials."""
    _check_structure(structure)
    sources = _sources(potentials, labels)
    energies, forces, descriptions, fidelities = _evaluate(structure, sources)
    mean = np.mean(forces, axis=0)
    component_std = np.std(forces, axis=0, ddof=1)
    disagreement = np.sqrt(np.mean(np.sum((forces - mean[None, :, :]) ** 2, axis=2), axis=0))
    pairwise = np.zeros((len(sources), len(sources)), dtype=float)
    for first in range(len(sources)):
        for second in range(first + 1, len(sources)):
            value = float(np.sqrt(np.mean((forces[first] - forces[second]) ** 2)))
            pairwise[first, second] = pairwise[second, first] = value
    metrics = {
        "atom_ids": structure.ids.tolist(),
        "max_atom_disagreement_eV_A": float(np.max(disagreement)),
        "mean_atom_disagreement_eV_A": float(np.mean(disagreement)),
        "max_pairwise_force_rms_eV_A": float(np.max(pairwise)),
        "raw_energy_range_eV": float(np.ptp(energies)),
        "raw_energy_spread_is_uncertainty": False,
    }
    provenance = _provenance(
        "model-ensemble/classical-forces", {"structure": structure}, sources,
        descriptions, fidelities,
        "Absolute cross-potential energy offsets are retained for audit only.")
    return ForceEnsembleResult(
        tuple(label for label, _ in sources), energies, forces, mean, component_std,
        disagreement, pairwise, metrics, provenance)


def _same_atoms(before: Structure, after: Structure) -> bool:
    return bool(np.array_equal(before.ids, after.ids) and
                np.array_equal(before.numbers, after.numbers))


def energy_change_ensemble(before: Structure, after: Structure,
                           potentials: Sequence[Any], *,
                           labels: Optional[Sequence[str]] = None
                           ) -> EnergyChangeEnsembleResult:
    """Compare within-model energy changes so arbitrary energy zeros cancel."""
    _check_structure(before)
    _check_structure(after)
    if not _same_atoms(before, after):
        raise ModelEnsembleError(
            "Energy-change ensembles require the same stable atom-id and element order in "
            "both structures so every model compares corresponding atoms.")
    sources = _sources(potentials, labels)
    before_energy, _, descriptions, fidelities = _evaluate(before, sources)
    after_energy, _, after_descriptions, after_fidelities = _evaluate(after, sources)
    if [digest(value) for value in descriptions] != [digest(value) for value in after_descriptions]:
        raise ModelEnsembleError("A potential changed its declared parameters between states.")
    changes = after_energy - before_energy
    mean = float(np.mean(changes))
    std = float(np.std(changes, ddof=1))
    metrics = {
        "ensemble_size": len(sources),
        "minimum_change_eV": float(np.min(changes)),
        "maximum_change_eV": float(np.max(changes)),
        "range_eV": float(np.ptp(changes)),
        "sign_agreement": bool(np.all(changes >= 0) or np.all(changes <= 0)),
        "reference_offsets_cancelled_within_each_model": True,
    }
    provenance = _provenance(
        "model-ensemble/classical-energy-change", {"before": before, "after": after},
        sources, descriptions, fidelities + after_fidelities,
        "Every reported change is target minus reference within the same model.")
    return EnergyChangeEnsembleResult(
        tuple(label for label, _ in sources), before_energy, after_energy, changes,
        mean, std, metrics, provenance)
