"""The GPAW solver: configuration in, traced results out.

This is the only Tier-3 adapter in Materia that actually drives its code.  It
computes one self-consistent total energy and the forces that go with it, and
it declares nothing else: no relaxation, no density, no density of states, no
band structure, no spin or charged-system support.  Those are things GPAW can
do and this driver does not, so asking for them is refused with the reason
rather than answered.

A calculation that did not converge is never returned as a calculated result.
It comes back as an unsupported result carrying the iteration count, the
tolerances it failed to meet and the solver's own message, because a
half-solved Kohn-Sham problem is not an energy.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ...core_model.structure import Structure
from ...provenance import (
    Convergence, Fidelity, Origin, Provenance, Result, unsupported,
)
from ...version import __version__
from ..base import Capability, Solver, SolverResult, SupportReport
from . import conversion, runner
from .environment import GPAWEnvironment, discover
from .settings import GPAWSettings, GPAWSettingsError, validate

CAPABILITIES = {Capability.ENERGY, Capability.FORCES}

REFERENCE = ("J. Enkovaara et al., J. Phys.: Condens. Matter 22 (2010) 253202; "
             "J. J. Mortensen, L. B. Hansen and K. W. Jacobsen, "
             "Phys. Rev. B 71 (2005) 035109")


class GPAWSolver(Solver):
    """Single-point DFT through a GPAW running in its own interpreter."""

    name = "external:gpaw"
    fidelity = Fidelity.TIER3_EXTERNAL
    capabilities = CAPABILITIES
    description = ("Projector augmented-wave density functional theory, driven in a "
                   "separate GPAW interpreter. Single-point total energy and forces.")

    def __init__(self, environment: Optional[GPAWEnvironment] = None) -> None:
        self._environment = environment

    @property
    def environment(self) -> GPAWEnvironment:
        if self._environment is None:
            self._environment = discover()
        return self._environment

    @property
    def installed(self) -> bool:
        return self.environment.operational

    def supports(self, structure: Structure) -> SupportReport:
        environment = self.environment
        if not environment.operational:
            return SupportReport(False, [environment.blocking_reason()], [])
        blocking: List[str] = []
        warnings: List[str] = []
        if len(structure) == 0:
            blocking.append("The structure has no atoms.")
        symbols = sorted({s for s in structure.symbols()})
        known = set(environment.dataset_elements)
        if known:
            missing = [s for s in symbols if s not in known]
            if missing:
                blocking.append(
                    f"No PAW dataset is installed for {', '.join(missing)}. "
                    f"The dataset directory holds {len(known)} elements.")
        if len(structure) > 200:
            warnings.append(
                f"{len(structure)} atoms is a large first-principles calculation; "
                "expect it to take a long time on this machine.")
        return SupportReport(not blocking, blocking, warnings)

    def describe(self) -> dict:
        environment = self.environment
        description = super().describe()
        description.update({
            "available": environment.operational,
            "environment": environment.as_dict(),
            "reference": REFERENCE,
            "license_note": ("GPAW is LGPL-3.0-or-later and is neither bundled nor "
                             "linked: Materia runs it as a separate program. The PAW "
                             "datasets carry their own licence."),
            "install_hint": environment.install_hint(),
            "driven": True,
        })
        return description

    def run(self, structure: Structure, task: str = "energy", **kwargs) -> SolverResult:
        if task not in ("energy", "single_point"):
            out = SolverResult(solver=self.name, structure=structure)
            out.results[task] = unsupported(
                task, self.name,
                f"The GPAW driver runs single-point energy and forces only, and does "
                f"not implement {task!r}. GPAW itself can do more; this driver does "
                "not drive it, and will not pretend to.",
                suggested_models=[], fidelity=self.fidelity)
            return out
        return self.single_point(structure, **kwargs)

    def single_point(self, structure: Structure,
                     settings: Optional[Dict[str, Any]] = None,
                     progress: Optional[Callable[[dict], None]] = None,
                     cancelled: Optional[Callable[[], bool]] = None,
                     timeout_s: Optional[float] = None) -> SolverResult:
        """One self-consistent calculation. The structure is never mutated."""
        started = time.perf_counter()
        out = SolverResult(solver=self.name, structure=structure)
        environment = self.environment

        report = self.supports(structure)
        if not report.ok:
            reason = "; ".join(report.blocking)
            out.results["energy"] = unsupported(
                "energy", self.name, reason,
                suggested_models=[], unit="eV", fidelity=self.fidelity)
            out.results["energy"].extra["install_hint"] = environment.install_hint()
            out.results["energy"].extra["environment"] = environment.as_dict()
            out.log.extend(report.blocking)
            return out

        configuration = validate(settings, structure)
        spec = conversion.to_worker_spec(structure)

        run = runner.run(spec, configuration.as_dict(), environment,
                         progress=progress, cancelled=cancelled, timeout_s=timeout_s)
        out.wall_time_s = time.perf_counter() - started
        out.log.extend(report.warnings)
        out.log.extend(run.result.get("warnings", []) or [])

        provenance = self._provenance(configuration, structure, run, environment)
        convergence = Convergence(
            converged=run.converged,
            iterations=int(run.iterations),
            residual=None,
            residual_metric="SCF energy change per electron",
            tolerance=configuration.energy_tol_eV_per_electron,
            message=_convergence_message(run, configuration),
        )
        out.convergence = convergence

        if run.status != runner.STATUS_CONVERGED:
            blocked = unsupported(
                "energy", self.name, _refusal_reason(run, configuration),
                suggested_models=[], unit="eV", fidelity=self.fidelity)
            blocked.provenance = provenance
            blocked.provenance.origin = Origin.UNSUPPORTED
            blocked.convergence = convergence
            blocked.extra.update({
                "status": run.status,
                "iterations": run.iterations,
                "wall_time_s": run.wall_time_s,
                "gpaw_message": run.result.get("message", ""),
                "gpaw_error": run.error,
                "gpaw_log_tail": (run.result.get("log_tail", "") or "")[-2000:],
                "stderr_tail": run.stderr[-2000:],
            })
            out.results["energy"] = blocked
            out.log.append(blocked.unsupported_reason)
            return out

        energy = float(run.result["energy_eV"])
        forces = np.asarray(run.result["forces_eV_per_A"], dtype=float)
        by_id = conversion.forces_by_id(spec, run.result["forces_eV_per_A"])

        out.results["energy"] = Result(
            "energy", energy, "eV", provenance, convergence=convergence,
            extra={
                "energy_per_atom_eV": energy / max(1, len(structure)),
                "status": run.status,
                "iterations": run.iterations,
                "wall_time_s": run.wall_time_s,
                "fermi_level_eV": run.result.get("fermi_level_eV"),
                "n_electrons": run.result.get("n_electrons"),
                "n_bands": run.result.get("n_bands"),
                "n_kpoints": run.result.get("n_kpoints"),
                "magnetic_moment": run.result.get("magnetic_moment"),
            },
        )
        out.results["forces"] = Result(
            "forces", forces, "eV/A", provenance, convergence=convergence,
            extra={
                "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
                "by_atom_id": by_id,
                "constraint_applied": False,
                "note": ("Forces are the unconstrained Hellmann-Feynman forces GPAW "
                         "computed. Atoms marked fixed in Materia were passed to GPAW "
                         "as a constraint but their forces are reported as calculated."),
            },
        )
        return out

    def _provenance(self, configuration: GPAWSettings, structure: Structure,
                    run: runner.GPAWRun, environment: GPAWEnvironment) -> Provenance:
        result = run.result
        datasets = result.get("datasets", []) or []
        pbc = tuple(bool(p) for p in structure.cell.pbc)
        approximations = [
            f"Density functional theory in the {configuration.xc} approximation to "
            "exchange and correlation. The exchange-correlation functional is an "
            "approximation, and its error is not bounded by the convergence "
            "tolerances below.",
            "Projector augmented-wave treatment of the core region, with frozen "
            "cores from the PAW datasets identified in this record.",
        ]
        if configuration.mode == "pw":
            approximations.append(
                f"Plane-wave basis truncated at {configuration.cutoff_eV:g} eV; the "
                "energy is not converged with respect to this cutoff unless that was "
                "checked separately.")
        elif configuration.mode == "fd":
            approximations.append(
                f"Real-space grid of {configuration.grid_spacing_A:g} A; the energy is "
                "not converged with respect to the spacing unless that was checked "
                "separately.")
        else:
            approximations.append(
                f"LCAO basis {configuration.basis}; a finite atom-centred basis is not "
                "a variational limit.")
        approximations.append(
            f"Brillouin zone sampled on a {'x'.join(str(k) for k in configuration.kpoints)} "
            "Monkhorst-Pack grid.")
        if not all(pbc):
            approximations.append(
                "Open boundaries in at least one direction: the result depends on the "
                "amount of vacuum in the cell.")

        return Provenance(
            model="external:gpaw/paw-dft",
            fidelity=Fidelity.TIER3_EXTERNAL,
            origin=Origin.CALCULATED,
            approximations=approximations,
            tolerances={
                "energy_eV_per_electron": configuration.energy_tol_eV_per_electron,
                "density_electrons": configuration.density_tol_electrons,
                "max_iterations": configuration.max_iterations,
            },
            boundary_conditions=(
                "periodic in " + ", ".join(a for a, p in zip("abc", pbc) if p)
                if any(pbc) else "open in all directions"),
            parameters={
                "task": configuration.task,
                "xc": configuration.xc,
                "mode": configuration.mode,
                "cutoff_eV": configuration.cutoff_eV if configuration.mode == "pw" else None,
                "grid_spacing_A": (configuration.grid_spacing_A
                                   if configuration.mode == "fd" else None),
                "basis": configuration.basis if configuration.mode == "lcao" else None,
                "kpoints": list(configuration.kpoints),
                "occupations": configuration.occupations,
                "smearing_eV": configuration.smearing_eV,
                "charge_e": configuration.charge,
                "spin_polarized": configuration.spin_polarized,
                "n_atoms": len(structure),
                "formula": structure.formula(),
                "gpaw_version": result.get("gpaw_version", environment.gpaw_version),
                "ase_version": result.get("ase_version", environment.ase_version),
                "gpaw_python": environment.interpreter,
                "gpaw_python_version": result.get("worker_python",
                                                  environment.python_version),
                "mpi_world_size": result.get("mpi_world_size",
                                             environment.mpi_world_size),
                "paw_datasets": datasets,
                "paw_dataset_paths": list(environment.setup_paths),
                "scf_iterations": run.iterations,
                "converged": run.converged,
                "status": run.status,
                "wall_time_s": run.wall_time_s,
                "materia_version": __version__,
            },
            references=[REFERENCE],
            dataset="; ".join(
                f"{d.get('symbol')}:{d.get('file')}@{str(d.get('fingerprint'))[:8]}"
                for d in datasets),
            dataset_license=("PAW datasets are distributed with GPAW under their own "
                             "licence and are not redistributed by Materia."),
            notes=("Computed by GPAW " + str(result.get("gpaw_version", "")) +
                   " running as a separate program. Tier-3 external DFT: a "
                   "first-principles approximation, not an exact solution."),
        )


def _convergence_message(run: runner.GPAWRun, configuration: GPAWSettings) -> str:
    if run.status == runner.STATUS_CONVERGED:
        return (f"Self-consistent after {run.iterations} iterations, within "
                f"{configuration.energy_tol_eV_per_electron:g} eV/electron.")
    if run.status == runner.STATUS_NOT_CONVERGED:
        return (f"Stopped after {run.iterations} of at most "
                f"{configuration.max_iterations} iterations without reaching "
                f"{configuration.energy_tol_eV_per_electron:g} eV/electron.")
    if run.status == runner.STATUS_CANCELLED:
        return f"Cancelled after {run.iterations} iterations."
    return f"Failed after {run.iterations} iterations."


def _refusal_reason(run: runner.GPAWRun, configuration: GPAWSettings) -> str:
    if run.status == runner.STATUS_NOT_CONVERGED:
        detail = (run.result.get("message") or "").strip().replace("\n", " ")
        return (
            f"The GPAW calculation did not reach self-consistency: it ran "
            f"{run.iterations} of at most {configuration.max_iterations} iterations "
            f"without meeting {configuration.energy_tol_eV_per_electron:g} eV/electron. "
            "A half-solved Kohn-Sham problem is not a total energy, so no number is "
            "reported. Raise max_iterations, loosen the tolerance, or improve the "
            "starting geometry."
            + (f" GPAW said: {detail}" if detail else ""))
    if run.status == runner.STATUS_CANCELLED:
        return (f"The calculation was cancelled after {run.iterations} SCF "
                "iterations. No energy was produced.")
    return (f"The GPAW calculation failed after {run.iterations} SCF iterations: "
            f"{run.error or 'no error message was produced'}.")
