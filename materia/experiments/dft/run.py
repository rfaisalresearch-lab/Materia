"""Running one ground-state specification and turning GPAW's output into results.

:func:`execute` takes a checked :class:`~.spec.GroundStateSpec`, runs it in the
GPAW interpreter, and returns a :class:`GroundStateOutcome`: a run record,
one :class:`~materia.provenance.Result` per observable, and the large arrays
to be stored in chunks.  It never touches a structure or a project; storing
the outcome is :func:`~.records.store`'s job, so a run that fails, is
cancelled or does not converge cannot leave anything half-written.

Only a converged run yields numbers.  Anything else yields the run record
alone, with origin ``unsupported``, the reason, and the SCF iterations that
were completed, so it is visible what happened without any of it being
mistaken for a result.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ...project_format.arrays import StoredArray
from ...provenance import Convergence, Fidelity, Origin, Provenance, Result, unsupported
from ...provenance.classification import Classification
from ...solvers.gpaw_driver import runner
from ...version import __version__
from . import spec as specs
from .restart import RestartStore

WORKER = Path(__file__).resolve().parents[2] / "solvers" / "gpaw_driver" / \
    "ground_state_worker.py"

MODEL = "external:gpaw/ground-state-dft"

REFERENCES = [
    "J. Enkovaara et al., J. Phys.: Condens. Matter 22 (2010) 253202",
    "J. J. Mortensen et al., J. Chem. Phys. 160 (2024) 092503",
    "P. E. Bloechl, Phys. Rev. B 50 (1994) 17953",
]

EV_PER_A3_TO_GPA = 160.21766208

STATUS_CONVERGED = runner.STATUS_CONVERGED
STATUS_NOT_CONVERGED = runner.STATUS_NOT_CONVERGED
STATUS_CANCELLED = runner.STATUS_CANCELLED
STATUS_FAILED = runner.STATUS_FAILED

UNITS = {
    "energy": "eV", "forces": "eV/A", "stress": "eV/A^3", "fermi_level": "eV",
    "magnetic_moment": "mu_B", "eigenvalues": "eV",
    "occupations": "electrons per state and spin (0 to 1)",
    "density": "electrons/A^3", "spin_density": "electrons/A^3",
    "electrostatic_potential": "eV", "charge_accounting": "e",
    "scf_history": "", "run": "",
}

VOLUME_DESCRIPTIONS = {
    "density": "All-electron density including the frozen core, on GPAW's fine grid "
               "(twice the coarse grid).",
    "spin_density": "All-electron spin density n_up - n_down on the fine grid.",
    "electrostatic_potential": "Electrostatic potential energy of an electron, -e phi, "
                               "on the fine grid. GPAW convention.",
}

CLASSIFICATION = Classification.COMPUTATIONAL_PREDICTION.value


def new_run_id() -> str:
    return f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:6]}"


@dataclass
class GroundStateOutcome:
    """Everything one execution produced, not yet stored anywhere."""

    run_id: str
    spec: specs.GroundStateSpec
    status: str
    results: Dict[str, Result] = field(default_factory=dict)
    arrays: Dict[str, StoredArray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    wall_time_s: float = 0.0
    reason: str = ""

    @property
    def converged(self) -> bool:
        return self.status == STATUS_CONVERGED

    @property
    def record(self) -> Result:
        return self.results["run"]


def _approximations(spec: specs.GroundStateSpec) -> List[str]:
    out = [
        f"Kohn-Sham density functional theory with the {spec.xc} exchange-correlation "
        "functional. The functional is an approximation whose systematic error is not "
        "reduced by any numerical setting and is not estimated here.",
        "Projector augmented-wave method with frozen cores taken from the PAW datasets "
        "named in this record.",
        "Born-Oppenheimer approximation: nuclei are fixed classical point charges. The "
        "electronic ground state does not depend on nuclear mass, so isotopes do not "
        "change any value in this record.",
        "Scalar-relativistic datasets; no spin-orbit coupling.",
        "Energies are relative to the sum of the datasets' reference atomic energies; "
        "only differences between calculations with the same datasets are meaningful.",
    ]
    if spec.representation == "pw":
        out.append(f"Plane-wave basis truncated at {spec.cutoff_eV:g} eV.")
    elif spec.representation == "fd":
        out.append(f"Real-space grid with {spec.grid_spacing_A:g} A spacing.")
    else:
        out.append(f"LCAO basis {spec.basis} on a {spec.grid_spacing_A:g} A grid; a finite "
                   "atom-centred basis is not a variational limit.")
    if spec.boundary == "cluster":
        out.append("Isolated system in a finite box with zero boundary conditions; "
                   "values depend on the vacuum unless its convergence was checked.")
    else:
        k = "x".join(str(n) for n in spec.kpoints)
        out.append(f"Brillouin zone sampled on a {k} "
                   f"{'Gamma-centred' if spec.kpoints_gamma_centered else 'Monkhorst-Pack'} "
                   "grid, reduced by symmetry.")
    if spec.boundary in ("slab", "wire"):
        out.append("Open along the vacuum direction(s) with zero boundary conditions; "
                   "values depend on the vacuum thickness.")
    if spec.poisson == "dipole-layer":
        out.append("Dipole-layer correction across the vacuum of the slab.")
    if spec.poisson == "moment-corrected":
        out.append("Monopole and dipole of the charge distribution removed analytically "
                   "before the grid Poisson solve.")
    if abs(spec.charge_e) > 1e-9 and spec.boundary == "bulk":
        out.append("Charged cell neutralised by a uniform background; the finite-size "
                   "error is not corrected.")
    if spec.occupations != "fixed":
        out.append(f"{spec.occupations} occupation smearing of {spec.smearing_eV:g} eV; "
                   "the energy reported is the free energy E - TS, with the zero-width "
                   "extrapolation alongside.")
    if np.any(np.asarray(spec.external_field_V_per_A)):
        out.append("Uniform static external field; the finite box truncates any charge "
                   "that the field would pull into the vacuum beyond it.")
    if spec.spin_polarized:
        out.append("Collinear spin polarisation. Initial moments are a starting guess; "
                   "other magnetic states were not explored.")
    else:
        out.append("Spin-paired: both spin channels are constrained to be equal.")
    return out


def _boundary_text(spec: specs.GroundStateSpec) -> str:
    axes = "abc"
    periodic = [axes[i] for i in range(3) if spec.pbc[i]]
    open_axes = [axes[i] for i in range(3) if not spec.pbc[i]]
    parts = [spec.boundary]
    if periodic:
        parts.append("periodic along " + ", ".join(periodic))
    if open_axes:
        parts.append("open along " + ", ".join(open_axes))
    if spec.poisson != "gpaw-default":
        parts.append(spec.poisson)
    if abs(spec.charge_e) > 1e-9:
        parts.append(f"net charge {spec.charge_e:+g} e"
                     + (" with uniform background" if spec.boundary == "bulk" else ""))
    return "; ".join(parts)


def provenance(spec: specs.GroundStateSpec, run: Optional[runner.GPAWRun],
               origin: Origin, warnings: List[str], restart: dict) -> Provenance:
    reported = (run.result if run is not None else {}) or {}
    datasets = reported.get("datasets") or [
        {"symbol": s, "path": p, "sha256": h} for s, p, h in spec.paw_datasets]
    software = dict(spec.software)
    versions = reported.get("versions") or {}
    return Provenance(
        model=MODEL,
        fidelity=Fidelity.TIER3_EXTERNAL,
        origin=origin,
        approximations=_approximations(spec),
        tolerances={
            "energy_eV_per_electron": spec.energy_tol_eV_per_electron,
            "density_electrons_per_electron": spec.density_tol_electrons_per_electron,
            "eigenstates_eV2_per_electron": spec.eigenstates_tol_eV2_per_electron,
            "forces_eV_A": spec.forces_tol_eV_A,
            "max_scf_iterations": spec.max_scf_iterations,
        },
        boundary_conditions=_boundary_text(spec),
        parameters={
            "experiment_spec": spec.as_dict(),
            "spec_digest": spec.digest,
            "gpaw_parameters": specs.gpaw_parameters(spec),
            "gpaw_parameters_used": reported.get("parameters_used"),
            "software_pinned": software,
            "software_reported": versions,
            "paw_datasets": datasets,
            "grid": {"coarse": reported.get("coarse_grid"),
                     "fine": reported.get("fine_grid")},
            "n_bands": reported.get("n_bands"),
            "n_spins": reported.get("n_spins"),
            "n_ibz_kpoints": len(reported.get("ibz_kpoints") or []) or None,
            "n_bz_kpoints": reported.get("n_bz_kpoints"),
            "symmetry_operations": reported.get("symmetry_operations"),
            "scf_iterations": run.iterations if run is not None else 0,
            "status": run.status if run is not None else "",
            "wall_time_s": run.wall_time_s if run is not None else 0.0,
            "restart": restart,
            "warnings": list(warnings),
            "units": {k: v for k, v in spec.units().items()},
            "classification": CLASSIFICATION,
            "materia_version": __version__,
        },
        references=list(REFERENCES),
        dataset="; ".join(f"{d.get('symbol')}:{os.path.basename(str(d.get('path', '')))}"
                          f"@{str(d.get('sha256', ''))[:12]}" for d in datasets),
        dataset_license=("PAW datasets distributed with GPAW (gpaw-data) under their own "
                         "licence; not redistributed by Materia."),
        inputs_digest=spec.digest,
        notes=("Computed by GPAW " + str(versions.get("gpaw", software.get("gpaw", "")))
               + " in a separate process. Tier 3 external first-principles result: a "
               "density-functional approximation, not an exact solution. Classified as "
               "a computational prediction."),
    )


def _convergence(spec: specs.GroundStateSpec, run: runner.GPAWRun) -> Convergence:
    history = run.result.get("scf_history") or run.events or []
    last = history[-1] if history else {}
    residual = last.get("energy_change_eV_per_electron")
    if run.status == STATUS_CONVERGED:
        message = (f"Self-consistent after {run.iterations} iterations: energy, density "
                   "and eigenstate criteria all met.")
    elif run.status == STATUS_NOT_CONVERGED:
        message = (f"Stopped after {run.iterations} of at most {spec.max_scf_iterations} "
                   "iterations without meeting every criterion.")
    elif run.status == STATUS_CANCELLED:
        message = f"Cancelled after {run.iterations} iterations."
    else:
        message = f"Failed after {run.iterations} iterations."
    return Convergence(
        converged=run.status == STATUS_CONVERGED, iterations=int(run.iterations),
        residual=None if residual is None else float(residual),
        residual_metric="largest energy change over the last 3 iterations, eV/electron",
        tolerance=spec.energy_tol_eV_per_electron, message=message)


def _refusal(spec: specs.GroundStateSpec, run: runner.GPAWRun) -> str:
    if run.status == STATUS_NOT_CONVERGED:
        detail = (run.result.get("message") or "").strip().replace("\n", " ")
        return ("The SCF cycle did not reach self-consistency within "
                f"{spec.max_scf_iterations} iterations, so there is no ground state to "
                "report: no energy, density or other value is kept. Raise the iteration "
                "limit, use smeared occupations for a metal, or check the geometry and "
                "spin state." + (f" GPAW said: {detail}" if detail else ""))
    if run.status == STATUS_CANCELLED:
        return (f"The calculation was cancelled after {run.iterations} SCF iterations. "
                "No value is kept.")
    return (f"The GPAW calculation failed after {run.iterations} SCF iterations: "
            f"{run.error or 'no error message was produced'}. No value is kept.")


def execute(spec: specs.GroundStateSpec, environment=None, *,
            run_id: Optional[str] = None,
            progress: Optional[Callable[[dict], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None,
            restart_store: Optional[RestartStore] = None,
            reuse_restart: bool = False, keep_restart: bool = False,
            study_id: Optional[str] = None) -> GroundStateOutcome:
    """Run one specification.  Raises :class:`~.spec.SpecRefused` if it may not run."""
    environment = specs._environment(environment)
    report = specs.require(spec, environment)
    run_id = run_id or new_run_id()
    started = time.perf_counter()
    warnings: List[str] = list(report.warnings)

    key = specs.restart_key(spec)
    restart = {"key": key, "reused_from": None, "loaded": False, "kept": False,
               "reason": "restart reuse was not requested"}
    job: Dict[str, Any] = {
        "structure": specs.worker_structure(spec),
        "parameters": specs.gpaw_parameters(spec),
        "observables": list(spec.observables),
        "expected_datasets": {s: {"path": p, "sha256": h} for s, p, h in spec.paw_datasets},
    }
    store = restart_store
    if (reuse_restart or keep_restart) and store is None:
        store = RestartStore()
    if reuse_restart and store is not None:
        entry = store.find(key)
        if entry is not None:
            job["restart_in"] = entry["path"]
            restart["reused_from"] = entry.get("run_id")
            restart["reason"] = ("a converged state with the same atoms, discretisation, "
                                 "electron count and spin setup was reused as the "
                                 "starting point")
        else:
            restart["reason"] = "no compatible restart data was stored"
    if keep_restart and store is not None:
        job["restart_out"] = store.path_for(key, run_id)

    run = runner.run_job(job, environment, worker=WORKER, progress=progress,
                         cancelled=cancelled, timeout_s=timeout_s)
    reported_restart = run.result.get("restart") or {}
    restart["loaded"] = bool(reported_restart.get("loaded"))
    if reported_restart.get("error"):
        restart["reason"] = ("the stored state could not be loaded, so the calculation "
                             f"started fresh: {reported_restart['error']}")
        restart["reused_from"] = None
    if run.status == STATUS_CONVERGED and reported_restart.get("written") and store:
        entry = store.register(key, run_id, reported_restart["written"], spec.digest)
        restart["kept"] = entry is not None
    elif job.get("restart_out") and os.path.exists(job["restart_out"]) \
            and run.status != STATUS_CONVERGED:
        try:
            os.remove(job["restart_out"])
        except OSError:
            pass

    if run.status == STATUS_CONVERGED:
        mismatch = parameter_mismatch(job["parameters"], run.result.get("parameters_used"))
        if mismatch:
            run.status = STATUS_FAILED
            run.error = ("GPAW did not run with the parameters Materia sent, so the result "
                         f"would not describe this specification: {mismatch}")
    software_warnings = [str(w) for w in (run.result.get("warnings") or [])]
    outcome = GroundStateOutcome(run_id=run_id, spec=spec, status=run.status,
                                 warnings=warnings,
                                 wall_time_s=time.perf_counter() - started)
    convergence = _convergence(spec, run)
    history = run.result.get("scf_history") or run.events or []
    base_extra = {
        "run_id": run_id, "structure_key": spec.structure_key,
        "geometry_digest": spec.geometry_digest, "spec_digest": spec.digest,
        "mass_numbers": list(spec.mass_numbers), "status": run.status,
        "classification": CLASSIFICATION, "warnings": list(warnings),
        "software_warnings": software_warnings, "study_id": study_id, "n_atoms": spec.n_atoms, "formula": spec.formula,
        "boundary": spec.boundary,
    }

    if run.status != STATUS_CONVERGED:
        reason = _refusal(spec, run)
        record = unsupported("dft_run", MODEL, reason, fidelity=Fidelity.TIER3_EXTERNAL)
        record.provenance = provenance(spec, run, Origin.UNSUPPORTED, warnings, restart)
        record.convergence = convergence
        record.extra.update(base_extra)
        record.extra.update({
            "scf_history": _clean_history(history),
            "iterations": run.iterations, "wall_time_s": run.wall_time_s,
            "gpaw_error": run.error, "gpaw_message": run.result.get("message", ""),
            "gpaw_log_tail": (run.result.get("log_tail") or "")[-3000:],
            "stderr_tail": run.stderr[-2000:], "observables": [], "arrays": {},
        })
        outcome.results["run"] = record
        outcome.reason = reason
        return outcome

    collect(outcome, spec, run, report, restart, base_extra, convergence, history)
    return outcome


def collect(outcome: GroundStateOutcome, spec: specs.GroundStateSpec, run: runner.GPAWRun,
            report, restart: dict, base_extra: Dict[str, Any], convergence: Convergence,
            history: List[dict]) -> None:
    """Fill ``outcome`` with the results of a converged self-consistent field.

    ``spec`` describes the geometry the values belong to.  A relaxation calls
    this with the specification of its final geometry, so the observables it
    stores follow exactly the same contract as a ground-state run.
    """
    warnings = outcome.warnings
    result = run.result
    prov = provenance(spec, run, Origin.CALCULATED, warnings, restart)

    def make(name: str, value: Any, unit: str, **extra: Any) -> Result:
        item = Result(name, value, unit, _copy_provenance(prov), convergence=convergence)
        item.extra.update(base_extra)
        item.extra.update(extra)
        return item

    ids = list(spec.atom_ids)
    energy_free = float(result["energy_free_eV"])
    outcome.results["energy"] = make(
        "total_energy", energy_free, "eV",
        free_energy_eV=energy_free,
        extrapolated_energy_eV=float(result["energy_extrapolated_eV"]),
        energy_per_atom_eV=energy_free / max(1, spec.n_atoms),
        reference_energy_eV=result.get("reference_energy_eV"),
        contributions_eV=result.get("energy_contributions_eV"),
        note=("Free energy E - TS, the quantity forces and stress are derivatives of. "
              "Identical to the extrapolated energy for integer occupations. Relative "
              "to the datasets' reference atoms."))
    if "forces_eV_per_A" in result:
        forces = np.asarray(result["forces_eV_per_A"], dtype=float)
        outcome.results["forces"] = make(
            "forces", forces, "eV/A",
            by_atom_id={int(i): [float(v) for v in row] for i, row in zip(ids, forces)},
            max_force_eV_A=float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0,
            net_force_eV_A=[float(v) for v in forces.sum(axis=0)],
            atom_ids=ids)
    if "stress_eV_per_A3" in result:
        stress = np.asarray(result["stress_eV_per_A3"], dtype=float)
        outcome.results["stress"] = make(
            "stress", stress, "eV/A^3",
            pressure_GPa=float(-np.trace(stress) / 3.0 * EV_PER_A3_TO_GPA),
            stress_GPa=(stress * EV_PER_A3_TO_GPA).tolist(),
            sign_convention="ASE: positive stress is tensile; pressure is -trace/3.")
    if "fermi_level" in spec.observables:
        outcome.results["fermi_level"] = make(
            "fermi_level", result.get("fermi_level_eV"), "eV",
            homo_eV=result.get("homo_eV"), lumo_eV=result.get("lumo_eV"),
            note=specs.OBSERVABLE_NOTES["fermi_level"])
    if "magnetic_moment" in spec.observables and "magnetic_moment_muB" in result:
        local = result.get("local_magnetic_moments_muB")
        outcome.results["magnetic_moment"] = make(
            "magnetic_moment", float(result["magnetic_moment_muB"]), "mu_B",
            local_moments_by_atom_id=(None if local is None else
                                      {int(i): float(m) for i, m in zip(ids, local)}),
            note=("Total moment of the cell. Local moments are integrated inside atomic "
                  "spheres and do not add up to the total; they are indicative only."))

    kpoints = result.get("ibz_kpoints") or []
    weights = result.get("kpoint_weights") or []
    for name in ("eigenvalues", "occupations"):
        if name in run.arrays:
            data = np.asarray(run.arrays[name], dtype=float)
            key = f"{name}"
            outcome.arrays[key] = StoredArray(
                data=data, unit=UNITS[name], kind="table",
                description=("Kohn-Sham eigenvalues" if name == "eigenvalues" else
                             "Occupation of each state, 0 to 1 per spin channel")
                + " indexed [spin, IBZ k-point, band].",
                meta={"axes": ["spin", "kpoint", "band"], "ibz_kpoints": kpoints,
                      "kpoint_weights": weights})
            summary = {"array": key, "shape": list(data.shape)}
            extra: Dict[str, Any] = {"ibz_kpoints": kpoints, "kpoint_weights": weights}
            if name == "eigenvalues" and "occupations" in run.arrays:
                extra["band_edges"] = _band_edges(data, np.asarray(run.arrays["occupations"]))
            outcome.results[name] = make(name, summary, UNITS[name], **extra)

    cell = np.asarray(spec.cell_A, dtype=float)
    for name in ("density", "spin_density", "electrostatic_potential"):
        if name not in run.arrays:
            continue
        data = np.asarray(run.arrays[name], dtype=float)
        volume = abs(float(np.linalg.det(cell)))
        dv = volume / float(np.prod(data.shape))
        meta = {"grid_shape": list(data.shape), "cell_A": cell.tolist(),
                "pbc": list(spec.pbc), "origin_A": [0.0, 0.0, 0.0],
                "convention": "point (i, j, k) sits at fractional coordinates "
                              "(i/N1, j/N2, k/N3); the end point of each axis is not "
                              "repeated",
                "grid": "fine", "voxel_volume_A3": dv}
        outcome.arrays[name] = StoredArray(data=data, unit=UNITS[name], kind="volumetric",
                                           description=VOLUME_DESCRIPTIONS[name], meta=meta)
        summary = {"array": name, "shape": list(data.shape),
                   "min": float(data.min()), "max": float(data.max())}
        if name != "electrostatic_potential":
            summary["integral"] = float(data.sum() * dv)
        outcome.results[name] = make(name, summary, UNITS[name], grid=meta)

    accounting = _accounting(spec, result, report)
    outcome.results["charge_accounting"] = make(
        "charge_accounting", accounting, "e",
        note="Electron bookkeeping from nuclear charge, frozen cores, net charge and the "
             "integral of the all-electron density.")
    if not accounting["balanced"]:
        outcome.warnings.append(accounting["message"])
    outcome.results["scf_history"] = make("scf_history", _clean_history(history), "",
                                          columns=list(HISTORY_COLUMNS))
    record = make("dft_run", {
        "status": run.status, "spec_digest": spec.digest,
        "observables": sorted(k for k in outcome.results if k not in ("run",)),
        "arrays": {k: v.summary() for k, v in outcome.arrays.items()},
    }, "")
    record.extra.update({"iterations": run.iterations, "wall_time_s": run.wall_time_s,
                         "gpaw_log_tail": (result.get("log_tail") or "")[-3000:],
                         "warnings": list(outcome.warnings)})
    outcome.results["run"] = record
    for item in outcome.results.values():
        item.extra["warnings"] = list(outcome.warnings)
        item.provenance.parameters["warnings"] = list(outcome.warnings)


def parameter_mismatch(sent: Dict[str, Any], used: Optional[Dict[str, Any]],
                       path: str = "") -> str:
    """The first parameter GPAW reports differently from what was sent, or ''.

    GPAW echoes its parameters after a run.  Every value Materia sent must come
    back unchanged: that is the check that no variable was accepted and then
    ignored, including after a restart.
    """
    if used is None:
        return "GPAW reported no parameters"
    for name, value in sent.items():
        where = f"{path}{name}"
        if name not in used:
            return f"{where} is missing from the parameters GPAW used"
        other = used[name]
        if isinstance(value, dict):
            if not isinstance(other, dict):
                return f"{where} was {other!r}, not {value!r}"
            inner = parameter_mismatch(value, other, where + ".")
            if inner:
                return inner
        elif not _same(value, other):
            return f"{where} was {other!r}, not {value!r}"
    return ""


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)))
    return a == b


HISTORY_COLUMNS = ("iteration", "time_s", "energy_eV", "free_energy_eV",
                   "energy_change_eV_per_electron", "density_error_electrons_per_electron",
                   "eigenstate_error_eV2_per_electron", "magnetic_moment_muB", "converged")


def _clean_history(rows: List[dict]) -> List[dict]:
    out = []
    for row in rows:
        out.append({key: row.get(key) for key in HISTORY_COLUMNS}
                   | {"criteria_met": dict(row.get("criteria_met") or {})})
    return out


def _band_edges(eigenvalues: np.ndarray, occupations: np.ndarray) -> Dict[str, Any]:
    occupied = eigenvalues[occupations > 0.5]
    empty = eigenvalues[occupations <= 0.5]
    out: Dict[str, Any] = {
        "highest_occupied_eV": float(occupied.max()) if occupied.size else None,
        "lowest_unoccupied_eV": float(empty.min()) if empty.size else None,
    }
    if occupied.size and empty.size:
        out["gap_eV"] = max(0.0, float(empty.min() - occupied.max()))
        out["note"] = ("Kohn-Sham gap from the computed states, occupation above or below "
                       "one half. Semilocal functionals underestimate fundamental gaps; "
                       "on a coarse k-grid the gap is sampled, not located.")
    return out


def _accounting(spec: specs.GroundStateSpec, result: dict, report) -> Dict[str, Any]:
    datasets = result.get("datasets") or []
    by_symbol = {d["symbol"]: d for d in datasets}
    symbols = spec.symbols
    nuclear = float(sum(spec.numbers))
    core = float(sum(by_symbol.get(s, {}).get("core_electrons", 0.0) for s in symbols))
    valence_neutral = float(sum(by_symbol.get(s, {}).get("valence_electrons", 0.0)
                                for s in symbols))
    charge = float(spec.charge_e)
    expected_valence = valence_neutral - charge
    expected_total = nuclear - charge
    reported = result.get("accounting") or {}
    integral = reported.get("density_integral_e")
    gpaw_valence = reported.get("gpaw_valence_electrons")
    occupied = reported.get("occupied_electrons")
    out: Dict[str, Any] = {
        "nuclear_charge_e": nuclear,
        "frozen_core_electrons": core,
        "net_charge_e": charge,
        "expected_valence_electrons": expected_valence,
        "expected_total_electrons": expected_total,
        "gpaw_valence_electrons": gpaw_valence,
        "occupied_electrons": occupied,
        "density_integral_e": integral,
        "density_integral_error_e": (None if integral is None
                                     else float(integral) - expected_total),
        "background_charge_e": charge if (spec.boundary == "bulk" and abs(charge) > 1e-9)
        else 0.0,
        "density_grid": reported.get("density_grid"),
    }
    if spec.spin_polarized:
        out.update({"spin_up_e": reported.get("spin_up_integral_e"),
                    "spin_down_e": reported.get("spin_down_integral_e"),
                    "magnetisation_integral_muB": reported.get(
                        "magnetisation_integral_muB")})
    problems = []
    if gpaw_valence is None or abs(float(gpaw_valence) - expected_valence) > 1e-6:
        problems.append(f"GPAW counted {gpaw_valence} valence electrons, the datasets and "
                        f"net charge give {expected_valence:g}")
    if occupied is not None and abs(float(occupied) - expected_valence) > 1e-4:
        problems.append(f"the occupied states hold {float(occupied):.6f} electrons, not "
                        f"{expected_valence:g}")
    if integral is not None and abs(float(integral) - expected_total) > 1e-3 * max(
            1.0, expected_total / 10.0):
        problems.append(f"the all-electron density integrates to {float(integral):.5f} "
                        f"electrons, not {expected_total:g}; the fine grid is too coarse "
                        "to integrate the core region accurately")
    out["balanced"] = not problems
    out["message"] = ("Electron count balanced: nuclear charge, frozen cores, net charge, "
                      "occupied states and density integral agree." if not problems else
                      "Electron accounting does not balance: " + "; ".join(problems) + ".")
    return out


def _copy_provenance(prov: Provenance) -> Provenance:
    import copy

    return copy.deepcopy(prov)
