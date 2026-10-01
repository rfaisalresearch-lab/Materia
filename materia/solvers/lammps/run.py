"""Executing a frozen LAMMPS specification and turning its output into results.

:func:`execute` is the whole pipeline: check the specification against the
LAMMPS found now, write the native inputs into a private run directory, run
LAMMPS out of process, and read back what it wrote.  Every check below has to
pass before a single :class:`~materia.provenance.Result` is built, so an
outcome is either complete or carries no values at all:

* the process exited with status 0, printed no ``ERROR`` line and reached the
  final marker of the input script;
* the log header names the LAMMPS version the specification was frozen for;
* every thermodynamic row and every dumped value is finite;
* the atom count never changed, and the dumped identifiers, types and masses
  are exactly the ones written;
* the steps printed are the steps requested, and the trajectory has one frame
  per stored step;
* the potential energy of the final evaluation equals the last energy of the
  run, and the kinetic energy recomputed from the dumped velocities and masses
  equals the one LAMMPS printed.  The second check exercises the unit
  conversion of velocities and masses on every run.
"""

from __future__ import annotations

import os
import shlex
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ...project_format.arrays import StoredArray
from ...provenance import Convergence, Fidelity, Origin, Provenance, Result
from . import native, units
from .environment import discover
from .parse import DumpFrame, OutputError, ParsedLog, parse_dump, parse_log
from .runner import STATUS_CANCELLED, STATUS_NOT_STARTED, STATUS_TIMEOUT, run_process
from .spec import LAMMPSRunSpec, SpecRefused, check

PREFIX = "lammps::"
MODEL = "external:lammps"

STATUS_COMPLETE = "complete"
STATUS_CONVERGED = "converged"
STATUS_NOT_CONVERGED = "not-converged"
STATUS_FAILED = "failed"
STATUS_RUN_CANCELLED = "cancelled"
SUCCESS = (STATUS_COMPLETE, STATUS_CONVERGED, STATUS_NOT_CONVERGED)

LOG_TAIL_LINES = 200
ENERGY_RELATIVE_TOLERANCE = 1e-9
KINETIC_RELATIVE_TOLERANCE = 1e-6
POSITION_ROUND_TRIP_A = 1e-6

REFERENCE = "A. P. Thompson et al., LAMMPS, Comp. Phys. Comm. 271 (2022) 108171"


def key(run_id: str, quantity: str) -> str:
    return f"{PREFIX}{run_id}::{quantity}"


def new_run_id() -> str:
    return f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:6]}"


@dataclass
class LAMMPSOutcome:
    run_id: str
    spec: LAMMPSRunSpec
    status: str
    reason: str = ""
    results: Dict[str, Result] = field(default_factory=dict)
    arrays: Dict[str, StoredArray] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    output_state_digest: str = ""

    @property
    def ok(self) -> bool:
        return self.status in SUCCESS

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "status": self.status, "ok": self.ok,
                "reason": self.reason, "spec_digest": self.spec.digest,
                "audit": dict(self.audit)}


class _Failure(Exception):
    pass


def execute(spec: LAMMPSRunSpec, *, environment=None, run_id: Optional[str] = None,
            progress: Optional[Callable[[float, str], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            timeout_s: Optional[float] = None, keep_files: bool = False,
            workdir_root: Optional[str] = None) -> LAMMPSOutcome:
    """Run ``spec`` and return what happened.

    Raises :class:`~materia.solvers.lammps.spec.SpecRefused` before anything is
    started if the specification must not run.  Otherwise returns an outcome
    whose status is ``complete``, ``converged``, ``not-converged``, ``failed``
    or ``cancelled``; only the first three carry results.
    """
    environment = environment if environment is not None else discover(refresh=True)
    report = check(spec, environment)
    if not report.ok:
        raise SpecRefused(report)
    run_id = run_id or new_run_id()
    audit: Dict[str, Any] = {"run_id": run_id, "spec_digest": spec.digest,
                             "lammps": dict(spec.lammps), "task": spec.task,
                             "warnings_from_check": report.warnings}
    if cancelled is not None and cancelled():
        return LAMMPSOutcome(run_id, spec, STATUS_RUN_CANCELLED,
                             "Cancelled before LAMMPS was started.", audit=audit)
    root = workdir_root or os.environ.get("MATERIA_LAMMPS_RUN_DIR") or None
    if root:
        os.makedirs(root, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix=f"materia-lammps-{run_id}-", dir=root)
    audit["workdir"] = workdir if keep_files else ""
    try:
        try:
            inputs = native.write_inputs(spec, workdir)
        except OSError as exc:
            return LAMMPSOutcome(run_id, spec, STATUS_FAILED,
                                 f"The run directory could not be prepared: {exc}",
                                 audit=audit)
        command = environment.command(native.INPUT_FILE, native.LOG_FILE)
        audit.update({"command": command, "command_line": shlex.join(command),
                      "inputs": inputs,
                      "input_script": (Path(workdir) / native.INPUT_FILE).read_text()})
        if progress is not None:
            progress(0.01, f"Starting LAMMPS {environment.version}")
        total = {"energy": 1, "relax": spec.max_iterations or 1,
                 "md": spec.steps or 1}[spec.task]
        process = run_process(command, workdir, total_steps=total, progress=progress,
                              cancelled=cancelled, timeout_s=timeout_s)
        audit["process"] = process.as_dict()
        audit["wall_time_s"] = process.wall_time_s
        log_path = Path(workdir) / native.LOG_FILE
        log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        audit["log_tail"] = "\n".join(log_text.splitlines()[-LOG_TAIL_LINES:])
        audit["log_lines"] = len(log_text.splitlines())
        if process.status == STATUS_CANCELLED:
            return LAMMPSOutcome(run_id, spec, STATUS_RUN_CANCELLED,
                                 "The LAMMPS run was cancelled. Nothing was kept and the "
                                 "project was not changed.", audit=audit)
        if process.status in (STATUS_TIMEOUT, STATUS_NOT_STARTED):
            reason = (process.error if process.status == STATUS_NOT_STARTED else
                      f"LAMMPS was stopped after the {timeout_s:g} s time limit.")
            return LAMMPSOutcome(run_id, spec, STATUS_FAILED, reason, audit=audit)
        try:
            outcome = _interpret(spec, run_id, workdir, log_text, process, audit)
        except (OutputError, _Failure) as exc:
            return LAMMPSOutcome(run_id, spec, STATUS_FAILED, str(exc), audit=audit)
        if cancelled is not None and cancelled():
            return LAMMPSOutcome(run_id, spec, STATUS_RUN_CANCELLED,
                                 "The LAMMPS run was cancelled as it finished. Nothing was "
                                 "kept and the project was not changed.", audit=audit)
        return outcome
    finally:
        if not keep_files:
            shutil.rmtree(workdir, ignore_errors=True)


def _first_error(parsed: Optional[ParsedLog], process) -> str:
    if parsed is not None and parsed.errors:
        return parsed.errors[0]
    for line in reversed(process.stderr_tail):
        if line.strip():
            return line.strip()[:500]
    for line in reversed(process.stdout_tail):
        if line.strip().startswith("ERROR"):
            return line.strip()[:500]
    return ""


def _finite(array: np.ndarray, what: str) -> None:
    if not np.all(np.isfinite(array)):
        raise _Failure(f"LAMMPS reported a value that is not finite in {what}; the run is "
                       "not used.")


def _interpret(spec: LAMMPSRunSpec, run_id: str, workdir: str, log_text: str, process,
               audit: Dict[str, Any]) -> LAMMPSOutcome:
    parsed: Optional[ParsedLog] = None
    try:
        parsed = parse_log(log_text) if log_text else None
    except OutputError as exc:
        if process.returncode == 0:
            raise
        audit["parse_error"] = str(exc)
    if process.returncode != 0:
        detail = _first_error(parsed, process)
        raise _Failure(f"LAMMPS exited with status {process.returncode}."
                       + (f" {detail}" if detail else ""))
    if parsed is None:
        raise _Failure("LAMMPS exited without writing a log file.")
    audit["log_version"] = parsed.version
    audit["log_warnings"] = parsed.warnings[:50]
    audit["total_wall_time"] = parsed.total_wall_time
    if parsed.errors:
        raise _Failure(f"LAMMPS reported an error: {parsed.errors[0]}")
    if parsed.version != spec.lammps.get("version"):
        raise _Failure(f"The log was written by LAMMPS {parsed.version or '(no header)'}, "
                       f"not the {spec.lammps.get('version')} this run was frozen for.")
    if not parsed.complete:
        raise _Failure("The LAMMPS log stops before the end of the input script; the run "
                       "is truncated and nothing from it is used.")
    needed = ["final"] if spec.task == "energy" else ["main", "final"]
    for name in needed:
        if name not in parsed.sections:
            raise _Failure(f"The LAMMPS log has no thermodynamic table for the {name} "
                           "section.")
    for name, table in parsed.sections.items():
        _finite(table.rows, f"the {name} thermodynamic table")
        atoms = table.column("atoms")
        if np.any(atoms != spec.n_atoms):
            raise _Failure(f"The atom count changed during the {name} section (from "
                           f"{spec.n_atoms} to {int(atoms.min())}); atoms were lost.")
    final_table = parsed.sections["final"]
    if len(final_table.rows) != 1:
        raise _Failure("The final evaluation printed more than one thermodynamic row.")
    final_path = Path(workdir) / native.FINAL_DUMP
    if not final_path.is_file():
        raise _Failure("LAMMPS did not write the final dump.")
    frames = parse_dump(final_path.read_text(), native.FINAL_COLUMNS)
    if len(frames) != 1:
        raise _Failure(f"The final dump holds {len(frames)} frames, not one.")
    final = frames[0]
    order = _check_atoms(spec, final, "the final dump")
    _finite(final.data, "the final dump")
    masses = final.column("mass")[order]
    expected_masses = spec.masses_amu()
    if not np.allclose(masses, expected_masses, rtol=1e-12, atol=0.0):
        worst = int(np.argmax(np.abs(masses - expected_masses)))
        raise _Failure(f"Atom {spec.atom_ids[worst]} has mass {masses[worst]!r} in LAMMPS, "
                       f"not the {expected_masses[worst]!r} u that was written.")
    positions = np.column_stack([final.column(c) for c in ("xu", "yu", "zu")])[order]
    velocities_ps = np.column_stack([final.column(c) for c in ("vx", "vy", "vz")])[order]
    forces = np.column_stack([final.column(c) for c in ("fx", "fy", "fz")])[order]
    final_row = final_table.rows[0]
    final_step = int(final_row[0])
    if final.timestep != final_step:
        raise _Failure(f"The final dump is at step {final.timestep}, the final evaluation "
                       f"at step {final_step}.")
    pe = float(final_table.column("pe")[0])
    ke = float(final_table.column("ke")[0])
    recomputed = units.kinetic_energy_eV(masses, velocities_ps)
    if abs(recomputed - ke) > KINETIC_RELATIVE_TOLERANCE * max(abs(ke), 1e-6) + 1e-10:
        raise _Failure(f"The kinetic energy from the dumped velocities and masses "
                       f"({recomputed:.12g} eV) does not match the {ke:.12g} eV LAMMPS "
                       "printed; the velocities or masses were not read consistently.")
    main = parsed.sections.get("main")
    if main is not None:
        last_pe = float(main.column("pe")[-1])
        if abs(last_pe - pe) > ENERGY_RELATIVE_TOLERANCE * max(1.0, abs(pe)):
            raise _Failure(f"The final evaluation gives {pe!r} eV but the run ended at "
                           f"{last_pe!r} eV at the same geometry.")
        if int(main.column("step")[-1]) != final_step:
            raise _Failure("The final evaluation is not at the last step of the run.")
    if spec.task == "energy":
        if final_step != 0:
            raise _Failure(f"An energy evaluation ended at step {final_step}, not 0.")
        moved = float(np.abs(positions - spec.positions()).max()) if spec.n_atoms else 0.0
        if moved > POSITION_ROUND_TRIP_A:
            raise _Failure(f"Positions changed by {moved:.3g} A in an energy evaluation; "
                           "the coordinates did not survive the round trip.")
        written = units.velocity_to_lammps(spec.velocities())
        if spec.n_atoms and float(np.abs(velocities_ps - written).max()) > \
                POSITION_ROUND_TRIP_A * max(1.0, float(np.abs(written).max())):
            raise _Failure("Velocities changed in an energy evaluation; they did not "
                           "survive the round trip.")
        positions = spec.positions()
        velocities_ps = written
    fixed = spec.fixed_mask()
    if spec.task != "energy" and fixed.any():
        moved = float(np.abs(positions[fixed] - spec.positions()[fixed]).max())
        if moved > POSITION_ROUND_TRIP_A:
            raise _Failure(f"A fixed atom moved by {moved:.3g} A.")
    velocities = units.velocity_from_lammps(velocities_ps)

    from ...python_api.api import structure_state_digest

    produced = spec.structure()
    produced.positions = positions
    output_digest = (spec.input_state_digest if spec.task == "energy"
                     else structure_state_digest(produced))
    builder = _Builder(spec, run_id, audit, parsed)
    status = STATUS_COMPLETE
    convergence = None
    mobile = ~spec.fixed_mask()
    max_force = float(np.linalg.norm(forces[mobile], axis=1).max()) if mobile.any() else 0.0
    if spec.task == "relax":
        status, convergence = _relaxation(spec, parsed, main, max_force)
        builder.relaxation(parsed, main, max_force, convergence, status)
    if spec.task == "md":
        trajectory = _trajectory(spec, workdir, main)
        builder.trajectory(main, trajectory)
    builder.energy(pe, final_table, convergence, status, max_force)
    builder.forces(forces, max_force)
    if "stress" in spec.outputs:
        builder.stress(final_table)
    if "thermo" in spec.outputs:
        builder.thermo(parsed)
    builder.final_state(positions, velocities)
    outcome = LAMMPSOutcome(run_id, spec, status, results=builder.results,
                            arrays=builder.arrays, audit=audit,
                            output_state_digest=output_digest)
    builder.annotate(outcome, max_force)
    return outcome


def _check_atoms(spec: LAMMPSRunSpec, frame: DumpFrame, what: str) -> np.ndarray:
    """The row of ``frame`` for each atom of ``spec``, after checking ids and types."""
    ids_float = frame.column("id")
    types_float = frame.column("type")
    if len(ids_float) != spec.n_atoms:
        raise _Failure(f"{what.capitalize()} holds {len(ids_float)} atoms; {spec.n_atoms} "
                       "were written. Atoms were lost or created.")
    if not (np.all(ids_float == np.round(ids_float)) and
            np.all(types_float == np.round(types_float))):
        raise _Failure(f"{what.capitalize()} has an atom id or type that is not an integer.")
    ids = ids_float.astype(np.int64)
    types = types_float.astype(np.int64)
    expected = np.array(spec.atom_ids, dtype=np.int64)
    if len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != set(expected.tolist()):
        extra = sorted(set(ids.tolist()) - set(expected.tolist()))[:5]
        gone = sorted(set(expected.tolist()) - set(ids.tolist()))[:5]
        raise _Failure(f"{what.capitalize()} does not carry the atom identifiers that were "
                       f"written (unexpected {extra}, missing {gone}).")
    position = {int(i): k for k, i in enumerate(ids)}
    order = np.array([position[int(i)] for i in expected], dtype=np.int64)
    written = np.array(spec.atom_types, dtype=np.int64)
    if np.any(types[order] != written):
        bad = int(np.flatnonzero(types[order] != written)[0])
        raise _Failure(f"Atom {spec.atom_ids[bad]} has type {int(types[order][bad])} in "
                       f"{what}, but type {int(written[bad])} was written.")
    return order


def _relaxation(spec: LAMMPSRunSpec, parsed: ParsedLog, main, max_force: float):
    stats = parsed.minimization
    if not stats or "stopping_criterion" not in stats or "iterations" not in stats:
        raise _Failure("The log has no complete Minimization stats block.")
    if int(main.column("step")[0]) != 0:
        raise _Failure("The minimisation table does not start at step 0.")
    steps = main.column("step")
    if np.any(np.diff(steps) <= 0):
        raise _Failure("The minimisation steps are not increasing.")
    if int(steps[-1]) != int(stats["iterations"]):
        raise _Failure(f"The minimisation table ends at step {int(steps[-1])} but LAMMPS "
                       f"reports {stats['iterations']} iterations.")
    converged = max_force <= float(spec.ftol_eV_A)
    criterion = str(stats["stopping_criterion"])
    message = (f"LAMMPS stopped on '{criterion}' after {stats['iterations']} iterations; the "
               f"largest force on a mobile atom is {max_force:.3e} eV/A against a tolerance "
               f"of {spec.ftol_eV_A:g} eV/A.")
    if not converged:
        message += " The force tolerance was not met, so this is not an energy minimum."
    convergence = Convergence(converged=converged, iterations=int(stats["iterations"]),
                              residual=max_force, residual_metric="max |F| on mobile atoms, eV/A",
                              tolerance=float(spec.ftol_eV_A), message=message)
    return (STATUS_CONVERGED if converged else STATUS_NOT_CONVERGED), convergence


def _trajectory(spec: LAMMPSRunSpec, workdir: str, main) -> Dict[str, np.ndarray]:
    every = int(spec.sample_every)
    expected_steps = np.arange(0, int(spec.steps) + 1, every)
    printed = main.column("step").astype(np.int64)
    if len(printed) != len(expected_steps) or np.any(printed != expected_steps):
        raise _Failure(f"The run printed steps {printed[:3].tolist()}...{printed[-2:].tolist()} "
                       f"({len(printed)} rows); {len(expected_steps)} rows every {every} steps "
                       f"to {spec.steps} were requested.")
    if "trajectory" not in spec.outputs:
        return {}
    path = Path(workdir) / native.TRAJECTORY_DUMP
    if not path.is_file():
        raise _Failure("LAMMPS did not write the trajectory dump.")
    frames = parse_dump(path.read_text(), native.TRAJECTORY_COLUMNS)
    if len(frames) != len(expected_steps):
        raise _Failure(f"The trajectory holds {len(frames)} frames; {len(expected_steps)} "
                       "were requested. It is truncated.")
    positions = np.empty((len(frames), spec.n_atoms, 3))
    velocities = np.empty((len(frames), spec.n_atoms, 3))
    for index, (frame, step) in enumerate(zip(frames, expected_steps)):
        if frame.timestep != int(step):
            raise _Failure(f"Trajectory frame {index} is at step {frame.timestep}, not "
                           f"{int(step)}.")
        order = _check_atoms(spec, frame, f"trajectory frame {index}")
        _finite(frame.data, f"trajectory frame {index}")
        positions[index] = np.column_stack(
            [frame.column(c) for c in ("xu", "yu", "zu")])[order]
        velocities[index] = units.velocity_from_lammps(np.column_stack(
            [frame.column(c) for c in ("vx", "vy", "vz")])[order])
    return {"positions": positions, "velocities": velocities, "steps": expected_steps}


def _series(table) -> Dict[str, List[float]]:
    return {
        "step": [int(v) for v in table.column("step")],
        "time_fs": [float(v) for v in units.ps_to_fs(table.column("time"))],
        "potential_eV": [float(v) for v in table.column("pe")],
        "kinetic_eV": [float(v) for v in table.column("ke")],
        "total_eV": [float(v) for v in table.column("etotal")],
        "temperature_K": [float(v) for v in table.column("temp")],
        "pressure_bar": [float(v) for v in table.column("press")],
    }


class _Builder:
    """Results for one validated run.  Nothing here can fail on the output."""

    def __init__(self, spec: LAMMPSRunSpec, run_id: str, audit: Dict[str, Any],
                 parsed: ParsedLog) -> None:
        self.spec = spec
        self.run_id = run_id
        self.audit = audit
        self.parsed = parsed
        self.results: Dict[str, Result] = {}
        self.arrays: Dict[str, StoredArray] = {}

    def provenance(self, origin: Origin = Origin.CALCULATED) -> Provenance:
        spec = self.spec
        potential = spec.potential
        approximations = [
            f"Computed by LAMMPS {spec.lammps.get('version')} with pair_style "
            f"{potential.get('style')} and the parameter file {potential.get('file_name')} "
            f"(SHA-256 {potential.get('sha256')}). Materia wrote the input and read the "
            "output; the numbers are LAMMPS's.",
            "Embedded-atom method: a classical many-body potential with no explicit "
            "electrons, no angular terms and no magnetism.",
            "LAMMPS metal units converted to Materia units: time ps to fs and velocity "
            "A/ps to A/fs by exact factors of 1000; pressure bar to stress eV/A^3 with the "
            "nktv2p constant LAMMPS itself applies.",
        ]
        if spec.task in ("relax", "md"):
            approximations.append("The cell is held fixed; no barostat is applied.")
        if spec.task == "md" and spec.ensemble == "langevin":
            approximations.append("Langevin thermostat: stochastic, reproducible only with "
                                  "the same seed, LAMMPS build and processor count.")
        if not all(spec.pbc):
            approximations.append("Non-periodic directions use shrink-wrapped boundaries.")
        return Provenance(
            model=f"{MODEL}/{potential.get('style')}/{potential.get('id')}",
            fidelity=Fidelity.TIER1_CLASSICAL, origin=origin,
            approximations=approximations,
            boundary_conditions=f"boundary {' '.join(spec.boundary)}",
            parameters={"run_spec_digest": spec.digest, "potential": dict(potential),
                        "lammps": dict(spec.lammps), "task_settings": spec.settings(),
                        "units": "LAMMPS metal, converted to eV, A, fs, u, K"},
            references=[REFERENCE] + list(potential.get("citations") or []),
            seed=spec.seed if spec.task == "md" else None,
            inputs_digest=spec.input_state_digest)

    def _origin(self, status: str) -> Origin:
        return Origin.ESTIMATED if status == STATUS_NOT_CONVERGED else Origin.CALCULATED

    def energy(self, pe: float, final_table, convergence, status: str,
               max_force: float) -> None:
        result = Result("lammps_energy", float(pe), "eV", self.provenance(self._origin(status)),
                        convergence=convergence)
        self.results["energy"] = result

    def forces(self, forces: np.ndarray, max_force: float) -> None:
        array_key = key(self.run_id, "forces")
        self.arrays["forces"] = StoredArray(
            np.ascontiguousarray(forces, dtype=np.float64), "eV/A",
            "Forces on every atom at the final geometry, from LAMMPS", "per-atom",
            {"atom_ids": list(self.spec.atom_ids)})
        self.results["forces"] = Result("lammps_forces", np.array(forces), "eV/A",
                                        self.provenance(),
                                        extra={"array": array_key,
                                               "max_force_eV_A": max_force})

    def stress(self, final_table) -> None:
        values = [float(final_table.column(k)[0]) for k in
                  ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz")]
        pressure = units.pressure_voigt_to_tensor(*values)
        stress = units.pressure_bar_to_stress_eV_A3(pressure)
        self.results["stress"] = Result(
            "lammps_stress", stress.tolist(), "eV/A^3", self.provenance(),
            extra={"pressure_tensor_bar": pressure.tolist(),
                   "pressure_bar": float(final_table.column("press")[0]),
                   "convention": "stress = -P, the LAMMPS pressure tensor including the "
                                 "kinetic contribution of the atom velocities"})

    def thermo(self, parsed: ParsedLog) -> None:
        value = {name: _series(table) for name, table in parsed.sections.items()}
        self.results["thermo"] = Result("lammps_thermo", value, "", self.provenance(),
                                        extra={"units": {
                                            "time_fs": "fs", "potential_eV": "eV",
                                            "kinetic_eV": "eV", "total_eV": "eV",
                                            "temperature_K": "K", "pressure_bar": "bar"}})

    def relaxation(self, parsed: ParsedLog, main, max_force: float, convergence,
                   status: str) -> None:
        stats = dict(parsed.minimization or {})
        stats.update({"max_force_eV_A": max_force, "ftol_eV_A": self.spec.ftol_eV_A,
                      "min_style": self.spec.min_style, "status": status})
        self.results["relaxation"] = Result("lammps_relaxation", stats, "",
                                            self.provenance(self._origin(status)),
                                            convergence=convergence)

    def trajectory(self, main, trajectory: Dict[str, np.ndarray]) -> None:
        spec = self.spec
        series = _series(main)
        temperatures = np.array(series["temperature_K"])
        if len(temperatures) > 1:
            sample = temperatures[1:]
            self.results["mean_temperature"] = Result(
                "lammps_mean_temperature", float(sample.mean()), "K", self.provenance(),
                uncertainty=float(sample.std()), uncertainty_kind="stddev")
        if not trajectory:
            return
        drift = float(series["total_eV"][-1] - series["total_eV"][0])
        common = {
            "run_id": self.run_id, "backend": "lammps",
            "atom_ids": [int(i) for i in spec.atom_ids],
            "numbers": [int(z) for z in spec.numbers],
            "roles": list(spec.roles),
            "cell": [list(r) for r in spec.cell_A], "pbc": list(spec.pbc),
            "frame_steps": [int(s) for s in trajectory["steps"]],
            "frame_times_fs": series["time_fs"],
            "collision": dict(spec.collision),
        }
        self.arrays["positions"] = StoredArray(
            trajectory["positions"], "A", "Sampled atom positions from LAMMPS dynamics",
            "trajectory", dict(common))
        self.arrays["velocities"] = StoredArray(
            trajectory["velocities"], "A/fs", "Sampled atom velocities from LAMMPS dynamics",
            "trajectory", dict(common))
        value = {k: series[k] for k in ("step", "time_fs", "potential_eV", "kinetic_eV",
                                        "total_eV", "temperature_K")}
        value.update({"frames": int(trajectory["positions"].shape[0]),
                      "positions_array": key(self.run_id, "positions"),
                      "velocities_array": key(self.run_id, "velocities")})
        conservative = spec.ensemble == "nve"
        convergence = Convergence(
            converged=True, iterations=int(spec.steps), residual=drift,
            residual_metric="total energy change over the run, eV",
            message=("Total-energy change over constant-energy dynamics: a measure of "
                     "time-step error." if conservative else
                     "Total energy is not conserved under a thermostat; the change is "
                     "reported for reference."))
        self.results["trajectory"] = Result("lammps_trajectory", value, "",
                                            self.provenance(), convergence=convergence)

    def final_state(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        meta = {"atom_ids": list(self.spec.atom_ids)}
        self.arrays["final_positions"] = StoredArray(
            np.ascontiguousarray(positions, dtype=np.float64), "A",
            "Positions at the end of the LAMMPS run", "per-atom", dict(meta))
        self.arrays["final_velocities"] = StoredArray(
            np.ascontiguousarray(velocities, dtype=np.float64), "A/fs",
            "Velocities at the end of the LAMMPS run", "per-atom", dict(meta))

    def annotate(self, outcome: LAMMPSOutcome, max_force: float) -> None:
        spec = self.spec
        common = {"run_id": outcome.run_id, "task": spec.task, "backend": "lammps",
                  "status": outcome.status, "structure_key": spec.structure_key,
                  "spec_digest": spec.digest, "input_state_digest": spec.input_state_digest,
                  "output_state_digest": outcome.output_state_digest,
                  "n_atoms": spec.n_atoms, "potential_id": spec.potential.get("id"),
                  "potential_sha256": spec.potential.get("sha256"),
                  "lammps_version": spec.lammps.get("version"),
                  "wall_time_s": self.audit.get("wall_time_s")}
        for result in self.results.values():
            result.extra.update(common)
        self.results["energy"].extra["max_force_eV_A"] = max_force
        record_value = {k: v for k, v in self.audit.items()
                        if k not in ("warnings_from_check",)}
        record_value["status"] = outcome.status
        record_value["arrays"] = {name: key(outcome.run_id, name) for name in self.arrays}
        record = Result("lammps_run", record_value, "", self.provenance(
            self._origin(outcome.status)))
        record.provenance.parameters["run_spec"] = spec.as_dict()
        record.extra.update(common)
        self.results["run"] = record
