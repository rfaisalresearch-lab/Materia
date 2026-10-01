"""A deterministic stand-in for the GPAW worker's relaxation output.

:class:`FakeRelaxWorker` replaces :func:`materia.solvers.gpaw_driver.runner.run_job`.
It reads the job Materia actually built, moves every mobile atom a fixed
fraction of the way towards a target geometry, scales the cell for a
variable-cell job, and reports the result in the worker's JSON layout,
echoing the parameters and optimiser it was given.  It computes no physics
and proves nothing about GPAW; it lets the relaxation pipeline, storage,
ownership and interface be tested exactly without GPAW.

``mode`` selects a behaviour: ``"converge"``, ``"max_steps"``, ``"failed"``,
``"cancelled"``, ``"timeout"``, ``"wait_for_cancel"``, ``"echo_symmetry"``,
``"echo_optimizer"``, ``"lose_atom"``, ``"move_fixed"``, ``"change_cell"``,
``"nan_energy"``, ``"wrong_verdict"`` and ``"no_forces"``.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from materia.solvers.gpaw_driver import GPAWEnvironment, runner


def environment(tmp_path) -> GPAWEnvironment:
    setup = tmp_path / "setups"
    setup.mkdir(exist_ok=True)
    for symbol, number, core, valence in (("H", 1, 0, 1), ("Si", 14, 10, 4),
                                          ("Cu", 29, 18, 11)):
        for functional in ("LDA", "PBE"):
            (setup / f"{symbol}.{functional}").write_text(
                f'<paw_setup><atom symbol="{symbol}" Z="{number}" '
                f'core="{core}" valence="{valence}"/></paw_setup>', encoding="utf-8")
    return GPAWEnvironment(
        interpreter="/fake/gpaw/python", source="test", code_available=True,
        datasets_available=True, gpaw_version="25.7.0", ase_version="3.29.0",
        python_version="3.13.0", setup_paths=[str(setup)],
        dataset_dirs=[{"path": str(setup), "files": 6}])


VALENCE = {1: (1.0, 0.0), 14: (4.0, 10.0), 29: (11.0, 18.0)}
SYMBOLS = {1: "H", 14: "Si", 29: "Cu"}


class FakeRelaxWorker:
    def __init__(self, mode: str = "converge", shift=(0.0, 0.0, 0.05), cell_scale=0.98,
                 steps: int = 4) -> None:
        self.mode = mode
        self.shift = np.asarray(shift, dtype=float)
        self.cell_scale = float(cell_scale)
        self.steps = int(steps)
        self.jobs = []
        self.started = None

    def __call__(self, job, environment, **kwargs) -> runner.GPAWRun:
        self.jobs.append(copy.deepcopy(job))
        cancelled = kwargs.get("cancelled")
        progress = kwargs.get("progress")
        if self.started is not None:
            self.started.set()
        if self.mode == "wait_for_cancel":
            deadline = time.time() + 10
            while time.time() < deadline and not (cancelled and cancelled()):
                time.sleep(0.01)
            return runner.GPAWRun(status=runner.STATUS_CANCELLED, result={}, iterations=0)
        if self.mode == "cancelled":
            return runner.GPAWRun(status=runner.STATUS_CANCELLED, result={}, iterations=0)
        if self.mode == "failed":
            return runner.GPAWRun(status=runner.STATUS_FAILED,
                                  result={"status": "failed", "error": "RuntimeError: boom"},
                                  error="RuntimeError: boom")
        if self.mode == "timeout":
            return runner.GPAWRun(status=runner.STATUS_FAILED, result={},
                                  stderr="\nMateria stopped the calculation after 1 s.\n",
                                  error="Materia stopped the calculation after 1 s.")
        return self._finished(job, progress)

    def _finished(self, job, progress) -> runner.GPAWRun:
        structure = job["structure"]
        relax = job["relaxation"]
        numbers = [int(z) for z in structure["numbers"]]
        n = len(numbers)
        initial = np.asarray(structure["positions"], dtype=float)
        cell0 = np.asarray(structure["cell"], dtype=float)
        mobile = np.ones(n, dtype=bool)
        mobile[list(relax["fixed_indices"])] = False
        variable = relax["filter"] is not None
        converge = self.mode != "max_steps"
        total = self.steps if converge else int(relax["max_steps"])
        total = min(total, int(relax["max_steps"]))
        fmax = float(relax["fmax"])
        rows, positions, cells = [], [], []
        for step in range(total + 1):
            fraction = step / max(1, total)
            pos = initial.copy()
            pos[mobile] += self.shift * fraction
            scale = 1.0 + (self.cell_scale - 1.0) * fraction if variable else 1.0
            cell = cell0 * scale
            if variable:
                pos = pos * scale
            force = 1.0 * (1.0 - fraction) + (0.5 * fmax if converge else 2.0 * fmax)
            residual = (0.01 * (1.0 - fraction) + 0.5 * float(relax["stress_tol"])
                        if variable else None)
            row = {"step": step, "energy_free_eV": -10.0 - fraction,
                   "energy_eV": -10.0 - fraction, "max_force_eV_A": force,
                   "stress_residual_eV_A3": residual,
                   "volume_A3": float(abs(np.linalg.det(cell))), "scf_iterations": 5,
                   "stress_eV_A3": None}
            rows.append(row)
            positions.append(pos)
            cells.append(cell)
            if progress is not None:
                progress({"event": "relax", **{k: v for k, v in row.items()
                                               if k != "stress_eV_A3"}})
        final = positions[-1].copy()
        final_cell = cells[-1].copy()
        converged = converge
        if self.mode == "lose_atom":
            final = final[:-1]
        if self.mode == "move_fixed":
            final[~mobile] += 0.1
        if self.mode == "change_cell":
            final_cell = final_cell * 1.01
        if self.mode == "nan_energy":
            rows[-1]["energy_free_eV"] = float("nan")
        if self.mode == "wrong_verdict":
            converged = not converged
        used = {"mode": relax["mode"], "optimizer": relax["optimizer"],
                "maxstep": relax["maxstep"], "fmax": relax["fmax"],
                "stress_tol": relax["stress_tol"], "max_steps": relax["max_steps"],
                "fixed_indices": list(relax["fixed_indices"]),
                "filter": copy.deepcopy(relax["filter"])}
        if self.mode == "echo_optimizer":
            used["optimizer"] = "LBFGS"
        parameters = copy.deepcopy(job["parameters"])
        if self.mode == "echo_symmetry":
            sent = parameters["symmetry"]
            parameters["symmetry"] = {key: not value for key, value in sent.items()}
        valence = sum(VALENCE.get(z, (0.0, 0.0))[0] for z in numbers)
        charge = float(job["parameters"].get("charge", 0.0))
        total_electrons = float(sum(numbers)) - charge
        result = {
            "status": "converged" if converged else "not-converged",
            "converged": converged, "iterations": 5,
            "parameters_used": parameters,
            "relaxation_used": used,
            "relaxation": {"steps": rows, "converged": converged,
                           "stop_reason": "criteria met" if converged else "max_steps",
                           "final_positions": final.tolist(),
                           "final_cell": final_cell.tolist(),
                           "final_forces": np.zeros((n, 3)).tolist()},
            "relaxation_scf_iterations": 5 * len(rows),
            "energy_free_eV": -10.0 - 1.0, "energy_extrapolated_eV": -10.0 - 1.0,
            "reference_energy_eV": 0.0, "energy_contributions_eV": {},
            "forces_eV_per_A": (np.full((n, 3), 0.001)).tolist(),
            "datasets": [{"symbol": SYMBOLS[z], "path": "", "file": "",
                          "valence_electrons": VALENCE[z][0],
                          "core_electrons": VALENCE[z][1]} for z in sorted(set(numbers))],
            "accounting": {"gpaw_valence_electrons": valence - charge,
                           "occupied_electrons": valence - charge,
                           "density_integral_e": total_electrons,
                           "density_grid": [8, 8, 8]},
            "scf_history": [{"ionic_step": total, "iteration": i + 1, "energy_eV": -11.0,
                             "converged": i == 4} for i in range(5)],
            "versions": {"gpaw": "25.7.0", "ase": "3.29.0"},
            "warnings": [], "log_tail": "fake worker log",
        }
        if self.mode == "no_forces":
            del result["forces_eV_per_A"]
        if variable:
            p = float(relax["filter"]["scalar_pressure"])
            residual = rows[-1]["stress_residual_eV_A3"]
            result["stress_eV_per_A3"] = (-(p + residual) * np.eye(3)).tolist()
        if self.mode == "lose_atom":
            result["forces_eV_per_A"] = result["forces_eV_per_A"][:-1]
        arrays = {"relax_positions": np.asarray(positions), "relax_cells": np.asarray(cells)}
        return runner.GPAWRun(status=result["status"], result=result, iterations=5,
                              wall_time_s=0.01, command=["fake"], arrays=arrays)
