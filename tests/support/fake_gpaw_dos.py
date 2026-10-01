"""A deterministic stand-in for the GPAW worker's DOS output.

:class:`FakeGPAW` replaces :func:`materia.solvers.gpaw_driver.runner.run_job`.
A job with a ``dos`` block gets synthetic Kohn-Sham eigenvalues and the DOS
GPAW's own formulas give for them (Gaussian, or ASE's linear tetrahedron),
with every parameter it was sent echoed back.  A job with a ``relaxation``
block goes to :class:`tests.support.fake_gpaw_relax.FakeRelaxWorker`, and any
other job gets a converged ground state.  It computes no physics and proves
nothing about GPAW.

``mode`` selects a DOS fault: ``"complete"``, ``"failed"``, ``"timeout"``,
``"cancelled"``, ``"wait_for_cancel"``, ``"not_converged"``, ``"echo_nscf"``,
``"echo_width"``, ``"echo_grid"``, ``"bad_recompute"``, ``"spin_mismatch"``,
``"nan"``, ``"negative"``, ``"short_bands"``, ``"fermi_moved"``,
``"channels"``, ``"no_projector"`` and ``"nscf_unconverged"``.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from materia.solvers.gpaw_driver import GPAWEnvironment, runner
from tests.support.fake_gpaw_relax import FakeRelaxWorker

SETUPS = {
    "H": (1, 0, 1, [(1, 0)], [(None, 1)]),
    "Si": (14, 10, 4, [(3, 0), (3, 1)], [(None, 2)]),
    "O": (8, 2, 6, [(2, 0), (2, 1)], [(None, 2)]),
    "Cu": (29, 18, 11, [(4, 0), (4, 1), (3, 2)], []),
    "C": (6, 2, 4, [(2, 0), (2, 1)], [(None, 2)]),
}
SYMBOLS = {1: "H", 14: "Si", 8: "O", 29: "Cu", 6: "C"}
FERMI_EV = -2.0


def environment(tmp_path) -> GPAWEnvironment:
    setup = tmp_path / "dos-setups"
    setup.mkdir(exist_ok=True)
    for symbol, (number, core, valence, bound, unbound) in SETUPS.items():
        states = "".join(f'<state n="{n}" l="{l}" f="1" rc="1.0" e="-0.1" id="{symbol}-{n}{l}"/>'
                         for n, l in bound)
        states += "".join(f'<state l="{l}" rc="1.0" e="0.0" id="{symbol}-x{l}"/>'
                          for _, l in unbound)
        for functional in ("LDA", "PBE"):
            (setup / f"{symbol}.{functional}").write_text(
                f'<paw_setup><atom symbol="{symbol}" Z="{number}" core="{core}" '
                f'valence="{valence}"/><valence_states>{states}</valence_states></paw_setup>',
                encoding="utf-8")
    return GPAWEnvironment(
        interpreter="/fake/gpaw/python", source="test", code_available=True,
        datasets_available=True, gpaw_version="25.7.0", ase_version="3.29.0",
        python_version="3.13.0", setup_paths=[str(setup)],
        dataset_dirs=[{"path": str(setup), "files": 8}])


def gaussian(eig, weights, energies, width):
    out = np.zeros_like(energies)
    for w, row in zip(weights, eig):
        for e in row:
            out += w * np.exp(-((energies - e) / width) ** 2)
    return out / (math.sqrt(math.pi) * width)


class FakeGPAW:
    def __init__(self, mode: str = "complete") -> None:
        self.mode = mode
        self.relax = FakeRelaxWorker()
        self.jobs = []
        self.started = None

    def __call__(self, job, environment, **kwargs) -> runner.GPAWRun:
        self.jobs.append(copy.deepcopy(job))
        if "relaxation" in job:
            return self.relax(job, environment, **kwargs)
        if self.started is not None:
            self.started.set()
        cancelled = kwargs.get("cancelled")
        if "dos" in job:
            if self.mode == "wait_for_cancel":
                deadline = time.time() + 10
                while time.time() < deadline and not (cancelled and cancelled()):
                    time.sleep(0.01)
                return runner.GPAWRun(status=runner.STATUS_CANCELLED, result={})
            if self.mode == "cancelled":
                return runner.GPAWRun(status=runner.STATUS_CANCELLED, result={})
            if self.mode == "failed":
                return runner.GPAWRun(status=runner.STATUS_FAILED,
                                      result={"status": "failed", "error": "RuntimeError: boom"},
                                      error="RuntimeError: boom")
            if self.mode == "timeout":
                return runner.GPAWRun(status=runner.STATUS_FAILED, result={},
                                      stderr="\nMateria stopped the calculation after 1 s.\n",
                                      error="Materia stopped the calculation after 1 s.")
            if self.mode == "not_converged":
                return runner.GPAWRun(status=runner.STATUS_NOT_CONVERGED,
                                      result={"status": "not-converged"})
        result, arrays = self._ground_state(job)
        if "dos" in job:
            self._dos(job, result, arrays)
        return runner.GPAWRun(status=result["status"], result=result, iterations=6,
                              wall_time_s=0.01, command=["fake"], arrays=arrays)

    def _ground_state(self, job):
        structure = job["structure"]
        numbers = [int(z) for z in structure["numbers"]]
        charge = float(job["parameters"].get("charge", 0.0))
        valence = sum(SETUPS[SYMBOLS[z]][2] for z in numbers)
        n = len(numbers)
        result = {
            "status": "converged", "converged": True, "iterations": 6,
            "parameters_used": copy.deepcopy(job["parameters"]),
            "energy_free_eV": -11.0, "energy_extrapolated_eV": -11.0,
            "reference_energy_eV": 0.0, "energy_contributions_eV": {},
            "forces_eV_per_A": np.full((n, 3), 0.001).tolist(),
            "fermi_level_eV": FERMI_EV,
            "datasets": [{"symbol": SYMBOLS[z], "path": "", "file": "",
                          "valence_electrons": float(SETUPS[SYMBOLS[z]][2]),
                          "core_electrons": float(SETUPS[SYMBOLS[z]][1])}
                         for z in sorted(set(numbers))],
            "accounting": {"gpaw_valence_electrons": valence - charge,
                           "occupied_electrons": valence - charge,
                           "density_integral_e": float(sum(numbers)) - charge,
                           "density_grid": [8, 8, 8]},
            "scf_history": [{"iteration": i + 1, "energy_eV": -11.0, "converged": i == 5}
                            for i in range(6)],
            "versions": {"gpaw": "25.7.0", "ase": "3.29.0"},
            "warnings": [], "log_tail": "fake worker log",
        }
        if "stress" in job.get("observables", []):
            result["stress_eV_per_A3"] = np.zeros((3, 3)).tolist()
        return result, {}

    def _dos(self, job, result, arrays):
        settings = job["dos"]
        nscf = settings["nscf"]
        size = [int(v) for v in nscf["kpts"]["size"]]
        nk = int(np.prod(size))
        nbands = int(nscf["nbands"])
        nspins = 2 if job["parameters"].get("spinpol") else 1
        numbers = [int(z) for z in job["structure"]["numbers"]]
        valence = sum(SETUPS[SYMBOLS[z]][2] for z in numbers) - float(
            job["parameters"].get("charge", 0.0))
        occupied = int(math.ceil(valence / (2.0 if nspins == 1 else 2.0)))
        top = 30.0 if self.mode != "short_bands" else FERMI_EV + 1.0
        eig = np.zeros((nspins, nk, nbands))
        for s in range(nspins):
            for k in range(nk):
                below = np.linspace(-14.0, FERMI_EV - 1.0, max(1, occupied))
                above = np.linspace(FERMI_EV + 1.0, top, max(1, nbands - occupied))
                row = np.concatenate([below, above])[:nbands]
                eig[s, k] = row + 0.05 * k / max(1, nk) + 0.1 * s
        weights = np.full(nk, 1.0 / nk)
        reference = FERMI_EV if settings["reference"] == "fermi-level" else 0.0
        grid = settings["energy_min"] + settings["energy_step"] * np.arange(settings["npoints"])
        if self.mode == "echo_grid":
            grid = grid + 0.5
        absolute = grid + reference
        width = float(settings["width"])
        if width > 0:
            channels = [gaussian(eig[s], weights, absolute, width) for s in range(nspins)]
        else:
            from ase.dft.dos import linear_tetrahedron_integration

            cell = np.asarray(job["structure"]["cell"], dtype=float)
            channels = [np.asarray(linear_tetrahedron_integration(
                cell, eig[s].reshape(tuple(size) + (-1,)), absolute)) for s in range(nspins)]
        total = (2.0 / nspins) * sum(channels)
        if self.mode == "bad_recompute":
            total = total * 1.01
        if self.mode == "nan":
            total[0] = float("nan")
        if self.mode == "negative":
            total[0] = -1.0
        arrays["dos_energies"] = grid
        arrays["dos_total"] = total
        resolved = settings["spin"] == "resolved"
        if resolved:
            spin = np.asarray(channels)
            if self.mode == "spin_mismatch":
                spin = spin * 0.9
            arrays["dos_spin"] = spin
        projections = settings["projections"]
        counts = []
        if projections:
            fractions = [0.3 / (i + 1) for i in range(len(projections))]
            arrays["pdos_total"] = np.asarray([f * total for f in fractions])
            if resolved:
                arrays["pdos_spin"] = np.asarray([[f * c for c in channels] for f in fractions])
            for p in projections:
                counts.append([0 if self.mode == "no_projector" else 2 * "spdf".index(
                    p["angular"]) + 1 for _ in p["indices"]])
        arrays["dos_eigenvalues"] = eig
        arrays["dos_kweights"] = weights
        used_nscf = copy.deepcopy(nscf)
        if self.mode == "echo_nscf":
            used_nscf["nbands"] = nbands + 1
        used = {"reference": settings["reference"], "energy_min": float(grid[0]),
                "energy_step": float(settings["energy_step"]),
                "npoints": int(settings["npoints"]),
                "width": width + (0.01 if self.mode == "echo_width" else 0.0),
                "method": "tetrahedron" if width == 0 else "gaussian",
                "spin": settings["spin"],
                "projections": [{"label": p["label"], "indices": list(p["indices"]),
                                 "angular": p["angular"]} for p in projections]}
        channels_bound = copy.deepcopy(settings["channels"])
        if self.mode == "channels":
            channels_bound = {k: v + ["f"] for k, v in channels_bound.items()}
        result["dos"] = {
            "used": used, "nscf_parameters_used": used_nscf,
            "nscf_converged": self.mode != "nscf_unconverged",
            "fermi_level_scf_eV": FERMI_EV,
            "fermi_level_nscf_eV": FERMI_EV + (0.1 if self.mode == "fermi_moved" else 0.0),
            "reference_eV": reference, "n_spins": nspins, "n_ibz_kpoints": nk,
            "n_bz_kpoints": nk, "size": size, "bz2ibz": list(range(nk)),
            "cell": job["structure"]["cell"], "bound_channels": channels_bound,
            "projector_counts": counts, "nscf_iterations": 4,
        }
