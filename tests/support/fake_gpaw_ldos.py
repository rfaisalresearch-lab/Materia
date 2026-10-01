"""A deterministic stand-in for the GPAW worker's LDOS output.

:class:`FakeLDOSGPAW` replaces :func:`materia.solvers.gpaw_driver.runner.run_job`.
A job with an ``ldos`` block gets a converged ground state, eigenvalues with
known states in the window, and a synthetic map on a grid of the job's cell:
Gaussian bumps over every atom that decay as exp(-2 kappa z) into the vacuum,
scaled to 0.95 of the states in the window, as pseudo-wavefunction norms are.
Every other job goes to :class:`tests.support.fake_gpaw_dos.FakeGPAW`.  It
computes no physics and proves nothing about GPAW.

``mode`` selects a fault: ``"complete"``, ``"failed"``, ``"timeout"``,
``"cancelled"``, ``"wait_for_cancel"``, ``"not_converged"``,
``"nscf_unconverged"``, ``"echo_nscf"``, ``"echo_window"``, ``"fermi_moved"``,
``"nan"``, ``"negative"``, ``"wrong_grid"``, ``"wrong_cell"``, ``"bad_count"``,
``"missing_band"``, ``"short_bands"``, ``"empty"``, ``"spin_mismatch"``,
``"no_radii"``, ``"bad_norm"`` and ``"weights"``.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import FERMI_EV, SETUPS, SYMBOLS, FakeGPAW

KAPPA_PER_A = 1.1
GRID_SPACING_A = 0.25
RADII_A = {"H": 0.5, "Si": 1.05, "O": 0.7, "Cu": 1.1, "C": 0.64}


def synthetic_map(job, count):
    cell = np.asarray(job["structure"]["cell"], dtype=float)
    positions = np.asarray(job["structure"]["positions"], dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    shape = tuple(max(4, int(round(length / GRID_SPACING_A))) for length in lengths)
    frac = np.stack(np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij"), -1)
    points = frac @ cell
    density = np.zeros(shape)
    for atom in positions:
        lateral = np.sum((points[..., :2] - atom[:2]) ** 2, axis=-1)
        height = points[..., 2] - atom[2]
        density += np.exp(-lateral / 1.2) * np.exp(-2 * KAPPA_PER_A * np.abs(height))
    volume = abs(float(np.linalg.det(cell)))
    integral = density.sum() * volume / density.size
    return density * (0.95 * count / integral if integral > 0 else 0.0)


class FakeLDOSGPAW:
    def __init__(self, mode: str = "complete") -> None:
        self.mode = mode
        self.other = FakeGPAW()
        self.jobs = []
        self.started = None

    @property
    def relax(self):
        return self.other.relax

    def __call__(self, job, environment, **kwargs) -> runner.GPAWRun:
        self.jobs.append(copy.deepcopy(job))
        if "ldos" not in job:
            return self.other(job, environment, **kwargs)
        if self.started is not None:
            self.started.set()
        cancelled = kwargs.get("cancelled")
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
        result, arrays = self.other._ground_state(job)
        self._ldos(job, result, arrays)
        return runner.GPAWRun(status=result["status"], result=result, iterations=6,
                              wall_time_s=0.01, command=["fake"], arrays=arrays)

    def _ldos(self, job, result, arrays):
        settings = job["ldos"]
        nscf = settings["nscf"]
        size = [int(v) for v in nscf["kpts"]["size"]]
        nk = int(np.prod(size))
        total_bands = int(nscf["nbands"])
        n_bands = int(settings["n_bands"])
        nspins = 2 if job["parameters"].get("spinpol") else 1
        numbers = [int(z) for z in job["structure"]["numbers"]]
        valence = sum(SETUPS[SYMBOLS[z]][2] for z in numbers) - float(
            job["parameters"].get("charge", 0.0))
        occupied = int(math.ceil(valence / 2.0))
        low, high = float(settings["energy_min"]), float(settings["energy_max"])
        eig = np.zeros((nspins, nk, total_bands))
        for s in range(nspins):
            for n in range(total_bands):
                if n < occupied:
                    eig[s, :, n] = FERMI_EV - 0.5 - 2.0 * (occupied - 1 - n)
                else:
                    eig[s, :, n] = FERMI_EV + 0.5 + 2.0 * (n - occupied)
            eig[s] += 0.01 * np.arange(nk)[:, None] / nk
        if self.mode == "empty":
            eig += 50.0
        if self.mode == "short_bands":
            eig[:, :, :n_bands] = np.minimum(eig[:, :, :n_bands], FERMI_EV + high - 0.01)
            eig[:, :, :n_bands] = np.sort(eig[:, :, :n_bands], axis=2)
        weights = np.full(nk, 1.0 / nk)
        if self.mode == "weights":
            weights[0] *= 2.0
        degeneracy = 2.0 / nspins
        relative = eig[:, :, :n_bands] - FERMI_EV
        inside = (relative > low) & (relative < high)
        count = float(degeneracy * np.einsum("k,skn->", weights, inside))
        per_state = synthetic_map(job, 1.0)
        spin_maps = np.array([per_state * float(degeneracy * np.einsum(
            "k,kn->", weights, inside[s])) for s in range(nspins)])
        total = spin_maps.sum(axis=0)
        if self.mode == "bad_norm":
            total = total * 3.0
            spin_maps = spin_maps * 3.0
        if self.mode == "nan":
            total[0, 0, 0] = float("nan")
        if self.mode == "negative":
            total[0, 0, 0] = -1.0
        shape = list(total.shape)
        if self.mode == "wrong_grid":
            shape = [shape[0] + 1] + shape[1:]
        arrays["ldos_total"] = total
        if settings["spin"] == "resolved":
            arrays["ldos_spin"] = spin_maps * (0.9 if self.mode == "spin_mismatch" else 1.0)
        if self.mode == "missing_band":
            eig = eig[:, :, :-1]
        arrays["ldos_eigenvalues"] = eig
        arrays["ldos_kweights"] = weights
        used_nscf = copy.deepcopy(nscf)
        if self.mode == "echo_nscf":
            used_nscf["symmetry"] = {"point_group": True, "time_reversal": True}
        cell = copy.deepcopy(job["structure"]["cell"])
        if self.mode == "wrong_cell":
            cell[0][0] += 0.1
        radii = {SYMBOLS[z]: RADII_A[SYMBOLS[z]] for z in set(numbers)}
        if self.mode == "no_radii":
            radii = {}
        result["ldos"] = {
            "used": {"energy_min": low + (0.1 if self.mode == "echo_window" else 0.0),
                     "energy_max": high, "spin": settings["spin"], "n_bands": n_bands},
            "nscf_parameters_used": used_nscf,
            "nscf_converged": self.mode != "nscf_unconverged", "nscf_iterations": 7,
            "nscf_symmetry_operations": 1,
            "fermi_level_scf_eV": FERMI_EV,
            "fermi_level_nscf_eV": FERMI_EV + (0.1 if self.mode == "fermi_moved" else 0.0),
            "states_in_window": count + (0.5 if self.mode == "bad_count" else 0.0),
            "grid": shape, "cell_A": cell, "augmentation_radii_A": radii,
        }
