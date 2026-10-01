"""A deterministic stand-in for the GPAW worker's band-structure output.

:class:`FakeBandsGPAW` replaces :func:`materia.solvers.gpaw_driver.runner.run_job`.
A job with a ``bands`` block gets a converged ground state and synthetic,
smooth, ordered bands on exactly the k-points it was sent, with every
parameter echoed back, the reciprocal cell of the job's cell and the
cumulative path distance computed from it.  Every other job goes to
:class:`tests.support.fake_gpaw_dos.FakeGPAW`.  It computes no physics and
proves nothing about GPAW.

``mode`` selects a fault: ``"complete"``, ``"failed"``, ``"timeout"``,
``"cancelled"``, ``"wait_for_cancel"``, ``"not_converged"``, ``"nscf_unconverged"``,
``"echo_nscf"``, ``"echo_kpts"``, ``"echo_scf"``, ``"echo_labels"``,
``"wrong_order"``, ``"folded"``, ``"symmetry"``, ``"missing_bands"``,
``"missing_kpoint"``, ``"spin_mismatch"``, ``"flat_shape"``, ``"nan"``, ``"inf"``,
``"unsorted"``, ``"bad_distance"``, ``"bad_recip"``, ``"fermi_moved"``,
``"no_fermi"`` and ``"no_bands"``.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import FERMI_EV, SETUPS, SYMBOLS, FakeGPAW

GAP_EV = 1.2


def synthetic_bands(kpts, nbands, occupied, nspins):
    """Smooth ordered bands: occupied below the Fermi level, empty above, a gap between."""
    kpts = np.asarray(kpts, dtype=float)
    phase = np.cos(2 * math.pi * kpts).sum(axis=1)
    eig = np.zeros((nspins, len(kpts), nbands))
    for s in range(nspins):
        for n in range(nbands):
            if n < occupied:
                base = FERMI_EV - 0.5 - 2.5 * (occupied - 1 - n)
                eig[s, :, n] = base - 0.1 * (3.0 - phase) + 0.15 * s
            else:
                base = FERMI_EV + GAP_EV + 2.5 * (n - occupied)
                eig[s, :, n] = base + 0.1 * (3.0 + phase) + 0.15 * s
    return eig


class FakeBandsGPAW:
    def __init__(self, mode: str = "complete") -> None:
        self.mode = mode
        self.other = FakeGPAW()
        self.jobs = []
        self.started = None

    @property
    def relax(self):
        return self.other.relax

    def __call__(self, job, environment, **kwargs) -> runner.GPAWRun:
        if "bands" not in job:
            self.jobs.append(copy.deepcopy(job))
            return self.other(job, environment, **kwargs)
        self.jobs.append(copy.deepcopy(job))
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
        if self.mode == "echo_scf":
            result["parameters_used"]["symmetry"] = {"point_group": False,
                                                     "time_reversal": False} if \
                job["parameters"]["symmetry"]["point_group"] else {"point_group": True,
                                                                   "time_reversal": True}
        if self.mode != "no_bands":
            self._bands(job, result, arrays)
        return runner.GPAWRun(status=result["status"], result=result, iterations=6,
                              wall_time_s=0.01, command=["fake"], arrays=arrays)

    def _bands(self, job, result, arrays):
        settings = job["bands"]
        nscf = settings["nscf"]
        kpts = np.asarray(nscf["kpts"], dtype=float)
        nbands = int(nscf["nbands"])
        nspins = 2 if job["parameters"].get("spinpol") else 1
        numbers = [int(z) for z in job["structure"]["numbers"]]
        valence = sum(SETUPS[SYMBOLS[z]][2] for z in numbers) - float(
            job["parameters"].get("charge", 0.0))
        occupied = int(math.ceil(valence / 2.0))
        eig = synthetic_bands(kpts, nbands, occupied, nspins)
        cell = np.asarray(job["structure"]["cell"], dtype=float)
        recip = 2 * math.pi * np.linalg.inv(cell).T
        returned = kpts.copy()
        if self.mode == "wrong_order":
            returned = returned[::-1].copy()
            eig = eig[:, ::-1]
        if self.mode == "folded":
            returned = -returned
        cart = returned @ recip
        steps = np.linalg.norm(np.diff(cart, axis=0), axis=1)
        for index in settings["breaks"]:
            steps[index - 1] = 0.0
        distance = np.concatenate([[0.0], np.cumsum(steps)])
        if self.mode == "bad_distance":
            distance = distance * 1.001
        if self.mode == "bad_recip":
            recip = recip * 1.01
        if self.mode == "missing_bands":
            eig = eig[:, :, :-1]
        if self.mode == "missing_kpoint":
            eig = eig[:, :-1]
        if self.mode == "spin_mismatch":
            eig = np.concatenate([eig, eig + 0.2]) if nspins == 1 else eig[:1]
        if self.mode == "flat_shape":
            eig = eig[0]
        if self.mode == "nan":
            eig[0, 1, 0] = float("nan")
        if self.mode == "inf":
            eig[-1, -1, -1] = float("inf")
        if self.mode == "unsorted":
            eig[0, 2, [0, 1]] = eig[0, 2, [1, 0]]
        arrays["band_eigenvalues"] = eig
        arrays["band_kpoints"] = returned[:-1] if self.mode == "missing_kpoint" else returned
        arrays["band_bz_kpoints"] = returned
        arrays["band_distance"] = distance
        used_nscf = copy.deepcopy(nscf)
        if self.mode == "echo_nscf":
            used_nscf["nbands"] = nbands + 1
        if self.mode == "echo_kpts":
            used_nscf["kpts"][1][0] += 0.01
        labels = list(settings["labels"])
        if self.mode == "echo_labels":
            labels[0] = "Q"
        result["bands"] = {
            "used": {"reference": settings["reference"], "n_bands": settings["n_bands"],
                     "labels": labels, "breaks": list(settings["breaks"])},
            "nscf_parameters_used": used_nscf,
            "nscf_converged": self.mode != "nscf_unconverged",
            "nscf_iterations": 9,
            "nscf_symmetry_operations": 48 if self.mode == "symmetry" else 1,
            "fermi_level_scf_eV": None if self.mode == "no_fermi" else FERMI_EV,
            "fermi_level_nscf_eV": FERMI_EV + (0.1 if self.mode == "fermi_moved" else 0.0),
            "n_spins": nspins, "n_kpoints": len(kpts), "n_bands_computed": nbands,
            "reciprocal_cell_invA": recip.tolist(), "cell_A": cell.tolist(),
        }
