"""A deterministic stand-in for GPAW ground states along an equation of state.

:class:`FakeEOSGPAW` replaces :func:`materia.solvers.gpaw_driver.runner.run_job`.
Each job gets the converged ground state of
:class:`tests.support.fake_gpaw_dos.FakeGPAW`, with the energy replaced by a
third-order Birch-Murnaghan curve of the job's cell volume (known V0, B0 and
B') and the stress by the matching pressure.  It computes no physics and
proves nothing about GPAW.

The zero-width extrapolated energy follows the known curve exactly.  The
free energy E - TS reported alongside it adds an electronic entropy term that
grows with the cell volume, ``TS_SLOPE_EV_PER_A3`` per A^3, as smearing does
in a metal, and the stress is -dF/dV of that free energy, as GPAW's is.  A fit
to the free energy therefore misses the known V0.

``mode`` selects a fault at point ``at``: ``"complete"``, ``"failed"``,
``"not_converged"``, ``"cancelled"``, ``"echo"``, ``"nan"``, ``"forces"``,
``"pulay"``, ``"noisy"`` and ``"shifted"`` (a minimum outside the range).
"""

from __future__ import annotations

import copy

import numpy as np

from materia.experiments.dft.eos import EV_A3_TO_GPA, birch_murnaghan, birch_murnaghan_pressure
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import FakeGPAW

V0_PER_ATOM = 20.0
B0_GPA = 90.0
B1 = 4.3
E0_PER_ATOM = -5.4
TS_SLOPE_EV_PER_A3 = 0.004


class FakeEOSGPAW:
    def __init__(self, mode: str = "complete", at: int = 3) -> None:
        self.mode = mode
        self.at = at
        self.other = FakeGPAW()
        self.jobs = []

    @property
    def relax(self):
        return self.other.relax

    def __call__(self, job, environment, **kwargs) -> runner.GPAWRun:
        index = len(self.jobs)
        self.jobs.append(copy.deepcopy(job))
        if "dos" in job or "relaxation" in job:
            return self.other(job, environment, **kwargs)
        faulty = index == self.at
        if faulty and self.mode == "failed":
            return runner.GPAWRun(status=runner.STATUS_FAILED,
                                  result={"status": "failed", "error": "RuntimeError: boom"},
                                  error="RuntimeError: boom")
        if faulty and self.mode == "not_converged":
            return runner.GPAWRun(status=runner.STATUS_NOT_CONVERGED,
                                  result={"status": "not-converged"})
        if faulty and self.mode == "cancelled":
            return runner.GPAWRun(status=runner.STATUS_CANCELLED, result={})
        result, arrays = self.other._ground_state(job)
        n = len(job["structure"]["numbers"])
        cell = np.asarray(job["structure"]["cell"], dtype=float)
        volume = abs(float(np.linalg.det(cell)))
        v0 = V0_PER_ATOM * n * (1.3 if self.mode == "shifted" else 1.0)
        b0 = B0_GPA / EV_A3_TO_GPA
        energy = float(birch_murnaghan(volume, E0_PER_ATOM * n, v0, b0, B1))
        smeared = job["parameters"]["occupations"].get("width", 0.0) > 0.0
        entropy = TS_SLOPE_EV_PER_A3 * volume if smeared else 0.0
        if self.mode == "noisy":
            energy += 0.02 * n * (1 if index % 2 else -1)
        if faulty and self.mode == "nan":
            energy = float("nan")
        result["energy_free_eV"] = energy - entropy
        result["energy_extrapolated_eV"] = energy
        pressure = float(birch_murnaghan_pressure(volume, v0, b0, B1))
        pressure += TS_SLOPE_EV_PER_A3 if smeared else 0.0
        if self.mode == "pulay":
            pressure += 2.0 / EV_A3_TO_GPA
        if "stress" in job.get("observables", []):
            result["stress_eV_per_A3"] = (-pressure * np.eye(3)).tolist()
        if self.mode == "forces":
            result["forces_eV_per_A"] = np.full((n, 3), 0.2).tolist()
        if faulty and self.mode == "echo":
            result["parameters_used"]["kpts"] = {"size": [2, 2, 2], "gamma": True}
        return runner.GPAWRun(status=result["status"], result=result, iterations=6,
                              wall_time_s=0.01, command=["fake"], arrays=arrays)
