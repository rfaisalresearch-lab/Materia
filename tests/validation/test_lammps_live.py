"""Live parity between real LAMMPS and Materia's EAM on the shipped Cu potential.

These cases run only when a real LAMMPS is installed and found by
:func:`materia.solvers.lammps.discover`.  Without one they are skipped with
the reason ``BLOCKED``: the validation has not been performed, and a skip is
not a pass.

Both codes read the same checksummed setfl file and use the same cubic
Hermite interpolation, so energies, forces and stresses should agree to
round-off.  The tolerances below allow for LAMMPS's own table handling
(it stores ``r*phi`` splines and evaluates in single-pass double precision)
and for its ``nktv2p`` constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.physics import eam
from materia.solvers.lammps import discover
from materia.solvers.lammps import spec as specs
from materia.solvers.lammps.run import execute


@pytest.fixture(scope="module")
def real_lammps():
    environment = discover(refresh=True)
    if not environment.available:
        pytest.skip("BLOCKED: live LAMMPS parity was not run because no LAMMPS is "
                    f"installed. {environment.blocking_reason()} {environment.install_hint()}")
    if not environment.has_style("pair", "eam/alloy"):
        pytest.skip(f"BLOCKED: LAMMPS {environment.version} at {environment.path} has no "
                    "pair_style eam/alloy (MANYBODY package).")
    return environment


def copper(jitter: float) -> "object":
    from materia.python_api.api import Lab

    s = Lab().materials.load("copper").bulk(repeat=(3, 3, 3)).structure
    rng = np.random.default_rng(11)
    s.positions = s.positions + rng.normal(0, jitter, s.positions.shape)
    return s


def freeze(environment, structure, task="energy", **settings):
    return specs.build(structure, task, eam.load_shipped("Cu-Zhou04"), environment, None,
                       settings)


def test_live_energy_forces_and_stress_match_materia(real_lammps):
    s = copper(0.05)
    outcome = execute(freeze(real_lammps, s))
    assert outcome.ok, outcome.reason
    reference = eam.load_shipped("Cu-Zhou04").evaluate(s)
    n = len(s)
    assert outcome.results["energy"].value / n == pytest.approx(
        reference["energy_eV"] / n, abs=1e-6)
    assert np.abs(outcome.results["forces"].value - reference["forces_eV_A"]).max() < 1e-5
    assert np.abs(np.array(outcome.results["stress"].value)
                  - reference["stress_eV_A3"]).max() < 1e-6
    assert outcome.results["run"].value["log_version"] == real_lammps.version


def test_live_relaxation_reaches_the_same_minimum(real_lammps):
    s = copper(0.03)
    outcome = execute(freeze(real_lammps, s, "relax", ftol_eV_A=1e-4, max_iterations=5000,
                             max_evaluations=50000))
    assert outcome.status == "converged", outcome.reason
    relaxed = s.copy()
    relaxed.positions = outcome.arrays["final_positions"].data
    reference = eam.load_shipped("Cu-Zhou04").evaluate(relaxed)
    assert np.abs(reference["forces_eV_A"]).max() < 5e-4
    assert outcome.results["energy"].value == pytest.approx(reference["energy_eV"], abs=1e-5)


def test_live_nve_conserves_energy_and_replays(real_lammps):
    s = copper(0.0)
    outcome = execute(freeze(real_lammps, s, "md", steps=200, timestep_fs=1.0,
                             sample_every=20, initial_velocities="create",
                             temperature_K=300.0, seed=4242))
    assert outcome.ok, outcome.reason
    drift = outcome.results["trajectory"].convergence.residual
    assert abs(drift) / len(s) < 1e-4
    assert outcome.arrays["positions"].data.shape == (11, len(s), 3)
