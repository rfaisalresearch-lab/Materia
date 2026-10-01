"""Executing LAMMPS specifications through a deterministic fake process.

The fake (``tests/support/fake_lammps.py``) evaluates Materia's own EAM in
LAMMPS metal units and writes LAMMPS-format output.  Parity with Materia here
therefore proves that positions, types, masses, velocities, forces and stress
survive the round trip through the native files and back, with every unit
conversion applied once and correctly.  It says nothing about LAMMPS itself;
``tests/validation/test_lammps_live.py`` does that when LAMMPS is installed.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import eam
from materia.provenance import Origin
from materia.solvers.lammps import discover, native, runner
from materia.solvers.lammps import spec as specs
from materia.solvers.lammps.run import execute


def copper(repeat=(2, 2, 2), jitter=0.05, seed=3) -> Structure:
    from materia.python_api.api import Lab

    s = Lab().materials.load("copper").bulk(repeat=repeat).structure
    rng = np.random.default_rng(seed)
    s.positions = s.positions + rng.normal(0, jitter, s.positions.shape)
    return s


def freeze(structure, task="energy", **settings):
    environment = discover(refresh=True)
    potential = eam.load_shipped("Cu-Zhou04")
    return specs.build(structure, task, potential, environment, None, settings)


def reference(structure):
    return eam.load_shipped("Cu-Zhou04").evaluate(structure)


def test_energy_forces_and_stress_match_materia_through_the_round_trip(fake_lammps):
    s = copper()
    s.positions[:3] += s.cell.matrix[0] * np.array([1, -2, 3])[:, None]
    outcome = execute(freeze(s))
    assert outcome.status == "complete", outcome.reason
    ref = reference(s)
    assert outcome.results["energy"].value == pytest.approx(ref["energy_eV"], rel=1e-12)
    assert np.allclose(outcome.results["forces"].value, ref["forces_eV_A"], atol=1e-10)
    assert np.allclose(outcome.results["stress"].value, ref["stress_eV_A3"], atol=1e-12)
    assert outcome.results["stress"].unit == "eV/A^3"
    assert outcome.arrays["final_positions"].data.shape == (len(s), 3)
    assert np.allclose(outcome.arrays["final_positions"].data, s.positions, atol=0)
    record = outcome.results["run"].value
    assert record["status"] == "complete"
    assert record["command"][0] == str(fake_lammps)
    assert record["command"][1:] == ["-in", "in.lammps", "-log", "log.lammps", "-echo",
                                     "log", "-nocite"]
    assert record["log_version"] == "29 Aug 2024 - Update 1"
    assert set(record["inputs"]) == {"structure.data", "in.lammps", "Cu_Zhou04.eam.alloy"}
    assert record["inputs"]["Cu_Zhou04.eam.alloy"]["sha256"] == \
        eam.load_shipped("Cu-Zhou04").identity.sha256
    assert record["inputs"]["structure.data"]["sha256"] == native.sha256_text(
        native.render_data(outcome.spec))
    assert record["input_script"] == native.render_input(outcome.spec)
    assert len(record["log_tail"].splitlines()) <= 200
    prov = outcome.results["energy"].provenance
    assert prov.origin is Origin.CALCULATED
    assert prov.model == "external:lammps/eam/alloy/Cu-Zhou04"
    assert prov.fidelity.value == "tier1-classical"
    assert prov.parameters["lammps"]["version"] == "29 Aug 2024 - Update 1"
    assert outcome.output_state_digest == outcome.spec.input_state_digest


def test_slab_and_cluster_boundaries(fake_lammps):
    slab = copper()
    slab.cell = Cell(np.diag([slab.cell.matrix[0, 0], slab.cell.matrix[1, 1], 30.0]),
                     (True, True, False))
    spec = freeze(slab)
    assert spec.boundary == ("p", "p", "s") and "stress" not in spec.outputs
    outcome = execute(spec)
    assert outcome.ok, outcome.reason
    assert outcome.results["energy"].value == pytest.approx(
        reference(slab)["energy_eV"], rel=1e-12)
    cluster = copper(repeat=(1, 1, 1))
    cluster.cell = Cell.none()
    outcome = execute(freeze(cluster))
    assert outcome.ok, outcome.reason
    assert np.allclose(outcome.results["forces"].value, reference(cluster)["forces_eV_A"],
                       atol=1e-10)


def test_triclinic_cell(fake_lammps):
    s = copper()
    m = s.cell.matrix.copy()
    m[1, 0] = 0.25 * m[0, 0]
    s.cell = Cell(m, (True, True, True))
    outcome = execute(freeze(s))
    assert outcome.ok, outcome.reason
    assert "xy xz yz" in native.render_data(outcome.spec)
    assert outcome.results["energy"].value == pytest.approx(reference(s)["energy_eV"],
                                                            rel=1e-12)


def test_isotopic_masses_survive_and_are_checked(fake_lammps):
    s = copper()
    s.mass_numbers[::2] = 65
    s.velocities = np.full((len(s), 3), 0.002)
    outcome = execute(freeze(s))
    assert outcome.ok, outcome.reason
    assert len(outcome.spec.type_table) == 2


def test_relaxation_converges_and_is_verified_independently(fake_lammps):
    s = copper(jitter=0.03)
    outcome = execute(freeze(s, "relax", ftol_eV_A=0.01, max_iterations=2000,
                             max_evaluations=20000))
    assert outcome.status == "converged", outcome.reason
    energy = outcome.results["energy"]
    assert energy.convergence.converged
    assert energy.convergence.residual <= 0.01
    assert energy.value < reference(s)["energy_eV"]
    relaxation = outcome.results["relaxation"].value
    assert relaxation["stopping_criterion"] == "force tolerance"
    assert relaxation["iterations"] == outcome.results["thermo"].value["main"]["step"][-1]
    assert not np.allclose(outcome.arrays["final_positions"].data, s.positions)
    assert outcome.output_state_digest != outcome.spec.input_state_digest
    final = s.copy()
    final.positions = outcome.arrays["final_positions"].data
    assert np.allclose(outcome.results["forces"].value, reference(final)["forces_eV_A"],
                       atol=1e-10)


def test_unconverged_relaxation_is_marked_estimated(fake_lammps, monkeypatch):
    monkeypatch.setenv("FAKE_LAMMPS_MODE", "maxiter")
    outcome = execute(freeze(copper(), "relax", ftol_eV_A=1e-6))
    assert outcome.status == "not-converged"
    energy = outcome.results["energy"]
    assert energy.convergence.converged is False
    assert energy.provenance.origin is Origin.ESTIMATED
    assert "not an energy minimum" in energy.convergence.message


def test_fixed_atoms_stay_put(fake_lammps):
    s = copper(jitter=0.04)
    s.fixed[:4] = True
    outcome = execute(freeze(s, "relax", ftol_eV_A=0.02))
    assert outcome.ok, outcome.reason
    assert np.allclose(outcome.arrays["final_positions"].data[:4], s.positions[:4],
                       rtol=0, atol=1e-12)
    assert not np.allclose(outcome.arrays["final_positions"].data[4:], s.positions[4:])
    moving = execute(freeze(s, "md", steps=10, sample_every=5, timestep_fs=1.0,
                            initial_velocities="create", temperature_K=300.0, seed=4))
    assert moving.ok, moving.reason
    assert np.allclose(moving.arrays["positions"].data[-1, :4], s.positions[:4],
                       rtol=0, atol=1e-12)


def test_nve_dynamics_trajectory_and_units(fake_lammps):
    s = copper(jitter=0.02)
    rng = np.random.default_rng(7)
    s.velocities = rng.normal(0, 0.005, s.positions.shape)
    s.velocities -= s.velocities.mean(axis=0)
    outcome = execute(freeze(s, "md", steps=40, timestep_fs=1.0, sample_every=10))
    assert outcome.status == "complete", outcome.reason
    positions = outcome.arrays["positions"]
    velocities = outcome.arrays["velocities"]
    assert positions.data.shape == (5, len(s), 3) and positions.unit == "A"
    assert velocities.unit == "A/fs"
    assert np.allclose(positions.data[0], s.positions, atol=1e-12)
    assert np.allclose(velocities.data[0], s.velocities, rtol=1e-12, atol=1e-15)
    trajectory = outcome.results["trajectory"].value
    assert trajectory["step"] == [0, 10, 20, 30, 40]
    assert trajectory["time_fs"] == pytest.approx([0.0, 10.0, 20.0, 30.0, 40.0])
    assert trajectory["frames"] == 5
    assert trajectory["positions_array"].endswith("::positions")
    assert positions.meta["atom_ids"] == [int(i) for i in s.ids]
    assert positions.meta["backend"] == "lammps"
    drift = outcome.results["trajectory"].convergence.residual
    kinetic = outcome.results["trajectory"].value["kinetic_eV"][0]
    assert abs(drift) < 0.01 * kinetic
    displacement = positions.data[1] - positions.data[0]
    assert np.allclose(displacement, s.velocities * 10.0, atol=0.02)
    assert np.allclose(outcome.arrays["final_velocities"].data, velocities.data[-1])


def test_thermostatted_dynamics_is_reproducible_by_seed(fake_lammps):
    s = copper(jitter=0.0)
    runs = [execute(freeze(s, "md", steps=20, sample_every=10, ensemble="langevin",
                           temperature_K=600.0, damping_fs=20.0, seed=seed,
                           initial_velocities="create"))
            for seed in (11, 11, 12)]
    assert all(r.ok for r in runs), [r.reason for r in runs]
    first, again, other = (r.arrays["positions"].data for r in runs)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert runs[0].results["run"].provenance.seed == 11
    nvt = execute(freeze(s, "md", steps=20, sample_every=10, ensemble="nvt",
                         temperature_K=300.0, damping_fs=50.0, seed=5,
                         initial_velocities="create"))
    assert nvt.ok, nvt.reason
    assert "mean_temperature" in nvt.results


def test_collision_setup_replays_through_the_shared_contract(fake_lammps):
    from materia.experiments.collisions import prepare

    a = copper(repeat=(1, 1, 1), jitter=0.0)
    a.cell = Cell.none()
    b = a.copy()
    collision = prepare(a, b, relative_speed_A_fs=0.05, gap_A=3.0)
    outcome = execute(freeze(collision, "md", steps=20, sample_every=5, timestep_fs=1.0))
    assert outcome.ok, outcome.reason
    meta = outcome.arrays["positions"].meta
    assert meta["collision"]["projectile_atoms"] == len(a)
    assert meta["roles"][:len(a)] == ["projectile"] * len(a)
    assert np.allclose(outcome.arrays["velocities"].data[0], collision.velocities)


@pytest.mark.parametrize("mode, task, fragment", [
    ("exit1", "energy", "exited with status 1"),
    ("error_line", "energy", "reported an error"),
    ("truncate", "energy", "truncated"),
    ("truncate_dump", "energy", "truncated"),
    ("truncate_trajectory", "md", "truncated"),
    ("nan", "energy", "not finite"),
    ("nan_force", "energy", "not finite"),
    ("lose_atom", "energy", "atoms were lost"),
    ("lose_atom_dump_only", "energy", "Atoms were lost"),
    ("change_id", "energy", "identifiers"),
    ("change_type", "energy", "type"),
    ("bad_mass", "energy", "mass"),
    ("version_log", "energy", "not the 29 Aug 2024"),
    ("bad_ke", "md", "kinetic energy"),
    ("no_log", "energy", "without writing a log"),
    ("malformed_thermo", "energy", "values"),
])
def test_corrupt_output_is_refused_with_nothing_kept(fake_lammps, monkeypatch, mode, task,
                                                     fragment):
    s = copper()
    s.velocities = np.full((len(s), 3), 0.001)
    s.mass_numbers[0] = 65
    spec = freeze(s, task, **({"steps": 10, "sample_every": 5} if task == "md" else {}))
    if mode == "change_type":
        assert len(spec.type_table) == 2
    monkeypatch.setenv("FAKE_LAMMPS_MODE", mode)
    outcome = execute(spec)
    assert outcome.status == "failed"
    assert fragment.lower() in outcome.reason.lower(), outcome.reason
    assert outcome.results == {} and outcome.arrays == {}
    assert not outcome.ok


def test_mass_override_by_the_potential_file_is_detected(fake_lammps, monkeypatch):
    s = copper()
    s.mass_numbers[:] = 65
    monkeypatch.setenv("FAKE_LAMMPS_MODE", "ignore_mass_command")
    outcome = execute(freeze(s))
    assert outcome.status == "failed" and "mass" in outcome.reason


def test_cancellation_stops_the_process_and_keeps_nothing(fake_lammps, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_LAMMPS_MODE", "sleep")
    spec = freeze(copper(), "md", steps=400, sample_every=10)
    flag = threading.Event()
    seen = []

    def progress(fraction, message):
        seen.append(message)
        if "step" in message:
            flag.set()

    started = time.time()
    outcome = execute(spec, progress=progress, cancelled=flag.is_set)
    assert outcome.status == "cancelled"
    assert outcome.results == {} and outcome.arrays == {}
    assert time.time() - started < 15
    assert runner.active_runs() == 0
    assert list(Path(os.environ["MATERIA_LAMMPS_RUN_DIR"]).iterdir()) == []
    assert any("LAMMPS step" in m for m in seen)


def test_cancellation_before_start(fake_lammps):
    outcome = execute(freeze(copper()), cancelled=lambda: True)
    assert outcome.status == "cancelled" and not outcome.results


def test_time_limit(fake_lammps, monkeypatch):
    monkeypatch.setenv("FAKE_LAMMPS_MODE", "hang")
    outcome = execute(freeze(copper()), timeout_s=1.0)
    assert outcome.status == "failed" and "time limit" in outcome.reason
    assert runner.active_runs() == 0


def test_refused_specification_starts_no_process(fake_lammps):
    spec = freeze(copper(), "md", steps=10, sample_every=5, pressure_bar=1.0)
    with pytest.raises(specs.SpecRefused) as info:
        execute(spec)
    assert "pressure_bar" in info.value.report.blocking
    assert list(Path(os.environ["MATERIA_LAMMPS_RUN_DIR"]).glob("*")) == []


def test_a_changed_installation_is_refused(fake_lammps, monkeypatch):
    spec = freeze(copper())
    monkeypatch.setenv("FAKE_LAMMPS_VERSION", "7 Feb 2024")
    with pytest.raises(specs.SpecRefused, match="Freeze the specification again"):
        execute(spec)


def test_kept_run_directory_is_auditable(fake_lammps):
    outcome = execute(freeze(copper()), keep_files=True)
    workdir = Path(outcome.audit["workdir"])
    names = {p.name for p in workdir.iterdir()}
    assert {"in.lammps", "structure.data", "Cu_Zhou04.eam.alloy", "log.lammps",
            "final.dump"} <= names
    for name, identity in outcome.audit["inputs"].items():
        assert native.sha256_text((workdir / name).read_text(errors="replace")) == \
            identity["sha256"] or name.endswith(".eam.alloy")
    assert outcome.results["run"].value["workdir"] == str(workdir)


def test_python_module_route_runs_the_same_input(fake_lammps_module):
    s = copper()
    outcome = execute(freeze(s))
    assert outcome.ok, outcome.reason
    assert outcome.audit["command"][1].endswith("module_worker.py")
    assert outcome.results["energy"].value == pytest.approx(reference(s)["energy_eV"],
                                                            rel=1e-12)
