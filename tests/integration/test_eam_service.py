"""EAM through the service, the HTTP layer and the Python API."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from materia.desktop_ui.server import create_server, find_free_port
from materia.desktop_ui.service import Service
from materia.project_format.project import Project
from materia.python_api.api import structure_state_digest


@pytest.fixture
def svc(tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    handle = service.lab.materials.load("copper").bulk(repeat=(3, 3, 3))
    rng = np.random.default_rng(5)
    handle.structure.positions = handle.structure.positions + rng.normal(
        0, 0.04, handle.structure.positions.shape)
    return service


def wait(service, job_id, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.job_status(job_id)
        if status["status"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_status_names_the_selected_potential_and_its_identity(svc):
    status = svc.eam_status()
    assert status["structure"]["selected"] == "Cu-Zhou04"
    ids = {p["id"]: p for p in status["potentials"]}
    assert set(ids) == {"Cu-Zhou04", "Au-Zhou04", "W-Zhou04"}
    cu = ids["Cu-Zhou04"]
    assert cu["license"] == "Public domain" and len(cu["sha256"]) == 64
    assert cu["cutoff_A"] == pytest.approx(5.715751993501819)


def test_energy_attaches_forces_undoably(svc):
    s = svc.structure
    assert np.isnan(s.forces).all()
    out = svc.eam_run("energy", background=False)
    assert out["ok"] and out["applied"] and out["current"]
    assert out["origin"] == "calculated"
    assert not np.isnan(s.forces).any()
    assert np.abs(s.forces.sum(axis=0)).max() < 1e-9
    svc.undo()
    assert np.isnan(svc.structure.forces).all()


def test_converged_relaxation_is_applied_and_undoable(svc):
    s = svc.structure
    before = s.positions.copy()
    out = svc.eam_run("relax", settings={"fmax_eV_A": 0.001}, background=False)
    assert out["ok"] and out["status"] == "converged" and out["applied"]
    assert out["max_force_eV_A"] < 0.001
    assert out["energy_change_eV"] < 0
    assert not np.allclose(s.positions, before)
    relaxed = s.positions.copy()
    svc.undo()
    np.testing.assert_allclose(svc.structure.positions, before)
    assert svc.eam_result()["current"] is False
    svc.redo()
    np.testing.assert_allclose(svc.structure.positions, relaxed)
    assert svc.eam_result()["current"] is True


def test_unconverged_relaxation_is_stored_as_estimated_and_not_applied(svc):
    before = svc.structure.positions.copy()
    n_history = len(svc.project.history._undo)
    out = svc.eam_run("relax", settings={"fmax_eV_A": 1e-6, "max_steps": 3},
                      background=False)
    assert out["ok"] and out["status"] == "not-converged"
    assert out["origin"] == "estimated" and out["applied"] is False
    assert "did not reach" in out["apply_reason"]
    np.testing.assert_array_equal(svc.structure.positions, before)
    assert len(svc.project.history._undo) == n_history


def test_dynamics_applies_final_state(svc):
    out = svc.eam_run("md", settings={"steps": 50, "temperature_K": 300, "seed": 2,
                                               "sample_every": 5},
                      background=False)
    assert out["ok"] and out["applied"] and out["steps"] == 10
    assert out["frames"] == 10 and out["trajectory_available"]
    assert out["mean_temperature_K"] > 0
    assert np.abs(svc.structure.velocities).max() > 0
    meta = svc.eam_trajectory(out["run_id"])
    assert meta["frames"] == 10 and meta["steps"][-1] == 50
    frame = svc.eam_trajectory(out["run_id"], 3)
    assert frame["frame"] == 3 and len(frame["positions"]) == 3 * len(svc.structure)
    assert len(frame["velocities"]) == 3 * len(svc.structure)
    assert sum(piece["atoms"] for piece in frame["fragments"]) == len(svc.structure)


def test_trajectory_arrays_survive_project_reopen(svc, tmp_path):
    run = svc.eam_run("md", settings={"steps": 8, "temperature_K": 100,
                                               "sample_every": 2}, background=False)
    path = svc.save_project(str(tmp_path / "trajectory.materia"))["path"]
    reopened = Service(Project.load(path), recovery_dir=str(tmp_path / "r-trajectory"))
    meta = reopened.eam_trajectory(run["run_id"])
    frame = reopened.eam_trajectory(run["run_id"], 3)
    assert meta["frames"] == 4 and meta["times_fs"][-1] == pytest.approx(16.0)
    assert frame["frame"] == 3 and frame["positions"]
    assert reopened.eam_result(run["run_id"])["trajectory_available"] is True


def test_trajectory_frame_refuses_invalid_index(svc):
    run = svc.eam_run("md", settings={"steps": 2, "temperature_K": 100},
                      background=False)
    assert svc.eam_trajectory(run["run_id"], -1)["ok"] is False
    assert svc.eam_trajectory(run["run_id"], 2)["ok"] is False


def test_background_run_reports_progress_and_completes(svc):
    started = svc.eam_run("relax", settings={"fmax_eV_A": 0.001})
    final = wait(svc, started["job"]["id"])
    assert final["status"] == "done"
    assert final["result"]["run_id"] == started["run_id"]
    assert final["progress"] > 0


def test_cancellation_keeps_no_number_and_changes_nothing(svc):
    s = svc.structure
    digest = structure_state_digest(s)
    n_history = len(svc.project.history._undo)
    started = svc.eam_run("md", settings={"steps": 20000, "temperature_K": 300})
    job_id = started["job"]["id"]
    deadline = time.time() + 30
    while svc.job_status(job_id)["progress"] <= 0 and time.time() < deadline:
        time.sleep(0.01)
    svc.cancel_job(job_id)
    final = wait(svc, job_id)
    assert final["status"] == "cancelled"
    stored = {k: v for k, v in svc.project.results.items()
              if k.startswith(f"eam::{started['run_id']}::")}
    assert list(stored) == [f"eam::{started['run_id']}::energy"]
    energy = stored[f"eam::{started['run_id']}::energy"]
    assert not energy.supported and energy.value is None
    assert energy.extra["status"] == "cancelled"
    assert structure_state_digest(s) == digest
    assert len(svc.project.history._undo) == n_history


def test_result_is_not_applied_when_the_structure_changes_during_the_run(svc):
    gate = threading.Event()
    real = svc.lab.eam.run

    def slow(*args, **kwargs):
        gate.wait(10)
        return real(*args, **kwargs)

    svc.lab.eam.run = slow
    started = svc.eam_run("relax", settings={"fmax_eV_A": 0.001})
    svc.edit("move", atom_id=int(svc.structure.ids[0]), delta_A=[0.05, 0, 0])
    moved = svc.structure.positions.copy()
    gate.set()
    final = wait(svc, started["job"]["id"])
    assert final["status"] == "done"
    result = final["result"]
    assert result["ok"] and result["applied"] is False
    assert "changed while" in result["apply_reason"]
    assert result["current"] is False
    np.testing.assert_array_equal(svc.structure.positions, moved)


def test_refusal_for_uncovered_elements_leaves_no_trace(svc):
    s = svc.structure
    s.substitute(int(s.ids[0]), "Au")
    n_results, n_log = len(svc.project.results), len(svc.project.history.log)
    out = svc.eam_run("energy", background=False)
    assert out["ok"] is False and "does not combine files" in out["error"]
    out = svc.eam_run("energy", potential="Cu-Zhou04", background=False)
    assert out["ok"] is False and "Au" in out["error"]
    assert len(svc.project.results) == n_results
    assert len(svc.project.history.log) == n_log


def test_invalid_settings_are_refused(svc):
    for task, settings in (("relax", {"fmax_eV_A": -1}), ("md", {"thermostat": "berendsen"}),
                           ("md", {"temperature_K": 0, "thermostat": "langevin"}),
                           ("energy", {"steps": 3}), ("anneal", {})):
        out = svc.eam_run(task, settings=settings, background=False)
        assert out["ok"] is False, (task, settings)


def test_save_reopen_and_reproduce(svc, tmp_path):
    first = svc.eam_run("relax", settings={"fmax_eV_A": 0.001}, background=False)
    path = svc.save_project(str(tmp_path / "cu.materia"))["path"]
    reopened = Service(Project.load(path), recovery_dir=str(tmp_path / "r2"))
    stored = reopened.eam_result(first["run_id"])
    assert stored["energy_eV"] == first["energy_eV"]
    assert stored["current"] is True and stored["potential_file_matches"] is True
    assert stored["potential"]["sha256"] == first["potential"]["sha256"]
    again = reopened.eam_run("energy", background=False)
    assert again["energy_eV"] == pytest.approx(first["energy_eV"], abs=1e-9)


def test_historical_results_keep_their_model_after_the_recommendation_changed(tmp_path):
    from materia.materials import load
    from materia.solvers.classical import lennard_jones

    service = Service(recovery_dir=str(tmp_path / "recovery"))
    handle = service.lab.materials.load("gold").bulk(repeat=(2, 2, 2))
    old = lennard_jones(material=load("gold")).single_point(handle.structure)
    service.project.add_result("energy", old["energy"])
    path = service.save_project(str(tmp_path / "au.materia"))["path"]
    reopened = Service(Project.load(path), recovery_dir=str(tmp_path / "r2"))
    reopened.eam_run("energy", background=False)
    kept = reopened.project.results["energy"]
    assert kept.provenance.model == "classical/lennard-jones"
    assert kept.value == old["energy"].value
    assert any("no many-body metallic screening" in a for a in kept.provenance.approximations)


def test_python_script_drives_eam(svc):
    code = (
        "run = eam.run(task='relax', fmax_eV_A=0.002)\n"
        "print(run['energy'].convergence.converged, run['energy'].unit)\n"
        "print(run['energy'].provenance.parameters['potential']['id'])\n"
        "print(eam.is_current())\n"
        "print(len(eam.catalog()))\n"
    )
    out = svc.runner.run(code)
    assert out.ok, out.error
    assert out.stdout.split() == ["True", "eV", "Cu-Zhou04", "True", "3"]


def test_http_routes(tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.lab.materials.load("tungsten").bulk(repeat=(2, 2, 2))
    port = find_free_port(9851)
    httpd = create_server(service=service, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def post(route, payload=None):
        request = urllib.request.Request(
            f"{base}/api/{route}", method="POST", data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            envelope = json.loads(response.read())
        assert envelope["ok"] is True
        return envelope["data"]

    try:
        assert post("eam/status")["structure"]["selected"] == "W-Zhou04"
        run = post("eam/run", {"task": "energy", "background": False})
        assert run["ok"] and run["energy_per_atom_eV"] == pytest.approx(-8.76, abs=0.01)
        assert post("eam/result", {"run_id": run["run_id"]})["energy_eV"] == run["energy_eV"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_shipped_eam_example_runs(svc):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "examples" / "scripts"
              / "10_eam_metals.py").read_text()
    out = svc.runner.run(source)
    assert out.ok, out.error or out.traceback
    assert "W bcc lattice constant" in out.stdout
    assert "refused:" in out.stdout


def overlap(service):
    s = service.structure
    positions = s.positions.copy()
    positions[1] = positions[0] + 1e-9
    s.positions = positions
    return s


def trace(service):
    return (len(service.project.results), len(service.project.history.log),
            len(service.project.history._undo), structure_state_digest(service.structure))


@pytest.mark.parametrize("task", ["energy", "relax", "md"])
def test_service_refuses_coincident_atoms_without_trace(svc, task):
    overlap(svc)
    before = trace(svc)
    out = svc.eam_run(task, background=False)
    assert out["ok"] is False and "collision tolerance" in out["error"]
    assert trace(svc) == before
    status = svc.eam_status()["structure"]
    assert "collision tolerance" in status["blocking"]


def test_generic_solver_panel_refuses_coincident_atoms_without_trace(svc):
    overlap(svc)
    before = trace(svc)
    for task in ("energy", "relax", "md"):
        out = svc.solve(task, "eam/Cu-Zhou04", background=False)
        assert out["supported"] is False and "collision tolerance" in out["reason"]
    n_results, n_log, n_undo, digest = trace(svc)
    assert (n_log, n_undo, digest) == before[1:]
    assert all(r.value is None for k, r in svc.project.results.items() if k == "energy")


def test_python_api_refuses_coincident_atoms(svc):
    from materia.python_api.api import ApiError

    overlap(svc)
    before = trace(svc)
    for task in ("energy", "relax", "md"):
        with pytest.raises(ApiError, match="collision tolerance"):
            svc.lab.eam.run(task=task)
    assert trace(svc) == before
    out = svc.runner.run("try:\n    eam.run(task='energy')\nexcept ApiError as exc:\n"
                         "    print('refused', 'collision tolerance' in str(exc))\n")
    assert out.ok and out.stdout.strip() == "refused True"


def test_http_reports_the_refusal(tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.lab.materials.load("copper").bulk(repeat=(2, 2, 2))
    overlap(service)
    port = find_free_port(9871)
    httpd = create_server(service=service, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/eam/run", method="POST",
            data=json.dumps({"task": "relax", "background": False}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read())["data"]
        assert data["ok"] is False and "collision tolerance" in data["error"]
        status_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/eam/status", method="POST", data=b"{}",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(status_request, timeout=60) as response:
            status = json.loads(response.read())["data"]
        assert "collision tolerance" in status["structure"]["blocking"]
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("task", ["energy", "relax", "md"])
def test_coincidence_arising_part_way_is_rolled_back(svc, task, monkeypatch):
    from materia.physics import eam as eam_module
    from materia.physics.neighbors import CoincidentAtoms
    from materia.physics.potentials import refuse_overlap

    real = eam_module.EAMPotential.evaluate
    calls = {"n": 0}

    def failing(self, structure):
        calls["n"] += 1
        if calls["n"] >= 3 or task == "energy":
            raise refuse_overlap(structure, CoincidentAtoms(0, 1, (0, 0, 0), 0.0))
        return real(self, structure)

    monkeypatch.setattr(eam_module.EAMPotential, "evaluate", failing)
    monkeypatch.setattr(eam_module.EAMPotential, "supports", lambda self, s: (True, ""))
    before = trace(svc)
    run_id_before = set(svc.project.results)
    out = svc.eam_run(task, settings={"fmax_eV_A": 1e-6} if task == "relax" else
                      ({"steps": 20} if task == "md" else {}), background=False)
    assert out["ok"] is False and out["status"] == "refused"
    assert "collision tolerance" in out["reason"]
    new = {k: r for k, r in svc.project.results.items() if k not in run_id_before}
    assert len(new) == 1 and all(r.value is None for r in new.values())
    n_results, n_log, n_undo, digest = trace(svc)
    assert (n_log, n_undo, digest) == before[1:]


def test_generic_solver_rolls_back_a_coincidence_arising_part_way(svc, monkeypatch):
    from materia.physics import eam as eam_module
    from materia.physics.neighbors import CoincidentAtoms
    from materia.physics.potentials import refuse_overlap

    real = eam_module.EAMPotential.evaluate
    calls = {"n": 0}

    def failing(self, structure):
        calls["n"] += 1
        if calls["n"] >= 4:
            raise refuse_overlap(structure, CoincidentAtoms(0, 1, (0, 0, 0), 0.0))
        return real(self, structure)

    monkeypatch.setattr(eam_module.EAMPotential, "evaluate", failing)
    before = trace(svc)
    out = svc.solve("relax", "eam/Cu-Zhou04", background=False, fmax=1e-6)
    assert out["supported"] is False and "put back" in out["reason"]
    assert trace(svc)[1:] == before[1:]


def test_refusal_leaves_saved_project_clean(svc, tmp_path):
    first = svc.eam_run("energy", background=False)
    overlap(svc)
    svc.eam_run("relax", background=False)
    path = svc.save_project(str(tmp_path / "refused.materia"))["path"]
    reopened = Service(Project.load(path), recovery_dir=str(tmp_path / "r2"))
    runs = reopened.lab.eam.runs()
    assert [r["run_id"] for r in runs] == [first["run_id"]]
    assert reopened.eam_result()["current"] is False
    assert "collision tolerance" in reopened.eam_status()["structure"]["blocking"]


def test_editing_atoms_onto_each_other_warns_and_undo_recovers(svc):
    s = svc.structure
    target = s.positions[0] - s.positions[1]
    svc.edit("move", atom_id=int(s.ids[1]), delta_A=target.tolist())
    assert any("collision tolerance" in w["text"] for w in svc.warnings)
    assert svc.eam_run("energy", background=False)["ok"] is False
    svc.undo()
    assert svc.eam_run("energy", background=False)["ok"] is True
