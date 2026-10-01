"""LAMMPS through the Python API, the service, persistence and the HTTP layer.

Every run here goes through the deterministic fake process from
``tests/support/fake_lammps.py``.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import zipfile

import numpy as np
import pytest

from materia.desktop_ui.server import create_server, find_free_port
from materia.desktop_ui.service import Service
from materia.project_format.arrays import ArrayStoreError
from materia.project_format.project import Project
from materia.python_api.api import ApiError, LAMMPSRunFailed, structure_state_digest


@pytest.fixture
def svc(tmp_path, fake_lammps):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    handle = service.lab.materials.load("copper").bulk(repeat=(2, 2, 2))
    rng = np.random.default_rng(5)
    handle.structure.positions = handle.structure.positions + rng.normal(
        0, 0.03, handle.structure.positions.shape)
    yield service
    service.shutdown(grace_s=2.0)


def snapshot(project):
    return (sorted(project.results), sorted(project.arrays), len(project.history.log),
            project.history.can_undo, structure_state_digest(project.structure),
            project.structure.forces.copy(), project.structure.velocities.copy())


def same(a, b):
    return a[:5] == b[:5] and np.array_equal(a[5], b[5], equal_nan=True) and \
        np.array_equal(a[6], b[6])


def wait(service, job_id, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.job_status(job_id)
        if status["status"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_energy_run_attaches_forces_as_one_undoable_change(svc):
    lab = svc.lab
    s = svc.structure
    log_before = len(lab.project.history.log)
    run = lab.lammps.run(task="energy")
    assert {"run", "energy", "forces", "stress", "thermo"} <= set(run)
    assert len(lab.project.history.log) == log_before + 1
    entry = lab.project.history.log[-1]
    assert entry.undoable and entry.operation == "solver.lammps.energy"
    assert entry.parameters["run_id"] == run["run"].extra["run_id"]
    assert np.allclose(s.forces, run["forces"].value)
    state = lab.lammps.state()
    assert state["state"] == "current" and state["applied"] and not state["applicable"]
    lab.undo()
    assert np.isnan(s.forces).all()
    state = lab.lammps.state()
    assert state["state"] == "current" and not state["applied"] and state["applicable"]
    lab.redo()
    assert lab.lammps.state()["applied"]


def test_relaxation_is_applied_and_undo_restores_the_geometry(svc):
    lab = svc.lab
    s = svc.structure
    before = s.positions.copy()
    run = lab.lammps.run(task="relax", ftol_eV_A=0.02)
    assert run["energy"].convergence.converged
    assert not np.allclose(s.positions, before)
    assert lab.lammps.state()["state"] == "current"
    lab.undo()
    assert np.array_equal(s.positions, before)
    state = lab.lammps.state()
    assert state["state"] == "stale" and state["applicable"]
    assert "apply the run" in state["reason"]
    lab.lammps.apply()
    assert lab.lammps.state()["state"] == "current"
    with pytest.raises(ApiError, match="already applied"):
        lab.lammps.apply()


def test_unconverged_relaxation_is_stored_but_never_applied(svc, monkeypatch):
    monkeypatch.setenv("FAKE_LAMMPS_MODE", "maxiter")
    lab = svc.lab
    before = svc.structure.positions.copy()
    log_before = len(lab.project.history.log)
    run = lab.lammps.run(task="relax", ftol_eV_A=1e-6)
    assert run["energy"].provenance.origin.value == "estimated"
    assert np.array_equal(svc.structure.positions, before)
    assert len(lab.project.history.log) == log_before + 1
    assert lab.project.history.log[-1].undoable is False
    assert "did not reach" in run["energy"].extra["apply_reason"]
    assert not lab.lammps.state()["applicable"]
    with pytest.raises(ApiError, match="force tolerance"):
        lab.lammps.apply()


def test_dynamics_applies_positions_and_velocities(svc):
    lab = svc.lab
    run = lab.lammps.run(task="md", steps=20, sample_every=10, initial_velocities="create",
                         temperature_K=300.0, seed=9)
    final = lab.lammps.array(name="final_velocities").data
    assert np.allclose(svc.structure.velocities, final)
    assert np.allclose(svc.structure.positions, lab.lammps.array(name="positions").data[-1])
    assert run["trajectory"].value["frames"] == 3


def test_stale_geometry_is_refused_and_leaves_no_history(svc):
    lab = svc.lab
    lab.lammps.run(task="relax", ftol_eV_A=0.02, apply=False)
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + np.array([0.1, 0.0, 0.0])
    lab.project.record_change(s, before, lab.project.selection, "move", "edit.move")
    log_before = len(lab.project.history.log)
    state = lab.lammps.state()
    assert state["state"] == "stale" and not state["applicable"]
    assert "changed since the run" in state["reason"]
    with pytest.raises(ApiError, match="changed after the run started"):
        lab.lammps.apply()
    assert len(lab.project.history.log) == log_before
    lab.undo()
    assert lab.lammps.state()["applicable"]


def test_detached_structure(svc):
    lab = svc.lab
    loose = svc.structure.copy()
    lab.lammps.run(loose, task="energy")
    state = lab.lammps.state()
    assert state["state"] == "detached"
    with pytest.raises(ApiError, match="not held"):
        lab.lammps.apply()


@pytest.mark.parametrize("mode", ["exit1", "truncate", "lose_atom", "change_id", "nan"])
def test_failure_is_transactional(svc, monkeypatch, mode):
    before = snapshot(svc.project)
    monkeypatch.setenv("FAKE_LAMMPS_MODE", mode)
    with pytest.raises(LAMMPSRunFailed) as info:
        svc.lab.lammps.run(task="energy")
    assert info.value.status == "failed"
    assert same(snapshot(svc.project), before)


def test_refusal_is_transactional(svc):
    before = snapshot(svc.project)
    with pytest.raises(ApiError, match="refused"):
        svc.lab.lammps.run(task="md", steps=10, sample_every=5, pressure_bar=1.0)
    with pytest.raises(ApiError, match="Unknown setting"):
        svc.lab.lammps.run(task="energy", steps=3)
    assert same(snapshot(svc.project), before)


def test_missing_lammps_is_refused_without_trace(tmp_path, no_lammps):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.lab.materials.load("copper").bulk(repeat=(2, 2, 2))
    before = snapshot(service.project)
    with pytest.raises(ApiError, match="conda install"):
        service.lab.lammps.run(task="energy")
    out = service.lammps_run("energy", background=False)
    assert out["ok"] is False and out["status"] == "refused"
    assert "conda install" in out["error"]
    assert same(snapshot(service.project), before)
    status = service.lammps_status()
    assert status["environment"]["available"] is False
    assert any("conda install" in b for b in status["structure"]["blocking"])
    external = {e["name"]: e for e in service.solvers()["external"]}
    assert external["external:lammps"]["available"] is False
    assert external["external:lammps"]["driven"] is True


def test_project_round_trip_keeps_runs_arrays_and_state(svc, tmp_path):
    lab = svc.lab
    run = lab.lammps.run(task="md", steps=20, sample_every=5, initial_velocities="create",
                         temperature_K=300.0, seed=2)
    run_id = run["run"].extra["run_id"]
    path = svc.save_project(str(tmp_path / "lammps.materia"))["path"]
    loaded = Project.load(path)
    assert f"lammps::{run_id}::positions" in loaded.arrays
    assert np.array_equal(loaded.arrays[f"lammps::{run_id}::positions"].data,
                          lab.project.arrays[f"lammps::{run_id}::positions"].data)
    other = Service(loaded, recovery_dir=str(tmp_path / "r2"))
    assert other.lab.lammps.spec_of(run_id).digest == lab.lammps.spec_of(run_id).digest
    assert other.lab.lammps.state(run_id)["state"] == "current"
    forces = other.lab.lammps.result(run_id, "forces")
    assert np.allclose(np.asarray(forces.value), run["forces"].value)
    frame = other.eam_trajectory(run_id, 2)
    assert frame["ok"] and frame["backend"] == "lammps" and frame["frame"] == 2
    result = other.lammps_result(run_id)
    assert result["ok"] and result["trajectory_available"] and result["current"]
    other.shutdown(grace_s=1.0)


def test_corrupted_trajectory_chunk_is_refused_on_load(svc, tmp_path):
    run = svc.lab.lammps.run(task="md", steps=10, sample_every=5)
    run_id = run["run"].extra["run_id"]
    path = svc.save_project(str(tmp_path / "good.materia"))["path"]
    bad = tmp_path / "bad.materia"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(bad, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == f"arrays/lammps::{run_id}::positions/chunk-00000.npy":
                data = data[:-8] + b"\x00" * 8
            target.writestr(item, data)
    with pytest.raises(ArrayStoreError, match="SHA-256"):
        Project.load(str(bad))


def test_script_namespace_exposes_lammps(svc):
    out = svc.runner.run(
        "spec = lammps.spec(task='energy')\n"
        "assert lammps.check(spec)['ok']\n"
        "files = lammps.native_input(spec)\n"
        "assert 'pair_style eam/alloy' in files['in.lammps']\n"
        "run = lammps.run(spec=spec)\n"
        "print(round(run['energy'].value, 6), lammps.state()['state'])\n")
    assert out.ok, out.error or out.traceback
    assert out.stdout.strip().endswith("current")


def example_source():
    from pathlib import Path

    return (Path(__file__).resolve().parents[2] / "examples" / "scripts"
            / "13_lammps_backend.py").read_text()


def test_shipped_lammps_example_runs(svc):
    out = svc.runner.run(example_source())
    assert out.ok, out.error or out.traceback
    assert "LAMMPS available: True" in out.stdout
    assert "refusals: {}" in out.stdout and "state: current" in out.stdout


def test_shipped_lammps_example_reports_the_install_path_without_lammps(tmp_path, no_lammps):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    out = service.runner.run(example_source())
    assert out.ok, out.error or out.traceback
    assert "LAMMPS available: False" in out.stdout
    assert "conda install -c conda-forge lammps" in out.stdout


def test_service_background_run_progress_and_payload(svc):
    submitted = svc.lammps_run("md", settings={"steps": 20, "sample_every": 10,
                                                "initial_velocities": "create",
                                                "temperature_K": 300.0, "seed": 3})
    status = wait(svc, submitted["job"]["id"])
    assert status["status"] == "done", status
    result = status["result"]
    assert result["ok"] and result["backend"] == "lammps" and result["applied"]
    assert result["lammps"]["version"] == "29 Aug 2024 - Update 1"
    assert result["frames"] == 3 and result["trajectory_available"]
    assert result["inputs"]["in.lammps"]["sha256"]
    assert "Applied" in result["apply_reason"]
    assert result["state"] is not None


def test_service_cancellation_keeps_nothing(svc, monkeypatch):
    monkeypatch.setenv("FAKE_LAMMPS_MODE", "sleep")
    before = snapshot(svc.project)
    submitted = svc.lammps_run("md", settings={"steps": 400, "sample_every": 10})
    deadline = time.time() + 30
    while time.time() < deadline:
        job = svc.job_status(submitted["job"]["id"], include_result=False)
        if "LAMMPS step" in (job.get("message") or ""):
            break
        time.sleep(0.02)
    svc.cancel_job(submitted["job"]["id"])
    status = wait(svc, submitted["job"]["id"])
    assert status["status"] == "cancelled"
    assert status["result"]["status"] == "cancelled"
    assert same(snapshot(svc.project), before)
    assert svc.lammps_result()["ok"] is False


def test_structure_changed_during_the_run_is_not_applied(svc, monkeypatch):
    monkeypatch.setenv("FAKE_LAMMPS_MODE", "sleep")
    submitted = svc.lammps_run("md", settings={"steps": 30, "sample_every": 10})
    time.sleep(0.3)
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + 0.05
    svc.project.record_change(s, before, svc.project.selection, "nudge", "edit.move")
    moved = s.positions.copy()
    status = wait(svc, submitted["job"]["id"])
    result = status["result"]
    assert result["ok"] and not result["applied"]
    assert "changed after the run started" in result["apply_reason"]
    assert np.array_equal(s.positions, moved)
    assert result["run_state"] == "stale"
    assert result["state"] is not None


def test_replacing_the_project_cancels_the_run(svc, monkeypatch):
    monkeypatch.setenv("FAKE_LAMMPS_MODE", "sleep")
    old = svc.project
    submitted = svc.lammps_run("md", settings={"steps": 400, "sample_every": 10})
    time.sleep(0.3)
    svc.new_project("fresh")
    status = wait(svc, submitted["job"]["id"])
    assert status["status"] == "cancelled"
    assert not any(k.startswith("lammps::") for k in old.results)
    assert not any(k.startswith("lammps::") for k in svc.project.results)


def test_http_routes(svc):
    port = find_free_port(9931)
    httpd = create_server(svc, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def post(route, payload=None):
        request = urllib.request.Request(
            f"{base}/api/{route}", method="POST", data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    try:
        status = post("lammps/status")["data"]
        assert status["environment"]["available"] and status["structure"]["blocking"] == []
        spec = post("lammps/spec", {"task": "md", "settings": {"steps": 10,
                                                                 "sample_every": 5}})["data"]
        assert spec["ok"] and "run 10" in spec["native_input"]["in.lammps"]
        assert spec["spec"]["settings"]["steps"] == 10
        refused = post("lammps/spec", {"task": "md", "settings": {"steps": 10,
                                                                   "sample_every": 3}})["data"]
        assert refused["ok"] is False and "sample_every" in refused["spec"]["check"]["blocking"]
        run = post("lammps/run", {"task": "md", "background": False, "apply": False,
                                  "settings": {"steps": 10, "sample_every": 5,
                                               "initial_velocities": "create",
                                               "temperature_K": 300, "seed": 1}})["data"]
        assert run["ok"] and not run["applied"] and run["applicable"]
        run_id = run["run_id"]
        runs = post("lammps/runs")["data"]["runs"]
        assert runs[0]["run_id"] == run_id and runs[0]["state"] == "stale"
        meta = post("eam/trajectory", {"run_id": run_id})["data"]
        assert meta["frames"] == 3 and meta["backend"] == "lammps"
        frame = post("lammps/trajectory", {"run_id": run_id, "frame": 1})["data"]
        assert len(frame["positions"]) == 3 * len(svc.structure)
        assert frame["ids"] == [int(i) for i in svc.structure.ids]
        applied = post("lammps/apply", {"run_id": run_id})["data"]
        assert applied["ok"]
        again = post("lammps/apply", {"run_id": run_id})["data"]
        assert again["ok"] is False and "already applied" in again["error"]
        result = post("lammps/result", {"run_id": run_id})["data"]
        assert result["current"] and result["applied"]
        bad = post("lammps/run", {"task": "md", "background": False,
                                  "settings": {"steps": 10, "sample_every": 5,
                                               "pressure_bar": 5}})["data"]
        assert bad["ok"] is False and bad["status"] == "refused"
    finally:
        httpd.shutdown()
        httpd.server_close()
