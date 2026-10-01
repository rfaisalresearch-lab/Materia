"""DFT relaxation through the service, Python API, history, persistence and HTTP.

GPAW is replaced by the deterministic worker double in
``tests/support/fake_gpaw_relax.py``; ``tests/validation/test_dft_relaxation_live.py``
runs the real code when GPAW is installed.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import zipfile

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.desktop_ui.server import create_server, find_free_port
from materia.desktop_ui.service import Service
from materia.experiments.dft import records
from materia.project_format.arrays import ArrayStoreError
from materia.project_format.project import Project
from materia.python_api.api import ApiError, DFTRelaxationFailed
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_relax import FakeRelaxWorker
from tests.support.fake_gpaw_relax import environment as fake_environment

RELAX = {"forces_tol_eV_A": 0.01, "grid_spacing_A": 0.22}
BULK = {"mode": "variable-cell", "cutoff_eV": 500.0, "kpoints": [2, 2, 2],
        "occupations": "fixed", "smearing_eV": 0.0, "forces_tol_eV_A": 0.005}


def molecule(distance=0.8, box=8.0):
    c = box / 2
    s = Structure(np.array([1, 1]),
                  np.array([[c, c, c - distance / 2], [c, c, c + distance / 2]]),
                  Cell(np.eye(3) * box, (False, False, False)))
    s.roles[:] = ["adsorbate", "adsorbate"]
    s.info["label"] = "hydrogen molecule"
    return s


def silicon(a=5.6):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


@pytest.fixture
def gpaw(tmp_path, monkeypatch):
    import materia.solvers.gpaw_driver as package
    import materia.solvers.gpaw_driver.environment as module

    env = fake_environment(tmp_path)
    monkeypatch.setattr(package, "discover", lambda refresh=False: env)
    monkeypatch.setattr(module, "discover", lambda refresh=False: env)
    worker = FakeRelaxWorker()
    monkeypatch.setattr(runner, "run_job", worker)
    return worker


@pytest.fixture
def svc(tmp_path, gpaw):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(molecule())
    yield service
    service.shutdown(grace_s=2.0)


def snapshot(project):
    s = project.structure
    return (sorted(project.results), sorted(project.arrays), len(project.history.log),
            len(project.history._undo), s.positions.copy(), s.cell.matrix.copy(),
            s.forces.copy(), list(s.roles), dict(s.info))


def unchanged(before, after):
    return (before[:4] == after[:4] and np.array_equal(before[4], after[4])
            and np.array_equal(before[5], after[5])
            and np.array_equal(before[6], after[6], equal_nan=True)
            and before[7:] == after[7:])


def wait(service, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.job_status(job_id)
        if status["status"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_converged_relaxation_is_one_undoable_change(svc):
    s = svc.structure
    start = s.positions.copy()
    log = len(svc.project.history.log)
    out = svc.dft_relax(RELAX, background=False)
    assert out["ok"] and out["status"] == "converged", out
    assert out["run_state"] == "current" and out["applied"] and not out["applicable"]
    assert len(svc.project.history.log) == log + 1
    entry = svc.project.history.log[-1]
    assert entry.undoable and entry.operation == "solver.dft.relax.fixed-cell"
    assert entry.parameters["run_id"] == out["run_id"]
    assert np.allclose(s.positions - start, [[0, 0, 0.05], [0, 0, 0.05]])
    assert np.allclose(s.forces, 0.001)
    assert out["app_state"] is not None
    svc.undo()
    assert np.array_equal(s.positions, start) and np.isnan(s.forces).all()
    after_undo = svc.dft_relax_result(out["run_id"])
    assert after_undo["run_state"] == "stale" and after_undo["applicable"]
    svc.redo()
    again = svc.dft_relax_result(out["run_id"])
    assert again["run_state"] == "current" and again["applied"]
    assert list(s.roles) == ["adsorbate", "adsorbate"]
    assert s.info["label"] == "hydrogen molecule"


def test_variable_cell_apply_restores_cell_on_undo(tmp_path, gpaw):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    s = silicon()
    s.roles[:] = ["bulk", "bulk"]
    service.project.add_structure(s)
    cell = s.cell.matrix.copy()
    out = service.dft_relax(BULK, background=False)
    assert out["ok"] and out["mode"] == "variable-cell", out
    assert np.allclose(s.cell.matrix, cell * 0.98)
    assert out["cell"]["volume_change_percent"] == pytest.approx(100 * (0.98 ** 3 - 1))
    assert out["stress_eV_A3"] is not None
    service.undo()
    assert np.array_equal(s.cell.matrix, cell)
    assert s.cell.pbc == (True, True, True)
    service.redo()
    assert np.allclose(s.cell.matrix, cell * 0.98)
    service.shutdown(grace_s=1.0)


def test_apply_later_and_only_once(svc):
    log = len(svc.project.history.log)
    out = svc.dft_relax(RELAX, background=False, apply=False)
    assert out["ok"] and not out["applied"] and out["applicable"]
    assert out["run_state"] == "stale"
    assert len(svc.project.history.log) == log + 1
    assert svc.project.history.log[-1].undoable is False
    applied = svc.dft_relax_apply(out["run_id"])
    assert applied["ok"]
    assert len(svc.project.history.log) == log + 2
    assert svc.project.history.log[-1].undoable
    again = svc.dft_relax_apply(out["run_id"])
    assert again["ok"] is False and "already applied" in again["error"]
    assert len(svc.project.history.log) == log + 2


def test_stale_geometry_is_refused_without_trace(svc):
    out = svc.dft_relax(RELAX, background=False, apply=False)
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + 0.1
    svc.project.record_change(s, before, svc.project.selection, "nudge", "edit.move")
    state = snapshot(svc.project)
    refused = svc.dft_relax_apply(out["run_id"])
    assert refused["ok"] is False and "changed after the relaxation started" in refused["error"]
    assert unchanged(state, snapshot(svc.project))
    result = svc.dft_relax_result(out["run_id"])
    assert result["run_state"] == "stale" and not result["applicable"]
    assert "atom positions changed" in result["state_reason"]


def test_structure_changed_during_the_run_is_not_applied(svc, gpaw):
    started = threading.Event()
    gate = threading.Event()
    original = gpaw._finished

    def slow(job, progress):
        started.set()
        gate.wait(5)
        return original(job, progress)

    gpaw._finished = slow
    submitted = svc.dft_relax(RELAX, background=True)
    assert started.wait(5)
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + 0.2
    svc.project.record_change(s, before, svc.project.selection, "nudge", "edit.move")
    moved = s.positions.copy()
    gate.set()
    status = wait(svc, submitted["job"]["id"])
    result = status["result"]
    assert result["ok"] and not result["applied"]
    assert "changed after the relaxation started" in result["apply_reason"]
    assert np.array_equal(s.positions, moved)


def test_detached_structure(svc):
    loose = molecule()
    run = svc.lab.dft.relax(loose, **RELAX)
    run_id = run["relaxation"].extra["run_id"]
    assert svc.lab.dft.relax_state(run_id)["state"] == "detached"
    with pytest.raises(ApiError, match="not held"):
        svc.lab.dft.relax_apply(run_id)
    assert svc.lab.dft.relax_state(run_id, target=loose)["state"] == "stale"


def test_unconverged_relaxation_is_kept_but_never_applied(svc, gpaw):
    gpaw.mode = "max_steps"
    start = svc.structure.positions.copy()
    out = svc.dft_relax({**RELAX, "max_steps": 3}, background=False)
    assert out["ok"] and out["status"] == "not-converged"
    assert out["provenance"]["origin"] == "estimated"
    assert not out["applicable"] and not out["applied"]
    assert np.array_equal(svc.structure.positions, start)
    assert svc.project.history.log[-1].undoable is False
    refused = svc.dft_relax_apply(out["run_id"])
    assert refused["ok"] is False and "did not meet its criteria" in refused["error"]


@pytest.mark.parametrize("mode", ["failed", "timeout", "cancelled", "echo_symmetry",
                                  "lose_atom", "nan_energy"])
def test_failures_are_transactional(svc, gpaw, mode):
    gpaw.mode = mode
    before = snapshot(svc.project)
    out = svc.dft_relax(RELAX, background=False, timeout_s=1.0)
    assert out["ok"] is False
    assert out["status"] in ("failed", "cancelled")
    assert unchanged(before, snapshot(svc.project))
    assert svc.dft_relax_result()["ok"] is False
    with pytest.raises(DFTRelaxationFailed) as info:
        svc.lab.dft.relax(**RELAX)
    assert info.value.status in ("failed", "cancelled")
    assert unchanged(before, snapshot(svc.project))


def test_refusal_starts_nothing(svc, gpaw):
    before = snapshot(svc.project)
    out = svc.dft_relax({**RELAX, "mode": "variable-cell"}, background=False)
    assert out["ok"] is False and out["refused"] and "mode" in out["by_field"]
    bad = svc.dft_relax({**RELAX, "optimizer": "Nope"}, background=False)
    assert bad["ok"] is False and bad["refused"]
    assert gpaw.jobs == []
    assert unchanged(before, snapshot(svc.project))


def test_job_cancellation_stores_nothing(svc, gpaw):
    gpaw.mode = "wait_for_cancel"
    gpaw.started = threading.Event()
    before = snapshot(svc.project)
    submitted = svc.dft_relax(RELAX, background=True)
    assert gpaw.started.wait(5)
    svc.cancel_job(submitted["job"]["id"])
    status = wait(svc, submitted["job"]["id"])
    assert status["status"] == "cancelled"
    assert status["result"]["status"] == "cancelled"
    assert unchanged(before, snapshot(svc.project))


def test_replacing_the_project_cancels_and_writes_nowhere(svc, gpaw):
    gpaw.mode = "wait_for_cancel"
    gpaw.started = threading.Event()
    old = svc.project
    submitted = svc.dft_relax(RELAX, background=True)
    assert gpaw.started.wait(5)
    svc.new_project("fresh")
    status = wait(svc, submitted["job"]["id"])
    assert status["status"] == "cancelled"
    assert not any(k.startswith(records.RELAX_PREFIX) for k in old.results)
    assert not any(k.startswith(records.RELAX_PREFIX) for k in svc.project.results)


def test_failed_apply_rolls_back(svc, monkeypatch):
    out = svc.dft_relax(RELAX, background=False, apply=False)
    before = snapshot(svc.project)

    def broken(*args, **kwargs):
        raise RuntimeError("history is full")

    monkeypatch.setattr(svc.project, "record_change", broken)
    with pytest.raises(RuntimeError):
        records.apply_relaxation(svc.project, out["run_id"])
    assert unchanged(before, snapshot(svc.project))


def test_persistence_round_trip(svc, tmp_path):
    out = svc.dft_relax(RELAX, background=False)
    run_id = out["run_id"]
    path = svc.save_project(str(tmp_path / "relaxed"))["path"]
    loaded = Project.load(path)
    for name in ("final_positions", "initial_positions", "step_positions", "final_cell"):
        key = records.relax_key(run_id, name)
        assert np.array_equal(loaded.arrays[key].data, svc.project.arrays[key].data)
    other = Service(loaded, recovery_dir=str(tmp_path / "r2"))
    assert other.lab.dft.relax_spec_of(run_id).digest == svc.lab.dft.relax_spec_of(run_id).digest
    result = other.dft_relax_result(run_id)
    assert result["run_state"] == "current" and result["applied"]
    assert result["history"] == out["history"]
    assert loaded.history.log[-1].operation == "solver.dft.relax.fixed-cell"
    moved = other.structure.copy()
    moved.positions = moved.positions + 0.1
    other.structure.restore_from(moved)
    assert other.dft_relax_result(run_id)["run_state"] == "stale"
    other.shutdown(grace_s=1.0)


def test_corrupted_relaxation_array_is_refused(svc, tmp_path):
    out = svc.dft_relax(RELAX, background=False)
    path = svc.save_project(str(tmp_path / "good"))["path"]
    damaged = tmp_path / "bad.materia"
    target = f"arrays/{records.relax_key(out['run_id'], 'final_positions')}/chunk-00000.npy"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "w") as sink:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == target:
                data = data[:-8] + b"\x00" * 8
            sink.writestr(item, data)
    with pytest.raises(ArrayStoreError, match="SHA-256"):
        Project.load(str(damaged))


def test_status_lists_relaxations_and_fields(svc):
    svc.dft_relax(RELAX, background=False)
    status = svc.dft_status()
    assert status["relaxations"][0]["state"] == "current"
    names = {f["name"] for f in status["relaxation_fields"]}
    assert {"mode", "fmax_eV_A", "stress_tol_eV_A3", "symmetry"} <= names
    assert status["structure"]["fixed_atoms"] == 0


def test_http_routes(svc):
    port = find_free_port(9951)
    httpd = create_server(svc, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def post(route, payload=None):
        request = urllib.request.Request(
            f"{base}/api/{route}", method="POST", data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    try:
        spec = post("dft/relax/spec", {"variables": RELAX})["data"]
        assert spec["ok"] and spec["spec"]["relaxation_settings"]["optimizer"] == "BFGS"
        refused = post("dft/relax/spec", {"variables": {**RELAX, "mode": "variable-cell"}})
        assert refused["data"]["ok"] is False and "mode" in refused["data"]["by_field"]
        run = post("dft/relax/run", {"variables": RELAX, "background": False,
                                     "apply": False})["data"]
        assert run["ok"] and run["applicable"]
        runs = post("dft/relax/runs")["data"]["runs"]
        assert runs[0]["run_id"] == run["run_id"] and runs[0]["state"] == "stale"
        result = post("dft/relax/result", {"run_id": run["run_id"]})["data"]
        assert result["optimizer_steps"] == 4
        applied = post("dft/relax/apply", {"run_id": run["run_id"]})["data"]
        assert applied["ok"]
        again = post("dft/relax/apply", {"run_id": run["run_id"]})["data"]
        assert again["ok"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_script_namespace(svc):
    out = svc.runner.run(
        "spec = dft.relax_spec(forces_tol_eV_A=0.01, grid_spacing_A=0.22)\n"
        "assert dft.relax_check(spec)['ok']\n"
        "run = dft.relax(spec=spec)\n"
        "print(run['relaxation'].value['optimizer_steps'], dft.relax_state()['state'])\n")
    assert out.ok, out.error or out.traceback
    assert out.stdout.strip() == "4 current"


def test_shipped_relaxation_example_runs(svc):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "examples" / "scripts"
              / "14_dft_relaxation.py").read_text()
    out = svc.runner.run(source)
    assert out.ok, out.error or out.traceback
    assert "refusals: none" in out.stdout
    assert "state: current" in out.stdout
    assert "after undo: stale | can apply again: True" in out.stdout
    assert "lattice constant" in out.stdout
