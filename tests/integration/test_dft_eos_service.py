"""Equation of state through the service, Python API, history, persistence and HTTP.

GPAW is replaced by the deterministic double in ``tests/support/fake_gpaw_eos.py``;
``tests/validation/test_dft_eos_live.py`` exercises real GPAW.
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
from materia.project_format.arrays import ArrayStoreError, StoredArray
from materia.project_format.project import Project
from materia.python_api.api import ApiError, DFTEOSFailed
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import environment as fake_environment
from tests.support.fake_gpaw_eos import B0_GPA, V0_PER_ATOM, FakeEOSGPAW

GS = {"xc": "PBE", "cutoff_eV": 400.0, "kpoints": [4, 4, 4], "occupations": "fermi-dirac",
      "smearing_eV": 0.01}


def silicon(scale=1.0):
    a = (V0_PER_ATOM * 8.0 * scale) ** (1.0 / 3.0)
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
    fake = FakeEOSGPAW()
    monkeypatch.setattr(runner, "run_job", fake)
    return fake


@pytest.fixture
def svc(tmp_path, gpaw):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(silicon(1.03))
    yield service
    service.shutdown(grace_s=2.0)


def snapshot(project):
    s = project.structure
    return (sorted(project.results), sorted(project.arrays), len(project.history.log),
            len(project.history._undo), s.positions.copy(), s.cell.matrix.copy())


def unchanged(a, b):
    return a[:4] == b[:4] and np.array_equal(a[4], b[4]) and np.array_equal(a[5], b[5])


def wait(service, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.job_status(job_id)
        if status["status"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_equation_of_state_is_stored_and_changes_nothing(svc):
    before = snapshot(svc.project)
    out = svc.dft_eos_run(GS, background=False)
    assert out["ok"] and out["status"] == "complete", out
    after = snapshot(svc.project)
    assert np.array_equal(after[4], before[4]) and after[3] == before[3]
    assert after[2] == before[2] + 1
    entry = svc.project.history.log[-1]
    assert entry.operation == "dft.eos" and entry.undoable is False
    assert out["V0_A3_per_atom"] == pytest.approx(V0_PER_ATOM, rel=1e-7)
    assert out["B0_GPa"] == pytest.approx(B0_GPA, rel=1e-6)
    assert out["linear_scale"] == pytest.approx(1.03 ** (-1.0 / 3.0), rel=1e-7)
    assert out["run_state"] == "current" and out["applicable"] is True
    assert len(out["points"]) == 7 and len(out["fit_curve"]["volumes_A3"]) == 121
    assert out["provenance"]["model"] == "external:gpaw/equation-of-state"
    status = svc.dft_status()
    assert status["eos_runs"][0]["run_id"] == out["run_id"]
    assert {f["name"] for f in status["eos_fields"]} >= {"volume_min_scale", "n_points"}


def test_apply_is_one_undoable_change_and_tracks_state(svc):
    out = svc.dft_eos_run(GS, background=False)
    undo_before = len(svc.project.history._undo)
    applied = svc.dft_eos_apply(out["run_id"])
    assert applied["ok"], applied
    volume = abs(np.linalg.det(svc.structure.cell.matrix)) / len(svc.structure)
    assert volume == pytest.approx(V0_PER_ATOM, rel=1e-7)
    assert len(svc.project.history._undo) == undo_before + 1
    assert svc.dft_eos_result(out["run_id"])["run_state"] == "applied"
    again = svc.dft_eos_apply(out["run_id"])
    assert again["ok"] is False and "already applied" in again["error"]
    svc.undo()
    assert svc.dft_eos_result(out["run_id"])["run_state"] == "current"
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + 0.05
    svc.project.record_change(s, before, svc.project.selection, "nudge", "edit.move")
    stale = svc.dft_eos_result(out["run_id"])
    assert stale["run_state"] == "stale" and stale["applicable"] is False
    refused = svc.dft_eos_apply(out["run_id"])
    assert refused["ok"] is False and "changed after" in refused["error"]


def test_from_a_stored_ground_state(svc):
    ground = svc.dft_run({**GS, "observables": ["energy", "forces", "stress"]},
                         background=False)
    assert ground["ok"], ground
    source = {"kind": "ground-state", "run_id": ground["run_id"]}
    out = svc.dft_eos_run({"n_points": 5, "volume_min_scale": 0.92,
                           "volume_max_scale": 1.04}, source, background=False)
    assert out["ok"], out
    assert out["source"]["run_id"] == ground["run_id"]
    refused = svc.dft_eos_spec({"cutoff_eV": 500.0}, source)
    assert refused["ok"] is False and "keeps that run's geometry" in refused["error"]


def test_detached_structure_and_python_api(svc):
    loose = silicon(1.0)
    result = svc.lab.dft.eos(loose, **GS)
    run_id = result.extra["run_id"]
    assert svc.lab.dft.eos_state(run_id)["state"] == "detached"
    assert svc.lab.dft.eos_state(run_id, target=loose)["state"] == "current"
    assert svc.lab.dft.eos_array(run_id, "energies").data.shape == (7,)
    assert svc.lab.dft.eos_spec_of(run_id).n_points == 7
    with pytest.raises(ApiError, match="not held"):
        svc.lab.dft.eos_apply(run_id)


@pytest.mark.parametrize("mode", ["failed", "not_converged", "echo", "shifted", "noisy",
                                  "cancelled"])
def test_failures_are_transactional(svc, gpaw, mode):
    gpaw.mode = mode
    before = snapshot(svc.project)
    out = svc.dft_eos_run(GS, background=False)
    assert out["ok"] is False and out["status"] in ("failed", "cancelled")
    assert unchanged(before, snapshot(svc.project))
    gpaw.jobs.clear()
    with pytest.raises(DFTEOSFailed):
        svc.lab.dft.eos(**GS)
    assert unchanged(before, snapshot(svc.project))
    assert svc.dft_eos_result()["ok"] is False


def test_refusal_starts_nothing(svc, gpaw):
    before = snapshot(svc.project)
    out = svc.dft_eos_run({**GS, "representation": "fd"}, background=False)
    assert out["ok"] is False and out["refused"] and "representation" in out["by_field"]
    with pytest.raises(ApiError, match="refused"):
        svc.lab.dft.eos(**GS, n_points=3)
    assert gpaw.jobs == []
    assert unchanged(before, snapshot(svc.project))


def test_job_cancellation(svc, gpaw):
    started = threading.Event()
    original = gpaw.__call__

    def slow(job, environment, **kwargs):
        started.set()
        deadline = time.time() + 10
        cancelled = kwargs.get("cancelled")
        while time.time() < deadline and not (cancelled and cancelled()):
            time.sleep(0.01)
        return runner.GPAWRun(status=runner.STATUS_CANCELLED, result={})

    import materia.solvers.gpaw_driver.runner as module
    module.run_job = slow
    try:
        before = snapshot(svc.project)
        submitted = svc.dft_eos_run(GS, background=True)
        assert started.wait(5)
        svc.cancel_job(submitted["job"]["id"])
        status = wait(svc, submitted["job"]["id"])
        assert status["status"] == "cancelled" and status["result"]["status"] == "cancelled"
        assert unchanged(before, snapshot(svc.project))
    finally:
        module.run_job = original


def test_persistence_round_trip_and_corruption(svc, tmp_path):
    out = svc.dft_eos_run(GS, background=False)
    run_id = out["run_id"]
    path = svc.save_project(str(tmp_path / "eos"))["path"]
    loaded = Project.load(path)
    for name in out["array_sha256"]:
        key = records.eos_key(run_id, name)
        assert np.array_equal(loaded.arrays[key].data, svc.project.arrays[key].data)
    other = Service(loaded, recovery_dir=str(tmp_path / "r2"))
    again = other.dft_eos_result(run_id)
    assert again["run_state"] == "current"
    for name in ("V0_A3", "B0_GPa", "B1", "points", "checks"):
        assert again[name] == out[name], name
    other.shutdown(grace_s=1.0)
    damaged = tmp_path / "bad.materia"
    target = f"arrays/{records.eos_key(run_id, 'energies')}/chunk-00000.npy"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "w") as sink:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == target:
                data = data[:-8] + b"\x00" * 8
            sink.writestr(item, data)
    with pytest.raises(ArrayStoreError, match="SHA-256"):
        Project.load(str(damaged))
    key = records.eos_key(run_id, "energies")
    good = svc.project.arrays[key]
    svc.project.arrays[key] = StoredArray(good.data + 0.1, good.unit, good.description,
                                          good.kind, dict(good.meta))
    reloaded = Project.load(svc.save_project(str(tmp_path / "rewritten"))["path"])
    assert records.eos_status(reloaded, run_id)["state"] == "corrupt"
    third = Service(reloaded, recovery_dir=str(tmp_path / "r3"))
    assert third.dft_eos_result(run_id)["ok"] is False
    assert third.dft_eos_apply(run_id)["ok"] is False
    with pytest.raises(ValueError, match="not exported"):
        third.dft_eos_export(str(tmp_path / "x.csv"), run_id)
    third.shutdown(grace_s=1.0)


def test_export_writes_every_point(svc, tmp_path):
    out = svc.dft_eos_run(GS, background=False)
    written = svc.dft_eos_export(str(tmp_path / "eos.csv"), out["run_id"])
    lines = open(written["path"]).read().splitlines()
    table = [line for line in lines if not line.startswith("#")]
    assert table[0].split(",") == ["V_over_Vref", "volume_A3", "zero_width_energy_eV",
                                   "free_energy_eV", "fit_zero_width_energy_eV",
                                   "fit_pressure_0K_GPa", "free_energy_fit_pressure_GPa",
                                   "stress_pressure_GPa", "max_force_eV_A"]
    rows = [r.split(",") for r in table[1:]]
    assert len(rows) == written["rows"] == 7
    for row in rows:
        assert float(row[2]) == pytest.approx(float(row[4]), abs=1e-9)
        assert float(row[3]) < float(row[2])
        assert float(row[7]) == pytest.approx(float(row[6]), abs=1e-3)
    assert any("zero-width extrapolated energy" in line for line in lines[:4])
    assert out["spec_digest"] in lines[0]


def test_http_routes(svc, tmp_path):
    port = find_free_port(9991)
    httpd = create_server(svc, port=port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(route, payload=None):
        request = urllib.request.Request(
            f"{base}/api/{route}", method="POST", data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    try:
        spec = post("dft/eos/spec", {"variables": GS})["data"]
        assert spec["ok"] and len(spec["spec"]["volumes_A3"]) == 7
        refused = post("dft/eos/spec", {"variables": {**GS, "n_points": 3}})["data"]
        assert refused["ok"] is False and "n_points" in refused["by_field"]
        run = post("dft/eos/run", {"variables": GS, "background": False})["data"]
        assert run["ok"]
        runs = post("dft/eos/runs")["data"]["runs"]
        assert runs[0]["run_id"] == run["run_id"]
        assert post("dft/eos/result", {"run_id": run["run_id"]})["data"]["B1"] == run["B1"]
        exported = post("dft/eos/export", {"path": str(tmp_path / "e.csv"),
                                           "run_id": run["run_id"]})["data"]
        assert exported["rows"] == 7
        assert post("dft/eos/apply", {"run_id": run["run_id"]})["data"]["ok"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_shipped_eos_example_runs(svc):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "examples" / "scripts"
              / "17_dft_equation_of_state.py").read_text()
    out = svc.runner.run(source)
    assert out.ok, out.error or out.traceback
    assert "Si equation of state" in out.stdout and "state: applied" in out.stdout
