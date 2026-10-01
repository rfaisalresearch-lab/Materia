"""LDOS and Tersoff-Hamann images through the service, Python API, persistence and HTTP.

GPAW is replaced by the deterministic double in ``tests/support/fake_gpaw_ldos.py``;
``tests/validation/test_dft_ldos_live.py`` runs real GPAW against ASE's STM code.
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
from materia.python_api.api import ApiError, DFTLDOSFailed
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import environment as fake_environment
from tests.support.fake_gpaw_ldos import FakeLDOSGPAW

GS = {"xc": "PBE", "grid_spacing_A": 0.2, "kpoints": [4, 4, 1]}
LDOS = {"energy_min_eV": -1.0, "energy_max_eV": 0.0}


def sheet(vacuum=16.0):
    a = 3.84
    return Structure(np.array([14, 14]), np.array([[0, 0, 5.0], [a / 2, a / 2, 5.8]]),
                     Cell(np.array([[a, 0, 0], [0, a, 0], [0, 0, vacuum]]), (True, True, False)))


@pytest.fixture
def gpaw(tmp_path, monkeypatch):
    import materia.solvers.gpaw_driver as package
    import materia.solvers.gpaw_driver.environment as module

    env = fake_environment(tmp_path)
    monkeypatch.setattr(package, "discover", lambda refresh=False: env)
    monkeypatch.setattr(module, "discover", lambda refresh=False: env)
    fake = FakeLDOSGPAW()
    monkeypatch.setattr(runner, "run_job", fake)
    return fake


@pytest.fixture
def svc(tmp_path, gpaw):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(sheet())
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


def test_ldos_is_stored_and_changes_nothing(svc):
    before = snapshot(svc.project)
    out = svc.dft_ldos_run({**GS, **LDOS}, background=False)
    assert out["ok"] and out["status"] == "complete", out
    after = snapshot(svc.project)
    assert np.array_equal(after[4], before[4]) and after[3] == before[3]
    assert after[2] == before[2] + 1
    entry = svc.project.history.log[-1]
    assert entry.operation == "dft.ldos" and entry.undoable is False
    assert out["run_state"] == "current" and out["stm"]["supported"] is True
    assert out["pseudo_norm_ratio"] == pytest.approx(0.95, rel=1e-9)
    assert len(out["profile"]["planar_average"]) == out["grid_shape"][2]
    assert out["provenance"]["model"] == "external:gpaw/local-density-of-states"
    assert out["provenance"]["nscf_parameters_used"]["symmetry"]["point_group"] is False
    status = svc.dft_status()
    assert status["ldos_runs"][0]["run_id"] == out["run_id"]
    assert {f["name"] for f in status["ldos_fields"]} >= {"energy_min_eV", "n_bands"}


def test_slices_and_images(svc):
    out = svc.dft_ldos_run({**GS, **LDOS}, background=False)
    first = svc.dft_ldos_slice(out["run_id"])
    assert first["ok"] and len(first["values"]) == np.prod(first["shape"])
    assert len(first["lut"]) == 256
    top = svc.dft_ldos_slice(out["run_id"], first["n_planes"] - 1)
    assert top["ok"] and top["max"] < first["max"]
    assert svc.dft_ldos_slice(out["run_id"], 10 ** 6)["ok"] is False
    height = svc.dft_ldos_image(out["run_id"], "constant-height", height_A=4.0)
    assert height["ok"], height
    isovalue = float(np.median(height["values"]))
    current = svc.dft_ldos_image(out["run_id"], "constant-current", isovalue=isovalue)
    assert current["ok"], current
    assert current["min"] > height["valid_heights_A"][0] - 0.5
    too_close = svc.dft_ldos_image(out["run_id"], "constant-height", height_A=0.5)
    assert too_close["ok"] is False and "valid band" in too_close["error"]


def test_state_follows_the_geometry(svc):
    out = svc.dft_ldos_run({**GS, **LDOS}, background=False)
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + 0.05
    svc.project.record_change(s, before, svc.project.selection, "nudge", "edit.move")
    assert svc.dft_ldos_result(out["run_id"])["run_state"] == "stale"
    svc.undo()
    assert svc.dft_ldos_result(out["run_id"])["run_state"] == "current"


def test_from_a_stored_ground_state(svc, gpaw):
    ground = svc.dft_run({**GS, "observables": ["energy", "forces"]}, background=False)
    assert ground["ok"], ground
    source = {"kind": "ground-state", "run_id": ground["run_id"]}
    out = svc.dft_ldos_run(LDOS, source, background=False)
    assert out["ok"], out
    assert gpaw.jobs[-1]["parameters"] == gpaw.jobs[0]["parameters"]
    refused = svc.dft_ldos_spec({**LDOS, "grid_spacing_A": 0.18}, source)
    assert refused["ok"] is False and "keeps that run's geometry" in refused["error"]


def test_python_api_and_detached(svc):
    loose = sheet(18.0)
    results = svc.lab.dft.ldos(loose, **GS, **LDOS)
    run_id = results["ldos"].extra["run_id"]
    assert svc.lab.dft.ldos_state(run_id)["state"] == "detached"
    assert svc.lab.dft.ldos_state(run_id, target=loose)["state"] == "current"
    assert svc.lab.dft.ldos_array(run_id).data.ndim == 3
    image = svc.lab.dft.stm_image(run_id, "constant-height", height_A=4.0)
    assert image["values"].ndim == 2
    with pytest.raises(ApiError, match="valid band"):
        svc.lab.dft.stm_image(run_id, "constant-height", height_A=0.3)


def test_cluster_has_no_stm_image(svc):
    molecule = Structure(np.array([1, 1]), np.array([[4, 4, 3.63], [4, 4, 4.37]]),
                         Cell(np.eye(3) * 8.0, (False, False, False)))
    results = svc.lab.dft.ldos(molecule, xc="PBE", grid_spacing_A=0.2, energy_min_eV=-3.0,
                               energy_max_eV=0.0)
    run_id = results["ldos"].extra["run_id"]
    assert results["ldos"].value["stm"]["supported"] is False
    image = svc.dft_ldos_image(run_id, "constant-height", height_A=3.0)
    assert image["ok"] is False and "slab" in image["error"]


@pytest.mark.parametrize("mode", ["failed", "timeout", "cancelled", "not_converged",
                                  "bad_count", "short_bands", "empty", "echo_nscf"])
def test_failures_are_transactional(svc, gpaw, mode):
    gpaw.mode = mode
    before = snapshot(svc.project)
    out = svc.dft_ldos_run({**GS, **LDOS}, background=False, timeout_s=1.0)
    assert out["ok"] is False and out["status"] in ("failed", "cancelled")
    assert unchanged(before, snapshot(svc.project))
    with pytest.raises(DFTLDOSFailed):
        svc.lab.dft.ldos(**GS, **LDOS)
    assert unchanged(before, snapshot(svc.project))
    assert svc.dft_ldos_result()["ok"] is False


def test_refusal_starts_nothing(svc, gpaw):
    before = snapshot(svc.project)
    out = svc.dft_ldos_run({**GS, "energy_min_eV": 0.5, "energy_max_eV": 0.5},
                           background=False)
    assert out["ok"] is False and out["refused"] and "energy_max_eV" in out["by_field"]
    with pytest.raises(ApiError, match="refused"):
        svc.lab.dft.ldos(**GS, spin_channels="resolved")
    assert gpaw.jobs == []
    assert unchanged(before, snapshot(svc.project))


def test_job_cancellation_and_project_replacement(svc, gpaw):
    gpaw.mode = "wait_for_cancel"
    gpaw.started = threading.Event()
    before = snapshot(svc.project)
    submitted = svc.dft_ldos_run({**GS, **LDOS}, background=True)
    assert gpaw.started.wait(5)
    svc.cancel_job(submitted["job"]["id"])
    status = wait(svc, submitted["job"]["id"])
    assert status["status"] == "cancelled" and status["result"]["status"] == "cancelled"
    assert unchanged(before, snapshot(svc.project))
    gpaw.started = threading.Event()
    old = svc.project
    submitted = svc.dft_ldos_run({**GS, **LDOS}, background=True)
    assert gpaw.started.wait(5)
    svc.new_project("fresh")
    assert wait(svc, submitted["job"]["id"])["status"] == "cancelled"
    assert not any(k.startswith(records.LDOS_PREFIX) for k in old.results)
    assert not any(k.startswith(records.LDOS_PREFIX) for k in svc.project.results)


def test_persistence_round_trip_and_corruption(svc, tmp_path):
    out = svc.dft_ldos_run({**GS, **LDOS}, background=False)
    run_id = out["run_id"]
    path = svc.save_project(str(tmp_path / "ldos"))["path"]
    loaded = Project.load(path)
    for name in out["array_sha256"]:
        key = records.ldos_key(run_id, name)
        assert np.array_equal(loaded.arrays[key].data, svc.project.arrays[key].data)
    other = Service(loaded, recovery_dir=str(tmp_path / "r2"))
    again = other.dft_ldos_result(run_id)
    assert again["run_state"] == "current"
    assert again["states_in_window"] == out["states_in_window"]
    image = other.dft_ldos_image(run_id, "constant-height", height_A=4.0)
    assert image["ok"]
    other.shutdown(grace_s=1.0)
    damaged = tmp_path / "bad.materia"
    target = f"arrays/{records.ldos_key(run_id, 'ldos')}/chunk-00000.npy"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "w") as sink:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == target:
                data = data[:-8] + b"\x00" * 8
            sink.writestr(item, data)
    with pytest.raises(ArrayStoreError, match="SHA-256"):
        Project.load(str(damaged))
    key = records.ldos_key(run_id, "ldos")
    good = svc.project.arrays[key]
    svc.project.arrays[key] = StoredArray(good.data * 2.0, good.unit, good.description,
                                          good.kind, dict(good.meta))
    reloaded = Project.load(svc.save_project(str(tmp_path / "rewritten"))["path"])
    assert records.ldos_status(reloaded, run_id)["state"] == "corrupt"
    third = Service(reloaded, recovery_dir=str(tmp_path / "r3"))
    assert third.dft_ldos_result(run_id)["ok"] is False
    assert third.dft_ldos_image(run_id, "constant-height", height_A=4.0)["ok"] is False
    with pytest.raises(ValueError, match="refused"):
        third.dft_ldos_export(str(tmp_path / "x.cube"), run_id)
    third.shutdown(grace_s=1.0)


def test_exports(svc, tmp_path):
    out = svc.dft_ldos_run({**GS, **LDOS}, background=False)
    cube = svc.dft_ldos_export(str(tmp_path / "map.cube"), out["run_id"])
    lines = open(cube["path"]).read().splitlines()
    assert "LDOS" in lines[0] and out["spec_digest"][:16] in lines[0]
    counts = [int(lines[k].split()[0]) for k in (3, 4, 5)]
    assert counts == out["grid_shape"]
    written = svc.dft_ldos_image_export(str(tmp_path / "stm.csv"), out["run_id"],
                                        "constant-height", height_A=4.0)
    table = [line for line in open(written["path"]).read().splitlines()
             if not line.startswith("#")]
    assert table[0] == "i,j,x_A,y_A,value"
    assert len(table) - 1 == written["rows"] == out["grid_shape"][0] * out["grid_shape"][1]
    with pytest.raises(ValueError, match="valid band"):
        svc.dft_ldos_image_export(str(tmp_path / "bad.csv"), out["run_id"],
                                  "constant-height", height_A=0.2)


def test_http_routes(svc, tmp_path):
    port = find_free_port(9995)
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
        spec = post("dft/ldos/spec", {"variables": {**GS, **LDOS}})["data"]
        assert spec["ok"] and spec["spec"]["nscf_parameters"]["symmetry"]["point_group"] is False
        run = post("dft/ldos/run", {"variables": {**GS, **LDOS}, "background": False})["data"]
        assert run["ok"]
        assert post("dft/ldos/runs")["data"]["runs"][0]["run_id"] == run["run_id"]
        assert post("dft/ldos/result", {"run_id": run["run_id"]})["data"]["ok"]
        assert post("dft/ldos/slice", {"run_id": run["run_id"], "index": 3})["data"]["index"] == 3
        image = post("dft/ldos/image", {"run_id": run["run_id"], "mode": "constant-height",
                                        "height_A": 4.0})["data"]
        assert image["ok"]
        assert post("dft/ldos/export", {"path": str(tmp_path / "m.cube"),
                                        "run_id": run["run_id"]})["data"]["shape"]
        exported = post("dft/ldos/image/export", {"path": str(tmp_path / "i.csv"),
                                                  "run_id": run["run_id"],
                                                  "mode": "constant-height",
                                                  "height_A": 4.0})["data"]
        assert exported["rows"] > 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_shipped_ldos_example_runs(svc):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "examples" / "scripts"
              / "18_dft_ldos_stm.py").read_text()
    out = svc.runner.run(source)
    assert out.ok, out.error or out.traceback
    assert "graphene LDOS" in out.stdout and "state: current" in out.stdout
