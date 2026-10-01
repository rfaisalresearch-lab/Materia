"""Ground-state DFT through the service, HTTP layer and project format."""

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
from materia.provenance.classification import Claim, Classification, Evidence
from materia.solvers.gpaw_driver import runner


def molecule(distance=0.74, box=8.0):
    centre = box / 2
    return Structure(
        np.array([1, 1]),
        np.array([[centre, centre, centre - distance / 2],
                  [centre, centre, centre + distance / 2]]),
        Cell(np.eye(3) * box, (False, False, False)),
    )


def variables():
    return {
        "xc": "PBE",
        "grid_spacing_A": 0.25,
        "max_scf_iterations": 100,
        "observables": ["energy", "forces", "density"],
    }


@pytest.fixture(scope="module")
def saved_run(gpaw_environment, tmp_path_factory):
    if gpaw_environment is None:
        pytest.skip("GPAW or its PAW datasets are not available for the real service run.")
    root = tmp_path_factory.mktemp("dft-service")
    service = Service(recovery_dir=str(root / "recovery"))
    service.project.add_structure(molecule())
    undo_depth = len(service.project.history._undo)
    outcome = service.dft_run(variables(), background=False, timeout_s=180)
    assert outcome["ok"] and outcome["status"] == "converged"
    assert len(service.project.history._undo) == undo_depth
    run_id = outcome["run_id"]
    claim = Claim(
        "PBE predicts a bound H2 state for this geometry",
        Classification.COMPUTATIONAL_PREDICTION,
        [Evidence("result", records.key(run_id, "energy"), model="GPAW",
                  software="25.7.0", converged=True)],
    )
    service.project.add_claim(claim)
    path = service.save_project(str(root / "ground-state"))["path"]
    return path, run_id, outcome, claim.claim_id


def test_service_run_persists_results_arrays_claims_and_nonundoable_history(saved_run):
    path, run_id, outcome, claim_id = saved_run
    project = Project.load(path)
    assert project.results[records.key(run_id, "energy")].value == pytest.approx(
        outcome["energy_eV"])
    assert records.key(run_id, "density") in project.arrays
    assert claim_id in project.claims
    entries = [e for e in project.history.log if e.operation == "dft.ground_state"]
    assert len(entries) == 1
    assert entries[0].undoable is False
    assert entries[0].parameters["run_id"] == run_id


def test_result_state_tracks_geometry_but_not_isotope(saved_run, tmp_path):
    path, run_id, _, _ = saved_run
    service = Service(Project.load(path), recovery_dir=str(tmp_path / "recovery"))
    assert service.dft_result(run_id)["state"] == "current"
    original_mass = int(service.structure.mass_numbers[0])
    service.structure.mass_numbers[0] = original_mass + 1
    isotope = service.dft_result(run_id)
    assert isotope["state"] == "current"
    assert isotope["current"] is True
    service.structure.positions[0, 2] += 0.05
    stale = service.dft_result(run_id)
    assert stale["state"] == "stale"
    assert stale["current"] is False
    assert "positions" in stale["state_reason"]


def test_http_routes_expose_specification_result_and_array(saved_run, tmp_path):
    path, run_id, _, _ = saved_run
    service = Service(Project.load(path), recovery_dir=str(tmp_path / "recovery"))
    port = find_free_port(9831)
    httpd = create_server(service, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def post(route, payload=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/{route}",
            method="POST",
            data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())["data"]

    try:
        described = post("dft/spec", {"variables": variables()})
        assert described["ok"] is True
        result = post("dft/result", {"run_id": run_id})
        assert result["status"] == "converged"
        assert result["current"] is True
        profile = post("dft/array", {"run_id": run_id, "name": "density", "axis": 2})
        assert profile["ok"] is True
        assert len(profile["coordinate_A"]) == len(profile["values"])
        claims = post("claims/list")
        assert len(claims["claims"]) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_project_load_refuses_a_corrupted_dft_array(saved_run, tmp_path):
    path, run_id, _, _ = saved_run
    damaged = tmp_path / "damaged.materia"
    target = f"arrays/{records.key(run_id, 'density')}/chunk-00000.npy"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "w") as output:
        for name in source.namelist():
            payload = source.read(name)
            if name == target:
                payload = b"damaged"
            output.writestr(name, payload)
    with pytest.raises(ArrayStoreError, match="not a readable NumPy array"):
        Project.load(str(damaged))


def test_project_replacement_cancels_job_and_never_writes_into_new_project(
        monkeypatch, needs_gpaw, tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(molecule())
    original = service.project
    started = threading.Event()

    def interrupted(job, environment, **kwargs):
        started.set()
        cancelled = kwargs.get("cancelled")
        deadline = time.time() + 10
        while time.time() < deadline and not cancelled():
            time.sleep(0.01)
        return runner.GPAWRun(status="cancelled", result={"scf_history": []}, iterations=0)

    monkeypatch.setattr(runner, "run_job", interrupted)
    submitted = service.dft_run(variables(), background=True)
    assert started.wait(5)
    service.new_project("Replacement")
    deadline = time.time() + 10
    while time.time() < deadline:
        status = service.job_status(submitted["job"]["id"])
        if status["status"] in ("cancelled", "done", "failed"):
            break
        time.sleep(0.02)
    assert status["status"] == "cancelled"
    assert service.project.results == {}
    assert service.project.arrays == {}
    assert records.key(submitted["run_id"], "run") in original.results
    assert not original.results[records.key(submitted["run_id"], "run")].supported
