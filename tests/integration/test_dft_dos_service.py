"""DOS and PDOS through the service, Python API, history, persistence and HTTP.

GPAW is replaced by the deterministic double in ``tests/support/fake_gpaw_dos.py``;
``tests/validation/test_dft_dos_live.py`` runs the real code when GPAW is installed.
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
from materia.python_api.api import ApiError, DFTDOSFailed
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import FERMI_EV, FakeGPAW
from tests.support.fake_gpaw_dos import environment as fake_environment

GS = {"xc": "PBE", "grid_spacing_A": 0.22, "forces_tol_eV_A": 0.01}
DOS = {"n_bands": 4, "energy_min_eV": -15.0, "energy_max_eV": 1.0, "energy_step_eV": 0.02}


def molecule(distance=0.74, box=8.0):
    c = box / 2
    s = Structure(np.array([1, 1]),
                  np.array([[c, c, c - distance / 2], [c, c, c + distance / 2]]),
                  Cell(np.eye(3) * box, (False, False, False)))
    s.roles[:] = ["adsorbate", "adsorbate"]
    return s


@pytest.fixture
def gpaw(tmp_path, monkeypatch):
    import materia.solvers.gpaw_driver as package
    import materia.solvers.gpaw_driver.environment as module

    env = fake_environment(tmp_path)
    monkeypatch.setattr(package, "discover", lambda refresh=False: env)
    monkeypatch.setattr(module, "discover", lambda refresh=False: env)
    fake = FakeGPAW()
    monkeypatch.setattr(runner, "run_job", fake)
    return fake


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
            list(s.roles))


def unchanged(a, b):
    return a[:4] == b[:4] and np.array_equal(a[4], b[4]) and np.array_equal(a[5], b[5]) \
        and a[6] == b[6]


def wait(service, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.job_status(job_id)
        if status["status"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_dos_of_the_structure_is_stored_and_changes_nothing(svc):
    before = snapshot(svc.project)
    out = svc.dft_dos_run({**GS, **DOS}, background=False)
    assert out["ok"] and out["status"] == "complete", out
    after = snapshot(svc.project)
    assert np.array_equal(after[4], before[4]) and after[3] == before[3]
    assert after[2] == before[2] + 1
    entry = svc.project.history.log[-1]
    assert entry.operation == "dft.dos" and entry.undoable is False
    assert out["run_state"] == "current" and out["source"]["kind"] == "structure"
    assert out["fermi_level_eV"] == FERMI_EV and out["reference"] == "fermi-level"
    assert len(out["curves"]["energies_eV"]) == 801
    assert [p["label"] for p in out["projections"]] == ["H s"]
    assert out["checks"]["electron_count_difference_e"] == pytest.approx(0.0, abs=1e-6)
    assert out["provenance"]["model"] == "external:gpaw/density-of-states"
    assert out["provenance"]["nscf_parameters_used"]["nbands"] == 4


def test_state_follows_the_geometry(svc):
    out = svc.dft_dos_run({**GS, **DOS}, background=False)
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + 0.05
    svc.project.record_change(s, before, svc.project.selection, "nudge", "edit.move")
    result = svc.dft_dos_result(out["run_id"])
    assert result["run_state"] == "stale" and "atom positions changed" in result["state_reason"]
    svc.undo()
    assert svc.dft_dos_result(out["run_id"])["run_state"] == "current"


def test_dos_from_a_stored_ground_state(svc, gpaw):
    ground = svc.dft_run({**GS, "observables": ["energy", "forces"]}, background=False)
    assert ground["ok"]
    sources = svc.dft_status()["dos_sources"]
    assert sources[0]["kind"] == "ground-state" and sources[0]["run_id"] == ground["run_id"]
    source = {"kind": "ground-state", "run_id": ground["run_id"]}
    out = svc.dft_dos_run(DOS, source, background=False)
    assert out["ok"], out
    assert out["source"]["run_id"] == ground["run_id"]
    assert out["checks"]["source_energy_difference_eV"] == 0.0
    job = gpaw.jobs[-1]
    assert job["parameters"] == gpaw.jobs[0]["parameters"]
    refused = svc.dft_dos_run({**DOS, "grid_spacing_A": 0.18}, source, background=False)
    assert refused["ok"] is False and "keeps that run's geometry" in refused["error"]


def test_dos_from_a_relaxation_is_current_only_once_applied(svc):
    relax = svc.dft_relax({**GS}, background=False, apply=False)
    assert relax["ok"] and relax["status"] == "converged"
    out = svc.dft_dos_run(DOS, {"kind": "relaxation", "run_id": relax["run_id"]},
                          background=False)
    assert out["ok"], out
    assert out["run_state"] == "stale" and "Apply the relaxation" in out["state_reason"]
    assert svc.dft_relax_apply(relax["run_id"])["ok"]
    assert svc.dft_dos_result(out["run_id"])["run_state"] == "current"


def test_unconverged_sources_are_refused(svc, gpaw):
    gpaw.relax.mode = "max_steps"
    relax = svc.dft_relax({**GS, "max_steps": 2}, background=False)
    assert relax["status"] == "not-converged"
    out = svc.dft_dos_spec(DOS, {"kind": "relaxation", "run_id": relax["run_id"]})
    assert out["ok"] is False and "not a minimum" in out["error"]
    missing = svc.dft_dos_spec(DOS, {"kind": "ground-state", "run_id": "nope"})
    assert missing["ok"] is False and "no ground-state run" in missing["error"].lower()


def test_detached_structure(svc):
    loose = molecule()
    run = svc.lab.dft.dos(loose, **GS, **DOS)
    run_id = run["dos"].extra["run_id"]
    assert svc.lab.dft.dos_state(run_id)["state"] == "detached"
    assert svc.lab.dft.dos_state(run_id, target=loose)["state"] == "current"
    assert svc.lab.dft.dos_array(run_id, "dos_total").data.shape == (801,)


@pytest.mark.parametrize("mode", ["failed", "timeout", "cancelled", "not_converged",
                                  "bad_recompute", "short_bands", "echo_nscf"])
def test_failures_are_transactional(svc, gpaw, mode):
    gpaw.mode = mode
    before = snapshot(svc.project)
    out = svc.dft_dos_run({**GS, **DOS}, background=False, timeout_s=1.0)
    assert out["ok"] is False and out["status"] in ("failed", "cancelled")
    assert unchanged(before, snapshot(svc.project))
    with pytest.raises(DFTDOSFailed):
        svc.lab.dft.dos(**GS, **DOS)
    assert unchanged(before, snapshot(svc.project))
    assert svc.dft_dos_result()["ok"] is False


def test_refusal_starts_nothing(svc, gpaw):
    before = snapshot(svc.project)
    out = svc.dft_dos_run({**GS, **DOS, "spin_channels": "resolved"}, background=False)
    assert out["ok"] is False and out["refused"] and "spin_channels" in out["by_field"]
    with pytest.raises(ApiError, match="refused"):
        svc.lab.dft.dos(**GS, **DOS, broadening="tetrahedron")
    assert gpaw.jobs == []
    assert unchanged(before, snapshot(svc.project))


def test_job_cancellation_and_project_replacement(svc, gpaw):
    gpaw.mode = "wait_for_cancel"
    gpaw.started = threading.Event()
    before = snapshot(svc.project)
    submitted = svc.dft_dos_run({**GS, **DOS}, background=True)
    assert gpaw.started.wait(5)
    svc.cancel_job(submitted["job"]["id"])
    status = wait(svc, submitted["job"]["id"])
    assert status["status"] == "cancelled" and status["result"]["status"] == "cancelled"
    assert unchanged(before, snapshot(svc.project))
    gpaw.started = threading.Event()
    old = svc.project
    submitted = svc.dft_dos_run({**GS, **DOS}, background=True)
    assert gpaw.started.wait(5)
    svc.new_project("fresh")
    assert wait(svc, submitted["job"]["id"])["status"] == "cancelled"
    assert not any(k.startswith(records.DOS_PREFIX) for k in old.results)
    assert not any(k.startswith(records.DOS_PREFIX) for k in svc.project.results)


def test_persistence_round_trip_and_corruption(svc, tmp_path):
    out = svc.dft_dos_run({**GS, **DOS, "charge_e": 1.0}, background=False)
    assert out["ok"] and out["curves"].get("up") is not None
    run_id = out["run_id"]
    path = svc.save_project(str(tmp_path / "dos"))["path"]
    loaded = Project.load(path)
    for name in ("energies", "dos_total", "dos_spin", "pdos", "pdos_spin", "eigenvalues"):
        key = records.dos_key(run_id, name)
        assert np.array_equal(loaded.arrays[key].data, svc.project.arrays[key].data)
    other = Service(loaded, recovery_dir=str(tmp_path / "r2"))
    assert other.lab.dft.dos_spec_of(run_id).digest == svc.lab.dft.dos_spec_of(run_id).digest
    again = other.dft_dos_result(run_id)
    assert again["run_state"] == "current" and again["checks"] == out["checks"]
    assert loaded.history.log[-1].operation == "dft.dos"
    other.shutdown(grace_s=1.0)
    damaged = tmp_path / "bad.materia"
    target = f"arrays/{records.dos_key(run_id, 'dos_total')}/chunk-00000.npy"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "w") as sink:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == target:
                data = data[:-8] + b"\x00" * 8
            sink.writestr(item, data)
    with pytest.raises(ArrayStoreError, match="SHA-256"):
        Project.load(str(damaged))


def test_export_writes_full_resolution_csv(svc, tmp_path):
    out = svc.dft_dos_run({**GS, **DOS, "energy_step_eV": 0.001, "width_eV": 0.05},
                          background=False)
    assert out["display_stride"] > 1
    written = svc.dft_dos_export(str(tmp_path / "dos.csv"), out["run_id"])
    lines = open(written["path"]).read().splitlines()
    header = [line for line in lines if not line.startswith("#")][0]
    assert header == "E-E_F_eV,total_states_per_eV,pdos_H_s_states_per_eV"
    rows = [line for line in lines if not line.startswith("#")][1:]
    assert len(rows) == written["rows"] == 16001
    assert out["spec_digest"] in lines[0]


def test_http_routes(svc, tmp_path):
    port = find_free_port(9971)
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
        spec = post("dft/dos/spec", {"variables": {**GS, **DOS}})["data"]
        assert spec["ok"] and spec["spec"]["dos_settings"]["npoints"] == 801
        refused = post("dft/dos/spec", {"variables": {**GS, **DOS, "energy_step_eV": 0.03}})
        assert refused["data"]["ok"] is False and "energy_step_eV" in refused["data"]["by_field"]
        run = post("dft/dos/run", {"variables": {**GS, **DOS}, "background": False})["data"]
        assert run["ok"]
        runs = post("dft/dos/runs")["data"]["runs"]
        assert runs[0]["run_id"] == run["run_id"] and runs[0]["state"] == "current"
        result = post("dft/dos/result", {"run_id": run["run_id"], "max_points": 100})["data"]
        assert len(result["curves"]["energies_eV"]) <= 101
        exported = post("dft/dos/export", {"path": str(tmp_path / "h.csv"),
                                           "run_id": run["run_id"]})["data"]
        assert exported["rows"] == 801
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_shipped_dos_example_runs(svc):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "examples" / "scripts"
              / "15_dft_dos.py").read_text()
    out = svc.runner.run(source)
    assert out.ok, out.error or out.traceback
    assert "H2 DOS" in out.stdout and "Si DOS" in out.stdout
    assert "state: current" in out.stdout
