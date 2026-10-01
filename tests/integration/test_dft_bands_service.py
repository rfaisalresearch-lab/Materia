"""Band structure through the service, Python API, history, persistence and HTTP.

GPAW is replaced by the deterministic double in ``tests/support/fake_gpaw_bands.py``;
``tests/validation/test_dft_bands_live.py`` runs the real code when GPAW is installed.
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
from materia.python_api.api import ApiError, DFTBandsFailed
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_bands import FakeBandsGPAW
from tests.support.fake_gpaw_dos import FERMI_EV
from tests.support.fake_gpaw_dos import environment as fake_environment

GS = {"xc": "PBE", "cutoff_eV": 300.0, "occupations": "fixed", "smearing_eV": 0.0,
      "kpoints": [2, 2, 2], "forces_tol_eV_A": 0.01}
BANDS = {"path": "GXL", "segment_intervals": [4, 3], "n_bands": 6}


def silicon(a=5.43):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    s = Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                  Cell(cell, (True, True, True)))
    return s


def molecule(box=8.0):
    c = box / 2
    return Structure(np.array([1, 1]), np.array([[c, c, c - 0.37], [c, c, c + 0.37]]),
                     Cell(np.eye(3) * box, (False, False, False)))


@pytest.fixture
def gpaw(tmp_path, monkeypatch):
    import materia.solvers.gpaw_driver as package
    import materia.solvers.gpaw_driver.environment as module

    env = fake_environment(tmp_path)
    monkeypatch.setattr(package, "discover", lambda refresh=False: env)
    monkeypatch.setattr(module, "discover", lambda refresh=False: env)
    fake = FakeBandsGPAW()
    monkeypatch.setattr(runner, "run_job", fake)
    return fake


@pytest.fixture
def svc(tmp_path, gpaw):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(silicon())
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


def test_bands_of_the_structure_are_stored_and_change_nothing(svc):
    before = snapshot(svc.project)
    out = svc.dft_bands_run({**GS, **BANDS}, background=False)
    assert out["ok"] and out["status"] == "complete", out
    after = snapshot(svc.project)
    assert np.array_equal(after[4], before[4]) and after[3] == before[3]
    assert after[2] == before[2] + 1
    entry = svc.project.history.log[-1]
    assert entry.operation == "dft.bands" and entry.undoable is False
    assert out["run_state"] == "current" and out["source"]["kind"] == "structure"
    assert out["fermi_level_eV"] == FERMI_EV and out["n_kpoints"] == 8
    assert np.asarray(out["bands_relative_eV"]).shape == (1, 6, 8)
    assert [t["label"] for t in out["ticks"]] == ["G", "X", "L"]
    assert out["distance_invA"][0] == 0.0 and out["distance_invA"][-1] == pytest.approx(
        out["path_length_invA"])
    assert out["band_edges"]["gap_eV"] > 0
    assert out["provenance"]["model"] == "external:gpaw/band-structure"
    assert out["provenance"]["nscf_parameters_used"]["symmetry"] == "off"
    assert out["experimental_comparison"] is None
    status = svc.dft_status()
    assert status["bands_runs"][0]["run_id"] == out["run_id"]
    assert {f["name"] for f in status["bands_fields"]} >= {"path", "segment_intervals"}


def test_state_follows_the_geometry(svc):
    out = svc.dft_bands_run({**GS, **BANDS}, background=False)
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + 0.05
    svc.project.record_change(s, before, svc.project.selection, "nudge", "edit.move")
    result = svc.dft_bands_result(out["run_id"])
    assert result["run_state"] == "stale" and "atom positions changed" in result["state_reason"]
    svc.undo()
    assert svc.dft_bands_result(out["run_id"])["run_state"] == "current"


def test_bands_from_a_stored_ground_state(svc, gpaw):
    ground = svc.dft_run({**GS, "observables": ["energy", "forces"]}, background=False)
    assert ground["ok"], ground
    source = {"kind": "ground-state", "run_id": ground["run_id"]}
    spec = svc.dft_bands_spec(BANDS, source)
    assert spec["ok"] and spec["spec"]["source"]["run_id"] == ground["run_id"]
    out = svc.dft_bands_run(BANDS, source, background=False)
    assert out["ok"], out
    assert out["checks"]["source_energy_difference_eV"] == 0.0
    job = gpaw.jobs[-1]
    sent = {k: v for k, v in job["parameters"].items() if k != "symmetry"}
    assert sent == gpaw.jobs[0]["parameters"]
    refused = svc.dft_bands_run({**BANDS, "cutoff_eV": 400.0}, source, background=False)
    assert refused["ok"] is False and "keeps that run's geometry" in refused["error"]


def test_bands_from_a_relaxation_are_current_only_once_applied(svc):
    s = svc.structure
    before = s.copy()
    s.positions = s.positions + np.array([[0.0, 0.0, 0.0], [0.04, -0.02, 0.03]])
    svc.project.record_change(s, before, svc.project.selection, "nudge", "edit.move")
    relax = svc.dft_relax({**GS, "symmetry": "off"}, background=False, apply=False)
    assert relax["ok"] and relax["status"] == "converged", relax
    source = {"kind": "relaxation", "run_id": relax["run_id"]}
    spec = svc.dft_bands_spec(BANDS, source)
    assert spec["spec"]["settings"]["scf_symmetry"] == "off"
    out = svc.dft_bands_run(BANDS, source, background=False)
    assert out["ok"], out
    assert out["run_state"] == "stale" and "Apply the relaxation" in out["state_reason"]
    assert svc.dft_relax_apply(relax["run_id"])["ok"]
    assert svc.dft_bands_result(out["run_id"])["run_state"] == "current"


def test_detached_structure(svc):
    loose = silicon(5.5)
    run = svc.lab.dft.bands(loose, **GS, **BANDS)
    run_id = run["bands"].extra["run_id"]
    assert svc.lab.dft.bands_state(run_id)["state"] == "detached"
    assert svc.lab.dft.bands_state(run_id, target=loose)["state"] == "current"
    assert svc.lab.dft.bands_array(run_id, "eigenvalues").data.shape == (1, 8, 6)
    assert svc.lab.dft.bands_spec_of(run_id).n_kpoints == 8
    assert svc.lab.dft.bands_runs()[0]["state"] == "detached"


@pytest.mark.parametrize("mode", ["failed", "timeout", "cancelled", "not_converged",
                                  "wrong_order", "missing_bands", "nan", "bad_distance",
                                  "echo_nscf"])
def test_failures_are_transactional(svc, gpaw, mode):
    gpaw.mode = mode
    before = snapshot(svc.project)
    out = svc.dft_bands_run({**GS, **BANDS}, background=False, timeout_s=1.0)
    assert out["ok"] is False and out["status"] in ("failed", "cancelled")
    assert unchanged(before, snapshot(svc.project))
    with pytest.raises(DFTBandsFailed):
        svc.lab.dft.bands(**GS, **BANDS)
    assert unchanged(before, snapshot(svc.project))
    assert svc.dft_bands_result()["ok"] is False


def test_refusal_starts_nothing(svc, gpaw):
    before = snapshot(svc.project)
    out = svc.dft_bands_run({**GS, **BANDS, "energy_reference": "absolute"}, background=False)
    assert out["ok"] is False and out["refused"] and "energy_reference" in out["by_field"]
    with pytest.raises(ApiError, match="refused"):
        svc.lab.dft.bands(molecule(), xc="PBE")
    assert gpaw.jobs == []
    assert unchanged(before, snapshot(svc.project))


def test_job_cancellation_and_project_replacement(svc, gpaw):
    gpaw.mode = "wait_for_cancel"
    gpaw.started = threading.Event()
    before = snapshot(svc.project)
    submitted = svc.dft_bands_run({**GS, **BANDS}, background=True)
    assert gpaw.started.wait(5)
    svc.cancel_job(submitted["job"]["id"])
    status = wait(svc, submitted["job"]["id"])
    assert status["status"] == "cancelled" and status["result"]["status"] == "cancelled"
    assert unchanged(before, snapshot(svc.project))
    gpaw.started = threading.Event()
    old = svc.project
    submitted = svc.dft_bands_run({**GS, **BANDS}, background=True)
    assert gpaw.started.wait(5)
    svc.new_project("fresh")
    assert wait(svc, submitted["job"]["id"])["status"] == "cancelled"
    assert not any(k.startswith(records.BANDS_PREFIX) for k in old.results)
    assert not any(k.startswith(records.BANDS_PREFIX) for k in svc.project.results)


def test_persistence_round_trip_and_corruption(svc, tmp_path):
    out = svc.dft_bands_run({**GS, **BANDS, "spin_polarized": True,
                             "initial_magnetic_moments_muB": [1.0, 1.0],
                             "occupations": "fermi-dirac", "smearing_eV": 0.1},
                            background=False)
    assert out["ok"] and out["n_spins"] == 2, out
    run_id = out["run_id"]
    path = svc.save_project(str(tmp_path / "bands"))["path"]
    loaded = Project.load(path)
    for name in records.BANDS_ARRAYS:
        key = records.bands_key(run_id, name)
        assert np.array_equal(loaded.arrays[key].data, svc.project.arrays[key].data)
        assert loaded.arrays[key].data.dtype == svc.project.arrays[key].data.dtype
    other = Service(loaded, recovery_dir=str(tmp_path / "r2"))
    assert other.lab.dft.bands_spec_of(run_id).digest == svc.lab.dft.bands_spec_of(run_id).digest
    again = other.dft_bands_result(run_id)
    assert again["run_state"] == "current"
    for name in ("bands_relative_eV", "distance_invA", "ticks", "checks", "band_edges"):
        assert again[name] == out[name], name
    assert loaded.history.log[-1].operation == "dft.bands"
    other.shutdown(grace_s=1.0)

    damaged = tmp_path / "bad.materia"
    target = f"arrays/{records.bands_key(run_id, 'eigenvalues')}/chunk-00000.npy"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "w") as sink:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == target:
                data = data[:-8] + b"\x00" * 8
            sink.writestr(item, data)
    with pytest.raises(ArrayStoreError, match="SHA-256"):
        Project.load(str(damaged))

    key = records.bands_key(run_id, "eigenvalues")
    good = svc.project.arrays[key]
    svc.project.arrays[key] = StoredArray(good.data + 0.5, good.unit, good.description,
                                          good.kind, dict(good.meta))
    resaved = svc.save_project(str(tmp_path / "rewritten"))["path"]
    reloaded = Project.load(resaved)
    assert records.bands_status(reloaded, run_id)["state"] == "corrupt"
    third = Service(reloaded, recovery_dir=str(tmp_path / "r3"))
    refused = third.dft_bands_result(run_id)
    assert refused["ok"] is False and refused["run_state"] == "corrupt"
    assert "SHA-256" in refused["error"]
    with pytest.raises(ValueError, match="not exported"):
        third.dft_bands_export(str(tmp_path / "x.csv"), run_id)
    with pytest.raises(ApiError, match="refused"):
        third.lab.dft.bands_array(run_id)
    third.shutdown(grace_s=1.0)


def test_store_refuses_tampered_or_repeated_outcomes(svc, gpaw):
    from materia.experiments.dft import bands as B

    env = __import__("materia.solvers.gpaw_driver", fromlist=["discover"]).discover()
    spec = B.build(svc.structure, env, "s", **GS, **BANDS)
    outcome = B.execute(spec, env, run_id="r1")
    assert outcome.ok
    records.store_bands(svc.project, outcome)
    before = snapshot(svc.project)
    with pytest.raises(ValueError, match="already stored"):
        records.store_bands(svc.project, outcome)
    second = B.execute(spec, env, run_id="r2")
    second.arrays["distance"].data[1] += 1.0
    with pytest.raises(ValueError, match="changed after it was verified"):
        records.store_bands(svc.project, second)
    failed = B.BandStructureOutcome(run_id="r3", spec=spec, status="failed", reason="x")
    with pytest.raises(ValueError, match="not stored"):
        records.store_bands(svc.project, failed)
    assert unchanged(before, snapshot(svc.project))


def test_export_writes_full_precision_csv(svc, tmp_path):
    out = svc.dft_bands_run({**GS, **BANDS}, background=False)
    written = svc.dft_bands_export(str(tmp_path / "bands.csv"), out["run_id"])
    lines = open(written["path"]).read().splitlines()
    comments = [line for line in lines if line.startswith("#")]
    table = [line for line in lines if not line.startswith("#")]
    assert out["spec_digest"] in comments[0]
    header = table[0].split(",")
    assert header[:8] == ["spin", "k_index", "distance_invA", "k1_frac", "k2_frac", "k3_frac",
                          "label", "branch_start"]
    assert header[8:] == [f"band_{n}_E-E_F_eV" for n in range(1, 7)]
    rows = [r.split(",") for r in table[1:]]
    assert len(rows) == written["rows"] == 8
    assert rows[0][6] == "G" and rows[4][6] == "X" and rows[7][6] == "L"
    stored = svc.lab.dft.bands_array(out["run_id"], "eigenvalues").data
    assert float(rows[3][8]) == pytest.approx(stored[0, 3, 0] - FERMI_EV, abs=1e-9)


def test_http_routes(svc, tmp_path):
    port = find_free_port(9981)
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
        spec = post("dft/bands/spec", {"variables": {**GS}})["data"]
        assert spec["ok"] and spec["spec"]["settings"]["path"] == "GXWKGLUWLK,UX"
        assert spec["spec"]["ticks"][-2]["label"] == "K|U"
        refused = post("dft/bands/spec", {"variables": {**GS, "path": "GQ"}})["data"]
        assert refused["ok"] is False and "path" in refused["by_field"]
        run = post("dft/bands/run", {"variables": {**GS, **BANDS}, "background": False})["data"]
        assert run["ok"]
        runs = post("dft/bands/runs")["data"]["runs"]
        assert runs[0]["run_id"] == run["run_id"] and runs[0]["state"] == "current"
        result = post("dft/bands/result", {"run_id": run["run_id"]})["data"]
        assert result["n_kpoints"] == 8
        exported = post("dft/bands/export", {"path": str(tmp_path / "b.csv"),
                                             "run_id": run["run_id"]})["data"]
        assert exported["rows"] == 8
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_shipped_bands_example_runs(svc):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "examples" / "scripts"
              / "16_dft_band_structure.py").read_text()
    out = svc.runner.run(source)
    assert out.ok, out.error or out.traceback
    assert "Si band structure" in out.stdout and "state: current" in out.stdout
