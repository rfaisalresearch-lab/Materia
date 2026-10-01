"""Electrostatics through the service, the HTTP layer and the Python API."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.desktop_ui.server import create_server, find_free_port
from materia.desktop_ui.service import Service
from materia.project_format.project import Project

FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def gaas_bulk(repeat=1):
    a = 5.65325
    s = Structure([31] * 4 + [33] * 4, np.vstack([FCC * a, (FCC + 0.25) * a]),
                  Cell.cubic(a))
    return s.repeat(repeat, repeat, repeat) if repeat > 1 else s


@pytest.fixture
def svc(tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(gaas_bulk(), activate=True)
    return service


def assign(service, charges=None, source="integration test"):
    return service.electrostatics_assign("per-element", charges or {"Ga": 1.0, "As": -1.0},
                                         source=source)


def wait(service, job_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.job_status(job_id)
        if status["status"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_status_blocks_until_charges_are_assigned(svc):
    status = svc.electrostatics_status()
    assert not status["ready"]
    assert "No point-charge model" in status["blocking"][0]
    assert status["geometry"]["kind"] == "bulk-3d"
    refused = svc.electrostatics_run(background=False)
    assert refused["ok"] is False
    assert not any(k.startswith("electrostatics::") for k in svc.project.results)


def test_incomplete_charge_table_is_refused_without_changing_anything(svc):
    before = len(svc.project.history.log)
    out = svc.electrostatics_assign("per-element", {"Ga": 1.0}, source="")
    assert out["ok"] is False and "As" in out["error"]
    assert len(svc.project.history.log) == before
    assert svc.electrostatics_status()["charge_model"] is None


def test_assign_run_and_inspect(svc):
    out = assign(svc)
    assert out["ok"] and out["charge_accounting"]["total_charge_e"] == 0.0
    payload = svc.electrostatics_run(background=False)
    assert payload["ok"] and payload["current"]
    assert payload["origin"] == "calculated"
    r_nn = 5.65325 * np.sqrt(3) / 4
    madelung = -payload["energy_eV"] / 4 * r_nn / 14.3996454784
    assert madelung == pytest.approx(1.638055, abs=1e-6)
    assert {g["group"] for g in payload["site_summary"]} == {"As -1 e", "Ga +1 e"}
    atom_id = int(svc.structure.ids[0])
    site = svc.inspect_atom(atom_id)["electrostatics"]
    assert site["point_charges"] == 1.0
    assert site["site_potential"] == pytest.approx(-1.638055 * 14.3996454784 / r_nn, rel=1e-5)
    assert svc.project.history.log[-1].operation == "electrostatics.compute"


def test_edits_make_results_stale_and_undo_restores_them(svc):
    assign(svc)
    svc.electrostatics_run(background=False)
    atom_id = int(svc.structure.ids[0])
    svc.edit("move", atom_id=atom_id, delta_A=[0.2, 0.0, 0.0])
    assert svc.electrostatics_result()["current"] is False
    assert svc.inspect_atom(atom_id)["electrostatics"] is None
    svc.undo()
    assert svc.electrostatics_result()["current"] is True
    assert svc.inspect_atom(atom_id)["electrostatics"] is not None


def test_charge_model_assignment_is_undoable(svc):
    assign(svc)
    assert svc.electrostatics_status()["charge_model"]["kind"] == "per-element"
    svc.undo()
    assert svc.electrostatics_status()["charge_model"] is None
    svc.redo()
    assert svc.electrostatics_status()["charge_model"]["by_element"] == {"As": -1.0, "Ga": 1.0}
    svc.electrostatics_clear()
    assert svc.electrostatics_status()["charge_model"] is None
    svc.undo()
    assert svc.electrostatics_status()["charge_model"] is not None


def test_changing_the_charges_makes_results_stale(svc):
    assign(svc)
    svc.electrostatics_run(background=False)
    assign(svc, {"Ga": 0.5, "As": -0.5})
    assert svc.electrostatics_result()["current"] is False


def test_save_reopen_reproduces_the_result(svc, tmp_path):
    assign(svc)
    first = svc.electrostatics_run(background=False)
    path = svc.save_project(str(tmp_path / "es.materia"))["path"]
    reopened = Service(Project.load(path), recovery_dir=str(tmp_path / "r2"))
    stored = reopened.electrostatics_result()
    assert stored["energy_eV"] == first["energy_eV"]
    assert stored["current"] is True
    assert reopened.electrostatics_status()["charge_model"]["source"] == "integration test"
    again = reopened.electrostatics_run(background=False)
    assert again["energy_eV"] == first["energy_eV"]


def test_background_job_can_be_cancelled_and_stores_no_number(svc):
    svc.project.structures[svc.project.active_structure_key] = gaas_bulk(5)
    assign(svc)
    started = svc.electrostatics_run(settings={"accuracy": 1e-10})
    job_id = started["job"]["id"]
    deadline = time.time() + 30
    while svc.job_status(job_id)["status"] == "queued" and time.time() < deadline:
        time.sleep(0.005)
    svc.cancel_job(job_id)
    final = wait(svc, job_id)
    assert final["status"] == "cancelled"
    energy = svc.project.results[f"electrostatics::{started['run_id']}::energy"]
    assert not energy.supported and energy.extra["status"] == "cancelled"
    assert energy.value is None


def test_background_job_completes(svc):
    assign(svc)
    started = svc.electrostatics_run()
    final = wait(svc, started["job"]["id"])
    assert final["status"] == "done"
    assert final["result"]["ok"] and final["result"]["run_id"] == started["run_id"]


def test_result_belongs_to_the_project_that_started_it(svc):
    svc.project.structures[svc.project.active_structure_key] = gaas_bulk(4)
    assign(svc)
    original = svc.project
    gate = threading.Event()
    real = svc._run_electrostatics

    def slow(*args, **kwargs):
        gate.wait(10)
        return real(*args, **kwargs)

    svc._run_electrostatics = slow
    started = svc.electrostatics_run()
    svc.new_project("Replacement")
    gate.set()
    final = wait(svc, started["job"]["id"])
    assert not any(k.startswith("electrostatics::") for k in svc.project.results)
    if final["status"] == "done":
        assert any(k.startswith("electrostatics::") for k in original.results)


def test_slab_run_through_the_service(svc):
    slab = gaas_bulk(2)
    slab.cell = Cell(slab.cell.matrix, (True, True, False))
    svc.project.structures[svc.project.active_structure_key] = slab
    assign(svc)
    payload = svc.electrostatics_run(background=False)
    assert payload["ok"]
    assert payload["geometry"]["kind"] == "slab-2d"
    assert "internal_vacuum_gap_A" in payload["parameters"]


def test_charged_slab_is_refused_through_the_service(svc):
    slab = gaas_bulk()
    slab.cell = Cell(slab.cell.matrix, (True, True, False))
    svc.project.structures[svc.project.active_structure_key] = slab
    assign(svc, {"Ga": 1.0, "As": -0.5})
    out = svc.electrostatics_run(background=False)
    assert out["ok"] is False and "charged" in out["error"]


def test_generic_solve_refuses_relaxation_with_the_electrostatics_model(svc):
    out = svc.solve("relax", "electrostatics/ewald", background=False)
    assert out["supported"] is False
    assert "relax" in out["reason"]


def test_python_script_drives_electrostatics(svc):
    code = (
        "electrostatics.assign(by_element={'Ga': 1, 'As': -1}, source='script')\n"
        "run = electrostatics.compute()\n"
        "print(round(run['energy'].value, 6), run['energy'].unit)\n"
        "print(electrostatics.runs()[0]['origin'])\n"
        "print(electrostatics.result(quantity='site_potential').unit)\n"
        "undo = project.undo()\n"
        "print(electrostatics.charge_model())\n"
    )
    out = svc.runner.run(code)
    assert out.ok, out.error
    lines = out.stdout.strip().splitlines()
    assert lines[0].endswith(" eV")
    assert lines[1] == "calculated"
    assert lines[2] == "V"
    assert lines[3] == "None"


def test_http_routes(tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(gaas_bulk(), activate=True)
    port = find_free_port(9791)
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
        assert post("electrostatics/status")["ready"] is False
        assigned = post("electrostatics/assign", {"kind": "per-element",
                                                  "by_element": {"Ga": 1, "As": -1},
                                                  "source": "http"})
        assert assigned["ok"]
        run = post("electrostatics/run", {"background": False})
        assert run["ok"] and run["current"]
        assert post("electrostatics/result", {"run_id": run["run_id"]})["energy_eV"] == run["energy_eV"]
        assert post("electrostatics/clear")["removed"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_shipped_electrostatics_example_runs(svc):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "examples" / "scripts"
              / "09_electrostatics.py").read_text()
    out = svc.runner.run(source)
    assert out.ok, out.error or out.traceback
    assert "Madelung constant from the energy: 1.6380551" in out.stdout
    assert "charged slab supported: False" in out.stdout
