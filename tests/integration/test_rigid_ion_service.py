"""The rigid-ion model through the service, the HTTP layer and the Python API."""

from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.desktop_ui.server import create_server, find_free_port
from materia.desktop_ui.service import Service
from materia.materials import default_library
from materia.physics import rigid_ion as R
from materia.project_format.project import Project
from materia.structure_builder.lattice import bulk


def quartz():
    return bulk(default_library().get("silicon_dioxide"))


@pytest.fixture
def svc(tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(quartz(), activate=True)
    yield service
    service.shutdown(grace_s=1.0)


def test_recommended_model_for_silica_is_the_rigid_ion_model(svc):
    energy = svc.solve(task="energy", model="recommended", background=False)
    assert energy["supported"], energy
    assert energy["solver"] == "classical/rigid-ion/bks-silica"
    checks = energy["checks"]
    assert checks["ewald_converged"] is True
    assert checks["components_eV"]["short_range"] + checks["components_eV"]["coulomb"] \
        == pytest.approx(energy["energy_eV"], rel=1e-9)
    assert checks["barrier_margin_A"]["O-Si"] > 0.3
    assert energy["provenance"]["parameters"]["parameter_digest"] == R.BKS_SILICA.digest()


def test_relax_is_one_undoable_change_and_survives_save_and_reopen(svc, tmp_path):
    before = svc.structure.positions.copy()
    relax = svc.solve(task="relax", model="rigid-ion/bks-silica", fmax=1e-3, steps=2000,
                      background=False)
    assert relax["supported"] and relax["converged"], relax
    assert relax["energy_change_eV"] < 0
    assert 0 < relax["max_displacement_A"] < 0.1
    assert relax["checks"]["ewald_converged"] is True
    relaxed = svc.structure.positions.copy()
    assert not np.allclose(relaxed, before)
    path = svc.save_project(str(tmp_path / "quartz.materia"))["path"]
    reopened = Service(Project.load(path), recovery_dir=str(tmp_path / "r2"))
    try:
        assert np.allclose(reopened.structure.positions, relaxed, atol=1e-12)
        stored = reopened.project.results["relax::energy"]
        assert stored.value == pytest.approx(relax["energy_eV"], rel=1e-12)
        assert stored.extra["potential_checks"]["ewald_converged"] is True
    finally:
        reopened.shutdown(grace_s=1.0)
    svc.undo()
    assert np.allclose(svc.structure.positions, before, atol=1e-12)


def test_dynamics_reports_checks_and_temperature(svc):
    md = svc.solve(task="md", model="rigid-ion/bks-silica", steps=40, temperature_K=300,
                   thermostat="langevin", seed=4, background=False)
    assert md["supported"], md
    assert md["mean_temperature_K"] > 0
    assert md["checks"]["ewald_converged"] is True


def test_a_collapsed_configuration_is_refused_and_nothing_changes(svc):
    s = svc.structure
    si = int(np.flatnonzero(s.numbers == 14)[0])
    nl_o = int(np.flatnonzero(s.numbers == 8)[0])
    direction = s.positions[nl_o] - s.positions[si]
    s.positions[nl_o] = s.positions[si] + direction / np.linalg.norm(direction) * 1.1
    before = s.positions.copy()
    results_before = set(svc.project.results)
    out = svc.solve(task="relax", model="rigid-ion/bks-silica", background=False)
    assert out["supported"] is False
    assert "Buckingham catastrophe" in out["reason"]
    assert np.array_equal(svc.structure.positions, before)
    assert set(svc.project.results) == results_before


def test_other_compositions_are_refused_with_the_reason(svc):
    s = svc.structure
    s.numbers[0] = 13
    out = svc.solve(task="relax", model="rigid-ion/bks-silica", background=False)
    assert out["supported"] is False and "no parameters for Al" in out["reason"]


def test_http_solve_and_python_api(svc):
    port = find_free_port()
    server = create_server(svc, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        solvers = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/api/solvers", data=b"{}",
            headers={"Content-Type": "application/json"})).read())["data"]
        assert any(d["key"] == "rigid-ion/bks-silica" for d in solvers["registered"])
        body = json.dumps({"task": "energy", "model": "rigid-ion/bks-silica",
                           "background": False}).encode()
        reply = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/api/solve", data=body,
            headers={"Content-Type": "application/json"})).read())["data"]
        assert reply["supported"] and reply["checks"]["ewald_converged"] is True
    finally:
        server.shutdown()
        server.server_close()
    result = svc.lab.relax(svc.structure, model="recommended", fmax=1e-3, steps=2000)
    assert result.provenance.model == "classical/rigid-ion/bks-silica"
    assert result.convergence.converged


def test_every_model_the_solver_panel_offers_resolves(svc):
    """The panel sends ``key``; before it sent the instance name, which never resolved."""
    offered = [d for d in svc.solvers()["registered"] if d.get("available", True)]
    assert len({d["key"] for d in offered}) == len(offered)
    for d in offered:
        solver = svc.lab._resolve_solver(d["key"], svc.structure, None, "energy")
        assert solver.name == d["name"]


def test_shipped_rigid_ion_example_runs(svc):
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "examples" / "scripts"
              / "19_rigid_ion_silica.py").read_text()
    out = svc.runner.run(source)
    assert out.ok, out.error or out.traceback
    assert "classical/rigid-ion/bks-silica" in out.stdout
    assert "Independent split agrees" in out.stdout
    assert "supported: False" in out.stdout
    assert "refused:" in out.stdout and "Buckingham catastrophe" in out.stdout
