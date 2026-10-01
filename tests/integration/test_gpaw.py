"""GPAW through every layer Materia exposes it on.

The cases that need a running GPAW are skipped when there is none, but the
refusal path is checked either way, because declining clearly is what the
interface does most of the time on a machine without a first-principles code.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from materia.desktop_ui.service import Service
from materia.project_format import Project
from materia.python_api import Lab


@pytest.fixture
def service():
    return Service()


def slab(service):
    service.build_surface("silicon", [1, 1, 1], size=[1, 1, 2], vacuum_A=8.0)
    return service.structure


FAST = {"preset": "surface", "xc": "LDA", "grid_spacing_A": 0.25,
        "kpoints": [1, 1, 1], "energy_tol_eV_per_electron": 5e-3,
        "max_iterations": 60}


class TestStatusIsHonest:

    def test_the_service_reports_code_and_datasets_separately(self, service):
        status = service.gpaw_status()
        environment = status["environment"]
        assert "code_available" in environment
        assert "datasets_available" in environment
        assert environment["operational"] == (
            environment["code_available"] and environment["datasets_available"])
        assert environment["install_hint"]
        json.dumps(status)

    def test_presets_are_offered_with_notes(self, service):
        names = {p["name"] for p in service.gpaw_presets()}
        assert {"smoke", "molecule", "surface"} <= names
        assert all(p["note"] for p in service.gpaw_presets())

    def test_the_solver_list_marks_which_adapters_are_actually_driven(self, service):
        external = {row["name"]: row for row in service.solvers()["external"]}
        assert external["external:gpaw"]["driven"] is True
        assert external["external:gpaw"]["capabilities"] == ["energy", "forces"]
        assert external["external:quantum-espresso"]["driven"] is False


class TestConfigurationIsValidatedBeforeAnyJob:

    def test_a_bad_setting_is_refused_without_creating_a_job(self, service):
        slab(service)
        before = len(service.jobs.list(50))
        out = service.gpaw_energy(preset="surface", mode="pw", background=False)
        assert out["ok"] is False
        assert "periodic boundaries" in out["error"]
        assert len(service.jobs.list(50)) == before

    def test_settings_are_checked_against_the_active_structure(self, service):
        slab(service)
        assert service.gpaw_settings("surface", mode="pw")["ok"] is False
        good = service.gpaw_settings("surface", kpoints=[2, 2, 1])
        assert good["ok"] is True
        assert good["units"]["grid_spacing_A"] == "A"
        assert "PBE" in good["summary"]


class TestRefusalWithoutGPAW:

    def test_an_absent_gpaw_refuses_through_the_solver(self):
        from materia.solvers.gpaw_driver import GPAWEnvironment, GPAWSolver
        from materia.solvers.gpaw_driver.environment import INTERPRETER_ENV_VAR

        service = Service()
        structure = slab(service)
        solver = GPAWSolver(environment=GPAWEnvironment(interpreter=None))
        out = solver.single_point(structure)
        refusal = out.results["energy"]
        assert refusal.value is None
        assert INTERPRETER_ENV_VAR in refusal.unsupported_reason
        assert np.allclose(structure.positions, slab(Service()).positions)


class TestRealRuns:

    def test_python_api_end_to_end(self, needs_gpaw):
        lab = Lab()
        surface = lab.materials.load("silicon").create_surface(
            size=(1, 1, 2), orientation=(1, 1, 1), vacuum_angstrom=8.0)
        assert lab.dft.available() is True
        settings = lab.dft.settings("surface", surface, xc="LDA",
                                    grid_spacing_A=0.25, kpoints=(1, 1, 1),
                                    energy_tol_eV_per_electron=5e-3,
                                    max_iterations=60)
        energy = lab.dft.energy(surface, settings=settings)
        assert energy.supported
        assert energy.unit == "eV"
        assert energy.provenance.fidelity.value == "tier3-external-first-principles"
        keys = sorted(lab.project.results)
        assert len(keys) == 2
        run_id = energy.extra["run_id"]
        assert keys == [f"gpaw::{run_id}::energy", f"gpaw::{run_id}::forces"]
        assert energy.provenance.inputs_digest

    def test_applying_forces_is_one_undoable_change(self, needs_gpaw):
        lab = Lab()
        surface = lab.materials.load("silicon").create_surface(
            size=(1, 1, 2), orientation=(1, 1, 1), vacuum_angstrom=8.0)
        project = lab.project
        key = project.active_structure_key
        before = len(project.history.log)

        lab.dft.energy(surface, preset_name="surface", xc="LDA",
                       grid_spacing_A=0.25, kpoints=(1, 1, 1),
                       energy_tol_eV_per_electron=5e-3, max_iterations=60,
                       apply_forces=True)
        assert np.isfinite(surface.structure.forces).all()
        assert len(project.history.log) == before + 1
        assert project.history.log[-1].operation == "solver.gpaw.forces"

        project.undo()
        assert not np.isfinite(project.structures[key].forces).any()
        project.redo()
        assert np.isfinite(project.structures[key].forces).all()

    def test_service_run_and_project_persistence(self, needs_gpaw, service, tmp_path):
        slab(service)
        out = service.gpaw_energy(background=False, **FAST)
        assert out["ok"] is True
        assert out["converged"] is True
        assert out["status"] == "converged"
        assert out["iterations"] > 0
        assert out["energy_eV"] < 0
        assert out["max_force_eV_A"] >= 0
        json.dumps(out)

        path = str(tmp_path / "dft.materia")
        service.save_project(path)
        reloaded = Project.load(path)
        run_id = out["run_id"]
        assert sorted(reloaded.results) == [f"gpaw::{run_id}::energy",
                                            f"gpaw::{run_id}::forces"]
        energy = reloaded.results[f"gpaw::{run_id}::energy"]
        assert energy.provenance.inputs_digest == out["inputs_digest"]
        assert energy.value == pytest.approx(out["energy_eV"])
        assert energy.provenance.origin.value == "calculated"
        assert energy.provenance.fidelity.value == "tier3-external-first-principles"
        assert energy.provenance.parameters["gpaw_version"]
        assert energy.provenance.parameters["paw_datasets"][0]["fingerprint"]
        assert energy.provenance.dataset
        assert energy.convergence.converged is True

    def test_a_background_job_reports_progress_and_can_be_cancelled(
            self, needs_gpaw, service):
        slab(service)
        submitted = service.gpaw_energy(
            background=True, preset="surface", xc="LDA", grid_spacing_A=0.16,
            kpoints=[2, 2, 1], energy_tol_eV_per_electron=1e-10,
            max_iterations=400)
        job_id = submitted["job"]["id"]

        deadline = time.time() + 60
        while time.time() < deadline:
            status = service.job_status(job_id, include_result=False)
            if status["status"] == "running" and "SCF" in (status["message"] or ""):
                break
            if status["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.2)

        assert service.cancel_job(job_id)["cancelled"] is True
        deadline = time.time() + 60
        while time.time() < deadline:
            status = service.job_status(job_id, include_result=True)
            if status["status"] in ("done", "cancelled", "failed"):
                break
            time.sleep(0.2)
        assert status["status"] in ("cancelled", "done")
        result = status.get("result")
        if result:
            assert result["ok"] is False
            assert result["status"] == "cancelled"
            assert "cancelled" in result["reason"]

    def test_http_routes(self, needs_gpaw):
        from materia.desktop_ui.server import create_server, find_free_port
        import urllib.request

        port = find_free_port(9781)
        httpd = create_server(port=port)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"

        def post(route, payload=None):
            request = urllib.request.Request(
                f"{base}/api/{route}", method="POST",
                data=json.dumps(payload or {}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())

        try:
            status = post("gpaw/status")["data"]
            assert status["environment"]["operational"] is True
            assert {p["name"] for p in status["presets"]} >= {"smoke", "surface"}

            refused = post("gpaw/settings", {"preset": "surface", "mode": "pw"})["data"]
            post("project/new", {"name": "gpaw over http"})
            post("structure/build", {"material": "silicon", "miller": [1, 1, 1],
                                     "size": [1, 1, 2], "vacuum_A": 8.0})
            refused = post("gpaw/settings", {"preset": "surface", "mode": "pw"})["data"]
            assert refused["ok"] is False

            run = post("gpaw/energy", dict(FAST, background=False))["data"]
            assert run["ok"] is True
            assert run["converged"] is True
            assert run["energy_eV"] < 0
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_non_converged_run_is_reported_as_such_by_the_service(
            self, needs_gpaw, service):
        slab(service)
        out = service.gpaw_energy(
            background=False, preset="surface", xc="LDA", grid_spacing_A=0.25,
            kpoints=[1, 1, 1], energy_tol_eV_per_electron=1e-12,
            density_tol_electrons=1e-12, max_iterations=3)
        assert out["ok"] is False
        assert out["status"] == "not-converged"
        assert out["converged"] is False
        assert "did not reach self-consistency" in out["reason"]
        assert "energy_eV" not in out
        assert any(w["level"] == "unsupported" for w in service.warnings)


class TestChargeAndSpinAreRefusedEverywhere:
    """The same refusal, whichever layer asks."""

    def charged(self, service):
        structure = slab(service)
        structure.formal_charges[0] = 1.0
        return structure

    def test_the_service_refuses_a_requested_charge(self, service):
        slab(service)
        out = service.gpaw_energy(background=False, charge=1.0, **FAST)
        assert out["ok"] is False
        assert "Charged systems are not supported" in out["error"]

    def test_the_service_refuses_spin_polarisation(self, service):
        slab(service)
        out = service.gpaw_energy(background=False, spin_polarized=True, **FAST)
        assert out["ok"] is False
        assert "Spin-polarised calculations are not supported" in out["error"]

    def test_the_service_refuses_a_charged_structure(self, service):
        self.charged(service)
        out = service.gpaw_energy(background=False, **FAST)
        assert out["ok"] is False
        assert "net formal charge" in out["error"]

    def test_settings_validation_refuses_before_any_job(self, service):
        slab(service)
        before = len(service.jobs.list(50))
        assert service.gpaw_settings(charge=1.0, **FAST)["ok"] is False
        assert service.gpaw_settings(spin_polarized=True, **FAST)["ok"] is False
        assert service.gpaw_settings(spin_polarized="false", **FAST)["ok"] is False
        assert len(service.jobs.list(50)) == before

    def test_the_python_api_refuses(self):
        from materia.solvers.gpaw_driver import GPAWSettingsError

        lab = Lab()
        surface = lab.materials.load("silicon").create_surface(
            size=(1, 1, 2), orientation=(1, 1, 1), vacuum_angstrom=8.0)
        with pytest.raises(GPAWSettingsError, match="(?i)charged systems"):
            lab.dft.settings("surface", surface, charge=1.0)
        with pytest.raises(GPAWSettingsError, match="(?i)spin-polarised"):
            lab.dft.energy(surface, preset_name="surface", spin_polarized=True)

    def test_http_refuses(self, needs_gpaw):
        from materia.desktop_ui.server import create_server, find_free_port
        import urllib.request

        port = find_free_port(9791)
        httpd = create_server(port=port)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"

        def post(route, payload=None):
            request = urllib.request.Request(
                f"{base}/api/{route}", method="POST",
                data=json.dumps(payload or {}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())

        try:
            post("project/new", {"name": "charge refusal"})
            post("structure/build", {"material": "silicon", "miller": [1, 1, 1],
                                     "size": [1, 1, 2], "vacuum_A": 8.0})
            out = post("gpaw/energy", dict(FAST, background=False, charge=1.0))["data"]
            assert out["ok"] is False
            assert "Charged systems are not supported" in out["error"]
            spin = post("gpaw/settings", dict(FAST, spin_polarized=True))["data"]
            assert spin["ok"] is False
        finally:
            httpd.shutdown()
            httpd.server_close()
