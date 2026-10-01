"""Plug-in loading and the application service layer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from materia.plugin_system import PluginManager
from materia.desktop_ui.service import Service
from materia.version import PLUGIN_API_VERSION


def test_repository_example_plugin_loads():
    manager = PluginManager()
    loaded = manager.load_all()
    example = next((p for p in loaded if p.id == "example_plugin"), None)
    assert example is not None
    assert example.ok, example.error
    assert example.api_version == PLUGIN_API_VERSION
    provided = " ".join(example.provided)
    assert "solver:" in provided
    assert "palette:" in provided
    assert "material search path:" in provided


def test_plugin_registers_its_material_and_solver():
    PluginManager().load_all()
    from materia.materials import default_library
    from materia.solvers import available
    from materia.visualization.palette import PALETTES

    default_library().scan(force=True)
    assert "silicon_strained_1pct" in default_library().ids()
    assert "example/einstein-solid" in available()
    assert "example-teal" in PALETTES


def test_plugin_material_declares_that_it_is_hypothetical():
    from materia.materials import default_library
    PluginManager().load_all()
    default_library().scan(force=True)
    definition = default_library().get("silicon_strained_1pct")
    assert "not a measured structure" in definition.provenance_note
    assert "NOT applied" in definition.properties["band_gap"].note


def test_plugin_with_wrong_api_version_is_refused(tmp_path):
    (tmp_path / "old_plugin.py").write_text(
        'PLUGIN = {"id": "old", "name": "Old", "version": "0.1", "api_version": "0.5"}\n'
        "def register(ctx):\n    raise RuntimeError('must not run')\n")
    manager = PluginManager([tmp_path])
    loaded = [p for p in manager.load_all() if p.id == "old_plugin"]
    assert loaded and not loaded[0].ok
    assert "Not loaded" in loaded[0].error


def test_broken_plugin_is_reported_not_raised(tmp_path):
    (tmp_path / "broken_plugin.py").write_text(
        'PLUGIN = {"id": "broken", "api_version": "1.0"}\n'
        "def register(ctx):\n    raise ValueError('deliberate failure')\n")
    manager = PluginManager([tmp_path])
    loaded = [p for p in manager.load_all() if p.id == "broken_plugin"]
    assert loaded and not loaded[0].ok
    assert "deliberate failure" in loaded[0].error


def test_service_state_reports_units_and_provenance(service):
    service.create_wafer(material_id="silicon", orientation=(1, 1, 1))
    state = service.state()
    assert state["wafer"]["provenance"]["origin"] == "estimated"
    assert "synthetic" in state["wafer"]["provenance"]["notes"].lower() or \
        any("procedurally" in a for a in state["wafer"]["provenance"]["approximations"])
    assert state["scales"][0]["level"] == "wafer"


def test_service_autosaves_and_recovers(tmp_path):
    recovery_dir = str(tmp_path / "recovery")
    first = Service(recovery_dir=recovery_dir)
    first.create_wafer(material_id="silicon", orientation=(1, 1, 1))
    first.extract_region(size_nm=(1.0, 1.0), depth_layers=2, max_atoms=1000)
    candidate = first.recovery.info()
    assert candidate["available"] is True

    second = Service(recovery_dir=recovery_dir)
    assert second.state()["recovery"]["available"] is True
    restored = second.recover_project()
    assert restored["wafer"]["spec"]["material_id"] == "silicon"
    assert restored["structure"]["n_atoms"] > 0
    assert restored["recovery"]["available"] is False


def test_explicit_save_clears_recovery(tmp_path):
    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.create_wafer(material_id="silicon")
    assert service.recovery.info()["available"] is True
    saved = service.save_project(str(tmp_path / "saved"))
    assert saved["path"].endswith("saved.materia")
    assert service.recovery.info()["available"] is False


def test_service_rejects_unknown_settings_by_name(service):
    service.create_wafer(material_id="silicon", orientation=(1, 1, 1))
    service.extract_region(size_nm=(1.5, 1.5), depth_layers=3)
    with pytest.raises(ValueError) as excinfo:
        service.scan("stm", {"bias_volts": 1.0}, background=False)
    assert "bias_volts" in str(excinfo.value)
    assert "Accepted" in str(excinfo.value)


def test_service_warns_when_a_structure_becomes_charged(service):
    service.create_wafer(material_id="silicon", orientation=(1, 1, 1))
    service.extract_region(size_nm=(1.5, 1.5), depth_layers=3)
    service.select("ids", ids=[int(service.structure.ids[0])])
    service.edit("set_charge", charge=1.0)
    warnings = [w["text"] for w in service.state()["warnings"]]
    assert any("net charge" in w for w in warnings)


def test_service_edit_rejects_an_unknown_operation(service):
    service.create_wafer(material_id="silicon", orientation=(1, 1, 1))
    service.extract_region(size_nm=(1.5, 1.5), depth_layers=3)
    with pytest.raises(ValueError) as excinfo:
        service.edit("teleport")
    assert "substitute" in str(excinfo.value)


def test_atom_inspection_payload_is_complete(service):
    service.create_wafer(material_id="silicon", orientation=(1, 1, 1))
    service.extract_region(size_nm=(1.5, 1.5), depth_layers=3)
    atom_id = int(service.structure.ids[0])
    payload = service.inspect_atom(atom_id)
    for section in ("identity", "nucleus", "electrons", "environment", "energy"):
        assert section in payload
    assert payload["electrons"]["origin"] in ("reference", "estimated")
    assert payload["energy"]["note"]
    assert payload["nucleus"]["note"]


def test_job_queue_runs_and_reports(service):
    service.create_wafer(material_id="silicon", orientation=(1, 1, 1))
    service.extract_region(size_nm=(1.2, 1.2), depth_layers=2)
    submitted = service.solve(task="energy", background=True)
    job_id = submitted["job"]["id"]
    for _ in range(200):
        status = service.job_status(job_id)
        if status["status"] in ("done", "failed"):
            break
        import time
        time.sleep(0.05)
    assert status["status"] == "done"
    assert status["result"]["supported"] is True


def test_instrument_options_expose_every_mode(service):
    options = service.instrument_options()
    assert "constant-current" in options["stm_modes"]
    assert set(options["afm_modes"]) == {"contact", "constant-height", "fm-afm",
                                         "constant-frequency-shift"}
    assert "silver" == options["palettes"][0]["id"]
    assert "quiet" in options["noise_presets"]


def test_trusted_script_mode_requires_confirmation(service):
    response = service.set_script_mode("trusted", confirm=False)
    assert response["ok"] is False
    assert response["requires_confirmation"] is True
    assert "not a security boundary" in response["message"]
    assert service.runner.mode == "restricted"

    response = service.set_script_mode("trusted", confirm=True)
    assert response["ok"] is True
    assert service.runner.mode == "trusted"
    operations = [e.operation for e in service.project.history.log]
    assert "script.mode" in operations
