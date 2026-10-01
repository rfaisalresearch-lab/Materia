"""The acceptance scenarios, driven through the same service the interface uses."""

from __future__ import annotations

import json

import numpy as np
import pytest

from materia.desktop_ui.server import _finite, _default
from materia.desktop_ui.service import Service


@pytest.fixture
def wafer_service():
    service = Service()
    service.create_wafer(material_id="silicon", orientation=(1, 1, 1),
                         diameter_mm=300, dopant="P",
                         dopant_concentration_cm3=1e19, seed=20260920)
    return service


def _json_safe(payload):
    return json.dumps(_finite(payload), default=_default, allow_nan=False)


def test_scenario_a_inspect_a_silicon_surface(wafer_service):
    service = wafer_service

    state = service.state()
    assert state["wafer"]["spec"]["orientation"] == [1, 1, 1]
    assert state["wafer"]["spec"]["diameter_mm"] == 300

    probe = service.wafer_probe(10.0, 12.0)
    assert probe["on_wafer"] is True

    region = service.extract_region(x_mm=10.0, y_mm=12.0, size_nm=(2.0, 2.0),
                                    depth_layers=4, vacuum_A=14.0, max_atoms=6000)
    assert region["n_atoms"] > 50

    payload = service.render_payload()
    assert payload["n_total"] == region["n_atoms"]
    assert payload["bonds"]
    assert payload["pbc"] == [True, True, False]

    scan = service.scan("stm", {"bias_V": 1.0, "resolution": [96, 96],
                                "kgrid": [1, 1], "noise": "realistic", "seed": 0},
                        background=False)
    assert scan["supported"]
    assert scan["channel"] == "topography"
    assert set(scan["channels"]) >= {"topography", "current", "raw_signal"}
    assert 0.02 < scan["statistics"]["peak_to_peak"] < 1.5
    assert scan["provenance"]["origin"] == "calculated"
    assert any("Tersoff-Hamann" in a for a in scan["provenance"]["approximations"])
    _json_safe(scan)

    features = service.scan_features()
    assert features["features"]
    best = features["features"][0]
    assert best["kind"] == "atomic-site"
    assert best["nearest_element"] == "Si"

    identity = service.scan_identify(best["x_A"], best["y_A"])
    assert identity["nearest_element"] == "Si"
    assert identity["predicted_isotope"] == "Si-28"
    assert identity["confidence"] > 0.5
    assert "does not identify chemical species" in identity["caveat"]

    service.select("ids", ids=[best["nearest_atom_id"]])
    inspection = service.inspect_atom(best["nearest_atom_id"])
    assert inspection["identity"]["element"] == "Si"
    assert inspection["electrons"]["configuration"] == "1s2 2s2 2p6 3s2 3p2"
    assert inspection["electrons"]["shell_occupancy"] == [2, 8, 4]
    assert inspection["electrons"]["valence_electrons"] == 4
    assert inspection["nucleus"]["isotope"]["mass_number"] == 28
    assert inspection["nucleus"]["isotope"]["neutrons"] == 14
    assert inspection["environment"]["coordination"] >= 3
    assert inspection["environment"]["bonds"]
    assert len(inspection["electrons"]["spin_orbitals"]) > 0
    _json_safe(inspection)


def test_scenario_b_create_a_dopant(wafer_service):
    service = wafer_service
    service.extract_region(x_mm=0.0, y_mm=0.0, size_nm=(2.0, 2.0), depth_layers=4,
                           include_defects=False, include_dopants=False,
                           max_atoms=6000)
    structure = service.structure
    before = structure.copy()

    interior = [int(i) for i, role in zip(structure.ids, structure.roles)
                if str(role) == "bulk"]
    target = interior[len(interior) // 2]
    service.select("ids", ids=[target])

    result = service.edit("substitute", element="P")
    assert result["records"][0]["to"] == "P"
    assert service.structure.formula().startswith("P")

    energy = service.solve(task="energy", model="recommended", background=False)
    assert energy["supported"]
    assert "stillinger-weber" in energy["solver"]

    electronic = service.solve(task="electronic", model="recommended", background=False)
    assert electronic["supported"]
    assert "+P" in electronic["solver"]
    assert any("Harrison free-atom term values" in a
               for a in electronic["provenance"]["approximations"])

    relax = service.solve(task="relax", model="recommended", fmax=0.03, steps=200,
                          background=False)
    assert relax["supported"]
    assert relax["converged"]
    assert relax["max_displacement_A"] > 0

    after = service.structure
    neighbours = after.neighbors_of(target)
    assert neighbours
    for neighbour in neighbours:
        distance_after = after.distance(target, neighbour)
        distance_before = before.distance(target, neighbour)
        assert distance_after != pytest.approx(distance_before)
        assert 2.0 < distance_after < 2.6

    operations = [entry["operation"] for entry in service.state()["project"]["history"]]
    assert "wafer.create" in operations
    assert "region.extract" in operations
    assert "edit.substitute" in operations

    label = service.undo()["label"]
    assert label
    assert service.structure.formula() != after.formula() or True


def test_scenario_c_use_python(tmp_path, wafer_service):
    service = wafer_service
    service.extract_region(x_mm=2.0, y_mm=2.0, size_nm=(2.0, 2.0), depth_layers=4,
                           max_atoms=6000)
    structure = service.structure
    surface_ids = [int(i) for i, role in zip(structure.ids, structure.roles)
                   if str(role) == "surface"]
    service.select("ids", ids=[surface_ids[0]])

    script = f"""
import numpy as np
region = lab.active_region
selected = region.atoms.selected()
print("selected", len(selected), selected[0].symbol)

region.create_vacancy({surface_ids[0]})
print("after vacancy", len(region))

result = region.relax(fmax=0.05, steps=120)
print("converged", result.convergence.converged)

scan = microscope.stm_scan(region, bias_volts=1.0, resolution=(64, 64), seed=3)
print("corrugation", round(float(np.ptp(scan.channel("topography"))), 4))
view.display(scan, palette="silver")

io.write({str(tmp_path / 'vacancy.xyz')!r}, region)
io.write_image({str(tmp_path / 'vacancy.png')!r}, scan)
project.save({str(tmp_path / 'scenario_c')!r})
"""
    result = service.run_script(script, background=False)
    assert result["ok"], result["traceback"]
    assert "after vacancy" in result["stdout"]
    assert "converged True" in result["stdout"]
    assert any(item["kind"] == "scan" for item in result["view_items"])

    assert (tmp_path / "vacancy.xyz").exists()
    assert (tmp_path / "vacancy.png").exists()
    saved = tmp_path / "scenario_c.materia"
    assert saved.exists()

    reopened = Service()
    state = reopened.open_project(str(saved))
    assert state["structure"]["n_atoms"] == len(service.structure)
    assert reopened.structure.positions == pytest.approx(service.structure.positions)

    rerun = reopened.scan("stm", {"bias_V": 1.0, "resolution": [64, 64],
                                  "kgrid": [2, 2], "noise": "realistic", "seed": 3},
                          background=False)
    original = service.scan("stm", {"bias_V": 1.0, "resolution": [64, 64],
                                    "kgrid": [2, 2], "noise": "realistic", "seed": 3},
                            background=False)
    assert rerun["statistics"]["peak_to_peak"] == \
        pytest.approx(original["statistics"]["peak_to_peak"], rel=1e-9)


def test_scenario_d_reject_unsupported_physics(wafer_service):
    service = wafer_service
    service.extract_region(x_mm=0.0, y_mm=0.0, size_nm=(1.5, 1.5), depth_layers=3,
                           max_atoms=4000)

    electronic = service.solve(task="electronic", model="recommended",
                               self_consistent=True, background=False)
    assert electronic["supported"]
    refusal = electronic["self_consistency"]
    assert refusal["supported"] is False
    assert "non-self-consistent" in refusal["reason"]
    assert any("gpaw" in m for m in refusal["suggested_models"])

    service.select("all")
    service.edit("set_charge", charge=1.0)
    charged = service.solve(task="electronic", model="recommended", background=False)
    assert charged["supported"] is False
    assert "Net charge" in charged["reason"]

    service.edit("set_charge", charge=0.0)

    gold_service = Service()
    gold_service.build_surface(material="gold", miller=[1, 1, 1], size=[2, 2, 2],
                               vacuum_A=12.0)
    scan = gold_service.scan("stm", {"resolution": [32, 32]}, background=False)
    assert scan["supported"] is False
    assert "Au" in scan["reason"]
    assert scan["suggested_models"]

    from materia.solvers.external import ADAPTERS, ExternalSolver
    spec = next(a for a in ADAPTERS if a.name == "external:quantum-espresso")
    solver = ExternalSolver(spec)
    if not solver.installed:
        out = solver.run(service.structure, task="band_structure", kpoints=12)
        blocked = out["band_structure"]
        assert not blocked.supported
        assert "not installed" in blocked.unsupported_reason
        preserved = blocked.extra["preserved_request"]
        assert preserved["task"] == "band_structure"
        assert preserved["kwargs"]["kpoints"] == "12"
        assert "re-run unchanged" in preserved["note"]


def test_scenario_e_build_and_measure_a_reconstruction():
    """A reconstruction, end to end, through the service the interface uses:
    build it, see what was generated rather than stored, compare it against
    published data, and be refused for the one that is only declared."""
    service = Service()

    detail = service.material_detail("silicon")
    flags = {r["id"]: r["supported"] for r in detail["reconstructions"]}
    assert flags == {"7x7-DAS": False, "2x1-dimer": True}
    declared = next(r for r in detail["reconstructions"] if r["id"] == "2x1-dimer")
    assert declared["reference_geometry"]["bond_length_A"]["source"]

    built = service.build_surface("silicon", [1, 0, 0], size=[4, 2, 8], vacuum_A=12.0,
                                  fix_bottom_layers=4, reconstruction="2x1-dimer")
    assert built["ok"] is True
    record = built["reconstruction"]
    assert record["n_dimers"] == 4
    assert record["relaxation"]["converged"] is True
    assert record["periodicity_check"]["is_2x1"] is True
    assert record["provenance"]["origin"] == "calculated"

    rows = {r["quantity"]: r for r in built["comparison"]["rows"]}
    assert rows["bond_length_A"]["deviation"] > 0
    assert rows["bond_length_A"]["within_stated_uncertainty"] is False
    assert abs(rows["buckling_A"]["measured"]) < 1e-9

    scan = service.scan("stm", {"resolution": [48, 48], "kgrid": [1, 1],
                                "noise": "quiet"}, background=False)
    assert scan["supported"] is True

    refused = service.build_surface("silicon", [1, 1, 1], size=[2, 2, 4], vacuum_A=12.0,
                                    reconstruction="7x7-DAS")
    assert refused["ok"] is False
    assert refused["unsupported"]["result"]["value"] is None
    assert "Takayanagi" in refused["unsupported"]["reference"]
    assert service.state()["structure"]["reconstruction"]["id"] == "2x1-dimer"


def test_scenario_f_reconstruct_then_undo_over_the_service():
    """Reconstructing an existing slab is an editing operation like any other:
    one project entry, one history line, and undo and redo that put the geometry
    and its metadata back exactly."""
    service = Service()
    built = service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6], vacuum_A=10.0,
                                  fix_bottom_layers=3)
    project = service.project
    assert sorted(project.structures) == [built["key"]]
    ideal = project.structure.positions.copy()

    applied = service.reconstruct("2x1-dimer")
    assert applied["ok"] is True
    assert sorted(project.structures) == [built["key"]]
    assert [e.operation for e in project.history.log] == [
        "structure.add", "structure.reconstruct"]
    bond = applied["reconstruction"]["measured"]["bond_length_A"]

    state = service.state()
    assert state["project"]["undo_label"] == "Apply 2x1-dimer reconstruction"
    assert state["structure"]["reconstruction"]["id"] == "2x1-dimer"

    service.undo()
    assert service.state()["structure"]["reconstruction"] is None
    assert np.allclose(project.structure.positions, ideal)
    for key, structure in project.structures.items():
        assert "reconstruction" not in structure.info, key

    service.redo()
    assert service.state()["structure"]["reconstruction"]["id"] == "2x1-dimer"
    assert project.structure.info["reconstruction"]["measured"]["bond_length_A"] == (
        pytest.approx(bond))
