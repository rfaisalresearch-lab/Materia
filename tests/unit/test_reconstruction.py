"""Software behaviour of the surface-reconstruction subsystem.

These cases exercise the registry, the refusal paths, the records written into
a structure and the service and Python-API surfaces.  They make no claim about
the physics; the geometry itself is checked in the physical-validation suite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from materia.materials.schema import MaterialValidationError, validate
from materia.solvers.classical import ClassicalSolver
from materia.structure_builder import reconstruction as reconstruction_module
from materia.python_api import Lab, UnsupportedRequest
from materia.structure_builder import make_surface
from materia.structure_builder.reconstruction import (
    GEOMETRY_CONVERGED,
    GEOMETRY_NOT_CONVERGED,
    GEOMETRY_UNRELAXED,
    ReconstructionError,
    ReconstructionNotImplemented,
    apply_reconstruction,
    available_reconstructions,
    compare_to_reference,
    dimer_statistics,
    geometry_status_of,
    registered_reconstructions,
    verify_periodicity,
)


@pytest.fixture(scope="module")
def si100(silicon):
    return make_surface(silicon, (1, 0, 0), size=(4, 2, 8), vacuum_A=12.0,
                        fix_bottom_layers=4, reconstruction="2x1-dimer")


def test_registry_lists_only_generators_that_exist():
    ids = {entry["id"] for entry in registered_reconstructions()}
    assert "2x1-dimer" in ids
    assert "7x7-DAS" not in ids
    entry = next(e for e in registered_reconstructions() if e["id"] == "2x1-dimer")
    assert entry["prototypes"] == ["diamond-cubic"]
    assert entry["orientation"] == [1, 0, 0]
    assert entry["periodicity"] == [2, 1]
    assert entry["references"]


def test_availability_is_resolved_against_the_registry(library, silicon):
    by_id = {r["id"]: r for r in available_reconstructions(silicon)}
    assert by_id["2x1-dimer"]["supported"] is True
    assert by_id["7x7-DAS"]["supported"] is False
    assert "no generator" in by_id["7x7-DAS"]["reason"]
    gold = library.get("gold")
    herringbone = available_reconstructions(gold)[0]
    assert herringbone["supported"] is False
    assert herringbone["reason"]


def test_availability_filters_by_orientation(silicon):
    ids = [r["id"] for r in available_reconstructions(silicon, (1, 0, 0))]
    assert ids == ["2x1-dimer"]


def test_unknown_reconstruction_lists_what_is_declared(silicon):
    slab = make_surface(silicon, (1, 0, 0), size=(2, 2, 6), vacuum_A=10.0)
    with pytest.raises(ReconstructionError) as excinfo:
        apply_reconstruction(slab, silicon, "not-a-reconstruction")
    assert "7x7-DAS" in str(excinfo.value)
    assert "2x1-dimer" in str(excinfo.value)


def test_declared_but_unimplemented_refuses_with_a_record(silicon):
    slab = make_surface(silicon, (1, 1, 1), size=(2, 2, 4), vacuum_A=10.0)
    with pytest.raises(ReconstructionNotImplemented) as excinfo:
        apply_reconstruction(slab, silicon, "7x7-DAS")
    record = excinfo.value.as_dict()
    assert record["supported"] is False
    assert record["reconstruction"] == "7x7-DAS"
    assert "Takayanagi" in record["reference"]
    assert record["result"]["value"] is None
    assert record["result"]["provenance"]["origin"] == "unsupported"
    assert record["suggested"]
    json.dumps(record)
    assert "reconstruction" not in slab.info


def test_wrong_orientation_is_refused(silicon):
    slab = make_surface(silicon, (1, 1, 1), size=(2, 2, 4), vacuum_A=10.0)
    with pytest.raises(ReconstructionError, match=r"\(100\)"):
        apply_reconstruction(slab, silicon, "2x1-dimer")


def test_odd_repeat_along_the_dimer_axis_is_refused(silicon):
    with pytest.raises(ReconstructionError, match="even number"):
        make_surface(silicon, (1, 0, 0), size=(3, 2, 6), vacuum_A=10.0,
                     reconstruction="2x1-dimer")


def test_reconstructing_twice_is_refused(silicon):
    slab = make_surface(silicon, (1, 0, 0), size=(2, 2, 6), vacuum_A=10.0,
                        reconstruction="2x1-dimer", relax_reconstruction=False)
    with pytest.raises(ReconstructionError, match="already carries"):
        apply_reconstruction(slab, silicon, "2x1-dimer")


def test_every_surface_site_joins_exactly_one_dimer(si100):
    record = si100.info["reconstruction"]
    flat = [atom for pair in record["pairs"] for atom in pair]
    assert len(flat) == len(set(flat))
    z = si100.positions[:, 2]
    top = {int(si100.ids[i]) for i in np.where(z >= z.max() - 1.0)[0]}
    assert set(flat) == top
    assert record["n_dimers"] * 2 == len(top)
    assert sorted(set(si100.roles[[si100.index_of(a) for a in flat]])) == ["dimer"]


def test_unrelaxed_geometry_is_labelled_estimated(silicon):
    slab = make_surface(silicon, (1, 0, 0), size=(2, 2, 6), vacuum_A=10.0,
                        reconstruction="2x1-dimer", relax_reconstruction=False)
    record = slab.info["reconstruction"]
    assert record["relaxed"] is False
    assert record["provenance"]["origin"] == "estimated"
    assert record["provenance"]["fidelity"] == "tier0-structural"
    assert any("not a predicted bond length" in a
               for a in record["provenance"]["approximations"])
    assert "measured" not in record


def test_relaxed_geometry_is_labelled_calculated(si100):
    record = si100.info["reconstruction"]
    assert record["relaxed"] is True
    assert record["provenance"]["origin"] == "calculated"
    assert record["provenance"]["fidelity"] == "tier1-classical"
    assert record["relaxation"]["performed"] is True
    assert record["relaxation"]["converged"] is True
    assert record["relaxation"]["model"].startswith("classical/stillinger-weber")
    assert record["provenance"]["software_version"]
    assert record["provenance"]["references"]


def test_slab_provenance_no_longer_claims_an_ideal_truncation(si100):
    approximations = si100.info["provenance"]["approximations"]
    assert not any("no surface reconstruction" in a for a in approximations)
    assert any("2x1-dimer" in a for a in approximations)


def test_periodicity_check_is_recorded(si100):
    check = si100.info["reconstruction"]["periodicity_check"]
    assert check["invariant_under_1x1_shift"] is False
    assert check["invariant_under_2x1_shift"] is True
    assert check["2x1_shift_is_lattice_vector"] is False
    assert check["is_2x1"] is True
    assert verify_periodicity(si100)["is_2x1"] is True


def test_statistics_require_a_reconstruction_record(si111_small):
    with pytest.raises(ReconstructionError, match="no reconstruction record"):
        dimer_statistics(si111_small)


def test_comparison_reports_deviation_without_asserting_agreement(si100, silicon):
    comparison = compare_to_reference(si100, silicon)
    quantities = {row["quantity"]: row for row in comparison["rows"]}
    assert set(quantities) == {"bond_length_A", "buckling_A"}
    for row in quantities.values():
        assert row["source"]
        assert row["reference_uncertainty"] > 0
        assert row["deviation"] == pytest.approx(row["measured"] - row["reference"])
        assert isinstance(row["within_stated_uncertainty"], bool)
    assert "reported, not corrected" in comparison["note"]


def test_record_survives_a_project_round_trip(tmp_path, silicon):
    lab = Lab()
    lab.materials.load("silicon").create_surface(
        size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0,
        fix_bottom_layers=3, reconstruction="2x1-dimer")
    path = tmp_path / "recon.materia"
    lab.project.save(str(path))

    from materia.project_format import Project

    reloaded = Project.load(str(path))
    structure = reloaded.structures[sorted(reloaded.structures)[0]]
    record = structure.info["reconstruction"]
    assert record["id"] == "2x1-dimer"
    assert record["measured"]["bond_length_A"] > 0
    assert sorted(set(structure.roles)) == ["bulk", "dimer", "subsurface", "surface"]
    assert compare_to_reference(structure, silicon)["rows"]


def test_python_api_exposes_declared_and_supported_reconstructions():
    lab = Lab()
    silicon = lab.materials.load("silicon")
    flags = {r["id"]: r["supported"] for r in silicon.reconstructions()}
    assert flags == {"7x7-DAS": False, "2x1-dimer": True}
    assert [r["id"] for r in silicon.reconstructions((1, 0, 0))] == ["2x1-dimer"]


def test_python_api_reconstruct_and_compare():
    lab = Lab()
    surface = lab.materials.load("silicon").create_surface(
        size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0, fix_bottom_layers=3)
    assert surface.reconstruction() is None
    record = surface.reconstruct("2x1-dimer")
    assert record["n_dimers"] == 4
    assert surface.reconstruction()["id"] == "2x1-dimer"
    assert surface.compare_reconstruction()["rows"]
    labels = [entry.label for entry in lab.project.history.log]
    assert any("Reconstruct 2x1-dimer" in label for label in labels)


def test_python_api_raises_a_carrying_error_for_unsupported():
    lab = Lab()
    surface = lab.materials.load("silicon").create_surface(
        size=(2, 2, 4), orientation=(1, 1, 1), vacuum_angstrom=10.0)
    with pytest.raises(UnsupportedRequest) as excinfo:
        surface.reconstruct("7x7-DAS")
    assert excinfo.value.as_dict()["supported"] is False
    assert surface.reconstruction() is None


def test_service_build_returns_the_record_and_the_comparison(service):
    out = service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6], vacuum_A=10.0,
                                reconstruction="2x1-dimer", fix_bottom_layers=3)
    assert out["ok"] is True
    assert out["reconstruction"]["n_dimers"] == 4
    assert out["comparison"]["rows"]
    json.dumps(out["state"])
    assert out["state"]["structure"]["reconstruction"]["id"] == "2x1-dimer"


def test_service_reports_unsupported_without_raising(service):
    out = service.build_surface("silicon", [1, 1, 1], size=[2, 2, 4], vacuum_A=10.0,
                                reconstruction="7x7-DAS")
    assert out["ok"] is False
    assert out["unsupported"]["supported"] is False
    assert service.state()["structure"] is None
    assert any(w["level"] == "unsupported" for w in service.warnings)


def test_service_reports_a_refused_slab_without_raising(service):
    out = service.build_surface("silicon", [1, 0, 0], size=[3, 2, 6], vacuum_A=10.0,
                                reconstruction="2x1-dimer")
    assert out["ok"] is False
    assert "even number" in out["error"]


def test_service_reconstruct_and_report(service):
    service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6], vacuum_A=10.0,
                          fix_bottom_layers=3)
    before = service.reconstruction_report()
    assert before["reconstructed"] is False
    assert [d["id"] for d in before["declared"]] == ["2x1-dimer"]

    out = service.reconstruct("2x1-dimer")
    assert out["ok"] is True
    after = service.reconstruction_report()
    assert after["reconstructed"] is True
    assert after["record"]["id"] == "2x1-dimer"
    assert after["comparison"]["rows"]
    assert service.project.history.log[-1].label == "Apply 2x1-dimer reconstruction"


def test_service_reconstruct_refuses_unsupported(service):
    service.build_surface("silicon", [1, 1, 1], size=[2, 2, 4], vacuum_A=10.0)
    out = service.reconstruct("7x7-DAS")
    assert out["ok"] is False
    assert out["unsupported"]["reconstruction"] == "7x7-DAS"
    assert service.state()["structure"]["reconstruction"] is None


def test_reference_geometry_requires_a_value_and_a_citation():
    base = {
        "schema_version": "1.0", "id": "x", "name": "X", "formula": "X",
        "category": "test",
        "structure": {"prototype": "diamond-cubic",
                      "lattice": {"a": 5.0, "unit": "A", "system": "cubic"},
                      "basis": [{"element": "Si", "fractional": [0, 0, 0]}]},
        "reconstructions": [{"id": "r", "orientation": [1, 0, 0],
                             "reference_geometry": {"bond_length_A": {"value": 2.0}}}],
    }
    with pytest.raises(MaterialValidationError, match="source"):
        validate(base)
    base["reconstructions"][0]["reference_geometry"]["bond_length_A"] = {"source": "s"}
    with pytest.raises(MaterialValidationError, match="value"):
        validate(base)


def test_a_generator_works_for_a_plugin_material_without_reference_data():
    """The strained-silicon plug-in material shares silicon's prototype, so the
    dimer generator applies to it, and it deliberately ships no reference
    geometry because the LEED determination is for the unstrained surface. The
    comparison must then come back empty rather than borrowing a number that
    does not describe this system."""
    from materia.materials import default_library
    from materia.plugin_system import default_manager

    default_manager().load_all()
    strained = default_library().get("silicon_strained_1pct")
    entry = next(r for r in available_reconstructions(strained) if r["id"] == "2x1-dimer")
    assert entry["supported"] is True
    assert entry["reference_geometry"] == {}

    slab = make_surface(strained, (1, 0, 0), size=(2, 2, 6), vacuum_A=10.0,
                        fix_bottom_layers=3, reconstruction="2x1-dimer")
    assert slab.info["reconstruction"]["measured"]["bond_length_A"] > 0
    assert compare_to_reference(slab, strained)["rows"] == []


@pytest.fixture(scope="module")
def stalled(silicon):
    """A relaxation stopped after a single step: attempted, nowhere near a minimum."""
    slab = make_surface(silicon, (1, 0, 0), size=(4, 2, 8), vacuum_A=12.0,
                        fix_bottom_layers=4)
    apply_reconstruction(slab, silicon, "2x1-dimer", max_steps=1)
    return slab


def test_non_converged_relaxation_is_not_reported_as_relaxed(stalled):
    record = stalled.info["reconstruction"]
    assert record["relaxation"]["performed"] is True
    assert record["relaxation"]["converged"] is False
    assert record["relaxation_attempted"] is True
    assert record["relaxed"] is False
    assert record["geometry_status"] == GEOMETRY_NOT_CONVERGED


def test_non_converged_geometry_is_not_called_an_energy_minimum(stalled):
    provenance = stalled.info["reconstruction"]["provenance"]
    assert provenance["origin"] == "estimated"
    assert not any("energy minimum of this slab" in a
                   for a in provenance["approximations"])
    assert any("did not converge" in a for a in provenance["approximations"])
    assert any("NOT an energy minimum" in a for a in provenance["approximations"])
    assert provenance["parameters"]["geometry_status"] == GEOMETRY_NOT_CONVERGED
    assert provenance["parameters"]["max_steps"] == 1


def test_non_converged_slab_provenance_does_not_claim_a_relaxed_surface(stalled):
    statements = stalled.info["provenance"]["approximations"]
    applied = next(a for a in statements if "2x1-dimer" in a)
    assert "did not converge" in applied
    assert "relaxed to a converged geometry" not in applied


def test_non_converged_measurements_are_kept_but_flagged(stalled):
    measured = stalled.info["reconstruction"]["measured"]
    assert measured["bond_length_A"] > 0
    assert measured["is_energy_minimum"] is False
    assert measured["geometry_status"] == GEOMETRY_NOT_CONVERGED


def test_comparison_refuses_to_validate_a_non_converged_geometry(stalled, silicon):
    comparison = compare_to_reference(stalled, silicon)
    assert comparison["comparable"] is False
    assert comparison["geometry_status"] == GEOMETRY_NOT_CONVERGED
    assert "did not converge" in comparison["blocked_reason"]
    assert comparison["rows"]
    for row in comparison["rows"]:
        assert row["comparable"] is False
        assert row["within_stated_uncertainty"] is None


def test_comparison_refuses_to_validate_an_unrelaxed_geometry(silicon):
    slab = make_surface(silicon, (1, 0, 0), size=(2, 2, 6), vacuum_A=10.0,
                        reconstruction="2x1-dimer", relax_reconstruction=False)
    assert slab.info["reconstruction"]["geometry_status"] == GEOMETRY_UNRELAXED
    comparison = compare_to_reference(slab, silicon)
    assert comparison["comparable"] is False
    assert "No relaxation was run" in comparison["blocked_reason"]
    assert all(r["within_stated_uncertainty"] is None for r in comparison["rows"])


def test_converged_geometry_is_still_comparable_and_calculated(si100, silicon):
    record = si100.info["reconstruction"]
    assert record["geometry_status"] == GEOMETRY_CONVERGED
    assert record["relaxed"] is True
    assert record["measured"]["is_energy_minimum"] is True
    comparison = compare_to_reference(si100, silicon)
    assert comparison["comparable"] is True
    assert comparison["blocked_reason"] == ""
    verdicts = {r["quantity"]: r["within_stated_uncertainty"] for r in comparison["rows"]}
    assert verdicts == {"bond_length_A": False, "buckling_A": False}


def test_service_comparison_carries_the_geometry_status(service):
    out = service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6], vacuum_A=10.0,
                                fix_bottom_layers=3, reconstruction="2x1-dimer",
                                reconstruction_max_steps=1)
    assert out["ok"] is True
    assert out["reconstruction"]["geometry_status"] == GEOMETRY_NOT_CONVERGED
    assert out["comparison"]["comparable"] is False
    assert all(r["within_stated_uncertainty"] is None for r in out["comparison"]["rows"])
    state = service.state()["structure"]["reconstruction"]
    assert state["relaxed"] is False
    assert state["geometry_status"] == GEOMETRY_NOT_CONVERGED


def test_building_a_surface_creates_exactly_one_project_entry(service):
    out = service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6], vacuum_A=10.0,
                                fix_bottom_layers=3)
    project = service.project
    assert sorted(project.structures) == [out["key"]]
    assert project.active_structure_key == out["key"]
    assert project.structures[out["key"]] is service.structure
    adds = [e for e in project.history.log if e.operation == "structure.add"]
    assert len(adds) == 1
    assert adds[0].label == "Build silicon (100) surface"


def test_building_a_reconstructed_surface_creates_one_entry_and_one_log_line(service):
    out = service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6], vacuum_A=10.0,
                                fix_bottom_layers=3, reconstruction="2x1-dimer")
    project = service.project
    assert sorted(project.structures) == [out["key"]]
    assert [e.label for e in project.history.log] == [
        "Build silicon (100) surface, 2x1-dimer"]


def test_registering_the_same_structure_twice_reuses_its_key(service):
    service.build_surface("silicon", [1, 0, 0], size=[2, 2, 4], vacuum_A=10.0)
    project = service.project
    key = project.active_structure_key
    again = project.add_structure(project.structures[key])
    assert again == key
    assert sorted(project.structures) == [key]
    assert len([e for e in project.history.log if e.operation == "structure.add"]) == 1


def test_reconstructing_an_existing_slab_records_its_own_undo_point(service):
    service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6], vacuum_A=10.0,
                          fix_bottom_layers=3)
    project = service.project
    before = len(project.history.log)

    out = service.reconstruct("2x1-dimer")
    assert out["ok"] is True
    assert len(project.history.log) == before + 1
    entry = project.history.log[-1]
    assert entry.operation == "structure.reconstruct"
    assert entry.label == "Apply 2x1-dimer reconstruction"
    assert entry.parameters["reconstruction"] == "2x1-dimer"
    assert "converged" in entry.result_summary
    assert project.history.log[0].label == "Build silicon (100) surface"
    assert project.history.undo_labels()[-1] == "Apply 2x1-dimer reconstruction"


def test_undo_and_redo_restore_the_reconstruction_everywhere(service):
    service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6], vacuum_A=10.0,
                          fix_bottom_layers=3)
    project = service.project
    key = project.active_structure_key
    ideal_positions = project.structures[key].positions.copy()

    service.reconstruct("2x1-dimer")
    bond = project.structure.info["reconstruction"]["measured"]["bond_length_A"]
    assert project.structure.info["surface"]["reconstruction"] == "2x1-dimer"

    undone = service.undo()
    assert undone["label"] == "Apply 2x1-dimer reconstruction"
    restored = project.structures[key]
    assert restored is service.structure
    assert "reconstruction" not in restored.info
    assert "reconstruction" not in restored.info["surface"]
    assert np.allclose(restored.positions, ideal_positions)
    assert "dimer" not in set(restored.roles)
    assert service.state()["structure"]["reconstruction"] is None
    assert service.reconstruction_report()["reconstructed"] is False

    service.redo()
    again = project.structures[key]
    assert again is service.structure
    assert again.info["reconstruction"]["measured"]["bond_length_A"] == pytest.approx(bond)
    assert again.info["surface"]["reconstruction"] == "2x1-dimer"
    assert service.state()["structure"]["reconstruction"]["id"] == "2x1-dimer"


def test_a_refused_reconstruction_changes_neither_structure_nor_history(service):
    service.build_surface("silicon", [1, 0, 0], size=[3, 2, 6], vacuum_A=10.0)
    project = service.project
    before_log = [(e.operation, e.label) for e in project.history.log]
    before_positions = project.structure.positions.copy()
    before_info = sorted(project.structure.info)

    refused = service.reconstruct("2x1-dimer")
    assert refused["ok"] is False
    assert "even number" in refused["error"]
    assert [(e.operation, e.label) for e in project.history.log] == before_log
    assert np.allclose(project.structure.positions, before_positions)
    assert sorted(project.structure.info) == before_info
    assert "reconstruction" not in project.structure.info


def test_an_unsupported_reconstruction_changes_neither_structure_nor_history(service):
    service.build_surface("silicon", [1, 1, 1], size=[2, 2, 4], vacuum_A=10.0)
    project = service.project
    before_log = [(e.operation, e.label) for e in project.history.log]
    before_depth = len(project.history.log)
    before_positions = project.structure.positions.copy()

    refused = service.reconstruct("7x7-DAS")
    assert refused["ok"] is False
    assert refused["unsupported"]["reconstruction"] == "7x7-DAS"
    assert [(e.operation, e.label) for e in project.history.log] == before_log
    assert len(project.history.log) == before_depth
    assert np.allclose(project.structure.positions, before_positions)
    assert "reconstruction" not in project.structure.info


def _log(lab):
    return [(e.operation, e.label) for e in lab.project.history.log]


def test_api_refusal_leaves_history_and_undo_depth_untouched():
    lab = Lab()
    surface = lab.materials.load("silicon").create_surface(
        size=(2, 2, 4), orientation=(1, 1, 1), vacuum_angstrom=10.0)
    before_log = _log(lab)
    before_depth = len(lab.project.history._undo)
    before_positions = surface.structure.positions.copy()
    before_info = sorted(surface.structure.info)

    with pytest.raises(UnsupportedRequest):
        surface.reconstruct("7x7-DAS")

    assert _log(lab) == before_log
    assert len(lab.project.history._undo) == before_depth
    assert np.allclose(surface.structure.positions, before_positions)
    assert sorted(surface.structure.info) == before_info
    assert surface.reconstruction() is None


@pytest.mark.parametrize("reconstruction,error", [
    ("2x1-dimer", ReconstructionError),
    ("nonsense", ReconstructionError),
])
def test_api_invalid_requests_leave_history_untouched(reconstruction, error):
    lab = Lab()
    surface = lab.materials.load("silicon").create_surface(
        size=(2, 2, 4), orientation=(1, 1, 1), vacuum_angstrom=10.0)
    before_log = _log(lab)
    before_depth = len(lab.project.history._undo)

    with pytest.raises(error):
        surface.reconstruct(reconstruction)

    assert _log(lab) == before_log
    assert len(lab.project.history._undo) == before_depth
    assert surface.reconstruction() is None


def test_api_refusal_on_an_odd_slab_leaves_history_untouched():
    lab = Lab()
    surface = lab.materials.load("silicon").create_surface(
        size=(3, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0)
    before_log = _log(lab)
    before_positions = surface.structure.positions.copy()

    with pytest.raises(ReconstructionError, match="even number"):
        surface.reconstruct("2x1-dimer")

    assert _log(lab) == before_log
    assert np.allclose(surface.structure.positions, before_positions)


def test_api_success_records_one_entry_that_undo_and_redo_reverse():
    lab = Lab()
    surface = lab.materials.load("silicon").create_surface(
        size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0, fix_bottom_layers=3)
    project = lab.project
    ideal = surface.structure.positions.copy()
    before = len(project.history.log)

    record = surface.reconstruct("2x1-dimer")
    assert len(project.history.log) == before + 1
    entry = project.history.log[-1]
    assert entry.operation == "structure.reconstruct"
    assert entry.label == "Reconstruct 2x1-dimer"
    assert entry.result_summary == "geometry converged"
    bond = record["measured"]["bond_length_A"]

    assert project.undo() == "Reconstruct 2x1-dimer"
    assert "reconstruction" not in project.structure.info
    assert np.allclose(project.structure.positions, ideal)

    assert project.redo() == "Reconstruct 2x1-dimer"
    assert project.structure.info["reconstruction"]["measured"][
        "bond_length_A"] == pytest.approx(bond)


def test_api_does_not_log_for_a_structure_the_project_does_not_hold():
    lab = Lab()
    detached = lab.materials.load("silicon").create_surface(
        size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0,
        fix_bottom_layers=3, activate=False)
    before_log = _log(lab)
    record = detached.reconstruct("2x1-dimer")
    assert record["geometry_status"] == GEOMETRY_CONVERGED
    assert _log(lab) == before_log


@pytest.mark.parametrize("relaxation,expected", [
    ({"performed": True, "converged": True}, GEOMETRY_CONVERGED),
    ({"performed": True, "converged": False}, GEOMETRY_NOT_CONVERGED),
    ({"performed": False}, GEOMETRY_UNRELAXED),
])
def test_geometry_status_is_derived_for_records_without_the_key(relaxation, expected):
    assert geometry_status_of({"relaxation": relaxation}) == expected


def test_a_project_saved_before_the_status_existed_still_compares(tmp_path, silicon):
    """A record written by an earlier build carries only the relaxation flags.
    It must read back with the state it actually had, not default to the most
    cautious one and silently stop being comparable."""
    from materia.project_format import Project

    lab = Lab()
    lab.materials.load("silicon").create_surface(
        size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0,
        fix_bottom_layers=3, reconstruction="2x1-dimer")
    path = tmp_path / "legacy.materia"
    lab.project.save(str(path))

    reloaded = Project.load(str(path))
    structure = reloaded.structures[sorted(reloaded.structures)[0]]
    record = structure.info["reconstruction"]
    record.pop("geometry_status")
    record.pop("relaxation_attempted")

    assert geometry_status_of(record) == GEOMETRY_CONVERGED
    comparison = compare_to_reference(structure, silicon)
    assert comparison["comparable"] is True
    assert comparison["rows"]


def _slab_pair(lab):
    """An active Si(111) slab and a detached Si(100) slab in the same project."""
    silicon = lab.materials.load("silicon")
    active = silicon.create_surface(size=(2, 2, 4), orientation=(1, 1, 1),
                                    vacuum_angstrom=10.0)
    detached = silicon.create_surface(size=(4, 2, 6), orientation=(1, 0, 0),
                                      vacuum_angstrom=10.0, activate=False)
    return active, detached


def test_detached_handle_records_no_history_and_leaves_the_active_slab_alone():
    """A handle the project never registered is the caller's own object. It may
    be reconstructed, but it must not write an undo point that would restore its
    geometry over an unrelated structure."""
    lab = Lab()
    active, detached = _slab_pair(lab)
    project = lab.project
    assert project.key_of(detached.structure) is None
    active_positions = active.structure.positions.copy()
    active_atoms = len(active.structure)
    before_log = _log(lab)
    before_depth = len(project.history._undo)

    record = detached.reconstruct("2x1-dimer", relax=False)
    assert record["id"] == "2x1-dimer"
    assert detached.reconstruction() is not None
    assert _log(lab) == before_log
    assert len(project.history._undo) == before_depth

    assert project.active_structure_key == "struct0001"
    assert len(project.structure) == active_atoms
    assert np.allclose(project.structure.positions, active_positions)
    assert "reconstruction" not in project.structure.info

    project.undo()
    assert project.structures == {}
    assert project.active_structure_key is None
    project.redo()
    assert project.active_structure_key == "struct0001"
    assert len(project.structure) == active_atoms
    assert np.allclose(project.structure.positions, active_positions)
    assert "reconstruction" not in project.structure.info


def test_project_owned_inactive_handle_undoes_into_its_own_slot():
    """A structure the project holds but has not activated gets a real undo
    point, bound to its own slot: undo restores it, not whatever is active."""
    lab = Lab()
    active, other = _slab_pair(lab)
    project = lab.project
    other_key = project.add_structure(other.structure, activate=False)
    assert project.active_structure_key != other_key

    active_positions = active.structure.positions.copy()
    other_positions = other.structure.positions.copy()
    before = len(project.history.log)

    other.reconstruct("2x1-dimer", relax=False)
    assert len(project.history.log) == before + 1
    assert project.active_structure_key != other_key
    assert "reconstruction" in project.structures[other_key].info
    assert "reconstruction" not in project.structures["struct0001"].info

    assert project.undo() == "Reconstruct 2x1-dimer"
    assert project.active_structure_key != other_key
    assert "reconstruction" not in project.structures[other_key].info
    assert np.allclose(project.structures[other_key].positions, other_positions)
    assert np.allclose(project.structures["struct0001"].positions, active_positions)
    assert len(project.structures["struct0001"]) == len(active_positions)

    assert project.redo() == "Reconstruct 2x1-dimer"
    assert "reconstruction" in project.structures[other_key].info
    assert np.allclose(project.structures["struct0001"].positions, active_positions)


def test_active_handle_undo_is_unchanged():
    lab = Lab()
    surface = lab.materials.load("silicon").create_surface(
        size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0, fix_bottom_layers=3)
    project = lab.project
    assert project.key_of(surface.structure) == project.active_structure_key
    ideal = surface.structure.positions.copy()

    surface.reconstruct("2x1-dimer", relax=False)
    assert project.history.log[-1].label == "Reconstruct 2x1-dimer"
    assert project.undo() == "Reconstruct 2x1-dimer"
    assert "reconstruction" not in project.structure.info
    assert np.allclose(project.structure.positions, ideal)
    assert project.redo() == "Reconstruct 2x1-dimer"
    assert project.structure.info["reconstruction"]["id"] == "2x1-dimer"


def test_key_of_identifies_the_slot_by_identity():
    lab = Lab()
    active, detached = _slab_pair(lab)
    project = lab.project
    assert project.key_of(active.structure) == "struct0001"
    assert project.key_of(detached.structure) is None
    assert project.key_of(active.structure.copy()) is None


def test_snapshots_carry_the_slot_they_were_taken_from():
    lab = Lab()
    active, other = _slab_pair(lab)
    project = lab.project
    other_key = project.add_structure(other.structure, activate=False)
    other.reconstruct("2x1-dimer", relax=False)
    assert project.history.pending_undo().structure_key == other_key


class TestReconstructionIsTransactional:
    """A reconstruction that fails must leave nothing behind.

    The failure modes worth separating are a bad argument, which should be
    refused before anything moves, and a failure part-way through the
    generator or the relaxation, which has to be rolled back.  The second is
    the one a validation check alone cannot cover.
    """

    @staticmethod
    def _detached(lab):
        slab = lab.materials.load("silicon").create_surface(
            size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0,
            fix_bottom_layers=3, activate=False)
        slab.structure.ensure_bonds()
        return slab

    @staticmethod
    def _state(structure):
        return {
            "positions": structure.positions.copy(),
            "roles": list(structure.roles),
            "labels": list(structure.labels),
            "fixed": structure.fixed.copy(),
            "info": json.dumps(structure.info, sort_keys=True, default=str),
            "bonds": len(structure.ensure_bonds()),
            "ids": structure.ids.copy(),
        }

    def _assert_unchanged(self, structure, before):
        after = self._state(structure)
        assert np.allclose(after["positions"], before["positions"])
        assert after["roles"] == before["roles"]
        assert after["labels"] == before["labels"]
        assert np.array_equal(after["fixed"], before["fixed"])
        assert np.array_equal(after["ids"], before["ids"])
        assert after["bonds"] == before["bonds"]
        assert after["info"] == before["info"]
        assert "reconstruction" not in structure.info
        assert "reconstruction" not in structure.info.get("surface", {})
        assert not any("2x1" in a
                       for a in structure.info["provenance"]["approximations"])

    @pytest.mark.parametrize("kwargs,message", [
        ({"steps": "invalid"}, "positive whole number"),
        ({"steps": 3.5}, "positive whole number"),
        ({"steps": 0}, "at least 1"),
        ({"steps": -5}, "at least 1"),
        ({"fmax": "x"}, "must be a force"),
        ({"fmax": 0.0}, "positive, finite"),
        ({"fmax": float("inf")}, "positive, finite"),
        ({"model": 42}, "must be a solver name"),
    ])
    def test_bad_arguments_are_refused_before_anything_moves(self, kwargs, message):
        lab = Lab()
        slab = self._detached(lab)
        before = self._state(slab.structure)
        with pytest.raises(ReconstructionError, match=message):
            slab.reconstruct("2x1-dimer", **kwargs)
        self._assert_unchanged(slab.structure, before)
        assert _log(lab) == []

    def test_a_generator_failure_after_it_has_moved_atoms_is_rolled_back(self, monkeypatch):
        lab = Lab()
        slab = self._detached(lab)
        before = self._state(slab.structure)

        def explode(structure, row, axis):
            structure.positions = structure.positions + 0.5
            structure.roles[0] = "dimer"
            raise RuntimeError("generator failed after moving atoms")

        monkeypatch.setattr(reconstruction_module, "_pair_along_axis", explode)
        with pytest.raises(RuntimeError, match="after moving atoms"):
            slab.reconstruct("2x1-dimer", relax=False)
        self._assert_unchanged(slab.structure, before)
        assert _log(lab) == []

    def test_a_solver_failure_mid_relaxation_is_rolled_back(self, monkeypatch):
        lab = Lab()
        slab = self._detached(lab)
        before = self._state(slab.structure)

        def explode(self, structure, **kwargs):
            structure.positions = structure.positions + 0.3
            raise RuntimeError("solver failed mid-relaxation")

        monkeypatch.setattr(ClassicalSolver, "relax", explode)
        with pytest.raises(RuntimeError, match="mid-relaxation"):
            slab.reconstruct("2x1-dimer")
        self._assert_unchanged(slab.structure, before)
        assert _log(lab) == []

    def test_a_failure_after_the_relaxation_is_rolled_back(self, monkeypatch):
        """The generator has run, the relaxation has converged and the records
        are being written: the latest point at which anything can go wrong."""
        lab = Lab()
        slab = self._detached(lab)
        before = self._state(slab.structure)

        def explode(structure):
            raise RuntimeError("failed while measuring")

        monkeypatch.setattr(reconstruction_module, "dimer_statistics", explode)
        with pytest.raises(RuntimeError, match="while measuring"):
            slab.reconstruct("2x1-dimer")
        self._assert_unchanged(slab.structure, before)
        assert _log(lab) == []

    def test_a_rolled_back_slab_can_still_be_reconstructed(self, monkeypatch):
        lab = Lab()
        slab = self._detached(lab)

        def explode(self, structure, **kwargs):
            raise RuntimeError("solver failed mid-relaxation")

        monkeypatch.setattr(ClassicalSolver, "relax", explode)
        with pytest.raises(RuntimeError):
            slab.reconstruct("2x1-dimer")
        monkeypatch.undo()

        record = slab.reconstruct("2x1-dimer")
        assert record["geometry_status"] == GEOMETRY_CONVERGED
        assert record["measured"]["bond_length_A"] == pytest.approx(2.4035, abs=1e-3)

    def test_a_failure_on_an_active_handle_leaves_the_project_untouched(self, monkeypatch):
        lab = Lab()
        surface = lab.materials.load("silicon").create_surface(
            size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0,
            fix_bottom_layers=3)
        project = lab.project
        surface.structure.ensure_bonds()
        before = self._state(surface.structure)
        before_log = _log(lab)
        before_depth = len(project.history._undo)

        def explode(self, structure, **kwargs):
            structure.positions = structure.positions + 0.3
            raise RuntimeError("solver failed mid-relaxation")

        monkeypatch.setattr(ClassicalSolver, "relax", explode)
        with pytest.raises(RuntimeError):
            surface.reconstruct("2x1-dimer")

        self._assert_unchanged(surface.structure, before)
        assert surface.structure is project.structure
        self._assert_unchanged(project.structure, before)
        assert _log(lab) == before_log
        assert len(project.history._undo) == before_depth
