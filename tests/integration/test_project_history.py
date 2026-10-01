"""Undo and redo of project-level operations, driven through the service.

A structure's coordinates are only part of what an edit changes.  Building a
surface adds a dictionary entry and moves the active key; extracting a region
adds a region, a structure and an active region; creating a wafer replaces the
wafer.  These cases check that undo reverses the whole operation and that redo
puts it back, through the same service the interface uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.desktop_ui.service import Service


@pytest.fixture
def service():
    return Service()


def log(project):
    return [(e.operation, e.label) for e in project.history.log]


def assert_no_aliased_slots(project):
    """Two keys sharing one object would drift apart the moment undo restored
    a snapshot into one of them."""
    identities = [id(v) for v in project.structures.values()]
    assert len(set(identities)) == len(identities)


class TestStructureLifecycle:

    def test_building_the_first_surface_undoes_to_an_empty_project(self, service):
        project = service.project
        assert project.structures == {}

        built = service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        key = built["key"]
        atoms = len(project.structure)
        assert sorted(project.structures) == [key]
        assert project.active_structure_key == key

        assert service.undo()["label"] == "Build silicon (111) surface"
        assert project.structures == {}
        assert project.active_structure_key is None
        assert project.structure is None
        assert service.state()["structure"] is None

        assert service.redo()["label"] == "Build silicon (111) surface"
        assert sorted(project.structures) == [key]
        assert project.active_structure_key == key
        assert len(project.structure) == atoms
        assert_no_aliased_slots(project)

    def test_building_a_second_surface_undoes_back_to_the_first(self, service):
        project = service.project
        first = service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3],
                                      vacuum_A=10.0)["key"]
        first_atoms = len(project.structure)
        first_positions = project.structure.positions.copy()

        second = service.build_surface("gold", [1, 1, 1], size=[2, 2, 3],
                                       vacuum_A=10.0)["key"]
        assert second != first
        assert sorted(project.structures) == sorted([first, second])
        assert project.active_structure_key == second

        service.undo()
        assert sorted(project.structures) == [first]
        assert project.active_structure_key == first
        assert len(project.structure) == first_atoms
        assert np.allclose(project.structure.positions, first_positions)
        assert project.structure.formula().startswith("Si")

        service.redo()
        assert sorted(project.structures) == sorted([first, second])
        assert project.active_structure_key == second
        assert project.structure.formula().startswith("Au")
        assert_no_aliased_slots(project)

    def test_importing_a_structure_adds_and_removes_like_any_build(self, service, tmp_path):
        project = service.project
        built = service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3],
                                      vacuum_A=10.0)["key"]
        path = str(tmp_path / "exported.xyz")
        service.export_structure(path)

        imported = service.import_structure(path)["key"]
        assert sorted(project.structures) == sorted([built, imported])
        assert project.active_structure_key == imported

        service.undo()
        assert sorted(project.structures) == [built]
        assert project.active_structure_key == built

        service.redo()
        assert sorted(project.structures) == sorted([built, imported])
        assert project.active_structure_key == imported
        assert_no_aliased_slots(project)

    def test_undo_reverses_several_builds_in_order(self, service):
        project = service.project
        keys = [service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3],
                                      vacuum_A=10.0)["key"] for _ in range(3)]
        assert sorted(project.structures) == sorted(keys)

        for expected in reversed(range(3)):
            service.undo()
            assert sorted(project.structures) == sorted(keys[:expected])
            assert project.active_structure_key == (keys[expected - 1]
                                                    if expected else None)

        for expected in range(3):
            service.redo()
            assert sorted(project.structures) == sorted(keys[:expected + 1])
            assert project.active_structure_key == keys[expected]
        assert_no_aliased_slots(project)


class TestWafer:

    def test_creating_a_wafer_undoes_and_redoes(self, service):
        project = service.project
        assert project.wafer is None

        service.create_wafer(material_id="silicon", orientation=(1, 1, 1),
                             diameter_mm=300, seed=1)
        assert project.wafer is not None
        assert project.wafer.spec.diameter_mm == 300

        service.undo()
        assert project.wafer is None
        assert service.state()["wafer"] is None

        service.redo()
        assert project.wafer is not None
        assert project.wafer.spec.diameter_mm == 300

    def test_replacing_a_wafer_restores_the_previous_one(self, service):
        project = service.project
        service.create_wafer(material_id="silicon", orientation=(1, 1, 1),
                             diameter_mm=300, seed=1)
        service.create_wafer(material_id="gold", orientation=(1, 1, 1),
                             diameter_mm=200, seed=2)
        assert project.wafer.spec.material_id == "gold"
        assert project.wafer.spec.diameter_mm == 200

        service.undo()
        assert project.wafer.spec.material_id == "silicon"
        assert project.wafer.spec.diameter_mm == 300

        service.redo()
        assert project.wafer.spec.material_id == "gold"
        assert project.wafer.spec.diameter_mm == 200


class TestRegions:

    @pytest.fixture
    def with_wafer(self, service):
        service.create_wafer(material_id="silicon", orientation=(1, 1, 1),
                             diameter_mm=300, seed=20260920)
        return service

    def test_extracting_a_region_undoes_its_region_and_its_structure(self, with_wafer):
        service = with_wafer
        project = service.project
        base = service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3],
                                     vacuum_A=10.0)["key"]

        service.extract_region(x_mm=0.0, y_mm=0.0, size_nm=(1.5, 1.5),
                               depth_layers=3, max_atoms=3000)
        region_id = project.active_region_id
        region_key = project.active_structure_key
        assert region_id is not None
        assert sorted(project.structures) == sorted([base, region_key])
        region_atoms = len(project.structure)

        service.undo()
        assert project.regions == {}
        assert project.active_region_id is None
        assert sorted(project.structures) == [base]
        assert project.active_structure_key == base

        service.redo()
        assert sorted(project.regions) == [region_id]
        assert project.active_region_id == region_id
        assert sorted(project.structures) == sorted([base, region_key])
        assert project.active_structure_key == region_key
        assert len(project.structure) == region_atoms
        assert_no_aliased_slots(project)

    def test_extracting_a_region_from_an_empty_project_undoes_to_empty(self, with_wafer):
        service = with_wafer
        project = service.project
        service.extract_region(x_mm=0.0, y_mm=0.0, size_nm=(1.5, 1.5),
                               depth_layers=3, max_atoms=3000)
        assert len(project.structures) == 1

        service.undo()
        assert project.structures == {}
        assert project.regions == {}
        assert project.active_structure_key is None
        assert project.active_region_id is None
        assert project.wafer is not None

        service.redo()
        assert len(project.structures) == 1
        assert len(project.regions) == 1


class TestReconstructionStillUndoesIntoTheRightSlot:

    def test_active_structure(self, service):
        project = service.project
        key = service.build_surface("silicon", [1, 0, 0], size=[4, 2, 6],
                                    vacuum_A=10.0, fix_bottom_layers=3)["key"]
        ideal = project.structure.positions.copy()

        service.reconstruct("2x1-dimer")
        assert project.structure.info["reconstruction"]["id"] == "2x1-dimer"
        assert sorted(project.structures) == [key]

        service.undo()
        assert sorted(project.structures) == [key]
        assert "reconstruction" not in project.structure.info
        assert np.allclose(project.structure.positions, ideal)

        service.redo()
        assert project.structure.info["reconstruction"]["id"] == "2x1-dimer"
        assert_no_aliased_slots(project)

    def test_inactive_structure_through_the_python_api(self):
        from materia.python_api import Lab

        lab = Lab()
        project = lab.project
        silicon = lab.materials.load("silicon")
        active = silicon.create_surface(size=(2, 2, 4), orientation=(1, 1, 1),
                                        vacuum_angstrom=10.0)
        other = silicon.create_surface(size=(4, 2, 6), orientation=(1, 0, 0),
                                       vacuum_angstrom=10.0, activate=False)
        other_key = project.add_structure(other.structure, activate=False)
        active_positions = active.structure.positions.copy()
        active_key = project.active_structure_key

        other.reconstruct("2x1-dimer", relax=False)
        assert project.active_structure_key == active_key
        assert "reconstruction" in project.structures[other_key].info

        project.undo()
        assert project.active_structure_key == active_key
        assert sorted(project.structures) == sorted([active_key, other_key])
        assert "reconstruction" not in project.structures[other_key].info
        assert np.allclose(project.structures[active_key].positions, active_positions)

        project.redo()
        assert "reconstruction" in project.structures[other_key].info
        assert np.allclose(project.structures[active_key].positions, active_positions)
        assert_no_aliased_slots(project)


class TestRefusalsLeaveNoHistory:

    def test_an_unsupported_reconstruction_logs_nothing(self, service):
        project = service.project
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 4], vacuum_A=10.0)
        before = log(project)
        depth = len(project.history._undo)

        out = service.reconstruct("7x7-DAS")
        assert out["ok"] is False
        assert log(project) == before
        assert len(project.history._undo) == depth

    def test_a_refused_build_logs_nothing_and_adds_no_structure(self, service):
        project = service.project
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 4], vacuum_A=10.0)
        before = log(project)
        keys = sorted(project.structures)

        out = service.build_surface("silicon", [1, 0, 0], size=[3, 2, 6], vacuum_A=10.0,
                                    reconstruction="2x1-dimer")
        assert out["ok"] is False
        assert log(project) == before
        assert sorted(project.structures) == keys

    def test_extracting_a_region_without_a_wafer_logs_nothing(self, service):
        project = service.project
        before = log(project)
        with pytest.raises(Exception):
            service.extract_region(x_mm=0.0, y_mm=0.0, size_nm=(1.5, 1.5),
                                   depth_layers=3)
        assert log(project) == before
        assert project.structures == {}


class TestSelectionAndIdentity:

    def test_selection_and_atom_ids_survive_undo_and_redo(self, service):
        project = service.project
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        ids = project.structure.ids.copy()
        service.select("element", elements=["Si"])
        selected = list(project.selection.ids)
        assert selected

        victim = int(ids[0])
        service.edit("vacancy", ids=[victim])
        assert len(project.structure) == len(ids) - 1

        service.undo()
        assert len(project.structure) == len(ids)
        assert np.array_equal(project.structure.ids, ids)
        assert list(project.selection.ids) == selected

        service.redo()
        assert len(project.structure) == len(ids) - 1
        assert victim not in set(int(i) for i in project.structure.ids)

    def test_a_handle_keeps_pointing_at_its_slot_across_undo(self):
        from materia.python_api import Lab

        lab = Lab()
        surface = lab.materials.load("silicon").create_surface(
            size=(4, 2, 6), orientation=(1, 0, 0), vacuum_angstrom=10.0,
            fix_bottom_layers=3)
        project = lab.project
        key = project.active_structure_key
        surface.reconstruct("2x1-dimer", relax=False)

        project.undo()
        assert surface.structure is project.structures[key]
        assert "reconstruction" not in surface.structure.info

        project.redo()
        assert surface.structure is project.structures[key]
        assert surface.structure.info["reconstruction"]["id"] == "2x1-dimer"


class TestHistoryBookkeeping:

    def test_the_permanent_log_survives_undo_and_marks_entries_undone(self, service):
        project = service.project
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        service.build_surface("gold", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        assert len(project.history.log) == 2

        service.undo()
        assert len(project.history.log) == 2
        assert project.history.log[-1].undone is True
        assert project.history.log[0].undone is False

        service.redo()
        assert len(project.history.log) == 2
        assert all(not e.undone for e in project.history.log)

    def test_a_topology_only_snapshot_costs_no_structure_copy(self, service):
        project = service.project
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        add = project.history._undo[-1]
        assert add.operation == "structure.add"
        assert add.structure is None
        assert add.nbytes() == 0
        assert sorted(add.state.structures) == ["struct0001"]

    def test_a_content_change_snapshot_holds_one_copy(self, service):
        project = service.project
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        service.edit("vacancy", ids=[int(project.structure.ids[0])])
        snapshot = project.history._undo[-1]
        assert snapshot.structure is not None
        assert snapshot.structure is not project.structure
        assert snapshot.nbytes() > 0
        assert snapshot.state.structures["struct0001"] is project.structures["struct0001"]

    def test_undoing_past_the_start_is_reported(self, service):
        project = service.project
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        service.undo()
        assert not project.history.can_undo
        with pytest.raises(IndexError):
            project.undo()

    def test_a_new_operation_clears_the_redo_stack(self, service):
        project = service.project
        service.build_surface("silicon", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        service.undo()
        assert project.history.can_redo
        service.build_surface("gold", [1, 1, 1], size=[2, 2, 3], vacuum_A=10.0)
        assert not project.history.can_redo
        assert len(project.structures) == 1
        assert_no_aliased_slots(project)
