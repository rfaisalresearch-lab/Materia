"""Project format, history, and structure import/export."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from materia.dataio import detect_format, read_structure, structure_table, write_csv, write_cube, write_structure
from materia.dataio.formats import FormatError
from materia.materials import load
from materia.multiscale import RegionSpec, WaferSpec
from materia.project_format import Project, RecoveryStore
from materia.project_format.migrations import MigrationError, migrate_manifest
from materia.structure_builder import make_surface
from materia.structure_builder.defects import create_vacancy, substitute_atom


@pytest.fixture
def project():
    p = Project("Test project")
    p.create_wafer(WaferSpec(material_id="silicon", orientation=(1, 1, 1),
                             diameter_mm=200, dopant="P",
                             dopant_concentration_cm3=1e19))
    p.extract_region(RegionSpec(x_mm=1.0, y_mm=2.0, size_nm=(1.5, 1.5),
                                depth_layers=3))
    return p


def test_undo_and_redo_restore_the_structure(project):
    structure = project.structure
    before = structure.formula()
    target = int(structure.ids[3])

    project.begin("Substitute P", "edit.substitute", {"atom_id": target})
    substitute_atom(structure, target, "P")
    assert project.structure.formula() != before

    label = project.undo()
    assert label == "Substitute P"
    assert project.structure.formula() == before

    label = project.redo()
    assert label == "Substitute P"
    assert project.structure.formula() != before


def test_undo_stack_survives_several_operations(project):
    structure = project.structure
    formulas = [structure.formula()]
    for index in (2, 5, 7):
        project.begin(f"Vacancy {index}", "edit.vacancy", {})
        create_vacancy(project.structure, int(project.structure.ids[index]))
        formulas.append(project.structure.formula())
    for expected in reversed(formulas[:-1]):
        project.undo()
        assert project.structure.formula() == expected
    assert project.history.can_undo
    assert project.history.redo_labels() == ["Vacancy 7", "Vacancy 5", "Vacancy 2"]


def test_redo_is_cleared_by_a_new_edit(project):
    project.begin("first", "edit.vacancy", {})
    create_vacancy(project.structure, int(project.structure.ids[0]))
    project.undo()
    assert project.history.can_redo
    project.begin("second", "edit.vacancy", {})
    create_vacancy(project.structure, int(project.structure.ids[1]))
    assert not project.history.can_redo


def test_checkpoints_restore_an_earlier_state(project):
    project.save_checkpoint("pristine")
    count = len(project.structure)
    project.begin("Vacancy", "edit.vacancy", {})
    create_vacancy(project.structure, int(project.structure.ids[0]))
    assert len(project.structure) == count - 1
    project.restore_checkpoint("pristine")
    assert len(project.structure) == count


def test_restoring_an_unknown_checkpoint_lists_the_known_ones(project):
    project.save_checkpoint("alpha")
    with pytest.raises(Exception) as excinfo:
        project.restore_checkpoint("beta")
    assert "alpha" in str(excinfo.value)


def test_project_round_trip(tmp_path, project):
    project.begin("Substitute P", "edit.substitute", {})
    substitute_atom(project.structure, int(project.structure.ids[4]), "P")
    project.save_checkpoint("doped", "phosphorus in place")
    path = project.save(str(tmp_path / "test"))
    assert path.endswith(".materia")

    reopened = Project.load(path)
    assert reopened.name == project.name
    assert reopened.structure.formula() == project.structure.formula()
    assert reopened.structure.positions == pytest.approx(project.structure.positions)
    assert list(reopened.structure.ids) == list(project.structure.ids)
    assert reopened.wafer.spec.dopant == "P"
    assert "doped" in reopened.checkpoints
    assert len(reopened.history.log) == len(project.history.log)


def test_saved_project_contains_a_readable_provenance_report(tmp_path, project):
    path = project.save(str(tmp_path / "prov"))
    with zipfile.ZipFile(path) as archive:
        text = archive.read("PROVENANCE.txt").decode()
        manifest = json.loads(archive.read("manifest.json"))
    assert "Materia project provenance" in text
    assert "procedural seed" in text
    assert manifest["schema_version"] == "1.0"
    assert manifest["wafer"]["material_id"] == "silicon"


def test_recovery_store_round_trip_and_discard(tmp_path, project):
    store = RecoveryStore(str(tmp_path / "recovery"))
    path = store.save(project)
    info = store.info()
    assert path.endswith("autosave.materia")
    assert info["available"] is True
    assert info["project_name"] == "Test project"
    assert info["structures"] >= 1
    recovered = store.load()
    assert recovered.structure.formula() == project.structure.formula()
    assert store.discard() is True
    assert store.info()["available"] is False


def test_recovery_store_reports_corrupt_autosave(tmp_path):
    store = RecoveryStore(str(tmp_path / "recovery"))
    store.directory.mkdir(parents=True)
    store.path.write_text("not a project")
    info = store.info()
    assert info["available"] is False
    assert info["corrupt"] is True
    assert info["error"]


def test_region_regeneration_is_deterministic():
    first = Project("a")
    first.create_wafer(WaferSpec(material_id="silicon", orientation=(1, 1, 1), seed=99,
                                 vacancy_density_cm2=5e13))
    region_a = first.extract_region(RegionSpec(x_mm=3.0, y_mm=4.0, size_nm=(2, 2),
                                               depth_layers=3))
    second = Project("b")
    second.create_wafer(WaferSpec(material_id="silicon", orientation=(1, 1, 1), seed=99,
                                  vacancy_density_cm2=5e13))
    region_b = second.extract_region(RegionSpec(x_mm=3.0, y_mm=4.0, size_nm=(2, 2),
                                                depth_layers=3))
    assert len(region_a.structure) == len(region_b.structure)
    assert region_a.structure.positions == pytest.approx(region_b.structure.positions)
    assert region_a.region_id == region_b.region_id

    elsewhere = second.extract_region(RegionSpec(x_mm=9.0, y_mm=4.0, size_nm=(2, 2),
                                                 depth_layers=3))
    assert elsewhere.region_id != region_a.region_id


def test_newer_schema_is_refused_clearly():
    with pytest.raises(MigrationError) as excinfo:
        migrate_manifest({"schema_version": "99.0"})
    assert "newer" in str(excinfo.value)
    assert "Nothing has been modified" in str(excinfo.value)


@pytest.mark.parametrize(
    "fmt,suffix",
    [("xyz", ".xyz"), ("extxyz", ".extxyz"), ("cif", ".cif"),
     ("poscar", "POSCAR"), ("pdb", ".pdb"), ("lammps-data", ".data")],
)
def test_structure_formats_round_trip(tmp_path, fmt, suffix):
    slab = make_surface(load("gallium_arsenide"), (1, 1, 0), size=(2, 2, 2), vacuum_A=10.0)
    path = str(tmp_path / (f"structure{suffix}" if suffix.startswith(".") else suffix))
    write_structure(path, slab, fmt)
    restored = read_structure(path, fmt)
    assert len(restored) == len(slab)
    assert restored.formula() == slab.formula()
    tolerance = 2e-3 if fmt == "pdb" else 1e-5
    assert np.sort(restored.positions, axis=0) == \
        pytest.approx(np.sort(slab.positions, axis=0), abs=tolerance)
    assert restored.info["provenance"]["origin"] == "imported"


def test_extended_xyz_preserves_ids_charges_and_roles(tmp_path):
    slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2), vacuum_A=10.0)
    slab.formal_charges[0] = -1.0
    path = str(tmp_path / "s.extxyz")
    write_structure(path, slab, "extxyz")
    restored = read_structure(path, "extxyz")
    assert list(restored.ids) == list(slab.ids)
    assert restored.formal_charges[0] == pytest.approx(-1.0)
    assert list(restored.roles) == list(slab.roles)
    assert restored.cell.pbc == slab.cell.pbc


def test_cif_declares_p1_symmetry(tmp_path):
    slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2))
    path = str(tmp_path / "s.cif")
    write_structure(path, slab, "cif")
    text = open(path).read()
    assert "_symmetry_space_group_name_H-M   'P 1'" in text
    assert "written as P1" in text


def test_format_detection_and_unsupported_formats(tmp_path):
    assert detect_format("/tmp/POSCAR") == "poscar"
    assert detect_format("/tmp/a.extxyz") == "extxyz"
    with pytest.raises(FormatError):
        write_structure(str(tmp_path / "x.docx"), make_surface(load("silicon"), (1, 1, 1)))


def test_cube_and_csv_export(tmp_path):
    slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2))
    data = np.random.default_rng(0).random((6, 6, 8))
    cube = write_cube(str(tmp_path / "d.cube"), slab, data)
    text = open(cube).read().splitlines()
    assert text[0].startswith("Materia")
    assert int(text[2].split()[0]) == len(slab)

    csv = write_csv(str(tmp_path / "atoms.csv"), structure_table(slab), "test")
    lines = open(csv).read().splitlines()
    assert lines[0].startswith("# test")
    assert "element" in lines[2]
    assert len(lines) == len(slab) + 3
