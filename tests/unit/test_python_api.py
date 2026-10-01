"""The public Python API and the embedded script runner."""

from __future__ import annotations

import numpy as np
import pytest

from materia.python_api import ApiError, Lab, ScriptRunner
from materia.python_api.execution import ALLOWED_MODULES


@pytest.fixture
def lab():
    return Lab()


@pytest.fixture
def runner(lab):
    return ScriptRunner(lab, mode="restricted")


def test_material_handle_exposes_summary_and_properties(lab):
    silicon = lab.materials.load("silicon", orientation="111")
    assert silicon.id == "silicon"
    assert silicon.orientation == (1, 1, 1)
    assert silicon.property("band_gap").value == pytest.approx(1.12)
    with pytest.raises(ApiError):
        silicon.property("unobtainium_gap")


def test_surface_construction_and_editing(lab):
    silicon = lab.materials.load("silicon", orientation="111")
    surface = silicon.create_surface(size=(2, 2, 3), vacuum_angstrom=12.0)
    assert len(surface) > 0
    site = surface.nearest_site((3.0, 3.0, 30.0))
    record = surface.add_dopant("phosphorus", site=site)
    assert record["to"] == "P"
    assert surface.atoms[site].symbol == "P"
    assert surface.atoms[site].coordination >= 1


def test_atom_view_reads_and_writes_through_the_structure(lab):
    surface = lab.materials.load("silicon", "111").create_surface(size=(1, 1, 2))
    atom = surface.atoms[int(surface.structure.ids[0])]
    assert atom.symbol == "Si"
    assert atom.electron_configuration == "1s2 2s2 2p6 3s2 3p2"
    atom.charge = 1
    assert surface.structure.total_charge() == pytest.approx(1.0)
    assert atom.electron_configuration == "1s2 2s2 2p6 3s2 3p1"
    atom.set_spin(0.5)
    assert atom.magnetic_moment == pytest.approx(1.0)


def test_energy_and_relaxation_through_the_api(lab):
    surface = lab.materials.load("silicon", "111").create_surface(size=(2, 2, 3))
    energy = surface.energy()
    assert energy.unit == "eV"
    assert energy.value < 0
    rng = np.random.default_rng(1)
    surface.structure.positions = surface.structure.positions + \
        rng.normal(0, 0.05, surface.structure.positions.shape)
    result = surface.relax(fmax=0.05, steps=200)
    assert result.convergence.converged


def test_phonons_run_through_surface_and_persist_results(lab, tmp_path):
    from materia.project_format import Project

    crystal = lab.materials.load("silicon").bulk()
    before = crystal.structure.positions.copy()
    run = crystal.phonons(stencil=4, temperatures_K=[0.0, 300.0])
    frequencies = run["frequencies"]
    assert frequencies.supported
    assert frequencies.unit == "THz"
    assert frequencies.extra["kinds"].count("rigid-body") == 3
    assert "imaginary" not in frequencies.extra["kinds"]
    assert run["zero_point_energy"].supported
    assert run["heat_capacity_eV_K"].supported
    assert run["heat_capacity_eV_K"].value.shape == (2,)
    assert np.array_equal(crystal.structure.positions, before)
    assert lab.project.results["phonons::frequencies"] is frequencies
    assert "phonons::force_constants" in lab.project.results
    assert np.array_equal(
        lab.project.arrays["phonons::force_constants"].data,
        run["force_constants"].value)
    assert lab.project.arrays["phonons::displacements"].kind == "trajectory"
    assert np.array_equal(
        lab.project.arrays["phonons::heat_capacity_eV_K"].data,
        run["heat_capacity_eV_K"].value)
    saved = lab.save(str(tmp_path / "phonons.materia"))
    reopened = Project.load(saved)
    assert np.array_equal(
        reopened.arrays["phonons::force_constants"].data,
        run["force_constants"].value)
    assert np.array_equal(
        reopened.arrays["phonons::heat_capacity_eV_K"].data,
        run["heat_capacity_eV_K"].value)


def test_lab_phonons_validates_settings_and_supports_top_level_namespace(lab):
    from materia.physics.phonons import PhononSettings
    from materia.python_api.api import build_namespace

    crystal = lab.materials.load("silicon").bulk()
    run = build_namespace(lab)["phonons"](
        crystal, settings=PhononSettings(stencil=2))
    assert run["frequencies"].supported
    with pytest.raises(ApiError, match="either settings"):
        lab.phonons(crystal, settings=PhononSettings(), stencil=4)
    with pytest.raises(ApiError, match="PhononSettings"):
        lab.phonons(crystal, settings={"stencil": 4})


def test_recommended_model_selection_handles_dopants(lab):
    surface = lab.materials.load("silicon", "111").create_surface(size=(2, 2, 3))
    surface.substitute(int(surface.structure.ids[0]), "P")
    energy = surface.energy()
    assert energy.value < 0
    assert "stillinger-weber" in energy.provenance.model
    electronic = surface.solve()
    assert "sp3s*-Si+P" in electronic.solver


def test_unsupported_material_raises_with_alternatives(lab):
    gold = lab.materials.load("gold", "111").create_surface(size=(2, 2, 2))
    with pytest.raises(ApiError) as excinfo:
        gold.solve()
    assert "tight-binding" in str(excinfo.value)


def test_measurements_match_the_structure(lab):
    surface = lab.materials.load("silicon", "111").create_surface(size=(2, 2, 2))
    surface.bonds()
    first = int(surface.structure.ids[0])
    neighbours = surface.structure.neighbors_of(first)
    assert neighbours
    distance = lab.measure.distance(first, neighbours[0], surface)
    assert 2.0 < distance < 2.6


def test_view_collects_items_headlessly(lab):
    lab.view.plot([0, 1, 2], [1, 2, 3], title="test")
    lab.view.message("hello", "info")
    kinds = [item["kind"] for item in lab.view.items]
    assert kinds == ["plot", "message"]


def test_notebook_export_is_available_through_python_io(lab, tmp_path):
    lab.materials.load("silicon").bulk()
    project_path = lab.save(str(tmp_path / "project.materia"))
    exported = lab.io.write_notebook(
        str(tmp_path / "project.ipynb"), project_path=project_path)
    assert exported["cells"] > 0
    assert exported["metadata"]["project_sha256"]
    assert exported["path"].endswith("project.ipynb")


def test_script_runner_executes_real_python(runner):
    result = runner.run("total = sum(range(10))\nprint(total)\ntotal")
    assert result.ok
    assert "45" in result.stdout
    assert result.result_repr == "45"


def test_script_runner_reports_errors_with_a_traceback(runner):
    result = runner.run("1 / 0")
    assert not result.ok
    assert "ZeroDivisionError" in result.error
    assert "ZeroDivisionError" in result.traceback


def test_script_runner_supports_the_documented_workflow(runner):
    result = runner.run(
        "si = materials.load('silicon', orientation='111')\n"
        "surface = si.create_surface(size=(2, 2, 3), vacuum_angstrom=12)\n"
        "print(len(surface), surface.structure.formula())\n"
        "site = surface.nearest_site((3.0, 3.0, 25.0))\n"
        "surface.add_dopant('phosphorus', site=site)\n"
        "print(surface.structure.formula())\n")
    assert result.ok, result.traceback
    assert "P" in result.stdout


def test_restricted_mode_blocks_dangerous_imports(runner):
    for module in ("os", "sys", "subprocess", "socket", "shutil"):
        result = runner.run(f"import {module}")
        assert not result.ok
        assert "not allowed in restricted mode" in result.error


def test_restricted_mode_allows_scientific_modules(runner):
    result = runner.run("import numpy as np\nprint(np.arange(3).sum())")
    assert result.ok
    assert "3" in result.stdout


def test_restricted_mode_removes_file_and_eval_builtins(runner):
    for expression in ("open('/etc/passwd')", "eval('1')", "exec('x=1')",
                       "compile('1', '<s>', 'eval')", "input()"):
        result = runner.run(expression)
        assert not result.ok
        assert "NameError" in result.error


def test_allowed_module_list_is_explicit():
    assert "numpy" in ALLOWED_MODULES
    assert "materia" in ALLOWED_MODULES
    assert "os" not in ALLOWED_MODULES
    assert "subprocess" not in ALLOWED_MODULES


def test_trusted_mode_restores_full_builtins(lab):
    runner = ScriptRunner(lab, mode="trusted")
    result = runner.run("import os\nprint(bool(os.sep))")
    assert result.ok
    assert "True" in result.stdout


def test_statement_and_expression_on_one_line(runner):
    result = runner.run("a = 2; a + 3")
    assert result.ok


def test_script_view_items_are_returned(runner):
    result = runner.run("view.plot([0,1],[1,2], title='x')")
    assert result.ok
    assert result.view_items[0]["kind"] == "plot"
