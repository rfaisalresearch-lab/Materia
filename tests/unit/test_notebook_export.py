"""Notebook export: content, safety, determinism, atomic writes and validation."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.dataio import notebook as N
from materia.materials import default_library
from materia.microscopy.scan import ScanResult
from materia.project_format.arrays import StoredArray
from materia.project_format.project import Project
from materia.provenance import Fidelity, Origin, Provenance, Result, unsupported
from materia.solvers import registry
from materia.structure_builder.lattice import bulk

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = 'x"""\n\'); import os; os.system("echo pwned") #\n<script>alert(1)</script>\n# heading'


def scan(name="height"):
    prov = Provenance("stm-test", Fidelity.TIER2_SEMI_EMPIRICAL, Origin.CALCULATED)
    return ScanResult("STM", "constant-current", {name: np.arange(12.0).reshape(3, 4)}, name,
                      {name: "A"}, (0.0, 4.0, 0.0, 3.0), (4, 3), None, prov)


def full_project(name="Full project"):
    p = Project(name)
    s = bulk(default_library().get("silicon"))
    s.info["material_id"] = "silicon"
    p.add_structure(s, activate=True)
    for key, result in registry.create("stillinger-weber-si").relax(
            s.copy(), max_steps=5).results.items():
        p.add_result(f"relax::{key}", result)
    p.add_result("dft::run42::energy", Result(
        "total_energy", -12.5, "eV",
        Provenance("dft/gpaw", Fidelity.TIER3_EXTERNAL, Origin.CALCULATED,
                   inputs_digest="abc123"), extra={"run_id": "run42"}))
    p.add_result("missing", unsupported("band_gap", "test", "not computed here"))
    p.add_array("dft::run42::density", StoredArray(
        np.random.default_rng(0).normal(size=(30, 30, 30)), "e/A^3", "grid", "volumetric"))
    p.add_scan(scan(), key="scan0001")
    return p


def saved(project, tmp_path, name="project.materia"):
    return project.save(str(tmp_path / name))


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def markdown(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def run_cells(nb):
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", "\n\n".join(code_cells(nb))],
                          capture_output=True, text=True, env=env, timeout=300)


def test_full_project_round_trip_and_cells_run(tmp_path):
    path = saved(full_project(), tmp_path)
    out = N.export(str(tmp_path / "full.ipynb"), project_path=path)
    nb = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    N.validate(nb)
    meta = nb["metadata"]["materia"]
    assert meta["project_path"] == os.path.abspath(path)
    assert meta["project_sha256"] == N.file_sha256(path)
    assert meta["materia_version"] and meta["project_schema_version"]
    assert meta["run_identifiers"] == {"dft": ["run42"]}
    assert set(meta["array_sha256"]) == {"dft::run42::density"}
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert all(c["outputs"] == [] and c["execution_count"] is None
               for c in nb["cells"] if c["cell_type"] == "code")
    text = markdown(nb)
    for fragment in ("Si8", "e/A^3", "unsupported: not computed here", "abc123",
                     "run42", "tier3-external-first-principles", "height (A)",
                     "Limitations of this notebook", meta["structure_digests"]["struct0001"]):
        assert fragment.replace("_", "\\_") in text or fragment in text, fragment
    run = run_cells(nb)
    assert run.returncode == 0, run.stderr
    assert "struct0001 Si8 8 atoms, matches" in run.stdout
    assert "dft::run42::density (30, 30, 30) e/A^3 matches" in run.stdout
    assert "The project file has changed" not in run.stdout
    assert "Materia project provenance" in run.stdout


def test_arrays_are_summarised_not_copied(tmp_path):
    project = full_project()
    out = N.export(str(tmp_path / "a.ipynb"), project=project)
    assert out["bytes"] < 60_000
    data = project.arrays["dft::run42::density"].data
    text = Path(out["path"]).read_text(encoding="utf-8")
    assert f"{data.ravel()[123]:.6g}" not in text
    assert "30x30x30" in text and "SHA-256" in text


def test_changed_project_file_is_reported(tmp_path):
    project = full_project()
    path = saved(project, tmp_path)
    out = N.export(str(tmp_path / "c.ipynb"), project_path=path)
    project.structure.positions = project.structure.positions + 0.01
    project.arrays["dft::run42::density"].data[0, 0, 0] += 1.0
    project.save(path)
    run = run_cells(json.loads(Path(out["path"]).read_text(encoding="utf-8")))
    assert run.returncode == 0, run.stderr
    assert "The project file has changed" in run.stdout
    assert "struct0001 Si8 8 atoms, differs from the record" in run.stdout
    assert "differs from the record" in run.stdout.split("dft::run42::density")[1]


def test_empty_unsaved_project_embeds_a_summary(tmp_path):
    out = N.export(str(tmp_path / "empty.ipynb"), project=Project("Empty"))
    nb = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    assert nb["metadata"]["materia"]["project_path"] is None
    text = markdown(nb)
    for fragment in ("an unsaved project", "The project has no structures.",
                     "The project has no results.", "The project has no stored arrays.",
                     "The project has no scans.", "No run identifiers"):
        assert fragment in text
    run = run_cells(nb)
    assert run.returncode != 0 and "unsaved project" in run.stderr


def test_empty_saved_project_runs(tmp_path):
    path = saved(Project("Empty saved"), tmp_path)
    run = run_cells(N.build(project_path=path))
    assert run.returncode == 0, run.stderr
    assert "Project: Empty saved" in run.stdout


def test_unicode_names(tmp_path):
    name = "Kristall äöü 石英 שלום \U0001f9ea"
    project = Project(name)
    s = Structure([14], np.zeros((1, 3)), Cell(np.eye(3) * 5, (True, True, True)))
    project.add_structure(s, activate=True)
    project.structures["石英 \U0001f9ea"] = project.structures.pop(project.active_structure_key)
    project.active_structure_key = "石英 \U0001f9ea"
    path = saved(project, tmp_path, "unicode 石英.materia")
    out = N.export(str(tmp_path / "u \U0001f9ea.ipynb"), project_path=path)
    raw = Path(out["path"]).read_text(encoding="utf-8")
    assert "\U0001f9ea" in raw
    run = run_cells(json.loads(raw))
    assert run.returncode == 0, run.stderr
    assert "石英 \U0001f9ea Si 1 atoms, matches" in run.stdout
    assert f"Project: {name}" in run.stdout


def test_malicious_text_stays_data(tmp_path):
    project = full_project(name=PAYLOAD)
    project.structures[PAYLOAD] = project.structures.pop("struct0001")
    project.active_structure_key = PAYLOAD
    project.results[PAYLOAD] = project.results.pop("relax::energy")
    project.arrays[PAYLOAD] = project.arrays.pop("dft::run42::density")
    project.scans[PAYLOAD] = project.scans.pop("scan0001")
    path = saved(project, tmp_path)
    nb = N.build(project_path=path)
    for source in code_cells(nb):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert {a.name for a in node.names} <= {"hashlib", "math", "numpy", "materia",
                                                        "matplotlib.pyplot"}
            if isinstance(node, ast.Attribute):
                assert node.attr != "system"
    constants = {n.value for s in code_cells(nb) for n in ast.walk(ast.parse(s))
                 if isinstance(n, ast.Constant)}
    assert PAYLOAD in constants
    text = markdown(nb)
    assert "<script" not in text and "&lt;script&gt;" in text
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            headings = [line for line in "".join(cell["source"]).split("\n")
                        if line.startswith("#")]
            assert all(line.startswith(("# Materia project", "## ")) for line in headings)
    run = run_cells(nb)
    assert run.returncode == 0, run.stderr
    assert "pwned" not in run.stdout


def test_content_is_deterministic(tmp_path):
    path = saved(full_project(), tmp_path)
    first = N.export(str(tmp_path / "one.ipynb"), project_path=path)
    second = N.export(str(tmp_path / "two.ipynb"), project_path=path)
    assert first["sha256"] == second["sha256"]
    assert Path(first["path"]).read_bytes() == Path(second["path"]).read_bytes()
    assert N.serialise(N.build(project_path=path)) == Path(first["path"]).read_text("utf-8")


def test_long_lists_and_values_are_bounded(tmp_path):
    project = Project("Many")
    for k in range(5):
        project.add_structure(Structure([14], np.zeros((1, 3)),
                                        Cell(np.eye(3) * 5, (True, True, True))))
    prov = Provenance("test", Fidelity.NON_PHYSICAL, Origin.CALCULATED)
    project.add_result("long", Result("text", "y" * 5000, "", prov))
    project.add_result("big", Result("series", np.arange(10_000.0), "eV", prov))
    nb = N.build(project, max_items=2, preview_elements=3)
    text = markdown(nb)
    assert "Only the first 2 of 5 structures" in text
    assert "y" * 300 not in text and "(10000,) float64: min 0, max 9999, mean 4999.5" in text


def test_atomic_failure_leaves_nothing(tmp_path, monkeypatch):
    target = tmp_path / "keep.ipynb"
    target.write_text("previous", encoding="utf-8")

    def broken(notebook):
        raise N.NotebookError("simulated validation failure")

    project = full_project()
    notebook = N.build(project)
    monkeypatch.setattr(N, "build", lambda *a, **k: notebook)
    monkeypatch.setattr(N, "validate", broken)
    with pytest.raises(N.NotebookError, match="simulated"):
        N.export(str(target), project=project)
    assert target.read_text(encoding="utf-8") == "previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.ipynb"]


@pytest.mark.parametrize("name, fragment", [("out.json", "end in .ipynb"),
                                            ("missing/out.ipynb", "does not exist")])
def test_bad_targets(tmp_path, name, fragment):
    with pytest.raises(N.NotebookError, match=fragment):
        N.export(str(tmp_path / name), project=Project("x"))


def test_bad_sources(tmp_path):
    with pytest.raises(N.NotebookError, match="does not exist"):
        N.build(project_path=str(tmp_path / "absent.materia"))
    with pytest.raises(N.NotebookError, match="Give a project"):
        N.build()
    with pytest.raises(N.NotebookError, match="at least 1"):
        N.build(Project("x"), max_items=0)


def valid():
    return N.build(Project("v"))


@pytest.mark.parametrize("damage, fragment", [
    (lambda nb: nb.pop("metadata"), "missing"),
    (lambda nb: nb.update(extra=1), "unexpected keys"),
    (lambda nb: nb.update(nbformat=3), "nbformat"),
    (lambda nb: nb.update(nbformat_minor=4), "nbformat_minor"),
    (lambda nb: nb["cells"][0].pop("id"), "missing"),
    (lambda nb: nb["cells"][0].update(id="bad id!"), "id must be"),
    (lambda nb: nb["cells"][1].update(id=nb["cells"][0]["id"]), "duplicate"),
    (lambda nb: nb["cells"][0].update(outputs=[]), "unexpected keys"),
    (lambda nb: nb["cells"][0].update(cell_type="picture"), "unknown cell_type"),
    (lambda nb: nb["cells"][0].update(source=[1, 2]), "source"),
    (lambda nb: next(c for c in nb["cells"] if c["cell_type"] == "code").update(
        execution_count=-1), "execution_count"),
    (lambda nb: next(c for c in nb["cells"] if c["cell_type"] == "code").update(
        outputs={}), "outputs"),
    (lambda nb: nb["metadata"].update(kernelspec={"name": "python3"}), "kernelspec"),
])
def test_validator_rejects_broken_notebooks(damage, fragment):
    nb = valid()
    damage(nb)
    with pytest.raises(N.NotebookError, match=fragment):
        N.validate(nb)


def test_nbformat_accepts_the_notebook_when_installed(tmp_path):
    nbformat = pytest.importorskip(
        "nbformat", reason="nbformat is not installed; the schema check did not run")
    path = saved(full_project(), tmp_path)
    out = N.export(str(tmp_path / "n.ipynb"), project_path=path)
    notebook = nbformat.read(out["path"], as_version=4)
    nbformat.validate(notebook)


def test_markdown_escaping():
    assert N.md("<b>&</b>") == "&lt;b&gt;&amp;&lt;/b&gt;"
    assert N.md("a|b`c*d_e#f") == "a\\|b\\`c\\*d\\_e\\#f"
    assert N.md("line one\n# two") == "line one \\# two"
    assert N.md("x" * 500).endswith("(500 characters)")
