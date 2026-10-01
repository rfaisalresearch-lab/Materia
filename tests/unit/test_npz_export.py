"""NumPy archive export: exact round trip without pickling, manifest, refusals."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from materia.dataio import npz
from materia.materials import default_library
from materia.microscopy.scan import ScanResult
from materia.project_format.arrays import StoredArray
from materia.project_format.project import Project
from materia.provenance import Fidelity, Origin, Provenance, Result, unsupported
from materia.solvers import registry
from materia.structure_builder.lattice import bulk


@pytest.fixture
def project():
    p = Project("npz test")
    structure = bulk(default_library().get("silicon"))
    p.add_structure(structure, activate=True)
    solver = registry.create("stillinger-weber-si")
    for key, result in solver.relax(structure.copy(), max_steps=5).results.items():
        p.add_result(f"relax::{key}", result)
    grid = np.random.default_rng(0).normal(size=(4, 5, 6))
    p.add_array("dft::r1::density", StoredArray(grid, "e/A^3", "test grid", "volumetric"))
    p.add_result("note", Result("note", "text only", "", Provenance("test", Fidelity.NON_PHYSICAL, Origin.CALCULATED)))
    p.add_result("missing", unsupported("band_gap", "test", "not computed here"))
    prov = Provenance(model="stm-test", fidelity=Fidelity.TIER2_SEMI_EMPIRICAL,
                      origin=Origin.CALCULATED)
    p.add_scan(ScanResult("STM", "constant-current", {"height": np.arange(12.0).reshape(3, 4)},
                          "height", {"height": "A"}, (0.0, 4.0, 0.0, 3.0), (4, 3), None, prov),
               key="scan1")
    return p


def test_round_trip_is_exact_and_needs_no_pickle(project, tmp_path):
    out = npz.export(str(tmp_path / "all.npz"), project)
    with np.load(out["path"], allow_pickle=False) as archive:
        manifest = json.loads(str(archive["__manifest__"]))
        s = project.structure
        assert np.array_equal(archive["structure/positions_A"], s.positions)
        assert np.array_equal(archive["structure/numbers"], s.numbers)
        assert np.array_equal(archive["structure/cell_A"], s.cell.matrix)
        assert archive["structure/pbc"].tolist() == list(s.cell.pbc)
        grid = project.arrays["dft::r1::density"].data
        assert np.array_equal(archive["arrays/dft::r1::density"], grid)
        energy = project.results["relax::energy"]
        assert archive["results/relax::energy"].shape == ()
        assert float(archive["results/relax::energy"]) == energy.value
        history = project.results["relax::relaxation_history"].value
        assert np.array_equal(archive["results/relax::relaxation_history/energy_eV"],
                              np.asarray(history["energy_eV"]))
        assert np.array_equal(archive["scans/scan1/height"], np.arange(12.0).reshape(3, 4))
        assert np.array_equal(archive["results/relax::forces"],
                              project.results["relax::forces"].value)
    entry = manifest["entries"]["arrays/dft::r1::density"]
    assert entry["unit"] == "e/A^3" and entry["stored_sha256"] == entry["sha256"]
    assert manifest["entries"]["results/relax::energy"]["unit"] == "eV"
    assert manifest["entries"]["results/relax::energy"]["model"].startswith("classical/")
    assert manifest["entries"]["scans/scan1/height"]["unit"] == "A"
    assert "results/note" in manifest["skipped"]
    assert "unsupported" in manifest["skipped"]["results/missing"]
    assert "results/relax::relaxed_structure" in manifest["skipped"]
    assert manifest["schema"] == npz.SCHEMA and manifest["version"] == npz.VERSION
    assert npz.verify(out["path"])["entries"] == manifest["entries"]


def test_parts_select_what_is_written(project, tmp_path):
    out = npz.export(str(tmp_path / "s.npz"), project, ("structure",))
    with np.load(out["path"], allow_pickle=False) as archive:
        assert all(n.startswith("structure/") for n in archive.files if n != "__manifest__")
    with pytest.raises(npz.NpzExportError, match="Unknown export part"):
        npz.export(str(tmp_path / "x.npz"), project, ("everything",))


@pytest.mark.parametrize("name, fragment", [("out.npy", "end in .npz"),
                                            ("missing/out.npz", "does not exist")])
def test_bad_paths_are_refused(project, tmp_path, name, fragment):
    with pytest.raises(npz.NpzExportError, match=fragment):
        npz.export(str(tmp_path / name), project)


def test_nothing_numeric_is_refused_and_writes_nothing(tmp_path):
    with pytest.raises(npz.NpzExportError, match="nothing numeric"):
        npz.export(str(tmp_path / "empty.npz"), Project("empty"))
    assert os.listdir(tmp_path) == []


def test_a_failed_verification_leaves_no_file(project, tmp_path, monkeypatch):
    def broken(path, manifest=None):
        raise npz.NpzExportError("checksum mismatch")
    monkeypatch.setattr(npz, "verify", broken)
    with pytest.raises(npz.NpzExportError, match="checksum"):
        npz.export(str(tmp_path / "bad.npz"), project)
    assert os.listdir(tmp_path) == []


def test_existing_file_is_replaced_only_after_success(project, tmp_path, monkeypatch):
    target = tmp_path / "keep.npz"
    target.write_bytes(b"previous")
    monkeypatch.setattr(npz, "verify", lambda *a, **k: (_ for _ in ()).throw(
        npz.NpzExportError("checksum mismatch")))
    with pytest.raises(npz.NpzExportError):
        npz.export(str(target), project)
    assert target.read_bytes() == b"previous"


def test_verify_detects_a_tampered_archive(project, tmp_path):
    out = npz.export(str(tmp_path / "t.npz"), project, ("structure",))
    with np.load(out["path"], allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    data["structure/positions_A"] = data["structure/positions_A"] + 1e-12
    np.savez(str(tmp_path / "tampered.npz"), **data)
    with pytest.raises(npz.NpzExportError, match="checksum"):
        npz.verify(str(tmp_path / "tampered.npz"))
