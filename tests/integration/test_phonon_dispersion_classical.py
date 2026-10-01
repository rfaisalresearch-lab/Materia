"""Periodic phonons through the classical solver and Materia-to-ASE adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import phonon_dispersion as P
from materia.physics.potentials import StillingerWeber
from materia.solvers.classical import ClassicalSolver


def silicon_primitive(a=5.431):
    cell = np.array([
        [0.0, a / 2, a / 2],
        [a / 2, 0.0, a / 2],
        [a / 2, a / 2, 0.0],
    ])
    return Structure(
        [14, 14], np.array([[0.0, 0.0, 0.0], [a / 4, a / 4, a / 4]]),
        Cell(cell, (True, True, True)))


def configured():
    return P.PeriodicPhononSettings(
        supercell=(3, 3, 3),
        q_path=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0)),
        q_labels=("G", "", "X"),
        dos_mesh=(2, 2, 2),
        dos_points=301,
        dos_width_THz=0.05,
    )


def test_classical_solver_returns_traceable_dispersion_dos_and_force_constants():
    structure = silicon_primitive()
    positions = structure.positions.copy()
    cell = structure.cell.matrix.copy()
    out = ClassicalSolver(StillingerWeber("Si")).run(
        structure, task="phonon_dispersion", settings=configured())
    assert set(out.keys()) == {
        "phonon_dispersion", "phonon_dos", "real_space_force_constants"}
    dispersion = out["phonon_dispersion"]
    assert dispersion.supported and dispersion.unit == "THz"
    assert dispersion.value.shape == (3, 6)
    assert dispersion.extra["mode_kinds"][0].count("acoustic") == 3
    assert np.array_equal(dispersion.value[0, :3], np.zeros(3))
    assert out["phonon_dos"].unit == "states/THz"
    assert out["phonon_dos"].extra["integral_states"] == pytest.approx(6.0, abs=1e-10)
    assert out["real_space_force_constants"].value.shape == (27, 6, 6)
    assert dispersion.provenance.parameters["lo_to_correction"] is False
    assert len(dispersion.provenance.inputs_digest) == 16
    assert any("LO-TO" in item for item in dispersion.provenance.approximations)
    assert np.array_equal(structure.positions, positions)
    assert np.array_equal(structure.cell.matrix, cell)


def test_result_arrays_have_stable_checksums_and_complete_shapes():
    result = P.periodic_phonon_analysis(
        silicon_primitive(), StillingerWeber("Si"), configured())
    stored = result.stored_arrays()
    record = result.as_dict()
    assert set(record["array_checksums_sha256"]) == set(stored)
    assert all(len(value) == 64 for value in record["array_checksums_sha256"].values())
    assert record["array_shapes"]["phonon_frequencies"] == [3, 6]
    assert record["array_shapes"]["phonon_real_space_force_constants"] == [27, 6, 6]
    assert all(np.asarray(value.data).flags.c_contiguous or np.asarray(value.data).ndim == 1
               for value in stored.values())


def test_cancellation_removes_displacement_cache(tmp_path, monkeypatch):
    real_temporary_directory = P.tempfile.TemporaryDirectory
    paths = []

    class RecordingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            kwargs["dir"] = tmp_path
            self.inner = real_temporary_directory(*args, **kwargs)

        def __enter__(self):
            path = self.inner.__enter__()
            paths.append(Path(path))
            return path

        def __exit__(self, exc_type, exc, traceback):
            return self.inner.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(P.tempfile, "TemporaryDirectory", RecordingTemporaryDirectory)
    with pytest.raises(P.PhononCancelled):
        P.periodic_phonon_analysis(
            silicon_primitive(), StillingerWeber("Si"), configured(),
            progress=lambda fraction, message: fraction < 0.15)
    assert len(paths) == 1
    assert not paths[0].exists()
