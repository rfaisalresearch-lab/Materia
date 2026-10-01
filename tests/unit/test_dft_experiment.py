"""Ground-state DFT specification, storage and refusal behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import zipfile

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import spec as specs
from materia.experiments.dft.run import execute, parameter_mismatch
from materia.project_format.arrays import ArrayStoreError, StoredArray, read, write
from materia.provenance.classification import (
    Claim,
    Classification,
    ClassificationError,
    Evidence,
)
from materia.solvers.gpaw_driver import GPAWEnvironment
from materia.solvers.gpaw_driver import runner


@pytest.fixture
def environment(tmp_path):
    setup = tmp_path / "setups"
    setup.mkdir()
    for symbol, number, core, valence in (("H", 1, 0, 1), ("Si", 14, 10, 4)):
        for functional in ("LDA", "PBE"):
            (setup / f"{symbol}.{functional}").write_text(
                f'<paw_setup><atom symbol="{symbol}" Z="{number}" '
                f'core="{core}" valence="{valence}"/></paw_setup>',
                encoding="utf-8",
            )
    return GPAWEnvironment(
        interpreter="/fake/gpaw/python",
        source="test",
        code_available=True,
        datasets_available=True,
        gpaw_version="25.7.0",
        ase_version="3.29.0",
        python_version="3.13.0",
        setup_paths=[str(setup)],
        dataset_dirs=[{"path": str(setup), "files": 4}],
    )


def molecule(distance=0.74, box=8.0, pbc=(False, False, False)):
    centre = box / 2
    return Structure(
        np.array([1, 1]),
        np.array([[centre, centre, centre - distance / 2],
                  [centre, centre, centre + distance / 2]]),
        Cell(np.eye(3) * box, pbc),
    )


def test_specification_is_immutable_versioned_and_round_trips(environment):
    spec = specs.build(molecule(), environment, structure_key="struct0001",
                       grid_spacing_A=0.22)
    rebuilt = specs.GroundStateSpec.from_dict(spec.as_dict())
    assert rebuilt == spec
    assert rebuilt.digest == spec.digest
    assert spec.schema == "materia.dft.ground-state"
    assert spec.version == "1.0"
    assert len(spec.paw_datasets) == 1
    assert len(spec.paw_datasets[0][2]) == 64
    with pytest.raises(FrozenInstanceError):
        spec.xc = "LDA"


def test_digest_and_restart_identity_separate_experiment_from_reusable_state(environment):
    base = specs.build(molecule(), environment, grid_spacing_A=0.22)
    tighter = specs.changed(base, environment,
                            energy_tol_eV_per_electron=1.0e-5)
    different_grid = specs.changed(base, environment, grid_spacing_A=0.18)
    different_geometry = specs.build(molecule(0.80), environment,
                                     grid_spacing_A=0.22)
    assert base.digest != tighter.digest
    assert specs.restart_key(base) == specs.restart_key(tighter)
    assert specs.restart_key(base) != specs.restart_key(different_grid)
    assert specs.restart_key(base) != specs.restart_key(different_geometry)


def test_every_requested_setting_reaches_the_gpaw_parameters(environment):
    spec = specs.build(
        molecule(), environment,
        xc="PBE",
        representation="fd",
        grid_spacing_A=0.19,
        occupations="fermi-dirac",
        smearing_eV=0.08,
        n_bands=3,
        kpoints=[1, 1, 1],
        energy_tol_eV_per_electron=2.0e-4,
        density_tol_electrons_per_electron=3.0e-5,
        eigenstates_tol_eV2_per_electron=2.0e-8,
        forces_tol_eV_A=0.04,
        max_scf_iterations=211,
        external_field_V_per_A=[0.0, 0.0, 0.02],
        observables=["energy", "forces", "density"],
    )
    parameters = specs.gpaw_parameters(spec)
    assert parameters["mode"] == {"name": "fd"}
    assert parameters["xc"] == "PBE"
    assert parameters["h"] == pytest.approx(0.19)
    assert parameters["occupations"] == {"name": "fermi-dirac", "width": 0.08}
    assert parameters["nbands"] == 3
    assert parameters["kpts"] == {"size": [1, 1, 1], "gamma": True}
    assert parameters["convergence"] == {
        "energy": 2.0e-4,
        "density": 3.0e-5,
        "eigenstates": 2.0e-8,
        "bands": "occupied",
        "forces": 0.04,
    }
    assert parameters["maxiter"] == 211
    assert parameters["external"]["strength"] == pytest.approx(0.02)
    assert specs.check(spec, environment).ok


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"representation": "pw", "cutoff_eV": 300.0,
          "grid_spacing_A": None, "poisson": "gpaw-default"}, "representation"),
        ({"kpoints": [2, 1, 1]}, "kpoints"),
        ({"charge_e": 0.5}, "charge_e"),
        ({"random_seed": 7}, "random_seed"),
        ({"observables": ["forces"]}, "observables"),
    ],
)
def test_unsupported_requests_are_refused_before_execution(environment, changes, field):
    spec = specs.build(molecule(), environment, **changes)
    report = specs.check(spec, environment)
    assert not report.ok
    assert field in report.by_field


def test_dataset_identity_change_is_refused(environment):
    spec = specs.build(molecule(), environment)
    symbol, path, digest = spec.paw_datasets[0]
    changed = replace(spec, paw_datasets=((symbol, path, "0" * len(digest)),))
    report = specs.check(changed, environment)
    assert "paw_datasets" in report.by_field


@pytest.mark.parametrize("status", ["not-converged", "cancelled", "failed"])
def test_unsuccessful_runs_keep_only_a_non_numeric_record(
        monkeypatch, environment, status):
    spec = specs.build(molecule(), environment, observables=["energy", "density"])
    fake = runner.GPAWRun(
        status=status,
        result={"message": "stopped", "scf_history": [{"iteration": 2}]},
        iterations=2,
        error="solver stopped" if status == "failed" else "",
    )
    monkeypatch.setattr(runner, "run_job", lambda *args, **kwargs: fake)
    outcome = execute(spec, environment)
    assert set(outcome.results) == {"run"}
    assert outcome.arrays == {}
    assert outcome.record.value is None
    assert not outcome.record.supported
    assert outcome.record.extra["status"] == status
    assert outcome.record.provenance.inputs_digest == spec.digest


def test_parameter_echo_detects_missing_and_changed_values():
    sent = {"xc": "PBE", "kpts": {"size": [2, 2, 2], "gamma": True},
            "charge": 0.0}
    assert parameter_mismatch(sent, sent) == ""
    assert "kpts.gamma" in parameter_mismatch(
        sent, {"xc": "PBE", "kpts": {"size": [2, 2, 2], "gamma": False},
               "charge": 0.0})
    assert "charge is missing" in parameter_mismatch(
        sent, {"xc": "PBE", "kpts": {"size": [2, 2, 2], "gamma": True}})


def test_chunked_array_checksum_refuses_corruption(tmp_path):
    path = tmp_path / "array.zip"
    damaged = tmp_path / "damaged.zip"
    stored = StoredArray(np.arange(30, dtype=float).reshape(5, 3, 2), "e/A^3")
    with zipfile.ZipFile(path, "w") as archive:
        write(archive, "density", stored)
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "w") as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "arrays/density/chunk-00000.npy":
                payload = b"damaged"
            target.writestr(name, payload)
    with zipfile.ZipFile(damaged) as archive:
        with pytest.raises(ArrayStoreError, match="not a readable NumPy array"):
            read(archive, "density")


def test_claim_vocabulary_refuses_proof_and_requires_evidence():
    with pytest.raises(ClassificationError, match="formal proof"):
        Claim("This proves a universal law", Classification.CONJECTURE).validate()
    with pytest.raises(ClassificationError, match="converged"):
        Claim("A prediction", Classification.COMPUTATIONAL_PREDICTION).validate()
    claim = Claim(
        "The model predicts a bound state under these conditions",
        Classification.COMPUTATIONAL_PREDICTION,
        [Evidence("result", "dft::run::energy", model="GPAW", converged=True)],
    )
    claim.validate()
    assert claim.as_dict()["classification"] == "computational-prediction"


def test_automatic_k_grid_is_dense_enough_for_metals_and_is_flagged(environment):
    a = 3.632
    copper = Structure(np.array([29]), np.zeros((1, 3)),
                       Cell(0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                            (True, True, True)))
    automatic = specs.build(copper, environment)
    assert automatic.kpoints == (12, 12, 12)
    warnings = specs.check(automatic, environment).warnings
    assert any("automatic choice" in w and "not comparable" in w for w in warnings)
    explicit = specs.changed(automatic, environment, kpoints=[16, 16, 16])
    assert not any("automatic choice" in w
                   for w in specs.check(explicit, environment).warnings)
    assert specs.default_kpoints(np.eye(3) * 2.5, (True, True, False)) == (12, 12, 1)


def test_supercell_study_says_whether_its_sampling_is_equivalent(environment):
    from materia.experiments.dft import convergence

    a = 5.43
    silicon = Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                        Cell(0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                             (True, True, True)))
    base = specs.build(silicon, environment, None, kpoints=[4, 4, 4])
    points = convergence.plan(convergence.StudySpec(base, "supercell", (1, 2, 3)),
                              environment)
    flags = {value: detail["sampling_equivalent"] for value, _, detail in points}
    assert flags == {1: True, 2: True, 3: False}
    three = next(detail for value, _, detail in points if value == 3)
    assert three["kpoints"] == [2, 2, 2] and "not equivalent" in three["description"]


def test_stress_below_the_stress_cutoff_is_flagged(environment):
    a = 3.632
    copper = Structure(np.array([29]), np.zeros((1, 3)),
                       Cell(0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                            (True, True, True)))
    low = specs.build(copper, environment, kpoints=[12, 12, 12], cutoff_eV=400.0,
                      observables=["energy", "stress"])
    assert any("Pulay" in w for w in specs.check(low, environment).warnings)
    high = specs.changed(low, environment, cutoff_eV=600.0)
    assert not any("Pulay" in w for w in specs.check(high, environment).warnings)
    energy_only = specs.changed(low, environment, observables=["energy"])
    assert not any("Pulay" in w for w in specs.check(energy_only, environment).warnings)
