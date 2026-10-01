"""DOS and PDOS: specification, refusals, sources, recomputation and output validation."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import dos as D
from materia.experiments.dft import spec as specs
from materia.experiments.dft.datasets import bound_channels
from materia.provenance import Origin
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import FERMI_EV, FakeGPAW
from tests.support.fake_gpaw_dos import environment as fake_environment


@pytest.fixture
def environment(tmp_path):
    return fake_environment(tmp_path)


def molecule(box=8.0):
    c = box / 2
    return Structure(np.array([1, 1]), np.array([[c, c, c - 0.37], [c, c, c + 0.37]]),
                     Cell(np.eye(3) * box, (False, False, False)))


def silicon(a=5.47):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


BULK = {"occupations": "fixed", "smearing_eV": 0.0, "kpoints": [2, 2, 2],
        "energy_min_eV": -15.0, "energy_max_eV": 5.0}


def bulk(environment, **extra):
    return D.build(silicon(), environment, "s1", **{**BULK, **extra})


class TestBoundChannels:
    def test_only_bound_states_are_projectable(self, environment):
        paths = environment.setup_paths
        assert bound_channels(f"{paths[0]}/Si.PBE") == ("s", "p")
        assert bound_channels(f"{paths[0]}/H.PBE") == ("s",)
        assert bound_channels(f"{paths[0]}/Cu.PBE") == ("s", "p", "d")

    def test_a_dataset_without_valence_states_offers_nothing(self, tmp_path):
        path = tmp_path / "X.PBE"
        path.write_text('<paw_setup><atom symbol="H" Z="1" core="0" valence="1"/></paw_setup>')
        assert bound_channels(str(path)) == ()


class TestSpecification:
    def test_frozen_versioned_and_round_trips(self, environment):
        spec = bulk(environment)
        assert spec.schema == D.SCHEMA and spec.version == D.VERSION
        with pytest.raises(FrozenInstanceError):
            spec.width_eV = 1.0
        again = D.DOSSpec.from_dict(json.loads(json.dumps(spec.as_dict())))
        assert again == spec and again.digest == spec.digest and len(spec.digest) == 64
        other = D.changed(spec, environment, width_eV=0.2)
        assert other.digest != spec.digest
        with pytest.raises(specs.SpecError, match="version"):
            D.DOSSpec.from_dict({**spec.as_dict(), "version": "2.0"})
        with pytest.raises(specs.SpecError, match="Unknown DOS field"):
            D.DOSSpec.from_dict({**spec.as_dict(), "colour": "blue"})

    def test_defaults(self, environment):
        spec = bulk(environment)
        assert spec.source["kind"] == "structure" and spec.source["run_id"] is None
        assert spec.ground_state.observables == ("energy", "fermi_level")
        assert spec.dos_kpoints == (4, 4, 4)
        assert spec.spin_channels == "total" and spec.broadening == "gaussian"
        assert spec.width_eV == 0.1 and spec.energy_reference == "fermi-level"
        assert spec.n_bands >= 12
        assert spec.projections == (("Si s", (1, 2), "s"), ("Si p", (1, 2), "p"))
        assert spec.projection_channels == (("Si", ("s", "p")),)
        assert spec.npoints == 2001
        cluster = D.build(molecule(), environment, "s1")
        assert cluster.dos_kpoints == (1, 1, 1)
        magnetic = D.build(molecule(), environment, "s1", charge_e=1.0)
        assert magnetic.ground_state.spin_polarized and magnetic.spin_channels == "resolved"
        tetra = bulk(environment, broadening="tetrahedron")
        assert tetra.width_eV is None and tetra.gpaw_width == 0.0

    def test_projection_selections(self, environment):
        spec = bulk(environment, projections=[{"element": "Si", "angular": "s"},
                                              {"atom_ids": [2], "angular": "p",
                                               "label": "second p"}])
        assert spec.projections == (("Si s", (1, 2), "s"), ("second p", (2,), "p"))
        assert bulk(environment, projections=[]).projections == ()
        for bad in ([{"element": "Si"}], [{"element": "Si", "angular": "g"}],
                    [{"element": "Si", "atom_ids": [1], "angular": "s"}],
                    [{"element": "Si", "angular": "s", "colour": 1}], "Si s"):
            with pytest.raises(specs.SpecError):
                bulk(environment, projections=bad)

    @pytest.mark.parametrize("name, value", [
        ("energy_reference", "vacuum"), ("broadening", "lorentzian"),
        ("spin_channels", "up"), ("energy_min_eV", "low"), ("n_bands", 1.5),
        ("dos_kpoints", 4), ("dos_kpoints_gamma_centered", "yes"), ("source", {}),
    ])
    def test_mistyped_variables_raise(self, environment, name, value):
        with pytest.raises(specs.SpecError):
            bulk(environment, **{name: value})

    def test_every_variable_reaches_the_job(self, environment):
        base = bulk(environment)
        job = D.worker_job(base)
        assert job["dos"]["nscf"] == {"kpts": {"size": [4, 4, 4], "gamma": True},
                                      "nbands": base.n_bands, "convergence": {"bands": "all"}}
        assert job["dos"]["npoints"] == 2001 and job["dos"]["width"] == 0.1
        assert job["dos"]["projections"][1] == {"label": "Si p", "indices": [0, 1],
                                                "angular": "p"}
        assert job["dos"]["channels"] == {"Si": ["s", "p"]}
        changes = {"energy_min_eV": -14.0, "energy_max_eV": 4.0, "energy_step_eV": 0.02,
                   "width_eV": 0.2, "broadening": "tetrahedron", "dos_kpoints": [6, 6, 6],
                   "dos_kpoints_gamma_centered": False, "n_bands": 20,
                   "projections": [{"element": "Si", "angular": "p"}]}
        for name, value in changes.items():
            spec = D.changed(base, environment, **{name: value})
            assert D.worker_job(spec)["dos"] != job["dos"], name
        electronic = D.changed(base, environment, cutoff_eV=500.0)
        assert D.worker_job(electronic)["parameters"] != job["parameters"]

    def test_valid_specifications_pass(self, environment):
        assert D.check(bulk(environment), environment).ok
        assert D.check(bulk(environment, broadening="tetrahedron"), environment).ok
        cluster = D.build(molecule(), environment, "s1", energy_reference="absolute",
                          energy_min_eV=-15.0, energy_max_eV=0.0, n_bands=4)
        assert D.check(cluster, environment).ok


REFUSALS = [
    ("absolute in bulk", lambda env: bulk(env, energy_reference="absolute"),
     "energy_reference"),
    ("absolute in a field", lambda env: D.build(
        molecule(), env, "s1", energy_reference="absolute",
        external_field_V_per_A=[0.0, 0.0, 0.1]), "energy_reference"),
    ("reversed window", lambda env: bulk(env, energy_min_eV=2.0, energy_max_eV=1.0),
     "energy_max_eV"),
    ("window too wide", lambda env: bulk(env, energy_min_eV=-100.0, energy_max_eV=10.0),
     "energy_max_eV"),
    ("not whole steps", lambda env: bulk(env, energy_step_eV=0.03), "energy_step_eV"),
    ("too many points", lambda env: bulk(env, energy_step_eV=0.0005, width_eV=0.5),
     "energy_step_eV"),
    ("step coarser than width", lambda env: bulk(env, energy_step_eV=0.1, width_eV=0.1),
     "energy_step_eV"),
    ("width range", lambda env: bulk(env, width_eV=5.0), "width_eV"),
    ("tetrahedron on a cluster", lambda env: D.build(molecule(), env, "s1",
                                                     broadening="tetrahedron"), "broadening"),
    ("tetrahedron one division", lambda env: bulk(env, broadening="tetrahedron",
                                                  dos_kpoints=[4, 4, 1]), "dos_kpoints"),
    ("tetrahedron with width", lambda env: bulk(env, broadening="tetrahedron", width_eV=0.1),
     "width_eV"),
    ("resolved spin-paired", lambda env: bulk(env, spin_channels="resolved"),
     "spin_channels"),
    ("k-points along an open axis", lambda env: D.build(molecule(), env, "s1",
                                                        dos_kpoints=[2, 1, 1]), "dos_kpoints"),
    ("too few bands", lambda env: bulk(env, n_bands=2), "n_bands"),
    ("unbound channel", lambda env: bulk(env, projections=[{"element": "Si", "angular": "d"}]),
     "projections"),
    ("unknown atom", lambda env: bulk(env, projections=[{"atom_ids": [9], "angular": "s"}]),
     "projections"),
    ("no atoms", lambda env: bulk(env, projections=[{"element": "O", "angular": "s"}]),
     "projections"),
    ("duplicate labels", lambda env: bulk(env, projections=[
        {"element": "Si", "angular": "s", "label": "x"},
        {"element": "Si", "angular": "p", "label": "x"}]), "projections"),
]


@pytest.mark.parametrize("label, factory, variable", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_variable(environment, label, factory, variable):
    report = D.check(factory(environment), environment)
    assert not report.ok
    assert variable in report.by_field, report.by_field


def test_ground_state_refusals_are_kept(environment):
    report = D.check(bulk(environment, kpoints=[0, 2, 2]), environment)
    assert "kpoints" in report.by_field


def test_tampered_source_and_channels_are_refused(environment):
    spec = bulk(environment)
    moved = D.DOSSpec.from_dict({**spec.as_dict(), "source": {
        **spec.source, "electronic_digest": "0" * 64}})
    assert "source" in D.check(moved, environment).by_field
    pinned = D.DOSSpec.from_dict({**spec.as_dict(),
                                  "projection_channels": [["Si", ["s", "p", "d"]]]})
    assert "projection_channels" in D.check(pinned, environment).by_field


def test_unavailable_gpaw_is_refused():
    from materia.solvers.gpaw_driver import GPAWEnvironment

    missing = GPAWEnvironment()
    spec = D.build(molecule(), missing, "s1")
    assert not D.check(spec, missing).ok
    with pytest.raises(specs.SpecRefused):
        D.execute(spec, missing)


class TestRecomputation:
    def test_gaussian_matches_gpaw_formula_and_integrates_exactly(self):
        eig = np.array([[-1.0, 0.5], [-0.8, 0.7]])
        weights = np.array([0.25, 0.75])
        energies = np.linspace(-5.0, 5.0, 10001)
        width = 0.2
        dos = D.gaussian_dos(eig, weights, energies, width)
        direct = sum(w * np.exp(-((energies - e) / width) ** 2) for w, row in zip(weights, eig)
                     for e in row) / (math.sqrt(math.pi) * width)
        assert np.allclose(dos, direct, rtol=1e-14, atol=0)
        exact = D.gaussian_window_count(eig, weights, -5.0, 5.0, width)
        assert exact == pytest.approx(2.0, abs=1e-12)
        assert np.trapezoid(dos, energies) == pytest.approx(exact, rel=1e-8)
        assert D.gaussian_window_count(eig, weights, -5.0, 0.0, width) == pytest.approx(1.0, abs=1e-4)

    def test_tetrahedron_counts_states(self):
        cell = np.eye(3) * 4.0
        size = (4, 4, 4)
        grid = np.stack(np.meshgrid(*[np.arange(n) / n for n in size], indexing="ij"), -1)
        eig = (np.cos(2 * np.pi * grid).sum(-1)).reshape(-1, 1)
        energies = np.linspace(-4.0, 4.0, 4001)
        dos = D.tetrahedron_dos(eig, energies, cell, size, list(range(64)))
        assert np.trapezoid(dos, energies) == pytest.approx(1.0, abs=1e-3)


class TestExecute:
    def test_complete_run_builds_every_result(self, environment, monkeypatch):
        worker = FakeGPAW()
        monkeypatch.setattr(runner, "run_job", worker)
        spec = bulk(environment)
        outcome = D.execute(spec, environment, run_id="d1")
        assert outcome.status == "complete", outcome.reason
        assert {"dos", "fermi_level", "energy", "charge_accounting", "scf_history",
                "run"} <= set(outcome.results)
        assert set(outcome.arrays) == {"energies", "dos_total", "eigenvalues",
                                       "kpoint_weights", "pdos"}
        summary = outcome.results["dos"].value
        assert summary["fermi_level_eV"] == FERMI_EV
        checks = summary["checks"]
        assert checks["recompute_max_relative_error"] < 1e-12
        assert checks["electron_count_difference_e"] == pytest.approx(0.0, abs=1e-6)
        assert checks["window_integral_error_states"] == pytest.approx(0.0, abs=1e-6)
        assert checks["projection_integrals_states"]["Si s"] > 0
        record = outcome.results["dos"]
        assert record.provenance.model == D.MODEL
        assert record.provenance.origin is Origin.CALCULATED
        assert record.provenance.inputs_digest == spec.digest
        assert record.provenance.parameters["nscf_parameters"]["nbands"] == spec.n_bands
        assert outcome.arrays["energies"].unit == "eV"
        assert outcome.arrays["pdos"].meta["labels"] == ["Si s", "Si p"]
        assert worker.jobs[0]["observables"] == ["energy", "fermi_level"]

    def test_spin_resolved_and_tetrahedron(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeGPAW())
        spin = D.build(molecule(), environment, "s1", charge_e=1.0, n_bands=4,
                       energy_min_eV=-15.0, energy_max_eV=0.0)
        outcome = D.execute(spin, environment)
        assert outcome.ok, outcome.reason
        assert {"dos_spin", "pdos_spin"} <= set(outcome.arrays)
        assert outcome.arrays["dos_spin"].data.shape == (2, spin.npoints)
        tetra = bulk(environment, broadening="tetrahedron", energy_step_eV=0.05)
        assert D.execute(tetra, environment).ok

    @pytest.mark.parametrize("mode, fragment", [
        ("failed", "boom"), ("timeout", "time limit"), ("not_converged", "did not converge"),
        ("echo_nscf", "non-self-consistent"), ("echo_width", "different DOS"),
        ("echo_grid", "different DOS"), ("bad_recompute", "recomputed"),
        ("nan", "non-finite"), ("negative", "negative"), ("short_bands", "highest computed band"),
        ("fermi_moved", "Fermi level"), ("channels", "bound channels"),
        ("no_projector", "projector"), ("nscf_unconverged", "every band"),
    ])
    def test_bad_output_yields_nothing(self, environment, monkeypatch, mode, fragment):
        monkeypatch.setattr(runner, "run_job", FakeGPAW(mode))
        outcome = D.execute(bulk(environment), environment, timeout_s=1.0)
        assert outcome.status == "failed"
        assert fragment in outcome.reason, outcome.reason
        assert outcome.results == {} and outcome.arrays == {}

    def test_spin_mismatch_is_refused(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeGPAW("spin_mismatch"))
        spec = D.build(molecule(), environment, "s1", charge_e=1.0, n_bands=4,
                       energy_min_eV=-15.0, energy_max_eV=0.0)
        outcome = D.execute(spec, environment)
        assert outcome.status == "failed" and "Spin up and spin down" in outcome.reason

    def test_cancellation_yields_nothing(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeGPAW("cancelled"))
        assert D.execute(bulk(environment), environment).status == "cancelled"
        early = D.execute(bulk(environment), environment, cancelled=lambda: True)
        assert early.status == "cancelled" and not early.results

    def test_refused_specification_starts_nothing(self, environment, monkeypatch):
        worker = FakeGPAW()
        monkeypatch.setattr(runner, "run_job", worker)
        with pytest.raises(specs.SpecRefused):
            D.execute(bulk(environment, spin_channels="resolved"), environment)
        assert worker.jobs == []
