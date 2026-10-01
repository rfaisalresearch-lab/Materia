"""Equation of state: specification, pinned settings, refusals, fit and output validation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import eos as E
from materia.experiments.dft import spec as specs
from materia.provenance import Origin
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import environment as fake_environment
from tests.support.fake_gpaw_eos import B0_GPA, B1, E0_PER_ATOM, V0_PER_ATOM, FakeEOSGPAW


@pytest.fixture
def environment(tmp_path):
    return fake_environment(tmp_path)


def silicon(a=(V0_PER_ATOM * 8.0) ** (1.0 / 3.0)):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


def build(environment, **extra):
    return E.build(silicon(), environment, "s1", **{"kpoints": [4, 4, 4], **extra})


def test_birch_murnaghan_fit_recovers_known_parameters():
    volumes = np.linspace(36.0, 44.0, 9)
    b0 = 0.55
    energies = E.birch_murnaghan(volumes, -10.8, 40.0, b0, 4.3)
    fit = E.fit_birch_murnaghan(volumes, energies)
    assert fit["V0_A3"] == pytest.approx(40.0, rel=1e-9)
    assert fit["B0_eV_A3"] == pytest.approx(b0, rel=1e-7)
    assert fit["B1"] == pytest.approx(4.3, rel=1e-6)
    step = 1e-5
    numeric = -(E.birch_murnaghan(41.0 + step, -10.8, 40.0, b0, 4.3)
                - E.birch_murnaghan(41.0 - step, -10.8, 40.0, b0, 4.3)) / (2 * step)
    assert E.birch_murnaghan_pressure(41.0, 40.0, b0, 4.3) == pytest.approx(numeric, rel=1e-7)


class TestSpecification:
    def test_frozen_versioned_and_round_trips(self, environment):
        spec = build(environment)
        assert spec.schema == E.SCHEMA and spec.version == E.VERSION
        with pytest.raises(FrozenInstanceError):
            spec.n_points = 3
        again = E.EOSSpec.from_dict(json.loads(json.dumps(spec.as_dict())))
        assert again == spec and again.digest == spec.digest
        assert E.changed(spec, environment, n_points=9).digest != spec.digest
        with pytest.raises(specs.SpecError, match="version"):
            E.EOSSpec.from_dict({**spec.as_dict(), "version": "1.0"})
        with pytest.raises(specs.SpecError, match="Unknown"):
            E.EOSSpec.from_dict({**spec.as_dict(), "colour": 1})

    def test_defaults_and_observables(self, environment):
        spec = build(environment)
        assert spec.volume_scales == (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06)
        assert spec.ground_state.observables == ("energy", "forces", "stress")
        assert E.check(spec, environment).ok

    def test_every_point_uses_the_reference_grid(self, environment):
        spec = E.build(silicon(), environment, "s1")
        automatic = spec.ground_state.kpoints
        points = E.point_specs(spec, environment)
        assert [p.kpoints for p in points] == [automatic] * 7
        volumes = [abs(np.linalg.det(np.asarray(p.cell_A))) for p in points]
        assert np.allclose(volumes, spec.volumes(), rtol=1e-12)
        for p in points:
            assert specs.gpaw_parameters(p) == specs.gpaw_parameters(spec.ground_state)
        compressed = specs.build(E.structure_of(spec.ground_state, 0.75), environment, None)
        assert compressed.kpoints != automatic
        report = E.check(spec, environment)
        assert not any("automatic choice" in w for w in report.warnings)
        assert any("same" in w and "k-point grid" in w for w in report.warnings)

    @pytest.mark.parametrize("name, value", [
        ("volume_min_scale", "small"), ("n_points", 5.5), ("volume_scales", [1.0]),
        ("fit", "murnaghan"), ("source", {}),
    ])
    def test_mistyped_variables_raise(self, environment, name, value):
        with pytest.raises(specs.SpecError):
            build(environment, **{name: value})


def slab():
    a = 3.84
    cell = np.diag([a, a, 20.0])
    return Structure(np.array([14, 14]), np.array([[0, 0, 9.0], [a / 2, a / 2, 10.3]]),
                     Cell(cell, (True, True, False)))


REFUSALS = [
    ("slab", lambda env: E.build(slab(), env, "s1"), "source"),
    ("real-space grid", lambda env: build(env, representation="fd"), "representation"),
    ("charged", lambda env: build(env, charge_e=1.0,
                                  charged_periodic_policy="uniform-background"), "charge_e"),
    ("reversed range", lambda env: build(env, volume_min_scale=1.05, volume_max_scale=0.95),
     "volume_min_scale"),
    ("range too wide", lambda env: build(env, volume_min_scale=0.5), "volume_min_scale"),
    ("too few points", lambda env: build(env, n_points=4), "n_points"),
    ("too many points", lambda env: build(env, n_points=16), "n_points"),
    ("tampered volumes", lambda env: E.EOSSpec.from_dict({**build(env).as_dict(),
                                                           "volume_scales": [0.9] * 7}),
     "volume_scales"),
    ("tampered source", lambda env: E.EOSSpec.from_dict({**build(env).as_dict(), "source": {
        **build(env).source, "electronic_digest": "0" * 64}}), "source"),
]


@pytest.mark.parametrize("label, factory, variable", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_variable(environment, label, factory, variable):
    report = E.check(factory(environment), environment)
    assert not report.ok
    assert variable in report.by_field, report.by_field


class TestExecute:
    def test_complete_run_recovers_the_curve(self, environment, monkeypatch):
        worker = FakeEOSGPAW()
        monkeypatch.setattr(runner, "run_job", worker)
        spec = build(environment, occupations="fixed", smearing_eV=0.0)
        outcome = E.execute(spec, environment, run_id="e1")
        assert outcome.ok, outcome.reason
        value = outcome.results["eos"].value
        assert value["V0_A3_per_atom"] == pytest.approx(V0_PER_ATOM, rel=1e-8)
        assert value["B0_GPa"] == pytest.approx(B0_GPA, rel=1e-6)
        assert value["B1"] == pytest.approx(B1, rel=1e-5)
        assert value["E0_eV_per_atom"] == pytest.approx(E0_PER_ATOM, abs=1e-9)
        assert value["checks"]["stress_pressure_max_difference_GPa"] < 1e-6
        assert len(worker.jobs) == 7
        assert {json.dumps(j["parameters"], sort_keys=True) for j in worker.jobs} == \
            {json.dumps(specs.gpaw_parameters(spec.ground_state), sort_keys=True)}
        record = outcome.results["eos"]
        assert record.provenance.model == E.MODEL
        assert record.provenance.origin is Origin.CALCULATED
        assert record.provenance.inputs_digest == spec.digest
        assert E.REFERENCES[0] in record.provenance.references
        assert set(outcome.arrays) == {"scales", "volumes", "energies", "free_energies",
                                       "fit_pressures", "free_energy_fit_pressures",
                                       "stress_pressures", "max_forces"}
        assert np.array_equal(outcome.arrays["energies"].data,
                              outcome.arrays["free_energies"].data)
        assert value["array_sha256"]["energies"] == outcome.arrays["energies"].sha256()
        assert value["experimental_comparison"] is None

    @pytest.mark.parametrize("mode, fragment", [
        ("failed", "did not produce a result"), ("not_converged", "did not produce a result"),
        ("echo", "did not produce a result"), ("nan", "not finite"),
        ("shifted", "outside the sampled volumes"), ("noisy", "not a smooth curve"),
    ])
    def test_bad_points_yield_nothing(self, environment, monkeypatch, mode, fragment):
        monkeypatch.setattr(runner, "run_job", FakeEOSGPAW(mode))
        outcome = E.execute(build(environment), environment)
        assert outcome.status == "failed"
        assert fragment in outcome.reason, outcome.reason
        assert outcome.results == {} and outcome.arrays == {}

    def test_warnings_for_forces_and_pulay_stress(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeEOSGPAW("forces"))
        outcome = E.execute(build(environment), environment)
        assert outcome.ok and any("free internal parameters" in w for w in outcome.warnings)
        monkeypatch.setattr(runner, "run_job", FakeEOSGPAW("pulay"))
        outcome = E.execute(build(environment), environment)
        assert outcome.ok and any("Pulay" in w for w in outcome.warnings)

    def test_cancellation_yields_nothing(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeEOSGPAW("cancelled"))
        outcome = E.execute(build(environment), environment)
        assert outcome.status == "cancelled" and not outcome.results and not outcome.arrays
        early = E.execute(build(environment), environment, cancelled=lambda: True)
        assert early.status == "cancelled" and "after 0" in early.reason

    def test_refused_specification_starts_nothing(self, environment, monkeypatch):
        worker = FakeEOSGPAW()
        monkeypatch.setattr(runner, "run_job", worker)
        with pytest.raises(specs.SpecRefused):
            E.execute(build(environment, n_points=3), environment)
        assert worker.jobs == []


def test_the_fit_is_to_the_zero_width_energy_not_the_free_energy(environment, monkeypatch):
    """With smeared occupations GPAW reports E - TS and its zero-width extrapolation.

    The zero-width energy follows the known curve; the free energy adds an
    entropy term that grows with volume and moves the minimum. A 0 K static
    lattice equation of state must recover the known V0 from the zero-width
    energy, keep the free energy as a separate curve, and compare GPAW's
    stress, the derivative of the free energy, with -dF/dV. The free energy here is
    a Birch-Murnaghan curve plus a linear entropy term, which a Birch-Murnaghan
    form represents only to about 1e-4 GPa in pressure, hence the 1e-3 GPa bound.
    """
    from tests.support.fake_gpaw_eos import TS_SLOPE_EV_PER_A3

    monkeypatch.setattr(runner, "run_job", FakeEOSGPAW())
    spec = build(environment, occupations="fermi-dirac", smearing_eV=0.1)
    outcome = E.execute(spec, environment)
    assert outcome.ok, outcome.reason
    value = outcome.results["eos"].value
    assert value["energy_definition"] == "zero-width extrapolated energy"
    assert value["V0_A3_per_atom"] == pytest.approx(V0_PER_ATOM, rel=1e-8)
    assert value["B0_GPa"] == pytest.approx(B0_GPA, rel=1e-6)
    free = value["free_energy_fit"]
    assert free["V0_A3_per_atom"] != pytest.approx(V0_PER_ATOM, rel=1e-3)
    points = value["points"]
    for row in points:
        assert row["free_energy_eV"] - row["energy_eV"] == pytest.approx(
            -TS_SLOPE_EV_PER_A3 * row["volume_A3"], abs=1e-12)
    assert value["checks"]["stress_pressure_max_difference_GPa"] < 1e-3
    assert value["checks"]["stress_compared_with"] == "-dF/dV of the free-energy fit"
    zero_width_pressure = np.array(value["fit_pressure_GPa"])
    stress = np.array([r["pressure_GPa"] for r in points])
    assert np.abs(stress - zero_width_pressure).max() > 0.5
    assert np.allclose(outcome.arrays["energies"].data, [r["energy_eV"] for r in points])
    assert np.allclose(outcome.arrays["free_energies"].data,
                       [r["free_energy_eV"] for r in points])
