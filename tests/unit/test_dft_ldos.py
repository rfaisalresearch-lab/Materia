"""LDOS and Tersoff-Hamann images: specification, refusals, output validation, image geometry."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import ldos as L
from materia.experiments.dft import spec as specs
from materia.provenance import Origin
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_dos import environment as fake_environment
from tests.support.fake_gpaw_ldos import FakeLDOSGPAW


@pytest.fixture
def environment(tmp_path):
    return fake_environment(tmp_path)


def sheet(vacuum=16.0):
    a = 3.84
    cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, vacuum]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 5.0], [a / 2, a / 2, 5.8]]),
                     Cell(cell, (True, True, False)))


def molecule():
    return Structure(np.array([1, 1]), np.array([[4, 4, 3.63], [4, 4, 4.37]]),
                     Cell(np.eye(3) * 8.0, (False, False, False)))


SLAB = {"kpoints": [4, 4, 1], "grid_spacing_A": 0.2}


def slab(environment, **extra):
    return L.build(sheet(), environment, "s1", **{**SLAB, **extra})


class TestSpecification:
    def test_frozen_versioned_and_round_trips(self, environment):
        spec = slab(environment)
        assert spec.schema == L.SCHEMA and spec.version == L.VERSION
        with pytest.raises(FrozenInstanceError):
            spec.n_bands = 3
        again = L.LDOSSpec.from_dict(json.loads(json.dumps(spec.as_dict())))
        assert again == spec and again.digest == spec.digest
        assert L.changed(spec, environment, energy_min_eV=-2.0).digest != spec.digest
        with pytest.raises(specs.SpecError, match="version"):
            L.LDOSSpec.from_dict({**spec.as_dict(), "version": "2.0"})

    def test_defaults_and_job(self, environment):
        spec = slab(environment)
        assert (spec.energy_min_eV, spec.energy_max_eV) == (-1.0, 0.0)
        assert spec.spin_channels == "total"
        job = L.worker_job(spec)
        nscf = job["ldos"]["nscf"]
        assert nscf["symmetry"] == {"point_group": False, "time_reversal": True}
        assert nscf["convergence"] == {"bands": spec.n_bands}
        assert nscf["nbands"] == spec.n_bands + L.buffer_bands(spec.n_bands)
        assert nscf["kpts"]["size"] == [4, 4, 1]
        assert job["ldos"]["n_bands"] == spec.n_bands
        report = L.check(spec, environment)
        assert report.ok and report.options["stm_images"] is True
        assert any("pseudo-wavefunctions" in w for w in report.warnings)

    def test_every_variable_reaches_the_job(self, environment):
        base = slab(environment)
        job = L.worker_job(base)["ldos"]
        for name, value in (("energy_min_eV", -2.0), ("energy_max_eV", 0.5), ("n_bands", 20)):
            assert L.worker_job(L.changed(base, environment, **{name: value}))["ldos"] != job


REFUSALS = [
    ("narrow window", lambda env: slab(env, energy_min_eV=0.0, energy_max_eV=0.001),
     "energy_max_eV"),
    ("reversed window", lambda env: slab(env, energy_min_eV=1.0, energy_max_eV=-1.0),
     "energy_max_eV"),
    ("far window", lambda env: slab(env, energy_min_eV=-30.0), "energy_min_eV"),
    ("resolved spin-paired", lambda env: slab(env, spin_channels="resolved"), "spin_channels"),
    ("too few bands", lambda env: slab(env, n_bands=2), "n_bands"),
    ("tampered source", lambda env: L.LDOSSpec.from_dict({**slab(env).as_dict(), "source": {
        **slab(env).source, "electronic_digest": "0" * 64}}), "source"),
]


@pytest.mark.parametrize("label, factory, variable", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_variable(environment, label, factory, variable):
    report = L.check(factory(environment), environment)
    assert not report.ok and variable in report.by_field, report.by_field


def test_stm_support_depends_on_the_geometry(environment):
    assert L.stm_supported(slab(environment).ground_state)[0]
    ok, reason = L.stm_supported(L.build(molecule(), environment, "m").ground_state)
    assert not ok and "slab" in reason


class TestImages:
    def column_map(self):
        nx, ny, nz = 6, 5, 80
        cell = np.diag([3.0, 3.0, 20.0])
        z = np.arange(nz) * 20.0 / nz
        base = np.exp(-2.0 * 1.1 * np.abs(z - 6.0))
        bump = 1.0 + 0.3 * np.cos(2 * np.pi * np.arange(nx) / nx)[:, None, None] \
            * np.ones((1, ny, 1))
        return base[None, None, :] * bump, cell, np.array([[1.5, 1.5, 6.0]])

    def test_constant_height_is_linear_interpolation(self):
        ldos, cell, positions = self.column_map()
        image = L.stm_image(ldos, cell, positions, "constant-height", height_A=4.1)
        index = (6.0 + 4.1) / (20.0 / 80)
        lower = int(index)
        expected = (1 - (index - lower)) * ldos[:, :, lower] + (index - lower) * ldos[:, :,
                                                                                      lower + 1]
        assert np.allclose(image["values"], expected, rtol=1e-14)
        assert image["x_A"].shape == (6, 5)

    def test_constant_current_matches_ase(self):
        from ase.dft.stm import find_height

        ldos, cell, positions = self.column_map()
        isovalue = float(ldos[0, 0, int((6.0 + 4.0) / 0.25)])
        image = L.stm_image(ldos, cell, positions, "constant-current", isovalue=isovalue)
        spacing = 20.0 / 80
        for i in range(6):
            for j in range(5):
                reference = find_height(ldos[i, j], isovalue, spacing,
                                        z0=20.0 - L.FACE_MARGIN_A)
                assert image["values"][i, j] + 6.0 == pytest.approx(reference, abs=1e-12)
        assert image["values"].max() > image["values"].min()

    @pytest.mark.parametrize("kwargs, fragment", [
        ({"mode": "constant-height", "height_A": 1.0}, "outside the valid band"),
        ({"mode": "constant-height", "height_A": 13.5}, "outside the valid band"),
        ({"mode": "constant-current", "isovalue": 10.0}, "not reached"),
        ({"mode": "constant-current", "isovalue": -1.0}, "positive"),
        ({"mode": "sideways"}, "mode must be"),
    ])
    def test_invalid_requests_are_refused(self, kwargs, fragment):
        ldos, cell, positions = self.column_map()
        with pytest.raises(ValueError, match=fragment):
            L.stm_image(ldos, cell, positions, **kwargs)

    def test_augmentation_radius_raises_the_floor(self):
        ldos, cell, positions = self.column_map()
        with pytest.raises(ValueError, match="outside the valid band"):
            L.stm_image(ldos, cell, positions, "constant-height", height_A=2.2,
                        augmentation_radius_A=1.5)


class TestExecute:
    def test_complete_run(self, environment, monkeypatch):
        worker = FakeLDOSGPAW()
        monkeypatch.setattr(runner, "run_job", worker)
        spec = slab(environment)
        outcome = L.execute(spec, environment, run_id="l1")
        assert outcome.ok, outcome.reason
        value = outcome.results["ldos"].value
        assert value["states_in_window"] > 0
        assert value["pseudo_norm_ratio"] == pytest.approx(0.95, rel=1e-9)
        assert value["augmentation_radii_A"] == {"Si": 1.05}
        assert value["stm"]["supported"] is True
        assert set(outcome.arrays) == {"ldos", "eigenvalues", "kpoint_weights"}
        assert outcome.arrays["ldos"].kind == "volumetric"
        assert value["array_sha256"]["ldos"] == outcome.arrays["ldos"].sha256()
        record = outcome.results["ldos"]
        assert record.provenance.model == L.MODEL
        assert record.provenance.origin is Origin.CALCULATED
        assert L.REFERENCES[0] in record.provenance.references
        assert worker.jobs[0]["observables"] == ["energy", "fermi_level"]

    def test_spin_resolved(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeLDOSGPAW())
        spec = L.build(molecule(), environment, "m", charge_e=1.0, energy_min_eV=-3.0,
                       energy_max_eV=0.0)
        assert spec.spin_channels == "resolved"
        outcome = L.execute(spec, environment)
        assert outcome.ok, outcome.reason
        assert outcome.arrays["ldos_spin"].data.shape[0] == 2

    @pytest.mark.parametrize("mode, fragment", [
        ("failed", "boom"), ("timeout", "time limit"), ("not_converged", "did not converge"),
        ("nscf_unconverged", "every band"), ("echo_nscf", "non-self-consistent"),
        ("echo_window", "different window"), ("fermi_moved", "Fermi level"),
        ("nan", "non-finite"), ("negative", "negative"), ("wrong_grid", "grid"),
        ("wrong_cell", "different cell"), ("bad_count", "counted"),
        ("missing_band", "band may be missing"), ("short_bands", "highest computed band"),
        ("empty", "No Kohn-Sham state"), ("no_radii", "augmentation radius"),
        ("bad_norm", "integrates to"), ("weights", "weights"),
    ])
    def test_bad_output_yields_nothing(self, environment, monkeypatch, mode, fragment):
        monkeypatch.setattr(runner, "run_job", FakeLDOSGPAW(mode))
        outcome = L.execute(slab(environment), environment, timeout_s=1.0)
        assert outcome.status == "failed"
        assert fragment in outcome.reason, outcome.reason
        assert outcome.results == {} and outcome.arrays == {}

    def test_spin_mismatch_is_refused(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeLDOSGPAW("spin_mismatch"))
        spec = L.build(molecule(), environment, "m", charge_e=1.0, energy_min_eV=-3.0,
                       energy_max_eV=0.0)
        outcome = L.execute(spec, environment)
        assert outcome.status == "failed" and "Spin up and spin down" in outcome.reason

    def test_cancellation_and_refusal_keep_nothing(self, environment, monkeypatch):
        worker = FakeLDOSGPAW("cancelled")
        monkeypatch.setattr(runner, "run_job", worker)
        assert L.execute(slab(environment), environment).status == "cancelled"
        early = L.execute(slab(environment), environment, cancelled=lambda: True)
        assert early.status == "cancelled" and not early.results
        worker.jobs.clear()
        with pytest.raises(specs.SpecRefused):
            L.execute(slab(environment, n_bands=1), environment)
        assert worker.jobs == []


def test_window_count_is_the_sharp_open_window():
    eigen = np.array([[[-2.0, -0.5, 0.0, 0.7]], [[-2.0, -0.4, 0.3, 1.5]]])
    assert L.window_count(eigen, np.array([1.0]), 0.0, -1.0, 0.5) == pytest.approx(4.0)
    assert L.window_count(eigen[:1], np.array([1.0]), 0.0, -1.0, 0.0) == pytest.approx(2.0)
    assert math.isclose(L.window_count(eigen, np.array([1.0]), 0.0, 2.0, 3.0), 0.0)
