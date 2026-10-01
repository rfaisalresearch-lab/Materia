"""Band structure: specification, path geometry, refusals and output validation."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import bands as B
from materia.experiments.dft import spec as specs
from materia.provenance import Origin
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_bands import FakeBandsGPAW
from tests.support.fake_gpaw_dos import FERMI_EV
from tests.support.fake_gpaw_dos import environment as fake_environment


@pytest.fixture
def environment(tmp_path):
    return fake_environment(tmp_path)


def silicon(a=5.43):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


def molecule(box=8.0):
    c = box / 2
    return Structure(np.array([1, 1]), np.array([[c, c, c - 0.37], [c, c, c + 0.37]]),
                     Cell(np.eye(3) * box, (False, False, False)))


def graphene_like(vacuum=12.0):
    a = 2.46
    cell = np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, vacuum]])
    positions = np.array([[0, 0, vacuum / 2], [0, a / math.sqrt(3), vacuum / 2]])
    return Structure(np.array([1, 1]), positions, Cell(cell, (True, True, False)))


def chain(length=2.5, box=10.0):
    cell = np.diag([box, box, length])
    return Structure(np.array([1, 1]), np.array([[box / 2, box / 2, 0.2],
                                                 [box / 2, box / 2, 1.2]]),
                     Cell(cell, (False, False, True)))


BULK = {"occupations": "fixed", "smearing_eV": 0.0, "kpoints": [2, 2, 2]}


def bulk(environment, **extra):
    return B.build(silicon(), environment, "s1", **{**BULK, **extra})


class TestPathGeometry:
    def test_points_labels_and_breaks(self):
        points = {"G": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.5), "L": (0.5, 0.5, 0.5)}
        kpts, labels, breaks = B.path_kpoints((("G", "X"), ("L", "G")), points, (4, 2))
        assert kpts.shape == (8, 3)
        assert labels == ["G", "", "", "", "X", "L", "", "G"]
        assert breaks == [5]
        assert np.allclose(kpts[2], [0.25, 0.0, 0.25])

    def test_distance_uses_two_pi_and_skips_breaks(self):
        cell = np.eye(3) * 2.0
        kpts = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0.5, 0.5]])
        x = B.path_distance(kpts, cell, [2])
        assert x == pytest.approx([0.0, math.pi / 2, math.pi / 2, math.pi])

    def test_parse_path(self):
        assert B.parse_path("GXW,UX") == (("G", "X", "W"), ("U", "X"))
        assert B.parse_path("GY1S0") == (("G", "Y1", "S0"),)
        assert B.parse_path([["G", "X"]]) == (("G", "X"),)
        for bad in ("gx", "", "G-X", 7, [["G", 1]]):
            with pytest.raises(specs.SpecError):
                B.parse_path(bad)


class TestSpecification:
    def test_frozen_versioned_and_round_trips(self, environment):
        spec = bulk(environment)
        assert spec.schema == B.SCHEMA and spec.version == B.VERSION
        with pytest.raises(FrozenInstanceError):
            spec.n_bands = 3
        again = B.BandStructureSpec.from_dict(json.loads(json.dumps(spec.as_dict())))
        assert again == spec and again.digest == spec.digest and len(spec.digest) == 64
        assert B.changed(spec, environment, n_bands=12).digest != spec.digest
        with pytest.raises(specs.SpecError, match="version"):
            B.BandStructureSpec.from_dict({**spec.as_dict(), "version": "2.0"})
        with pytest.raises(specs.SpecError, match="Unknown band-structure field"):
            B.BandStructureSpec.from_dict({**spec.as_dict(), "colour": "blue"})
        with pytest.raises(specs.SpecError, match="Missing"):
            data = spec.as_dict()
            data.pop("kpoints_frac")
            B.BandStructureSpec.from_dict(data)

    def test_standard_fcc_path_is_generated_once_and_stored(self, environment):
        spec = bulk(environment)
        assert B.path_text(spec.path) == "GXWKGLUWLK,UX"
        assert spec.path_origin["generator"] == "ase" and spec.path_origin["lattice"] == "FCC"
        assert spec.path_origin["ase_version"]
        assert dict(spec.special_points)["X"] == (0.5, 0.0, 0.5)
        assert spec.n_kpoints == len(spec.kpoints_frac) == 76
        assert spec.kpoints_frac[0] == (0.0, 0.0, 0.0)
        assert spec.breaks() == [71]
        ticks = [t["label"] for t in spec.ticks()]
        assert ticks == ["G", "X", "W", "K", "G", "L", "U", "W", "L", "K|U", "X"]
        assert spec.source["kind"] == "structure"
        assert spec.ground_state.observables == ("energy", "fermi_level")
        assert spec.scf_symmetry == "preserve" and spec.energy_reference == "fermi-level"
        assert spec.n_bands == 8 and spec.extra_bands == 4

    def test_changes_keep_the_stored_path(self, environment, monkeypatch):
        spec = bulk(environment)
        calls = []
        original = B.standard_path
        monkeypatch.setattr(B, "standard_path", lambda *a: calls.append(a) or original(*a))
        other = B.changed(spec, environment, n_bands=10, cutoff_eV=500.0, scf_symmetry="off")
        assert calls == []
        assert other.kpoints_frac == spec.kpoints_frac and other.path == spec.path
        assert other.special_points == spec.special_points
        B.changed(spec, environment, path="standard")
        assert len(calls) == 1

    def test_explicit_path_and_intervals(self, environment):
        spec = bulk(environment, path="GX,LG", segment_intervals=[3, 2])
        assert spec.n_kpoints == 7 and spec.sampling_density_per_invA is None
        assert spec.labels() == ["G", "", "", "X", "L", "", "G"]
        assert spec.path_origin["path_chosen"] == "GX,LG"
        assert B.check(spec, environment).ok
        kept = B.changed(spec, environment, n_bands=9)
        assert kept.segment_intervals == (3, 2)
        denser = B.changed(spec, environment, sampling_density_per_invA=20.0)
        assert sum(denser.segment_intervals) > 5
        custom = bulk(environment, special_points={"G": [0, 0, 0], "A": [0.25, 0.25, 0.25]},
                      path="GA")
        assert custom.path_origin["generator"] == "user" and custom.n_kpoints > 2
        assert B.check(custom, environment).ok

    def test_every_variable_reaches_the_job(self, environment):
        base = bulk(environment)
        job = B.worker_job(base)
        assert job["parameters"]["symmetry"] == {"point_group": True, "time_reversal": True}
        assert job["bands"]["nscf"]["symmetry"] == "off"
        assert job["bands"]["nscf"]["nbands"] == 12
        assert job["bands"]["nscf"]["convergence"] == {"bands": 8}
        assert job["bands"]["nscf"]["kpts"][12] == [0.5, 0.0, 0.5]
        assert job["bands"]["labels"][12] == "X" and job["bands"]["breaks"] == [71]
        changes = {"n_bands": 10, "extra_bands": 6, "sampling_density_per_invA": 5.0,
                   "path": "GXL", "segment_intervals": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]}
        for name, value in changes.items():
            spec = B.changed(base, environment, **{name: value})
            assert B.worker_job(spec)["bands"] != job["bands"], name
        off = B.changed(base, environment, scf_symmetry="off")
        assert B.worker_job(off)["parameters"]["symmetry"]["point_group"] is False
        electronic = B.changed(base, environment, cutoff_eV=500.0)
        assert B.worker_job(electronic)["parameters"] != job["parameters"]

    @pytest.mark.parametrize("name, value", [
        ("energy_reference", "vacuum"), ("scf_symmetry", "partial"), ("path", "gx"),
        ("special_points", {"G": [0, 0]}), ("special_points", "G"),
        ("sampling_density_per_invA", "dense"), ("segment_intervals", 5),
        ("segment_intervals", [1.5]), ("n_bands", 1.5), ("extra_bands", True),
        ("kpoints_frac", [[0, 0, 0]]), ("path_origin", {}), ("source", {}),
    ])
    def test_mistyped_variables_raise(self, environment, name, value):
        with pytest.raises(specs.SpecError):
            bulk(environment, **{name: value})


class TestBoundaries:
    def test_slab_path_is_in_plane(self, environment):
        spec = B.build(graphene_like(), environment, "s1", kpoints=[4, 4, 1])
        assert spec.path_origin["lattice"] == "HEX2D"
        assert B.path_text(spec.path) == "GMKG"
        assert all(abs(k[2]) < 1e-12 for k in spec.kpoints_frac)
        report = B.check(spec, environment)
        assert "path" not in report.by_field and "special_points" not in report.by_field

    def test_wire_path_runs_along_the_axis(self, environment):
        spec = B.build(chain(), environment, "s1", kpoints=[1, 1, 6])
        assert B.path_text(spec.path) == "GX"
        assert dict(spec.special_points)["X"] == (0.0, 0.0, 0.5)
        report = B.check(spec, environment)
        assert "path" not in report.by_field and "special_points" not in report.by_field

    def test_out_of_plane_point_on_a_slab_is_refused(self, environment):
        spec = B.build(graphene_like(), environment, "s1", kpoints=[4, 4, 1],
                       special_points={"G": [0, 0, 0], "Z": [0, 0, 0.5]}, path="GZ")
        assert "special_points" in B.check(spec, environment).by_field

    def test_cluster_is_refused(self, environment, monkeypatch):
        spec = B.build(molecule(), environment, "s1")
        report = B.check(spec, environment)
        assert not report.ok and "no Brillouin zone" in " ".join(report.by_field["path"])
        worker = FakeBandsGPAW()
        monkeypatch.setattr(runner, "run_job", worker)
        with pytest.raises(specs.SpecRefused):
            B.execute(spec, environment)
        assert worker.jobs == []


def _tampered(spec, **fields):
    return B.BandStructureSpec.from_dict({**spec.as_dict(), **fields})


REFUSALS = [
    ("absolute energies", lambda env: bulk(env, energy_reference="absolute"),
     "energy_reference"),
    ("unknown label", lambda env: bulk(env, path="GQ"), "path"),
    ("one-point branch", lambda env: bulk(env, path="GX,L", segment_intervals=[3]), "path"),
    ("zero-length segment", lambda env: bulk(env, special_points={
        "G": [0, 0, 0], "A": [0, 0, 0]}, path="GA", segment_intervals=[2]), "path"),
    ("fraction out of range", lambda env: bulk(env, special_points={
        "G": [0, 0, 0], "A": [1.5, 0, 0]}, path="GA"), "special_points"),
    ("bad label", lambda env: bulk(env, special_points={
        "G": [0, 0, 0], "Abcdefghij": [0.5, 0, 0]}, path=[["G", "Abcdefghij"]]),
     "special_points"),
    ("interval count", lambda env: bulk(env, path="GXL", segment_intervals=[3]),
     "segment_intervals"),
    ("zero intervals", lambda env: bulk(env, path="GX", segment_intervals=[0]),
     "segment_intervals"),
    ("too many points", lambda env: bulk(env, sampling_density_per_invA=190.0),
     "segment_intervals"),
    ("density range", lambda env: bulk(env, sampling_density_per_invA=0.1),
     "sampling_density_per_invA"),
    ("too few bands", lambda env: bulk(env, n_bands=2), "n_bands"),
    ("buffer range", lambda env: bulk(env, extra_bands=500), "extra_bands"),
    ("reordered k-points", lambda env: _tampered(
        bulk(env), kpoints_frac=list(reversed(bulk(env).as_dict()["kpoints_frac"]))),
     "kpoints_frac"),
    ("moved k-point", lambda env: _tampered(
        bulk(env), kpoints_frac=[[0.01, 0, 0]] + bulk(env).as_dict()["kpoints_frac"][1:]),
     "kpoints_frac"),
    ("path for another cell", lambda env: _tampered(
        bulk(env), path_origin={**bulk(env).path_origin, "cell_A": np.eye(3).tolist()}),
     "path_origin"),
    ("source electronic settings", lambda env: _tampered(
        bulk(env), source={**bulk(env).source, "electronic_digest": "0" * 64}), "source"),
]


@pytest.mark.parametrize("label, factory, variable", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_variable(environment, label, factory, variable):
    report = B.check(factory(environment), environment)
    assert not report.ok
    assert variable in report.by_field, report.by_field


def test_ground_state_refusals_are_kept(environment):
    assert "kpoints" in B.check(bulk(environment, kpoints=[0, 2, 2]), environment).by_field


def test_unavailable_gpaw_is_refused():
    from materia.solvers.gpaw_driver import GPAWEnvironment

    missing = GPAWEnvironment()
    spec = B.build(silicon(), missing, "s1", **BULK)
    assert not B.check(spec, missing).ok
    with pytest.raises(specs.SpecRefused):
        B.execute(spec, missing)


def test_missing_ase_is_an_explicit_refusal(environment, monkeypatch):
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "ase" or name.startswith("ase."):
            raise ImportError("no ase")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    spec = bulk(environment)
    assert spec.path == () and spec.path_origin["generator"] == "none"
    report = B.check(spec, environment)
    assert "ASE is not installed" in " ".join(report.by_field["path"])


class TestBandEdges:
    def test_indirect_gap_and_metal(self):
        eig = np.array([[[-2.0, 1.0], [-1.0, 2.0], [-1.5, 0.5]]])
        labels = ["G", "", "X"]
        kpts = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
        edges = B.band_edges(eig, 0.0, labels, kpts)
        assert edges["gap_eV"] == pytest.approx(1.5) and edges["gap_kind"] == "indirect"
        assert edges["vbm"]["index"] == 1 and edges["cbm"]["index"] == 2
        assert edges["cbm"]["label"] == "X" and edges["vbm"]["label"] is None
        assert edges["per_spin"][0]["direct_gap_eV"] == pytest.approx(2.0)
        assert edges["per_spin"][0]["direct_gap_at"]["index"] == 2
        direct = B.band_edges(eig[:, :2], 0.0, labels[:2], kpts[:2])
        assert direct["gap_kind"] == "indirect" and direct["gap_eV"] == pytest.approx(2.0)
        metal = B.band_edges(eig, 0.7, labels, kpts)
        assert metal["metallic_on_path"] and "gap_eV" not in metal
        assert metal["per_spin"][0]["crossing_bands"] == [1]


class TestExecute:
    def test_complete_run_builds_every_result(self, environment, monkeypatch):
        worker = FakeBandsGPAW()
        monkeypatch.setattr(runner, "run_job", worker)
        spec = bulk(environment)
        outcome = B.execute(spec, environment, run_id="b1")
        assert outcome.status == "complete", outcome.reason
        assert {"bands", "fermi_level", "energy", "charge_accounting", "scf_history",
                "run"} <= set(outcome.results)
        assert set(outcome.arrays) == {"eigenvalues", "kpoints_frac", "kpoints_cartesian",
                                       "distance"}
        assert outcome.arrays["eigenvalues"].data.shape == (1, 76, 8)
        assert outcome.arrays["distance"].unit == "1/A"
        summary = outcome.results["bands"].value
        assert summary["fermi_level_eV"] == FERMI_EV
        assert summary["band_edges"]["gap_kind"] == "indirect"
        assert summary["checks"]["distance_max_relative_error"] < 1e-12
        assert summary["array_sha256"]["eigenvalues"] == outcome.arrays["eigenvalues"].sha256()
        assert summary["experimental_comparison"] is None
        assert summary["started_unix"] <= summary["finished_unix"]
        record = outcome.results["bands"]
        assert record.provenance.model == B.MODEL
        assert record.provenance.origin is Origin.CALCULATED
        assert record.provenance.inputs_digest == spec.digest
        assert B.PATH_CITATIONS[0] in record.provenance.references
        assert record.provenance.parameters["path_origin"]["lattice"] == "FCC"
        assert record.provenance.parameters["gpaw_parameters"]["symmetry"]["point_group"]
        assert worker.jobs[0]["observables"] == ["energy", "fermi_level"]

    def test_spin_polarised_bulk_returns_both_channels(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeBandsGPAW())
        spec = bulk(environment, spin_polarized=True, initial_magnetic_moments_muB=[1.0, 1.0],
                    occupations="fermi-dirac", smearing_eV=0.1)
        outcome = B.execute(spec, environment)
        assert outcome.ok, outcome.reason
        assert outcome.arrays["eigenvalues"].data.shape[0] == 2
        assert len(outcome.results["bands"].value["band_edges"]["per_spin"]) == 2

    @pytest.mark.parametrize("mode, fragment", [
        ("failed", "boom"), ("timeout", "time limit"), ("not_converged", "did not converge"),
        ("nscf_unconverged", "every requested band"), ("echo_nscf", "non-self-consistent"),
        ("echo_kpts", "nscf.kpts"), ("echo_scf", "ground state with the parameters"),
        ("echo_labels", "different path"), ("wrong_order", "path's order"),
        ("folded", "path's order"), ("symmetry", "symmetry operations"),
        ("missing_bands", "band is missing"), ("missing_kpoint", "k-points"),
        ("spin_mismatch", "spin channel"), ("flat_shape", "dimensions"),
        ("nan", "non-finite"), ("inf", "non-finite"), ("unsorted", "ascending"),
        ("bad_distance", "cumulative path distance"), ("bad_recip", "reciprocal cell"),
        ("fermi_moved", "Fermi level"), ("no_fermi", "Fermi level"),
        ("no_bands", "no band structure"),
    ])
    def test_bad_output_yields_nothing(self, environment, monkeypatch, mode, fragment):
        monkeypatch.setattr(runner, "run_job", FakeBandsGPAW(mode))
        outcome = B.execute(bulk(environment), environment, timeout_s=1.0)
        assert outcome.status == "failed"
        assert fragment in outcome.reason, outcome.reason
        assert outcome.results == {} and outcome.arrays == {}

    def test_cancellation_yields_nothing(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeBandsGPAW("cancelled"))
        assert B.execute(bulk(environment), environment).status == "cancelled"
        early = B.execute(bulk(environment), environment, cancelled=lambda: True)
        assert early.status == "cancelled" and not early.results and not early.arrays

    def test_refused_specification_starts_nothing(self, environment, monkeypatch):
        worker = FakeBandsGPAW()
        monkeypatch.setattr(runner, "run_job", worker)
        with pytest.raises(specs.SpecRefused):
            B.execute(bulk(environment, energy_reference="absolute"), environment)
        assert worker.jobs == []
