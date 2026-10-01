"""DFT relaxation: specification, refusals, worker loop and output validation."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import relaxation as relax
from materia.experiments.dft import spec as specs
from materia.provenance import Origin
from materia.solvers.gpaw_driver import runner
from tests.support.fake_gpaw_relax import FakeRelaxWorker
from tests.support.fake_gpaw_relax import environment as fake_environment

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def environment(tmp_path):
    return fake_environment(tmp_path)


def molecule(distance=0.8, box=8.0):
    c = box / 2
    return Structure(np.array([1, 1]),
                     np.array([[c, c, c - distance / 2], [c, c, c + distance / 2]]),
                     Cell(np.eye(3) * box, (False, False, False)))


def silicon(a=5.6):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


def bulk_variable(environment, **extra):
    variables = {"mode": "variable-cell", "cutoff_eV": 500.0, "kpoints": [2, 2, 2],
                 "occupations": "fixed", "smearing_eV": 0.0, "forces_tol_eV_A": 0.005}
    variables.update(extra)
    return relax.build(silicon(), environment, "s1", **variables)


def worker_module():
    path = ROOT / "materia" / "solvers" / "gpaw_driver" / "ground_state_worker.py"
    spec = importlib.util.spec_from_file_location("materia_gs_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSpecification:
    def test_frozen_versioned_and_round_trips(self, environment):
        spec = relax.build(molecule(), environment, "s1", forces_tol_eV_A=0.01)
        assert spec.schema == relax.SCHEMA and spec.version == relax.VERSION
        with pytest.raises(FrozenInstanceError):
            spec.fmax_eV_A = 1.0
        again = relax.RelaxationSpec.from_dict(json.loads(json.dumps(spec.as_dict())))
        assert again == spec and again.digest == spec.digest
        assert len(spec.digest) == 64
        other = relax.changed(spec, environment, fmax_eV_A=0.02)
        assert other.digest != spec.digest and other.fmax_eV_A == 0.02
        electronic = relax.changed(spec, environment, grid_spacing_A=0.18)
        assert electronic.ground_state.grid_spacing_A == 0.18
        with pytest.raises(specs.SpecError, match="version"):
            relax.RelaxationSpec.from_dict({**spec.as_dict(), "version": "9"})
        with pytest.raises(specs.SpecError, match="Unknown relaxation field"):
            relax.RelaxationSpec.from_dict({**spec.as_dict(), "extra": 1})

    def test_defaults_depend_on_mode(self, environment):
        fixed = relax.build(molecule(), environment, "s1")
        assert fixed.mode == "fixed-cell" and fixed.stress_tol_eV_A3 is None
        assert fixed.cell_mask is None and fixed.target_pressure_GPa is None
        assert "forces" in fixed.ground_state.observables
        variable = bulk_variable(environment)
        assert variable.cell_mask == (True,) * 6 and variable.hydrostatic_strain is False
        assert variable.target_pressure_GPa == 0.0 and variable.stress_tol_eV_A3 == 0.005
        assert {"forces", "stress"} <= set(variable.ground_state.observables)
        switched = relax.changed(fixed, environment, mode="variable-cell")
        assert switched.stress_tol_eV_A3 == 0.005

    def test_fixed_atoms_and_moments_come_from_the_structure(self, environment):
        s = molecule()
        s.fixed[0] = True
        s.magnetic_moments[:] = [0.5, 0.5]
        spec = relax.build(s, environment, "s1", spin_polarized=True)
        assert spec.fixed_atom_ids == (int(s.ids[0]),)
        assert spec.fixed_indices() == [0]
        assert spec.symmetry == "off"
        assert relax.build(molecule(), environment, "s1").symmetry == "preserve"
        preserved = relax.build(s, environment, "s1", spin_polarized=True,
                                symmetry="preserve", forces_tol_eV_A=0.01)
        report = relax.check(preserved, environment)
        assert "symmetry" in report.by_field
        assert "broken by the first step" in report.by_field["symmetry"][0]
        assert spec.structure_magnetic_moments == (0.5, 0.5)
        with pytest.raises(specs.SpecError, match="comes from the structure"):
            relax.build(s, environment, "s1", fixed_atom_ids=[1])

    @pytest.mark.parametrize("name, value", [
        ("mode", "sideways"), ("optimizer", "MDMin"), ("symmetry", "some"),
        ("max_steps", 1.5), ("fmax_eV_A", "0.1"), ("cell_mask", "xx"),
        ("cell_mask", [True] * 5), ("hydrostatic_strain", 1),
    ])
    def test_mistyped_variables_raise(self, environment, name, value):
        with pytest.raises(specs.SpecError):
            relax.build(molecule(), environment, "s1", **{name: value})

    def test_every_variable_reaches_the_worker_job(self, environment):
        base = bulk_variable(environment)
        job = relax.worker_job(base)
        assert job["parameters"]["symmetry"] == {"point_group": True, "time_reversal": True}
        assert job["relaxation"] == {
            "mode": "variable-cell", "optimizer": "BFGS", "maxstep": 0.2, "fmax": 0.05,
            "stress_tol": 0.005, "max_steps": 100, "fixed_indices": [],
            "filter": {"name": "FrechetCellFilter", "mask": [True] * 6,
                       "hydrostatic_strain": False, "scalar_pressure": 0.0}}
        changes = {"optimizer": "FIRE", "maxstep_A": 0.1, "fmax_eV_A": 0.02,
                   "stress_tol_eV_A3": 0.001, "max_steps": 7,
                   "cell_mask": [True, True, True, False, False, False],
                   "hydrostatic_strain": True, "target_pressure_GPa": 16.021766208,
                   "symmetry": "off"}
        for name, value in changes.items():
            if name == "hydrostatic_strain":
                spec = relax.changed(base, environment, hydrostatic_strain=True)
            else:
                spec = relax.changed(base, environment, **{name: value})
            assert relax.worker_job(spec) != job, name
        pressured = relax.changed(base, environment, target_pressure_GPa=16.021766208)
        assert relax.worker_job(pressured)["relaxation"]["filter"]["scalar_pressure"] == \
            pytest.approx(0.1)
        off = relax.changed(base, environment, symmetry="off")
        assert relax.worker_job(off)["parameters"]["symmetry"] == {
            "point_group": False, "time_reversal": False}
        for name, value in {"xc": "LDA", "cutoff_eV": 600.0, "kpoints": [3, 3, 3],
                            "energy_tol_eV_per_electron": 1e-4, "forces_tol_eV_A": 0.001,
                            "max_scf_iterations": 50}.items():
            spec = relax.changed(base, environment, **{name: value})
            assert relax.worker_job(spec)["parameters"] != job["parameters"], name

    def test_valid_specifications_pass(self, environment):
        fixed = relax.build(molecule(), environment, "s1", forces_tol_eV_A=0.01)
        report = relax.check(fixed, environment)
        assert report.ok, report.blocking
        variable = bulk_variable(environment)
        assert relax.check(variable, environment).ok, relax.check(variable, environment).blocking


REFUSALS = [
    ("cluster variable cell", lambda env: relax.build(molecule(), env, "s1",
                                                      mode="variable-cell"), "mode"),
    ("fd variable cell", lambda env: relax.build(silicon(), env, "s1", mode="variable-cell",
                                                 representation="fd", kpoints=[2, 2, 2],
                                                 observables=["energy", "forces"]), "mode"),
    ("stress missing", lambda env: bulk_variable(env, observables=["energy", "forces"]),
     "observables"),
    ("forces missing", lambda env: relax.build(molecule(), env, "s1",
                                               observables=["energy"]), "observables"),
    ("stress tolerance at fixed cell", lambda env: relax.build(
        molecule(), env, "s1", stress_tol_eV_A3=0.01), "stress_tol_eV_A3"),
    ("pressure at fixed cell", lambda env: relax.build(
        molecule(), env, "s1", target_pressure_GPa=1.0), "target_pressure_GPa"),
    ("loose SCF forces", lambda env: relax.build(molecule(), env, "s1", fmax_eV_A=0.01,
                                                 forces_tol_eV_A=0.05), "forces_tol_eV_A"),
    ("fmax", lambda env: relax.build(molecule(), env, "s1", fmax_eV_A=2.0), "fmax_eV_A"),
    ("steps", lambda env: relax.build(molecule(), env, "s1", max_steps=0), "max_steps"),
    ("step length", lambda env: relax.build(molecule(), env, "s1", maxstep_A=1.0),
     "maxstep_A"),
    ("empty mask", lambda env: bulk_variable(env, cell_mask=[False] * 6), "cell_mask"),
    ("hydrostatic with mask", lambda env: bulk_variable(
        env, hydrostatic_strain=True, cell_mask=[True, True, True, False, False, False]),
     "hydrostatic_strain"),
    ("pressure range", lambda env: bulk_variable(env, target_pressure_GPa=500.0),
     "target_pressure_GPa"),
    ("stress range", lambda env: bulk_variable(env, stress_tol_eV_A3=1.0),
     "stress_tol_eV_A3"),
    ("charged cell", lambda env: bulk_variable(
        env, charge_e=1.0, charged_periodic_policy="uniform-background",
        spin_polarized=True), "charge_e"),
    ("field on a charged cluster", lambda env: relax.build(
        molecule(), env, "s1", charge_e=1.0, external_field_V_per_A=[0.0, 0.0, 0.1],
        spin_polarized=True), "external_field_V_per_A"),
]


@pytest.mark.parametrize("label, factory, variable", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_variable(environment, label, factory, variable):
    report = relax.check(factory(environment), environment)
    assert not report.ok
    assert variable in report.by_field, report.by_field


def test_fixed_atom_refusals(environment):
    s = molecule()
    s.fixed[:] = True
    report = relax.check(relax.build(s, environment, "s1"), environment)
    assert "fixed_atom_ids" in report.by_field
    bulk = silicon()
    bulk.fixed[0] = True
    report = relax.check(bulk_variable(environment).__class__(
        **{**bulk_variable(environment).__dict__,
           "fixed_atom_ids": (int(bulk.ids[0]),)}), environment)
    assert "fixed_atom_ids" in report.by_field
    assert "change of cell moves every atom" in report.by_field["fixed_atom_ids"][0]


def test_ground_state_refusals_are_kept(environment):
    report = relax.check(relax.build(molecule(), environment, "s1", kpoints=[2, 1, 1]),
                         environment)
    assert "kpoints" in report.by_field


def test_warnings(environment):
    report = relax.check(relax.build(molecule(), environment, "s1"), environment)
    assert any("No SCF force tolerance" in w for w in report.warnings)
    assert any("Symmetry preserved" in w for w in report.warnings)
    low = relax.check(bulk_variable(environment, cutoff_eV=400.0), environment)
    assert any("Pulay" in w for w in low.warnings)


def test_unavailable_gpaw_is_refused(tmp_path):
    from materia.solvers.gpaw_driver import GPAWEnvironment

    missing = GPAWEnvironment()
    spec = relax.build(molecule(), missing, "s1")
    report = relax.check(spec, missing)
    assert not report.ok
    with pytest.raises(specs.SpecRefused):
        relax.execute(spec, missing)


class TestWorkerLoop:
    """The worker's ionic loop, run in-process with ASE's EMT calculator."""

    def run(self, atoms, **settings):
        from ase.calculators.emt import EMT

        w = worker_module()
        atoms.calc = EMT()
        base = {"mode": "fixed-cell", "optimizer": "BFGS", "maxstep": 0.2, "fmax": 0.01,
                "stress_tol": None, "max_steps": 200, "fixed_indices": [], "filter": None}
        base.update(settings)
        history = w.History(0.0)
        result, arrays, events = {}, {}, []
        done = w.relax(atoms, base, history, result, arrays,
                       report=lambda kind, **p: events.append(p))
        return done, result, arrays, events

    def cu(self, rattle=0.05):
        from ase.build import bulk

        atoms = bulk("Cu", "fcc", a=3.7, cubic=True) * (2, 1, 1)
        atoms.rattle(rattle, seed=3)
        return atoms

    def test_fixed_cell_keeps_fixed_atoms_and_meets_the_criterion(self):
        atoms = self.cu()
        start = atoms.get_positions().copy()
        cell = atoms.cell.array.copy()
        done, result, arrays, events = self.run(atoms, fixed_indices=[0, 3])
        assert done
        final = np.asarray(result["relaxation"]["final_positions"])
        assert np.array_equal(final[[0, 3]], start[[0, 3]])
        assert np.array_equal(np.asarray(result["relaxation"]["final_cell"]), cell)
        steps = result["relaxation"]["steps"]
        assert steps[-1]["max_force_eV_A"] <= 0.01
        assert [row["step"] for row in steps] == list(range(len(steps)))
        assert result["relaxation_used"]["fixed_indices"] == [0, 3]
        assert arrays["relax_positions"].shape == (len(steps), 8, 3)
        assert len(events) == len(steps)

    def test_variable_cell_reaches_the_target_pressure(self):
        atoms = self.cu(0.0)
        pressure = 0.01
        done, result, _, _ = self.run(atoms, mode="variable-cell", stress_tol=5e-4,
                                      filter={"name": "FrechetCellFilter",
                                              "mask": [True] * 6,
                                              "hydrostatic_strain": False,
                                              "scalar_pressure": pressure})
        assert done
        stress = np.asarray(atoms.get_stress(voigt=False))
        assert np.allclose(np.diag(stress), -pressure, atol=5e-4)
        used = result["relaxation_used"]["filter"]
        assert used == {"name": "FrechetCellFilter", "mask": [True] * 6,
                        "hydrostatic_strain": False, "scalar_pressure": pressure}

    def test_mask_is_echoed_in_voigt_order(self):
        atoms = self.cu(0.0)
        mask = [True, True, False, False, False, False]
        _, result, _, _ = self.run(atoms, mode="variable-cell", stress_tol=1e-3,
                                   max_steps=3,
                                   filter={"name": "FrechetCellFilter", "mask": mask,
                                           "hydrostatic_strain": False,
                                           "scalar_pressure": 0.0})
        assert result["relaxation_used"]["filter"]["mask"] == mask

    def test_step_limit_is_not_convergence(self):
        done, result, _, _ = self.run(self.cu(0.2), max_steps=2, fmax=1e-6)
        assert not done
        assert result["relaxation"]["stop_reason"] == "max_steps"
        assert len(result["relaxation"]["steps"]) == 3

    def test_stress_residual(self):
        w = worker_module()
        stress = np.diag([-0.1, -0.1, -0.1])
        assert w.stress_residual(stress, 0.1, [True] * 6, False) == pytest.approx(0.0)
        shear = np.array([[0.0, 0.02, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert w.stress_residual(shear, 0.0, [True] * 6, False) == pytest.approx(0.02)
        assert w.stress_residual(shear, 0.0, [True] * 5 + [False], False) == 0.0
        assert w.stress_residual(np.diag([0.3, 0.0, 0.0]), 0.0, [True] * 6, True) == \
            pytest.approx(0.1)


class TestExecute:
    def spec(self, environment, **extra):
        s = molecule()
        s.fixed[0] = True
        variables = {"forces_tol_eV_A": 0.01}
        variables.update(extra)
        return relax.build(s, environment, "s1", **variables)

    def test_finished_run_builds_every_result(self, environment, monkeypatch):
        worker = FakeRelaxWorker()
        monkeypatch.setattr(runner, "run_job", worker)
        spec = self.spec(environment)
        outcome = relax.execute(spec, environment, run_id="r1")
        assert outcome.status == "converged", outcome.reason
        assert {"relaxation", "run", "energy", "forces", "charge_accounting",
                "scf_history"} <= set(outcome.results)
        assert {"initial_positions", "final_positions", "initial_cell", "final_cell",
                "final_forces", "step_positions", "step_cells"} <= set(outcome.arrays)
        summary = outcome.results["relaxation"].value
        assert summary["optimizer_steps"] == 4
        assert summary["displacement"]["max_A"] == pytest.approx(0.05)
        assert summary["displacement"]["by_atom_id"][int(spec.ground_state.atom_ids[0])] == \
            [0.0, 0.0, 0.0]
        assert summary["cell"]["volume_change_percent"] == 0.0
        assert summary["echo"]["optimizer"] == "BFGS"
        assert summary["history"][0]["step"] == 0
        energy = outcome.results["energy"]
        assert energy.provenance.model == relax.MODEL
        assert energy.provenance.inputs_digest == spec.digest
        assert energy.provenance.parameters["relaxation_spec"]["mode"] == "fixed-cell"
        assert energy.extra["geometry_digest"] == outcome.output_geometry_digest
        assert outcome.output_geometry_digest != spec.geometry_digest
        assert outcome.final_spec.positions_A != spec.ground_state.positions_A
        sent = worker.jobs[0]
        assert sent["relaxation"]["fixed_indices"] == [0]
        assert sent["parameters"]["symmetry"] == {"point_group": False, "time_reversal": False}

    def test_variable_cell_reports_the_cell_change(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeRelaxWorker(cell_scale=0.98))
        spec = bulk_variable(environment)
        outcome = relax.execute(spec, environment)
        assert outcome.ok, outcome.reason
        cell = outcome.results["relaxation"].value["cell"]
        assert cell["volume_change_percent"] == pytest.approx(100 * (0.98 ** 3 - 1))
        assert np.allclose(cell["strain"], -0.02 * np.eye(3), atol=1e-12)
        assert "stress" in outcome.results

    def test_step_limit_is_kept_as_an_estimate(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeRelaxWorker("max_steps"))
        outcome = relax.execute(self.spec(environment, max_steps=3), environment)
        assert outcome.status == "not-converged" and outcome.ok
        record = outcome.results["relaxation"]
        assert record.convergence.converged is False
        assert record.provenance.origin is Origin.ESTIMATED
        assert "cannot be applied" in record.convergence.message

    @pytest.mark.parametrize("mode, fragment", [
        ("failed", "boom"), ("timeout", "time limit"),
        ("echo_symmetry", "symmetry"), ("echo_optimizer", "optimiser"),
        ("lose_atom", "atoms were lost"), ("move_fixed", "fixed atom moved"),
        ("change_cell", "cell changed"), ("nan_energy", "non-finite"),
        ("wrong_verdict", "verdict"), ("no_forces", "forces"),
    ])
    def test_bad_output_yields_nothing(self, environment, monkeypatch, mode, fragment):
        monkeypatch.setattr(runner, "run_job", FakeRelaxWorker(mode))
        outcome = relax.execute(self.spec(environment), environment, timeout_s=1.0)
        assert outcome.status == "failed"
        assert fragment in outcome.reason
        assert outcome.results == {} and outcome.arrays == {}
        assert "Nothing was kept" in outcome.reason

    def test_cancellation_yields_nothing(self, environment, monkeypatch):
        monkeypatch.setattr(runner, "run_job", FakeRelaxWorker("cancelled"))
        outcome = relax.execute(self.spec(environment), environment)
        assert outcome.status == "cancelled" and not outcome.results
        before = relax.execute(self.spec(environment), environment, cancelled=lambda: True)
        assert before.status == "cancelled" and not before.results

    def test_refused_specification_starts_nothing(self, environment, monkeypatch):
        worker = FakeRelaxWorker()
        monkeypatch.setattr(runner, "run_job", worker)
        with pytest.raises(specs.SpecRefused):
            relax.execute(self.spec(environment, fmax_eV_A=5.0), environment)
        assert worker.jobs == []
