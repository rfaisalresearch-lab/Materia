"""A GPAW job belongs to the project it was submitted from.

A first-principles calculation can run for minutes.  Anything the user does in
the meantime -- starting a new project, opening one, accepting a crash recovery
-- must not cause the result to land somewhere it does not belong, and must not
cause forces computed for one geometry to be attached to another.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from materia.desktop_ui.service import Service


def slab(service):
    service.build_surface("silicon", [1, 1, 1], size=[1, 1, 2], vacuum_A=8.0)
    return service.structure


SLOW = {"preset": "surface", "xc": "LDA", "grid_spacing_A": 0.16,
        "kpoints": [2, 2, 1], "energy_tol_eV_per_electron": 1e-10,
        "max_iterations": 400}

FAST = {"preset": "surface", "xc": "LDA", "grid_spacing_A": 0.25,
        "kpoints": [1, 1, 1], "energy_tol_eV_per_electron": 5e-3,
        "max_iterations": 60}


def wait_for(service, job_id, timeout_s=180.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = service.job_status(job_id, include_result=True)
        if status["status"] in ("done", "cancelled", "failed"):
            return status
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s} s")


def _stretched(structure):
    from materia.core_model.cell import Cell

    matrix = np.array(structure.cell.matrix, dtype=float)
    matrix[2] = matrix[2] * 1.05
    return Cell(matrix, tuple(structure.cell.pbc))


def _opened(structure):
    from materia.core_model.cell import Cell

    return Cell(np.array(structure.cell.matrix, dtype=float), (True, False, False))


def gpaw_result_keys(project):
    return sorted(k for k in project.results if k.startswith("gpaw"))


class TestOwnership:

    def test_a_running_job_does_not_write_into_a_new_project(self, needs_gpaw):
        service = Service()
        slab(service)
        original = service.project
        submitted = service.gpaw_energy(background=True, **SLOW)
        job_id = submitted["job"]["id"]

        deadline = time.time() + 60
        while time.time() < deadline:
            if service.job_status(job_id, include_result=False)["status"] == "running":
                break
            time.sleep(0.1)

        service.new_project("Replacement")
        replacement = service.project
        assert replacement is not original

        wait_for(service, job_id)
        assert gpaw_result_keys(replacement) == []

    def test_opening_a_project_while_a_job_runs_leaves_it_clean(self, needs_gpaw, tmp_path):
        service = Service()
        slab(service)
        path = str(tmp_path / "other.materia")
        service.save_project(path)
        service.new_project("scratch")
        slab(service)

        submitted = service.gpaw_energy(background=True, **SLOW)
        job_id = submitted["job"]["id"]
        time.sleep(3)
        service.open_project(path)
        opened = service.project
        wait_for(service, job_id)
        assert gpaw_result_keys(opened) == []


    def test_recovering_a_project_while_a_job_runs_leaves_it_clean(self, needs_gpaw):
        service = Service()
        slab(service)
        service._autosave()
        submitted = service.gpaw_energy(background=True, **SLOW)
        job_id = submitted["job"]["id"]
        time.sleep(3)
        service.recover_project()
        recovered = service.project
        wait_for(service, job_id)
        assert gpaw_result_keys(recovered) == []

    def test_a_replaced_project_cancels_its_jobs_and_says_so(self, needs_gpaw):
        service = Service()
        slab(service)
        original = service.project
        submitted = service.gpaw_energy(background=True, **SLOW)
        job_id = submitted["job"]["id"]
        time.sleep(3)
        service.new_project("Replacement")

        status = wait_for(service, job_id)
        assert status["status"] == "cancelled"
        assert "project was replaced" in status["cancel_reason"]
        assert any("cancelled because the project was replaced" in w["text"].lower()
                   for w in service.warnings)
        assert gpaw_result_keys(service.project) == []
        owned = gpaw_result_keys(original)
        assert owned, "the owning project keeps the record of its own cancelled run"
        assert all(not original.results[k].supported for k in owned if k.endswith("energy"))


class TestInputDigest:

    def test_an_unchanged_structure_accepts_the_forces(self, needs_gpaw):
        service = Service()
        structure = slab(service)
        out = service.gpaw_energy(background=False, apply_forces=True, **FAST)
        assert out["ok"] is True
        assert out["forces_applied"] is True
        assert np.isfinite(structure.forces).all()

    @pytest.mark.parametrize("mutate,label", [
        (lambda s: s.__setattr__("positions", s.positions + 0.1), "positions"),
        (lambda s: s.set_position(int(s.ids[0]), s.positions[0] + [0.05, 0, 0]), "one atom"),
        (lambda s: s.substitute(int(s.ids[0]), "Ge"), "atomic number"),
        (lambda s: s.magnetic_moments.__setitem__(0, 1.0), "magnetic moment"),
        (lambda s: s.fixed.__setitem__(0, True), "fixed flag"),
        (lambda s: s.formal_charges.__setitem__(0, 1.0), "formal charge"),
        (lambda s: s.__setattr__("cell", _stretched(s)), "cell"),
        (lambda s: s.__setattr__("cell", _opened(s)), "periodicity"),
        (lambda s: s.ids.__setitem__(slice(None), list(reversed(list(s.ids)))), "ordering"),
    ])
    def test_a_changed_structure_keeps_the_result_but_refuses_the_forces(
            self, needs_gpaw, mutate, label):
        service = Service()
        structure = slab(service)
        settings = service.gpaw_settings(**FAST)["settings"]

        from materia.desktop_ui.service import _NullJob

        submission = service._gpaw_submission(structure, settings, apply_forces=True)
        mutate(structure)
        result = service._run_gpaw(submission, _NullJob())

        assert result["ok"] is True
        assert result.get("forces_applied") is not True
        assert "forces_refused" in result
        assert label.split()[0] in result["forces_refused"] or result["forces_refused"]
        assert not np.isfinite(structure.forces).all()


    def test_synchronous_runs_follow_the_same_rules(self, needs_gpaw):
        service = Service()
        structure = slab(service)
        out = service.gpaw_energy(background=False, apply_forces=True, **FAST)
        assert out["run_id"]
        assert out["inputs_digest"]
        keys = gpaw_result_keys(service.project)
        assert all(out["run_id"] in k for k in keys)
        energy = service.project.results[f"gpaw::{out['run_id']}::energy"]
        assert energy.provenance.inputs_digest == out["inputs_digest"]
        assert energy.extra["run_id"] == out["run_id"]

    def test_the_digest_is_recorded_on_every_result_of_a_run(self, needs_gpaw):
        service = Service()
        slab(service)
        out = service.gpaw_energy(background=False, **FAST)
        for key in gpaw_result_keys(service.project):
            assert service.project.results[key].provenance.inputs_digest == out[
                "inputs_digest"]


class TestDistinctRuns:

    def test_two_runs_both_survive_with_distinct_digests(self, needs_gpaw):
        service = Service()
        slab(service)
        first = service.gpaw_energy(background=False, **FAST)
        second = service.gpaw_energy(background=False, **dict(FAST, xc="PBE"))
        assert first["ok"] and second["ok"]

        keys = gpaw_result_keys(service.project)
        assert len(keys) >= 4
        energies = [service.project.results[k] for k in keys if k.endswith("energy")]
        assert len(energies) == 2
        digests = {r.provenance.inputs_digest for r in energies}
        assert len(digests) == 2
        assert all(digests)
        assert first["run_id"] != second["run_id"]
