"""Solver registry, capabilities and honest refusal."""

from __future__ import annotations

import numpy as np
import pytest

from materia.materials import load
from materia.provenance import Origin
from materia.solvers import Capability, available, create, describe_all, solvers_providing
from materia.solvers.classical import stillinger_weber
from materia.solvers.external import ADAPTERS, ExternalSolver, availability_report
from materia.solvers.tight_binding import TightBinding
from materia.structure_builder import bulk, make_surface


def test_registry_exposes_the_built_in_solvers():
    names = available()
    assert "stillinger-weber" in names
    assert "tight-binding/sp3s*-Si" in names
    assert any(n.startswith("external:") for n in names)


def test_every_registered_solver_declares_its_capabilities():
    for description in describe_all():
        assert "capabilities" in description
        assert "missing_capabilities" in description
        assert description["fidelity"]


def test_capability_lookup_finds_providers():
    providers = solvers_providing(Capability.LOCAL_DOS)
    assert any("tight-binding" in p for p in providers)
    assert not any(p.startswith("classical/") for p in providers)


def test_classical_solver_refuses_electronic_questions():
    solver = stillinger_weber("Si")
    refusal = solver.require(Capability.LOCAL_DOS)
    assert refusal is not None
    assert refusal.provenance.origin is Origin.UNSUPPORTED
    assert refusal.value is None
    assert refusal.suggested_models


def test_classical_solver_energy_and_forces_carry_provenance(si111_small):
    out = stillinger_weber("Si").single_point(si111_small)
    energy = out["energy"]
    assert energy.provenance.origin is Origin.CALCULATED
    assert energy.provenance.fidelity.value == "tier1-classical"
    assert any("no electrons" in a.lower() or "no electronic" in a.lower()
               for a in energy.provenance.approximations)
    assert energy.unit == "eV"


def test_relaxation_reports_convergence_honestly(si111_small):
    slab = si111_small.copy()
    rng = np.random.default_rng(2)
    slab.positions = slab.positions + rng.normal(0, 0.1, slab.positions.shape)
    solver = stillinger_weber("Si")
    out = solver.relax(slab, fmax_eV_A=0.02, max_steps=400)
    assert out.convergence.converged
    assert out.convergence.residual < 0.02
    assert out["energy"].value < out["energy"].extra["initial_energy_eV"]

    truncated = solver.relax(slab, fmax_eV_A=1e-9, max_steps=3)
    assert not truncated.convergence.converged
    assert "Not converged" in truncated.convergence.message


def test_relaxation_holds_fixed_atoms(si111_small):
    slab = si111_small.copy()
    slab.fixed[:] = False
    slab.fixed[0] = True
    before = slab.positions[0].copy()
    rng = np.random.default_rng(4)
    slab.positions = slab.positions + rng.normal(0, 0.05, slab.positions.shape)
    slab.positions[0] = before
    stillinger_weber("Si").relax(slab, fmax_eV_A=0.05, max_steps=50, in_place=True)
    assert slab.positions[0] == pytest.approx(before)


def test_microcanonical_dynamics_conserves_energy():
    silicon = bulk(load("silicon"), (2, 2, 2))
    out = stillinger_weber("Si").dynamics(
        silicon, steps=200, dt_fs=1.0, temperature_K=300, thermostat="none", seed=1)
    trajectory = out["trajectory"].value
    drift = abs(trajectory["total_eV"][-1] - trajectory["total_eV"][0])
    assert drift / len(silicon) < 1e-3


def test_dynamics_is_reproducible_from_its_seed():
    silicon = bulk(load("silicon"), (2, 2, 2))
    solver = stillinger_weber("Si")
    first = solver.dynamics(silicon, steps=40, temperature_K=300, seed=7)
    second = solver.dynamics(silicon, steps=40, temperature_K=300, seed=7)
    assert first["trajectory"].value["total_eV"] == \
        pytest.approx(second["trajectory"].value["total_eV"])
    different = solver.dynamics(silicon, steps=40, temperature_K=300, seed=8)
    assert different["trajectory"].value["total_eV"][-1] != \
        pytest.approx(first["trajectory"].value["total_eV"][-1])


def test_langevin_thermostat_reaches_the_target_temperature():
    silicon = bulk(load("silicon"), (2, 2, 2))
    out = stillinger_weber("Si").dynamics(
        silicon, steps=600, dt_fs=1.0, temperature_K=300, thermostat="langevin",
        friction_per_fs=0.02, seed=3)
    mean = out["mean_temperature"]
    assert mean.value == pytest.approx(300, rel=0.25)
    assert mean.uncertainty is not None


def test_unknown_thermostat_is_rejected_by_name():
    silicon = bulk(load("silicon"))
    with pytest.raises(ValueError) as excinfo:
        stillinger_weber("Si").dynamics(silicon, thermostat="nose-hoover",
                                        temperature_K=300)
    assert "not implemented" in str(excinfo.value)


def test_tight_binding_refuses_unparameterised_species():
    gold = make_surface(load("gold"), (1, 1, 1), size=(1, 1, 2))
    report = TightBinding("sp3s*-Si").supports(gold)
    assert not report.ok
    assert "Au" in report.blocking[0]


def test_tight_binding_refuses_charged_systems():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2))
    slab.formal_charges[0] = 1.0
    report = TightBinding("sp3s*-Si").supports(slab)
    assert not report.ok
    assert "Net charge" in report.blocking[0]


def test_self_consistency_request_is_declined_with_alternatives(si111_small):
    out = TightBinding("sp3s*-Si").eigenstates(si111_small, self_consistent=True)
    refusal = out["self_consistency"]
    assert not refusal.supported
    assert "non-self-consistent" in refusal.unsupported_reason
    assert any("gpaw" in m for m in refusal.suggested_models)


def test_eigenstates_carry_the_expected_diagnostics(si111_small):
    out = TightBinding("sp3s*-Si").eigenstates(si111_small)
    assert out["eigenvalues"].value.size == len(si111_small) * 5
    assert out.convergence.converged
    assert out["fermi_level"].unit == "eV"
    assert out["local_dos"].value["ldos"].shape[1] == len(si111_small)
    assert any("NOT self-consistent" in a for a in out["eigenvalues"].provenance.approximations)


def test_external_adapters_are_registered_even_when_absent():
    report = availability_report()
    assert {r["name"] for r in report} == {a.name for a in ADAPTERS} | {"external:gpaw"}
    for row in report:
        assert row["install_hint"]
        assert row["license_note"]


def test_only_driven_adapters_claim_to_be_driven():
    """An adapter that can only be run by handing it a calculator object, or not
    at all, must not present itself as a working solver."""
    driven = {r["name"] for r in availability_report() if r.get("driven")}
    assert driven == {"external:gpaw", "external:ase", "external:lammps"}


def test_missing_external_solver_preserves_the_request():
    spec = next(a for a in ADAPTERS if a.name == "external:quantum-espresso")
    solver = ExternalSolver(spec)
    if solver.installed:
        pytest.skip("Quantum ESPRESSO is installed in this environment")
    out = solver.run(make_surface(load("silicon"), (1, 1, 1), size=(1, 1, 2)),
                     task="energy", cutoff=400)
    refusal = out["energy"]
    assert not refusal.supported
    assert "pw.x" in refusal.unsupported_reason
    assert refusal.extra["preserved_request"]["task"] == "energy"
    assert refusal.extra["preserved_request"]["kwargs"]["cutoff"] == "400"


def test_unknown_solver_name_lists_the_available_ones():
    with pytest.raises(KeyError) as excinfo:
        create("does-not-exist")
    assert "stillinger-weber" in str(excinfo.value)
