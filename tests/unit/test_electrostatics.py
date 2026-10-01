"""Point-charge electrostatics: implementation tests.

These check the software: charge accounting, refusal paths, invariances,
consistency of forces with the energy, determinism, cancellation and the
provenance record.  Agreement with published Madelung constants is physical
validation and lives in ``tests/validation/test_electrostatics_validation.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import electrostatics as es
from materia.physics.electrostatics import ChargeModel, EwaldSettings
from materia.provenance import Fidelity, Origin
from materia.solvers.base import Capability
from materia.solvers.electrostatics import ElectrostaticsSolver

NACL = ChargeModel.per_element({"Na": 1.0, "Cl": -1.0}, "unit test")
FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def rocksalt(a=5.64, pbc=(True, True, True), jitter=0.0, seed=3):
    pos = np.vstack([FCC * a, (FCC + [0.5, 0, 0]) * a])
    if jitter:
        pos = pos + np.random.default_rng(seed).normal(0.0, jitter, pos.shape)
    cell = Cell.cubic(a, pbc) if any(pbc) else Cell.none()
    return Structure([11] * 4 + [17] * 4, pos, cell)


def moved(structure, index, axis, delta):
    out = structure.copy()
    positions = out.positions.copy()
    positions[index, axis] += delta
    out.positions = positions
    return out


GEOMETRIES = [(True, True, True), (True, True, False), (False, False, False)]


def test_charge_model_refuses_unlisted_element():
    s = rocksalt()
    with pytest.raises(es.ChargeModelError, match="Cl"):
        ChargeModel.per_element({"Na": 1.0}).charges(s)


def test_charge_model_explicit_zero_is_accepted():
    s = rocksalt()
    q = ChargeModel.per_element({"Na": 0.0, "Cl": 0.0}).charges(s)
    assert np.all(q == 0.0)


def test_per_atom_model_refuses_missing_and_stale_ids():
    s = rocksalt()
    table = {int(i): 1.0 for i in s.ids}
    model = ChargeModel.per_atom(table)
    s.remove_atoms([int(s.ids[0])])
    with pytest.raises(es.ChargeModelError, match="no longer exist"):
        model.charges(s)
    s.add_atom("Na", [1.0, 1.0, 1.0])
    with pytest.raises(es.ChargeModelError, match="have no charge"):
        model.charges(s)


def test_formal_point_ion_reads_formal_charges():
    s = rocksalt()
    s.formal_charges[:4] = 1.0
    s.formal_charges[4:] = -1.0
    q = ChargeModel.formal_point_ion().charges(s)
    assert q.tolist() == [1.0] * 4 + [-1.0] * 4
    with pytest.raises(es.ChargeModelError):
        ChargeModel("formal-point-ion", by_element=(("Na", 1.0),))


@pytest.mark.parametrize("bad", [
    {"kind": "guess"},
    {"kind": "per-element", "by_element": {}},
    {"kind": "per-element", "by_element": {"Na": float("nan")}},
])
def test_invalid_charge_models_are_refused(bad):
    with pytest.raises(es.ChargeModelError):
        ChargeModel.from_dict(bad)


def test_charge_model_round_trips_through_structure_info():
    s = rocksalt()
    es.store_charge_model(s, NACL)
    copy = Structure.from_dict(s.as_dict())
    assert es.stored_charge_model(copy) == NACL
    es.store_charge_model(s, None)
    assert es.stored_charge_model(s) is None


def test_geometry_classification():
    assert es.classify(Cell.cubic(5.0)).kind == "bulk-3d"
    assert es.classify(Cell.cubic(5.0, (True, True, False))).kind == "slab-2d"
    assert es.classify(Cell.none()).kind == "isolated"
    wire = es.classify(Cell.cubic(5.0, (False, False, True)))
    assert wire.kind == "wire-1d" and not wire.supported


def test_wire_is_refused():
    s = rocksalt(pbc=(False, False, True))
    with pytest.raises(es.UndefinedElectrostatics, match="one-dimensional"):
        es.compute(s, NACL)


def test_charged_bulk_needs_an_explicit_background():
    s = rocksalt()
    model = ChargeModel.per_element({"Na": 1.0, "Cl": -0.5})
    with pytest.raises(es.UndefinedElectrostatics, match="neutralising background"):
        es.compute(s, model)
    out = es.compute(s, model, EwaldSettings(background="uniform"))
    assert "background" in out.components_eV
    assert out.components_eV["background"] < 0.0


def test_charged_slab_is_refused_even_with_a_background():
    s = rocksalt(pbc=(True, True, False))
    model = ChargeModel.per_element({"Na": 1.0, "Cl": -0.5})
    for settings in (EwaldSettings(), EwaldSettings(background="uniform")):
        with pytest.raises(es.UndefinedElectrostatics, match="charged"):
            es.compute(s, model, settings)


def test_charged_cluster_is_allowed_but_a_background_is_not():
    s = Structure([11, 11], [[0, 0, 0], [3.0, 0, 0]], Cell.none())
    model = ChargeModel.per_element({"Na": 1.0})
    out = es.compute(s, model)
    assert out.energy_eV == pytest.approx(es.COULOMB_K_EV_A / 3.0, rel=1e-12)
    with pytest.raises(es.UndefinedElectrostatics, match="isolated cluster"):
        es.compute(s, model, EwaldSettings(background="uniform"))


def test_vacuum_surrounding_refused_for_charged_cell():
    s = rocksalt()
    model = ChargeModel.per_element({"Na": 1.0, "Cl": -0.5})
    with pytest.raises(es.UndefinedElectrostatics, match="origin"):
        es.compute(s, model, EwaldSettings(background="uniform", surrounding="vacuum"))


def test_coincident_atoms_are_refused():
    s = rocksalt()
    s.add_atom("Na", s.positions[1] + 1e-8)
    model = ChargeModel.per_element({"Na": 1.0, "Cl": -1.0})
    s.remove_atoms([int(s.ids[0])])
    with pytest.raises(es.UndefinedElectrostatics, match="coincide"):
        es.compute(s, model, EwaldSettings(background="uniform"))
    cluster = Structure([11, 17], [[0, 0, 0], [0, 0, 0]], Cell.none())
    with pytest.raises(es.UndefinedElectrostatics, match="coincide"):
        es.compute(cluster, NACL)


@pytest.mark.parametrize("bad", [
    {"accuracy": 0.5}, {"accuracy": -1}, {"surrounding": "water"},
    {"background": "gaussian"}, {"alpha_per_A": -1.0}, {"unknown_key": 1},
])
def test_settings_validation(bad):
    with pytest.raises(ValueError):
        EwaldSettings.from_dict(bad)


@pytest.mark.parametrize("pbc", GEOMETRIES)
def test_forces_are_the_negative_energy_gradient(pbc):
    s = rocksalt(pbc=pbc, jitter=0.12)
    settings = EwaldSettings(accuracy=1e-10, check_convergence=False)
    out = es.compute(s, NACL, settings)
    h = 1e-5
    worst = 0.0
    for i in range(len(s)):
        for axis in range(3):
            plus = es.compute(moved(s, i, axis, h), NACL, settings).energy_eV
            minus = es.compute(moved(s, i, axis, -h), NACL, settings).energy_eV
            worst = max(worst, abs(-(plus - minus) / (2 * h) - out.forces_eV_A[i, axis]))
    assert worst < 1e-6


@pytest.mark.parametrize("pbc", GEOMETRIES)
def test_forces_sum_to_zero_and_equal_q_times_field(pbc):
    s = rocksalt(pbc=pbc, jitter=0.12)
    out = es.compute(s, NACL)
    assert np.abs(out.forces_eV_A.sum(axis=0)).max() < 1e-9
    np.testing.assert_allclose(out.forces_eV_A, out.charges_e[:, None] * out.site_field_V_A)


@pytest.mark.parametrize("pbc", GEOMETRIES)
def test_energy_is_half_sum_of_charge_times_potential(pbc):
    s = rocksalt(pbc=pbc, jitter=0.12)
    out = es.compute(s, NACL)
    assert out.energy_eV == pytest.approx(0.5 * float(out.charges_e @ out.site_potential_V),
                                          rel=1e-12)


@pytest.mark.parametrize("pbc", GEOMETRIES)
def test_rigid_translation_changes_nothing(pbc):
    s = rocksalt(pbc=pbc, jitter=0.12)
    t = s.copy()
    t.translate([0.731, -1.29, 0.418])
    a, b = es.compute(s, NACL), es.compute(t, NACL)
    assert b.energy_eV == pytest.approx(a.energy_eV, abs=1e-8)
    np.testing.assert_allclose(b.forces_eV_A, a.forces_eV_A, atol=1e-8)


def test_periodic_image_choice_does_not_matter_in_bulk():
    s = rocksalt(jitter=0.12)
    t = s.copy()
    positions = t.positions.copy()
    positions[2] += s.cell.matrix[0] - 2 * s.cell.matrix[2]
    t.positions = positions
    a, b = es.compute(s, NACL), es.compute(t, NACL)
    assert b.energy_eV == pytest.approx(a.energy_eV, abs=1e-8)
    np.testing.assert_allclose(b.forces_eV_A, a.forces_eV_A, atol=1e-8)


def test_periodic_image_choice_does_not_matter_in_a_slab_plane():
    s = rocksalt(pbc=(True, True, False), jitter=0.12)
    t = s.copy()
    positions = t.positions.copy()
    positions[5] += s.cell.matrix[1] - s.cell.matrix[0]
    t.positions = positions
    a, b = es.compute(s, NACL), es.compute(t, NACL)
    assert b.energy_eV == pytest.approx(a.energy_eV, abs=1e-8)


def test_vacuum_surrounding_depends_on_the_stored_image():
    s = rocksalt(jitter=0.12)
    t = s.copy()
    positions = t.positions.copy()
    positions[2] += s.cell.matrix[0]
    t.positions = positions
    settings = EwaldSettings(surrounding="vacuum")
    a, b = es.compute(s, NACL, settings), es.compute(t, NACL, settings)
    assert abs(a.energy_eV - b.energy_eV) > 1e-3
    assert es.compute(s, NACL).energy_eV == pytest.approx(
        es.compute(t, NACL).energy_eV, abs=1e-8)


@pytest.mark.parametrize("pbc", GEOMETRIES[:2])
def test_result_does_not_depend_on_the_splitting_parameter(pbc):
    s = rocksalt(pbc=pbc, jitter=0.12)
    energies = [es.compute(s, NACL, EwaldSettings(accuracy=1e-10, alpha_per_A=a,
                                                  check_convergence=False)).energy_eV
                for a in (0.25, 0.4, 0.6)]
    assert max(energies) - min(energies) < 1e-7


def test_repeated_runs_are_bitwise_identical():
    s = rocksalt(pbc=(True, True, False), jitter=0.12)
    a, b = es.compute(s, NACL), es.compute(s, NACL)
    assert a.energy_eV == b.energy_eV
    assert np.array_equal(a.forces_eV_A, b.forces_eV_A)
    assert np.array_equal(a.site_potential_V, b.site_potential_V)


def test_convergence_check_flags_a_poor_user_cutoff():
    s = rocksalt(jitter=0.12)
    loose = es.compute(s, NACL, EwaldSettings(alpha_per_A=0.4, real_cutoff_A=3.0,
                                              kspace_cutoff_per_A=1.0))
    assert not loose.converged
    assert "Not converged" in loose.convergence_message
    tight = es.compute(s, NACL)
    assert tight.converged and tight.check["energy_difference_eV"] < 1e-5


def test_unchecked_result_is_not_called_converged():
    out = es.compute(rocksalt(), NACL, EwaldSettings(check_convergence=False))
    assert not out.converged
    assert "not checked" in out.convergence_message


def test_progress_is_monotonic_and_cancellation_stops_the_run():
    s = rocksalt(pbc=(True, True, False))
    seen = []
    es.compute(s, NACL, progress=lambda f, m: seen.append(f))
    assert seen and all(b >= a - 1e-12 for a, b in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(1.0)
    with pytest.raises(es.ElectrostaticsCancelled):
        es.compute(s, NACL, cancelled=lambda: True)
    with pytest.raises(es.ElectrostaticsCancelled):
        es.compute(s, NACL, progress=lambda f, m: False)


def test_charge_accounting():
    s = rocksalt()
    acct = es.charge_accounting(s, NACL.charges(s))
    assert acct["total_charge_e"] == 0.0 and acct["neutral"]
    assert acct["sum_abs_charge_e"] == 8.0
    assert acct["per_element"]["Na"] == {"count": 4, "total_e": 4.0, "min_e": 1.0,
                                         "max_e": 1.0}


def test_solver_records_provenance_and_units():
    s = rocksalt(jitter=0.05)
    out = ElectrostaticsSolver().single_point(s, charge_model=NACL)
    energy = out["energy"]
    prov = energy.provenance
    assert energy.unit == "eV" and out["forces"].unit == "eV/A"
    assert out["site_potential"].unit == "V" and out["site_field"].unit == "V/A"
    assert prov.fidelity is Fidelity.TIER1_CLASSICAL
    assert prov.origin is Origin.CALCULATED
    assert prov.parameters["charge_model"]["kind"] == "per-element"
    assert "tinfoil" in prov.boundary_conditions
    assert any("Ewald" in r for r in prov.references)
    assert any("unit test" in r for r in prov.references)
    assert energy.convergence.converged
    assert prov.inputs_digest and energy.extra["atom_ids"] == [int(i) for i in s.ids]


def test_unchecked_solver_result_is_labelled_estimated():
    out = ElectrostaticsSolver().single_point(
        rocksalt(), charge_model=NACL, settings=EwaldSettings(check_convergence=False))
    assert out["energy"].provenance.origin is Origin.ESTIMATED


def test_solver_refuses_without_a_charge_model():
    out = ElectrostaticsSolver().single_point(rocksalt())
    energy = out["energy"]
    assert not energy.supported and energy.value is None
    assert "No point-charge model" in energy.unsupported_reason


def test_solver_reports_cancellation_without_a_number():
    out = ElectrostaticsSolver().single_point(rocksalt(), charge_model=NACL,
                                              cancelled=lambda: True)
    assert not out["energy"].supported
    assert out["energy"].extra["status"] == "cancelled"


def test_solver_does_not_relax_or_run_dynamics():
    solver = ElectrostaticsSolver()
    assert not solver.can(Capability.RELAX) and not solver.can(Capability.DYNAMICS)
    assert solver.can(Capability.SITE_POTENTIAL)
    refused = solver.relax(rocksalt())["relaxed_structure"]
    assert not refused.supported and "short-range" in refused.unsupported_reason


def test_solver_is_registered():
    from materia.solvers import registry

    assert "electrostatics/ewald" in registry.available()
    assert "electrostatics/ewald" in registry.solvers_providing(Capability.SITE_POTENTIAL)
