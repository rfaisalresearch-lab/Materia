"""Physical constants and unit conversions."""

from __future__ import annotations

import pytest

from materia.core_model import units


def test_codata_derived_constants():
    assert units.COULOMB_K_EV_A == pytest.approx(14.399645, rel=1e-6)
    assert units.HBAR2_OVER_2M_EV_A2 == pytest.approx(3.8099821, rel=1e-6)
    assert units.BOLTZMANN_EV_K == pytest.approx(8.617333e-5, rel=1e-6)
    assert units.BOHR_A == pytest.approx(0.529177210903, rel=1e-12)
    assert units.HARTREE_EV == pytest.approx(27.211386, rel=1e-7)


def test_tunnelling_decay_constant_matches_rule_of_thumb():
    kappa = units.kappa_inv_angstrom(4.0)
    assert kappa == pytest.approx(1.0246, rel=2e-3)
    decades_per_angstrom = 2 * kappa / 2.302585
    assert 0.8 < decades_per_angstrom < 0.95


def test_barrier_must_be_positive():
    with pytest.raises(ValueError):
        units.kappa_inv_angstrom(0.0)


def test_md_energy_unit_conversion():
    assert units.U_A2_FS2_TO_EV == pytest.approx(103.6427, rel=1e-5)
    assert units.EV_TO_U_A2_FS2 * units.U_A2_FS2_TO_EV == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value,unit,angstrom",
    [(1, "nm", 10.0), (1, "um", 1e4), (1, "m", 1e10), (1, "bohr", units.BOHR_A)],
)
def test_length_conversions_round_trip(value, unit, angstrom):
    assert units.to_angstrom(value, unit) == pytest.approx(angstrom)
    assert units.from_angstrom(angstrom, unit) == pytest.approx(value)


def test_energy_conversions():
    assert units.to_eV(1, "hartree") == pytest.approx(27.211386, rel=1e-7)
    assert units.to_eV(1, "Ry") == pytest.approx(13.605693, rel=1e-7)
    assert units.to_eV(1, "kcal/mol") == pytest.approx(0.0433641, rel=1e-5)
    assert units.from_eV(units.to_eV(2.5, "kJ/mol"), "kJ/mol") == pytest.approx(2.5)


def test_unknown_units_are_rejected_by_name():
    with pytest.raises(units.UnitError) as excinfo:
        units.to_angstrom(1, "furlong")
    assert "furlong" in str(excinfo.value)


def test_every_internal_quantity_declares_a_unit():
    for quantity, unit in units.INTERNAL_UNITS.items():
        assert isinstance(unit, str) and unit
