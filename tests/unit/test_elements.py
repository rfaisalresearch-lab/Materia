"""Periodic-table, isotope and electron-configuration data."""

from __future__ import annotations

import pytest

from materia.elements import periodic_table as pt
from materia.elements.configuration import (
    configuration,
    configuration_string,
    shell_occupancy,
    spin_orbitals,
    term_spin_multiplicity,
    unpaired_electrons,
    valence_electrons,
)
from materia.elements.data import ELEMENTS


def test_table_covers_all_named_elements():
    assert len(ELEMENTS) == 118
    assert [e.number for e in ELEMENTS] == list(range(1, 119))
    assert len({e.symbol for e in ELEMENTS}) == 118


@pytest.mark.parametrize("key", ["Si", "si", "silicon", "Silicon", 14])
def test_lookup_accepts_symbol_name_and_number(key):
    assert pt.element(key).number == 14


def test_unknown_element_is_rejected():
    with pytest.raises(pt.UnknownElement):
        pt.element("Xx")


@pytest.mark.parametrize(
    "symbol,weight",
    [("H", 1.008), ("C", 12.011), ("Si", 28.085), ("Au", 196.96657), ("W", 183.84)],
)
def test_standard_atomic_weights(symbol, weight):
    assert pt.mass(symbol) == pytest.approx(weight, rel=1e-6)


def test_isotope_masses_and_abundances():
    silicon = pt.element("Si")
    assert [i.mass_number for i in silicon.isotopes] == [28, 29, 30]
    assert silicon.isotope(28).atomic_mass_u == pytest.approx(27.9769265, abs=1e-6)
    total = sum(i.natural_abundance for i in silicon.isotopes)
    assert total == pytest.approx(1.0, abs=1e-4)
    assert silicon.most_abundant_isotope.mass_number == 28


def test_isotopic_abundances_sum_to_one_where_tabulated():
    for element in ELEMENTS:
        abundances = [i.natural_abundance for i in element.isotopes
                      if i.natural_abundance]
        if len(abundances) > 1:
            assert sum(abundances) == pytest.approx(1.0, abs=2e-3), element.symbol


def test_standard_weight_matches_isotope_average():
    for symbol in ("Si", "Cu", "Ga", "Ge", "W", "Mo", "Ni"):
        element = pt.element(symbol)
        average = sum(i.atomic_mass_u * i.natural_abundance
                      for i in element.isotopes if i.natural_abundance)
        assert average == pytest.approx(element.standard_atomic_weight, rel=2e-3), symbol


@pytest.mark.parametrize(
    "z,expected",
    [
        (1, "1s1"),
        (6, "1s2 2s2 2p2"),
        (14, "1s2 2s2 2p6 3s2 3p2"),
        (18, "1s2 2s2 2p6 3s2 3p6"),
        (24, "1s2 2s2 2p6 3s2 3p6 3d5 4s1"),
        (29, "1s2 2s2 2p6 3s2 3p6 3d10 4s1"),
        (31, "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p1"),
        (33, "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p3"),
    ],
)
def test_ground_state_configurations(z, expected):
    assert configuration_string(z) == expected


def test_noble_gas_abbreviation():
    assert configuration_string(14, abbreviated=True) == "[Ne] 3s2 3p2"
    assert configuration_string(79, abbreviated=True) == "[Xe] 4f14 5d10 6s1"


def test_ion_configurations_remove_outermost_electrons():
    assert configuration_string(14, charge=4) == "1s2 2s2 2p6"
    assert configuration_string(8, charge=-2) == "1s2 2s2 2p6"
    assert configuration_string(26, charge=2).endswith("3d6")


def test_electron_count_is_conserved():
    for z in range(1, 104):
        for charge in (-1, 0, 1, 2):
            if z - charge <= 0:
                continue
            total = sum(s.electrons for s in configuration(z, charge))
            assert total == z - charge


def test_subshell_capacity_never_exceeded():
    for z in range(1, 119):
        for shell in configuration(z):
            assert 0 < shell.electrons <= shell.capacity


def test_pauli_exclusion_holds_for_enumerated_spin_orbitals():
    for z in (6, 14, 26, 79):
        seen = set()
        for orbital in spin_orbitals(z):
            if not orbital.occupied:
                continue
            key = (orbital.n, orbital.l, orbital.m_l, orbital.m_s)
            assert key not in seen
            seen.add(key)


def test_hund_rule_maximises_multiplicity():
    assert unpaired_electrons(7) == 3
    assert unpaired_electrons(8) == 2
    assert term_spin_multiplicity(26) == 5
    assert unpaired_electrons(10) == 0


def test_valence_and_shell_occupancy():
    assert valence_electrons(14) == 4
    assert valence_electrons(15) == 5
    assert shell_occupancy(14) == [2, 8, 4]
    assert sum(shell_occupancy(79)) == 79


def test_covalent_radii_are_physical():
    for element in ELEMENTS:
        if element.covalent_radius_A is not None:
            assert 0.2 < element.covalent_radius_A < 3.0, element.symbol
