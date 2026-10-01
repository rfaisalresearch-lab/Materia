"""Nuclear masses, Q values, the mass formula, decay chains and particle data."""

from __future__ import annotations

import math

import pytest

from materia.physics import nuclear as N
from materia.provenance import Origin


def test_ame2020_table_is_complete_and_checked():
    table = N.table()
    assert len(table) > 3500
    neutron = table[(0, 1)]
    assert neutron.atomic_mass_u == pytest.approx(1.00866491590, abs=1e-11)
    assert table[(26, 56)].binding_per_nucleon_keV == pytest.approx(8790.3563, abs=1e-4)
    assert any(entry.estimated for entry in table.values())


@pytest.mark.parametrize("element, a", [("H", 2), ("He", 4), ("Fe", 56), ("Pb", 208),
                                        ("U", 238)])
def test_binding_from_masses_equals_the_tabulated_binding(element, a):
    result = N.binding_energy(element, a)
    assert result.value == pytest.approx(result.extra["tabulated_MeV"], abs=2e-5 * a)
    assert result.provenance.origin is Origin.CALCULATED


def test_known_energies():
    assert N.binding_energy("H", 2).value == pytest.approx(2.224566, abs=2e-6)
    assert N.q_value("U", 238, "alpha").value == pytest.approx(4.2699, abs=2e-4)
    assert N.q_value("n", 1, "beta-").value == pytest.approx(0.78235, abs=2e-5)
    assert N.q_value("C", 14, "beta-").value == pytest.approx(0.156476, abs=2e-5)
    assert not N.q_value("Pb", 208, "alpha").extra["allowed"] or \
        N.q_value("Pb", 208, "alpha").value < 1.0
    assert N.separation_energy("Pb", 208, "n").value == pytest.approx(7.3679, abs=1e-3)
    beta_plus = N.q_value("F", 18, "beta+").value
    electron_capture = N.q_value("F", 18, "ec").value
    assert electron_capture - beta_plus == pytest.approx(1.021998, abs=1e-5)


def test_mass_formula_is_a_labelled_fit():
    model = N.fitted_mass_formula()
    assert 15 < model.a_v < 16.5 and 16 < model.a_s < 19 and 0.6 < model.a_c < 0.8
    assert 20 < model.a_a < 25 and model.rms_MeV < 4.0
    result = N.mass_formula_binding("Fe", 56)
    assert abs(result.value - result.extra["measured_MeV"]) < 3 * model.rms_MeV
    assert result.provenance.origin is Origin.ESTIMATED
    assert result.uncertainty == pytest.approx(model.rms_MeV)


def test_decay_chain_matches_the_bateman_solution():
    pytest.importorskip("radioactivedecay", reason="BLOCKED: radioactivedecay not installed")
    import radioactivedecay as rd
    half_life_y = rd.Nuclide("Sr-90").half_life("y")
    assert half_life_y == pytest.approx(
        N.decay_data("Sr-90")["half_life_s"] / (365.2422 * 86400), rel=1e-6)
    out = N.decay({"Sr-90": 1000.0}, 10.0, "y").value["activities_Bq"]
    assert out["Sr-90"] == pytest.approx(1000.0 * 2 ** (-10.0 / half_life_y), rel=1e-9)
    assert out["Y-90"] == pytest.approx(out["Sr-90"], rel=1e-3)
    assert "Th-234" in N.decay_data("U-238")["progeny"]
    with pytest.raises(N.NuclearError):
        N.decay_data("Xx-999")


def test_particle_data():
    pytest.importorskip("particle", reason="BLOCKED: particle not installed")
    assert N.particle_data("p")["mass_MeV"] == pytest.approx(938.27208943, abs=1e-6)
    assert N.particle_data(11)["mass_MeV"] == pytest.approx(0.51099895, abs=1e-8)
    with pytest.raises(N.NuclearError):
        N.particle_data("not-a-particle")


def test_refusals(monkeypatch):
    with pytest.raises(N.NuclearError, match="no mass"):
        N.nuclide("Fe", 3)
    with pytest.raises(N.NuclearError, match="mode"):
        N.q_value("U", 238, "fission")
    N.table.cache_clear()
    monkeypatch.setattr(N, "AME_SHA256", "0" * 64)
    with pytest.raises(N.NuclearError, match="checksum"):
        N.table()
    monkeypatch.undo()
    N.table.cache_clear()
    assert math.isfinite(N.binding_energy("O", 16).value)


def test_reaction_q_values():
    assert N.reaction_q_value(["d", "t"], ["a", "n"]).value == pytest.approx(17.5893, abs=2e-4)
    assert N.reaction_q_value(["d", "d"], ["He-3", "n"]).value == pytest.approx(3.2689, abs=2e-4)
    assert N.reaction_q_value(["d", "d"], ["t", "p"]).value == pytest.approx(4.0327, abs=2e-4)
    with pytest.raises(N.NuclearError, match="Charge"):
        N.reaction_q_value(["d", "t"], ["a", "p"])


@pytest.mark.parametrize("energy, pstar_stopping, pstar_range", [(10.0, 45.67, 0.1230),
                                                                 (100.0, 7.289, 7.718)])
def test_proton_stopping_in_water_against_nist_pstar(energy, pstar_stopping, pstar_range):
    stopping = N.bethe_stopping_power(energy, "water").value
    assert stopping == pytest.approx(pstar_stopping, rel=0.01)
    result = N.csda_range(energy, "water")
    omitted = result.extra["omitted_below_floor_g_cm2_at_most"]
    assert result.value < pstar_range < result.value + omitted + 0.002 * pstar_range


def test_stopping_refusals():
    with pytest.raises(N.NuclearError, match="1 MeV per nucleon"):
        N.bethe_stopping_power(0.5)
    with pytest.raises(N.NuclearError, match="density effect"):
        N.bethe_stopping_power(5000.0)
    with pytest.raises(N.NuclearError, match="material"):
        N.bethe_stopping_power(10.0, "unobtainium")
