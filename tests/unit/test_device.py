"""DEVSIM p-n junction against closed-form junction and diode theory."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.physics import device as D

try:
    D._devsim()
except D.DeviceError as exc:
    pytest.skip(f"BLOCKED: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def junction():
    return D.pn_junction(1e17, 1e16, biases_V=[0.0, -0.5, 0.3, 0.4, 0.5])


def test_built_in_potential_is_exact(junction):
    result = junction["built_in_potential"]
    assert result.value == pytest.approx(result.extra["analytic_V"], abs=1e-6)
    vt = D.K_J_K * 300 / D.Q_C
    assert result.value == pytest.approx(vt * math.log(1e33 / 9.65e9 ** 2), abs=1e-6)


def test_field_and_charge_follow_the_depletion_picture(junction):
    peak = junction["peak_field"]
    assert peak.value == pytest.approx(peak.extra["depletion_approximation_V_cm"], rel=0.05)
    assert peak.value < peak.extra["depletion_approximation_V_cm"]
    profile = junction["profile"].value
    x = np.array(profile["x_um"]) * 1e-4
    charge = np.array(profile["charge_C_cm3"])
    assert abs(np.trapezoid(charge, x)) < 1e-6 * np.trapezoid(np.abs(charge), x)


def test_forward_current_is_ideal_diffusion(junction):
    iv = junction["iv"]
    bias = np.array(iv.value["bias_V"])
    current = np.array(iv.value["current_A_cm2"])
    vt = D.K_J_K * 300 / D.Q_C
    ideality = 0.1 / vt / math.log(current[4] / current[3])
    assert 1.0 <= ideality < 1.05
    shockley = iv.extra["short_diode_J0_A_cm2"] * (math.exp(0.5 / vt) - 1)
    assert current[4] == pytest.approx(shockley, rel=0.25)
    assert current[1] < 0 and abs(current[1]) > 10 * iv.extra["resolution_A_cm2"]
    assert iv.extra["resolution_A_cm2"] < 1e-8
    assert iv.provenance.parameters["silicon"]["n_i_cm3"] == 9.65e9


def test_refusals():
    with pytest.raises(D.DeviceError, match="nondegenerate"):
        D.pn_junction(1e20, 1e16)
    with pytest.raises(D.DeviceError, match="high injection"):
        D.pn_junction(biases_V=[0.9])
    with pytest.raises(D.DeviceError, match="300 K"):
        D.pn_junction(temperature_K=350)


def test_mos_charge_equals_the_exact_boltzmann_solution():
    out = D.mos_capacitor(1e17, 10.0)
    cv = out["cv"]
    charge = np.array(cv.value["gate_charge_C_cm2"])
    exact = np.array(cv.extra["exact_gate_charge_C_cm2"])
    assert np.max(np.abs(charge - exact)) < 2e-4 * np.max(np.abs(exact))
    ratio = np.array(cv.value["capacitance_F_cm2"]) / cv.extra["oxide_capacitance_F_cm2"]
    assert ratio[0] > 0.9 and ratio[-1] > 0.9 and ratio.min() < 0.35
    assert 0 < ratio.min()
    gate = np.array(cv.value["gate_V"])
    assert abs(gate[np.argmin(ratio)] - cv.extra["flat_band_V"]) < 1.5


def test_mos_refusals():
    with pytest.raises(D.DeviceError, match="increasing"):
        D.mos_capacitor(gate_V=[0.0, -1.0, 1.0])
    with pytest.raises(D.DeviceError, match="oxide_nm"):
        D.mos_capacitor(oxide_nm=0.5)


@pytest.fixture(scope="module")
def illuminated():
    biases = [0.0, 0.1, 0.2, 0.25, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.45]
    return D.pn_junction(1e17, 1e16, length_um=2.0, biases_V=biases, lifetime_s=1e-3,
                         generation_cm3_s=1e19)


def test_short_base_photocurrent_is_half_of_the_neutral_generation(illuminated):
    iv = illuminated["iv"]
    bias = np.array(iv.value["bias_V"])
    photo = np.array(iv.value["current_A_cm2"]) - np.array(iv.extra["dark_current_A_cm2"])
    assert iv.extra["generated_current_A_cm2"] == pytest.approx(D.Q_C * 1e19 * 2e-4, rel=1e-6)
    for v, j in zip(bias, photo):
        width = D.analytic_junction(1e17, 1e16, bias_V=v)["depletion_width_um"] * 1e-4
        assert -j == pytest.approx(D.Q_C * 1e19 * (2e-4 + width) / 2, rel=0.015)


def test_open_circuit_voltage_balances_dark_current(illuminated):
    iv = illuminated["iv"]
    metrics = D.solar_metrics(iv)
    bias = np.array(iv.value["bias_V"])
    dark = np.array(iv.extra["dark_current_A_cm2"])
    voc = metrics["open_circuit_voltage_V"]
    dark_at_voc = math.exp(np.interp(voc, bias[1:], np.log(dark[1:])))
    photo = np.array(iv.value["current_A_cm2"]) - dark
    assert dark_at_voc == pytest.approx(-np.interp(voc, bias, photo), rel=0.05)
    assert 0.3 < voc < 0.4
    assert 0.6 < metrics["fill_factor"] < 0.9
    assert metrics["short_circuit_current_A_cm2"] == pytest.approx(-iv.value["current_A_cm2"][0])


def test_beer_lambert_collection_is_bounded_by_absorbed_photons():
    flux, alpha = 1e17, 1e4
    out = D.pn_junction(1e17, 1e16, length_um=2.0, biases_V=[0.0], lifetime_s=1e-3,
                        photon_flux_cm2_s=flux, absorption_per_cm=alpha)
    absorbed = D.Q_C * flux * (1 - math.exp(-alpha * 2e-4))
    assert out["iv"].extra["generated_current_A_cm2"] == pytest.approx(absorbed, rel=1e-3)
    assert 0 < -out["iv"].value["current_A_cm2"][0] < absorbed


def test_illumination_refusals(illuminated):
    with pytest.raises(D.DeviceError, match="absorption"):
        D.pn_junction(photon_flux_cm2_s=1e17)
    with pytest.raises(D.DeviceError, match="negative"):
        D.pn_junction(generation_cm3_s=-1.0)
    dark = D.pn_junction(1e17, 1e16, biases_V=[0.0, 0.3])
    with pytest.raises(D.DeviceError, match="not illuminated"):
        D.solar_metrics(dark["iv"])
