"""Tip models, instrumental noise, scanning and feature attribution."""

from __future__ import annotations

import numpy as np
import pytest

from materia.materials import load
from materia.microscopy.afm import AFMSettings, AFMSimulator
from materia.microscopy.noise import NoiseModel, apply_noise
from materia.microscopy.stm import STMSettings, STMSimulator
from materia.microscopy.tip import Tip
from materia.solvers.tight_binding import TightBinding
from materia.structure_builder import make_surface
from materia.structure_builder.defects import add_adatom, add_dopant


@pytest.fixture(scope="module")
def slab():
    return make_surface(load("silicon"), (1, 1, 1), size=(3, 3, 3), vacuum_A=12.0)


@pytest.fixture(scope="module")
def stm():
    return STMSimulator(TightBinding("sp3s*-Si"), Tip("W"))


def test_tip_work_function_comes_from_the_material():
    assert Tip("W").work_function_eV == pytest.approx(4.55)
    assert Tip("Pt").work_function_eV == pytest.approx(5.65)
    with pytest.raises(ValueError):
        Tip("unobtainium")


def test_unknown_apex_state_is_rejected():
    with pytest.raises(ValueError):
        Tip("W", apex_state="f")


def test_decay_constant_follows_the_barrier_height(stm):
    from materia.core_model.units import kappa_inv_angstrom

    kappa_low, phi_low = stm.decay_constant(0.0)
    kappa_high, phi_high = stm.decay_constant(3.0)
    assert phi_low == pytest.approx((4.55 + 4.85) / 2, rel=1e-9)
    assert phi_high == pytest.approx((4.55 + 4.85) / 2 - 1.5, rel=1e-9)
    assert kappa_low > kappa_high
    assert kappa_low == pytest.approx(kappa_inv_angstrom(phi_low), rel=1e-12)
    assert 0.9 < kappa_low < 1.3


def test_bias_beyond_the_barrier_is_refused(stm):
    with pytest.raises(ValueError) as excinfo:
        stm.decay_constant(20.0)
    assert "tunnelling approximation has broken down" in str(excinfo.value)


def test_noise_is_reproducible_from_its_seed():
    clean = np.zeros((32, 32))
    first, _ = apply_noise(clean + np.linspace(0, 1, 32)[None, :], NoiseModel.realistic(5))
    second, _ = apply_noise(clean + np.linspace(0, 1, 32)[None, :], NoiseModel.realistic(5))
    third, _ = apply_noise(clean + np.linspace(0, 1, 32)[None, :], NoiseModel.realistic(6))
    assert first == pytest.approx(second)
    assert not np.allclose(first, third)


def test_quiet_noise_model_changes_nothing():
    data = np.random.default_rng(0).random((16, 16))
    noisy, applied = apply_noise(data, NoiseModel.quiet(0))
    assert noisy == pytest.approx(data)
    assert applied == {}


def test_noise_components_are_individually_recorded():
    data = np.linspace(0, 1, 64).reshape(8, 8)
    _, applied = apply_noise(data, NoiseModel.noisy(1))
    assert "white_rms_A" in applied
    assert "pink_rms_A" in applied
    assert "line_offset_rms_A" in applied


def test_constant_current_scan_produces_realistic_corrugation(slab, stm):
    settings = STMSettings(bias_V=1.0, resolution=(96, 96), kgrid=(1, 1),
                           noise=NoiseModel.quiet(0))
    scan = stm.scan(slab, settings)
    assert scan.supported
    topography = scan.channel("topography")
    corrugation = float(np.ptp(topography))
    assert 0.02 < corrugation < 1.5
    assert scan.convergence.converged
    assert scan.convergence.residual < 1e-2


def test_scan_keeps_the_unfiltered_signal_separately(slab, stm):
    settings = STMSettings(bias_V=1.0, resolution=(64, 64), kgrid=(1, 1),
                           noise=NoiseModel.realistic(3))
    scan = stm.scan(slab, settings)
    assert "raw_signal" in scan.channels
    assert not np.allclose(scan.channel("topography"), scan.channel("raw_signal"))
    assert scan.noise_record["reproducible"] is True


def test_maxima_sit_on_the_surface_atoms(slab, stm):
    settings = STMSettings(bias_V=1.0, resolution=(128, 128), kgrid=(1, 1),
                           noise=NoiseModel.quiet(0))
    scan = stm.scan(slab, settings)
    topography = scan.channel("topography")
    z = slab.positions[:, 2]
    top = z.max()
    values = []
    for index in np.nonzero(z >= top - 0.3)[0]:
        x, y = slab.positions[index, :2]
        col, row = scan.xy_to_pixel(float(x), float(y))
        if 0 <= col < topography.shape[1] - 1 and 0 <= row < topography.shape[0] - 1:
            values.append(topography[int(round(row)), int(round(col))])
    assert values
    assert np.mean(values) > topography.mean()


def test_features_are_attributed_to_atomic_sites(slab, stm):
    settings = STMSettings(bias_V=1.0, resolution=(128, 128), kgrid=(1, 1),
                           noise=NoiseModel.quiet(0))
    scan = stm.scan(slab, settings)
    features = scan.detect_features()
    assert features
    assert all(f.kind == "atomic-site" for f in features)
    offsets = np.array([f.lateral_offset_A for f in features])
    _, spacing = scan._surface_lattice()
    assert np.median(offsets) < 0.25
    assert (offsets < 0.6).mean() > 0.85
    assert offsets.max() < 0.55 * spacing
    assert all(f.confidence > 0.3 for f in features)
    assert np.mean([f.confidence for f in features]) > 0.8


def test_hover_identification_degrades_between_sites(slab, stm):
    settings = STMSettings(bias_V=1.0, resolution=(96, 96), kgrid=(1, 1),
                           noise=NoiseModel.quiet(0))
    scan = stm.scan(slab, settings)
    _, spacing = scan._surface_lattice()
    z = slab.positions[:, 2]
    index = int(np.argmax(z))
    x, y = slab.positions[index, :2]

    on_site = scan.identify_at(float(x), float(y))
    assert on_site["kind"] == "atomic-site"
    assert on_site["nearest_element"] == "Si"
    assert on_site["confidence"] > 0.8
    assert "model-based" in on_site["caveat"]

    between = scan.identify_at(float(x + spacing / 2), float(y))
    assert between["kind"] in ("electronic-feature", "uncertain")
    assert between["confidence"] < on_site["confidence"]


def test_bias_polarity_changes_the_sampled_states(slab, stm):
    filled = stm.scan(slab, STMSettings(bias_V=-1.2, resolution=(64, 64),
                                        kgrid=(1, 1), noise=NoiseModel.quiet(0)))
    empty = stm.scan(slab, STMSettings(bias_V=+1.2, resolution=(64, 64),
                                       kgrid=(1, 1), noise=NoiseModel.quiet(0)))
    assert filled.provenance.parameters["energy_window_eV"][1] <= \
        empty.provenance.parameters["energy_window_eV"][1]
    assert not np.allclose(filled.channel("topography"), empty.channel("topography"))


def test_constant_height_mode_reports_current(slab, stm):
    scan = stm.scan(slab, STMSettings(mode="constant-height", height_A=5.0,
                                      resolution=(48, 48), kgrid=(1, 1),
                                      noise=NoiseModel.quiet(0)))
    assert scan.primary_channel == "current"
    assert np.all(scan.channel("topography") == scan.channel("topography")[0, 0])


def test_unknown_stm_mode_is_rejected(slab, stm):
    with pytest.raises(ValueError):
        stm.scan(slab, STMSettings(mode="constant-something"))


def test_stm_refuses_materials_without_parameters(stm):
    gold = make_surface(load("gold"), (1, 1, 1), size=(2, 2, 2), vacuum_A=10.0)
    scan = stm.scan(gold, STMSettings(resolution=(32, 32)))
    assert not scan.supported
    assert "Au" in scan.unsupported_reason
    assert scan.suggested_models


def test_dopant_is_visible_in_the_attribution(slab, stm):
    doped = slab.copy()
    z = doped.positions[:, 2]
    top_index = int(np.argmax(z))
    add_dopant(doped, "P", site=int(doped.ids[top_index]))
    solver = TightBinding("sp3s*-Si", impurities=["P"])
    simulator = STMSimulator(solver, Tip("W"))
    scan = simulator.scan(doped, STMSettings(bias_V=1.0, resolution=(96, 96),
                                             kgrid=(1, 1), noise=NoiseModel.quiet(0)))
    assert scan.supported
    x, y = doped.positions[top_index, :2]
    info = scan.identify_at(float(x), float(y))
    assert info["kind"] == "dopant-site"
    assert info["nearest_element"] == "P"


def test_adsorbate_is_attributed_as_such(slab, stm):
    decorated = slab.copy()
    add_adatom(decorated, "Si", (5.0, 5.0), height_A=2.2)
    scan = stm.scan(decorated, STMSettings(bias_V=1.0, resolution=(96, 96),
                                           kgrid=(1, 1), noise=NoiseModel.quiet(0)))
    info = scan.identify_at(5.0, 5.0)
    assert info["kind"] == "adsorbate"


def test_scan_filters_leave_the_raw_channel_untouched(slab, stm):
    scan = stm.scan(slab, STMSettings(resolution=(64, 64), kgrid=(1, 1),
                                      noise=NoiseModel.realistic(2)))
    raw = scan.channel("topography").copy()
    filtered = scan.filtered("topography", "plane+median")
    assert filtered.shape == raw.shape
    assert not np.allclose(filtered, raw)
    assert scan.channel("topography") == pytest.approx(raw)
    with pytest.raises(ValueError):
        scan.filtered("topography", "wavelet")


def test_spectroscopy_returns_a_curve_with_provenance(slab, stm):
    z = float(slab.positions[:, 2].max()) + 5.0
    result = stm.spectroscopy(slab, 4.0, 4.0, z, bias_range_V=(-1.5, 1.5), n_points=61)
    assert result.supported
    assert result.value["dIdV"].shape == (61,)
    assert np.all(result.value["dIdV"] >= 0)
    assert "arbitrary" in result.extra["note"]
    assert any("Tersoff-Hamann" in a for a in result.provenance.approximations)


def test_afm_frequency_shift_is_negative_above_the_surface(slab):
    afm = AFMSimulator(Tip("W", radius_A=40.0))
    scan = afm.scan(slab, AFMSettings(mode="fm-afm", height_A=4.5,
                                      resolution=(48, 48), noise=NoiseModel.quiet(0)))
    detuning = scan.channel("frequency_shift")
    assert detuning.max() < 0
    assert np.ptp(detuning) > 0


def test_afm_force_curve_has_a_minimum():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 3), vacuum_A=12.0)
    result = AFMSimulator(Tip("W")).force_curve(slab, 2.0, 2.0)
    assert result.supported
    force = result.value["force_nN"]
    assert force.min() < 0
    assert 1.5 < result.extra["height_of_minimum_A"] < 6.0
    assert force[-1] == pytest.approx(0.0, abs=0.2)


def test_afm_reports_pixels_that_never_reach_the_setpoint():
    slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 3), vacuum_A=12.0)
    afm = AFMSimulator(Tip("W"))
    scan = afm.scan(slab, AFMSettings(mode="constant-frequency-shift",
                                      frequency_shift_setpoint_Hz=-1e6,
                                      resolution=(24, 24), noise=NoiseModel.quiet(0)))
    assert scan.provenance.tolerances["pixels_without_setpoint_crossing"] > 0
    assert "never reached the setpoint" in scan.convergence.message
