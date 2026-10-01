"""PBE equations of state through Materia and GPAW against all-electron references.

These cases run only when GPAW and its PAW datasets are installed; without
them each case skips with the reason ``BLOCKED``, which is not a pass.

Seven volumes from 0.97 a0 to 1.03 a0, each a Materia ground-state
experiment with one explicit, fixed k-point grid, are fitted with a
third-order Birch-Murnaghan equation of state.  The references are WIEN2k
all-electron PBE values from the Delta project (K. Lejaeghere et al.,
Science 351 (2016) aad3000).  The tolerances allow for the difference between
GPAW's PAW datasets and an all-electron calculation, which for copper is the
dominant term: GPAW run directly, without Materia, gives the same copper
volume to 1e-4 A^3 at 600 and at 900 eV, 1.2 percent above the all-electron
value.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from materia.core_model import Cell, Structure
from materia.experiments.dft import spec as specs
from materia.experiments.dft.run import execute
from materia.solvers.gpaw_driver import discover

pytestmark = pytest.mark.validation

EV_A3_TO_GPA = 160.21766208

CASES = {
    "Si": ("diamond", 14, 5.469, {"cutoff_eV": 500.0, "kpoints": [12, 12, 12],
                                  "occupations": "fermi-dirac", "smearing_eV": 0.01},
           20.453, 88.5, 0.006, 0.03),
    "Al": ("fcc", 13, 4.040, {"cutoff_eV": 500.0, "kpoints": [18, 18, 18],
                              "occupations": "fermi-dirac", "smearing_eV": 0.1},
           16.480, 78.0, 0.006, 0.03),
    "Cu": ("fcc", 29, 3.632, {"cutoff_eV": 600.0, "kpoints": [16, 16, 16],
                              "occupations": "fermi-dirac", "smearing_eV": 0.1},
           11.951, 141.3, 0.015, 0.05),
}


@pytest.fixture(scope="module")
def gpaw():
    environment = discover()
    if not environment.operational:
        pytest.skip("BLOCKED: live GPAW equations of state were not run because GPAW is "
                    f"not usable here. {environment.blocking_reason()} "
                    f"{environment.install_hint()}")
    return environment


def birch_murnaghan(volume, e0, v0, b0, b1):
    x = (v0 / volume) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((x - 1) ** 3 * b1 + (x - 1) ** 2 * (6 - 4 * x))


@pytest.mark.parametrize("element", sorted(CASES))
def test_equation_of_state_against_all_electron_pbe(gpaw, element):
    kind, z, a_guess, settings, v_ref, b_ref, v_tol, b_tol = CASES[element]
    volumes, energies = [], []
    for factor in np.linspace(0.97, 1.03, 7):
        a = a_guess * factor
        cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        if kind == "diamond":
            crystal = Structure(np.array([z, z]), np.array([[0, 0, 0], [a / 4] * 3]),
                                Cell(cell, (True, True, True)))
        else:
            crystal = Structure(np.array([z]), np.zeros((1, 3)), Cell(cell, (True, True, True)))
        spec = specs.build(crystal, gpaw, None, xc="PBE", kpoints_gamma_centered=True,
                           observables=["energy"], **settings)
        report = specs.check(spec, gpaw)
        assert report.ok, report.blocking
        assert not any("automatic choice" in w for w in report.warnings)
        outcome = execute(spec, gpaw, timeout_s=3600)
        assert outcome.converged, outcome.reason
        volumes.append(abs(np.linalg.det(cell)) / len(crystal))
        energies.append(outcome.results["energy"].value / len(crystal))
    volumes, energies = np.array(volumes), np.array(energies)
    fit, _ = curve_fit(birch_murnaghan, volumes, energies,
                       p0=(energies.min(), volumes[np.argmin(energies)], 0.5, 4.5))
    residual = np.sqrt(np.mean((birch_murnaghan(volumes, *fit) - energies) ** 2))
    assert residual < 1e-3
    assert fit[1] == pytest.approx(v_ref, rel=v_tol)
    assert fit[2] * EV_A3_TO_GPA == pytest.approx(b_ref, rel=b_tol)
    assert 3.5 < fit[3] < 6.0


def silicon(a=5.469):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


def test_equation_of_state_tool_is_confirmed_by_gpaw_stress(gpaw, tmp_path):
    """The fitted equilibrium, applied, is checked by an independent stress calculation.

    Si, PBE, 500 eV, 12x12x12: the tool's V0 must match the all-electron value as
    above, and a fresh ground state at the applied geometry must show a pressure
    below 0.3 GPa from GPAW's analytic stress, a route that shares nothing with
    the energy fit.
    """
    from materia.desktop_ui.service import Service

    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(silicon(5.43))
    settings = {"xc": "PBE", "cutoff_eV": 500.0, "kpoints": [12, 12, 12],
                "kpoints_gamma_centered": True, "occupations": "fermi-dirac",
                "smearing_eV": 0.01}
    out = service.dft_eos_run({**settings, "volume_min_scale": 0.97, "volume_max_scale": 1.09},
                              background=False, timeout_s=3600)
    assert out["ok"], out
    assert out["V0_A3_per_atom"] == pytest.approx(20.453, rel=0.006)
    assert out["B0_GPa"] == pytest.approx(88.5, rel=0.03)
    assert out["checks"]["stress_pressure_max_difference_GPa"] < 1.0
    assert service.dft_eos_apply(out["run_id"])["ok"]
    check = service.dft_run({**settings, "observables": ["energy", "stress"]},
                            background=False, timeout_s=3600)
    assert check["ok"], check
    assert abs(check["pressure_GPa"]) < 0.3
    service.shutdown(grace_s=2.0)


def test_equation_of_state_tool_pins_an_automatic_metal_grid(gpaw):
    """Copper with the automatic grid over +-10 percent in volume, where a grid chosen per
    cell would be 13x13x13 at the small volumes and 12x12x12 at the large ones, gives a
    smooth curve and a physical B' because the tool pins the reference's grid. The
    residual that remains, a few tenths of a meV per atom, is the k-point sampling
    noise of a 12x12x12 grid for a smeared metal; a scan whose grid changed part way
    gave 4.1 meV per atom and B' = -13.9."""
    from materia.experiments.dft import eos

    a = 3.632
    copper = Structure(np.array([29]), np.zeros((1, 3)),
                       Cell(0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                            (True, True, True)))
    spec = eos.build(copper, gpaw, None, xc="PBE", volume_min_scale=0.9,
                     volume_max_scale=1.1, n_points=9)
    grids = {specs.default_kpoints(np.asarray(p.cell_A), p.pbc)
             for p in eos.point_specs(spec, gpaw)}
    assert len(grids) > 1
    outcome = eos.execute(spec, gpaw, timeout_s=3600)
    assert outcome.ok, outcome.reason
    value = outcome.results["eos"].value
    assert value["checks"]["fit_rms_meV_per_atom"] < 1.0
    assert 4.0 < value["B1"] < 6.5
    assert value["V0_A3_per_atom"] == pytest.approx(11.951, rel=0.02)
    assert value["energy_definition"] == "zero-width extrapolated energy"
    points = value["points"]
    entropy = [p["free_energy_eV"] - p["energy_eV"] for p in points]
    assert all(e < 0 for e in entropy) and max(entropy) - min(entropy) > 1e-4
    zero_width = eos.fit_birch_murnaghan([p["volume_A3"] for p in points],
                                         [p["energy_eV"] for p in points])
    assert value["V0_A3"] == pytest.approx(zero_width["V0_A3"], rel=1e-12)
    free = value["free_energy_fit"]
    assert free["V0_A3"] != pytest.approx(value["V0_A3"], rel=1e-5)
    assert value["checks"]["stress_compared_with"] == "-dF/dV of the free-energy fit"


def test_zero_width_and_free_energy_agree_without_smearing(gpaw):
    """With fixed occupations there is no entropy term: both curves coincide exactly."""
    from materia.experiments.dft import eos

    spec = eos.build(silicon(5.469), gpaw, None, xc="PBE", cutoff_eV=400.0,
                     kpoints=[6, 6, 6], occupations="fixed", smearing_eV=0.0, n_points=5)
    outcome = eos.execute(spec, gpaw, timeout_s=3600)
    assert outcome.ok, outcome.reason
    value = outcome.results["eos"].value
    for point in value["points"]:
        assert point["free_energy_eV"] == pytest.approx(point["energy_eV"], abs=1e-9)
    assert value["free_energy_fit"]["V0_A3"] == pytest.approx(value["V0_A3"], rel=1e-9)
