"""Physical validation of finite ground-state DFT calculations."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.python_api import Lab


def h2(distance=0.74, box=8.0):
    centre = box / 2
    return Structure(
        np.array([1, 1]),
        np.array([[centre, centre, centre - distance / 2],
                  [centre, centre, centre + distance / 2]]),
        Cell(np.eye(3) * box, (False, False, False)),
    )


def electron_integral(stored):
    return float(np.asarray(stored.data).sum() * stored.meta["voxel_volume_A3"])


@pytest.fixture(scope="module")
def finite_runs(gpaw_environment):
    if gpaw_environment is None:
        pytest.skip("GPAW or its PAW datasets are not available for finite validation.")
    lab = Lab()
    common = {
        "xc": "PBE",
        "grid_spacing_A": 0.22,
        "energy_tol_eV_per_electron": 2.0e-4,
        "density_tol_electrons_per_electron": 5.0e-5,
        "max_scf_iterations": 150,
    }
    base = lab.dft.ground_state(
        h2(), observables=["energy", "forces", "density", "eigenvalues", "occupations"],
        **common,
    )
    step = 0.01
    shorter = lab.dft.ground_state(h2(0.74 - step), observables=["energy"], **common)
    longer = lab.dft.ground_state(h2(0.74 + step), observables=["energy"], **common)
    cation = lab.dft.ground_state(
        h2(), charge_e=1.0,
        observables=["energy", "density", "spin_density", "magnetic_moment"],
        **common,
    )
    return {
        "lab": lab,
        "base": base,
        "shorter": shorter,
        "longer": longer,
        "cation": cation,
        "step": step,
    }


def test_neutral_h2_converges_with_complete_provenance(finite_runs):
    run = finite_runs["base"]
    energy = run["energy"]
    record = run["run"]
    assert record.extra["status"] == "converged"
    assert energy.supported
    assert energy.convergence.converged is True
    assert -7.0 < energy.value < -6.0
    assert energy.provenance.origin.value == "calculated"
    assert energy.provenance.fidelity.value == "tier3-external-first-principles"
    assert energy.provenance.inputs_digest
    assert energy.provenance.parameters["paw_datasets"][0]["sha256"]
    assert energy.provenance.parameters["software_reported"]["gpaw"]


def test_neutral_h2_density_and_occupations_account_for_two_electrons(finite_runs):
    lab = finite_runs["lab"]
    run = finite_runs["base"]
    run_id = run["run"].extra["run_id"]
    density = lab.dft.array(run_id, "density")
    assert electron_integral(density) == pytest.approx(2.0, abs=2.0e-6)
    accounting = run["charge_accounting"].value
    assert accounting["expected_total_electrons"] == pytest.approx(2.0)
    assert accounting["occupied_electrons"] == pytest.approx(2.0, abs=2.0e-6)
    assert accounting["balanced"] is True


def test_h2_force_matches_a_central_energy_difference(finite_runs):
    shorter = finite_runs["shorter"]["energy"].value
    longer = finite_runs["longer"]["energy"].value
    numerical_force_on_upper_atom = -(longer - shorter) / (2 * finite_runs["step"])
    analytic_force_on_upper_atom = finite_runs["base"]["forces"].value[1, 2]
    assert analytic_force_on_upper_atom == pytest.approx(
        numerical_force_on_upper_atom, abs=0.03)
    assert np.linalg.norm(finite_runs["base"]["forces"].value.sum(axis=0)) < 1.0e-6


def test_h2_cation_has_one_electron_and_one_bohr_magneton(finite_runs):
    lab = finite_runs["lab"]
    run = finite_runs["cation"]
    run_id = run["run"].extra["run_id"]
    assert run["run"].extra["status"] == "converged"
    assert electron_integral(lab.dft.array(run_id, "density")) == pytest.approx(
        1.0, abs=2.0e-6)
    assert run["charge_accounting"].value["expected_total_electrons"] == 1.0
    assert run["charge_accounting"].value["balanced"] is True
    assert abs(run["magnetic_moment"].value) == pytest.approx(1.0, abs=2.0e-3)
    ionisation = run["energy"].value - finite_runs["base"]["energy"].value
    assert 10.0 < ionisation < 22.0
