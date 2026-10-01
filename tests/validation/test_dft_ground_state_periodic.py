"""Physical validation of periodic ground-state DFT calculations."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model import Cell
from materia.python_api import Lab


EV_PER_A3_TO_GPA = 160.21766208


def scaled(structure, factor):
    result = structure.copy()
    result.positions = np.asarray(structure.positions) * factor
    result.cell = Cell(np.asarray(structure.cell.matrix) * factor, structure.cell.pbc)
    return result


def electron_integral(stored):
    return float(np.asarray(stored.data).sum() * stored.meta["voxel_volume_A3"])


@pytest.fixture(scope="module")
def periodic_runs(gpaw_environment):
    if gpaw_environment is None:
        pytest.skip("GPAW or its PAW datasets are not available for periodic validation.")
    lab = Lab()
    silicon = lab.materials.load("silicon").bulk(repeat=(1, 1, 1)).structure
    common = {
        "xc": "LDA",
        "representation": "pw",
        "cutoff_eV": 200.0,
        "grid_spacing_A": None,
        "kpoints": [1, 1, 1],
        "energy_tol_eV_per_electron": 5.0e-4,
        "density_tol_electrons_per_electron": 1.0e-4,
        "max_scf_iterations": 120,
    }
    centre = lab.dft.ground_state(
        silicon, observables=["energy", "forces", "stress", "density"], **common)
    strain = 0.004
    compressed = lab.dft.ground_state(
        scaled(silicon, 1.0 - strain), observables=["energy"], **common)
    expanded = lab.dft.ground_state(
        scaled(silicon, 1.0 + strain), observables=["energy"], **common)
    study = lab.dft.convergence_study(
        silicon,
        parameter="cutoff_eV",
        values=[150.0, 200.0],
        observable="energy_per_atom_eV",
        tolerance=0.2,
        observables=["energy"],
        **common,
    )
    return {
        "lab": lab,
        "silicon": silicon,
        "centre": centre,
        "compressed": compressed,
        "expanded": expanded,
        "strain": strain,
        "study": study,
    }


def test_bulk_silicon_converges_with_symmetric_forces(periodic_runs):
    run = periodic_runs["centre"]
    assert run["run"].extra["status"] == "converged"
    assert run["energy"].convergence.converged is True
    assert -5.5 < run["energy"].extra["energy_per_atom_eV"] < -3.5
    forces = np.asarray(run["forces"].value)
    assert np.linalg.norm(forces.sum(axis=0)) < 1.0e-6
    assert np.linalg.norm(forces, axis=1).max() < 0.05


def test_periodic_density_integrates_to_all_electrons(periodic_runs):
    lab = periodic_runs["lab"]
    run = periodic_runs["centre"]
    run_id = run["run"].extra["run_id"]
    density = lab.dft.array(run_id, "density")
    expected = float(np.asarray(periodic_runs["silicon"].numbers).sum())
    assert electron_integral(density) == pytest.approx(expected, abs=2.0e-4)
    accounting = run["charge_accounting"].value
    assert accounting["expected_total_electrons"] == expected
    assert accounting["balanced"] is True


def test_stress_pressure_matches_an_isotropic_energy_derivative(periodic_runs):
    structure = periodic_runs["silicon"]
    strain = periodic_runs["strain"]
    volume = abs(float(np.linalg.det(structure.cell.matrix)))
    volume_minus = volume * (1.0 - strain) ** 3
    volume_plus = volume * (1.0 + strain) ** 3
    energy_minus = periodic_runs["compressed"]["energy"].value
    energy_plus = periodic_runs["expanded"]["energy"].value
    numerical_pressure = -(energy_plus - energy_minus) / (volume_plus - volume_minus)
    numerical_pressure *= EV_PER_A3_TO_GPA
    stress_pressure = periodic_runs["centre"]["stress"].extra["pressure_GPa"]
    assert stress_pressure == pytest.approx(numerical_pressure, abs=1.0)


def test_cutoff_study_records_points_residual_and_physical_limit(periodic_runs):
    study = periodic_runs["study"]
    assert study.extra["status"] == "complete"
    assert study.convergence.iterations == 2
    assert [point["value"] for point in study.value["points"]] == [150.0, 200.0]
    assert study.value["residual"] >= 0.0
    assert study.value["tolerance_met"] is True
    assert "not an uncertainty" in study.value["note"]
    assert "starts fresh" in study.provenance.parameters["restart_policy"]
