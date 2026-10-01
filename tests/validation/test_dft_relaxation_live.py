"""Real GPAW relaxations, checked against independent calculations and references.

These cases run only when GPAW and its PAW datasets are installed.  Without
them each case is skipped with the reason ``BLOCKED``: the validation has not
been performed, and a skip is not a pass.

References and tolerances
-------------------------
* H2 with PBE: equilibrium bond length about 0.750 A (the experimental value is
  0.741 A; PBE overestimates it slightly).  On a 0.22 A grid in a 6 A box the
  tolerance is 0.01 A, which covers the grid and box error.
* Diamond-structure Si with PBE: lattice constant about 5.47 A.  With a 500 eV
  cutoff and a 4x4x4 k-grid the tolerance is 0.04 A.
* Every relaxed geometry is recomputed with an independent ground-state run
  (the existing DFT experiment, fresh SCF, no restart): its energy must match
  the relaxation's final energy to 1e-4 eV, and its forces, and for a variable
  cell its stress, must meet the relaxation's own criteria.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import relaxation as relax
from materia.experiments.dft.run import execute as ground_state
from materia.solvers.gpaw_driver import discover

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def gpaw():
    environment = discover()
    if not environment.operational:
        pytest.skip("BLOCKED: live GPAW relaxation was not run because GPAW is not usable "
                    f"here. {environment.blocking_reason()} {environment.install_hint()}")
    return environment


def hydrogen(distance=0.80, box=6.0):
    c = box / 2
    return Structure(np.array([1, 1]),
                     np.array([[c, c, c - distance / 2], [c, c, c + distance / 2]]),
                     Cell(np.eye(3) * box, (False, False, False)))


def silicon(a=5.60):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


H2 = {"xc": "PBE", "grid_spacing_A": 0.22, "forces_tol_eV_A": 0.01, "fmax_eV_A": 0.02,
      "observables": ["energy", "forces"]}
SI = {"mode": "variable-cell", "xc": "PBE", "cutoff_eV": 500.0, "kpoints": [4, 4, 4],
      "occupations": "fixed", "smearing_eV": 0.0, "fmax_eV_A": 0.02,
      "stress_tol_eV_A3": 0.001, "forces_tol_eV_A": 0.005,
      "observables": ["energy", "forces", "stress"]}


def recompute(outcome, gpaw):
    result = ground_state(outcome.final_spec, gpaw, timeout_s=600)
    assert result.converged, result.reason
    return result


def test_h2_fixed_cell_bond_length(gpaw):
    spec = relax.build(hydrogen(), gpaw, None, **H2)
    outcome = relax.execute(spec, gpaw, timeout_s=600)
    assert outcome.status == "converged", outcome.reason
    positions = outcome.arrays["final_positions"].data
    bond = float(np.linalg.norm(positions[1] - positions[0]))
    assert bond == pytest.approx(0.750, abs=0.01)
    summary = outcome.results["relaxation"].value
    assert summary["final_max_force_eV_A"] <= 0.02
    assert summary["energy_change_eV"] < 0
    check = recompute(outcome, gpaw)
    assert check.results["energy"].value == pytest.approx(summary["final_energy_eV"],
                                                          abs=1e-4)
    forces = np.asarray(check.results["forces"].value)
    assert float(np.linalg.norm(forces, axis=1).max()) <= 0.02


def test_h2_with_a_fixed_atom(gpaw):
    s = hydrogen()
    s.fixed[0] = True
    spec = relax.build(s, gpaw, None, **H2)
    outcome = relax.execute(spec, gpaw, timeout_s=600)
    assert outcome.status == "converged", outcome.reason
    positions = outcome.arrays["final_positions"].data
    assert np.array_equal(positions[0], s.positions[0])
    assert float(np.linalg.norm(positions[1] - positions[0])) == pytest.approx(0.750, abs=0.01)
    echo = outcome.results["relaxation"].value["echo"]
    assert echo["fixed_indices"] == [0]


def test_si_variable_cell_lattice_constant(gpaw):
    spec = relax.build(silicon(), gpaw, None, **SI)
    outcome = relax.execute(spec, gpaw, timeout_s=900)
    assert outcome.status == "converged", outcome.reason
    summary = outcome.results["relaxation"].value
    volume = summary["cell"]["final_volume_A3"]
    lattice = (4.0 * volume) ** (1.0 / 3.0)
    assert lattice == pytest.approx(5.47, abs=0.04)
    assert summary["final_stress_residual_eV_A3"] <= 0.001
    check = recompute(outcome, gpaw)
    stress = np.asarray(check.results["stress"].value)
    assert float(np.abs(stress).max()) <= 0.001 + 1e-5
    assert check.results["energy"].value == pytest.approx(summary["final_energy_eV"],
                                                          abs=1e-4)


def test_service_applies_and_undoes_a_real_relaxation(gpaw, tmp_path):
    from materia.desktop_ui.service import Service

    service = Service(recovery_dir=str(tmp_path / "recovery"))
    s = hydrogen()
    service.project.add_structure(s)
    start = s.positions.copy()
    variables = {k: v for k, v in H2.items()}
    out = service.dft_relax(variables, background=False, timeout_s=600)
    assert out["ok"] and out["status"] == "converged" and out["applied"], out
    assert out["provenance"]["relaxation_used"]["optimizer"] == "BFGS"
    assert out["provenance"]["gpaw_parameters_used"]["symmetry"]["point_group"] is True
    assert float(np.linalg.norm(s.positions[1] - s.positions[0])) == pytest.approx(0.750,
                                                                                   abs=0.01)
    service.undo()
    assert np.array_equal(s.positions, start)
    assert service.dft_relax_result(out["run_id"])["applicable"]
    service.shutdown(grace_s=2.0)
