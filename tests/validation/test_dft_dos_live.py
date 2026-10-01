"""Real GPAW DOS and projected DOS, checked against independent quantities.

These cases run only when GPAW and its PAW datasets are installed.  Without
them each case is skipped with the reason ``BLOCKED``: the validation has not
been performed, and a skip is not a pass.

What is checked, and against what
---------------------------------
* H2 (PBE, 0.22 A grid, 6 A box): the occupied Kohn-Sham level in the DOS
  equals the eigenvalue of an independent ground-state run to 1e-4 eV and lies
  near the PBE value of about -10.3 eV; the DOS integrates to 2 electrons below
  the Fermi level; the H s projection peaks at that level.
* Si (PBE, 300 eV, 4x4x4 ground state, 6x6x6 DOS grid): 8 valence electrons
  below the Fermi level with Gaussian broadening (to 1e-4) and with the linear
  tetrahedron method (to 0.05, the tetrahedron DOS being sampled on the grid);
  a Kohn-Sham gap on the grid between 0.5 and 0.9 eV, with no DOS at the Fermi
  level; s and p projections both present.
* H atom (spin-polarised): one electron in the up channel and none in the down
  channel below the Fermi level.
* A DOS taken from a real relaxation reproduces the relaxation's final energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import dos as D
from materia.experiments.dft import spec as specs
from materia.experiments.dft.run import execute as ground_state
from materia.solvers.gpaw_driver import discover

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def gpaw():
    environment = discover()
    if not environment.operational:
        pytest.skip("BLOCKED: live GPAW DOS was not run because GPAW is not usable here. "
                    f"{environment.blocking_reason()} {environment.install_hint()}")
    return environment


def hydrogen(box=6.0):
    c = box / 2
    return Structure(np.array([1, 1]), np.array([[c, c, c - 0.37], [c, c, c + 0.37]]),
                     Cell(np.eye(3) * box, (False, False, False)))


def silicon(a=5.47):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


SI = {"xc": "PBE", "cutoff_eV": 300.0, "kpoints": [4, 4, 4], "occupations": "fixed",
      "smearing_eV": 0.0, "dos_kpoints": [6, 6, 6], "n_bands": 12,
      "energy_min_eV": -14.0, "energy_max_eV": 6.0}


def test_h2_level_matches_an_independent_ground_state(gpaw):
    spec = D.build(hydrogen(), gpaw, None, xc="PBE", grid_spacing_A=0.22,
                   energy_reference="absolute", energy_min_eV=-15.0, energy_max_eV=0.0,
                   energy_step_eV=0.01, width_eV=0.05, n_bands=4)
    outcome = D.execute(spec, gpaw, timeout_s=600)
    assert outcome.ok, outcome.reason
    level = float(outcome.arrays["eigenvalues"].data[0, 0, 0])
    independent = ground_state(specs.build(hydrogen(), gpaw, None, xc="PBE",
                                           grid_spacing_A=0.22,
                                           observables=["energy", "eigenvalues",
                                                        "occupations"]), gpaw, timeout_s=600)
    assert independent.converged
    reference = float(np.asarray(independent.arrays["eigenvalues"].data)[0, 0, 0])
    assert level == pytest.approx(reference, abs=1e-4)
    assert level == pytest.approx(-10.3, abs=0.3)
    checks = outcome.results["dos"].value["checks"]
    assert checks["integral_to_fermi_level_e"] == pytest.approx(2.0, abs=1e-6)
    assert checks["recompute_max_relative_error"] < 1e-12
    energies = outcome.arrays["energies"].data
    pdos = outcome.arrays["pdos"].data[0]
    assert energies[int(np.argmax(pdos))] == pytest.approx(level, abs=0.01)


def test_si_gaussian_counts_electrons_and_has_a_gap(gpaw):
    outcome = D.execute(D.build(silicon(), gpaw, None, **SI, energy_step_eV=0.02,
                                width_eV=0.1), gpaw, timeout_s=900)
    assert outcome.ok, outcome.reason
    summary = outcome.results["dos"].value
    checks = summary["checks"]
    assert checks["integral_to_fermi_level_e"] == pytest.approx(8.0, abs=1e-4)
    bands = outcome.arrays["eigenvalues"].data[0] - summary["fermi_level_eV"]
    gap = bands[:, 4:].min() - bands[:, :4].max()
    assert 0.5 < gap < 0.9
    energies = outcome.arrays["energies"].data
    total = outcome.arrays["dos_total"].data
    assert total[int(np.argmin(np.abs(energies)))] < 1e-3
    integrals = checks["projection_integrals_states"]
    assert integrals["Si s"] > 1.0 and integrals["Si p"] > 1.0


def test_si_tetrahedron_counts_electrons(gpaw):
    outcome = D.execute(D.build(silicon(), gpaw, None, **SI, broadening="tetrahedron",
                                energy_step_eV=0.02), gpaw, timeout_s=900)
    assert outcome.ok, outcome.reason
    checks = outcome.results["dos"].value["checks"]
    assert checks["integral_to_fermi_level_e"] == pytest.approx(8.0, abs=0.05)
    assert checks["recompute_max_relative_error"] < 1e-6


def test_h_atom_spin_channels(gpaw):
    atom = Structure(np.array([1]), np.array([[3.0, 3.0, 3.0]]),
                     Cell(np.eye(3) * 6.0, (False, False, False)))
    spec = D.build(atom, gpaw, None, xc="PBE", grid_spacing_A=0.22, n_bands=3,
                   energy_min_eV=-12.0, energy_max_eV=2.0, width_eV=0.05)
    assert spec.spin_channels == "resolved"
    outcome = D.execute(spec, gpaw, timeout_s=600)
    assert outcome.ok, outcome.reason
    energies = outcome.arrays["energies"].data
    spin = outcome.arrays["dos_spin"].data
    below = energies <= 0.0
    up = float(np.trapezoid(spin[0][below], dx=spec.energy_step_eV))
    down = float(np.trapezoid(spin[1][below], dx=spec.energy_step_eV))
    assert up == pytest.approx(1.0, abs=1e-3)
    assert down == pytest.approx(0.0, abs=1e-3)


def test_dos_of_a_real_relaxation(gpaw, tmp_path):
    from materia.desktop_ui.service import Service

    service = Service(recovery_dir=str(tmp_path / "recovery"))
    s = hydrogen()
    s.positions = s.positions + np.array([[0, 0, -0.03], [0, 0, 0.03]])
    service.project.add_structure(s)
    relax = service.dft_relax({"xc": "PBE", "grid_spacing_A": 0.22, "fmax_eV_A": 0.02,
                               "forces_tol_eV_A": 0.01,
                               "observables": ["energy", "forces"]}, background=False,
                              timeout_s=600)
    assert relax["ok"] and relax["applied"], relax
    out = service.dft_dos_run({"n_bands": 4, "energy_min_eV": -8.0, "energy_max_eV": 4.0,
                               "width_eV": 0.05},
                              {"kind": "relaxation", "run_id": relax["run_id"]},
                              background=False, timeout_s=600)
    assert out["ok"], out
    assert out["run_state"] == "current"
    assert abs(out["checks"]["source_energy_difference_eV"]) < 1e-6
    service.shutdown(grace_s=2.0)
