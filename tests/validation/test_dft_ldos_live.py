"""Real GPAW LDOS and Tersoff-Hamann images, checked against independent references.

These cases run only when GPAW and its PAW datasets are installed; otherwise
each skips with the reason ``BLOCKED``, which is not a pass.

What is checked, and against what
---------------------------------
* Graphene (PBE, 0.20 A grid, 6x6x1, window -1 to 0 eV): Materia's map
  against ASE's ``ase.dft.stm.STM`` evaluated in a separate GPAW process on the
  same job, point by point to 1e-4 of the maximum; constant-current heights
  against ASE's ``scan`` to 1e-3 A; constant-height values against direct
  linear interpolation of ASE's map to 1e-4. The two processes each converge
  their own self-consistent field, so in general they agree only to the SCF
  criterion (density 1e-4 per electron). Both run single-threaded, as the
  worker does; measured here the maps were bit-identical and the heights agreed
  to 2e-15 A. A multithreaded reference differed by up to 8.6e-4.
* H2 (fixed occupations, window -20 to 0 eV, covering every occupied state):
  the map equals GPAW's pseudo valence density of an independent run to 1e-4
  of its maximum, and holds exactly 2 states. Hydrogen has no core, so the
  pseudo density contains nothing but these states; for heavier atoms GPAW
  adds a smooth pseudo-core charge that is not a Kohn-Sham state.
* Spin-polarised H atom, window -10 to 0 eV: one state, all in spin up.
* An LDOS taken from a real converged H2 relaxation is current once applied.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import ldos as L
from materia.solvers.gpaw_driver import discover

pytestmark = pytest.mark.validation

REFERENCE = Path(__file__).resolve().parents[1] / "support" / "gpaw_ldos_reference.py"


@pytest.fixture(scope="module")
def gpaw():
    environment = discover()
    if not environment.operational:
        pytest.skip("BLOCKED: live GPAW LDOS was not run because GPAW is not usable here. "
                    f"{environment.blocking_reason()} {environment.install_hint()}")
    return environment


def reference(environment, spec, tmp_path, *extra):
    job = tmp_path / "job.json"
    job.write_text(json.dumps(L.worker_job(spec)))
    out = tmp_path / "reference.npz"
    subprocess.run([environment.interpreter, str(REFERENCE), str(job), str(out),
                    *[str(v) for v in extra]], check=True, timeout=1800,
                   capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    return np.load(out)


def graphene():
    a = 2.46
    return Structure(np.array([6, 6]), np.array([[0, 0, 5.0], [0, a / math.sqrt(3), 5.0]]),
                     Cell(np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0],
                                    [0, 0, 16.0]]), (True, True, False)))


def hydrogen(distance=0.74, box=8.0):
    c = box / 2
    return Structure(np.array([1, 1]), np.array([[c, c, c - distance / 2],
                                                 [c, c, c + distance / 2]]),
                     Cell(np.eye(3) * box, (False, False, False)))


def test_graphene_map_and_images_match_ase(gpaw, tmp_path):
    spec = L.build(graphene(), gpaw, None, xc="PBE", grid_spacing_A=0.2, kpoints=[6, 6, 1],
                   smearing_eV=0.05, energy_min_eV=-1.0, energy_max_eV=0.0, n_bands=8)
    outcome = L.execute(spec, gpaw, timeout_s=1800)
    assert outcome.ok, outcome.reason
    mine = outcome.arrays["ldos"].data
    value = outcome.results["ldos"].value
    assert 0.9 < value["pseudo_norm_ratio"] < 1.0
    cell = np.asarray(spec.ground_state.cell_A)
    positions = np.asarray(spec.ground_state.positions_A)
    height = L.stm_image(mine, cell, positions, "constant-height", height_A=3.0,
                         augmentation_radius_A=value["max_augmentation_radius_A"])
    isovalue = float(height["values"].min())
    current = L.stm_image(mine, cell, positions, "constant-current", isovalue=isovalue,
                          augmentation_radius_A=value["max_augmentation_radius_A"])
    top = height["surface_z_A"]
    z0 = 16.0 - L.FACE_MARGIN_A
    ref = reference(gpaw, spec, tmp_path, isovalue / 2.0, z0, top + 3.0)
    ase_map = 2.0 * ref["ase_ldos"]
    assert mine.shape == ase_map.shape
    assert np.abs(mine - ase_map).max() <= 1e-4 * ase_map.max()
    assert np.allclose(height["values"], 2.0 * ref["interpolated"], rtol=1e-4,
                       atol=1e-9 * mine.max())
    assert np.allclose(current["values"] + top, ref["ase_heights"], rtol=0, atol=1e-3)
    assert current["values"].max() - current["values"].min() > 0.01


def test_full_window_equals_the_pseudo_density(gpaw, tmp_path):
    spec = L.build(hydrogen(), gpaw, None, xc="PBE", grid_spacing_A=0.2,
                   occupations="fixed", smearing_eV=0.0, energy_min_eV=-20.0,
                   energy_max_eV=0.0, n_bands=3)
    outcome = L.execute(spec, gpaw, timeout_s=1800)
    assert outcome.ok, outcome.reason
    assert outcome.results["ldos"].value["states_in_window"] == pytest.approx(2.0, abs=1e-12)
    ref = reference(gpaw, spec, tmp_path)
    mine = outcome.arrays["ldos"].data
    density = ref["pseudo_density"]
    assert mine.shape == density.shape
    assert np.abs(mine - density).max() <= 1e-4 * density.max()


def test_spin_resolved_hydrogen_atom(gpaw):
    atom = Structure(np.array([1]), np.array([[4.0, 4.0, 4.0]]),
                     Cell(np.eye(3) * 8.0, (False, False, False)))
    spec = L.build(atom, gpaw, None, xc="PBE", grid_spacing_A=0.2, occupations="fixed",
                   smearing_eV=0.0, energy_min_eV=-10.0, energy_max_eV=0.0, n_bands=3)
    assert spec.spin_channels == "resolved"
    outcome = L.execute(spec, gpaw, timeout_s=1800)
    assert outcome.ok, outcome.reason
    assert outcome.results["ldos"].value["states_in_window"] == pytest.approx(1.0, abs=1e-12)
    spin = outcome.arrays["ldos_spin"].data
    assert spin[1].max() == 0.0 and spin[0].max() > 0.0
    assert np.array_equal(spin.sum(axis=0), outcome.arrays["ldos"].data)


def test_ldos_of_a_real_relaxation(gpaw, tmp_path):
    from materia.desktop_ui.service import Service

    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(hydrogen(0.80))
    relax = service.dft_relax({"xc": "PBE", "grid_spacing_A": 0.22, "fmax_eV_A": 0.02,
                               "forces_tol_eV_A": 0.01, "observables": ["energy", "forces"]},
                              background=False, timeout_s=1800)
    assert relax["ok"] and relax["applied"], relax
    out = service.dft_ldos_run({"energy_min_eV": -15.0, "energy_max_eV": 0.0, "n_bands": 3},
                               {"kind": "relaxation", "run_id": relax["run_id"]},
                               background=False, timeout_s=1800)
    assert out["ok"], out
    assert out["run_state"] == "current"
    assert out["states_in_window"] == pytest.approx(2.0, abs=1e-12)
    service.shutdown(grace_s=2.0)
