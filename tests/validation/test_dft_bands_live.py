"""Real GPAW band structures, checked against independent quantities.

These cases run only when GPAW and its PAW datasets are installed.  Without
them each case is skipped with the reason ``BLOCKED``: the validation has not
been performed, and a skip is not a pass.

What is checked, and against what
---------------------------------
Lightweight setup throughout: PBE, plane waves at 300 eV, a 4x4x4
Gamma-centred ground-state grid, fixed occupations, a = 5.43 A.  The
tolerances below are for that setup, not for converged PBE.

* Si on the standard FCC path G-X-W-K-G-L: the Kohn-Sham gap along the path is
  indirect, the valence top is at Gamma, the conduction bottom lies on the
  Gamma to X segment between 0.7 and 0.95 of the way to X (published PBE
  places it near 0.85), the indirect gap is between 0.45 and 0.75 eV (PBE
  reference about 0.6 eV) and the direct gap at Gamma between 2.35 and
  2.75 eV (PBE reference about 2.55 eV).
* The same Si endpoints against an independent GPAW ground-state run in its own
  process: at Gamma and at X the occupied eigenvalues agree to 2 meV, the
  lowest conduction band to 20 meV (the independent run converges only the
  occupied bands), and the Fermi level to 1 meV.
* bcc Fe, spin-polarised: both channels are returned, a band crosses the Fermi
  level in each (a metal), and the spin-down d bands at Gamma lie 1 to 3 eV
  above the spin-up ones (the exchange splitting; PBE gives about 2 eV).
* A graphene sheet as a 2D slab (real-space grid, 0.18 A spacing, 9x9x1 grid,
  12 A vacuum): the path is G-M-K-G in the plane, and the two
  pi bands touch at K within 30 meV of each other and of the Fermi level.
* A band structure taken from a real converged relaxation reproduces the
  relaxation's final energy to 1 meV and is current once the relaxation is
  applied; its gap stays within 50 meV of the unrelaxed crystal's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.experiments.dft import bands as B
from materia.experiments.dft import spec as specs
from materia.experiments.dft.run import execute as ground_state
from materia.solvers.gpaw_driver import discover

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def gpaw():
    environment = discover()
    if not environment.operational:
        pytest.skip("BLOCKED: live GPAW band structure was not run because GPAW is not usable "
                    f"here. {environment.blocking_reason()} {environment.install_hint()}")
    return environment


def silicon(a=5.43, shift=None):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    positions = np.array([[0, 0, 0], [a / 4] * 3], dtype=float)
    if shift is not None:
        positions[1] += shift
    return Structure(np.array([14, 14]), positions, Cell(cell, (True, True, True)))


SI = {"xc": "PBE", "cutoff_eV": 300.0, "kpoints": [4, 4, 4], "kpoints_gamma_centered": True,
      "occupations": "fixed", "smearing_eV": 0.0}


@pytest.fixture(scope="module")
def si_bands(gpaw):
    spec = B.build(silicon(), gpaw, None, **SI, path="GXWKGL", n_bands=8)
    outcome = B.execute(spec, gpaw, timeout_s=900)
    assert outcome.ok, outcome.reason
    return spec, outcome


def test_si_indirect_gap_on_the_standard_path(si_bands):
    spec, outcome = si_bands
    assert spec.path_origin["lattice"] == "FCC" and spec.path_origin["generator"] == "ase"
    summary = outcome.results["bands"].value
    edges = summary["band_edges"]
    assert edges["gap_kind"] == "indirect" and not edges["metallic_on_path"]
    assert edges["vbm"]["label"] == "G"
    labels = spec.labels()
    gamma, x_point = labels.index("G"), labels.index("X")
    cbm = edges["cbm"]["index"]
    assert gamma < cbm < x_point
    fraction = (cbm - gamma) / (x_point - gamma)
    assert 0.7 <= fraction <= 0.95
    assert 0.45 < edges["gap_eV"] < 0.75
    direct = edges["per_spin"][0]
    assert direct["direct_gap_at"]["label"] == "G"
    assert 2.35 < direct["direct_gap_eV"] < 2.75
    assert direct["direct_gap_eV"] > edges["gap_eV"]
    checks = summary["checks"]
    assert checks["distance_max_relative_error"] < 1e-10
    assert checks["nscf_symmetry_operations"] == 1
    assert abs(checks["fermi_level_nscf_eV"] - checks["fermi_level_scf_eV"]) < 1e-9
    eigen = outcome.arrays["eigenvalues"].data
    assert eigen.shape == (1, spec.n_kpoints, 8) and np.all(np.isfinite(eigen))
    assert np.allclose(outcome.arrays["kpoints_frac"].data, spec.kpoints_frac, atol=1e-12)


def _x_star(kpoints):
    star = {(0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.0)}
    for index, k in enumerate(np.asarray(kpoints, dtype=float)):
        folded = tuple(round(float(v) % 1.0, 9) for v in k)
        if folded in star:
            return index
    return None


def test_si_endpoints_match_an_independent_ground_state(gpaw, si_bands):
    spec, outcome = si_bands
    independent = ground_state(specs.build(silicon(), gpaw, None, **SI, n_bands=8,
                                           observables=["energy", "eigenvalues", "fermi_level"]),
                               gpaw, timeout_s=900)
    assert independent.converged, independent.reason
    stored = independent.arrays["eigenvalues"]
    ibz = np.asarray(stored.meta["ibz_kpoints"], dtype=float)
    reference = np.asarray(stored.data, dtype=float)[0]
    gamma_ibz = int(np.argmin(np.linalg.norm(ibz, axis=1)))
    assert np.linalg.norm(ibz[gamma_ibz]) < 1e-12
    x_ibz = _x_star(ibz)
    assert x_ibz is not None, ibz
    path = outcome.arrays["eigenvalues"].data[0]
    labels = spec.labels()
    for name, k_path, k_ibz in (("G", labels.index("G"), gamma_ibz),
                                ("X", labels.index("X"), x_ibz)):
        assert np.allclose(path[k_path, :4], reference[k_ibz, :4], atol=2e-3), name
        assert path[k_path, 4] == pytest.approx(reference[k_ibz, 4], abs=2e-2), name
    fermi = independent.results["fermi_level"].value
    assert outcome.results["bands"].value["fermi_level_eV"] == pytest.approx(fermi, abs=1e-3)


def test_fe_spin_resolved_bands(gpaw):
    a = 2.87
    fe = Structure(np.array([26]), np.zeros((1, 3)),
                   Cell(0.5 * a * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
                        (True, True, True)))
    spec = B.build(fe, gpaw, None, xc="PBE", cutoff_eV=350.0, kpoints=[8, 8, 8],
                   occupations="fermi-dirac", smearing_eV=0.1, spin_polarized=True,
                   initial_magnetic_moments_muB=[2.5], path="GHNG",
                   sampling_density_per_invA=6.0, n_bands=10)
    report = B.check(spec, gpaw)
    assert report.ok, report.blocking
    outcome = B.execute(spec, gpaw, timeout_s=1800)
    assert outcome.ok, outcome.reason
    eigen = outcome.arrays["eigenvalues"].data
    fermi = outcome.results["bands"].value["fermi_level_eV"]
    assert eigen.shape[0] == 2
    edges = outcome.results["bands"].value["band_edges"]
    assert edges["metallic_on_path"]
    assert all(p["metallic_on_path"] for p in edges["per_spin"])
    gamma = spec.labels().index("G")
    up, down = eigen[0, gamma] - fermi, eigen[1, gamma] - fermi
    d_up, d_down = np.sort(up)[1:6].mean(), np.sort(down)[1:6].mean()
    assert 1.0 < d_down - d_up < 3.0
    assert not np.allclose(eigen[0], eigen[1], atol=0.1)


def test_graphene_slab_bands_touch_at_k(gpaw):
    a = 2.46
    vacuum = 12.0
    cell = np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, vacuum]])
    positions = np.array([[0.0, 0.0, vacuum / 2], [0.0, a / math.sqrt(3), vacuum / 2]])
    sheet = Structure(np.array([6, 6]), positions, Cell(cell, (True, True, False)))
    spec = B.build(sheet, gpaw, None, xc="PBE", grid_spacing_A=0.18, kpoints=[9, 9, 1],
                   kpoints_gamma_centered=True, occupations="fermi-dirac", smearing_eV=0.05,
                   n_bands=6)
    assert spec.ground_state.boundary == "slab"
    assert B.path_text(spec.path) == "GMKG" and spec.path_origin["lattice"] == "HEX2D"
    assert all(abs(k[2]) < 1e-12 for k in spec.kpoints_frac)
    report = B.check(spec, gpaw)
    assert report.ok, report.blocking
    outcome = B.execute(spec, gpaw, timeout_s=900)
    assert outcome.ok, outcome.reason
    eigen = outcome.arrays["eigenvalues"].data[0]
    fermi = outcome.results["bands"].value["fermi_level_eV"]
    k = spec.labels().index("K")
    relative = np.sort(eigen[k] - fermi)
    nearest = relative[np.argsort(np.abs(relative))[:2]]
    assert abs(nearest[0] - nearest[1]) < 0.03
    assert np.all(np.abs(nearest) < 0.03)


def test_bands_of_a_real_relaxation(gpaw, si_bands, tmp_path):
    from materia.desktop_ui.service import Service

    service = Service(recovery_dir=str(tmp_path / "recovery"))
    service.project.add_structure(silicon(shift=np.array([0.04, -0.03, 0.02])))
    relax = service.dft_relax({**SI, "symmetry": "off", "fmax_eV_A": 0.02,
                               "forces_tol_eV_A": 0.005,
                               "observables": ["energy", "forces"]}, background=False,
                              timeout_s=1800)
    assert relax["ok"] and relax["status"] == "converged" and relax["applied"], relax
    out = service.dft_bands_run({"path": "GXWKGL", "n_bands": 8},
                                {"kind": "relaxation", "run_id": relax["run_id"]},
                                background=False, timeout_s=1800)
    assert out["ok"], out
    assert out["run_state"] == "current"
    assert out["settings"]["scf_symmetry"] == "off"
    assert abs(out["checks"]["source_energy_difference_eV"]) < 1e-3
    unrelaxed = si_bands[1].results["bands"].value["band_edges"]["gap_eV"]
    assert out["band_edges"]["gap_eV"] == pytest.approx(unrelaxed, abs=0.05)
    service.shutdown(grace_s=2.0)
