"""p-wave tip images and energy-resolved LDOS from real GPAW wavefunctions.

Skipped as BLOCKED without GPAW.  A hydrogen atom in a slab cell:

* Chen's p-wave tip images an s orbital as a ring with zero on the axis, whose
  radius at height z maximises (rho^2/r^2) exp(-2 kappa r) for the decay
  constant kappa measured from the s-wave LDOS above the atom.
* The point spectrum above the atom peaks where the broadened density of
  states computed independently from the eigenvalues peaks.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.experiments.dft import ldos as L
from materia.experiments.dft import stm_spectroscopy as S
from materia.solvers.gpaw_driver import discover

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def hydrogen():
    environment = discover()
    if not environment.operational:
        pytest.skip(f"BLOCKED: GPAW is not usable here. {environment.blocking_reason()}")
    atom = Structure([1], np.array([[4.0, 4.0, 5.0]]),
                     Cell(np.diag([8.0, 8.0, 14.0]), (True, True, False)))
    base = L.build(atom, environment, None, xc="PBE", grid_spacing_A=0.18, kpoints=[1, 1, 1],
                   occupations="fixed", smearing_eV=0.0, energy_min_eV=-10.0,
                   energy_max_eV=0.0, n_bands=4)
    spec = S.build(base, tip="p", energies_eV=list(np.linspace(-9, -3, 13)), broadening_eV=0.3)
    out = S.execute(spec, environment, timeout_s=1800)
    assert out.status == "complete", out.reason
    return spec, out


def test_p_tip_ring(hydrogen):
    spec, out = hydrogen
    ldos, ptip = out.arrays["ldos"].data, out.arrays["ptip"].data
    z = np.arange(ldos.shape[2]) * 14.0 / ldos.shape[2]
    column = ldos[ldos.shape[0] // 2, ldos.shape[1] // 2]
    window = (z > 7.0) & (z < 9.0)
    kappa = -np.polyfit(z[window], np.log(column[window]), 1)[0] / 2
    k = int(np.argmin(np.abs(z - 8.0)))
    plane = ptip[:, :, k]
    n = plane.shape[0]
    x = np.arange(n) * 8.0 / n - 4.0
    rho = np.hypot(*np.meshgrid(x, x, indexing="ij"))
    edges = np.arange(0, 3.5, 0.18)
    profile = [plane[(rho >= a) & (rho < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
    ring = edges[int(np.argmax(profile))] + 0.09
    height = z[k] - 5.0
    predicted = minimize_scalar(
        lambda r: -(r * r / (r * r + height * height)) * np.exp(-2 * kappa * np.hypot(r, height)),
        bounds=(0.01, 4.0), method="bounded").x
    assert ring == pytest.approx(predicted, abs=0.2)
    assert plane[n // 2, n // 2] < 1e-6 * plane.max()


def test_point_spectrum_follows_the_density_of_states(hydrogen):
    spec, out = hydrogen
    stack = out.arrays["stack"].data
    spectrum = S.point_spectrum(stack, np.diag([8.0, 8.0, 14.0]), spec.energies_eV,
                                4.0, 4.0, 7.0)["ldos_per_eV"]
    dos = out.results["dos"].value
    assert int(np.argmax(spectrum)) == int(np.argmax(dos))
    integrals = out.results["dos"].extra["stack_integrals"]
    peak = int(np.argmax(dos))
    assert integrals[peak] == pytest.approx(dos[peak], rel=0.1)
