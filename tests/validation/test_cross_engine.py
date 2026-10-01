"""The same physics through different engines driven by Materia.

Silicon, PBE, seven volumes from 0.97 to 1.03 of 5.469 A, third-order
Birch-Murnaghan fits.  GPAW (PAW, plane waves 500 eV, 12^3 k) and Quantum
ESPRESSO (PSlibrary ultrasoft, 45/360 Ry, 8^3 k) are independent codes with
independent pseudopotentials; both are compared with each other and with the
WIEN2k all-electron reference of the Delta project (Lejaeghere et al.,
Science 351 (2016) aad3000: V0 = 20.453 A^3, B0 = 88.5 GPa).  MACE-MP-0, a
machine-learned surrogate of PBE, is held to the volume only; its bulk
modulus is measured about 18 percent soft and asserted to be soft.
Each engine skips as BLOCKED when it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from materia.core_model import Cell, Structure

pytestmark = pytest.mark.validation
EV_A3_GPA = 160.21766208
AE_V0, AE_B0 = 20.453, 88.5
LATTICE = 5.469 * np.linspace(0.97, 1.03, 7)


def birch_murnaghan(volume, e0, v0, b0, b1):
    x = (v0 / volume) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((x - 1) ** 3 * b1 + (x - 1) ** 2 * (6 - 4 * x))


def silicon(a):
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return Structure(np.array([14, 14]), np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(cell, (True, True, True))), abs(np.linalg.det(cell)) / 2


def fit(energies):
    volumes = np.array([silicon(a)[1] for a in LATTICE])
    energies = np.asarray(energies)
    params = curve_fit(birch_murnaghan, volumes, energies,
                       p0=(energies.min(), volumes[3], 0.5, 4.5))[0]
    return params[1], params[2] * EV_A3_GPA


@pytest.fixture(scope="module")
def gpaw_eos():
    from materia.experiments.dft import spec as specs
    from materia.experiments.dft.run import execute
    from materia.solvers.gpaw_driver import discover

    environment = discover()
    if not environment.operational:
        pytest.skip("BLOCKED: GPAW is not usable here.")
    energies = []
    for a in LATTICE:
        structure, _ = silicon(a)
        spec = specs.build(structure, environment, None, xc="PBE", kpoints_gamma_centered=True,
                           observables=["energy"], cutoff_eV=500.0, kpoints=[12, 12, 12],
                           occupations="fermi-dirac", smearing_eV=0.01)
        outcome = execute(spec, environment, timeout_s=3600)
        assert outcome.converged, outcome.reason
        energies.append(outcome.results["energy"].value / 2)
    return fit(energies)


@pytest.fixture(scope="module")
def qe_eos():
    from materia.physics import espresso as E

    if E.find_pw() is None:
        pytest.skip("BLOCKED: Quantum ESPRESSO is not installed.")
    potential = E.EspressoPotential(ecutwfc_Ry=45, ecutrho_Ry=360, kpts=(8, 8, 8),
                                    smearing="gaussian", degauss_Ry=0.001)
    return fit([potential.energy(silicon(a)[0]) / 2 for a in LATTICE])


def test_gpaw_and_quantum_espresso_agree(gpaw_eos, qe_eos):
    assert gpaw_eos[0] == pytest.approx(qe_eos[0], rel=0.005)
    assert gpaw_eos[1] == pytest.approx(qe_eos[1], rel=0.01)
    for v0, b0 in (gpaw_eos, qe_eos):
        assert v0 == pytest.approx(AE_V0, rel=0.006)
        assert b0 == pytest.approx(AE_B0, rel=0.02)


def test_mace_volume_and_soft_bulk_modulus(qe_eos):
    from materia.solvers.ml_driver import MLPotential, find_python

    if find_python() is None:
        pytest.skip("BLOCKED: no materia-ml interpreter.")
    model = MLPotential("small")
    try:
        v0, b0 = fit([model.energy(silicon(a)[0]) / 2 for a in LATTICE])
    finally:
        model.close()
    assert v0 == pytest.approx(qe_eos[0], rel=0.01)
    assert b0 < 0.9 * qe_eos[1]
