"""MACE-MP-0 through Materia, against Quantum ESPRESSO PBE on silicon.

Skipped as BLOCKED without the materia-ml environment.  Lattice constants and
forces are checked; the Gamma optical phonon is recorded as soft, a known
property of universal machine-learned potentials, so the test asserts the
softening rather than agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import phonons as P
from materia.solvers.ml_driver import MLPotential, find_python

pytestmark = pytest.mark.validation
QE_PBE_A0 = 5.4707
DFPT_GAMMA_THZ = 15.134


def silicon(a, z=0.0):
    return Structure([14, 14], np.array([[0, 0, z], [a / 4] * 3]),
                     Cell(np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]),
                          (True, True, True)))


@pytest.fixture(scope="module")
def mace():
    if find_python() is None:
        pytest.skip("BLOCKED: no materia-ml interpreter with MACE is installed.")
    potential = MLPotential("small")
    yield potential
    potential.close()


def test_forces_lattice_and_soft_phonons(mace):
    energy, forces = mace.energy_and_forces(silicon(5.43, 0.05))
    h = 1e-4
    fd = -(mace.energy(silicon(5.43, 0.05 + h)) - mace.energy(silicon(5.43, 0.05 - h))) / (2 * h)
    assert fd == pytest.approx(forces[0, 2], abs=1e-6)
    a0 = minimize_scalar(lambda a: mace.energy(silicon(a)), bounds=(5.3, 5.6),
                         method="bounded", options={"xatol": 1e-5}).x
    assert a0 == pytest.approx(QE_PBE_A0, rel=0.005)
    modes = P.harmonic_analysis(silicon(a0), mace, P.PhononSettings(stencil=4))
    optical = float(np.sort(modes.frequencies_THz)[-1])
    assert 0.6 * DFPT_GAMMA_THZ < optical < 0.9 * DFPT_GAMMA_THZ
    assert mace.info["mace_version"] and mace.info["model_sha256"]
    assert modes.provenance.model == "phonons/finite-displacement[ml/mace-mp-0-small]"


def test_refuses_slabs(mace):
    slab = Structure([14], np.zeros((1, 3)), Cell(np.eye(3) * 10, (True, True, False)))
    ok, why = mace.supports(slab)
    assert not ok and "vacuum" in why
