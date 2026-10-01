"""Energetics of the Mishin Cu EAM against the values published with the potential.

Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter and J. D. Kress,
Phys. Rev. B 63 (2001) 224106, Table III: a0 3.615 A, E_coh 3.54 eV, relaxed
vacancy formation 1.27 eV, surface energies 1239 (111), 1345 (100) and 1475
(110) mJ/m^2.  The file is downloaded from the LAMMPS repository.
"""

from __future__ import annotations

import pytest

from materia.materials.loader import load
from materia.physics import eam
from materia.physics import energetics as E
from materia.physics.potentials import UnsupportedSystem
from materia.structure_builder.lattice import bulk

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def mishin():
    try:
        return eam.load_lammps("Cu_mishin1.eam.alloy",
                               "Y. Mishin et al., Phys. Rev. B 63 (2001) 224106")
    except UnsupportedSystem as exc:
        pytest.skip(f"BLOCKED: {exc}")


def test_header_correction_is_recorded(mishin):
    assert any("atomic number 1 for Cu" in note for note in mishin.identity.notes)


def test_bulk_and_vacancy(mishin):
    copper = load("copper")
    ref = E.equilibrium_bulk(bulk(copper), mishin)
    assert ref["volume_per_atom_A3"] ** (1 / 3) * 4 ** (1 / 3) == pytest.approx(3.615, abs=2e-3)
    assert -ref["energy_per_atom_eV"] == pytest.approx(3.54, abs=0.005)
    vacancy = E.vacancy_formation(bulk(copper), mishin, supercells=((4, 4, 4), (5, 5, 5)))
    assert vacancy.convergence.converged
    assert vacancy.value == pytest.approx(1.27, abs=0.01)


@pytest.mark.parametrize("miller,expected", [((1, 1, 1), 1.239), ((1, 0, 0), 1.345),
                                             ((1, 1, 0), 1.475)])
def test_surface_energies(mishin, miller, expected):
    out = E.surface_energy(load("copper"), mishin, miller, layers=(9, 12))
    assert out.convergence.converged
    assert out.value == pytest.approx(expected, abs=0.003)
