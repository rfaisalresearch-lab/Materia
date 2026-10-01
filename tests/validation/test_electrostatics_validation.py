"""Physical validation of point-charge electrostatics.

Each case compares a computed electrostatic energy with an analytic value or a
published lattice sum.  These establish that the Ewald, slab-corrected Ewald
and direct summations reproduce the Coulomb energy of the point-charge model.
They say nothing about whether a point-charge model describes a real material;
real ionic charges are smaller than formal ones and real ions are polarisable.

Reference values
----------------
Madelung constants referred to the nearest-neighbour distance, for unit
charges:

* rock salt 1.747564594633: J. M. Borwein, M. L. Glasser, R. C. McPhedran,
  J. G. Wan and I. J. Zucker, *Lattice Sums Then and Now*, Cambridge
  University Press (2013), chapter 1.
* caesium chloride 1.762675 and zinc blende 1.638055: C. Kittel,
  *Introduction to Solid State Physics*, 8th ed., chapter 3; M. P. Tosi,
  Solid State Physics 16 (1964) 1.
* square lattice of alternating charges (one NaCl(100) plane) 1.615542626712:
  Borwein et al. (2013), chapter 1.
* simple cubic lattice of point charges in a uniform neutralising background,
  referred to the cell edge L, alpha = 2.8373 in E = -q^2 alpha / (2 L):
  G. Makov and M. C. Payne, Phys. Rev. B 51 (1995) 4014.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.core_model.units import COULOMB_K_EV_A
from materia.physics import electrostatics as es
from materia.physics.electrostatics import ChargeModel, EwaldSettings

pytestmark = pytest.mark.validation

K = COULOMB_K_EV_A
FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def madelung(energy_eV: float, pairs: float, r_nn: float) -> float:
    return -energy_eV / pairs * r_nn / K


def test_coulomb_constant_matches_codata():
    assert K == pytest.approx(14.3996454784, rel=1e-9)


def test_rock_salt_madelung_constant():
    a = 5.64
    s = Structure([11] * 4 + [17] * 4, np.vstack([FCC * a, (FCC + [0.5, 0, 0]) * a]),
                  Cell.cubic(a))
    out = es.compute(s, ChargeModel.per_element({"Na": 1, "Cl": -1}),
                     EwaldSettings(accuracy=1e-10))
    assert out.converged
    assert madelung(out.energy_eV, 4, a / 2) == pytest.approx(1.747564594633, rel=1e-8)
    assert out.site_potential_V[0] == pytest.approx(-1.747564594633 * K / (a / 2), rel=1e-8)


def test_rock_salt_in_the_primitive_cell_and_a_supercell_agree():
    a = 5.64
    primitive = Cell(np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]))
    s1 = Structure([11, 17], [[0, 0, 0], [a / 2, 0, 0]], primitive)
    s2 = Structure([11] * 4 + [17] * 4, np.vstack([FCC * a, (FCC + [0.5, 0, 0]) * a]),
                   Cell.cubic(a)).repeat(2, 2, 2)
    model = ChargeModel.per_element({"Na": 1, "Cl": -1})
    e1 = es.compute(s1, model, EwaldSettings(accuracy=1e-10)).energy_eV
    e2 = es.compute(s2, model, EwaldSettings(accuracy=1e-10)).energy_eV
    assert e2 / 32 == pytest.approx(e1, rel=1e-9)
    assert madelung(e1, 1, a / 2) == pytest.approx(1.747564594633, rel=1e-8)


def test_caesium_chloride_madelung_constant():
    a = 4.12
    s = Structure([55, 17], [[0, 0, 0], [a / 2] * 3], Cell.cubic(a))
    out = es.compute(s, ChargeModel.per_element({"Cs": 1, "Cl": -1}),
                     EwaldSettings(accuracy=1e-10))
    assert madelung(out.energy_eV, 1, a * math.sqrt(3) / 2) == pytest.approx(1.762675,
                                                                            abs=1e-6)


def test_zinc_blende_madelung_constant():
    a = 5.4093
    s = Structure([30] * 4 + [16] * 4, np.vstack([FCC * a, (FCC + 0.25) * a]),
                  Cell.cubic(a))
    out = es.compute(s, ChargeModel.per_element({"Zn": 1, "S": -1}),
                     EwaldSettings(accuracy=1e-10))
    assert madelung(out.energy_eV, 4, a * math.sqrt(3) / 4) == pytest.approx(1.638055,
                                                                            abs=1e-6)


def test_madelung_energy_scales_with_charge_squared():
    a = 4.21
    s = Structure([12] * 4 + [8] * 4, np.vstack([FCC * a, (FCC + [0.5, 0, 0]) * a]),
                  Cell.cubic(a))
    s.formal_charges[:4] = 2.0
    s.formal_charges[4:] = -2.0
    out = es.compute(s, ChargeModel.formal_point_ion(), EwaldSettings(accuracy=1e-10))
    assert madelung(out.energy_eV, 4 * 4, a / 2) == pytest.approx(1.747564594633, rel=1e-8)


def test_square_lattice_of_alternating_charges():
    d = 2.82
    s = Structure([11, 17, 17, 11], [[0, 0, 0], [d, 0, 0], [0, d, 0], [d, d, 0]],
                  Cell(np.diag([2 * d, 2 * d, 30.0]), (True, True, False)))
    out = es.compute(s, ChargeModel.per_element({"Na": 1, "Cl": -1}),
                     EwaldSettings(accuracy=1e-10))
    assert out.converged
    assert madelung(out.energy_eV, 2, d) == pytest.approx(1.615542626712, rel=1e-8)


def test_slab_layer_energy_converges_to_the_bulk_layer_energy():
    """Adding one NaCl(100) plane to a thick slab adds one bulk plane's energy.

    The surface contribution of a point-charge rock-salt slab decays
    exponentially with depth, so the energy difference between slabs of n+1
    and n planes approaches the bulk energy of one plane (two ion pairs in
    this cell).  This ties the slab method to the bulk method.
    """
    a = 5.64
    d = a / 2
    model = ChargeModel.per_element({"Na": 1, "Cl": -1})

    def slab(n_planes):
        numbers, positions = [], []
        for k in range(n_planes):
            for (x, y), (first, second) in (((0, 0), (11, 17)), ((d, 0), (17, 11)),
                                            ((0, d), (17, 11)), ((d, d), (11, 17))):
                numbers.append(first if k % 2 == 0 else second)
                positions.append([x, y, k * d])
        return Structure(numbers, positions,
                         Cell(np.diag([a, a, 60.0]), (True, True, False)))

    energies = [es.compute(slab(n), model, EwaldSettings(accuracy=1e-10)).energy_eV
                for n in (4, 5, 6)]
    per_plane_bulk = -1.747564594633 * K / d * 2
    assert energies[2] - energies[1] == pytest.approx(per_plane_bulk, rel=1e-7)
    assert energies[1] - energies[0] == pytest.approx(per_plane_bulk, rel=1e-6)


def test_point_charge_in_a_neutralising_background():
    L = 7.3
    s = Structure([1], [[1.1, 2.2, 3.3]], Cell.cubic(L))
    out = es.compute(s, ChargeModel.per_element({"H": 1}),
                     EwaldSettings(background="uniform", accuracy=1e-10))
    assert -out.energy_eV * 2 * L / K == pytest.approx(2.8373, abs=5e-5)


def test_isolated_ion_pair_is_coulomb_law():
    r = 2.36
    s = Structure([11, 17], [[0, 0, 0], [r, 0, 0]], Cell.none())
    out = es.compute(s, ChargeModel.per_element({"Na": 1, "Cl": -1}))
    assert out.energy_eV == pytest.approx(-K / r, rel=1e-14)
    assert out.forces_eV_A[0, 0] == pytest.approx(K / r ** 2, rel=1e-14)


def test_neutral_cluster_far_from_its_images_matches_the_isolated_sum():
    """A neutral, nonpolar cluster in a large cubic box approaches the isolated
    value as the box grows, with images interacting through the quadrupole."""
    rng = np.random.default_rng(7)
    a = 2.8
    pos = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                   dtype=float) * a
    pos += rng.normal(0.0, 0.05, pos.shape)
    numbers = [11 if (i + j + k) % 2 == 0 else 17
               for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    model = ChargeModel.per_element({"Na": 1, "Cl": -1})
    isolated = es.compute(Structure(numbers, pos, Cell.none()), model).energy_eV
    errors = []
    for box in (30.0, 60.0):
        periodic = es.compute(Structure(numbers, pos + box / 2 - a / 2, Cell.cubic(box)),
                              model, EwaldSettings(surrounding="vacuum")).energy_eV
        errors.append(abs(periodic - isolated))
    assert errors[1] < errors[0]
    assert errors[1] < 1e-3
