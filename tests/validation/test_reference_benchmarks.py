"""Benchmarks of Materia's own numbers against independent references.

Each case compares a value Materia computes or ships with a value obtained
by a route that shares no code with it: a separately maintained data table
(SciPy's CODATA constants, ASE's IUPAC, Cordero and Alvarez tables), an
independent implementation of the same model (ASE's neighbour list, a
direct transcription of Vogl's k-space Hamiltonian), or the published
properties of a model (Stillinger-Weber elastic constants).  The tolerance
of each case is stated with its reason.  None of these cases says anything
about agreement with experiment; docs/ACCURACY_BENCHMARKS.md keeps that
separate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.core_model import Cell, Structure
from materia.core_model import units
from materia.elements.data import ELEMENTS
from materia.physics.neighbors import neighbor_list
from materia.physics.potentials import LennardJones, StillingerWeber
from materia.solvers.classical import ClassicalSolver
from materia.solvers.tight_binding import TightBinding

pytestmark = pytest.mark.validation

EV_A3_TO_GPA = 160.21766208
DIAMOND = np.array([[0, 0, 0], [.25, .25, .25], [0, .5, .5], [.25, .75, .75],
                    [.5, 0, .5], [.75, .25, .75], [.5, .5, 0], [.75, .75, .25]])
FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def cubic(basis, a, n, z, strain=None):
    frac = np.concatenate([basis + np.array([i, j, k]) for i in range(n)
                           for j in range(n) for k in range(n)]) / n
    cell = np.eye(3) * a * n
    if strain is not None:
        cell = cell @ (np.eye(3) + strain).T
    return Structure(np.full(len(frac), z), frac @ cell, Cell(cell, (True, True, True)))


class TestConstantsAndData:
    def test_physical_constants_match_codata(self):
        import scipy.constants as c

        table = c.physical_constants
        pairs = {
            units.ELECTRON_MASS_KG: c.m_e,
            units.ATOMIC_MASS_UNIT_KG: table["atomic mass constant"][0],
            units.BOHR_RADIUS_M: table["Bohr radius"][0],
            units.VACUUM_PERMITTIVITY: c.epsilon_0,
            units.HARTREE_EV: table["Hartree energy in eV"][0],
            units.BOLTZMANN_EV_K: table["Boltzmann constant in eV/K"][0],
            units.COULOMB_K_EV_A: 1.0 / (4.0 * math.pi * c.epsilon_0) * c.e / 1e-10,
            units.HBAR_EV_FS: c.hbar / c.e * 1e15,
        }
        for mine, reference in pairs.items():
            assert mine == pytest.approx(reference, rel=5e-9)

    def test_tunnelling_prefactor_is_sqrt_2m_over_hbar(self):
        import scipy.constants as c

        expected = math.sqrt(2.0 * c.m_e * c.e) / c.hbar * 1e-10
        assert units.DECAY_PREFACTOR_INV_A_SQRT_EV == pytest.approx(expected, rel=5e-9)
        assert units.kappa_inv_angstrom(4.0) == pytest.approx(
            2.0 * units.DECAY_PREFACTOR_INV_A_SQRT_EV, rel=1e-12)

    def test_element_masses_and_radii_match_independent_tables(self):
        from ase.data import atomic_masses_iupac2016, covalent_radii
        from ase.data.vdw_alvarez import vdw_radii

        superheavy_mass_numbers = {"Lr", "Sg", "Rg", "Mc", "Ts"}
        for e in ELEMENTS:
            z = e.number
            if e.symbol not in superheavy_mass_numbers and e.symbol != "Tc":
                assert e.standard_atomic_weight == pytest.approx(
                    float(atomic_masses_iupac2016[z]), abs=0.011), e.symbol
            if e.covalent_radius_A is not None:
                assert e.covalent_radius_A == pytest.approx(float(covalent_radii[z]),
                                                            abs=0.006), e.symbol
            if e.vdw_radius_A is not None and np.isfinite(vdw_radii[z]):
                assert e.vdw_radius_A == pytest.approx(float(vdw_radii[z]), abs=0.006), \
                    e.symbol

    def test_aufbau_exceptions_follow_nist(self):
        from materia.elements.data import BY_SYMBOL

        expected = {"Cr": "3d5 4s1", "Cu": "3d10 4s1", "Nb": "4d4 5s1", "Mo": "4d5 5s1",
                    "Ru": "4d7 5s1", "Rh": "4d8 5s1", "Pd": "4p6 4d10", "Ag": "4d10 5s1",
                    "Pt": "5d9 6s1", "Au": "5d10 6s1", "La": "5d1 6s2",
                    "Gd": "4f7 5s2 5p6 5d1 6s2", "Lr": "7s2 7p1"}
        for symbol, tail in expected.items():
            assert BY_SYMBOL[symbol].electron_configuration.endswith(tail), symbol


    def test_isotopes_add_up_to_the_standard_atomic_weights(self):
        """Abundances sum to one and their weighted mass gives the standard weight.

        Two known exceptions come from IUPAC itself: lead varies in nature and
        207.2 is a conventional value, and the selenium standard weight 78.971(8)
        rests on a newer measurement than its representative composition.
        """
        for e in ELEMENTS:
            natural = [i for i in e.isotopes if i.natural_abundance]
            for isotope in e.isotopes:
                assert isotope.neutrons == isotope.mass_number - e.number
            if not natural:
                continue
            total = sum(i.natural_abundance for i in natural)
            assert total == pytest.approx(1.0, abs=2e-4), e.symbol
            weighted = sum(i.natural_abundance * i.atomic_mass_u for i in natural) / total
            tolerance = {"Pb": 0.02, "Se": 0.015}.get(e.symbol, 0.01)
            assert weighted == pytest.approx(e.standard_atomic_weight, abs=tolerance), \
                e.symbol


class TestNeighbourList:
    def test_matches_ase_in_skewed_cells_with_many_images(self):
        from ase import Atoms
        from ase.neighborlist import neighbor_list as ase_neighbor_list

        rng = np.random.default_rng(7)
        compared = 0
        for trial in range(60):
            cell = np.eye(3) * rng.uniform(2.5, 6) + rng.uniform(-1.5, 1.5, (3, 3))
            if np.linalg.det(cell) < 1:
                continue
            pbc = (True, True, True) if trial % 3 == 0 else \
                tuple(bool(x) for x in rng.integers(0, 2, 3))
            positions = rng.random((int(rng.integers(1, 9)), 3)) @ cell
            cutoff = float(rng.uniform(1.5, 7.0))
            nl = neighbor_list(positions, Cell(cell, pbc), cutoff)
            mine = sorted(zip(nl.i.tolist(), nl.j.tolist(),
                              map(tuple, np.asarray(nl.shifts).astype(int).tolist())))
            i, j, shifts = ase_neighbor_list("ijS", Atoms(positions=positions, cell=cell,
                                                          pbc=pbc), cutoff)
            assert mine == sorted(zip(i.tolist(), j.tolist(), map(tuple, shifts.tolist())))
            compared += 1
        assert compared > 40


def vogl_hamiltonian(k, a, e_sa, e_pa, e_ssa, e_sc, e_pc, e_ssc, v_ss, v_xx, v_xy,
                     v_sapc, v_pasc, v_ssapc, v_passc):
    """Vogl, Hjalmarson and Dow's 10x10 H(k), written out from the paper's g functions."""
    d = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * a / 4
    e = np.exp(1j * d @ k)
    g0 = e.sum() / 4
    g1 = (e[0] + e[1] - e[2] - e[3]) / 4
    g2 = (e[0] - e[1] + e[2] - e[3]) / 4
    g3 = (e[0] - e[1] - e[2] + e[3]) / 4
    sa, sc, xa, ya, za, xc, yc, zc, ssa, ssc = range(10)
    h = np.zeros((10, 10), complex)
    h[sa, sc] = v_ss * g0
    h[sa, xc], h[sa, yc], h[sa, zc] = v_sapc * g1, v_sapc * g2, v_sapc * g3
    h[xa, sc], h[ya, sc], h[za, sc] = -v_pasc * g1, -v_pasc * g2, -v_pasc * g3
    h[xa, xc], h[xa, yc], h[xa, zc] = v_xx * g0, v_xy * g3, v_xy * g2
    h[ya, xc], h[ya, yc], h[ya, zc] = v_xy * g3, v_xx * g0, v_xy * g1
    h[za, xc], h[za, yc], h[za, zc] = v_xy * g2, v_xy * g1, v_xx * g0
    h[ssa, xc], h[ssa, yc], h[ssa, zc] = v_ssapc * g1, v_ssapc * g2, v_ssapc * g3
    h[xa, ssc], h[ya, ssc], h[za, ssc] = -v_passc * g1, -v_passc * g2, -v_passc * g3
    h = h + h.conj().T
    for index, value in zip(range(10), (e_sa, e_sc, e_pa, e_pa, e_pa, e_pc, e_pc, e_pc,
                                        e_ssa, e_ssc)):
        h[index, index] = value
    return h


VOGL = {
    "sp3s*-Si": ((14, 14), 5.4310205, (-4.2, 1.715, 6.685, -4.2, 1.715, 6.685, -8.3, 1.715,
                                       4.575, 5.7292, 5.7292, 5.3749, 5.3749)),
    "sp3s*-Ge": ((32, 32), 5.65785, (-5.88, 1.61, 6.39, -5.88, 1.61, 6.39, -6.78, 1.61, 4.90,
                                     5.4649, 5.4649, 5.2191, 5.2191)),
    "sp3s*-GaAs": ((33, 31), 5.65325, (-8.3431, 1.0414, 8.5914, -2.6569, 3.6686, 6.7386,
                                       -6.4513, 1.9546, 5.0779, 4.48, 5.7839, 4.8422, 4.8077)),
}


class TestTightBinding:
    @pytest.mark.parametrize("model", sorted(VOGL))
    def test_bands_match_an_independent_vogl_hamiltonian(self, model):
        numbers, a, parameters = VOGL[model]
        cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        crystal = Structure(np.array(numbers), np.array([[0, 0, 0], [a / 4] * 3]),
                            Cell(cell, (True, True, True)))
        tb = TightBinding(model)
        rng = np.random.default_rng(3)
        kpoints = [np.zeros(3), np.array([.5, 0, .5]), np.array([.5, .5, .5]),
                   np.array([.375, .375, .75])] + [rng.random(3) for _ in range(12)]
        for k in kpoints:
            mine = np.linalg.eigvalsh(tb.build_hamiltonian(crystal, k))
            reference = np.linalg.eigvalsh(vogl_hamiltonian(k @ crystal.cell.reciprocal, a,
                                                            *parameters))
            assert np.abs(mine - reference).max() < 1e-10

    def test_graphene_bandwidth_is_six_t(self):
        a = 2.4612
        cell = np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, 20.0]])
        sheet = Structure(np.array([6, 6]), np.array([[0, 0, 10], [0, a / math.sqrt(3), 10]]),
                          Cell(cell, (True, True, False)))
        gamma = np.linalg.eigvalsh(TightBinding("pz-graphene").build_hamiltonian(
            sheet, np.zeros(3)))
        assert gamma.max() - gamma.min() == pytest.approx(16.2, abs=1e-9)


@pytest.fixture(scope="module")
def sw_silicon():
    sw = StillingerWeber("Si")
    scan = np.linspace(5.40, 5.46, 13)
    energies = [sw.energy(cubic(DIAMOND, a, 1, 14)) / 8 for a in scan]
    fit = np.polyfit(scan, energies, 4)
    roots = np.roots(np.polyder(fit))
    a0 = float(min(roots[np.isreal(roots)].real, key=lambda r: abs(r - 5.43)))
    return sw, a0, float(np.polyval(fit, a0))


class TestStillingerWeber:
    """Against the published properties of the same potential.

    H. Balamane, T. Halicioglu and W. A. Tiller, Phys. Rev. B 46 (1992) 2250 and
    R. A. Cowley, Phys. Rev. Lett. 60 (1988) 2379: a0 = 5.431 A, E_coh = -4.3366
    eV, C11 = 151.4, C12 = 76.4, C44 = 56.4 GPa (internally relaxed), B = 101.4
    GPa.  Tolerances cover the second-order finite differences used here and
    the rounding of the published values.
    """

    def test_lattice_constant_and_cohesive_energy(self, sw_silicon):
        _, a0, ecoh = sw_silicon
        assert a0 == pytest.approx(5.431, abs=1e-3)
        assert ecoh == pytest.approx(-4.3366, abs=1e-4)

    def test_elastic_constants(self, sw_silicon):
        sw, a0, _ = sw_silicon
        reference = cubic(DIAMOND, a0, 2, 14)
        e0 = sw.energy(reference)
        volume = abs(np.linalg.det(reference.cell.matrix))
        h = 1e-3

        def density(strain, relax=False):
            strained = cubic(DIAMOND, a0, 2, 14, strain)
            if relax:
                strained = ClassicalSolver(sw).relax(strained, fmax_eV_A=1e-7,
                                                     max_steps=5000).structure
            return (sw.energy(strained) - e0) / volume

        def central(strain, relax=False):
            return (density(strain, relax) + density(-strain, relax)) / 2

        axial = np.zeros((3, 3))
        axial[0, 0] = h
        c11 = 2 * central(axial) / h ** 2 * EV_A3_TO_GPA
        biaxial = np.zeros((3, 3))
        biaxial[0, 0] = biaxial[1, 1] = h
        c12 = central(biaxial) / h ** 2 * EV_A3_TO_GPA - c11
        shear = np.zeros((3, 3))
        shear[1, 2] = shear[2, 1] = h / 2
        c44 = 2 * central(shear, relax=True) / h ** 2 * EV_A3_TO_GPA
        assert c11 == pytest.approx(151.4, abs=1.0)
        assert c12 == pytest.approx(76.4, abs=1.0)
        assert c44 == pytest.approx(56.4, abs=0.5)
        assert (c11 + 2 * c12) / 3 == pytest.approx(101.4, abs=1.0)

    def test_vacancy_has_an_ideal_and_a_reconstructed_minimum(self, sw_silicon):
        """The ideal vacancy is a local minimum: its four neighbours sit beyond the
        SW cutoff, so local relaxation keeps 2 epsilon = 4.3366 eV. Seeding them
        inward finds the reconstructed vacancy, 2.6 to 2.9 eV in the literature for
        cells of this size; both are reported, neither is hidden."""
        sw, a0, _ = sw_silicon
        crystal = cubic(DIAMOND, a0, 3, 14)
        n = len(crystal)
        e_bulk = sw.energy(crystal)
        vacancy = Structure(crystal.numbers[1:], crystal.positions[1:], crystal.cell)
        ideal = ClassicalSolver(sw).relax(vacancy, fmax_eV_A=1e-5, max_steps=5000).structure
        assert sw.energy(ideal) - (n - 1) / n * e_bulk == pytest.approx(4.3366, abs=1e-3)
        offsets = vacancy.positions - crystal.positions[0]
        lattice = crystal.cell.matrix
        offsets -= np.round(offsets @ np.linalg.inv(lattice)) @ lattice
        nearest = np.argsort(np.linalg.norm(offsets, axis=1))[:4]
        seeded = vacancy.copy()
        seeded.positions[nearest] -= 0.5 * offsets[nearest] / np.linalg.norm(
            offsets[nearest], axis=1)[:, None]
        relaxed = ClassicalSolver(sw).relax(seeded, fmax_eV_A=1e-5, max_steps=50000).structure
        formation = sw.energy(relaxed) - (n - 1) / n * e_bulk
        assert 2.6 < formation < 2.9


class TestLennardJonesMetals:
    @pytest.mark.parametrize("material_id", ["copper", "gold", "silver", "platinum", "nickel"])
    def test_derived_potential_reproduces_its_own_inputs(self, material_id):
        from materia.elements import periodic_table as pt
        from materia.materials import load

        material = load(material_id)
        potential = LennardJones.from_material(material)
        a = material.lattice.parameters()[0]
        z = pt.atomic_number(material.elements()[0])
        energy = lambda scale: potential.energy(cubic(FCC, a * scale, 3, z)) / 108
        assert energy(1.0) == pytest.approx(-material.properties["cohesive_energy"].value,
                                            abs=1e-9)
        slope = (energy(1 + 1e-5) - energy(1 - 1e-5)) / (2e-5 * a)
        assert abs(slope) < 1e-5


FIRST_SHELL = {
    "copper": {"Cu": 12}, "gold": {"Au": 12}, "silver": {"Ag": 12}, "nickel": {"Ni": 12},
    "platinum": {"Pt": 12}, "tungsten": {"W": 8}, "titanium": {"Ti": 12},
    "silicon": {"Si": 4}, "germanium": {"Ge": 4}, "diamond": {"C": 4},
    "gallium_arsenide": {"Ga": 4, "As": 4}, "indium_phosphide": {"In": 4, "P": 4},
    "gallium_nitride": {"Ga": 4, "N": 4}, "silicon_carbide_3c": {"Si": 4, "C": 4},
    "silicon_carbide_4h": {"Si": 4, "C": 4}, "graphene": {"C": 3},
    "hexagonal_boron_nitride": {"B": 3, "N": 3}, "silicon_dioxide": {"Si": 4, "O": 2},
    "sapphire": {"Al": 6, "O": 4}, "molybdenum_disulfide": {"Mo": 6, "S": 3},
    "tungsten_diselenide": {"W": 6, "Se": 3},
}


class TestLibraryStructures:
    """Every shipped crystal has physical contacts and its textbook coordination.

    The first shell of an atom is every neighbour within 1.1 times its nearest
    distance. A mispositioned basis atom breaks this even when the density,
    which only counts atoms, is right; that is how a 4H-SiC basis with a 0.63 A
    Si-C contact was found.
    """

    @pytest.mark.parametrize("material_id", sorted(FIRST_SHELL))
    def test_contacts_and_coordination(self, material_id):
        from materia.elements import periodic_table as pt
        from materia.materials import load
        from materia.structure_builder.lattice import bulk

        crystal = bulk(load(material_id), (3, 3, 3))
        nl = neighbor_list(crystal.positions, crystal.cell, 4.0)
        radii = np.array([pt.covalent_radius(int(z)) for z in crystal.numbers])
        assert np.all(nl.d > 0.75 * (radii[nl.i] + radii[nl.j]))
        symbols = [pt.symbol(int(z)) for z in crystal.numbers]
        nearest = np.full(len(crystal), np.inf)
        np.minimum.at(nearest, nl.i, nl.d)
        shell = nl.d < 1.1 * nearest[nl.i]
        counts = np.bincount(nl.i[shell], minlength=len(crystal))
        for index, symbol in enumerate(symbols):
            assert counts[index] == FIRST_SHELL[material_id][symbol], (symbol, index)
