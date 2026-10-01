"""Physical validation cases.

These are **not** implementation unit tests. Each case compares a computed
result against an analytic value, a published reference, or a conservation
law, and states the tolerance and its justification. A passing case means the
implementation reproduces the model it claims to implement; it does not mean
the model is a correct description of nature.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.core_model.units import BOLTZMANN_EV_K
from materia.materials import load
from materia.physics.neighbors import neighbor_list
from materia.physics.potentials import StillingerWeber
from materia.solvers.classical import stillinger_weber
from materia.solvers.tight_binding import TightBinding
from materia.structure_builder import bulk, compare_to_reference, make_surface
from materia.structure_builder.lattice import lattice_planes
from materia.structure_builder.primitive import primitive_structure

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def silicon_cell():
    return primitive_structure(bulk(load("silicon")))


@pytest.fixture(scope="module")
def simulator():
    from materia.microscopy.stm import STMSimulator
    from materia.microscopy.tip import Tip
    return STMSimulator(TightBinding("sp3s*-Si"), Tip("W"))


@pytest.fixture(scope="module")
def slab():
    return make_surface(load("silicon"), (1, 1, 1), size=(3, 3, 3), vacuum_A=12.0)


class TestCrystallography:
    """Analytic lattice geometry. Exact, so the tolerances are numerical only."""

    @pytest.mark.parametrize(
        "material,prototype,coordination,formula",
        [("silicon", "diamond-cubic", 4, lambda a: a * math.sqrt(3) / 4),
         ("germanium", "diamond-cubic", 4, lambda a: a * math.sqrt(3) / 4),
         ("diamond", "diamond-cubic", 4, lambda a: a * math.sqrt(3) / 4),
         ("gold", "fcc", 12, lambda a: a / math.sqrt(2)),
         ("copper", "fcc", 12, lambda a: a / math.sqrt(2)),
         ("tungsten", "bcc", 8, lambda a: a * math.sqrt(3) / 2)],
    )
    def test_nearest_neighbour_distance_and_coordination(
            self, material, prototype, coordination, formula):
        definition = load(material)
        assert definition.prototype == prototype
        crystal = bulk(definition)
        a = definition.lattice.parameters()[0]
        expected = formula(a)
        nl = neighbor_list(crystal.positions, crystal.cell, expected * 1.1)
        assert set(nl.counts()) == {coordination}
        assert float(nl.d.min()) == pytest.approx(expected, rel=1e-9)

    def test_tabulated_nearest_neighbour_distances_are_consistent(self):
        for material in ("silicon", "gold", "gallium_arsenide"):
            definition = load(material)
            tabulated = definition.properties["nearest_neighbour_distance"].value
            crystal = bulk(definition)
            nl = neighbor_list(crystal.positions, crystal.cell, tabulated * 1.1)
            assert float(nl.d.min()) == pytest.approx(tabulated, rel=2e-4)

    @pytest.mark.parametrize(
        "miller,factor",
        [((1, 0, 0), 1.0), ((1, 1, 0), 1 / math.sqrt(2)), ((1, 1, 1), 1 / math.sqrt(3)),
         ((2, 0, 0), 0.5), ((3, 1, 1), 1 / math.sqrt(11))],
    )
    def test_cubic_interplanar_spacing(self, miller, factor):
        a = 5.4310205
        assert lattice_planes(load("silicon"), miller) == pytest.approx(a * factor, rel=1e-9)

    def test_densities_match_the_generated_cells(self):
        avogadro = 6.02214076e23
        for material in ("silicon", "germanium", "gold", "copper", "silver",
                         "diamond", "gallium_arsenide", "tungsten"):
            definition = load(material)
            crystal = bulk(definition)
            mass_g = crystal.masses().sum() / avogadro
            volume_cm3 = crystal.cell.volume * 1e-24
            computed = mass_g / volume_cm3
            tabulated = definition.properties["density"].value
            assert computed == pytest.approx(tabulated, rel=0.01), material

    def test_graphene_bond_length(self):
        sheet = bulk(load("graphene"))
        nl = neighbor_list(sheet.positions, sheet.cell, 1.8)
        assert float(nl.d.min()) == pytest.approx(1.42, abs=0.01)
        assert set(nl.counts()) == {3}

    def test_corundum_aluminium_oxygen_distances(self):
        """alpha-Al2O3 has two distinct Al-O bond lengths, 1.86 and 1.97 A.

        Reference: N. Ishizawa et al., Acta Cryst. B 36 (1980) 228.
        """
        sapphire = bulk(load("sapphire"))
        aluminium = np.nonzero(sapphire.numbers == 13)[0]
        nl = neighbor_list(sapphire.positions, sapphire.cell, 2.1)
        lengths = sorted({round(float(d), 3) for i, d in zip(nl.i, nl.d)
                          if i in set(aluminium.tolist())})
        assert len(lengths) == 2
        assert lengths[0] == pytest.approx(1.856, abs=0.02)
        assert lengths[1] == pytest.approx(1.971, abs=0.02)

    def test_quartz_silicon_oxygen_distance(self):
        """alpha-quartz Si-O is 1.605-1.614 A (Levien et al., Am. Mineral. 65, 920)."""
        quartz = bulk(load("silicon_dioxide"))
        nl = neighbor_list(quartz.positions, quartz.cell, 1.8)
        silicon = set(np.nonzero(quartz.numbers == 14)[0].tolist())
        lengths = [float(d) for i, d in zip(nl.i, nl.d) if int(i) in silicon]
        assert len(lengths) == 12
        assert min(lengths) == pytest.approx(1.605, abs=0.01)
        assert max(lengths) == pytest.approx(1.614, abs=0.01)

    def test_molybdenum_disulfide_bond_length(self):
        """Mo-S is 2.41 A (Schoenfeld et al., Acta Cryst. B 39, 404)."""
        crystal = bulk(load("molybdenum_disulfide"))
        nl = neighbor_list(crystal.positions, crystal.cell, 2.6)
        molybdenum = set(np.nonzero(crystal.numbers == 42)[0].tolist())
        lengths = [float(d) for i, d in zip(nl.i, nl.d) if int(i) in molybdenum]
        assert lengths
        assert np.mean(lengths) == pytest.approx(2.41, abs=0.02)

    def test_primitive_cell_volumes(self):
        for material, multiplicity in (("silicon", 4), ("gold", 4), ("tungsten", 2)):
            conventional = bulk(load(material))
            primitive = primitive_structure(conventional)
            assert primitive.cell.volume == pytest.approx(
                conventional.cell.volume / multiplicity, rel=1e-9)


class TestClassicalPotential:
    """Stillinger-Weber against the values its parameters were fitted to."""

    def test_cohesive_energy_is_exactly_two_epsilon(self):
        """SW is constructed so the ideal diamond lattice has E = -2 eps/atom.

        This is exact for silicon, whose tabulated lattice constant coincides
        with the potential's own equilibrium bond length 2^(1/6) sigma. The
        germanium parameterisation of Ding and Andersen was fitted with a
        slightly different lattice constant, so the shipped Ge structure sits
        a few parts in 10^6 off the potential's minimum; the test records that
        difference rather than hiding it.
        """
        silicon = bulk(load("silicon"), (2, 2, 2))
        energy, _ = StillingerWeber("Si").energy_and_forces(silicon)
        assert energy / len(silicon) == pytest.approx(-2 * 2.1683, rel=1e-7)

        germanium = bulk(load("germanium"), (2, 2, 2))
        energy, _ = StillingerWeber("Ge").energy_and_forces(germanium)
        assert energy / len(germanium) == pytest.approx(-2 * 1.93, rel=1e-4)
        assert energy / len(germanium) > -2 * 1.93

    def test_equilibrium_lattice_constant_is_a_minimum(self):
        """The fitted lattice constant must be the energy minimum."""
        definition = load("silicon")
        potential = StillingerWeber("Si")
        energies = {}
        for scale in (0.98, 0.99, 1.0, 1.01, 1.02):
            crystal = bulk(definition, (2, 2, 2))
            crystal.cell = type(crystal.cell)(crystal.cell.matrix * scale,
                                              crystal.cell.pbc)
            crystal.positions = crystal.positions * scale
            energies[scale] = potential.energy(crystal) / len(crystal)
        assert energies[1.0] == min(energies.values())
        assert energies[0.98] > energies[1.0]
        assert energies[1.02] > energies[1.0]

    def test_forces_are_the_negative_energy_gradient(self):
        slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 2), vacuum_A=10.0)
        rng = np.random.default_rng(17)
        slab.positions = slab.positions + rng.normal(0, 0.1, slab.positions.shape)
        potential = StillingerWeber("Si")
        _, forces = potential.energy_and_forces(slab)
        step = 1e-5
        errors = []
        for atom in range(0, len(slab), 5):
            for axis in range(3):
                plus, minus = slab.copy(), slab.copy()
                plus.positions[atom, axis] += step
                minus.positions[atom, axis] -= step
                numerical = -(potential.energy(plus) - potential.energy(minus)) / (2 * step)
                errors.append(abs(numerical - forces[atom, axis]))
        assert max(errors) < 1e-6

    def test_newtons_third_law(self):
        """Net force on an isolated cluster must vanish."""
        crystal = bulk(load("silicon"), (2, 2, 2))
        rng = np.random.default_rng(23)
        crystal.positions = crystal.positions + rng.normal(0, 0.06, crystal.positions.shape)
        _, forces = StillingerWeber("Si").energy_and_forces(crystal)
        assert np.abs(forces.sum(axis=0)).max() < 1e-9

    def test_translational_and_rotational_invariance(self):
        crystal = bulk(load("silicon"), (2, 2, 2))
        potential = StillingerWeber("Si")
        reference = potential.energy(crystal)
        moved = crystal.copy()
        moved.translate([1.234, -2.345, 0.567])
        assert potential.energy(moved) == pytest.approx(reference, rel=1e-12)

    def test_energy_conservation_in_microcanonical_dynamics(self):
        """NVE drift must stay small: the integrator's only real correctness test."""
        crystal = bulk(load("silicon"), (2, 2, 2))
        out = stillinger_weber("Si").dynamics(
            crystal, steps=500, dt_fs=1.0, temperature_K=400,
            thermostat="none", seed=5)
        trajectory = out["trajectory"].value
        total = np.array(trajectory["total_eV"])
        drift_per_atom = abs(total[-1] - total[0]) / len(crystal)
        assert drift_per_atom < 5e-4

    def test_equipartition_in_the_canonical_ensemble(self):
        """A Langevin thermostat must reproduce its target temperature.

        The mean over 1.5 ps of 216 atoms scatters by about 5 K between seeds,
        so 3 percent is three standard errors. A 12 ps run gives 501.2 +- 1.6 K.
        """
        crystal = bulk(load("silicon"), (3, 3, 3))
        out = stillinger_weber("Si").dynamics(
            crystal, steps=1500, dt_fs=1.0, temperature_K=500,
            thermostat="langevin", friction_per_fs=0.02, seed=11)
        mean = out["mean_temperature"]
        assert mean.value == pytest.approx(500, rel=0.03)


class TestTightBinding:
    """Vogl sp3s* against the band-structure values it was fitted to.

    Reference: P. Vogl, H. P. Hjalmarson and J. D. Dow,
    J. Phys. Chem. Solids 44 (1983) 365, Table II.
    """

    def test_valence_band_width(self, silicon_cell):
        """Vogl fits the silicon valence-band width to 12.5 eV."""
        solver = TightBinding("sp3s*-Si")
        out = solver.band_structure(
            silicon_cell, path=[(0.5, 0.5, 0.5), (0, 0, 0), (0.5, 0, 0.5)],
            labels=["L", "G", "X"], n_per_segment=40)
        bands = out["band_structure"].value["bands_eV"]
        width = float(bands[:, 3].max() - bands[:, 0].min())
        assert width == pytest.approx(12.5, abs=0.05)

    def test_valence_band_maximum_is_at_gamma(self, silicon_cell):
        solver = TightBinding("sp3s*-Si")
        out = solver.band_structure(
            silicon_cell, path=[(0.5, 0.5, 0.5), (0, 0, 0), (0.5, 0, 0.5)],
            labels=["L", "G", "X"], n_per_segment=40)
        bands = out["band_structure"].value["bands_eV"]
        kpoints = out["band_structure"].value["kpoints_frac"]
        top = int(np.argmax(bands[:, 3]))
        assert np.allclose(kpoints[top], [0, 0, 0], atol=0.02)
        assert bands[top, 3] == pytest.approx(0.0, abs=1e-3)

    def test_indirect_gap_of_silicon(self, silicon_cell):
        """The model's gap is indirect, 1.171 eV, at 0.731 of the way from Gamma to X.

        The experimental low-temperature gap of silicon is 1.17 eV, which is
        what the parameterisation was fitted to reproduce. The experimental
        conduction minimum lies near 0.85 of Gamma to X; this nearest-neighbour
        model puts it at 0.73, and that difference is a property of the model.
        The minimum is located by a bounded one-dimensional search, and a 12^3
        scan of the whole zone confirms that no lower conduction state exists.
        """
        from scipy.optimize import minimize_scalar

        solver = TightBinding("sp3s*-Si")
        levels = lambda kf: np.linalg.eigvalsh(solver.build_hamiltonian(silicon_cell,
                                                                        k_frac=kf))
        a = load("silicon").lattice.parameters()[0]
        x_point = np.array([2.0 * np.pi / a, 0.0, 0.0]) @ np.linalg.inv(
            silicon_cell.cell.reciprocal)
        line = minimize_scalar(lambda t: levels(t * x_point)[4], bounds=(0.0, 1.0),
                               method="bounded", options={"xatol": 1e-8})
        valence_max = levels(np.zeros(3))[3]
        gap = line.fun - valence_max
        assert gap == pytest.approx(1.1713, abs=5e-4)
        assert line.x == pytest.approx(0.731, abs=0.005)
        grid = 12
        scan_min = min(levels(np.array([i, j, k]) / grid)[4] for i in range(grid)
                       for j in range(grid) for k in range(grid))
        scan_max = max(levels(np.array([i, j, k]) / grid)[3] for i in range(grid)
                       for j in range(grid) for k in range(grid))
        assert scan_min >= line.fun - 1e-9
        assert scan_max <= valence_max + 1e-9

    def test_direct_gap_of_gallium_arsenide(self):
        """Vogl fits GaAs to a direct gap of 1.55 eV at Gamma."""
        cell = primitive_structure(bulk(load("gallium_arsenide")))
        out = TightBinding("sp3s*-GaAs").band_structure(
            cell, path=[(0.5, 0.5, 0.5), (0, 0, 0), (0.5, 0, 0.5)],
            labels=["L", "G", "X"], n_per_segment=40)
        assert out["band_gap"].value == pytest.approx(1.55, abs=0.02)
        assert out["band_gap"].extra["direct"] is True

    def test_graphene_is_gapless_at_the_dirac_point(self):
        """Nearest-neighbour pi bands touch at K and span 6|t|."""
        sheet = primitive_structure(bulk(load("graphene")))
        solver = TightBinding("pz-graphene")
        values = np.linalg.eigvalsh(
            solver.build_hamiltonian(sheet, k_frac=(1 / 3, 1 / 3, 0)))
        assert values[1] - values[0] == pytest.approx(0.0, abs=1e-5)

        out = solver.band_structure(sheet, path=[(0, 0, 0), (1 / 3, 1 / 3, 0),
                                                 (0.5, 0, 0)],
                                    labels=["G", "K", "M"], n_per_segment=60)
        bands = out["band_structure"].value["bands_eV"]
        assert float(bands.max() - bands.min()) == pytest.approx(6 * 2.7, abs=1e-9)

    def test_electron_hole_symmetry_of_the_pi_model(self):
        sheet = primitive_structure(bulk(load("graphene")))
        solver = TightBinding("pz-graphene")
        for kf in [(0.1, 0.2, 0), (0.3, 0.05, 0), (0.25, 0.25, 0)]:
            values = np.linalg.eigvalsh(solver.build_hamiltonian(sheet, k_frac=kf))
            assert values[0] == pytest.approx(-values[1], abs=1e-9)

    def test_hamiltonian_is_hermitian(self):
        cell = primitive_structure(bulk(load("silicon")))
        solver = TightBinding("sp3s*-Si")
        for kf in [(0, 0, 0), (0.25, 0.1, 0.3)]:
            H = solver.build_hamiltonian(cell, k_frac=kf)
            assert np.abs(H - H.conj().T).max() < 1e-12

    def test_electron_count_matches_the_valence_electrons(self):
        slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 2), vacuum_A=10.0)
        out = TightBinding("sp3s*-Si").eigenstates(slab)
        assert out["fermi_level"].extra["valence_electrons"] == 4 * len(slab)

    def test_donor_shifts_charge_towards_the_impurity(self):
        """A phosphorus donor must attract charge relative to the host."""
        slab = make_surface(load("silicon"), (1, 1, 1), size=(2, 2, 3), vacuum_A=12.0)
        site = len(slab) // 2
        slab.substitute(int(slab.ids[site]), "P")
        out = TightBinding("sp3s*-Si", impurities=["P"]).eigenstates(slab)
        charges = out["partial_charges"].value
        assert charges[site] < np.delete(charges, site).mean()


class TestScanningProbe:
    """Tersoff-Hamann behaviour against the physics it is built from."""

    def test_current_decays_exponentially_with_tip_height(self, simulator, slab):
        """The decay rate approaches 2 kappa from below as the tip retracts.

        Above an atom its own orbital decays as exp(-2 kappa z), while the tails
        of its neighbours decay more slowly with height, so the apparent decay is
        slower than 2 kappa close to the surface and converges to it far away:
        the apparent barrier of a superposition is lower than the true barrier.
        Measured here: 0.901, 0.932 and 0.965 of 2 kappa over 6 to 9, 10 to 16
        and 20 to 28 A.
        """
        blocks, positions, orbitals, _ = simulator.vacuum_states(
            slab, 1.0, 0.1, kgrid=(1, 1), window_A=(0, 0, 12, 10))
        kappa, _ = simulator.decay_constant(1.0)
        top = float(slab.positions[:, 2].max())
        index = int(np.argmax(slab.positions[:, 2]))
        xy = slab.positions[index, :2][None, :]

        def ratio(heights):
            currents = np.array([
                simulator.current_map(positions, orbitals, blocks, xy,
                                      np.array([top + h]), kappa)[0] for h in heights])
            return np.polyfit(heights, np.log(currents), 1)[0] / (-2 * kappa)

        near = ratio(np.array([6.0, 7.0, 8.0, 9.0]))
        middle = ratio(np.array([10.0, 12.0, 14.0, 16.0]))
        far = ratio(np.array([20.0, 24.0, 28.0]))
        assert near == pytest.approx(0.901, abs=0.005)
        assert near < middle < far < 1.0
        assert far == pytest.approx(0.965, abs=0.005)

    def test_corrugation_decreases_with_tip_height(self, simulator, slab):
        from materia.microscopy.noise import NoiseModel
        from materia.microscopy.stm import STMSettings

        close = simulator.scan(slab, STMSettings(
            mode="constant-height", height_A=4.0, resolution=(48, 48),
            kgrid=(1, 1), noise=NoiseModel.quiet(0)))
        far = simulator.scan(slab, STMSettings(
            mode="constant-height", height_A=8.0, resolution=(48, 48),
            kgrid=(1, 1), noise=NoiseModel.quiet(0)))

        def relative_corrugation(scan):
            data = scan.channel("current")
            return float(np.ptp(data) / data.mean())

        assert relative_corrugation(close) > relative_corrugation(far)

    def test_corrugation_is_in_the_experimental_range(self, simulator, slab):
        """Si(111) corrugation at a few angstrom is of order 0.1-1 A."""
        from materia.microscopy.noise import NoiseModel
        from materia.microscopy.stm import STMSettings

        scan = simulator.scan(slab, STMSettings(
            bias_V=1.0, resolution=(96, 96), kgrid=(2, 2),
            noise=NoiseModel.quiet(0)))
        corrugation = float(np.ptp(scan.channel("topography")))
        assert 0.05 < corrugation < 1.5

    def test_maxima_follow_the_lattice_periodicity(self, simulator, slab):
        from materia.microscopy.noise import NoiseModel
        from materia.microscopy.stm import STMSettings

        scan = simulator.scan(slab, STMSettings(
            bias_V=1.0, resolution=(128, 128), kgrid=(1, 1),
            noise=NoiseModel.quiet(0)))
        features = scan.detect_features()
        _, spacing = scan._surface_lattice()
        points = np.array([[f.x_A, f.y_A] for f in features])
        assert len(points) >= 6
        distances = []
        for i, point in enumerate(points):
            others = np.delete(points, i, axis=0)
            distances.append(np.linalg.norm(others - point, axis=1).min())
        assert np.median(distances) == pytest.approx(spacing, rel=0.12)


class TestAtomicForceMicroscopy:
    def test_frequency_shift_reduces_to_the_small_amplitude_limit(self):
        """For A -> 0, df -> -(f0 / 2k) dF/dz."""
        from materia.microscopy.afm import AFMSimulator, N_PER_M_TO_EV_A2

        simulator = AFMSimulator()
        f0, k = 30000.0, 1800.0

        def force(z):
            z = np.asarray(z, dtype=float)
            return -1.0 / z**2

        amplitude = 0.005
        z0 = np.array([5.0])
        derivative = 2.0 / (z0[0] + amplitude) ** 3
        expected = -(f0 / (2 * k * N_PER_M_TO_EV_A2)) * derivative
        computed = simulator.frequency_shift(force, z0, amplitude, f0, k, 33)[0]
        assert computed == pytest.approx(expected, rel=1e-3)

    def test_frequency_shift_quadrature_is_exact_for_a_linear_force(self):
        """For F(z) = c z the Giessibl integral has the closed form -f0 c / 2k."""
        from materia.microscopy.afm import AFMSimulator, N_PER_M_TO_EV_A2

        simulator = AFMSimulator()
        f0, k, slope = 25000.0, 1500.0, 0.37
        expected = -(f0 / (2 * k * N_PER_M_TO_EV_A2)) * slope
        for amplitude in (0.2, 1.0, 5.0):
            for n_quad in (9, 17, 33):
                computed = simulator.frequency_shift(
                    lambda z: slope * np.asarray(z, dtype=float),
                    np.array([6.0]), amplitude, f0, k, n_quad)[0]
                assert computed == pytest.approx(expected, rel=1e-10)

    def test_hamaker_background_scales_as_inverse_square(self):
        from materia.microscopy.afm import AFMSimulator
        from materia.microscopy.tip import Tip

        simulator = AFMSimulator(Tip("W", radius_A=50.0))
        empty = np.zeros((1, 0))
        values = []
        for height in (5.0, 10.0):
            values.append(simulator._force_kernel(
                empty, np.zeros(0), np.array([height]), np.zeros(0), np.zeros(0),
                1.0, 0.624, 50.0, 0.0)[0])
        assert values[0] / values[1] == pytest.approx(4.0, rel=1e-9)


ONE_BY_ONE_SPACING_A = 5.4310205 / math.sqrt(2)
BULK_BOND_A = 5.4310205 * math.sqrt(3) / 4
LEED_BOND_A = 2.24
LEED_BOND_UNCERTAINTY_A = 0.08
LEED_BUCKLING_A = 0.72


def _dimerised(nx=4, ny=2, nz=8, **options):
    return make_surface(load("silicon"), (1, 0, 0), size=(nx, ny, nz),
                        vacuum_A=12.0, fix_bottom_layers=4,
                        reconstruction="2x1-dimer", **options)


@pytest.fixture(scope="module")
def dimerised():
    return _dimerised()


class TestSiliconDimerReconstruction:
    """Si(100)-(2x1) built by the dimer generator and relaxed with Stillinger-Weber.

    The generator imposes the pairing only.  Every distance below is the output
    of the relaxation, so these cases test whether the implementation reproduces
    the reconstruction that the Stillinger-Weber model predicts, and report how
    far that model sits from the LEED structure determination of Over et al.,
    Phys. Rev. B 55 (1997) 4731.  They do not assert agreement with experiment.
    """

    def test_unreconstructed_spacing_matches_the_1x1_surface_lattice(self, dimerised):
        """The sites the generator paired sat a/sqrt(2) apart, as (100) requires."""
        record = dimerised.info["reconstruction"]
        assert record["unreconstructed_spacing_A"] == pytest.approx(
            ONE_BY_ONE_SPACING_A, abs=1e-4)

    def test_relaxation_converged(self, dimerised):
        relaxation = dimerised.info["reconstruction"]["relaxation"]
        assert relaxation["converged"] is True
        assert relaxation["max_force_eV_A"] <= relaxation["fmax_target_eV_A"]

    def test_dimer_bond_is_a_silicon_bond_not_a_lattice_spacing(self, dimerised):
        """Pairing must close a real bond: well below the 1x1 spacing of 3.840 A
        and within a chemically sensible window around the bulk bond of 2.352 A."""
        measured = dimerised.info["reconstruction"]["measured"]
        bond = measured["bond_length_A"]
        assert bond < ONE_BY_ONE_SPACING_A - 1.0
        assert 0.9 * BULK_BOND_A < bond < 1.15 * BULK_BOND_A
        assert measured["contraction_A"] == pytest.approx(
            ONE_BY_ONE_SPACING_A - bond, abs=1e-6)

    def test_all_dimers_in_the_cell_are_equivalent(self, dimerised):
        """A p(2x1) has one dimer per surface cell, so every dimer in a supercell
        must relax to the same length."""
        measured = dimerised.info["reconstruction"]["measured"]
        assert measured["n_dimers"] == 4
        assert measured["bond_length_spread_A"] < 1e-6

    def test_periodicity_is_2x1_and_not_1x1(self, dimerised):
        """A four-repeat cell makes the doubled translation a genuine test rather
        than the lattice vector itself."""
        check = dimerised.info["reconstruction"]["periodicity_check"]
        assert check["onexone_repeats_along_axis"] == pytest.approx(4.0)
        assert check["2x1_shift_is_lattice_vector"] is False
        assert check["invariant_under_1x1_shift"] is False
        assert check["invariant_under_2x1_shift"] is True

    @pytest.mark.parametrize("start", [2.2, 2.6, 3.0, 3.4])
    def test_geometry_is_independent_of_the_construction_guess(self, dimerised, start):
        """If the relaxed geometry moved with the starting separation, the number
        reported would be the guess rather than a minimum of the potential.  The
        tolerance is the residual scatter left by the 0.005 eV/A force criterion."""
        reference = dimerised.info["reconstruction"]["measured"]["bond_length_A"]
        slab = _dimerised(reconstruction_options={"initial_separation_A": start})
        assert slab.info["reconstruction"]["measured"]["bond_length_A"] == pytest.approx(
            reference, abs=1e-3)

    @pytest.mark.parametrize("size", [(2, 2, 8), (4, 3, 8), (4, 2, 10), (6, 2, 8)])
    def test_geometry_is_independent_of_cell_size(self, dimerised, size):
        """One dimer per 2x1 cell: widening or deepening the slab must not change
        the bond it settles at."""
        reference = dimerised.info["reconstruction"]["measured"]["bond_length_A"]
        slab = _dimerised(*size)
        assert slab.info["reconstruction"]["measured"]["bond_length_A"] == pytest.approx(
            reference, abs=1e-3)

    def test_reconstruction_lowers_the_energy_of_the_surface(self):
        """The physical claim behind any reconstruction: it must be the lower
        energy state.  Both slabs are relaxed with the same potential, the same
        cell and the same fixed layers, so the difference is the surface term."""
        solver = stillinger_weber("Si")
        ideal = make_surface(load("silicon"), (1, 0, 0), size=(4, 2, 8),
                             vacuum_A=12.0, fix_bottom_layers=4)
        ideal_energy = solver.relax(
            ideal, fmax_eV_A=0.005, max_steps=800, in_place=True
        ).results["energy"].value
        slab = _dimerised()
        record = slab.info["reconstruction"]
        gain_per_dimer = (ideal_energy - record["relaxation"]["energy_eV"]) / record["n_dimers"]
        assert gain_per_dimer > 0.5

    def test_deviation_from_the_leed_bond_length_is_bounded_and_reported(self, dimerised):
        """Stillinger-Weber is a short-range empirical potential fitted to bulk
        silicon, so it is expected to overestimate the dimer bond.  The case
        bounds that overestimate rather than claiming agreement: the measured
        value must stay within 15 per cent of the LEED result and must not be
        reported as lying inside the quoted experimental uncertainty."""
        comparison = compare_to_reference(dimerised, load("silicon"))
        row = next(r for r in comparison["rows"] if r["quantity"] == "bond_length_A")
        assert row["reference"] == LEED_BOND_A
        assert row["reference_uncertainty"] == LEED_BOND_UNCERTAINTY_A
        assert abs(row["relative_deviation"]) < 0.15
        assert row["deviation"] > 0
        assert row["within_stated_uncertainty"] is False

    def test_the_model_produces_no_buckling_and_says_so(self, dimerised):
        """The real dimer is buckled by 0.72 A because charge transfers between
        its two atoms.  Stillinger-Weber has no electronic degrees of freedom, so
        it returns a symmetric dimer.  The comparison must show that gap rather
        than hide it."""
        measured = dimerised.info["reconstruction"]["measured"]
        assert measured["buckling_max_A"] < 1e-6
        comparison = compare_to_reference(dimerised, load("silicon"))
        row = next(r for r in comparison["rows"] if r["quantity"] == "buckling_A")
        assert row["reference"] == LEED_BUCKLING_A
        assert row["measured"] == pytest.approx(0.0, abs=1e-6)
        assert row["within_stated_uncertainty"] is False

    def test_subsurface_relaxation_decays_into_the_slab(self, dimerised):
        """Dimerisation strains the layers beneath it.  The displacement pattern
        must decay with depth, otherwise the slab is too thin to carry the
        reconstruction."""
        ideal = make_surface(load("silicon"), (1, 0, 0), size=(4, 2, 8),
                             vacuum_A=12.0, fix_bottom_layers=4)
        shift = dimerised.positions - ideal.positions
        depth = ideal.positions[:, 2].max() - ideal.positions[:, 2]
        magnitude = np.linalg.norm(shift, axis=1)
        top = magnitude[depth < 0.5].mean()
        deep = magnitude[depth > 6.0].mean()
        assert top > 0.5
        assert deep < 0.25 * top
