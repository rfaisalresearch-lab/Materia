"""Rigid-ion potentials: parameters, pair sums, forces, the collapse barrier and refusals."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.core_model.units import COULOMB_K_EV_A
from materia.materials import default_library
from materia.physics import electrostatics as es
from materia.physics import rigid_ion as R
from materia.physics.potentials import UnsupportedSystem
from materia.solvers import registry
from materia.solvers.classical import ClassicalSolver
from materia.structure_builder.lattice import bulk


def quartz(repeat=(1, 1, 1)):
    return bulk(default_library().get("silicon_dioxide"), repeat)


def shaken(structure, seed=3, amplitude=0.05):
    out = structure.copy()
    out.positions = out.positions + np.random.default_rng(seed).uniform(
        -amplitude, amplitude, out.positions.shape)
    return out


def dimer(distance, a=14, b=8):
    return Structure([a, b], np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0 + distance]]),
                     Cell(np.eye(3) * 12.0, (False, False, False)))


def finite_difference(potential, structure, h=1e-5):
    out = np.zeros_like(structure.positions)
    for i in range(len(structure)):
        for c in range(3):
            plus, minus = structure.copy(), structure.copy()
            plus.positions[i, c] += h
            minus.positions[i, c] -= h
            out[i, c] = -(potential.energy(plus) - potential.energy(minus)) / (2 * h)
    return out


class TestParameters:
    def test_bks_values_are_the_published_ones(self):
        p = R.BKS_SILICA
        assert p.charges_e == {"Si": 2.4, "O": -1.2}
        assert p.pairs[("O", "Si")] == R.BuckinghamPair(18003.7572, 4.87318, 133.5381)
        assert p.pairs[("O", "O")] == R.BuckinghamPair(1388.7730, 2.76000, 175.0000)
        assert ("Si", "Si") not in p.pairs
        assert p.cutoff_A == 10.0
        assert "Phys. Rev. Lett. 64 (1990) 1955" in p.source

    def test_digest_is_stable_and_tracks_every_parameter(self):
        p = R.BKS_SILICA
        assert p.digest() == R.RigidIonParameters(**{
            "id": p.id, "name": p.name, "charges_e": dict(p.charges_e), "pairs": dict(p.pairs),
            "source": p.source, "references": p.references, "fitted_to": p.fitted_to}).digest()
        assert p.with_cutoff(9.0).digest() != p.digest()
        moved = dict(p.pairs)
        moved[("O", "O")] = R.BuckinghamPair(1388.7730, 2.76000, 175.0001)
        assert R.RigidIonParameters(p.id, p.name, p.charges_e, moved, p.source).digest() \
            != p.digest()

    @pytest.mark.parametrize("change, fragment", [
        ({"source": " "}, "stated source"),
        ({"charges_e": {}}, "at least one charge"),
        ({"pairs": {("Si", "Al"): R.BuckinghamPair(1.0, 1.0, 0.0)}}, "no charge"),
        ({"pairs": {("Si", "O"): R.BuckinghamPair(-1.0, 1.0, 0.0)}}, "A >= 0"),
        ({"pairs": {("Si", "O"): R.BuckinghamPair(1.0, 0.0, 0.0)}}, "b > 0"),
        ({"pairs": {("Si", "O"): R.BuckinghamPair(1.0, 1.0, float("nan"))}}, "non-finite"),
        ({"pairs": {("Si", "O"): R.BuckinghamPair(1.0, 1.0, 0.0),
                    ("O", "Si"): R.BuckinghamPair(2.0, 1.0, 0.0)}}, "given twice"),
        ({"cutoff_A": 0.0}, "cutoff_A"),
    ])
    def test_invalid_parameterisations_are_refused(self, change, fragment):
        base = dict(id="x", name="x", charges_e={"Si": 2.4, "O": -1.2},
                    pairs={("Si", "O"): R.BuckinghamPair(1.0, 1.0, 0.0)}, source="test")
        with pytest.raises(ValueError, match=fragment):
            R.RigidIonParameters(**{**base, **change})

    def test_selection_needs_exactly_the_covered_elements(self):
        assert R.select_shipped(["O", "Si"]) == "bks-silica"
        assert R.select_shipped(["Si"]) is None
        assert R.select_shipped(["Si", "O", "Al"]) is None


class TestPairSum:
    def test_dimer_matches_the_closed_form(self):
        potential = R.RigidIon()
        r = 1.62
        term = R.BKS_SILICA.pairs[("O", "Si")]
        shift = term.A_eV * math.exp(-term.b_per_A * 10.0) - term.C_eV_A6 / 10.0 ** 6
        expected = (COULOMB_K_EV_A * 2.4 * -1.2 / r + term.A_eV * math.exp(-term.b_per_A * r)
                    - term.C_eV_A6 / r ** 6 - shift)
        energy, forces = potential.energy_and_forces(dimer(r))
        assert energy == pytest.approx(expected, rel=1e-12)
        slope = (-COULOMB_K_EV_A * 2.4 * -1.2 / r ** 2
                 - term.A_eV * term.b_per_A * math.exp(-term.b_per_A * r)
                 + 6 * term.C_eV_A6 / r ** 7)
        assert forces[1, 2] == pytest.approx(-slope, rel=1e-10)
        assert forces[0] == pytest.approx(-forces[1], abs=1e-12)

    def test_short_range_equals_an_explicit_image_sum(self):
        structure = shaken(quartz())
        potential = R.RigidIon()
        energy, forces, closest = potential.short_range(structure)
        cutoff = R.BKS_SILICA.cutoff_A
        matrix = structure.cell.matrix
        reach = [int(math.ceil(cutoff / (abs(np.linalg.det(matrix)) / np.linalg.norm(
            np.cross(matrix[(k + 1) % 3], matrix[(k + 2) % 3]))))) + 1 for k in range(3)]
        symbols = ["Si" if z == 14 else "O" for z in structure.numbers]
        expected_e = 0.0
        expected_f = np.zeros_like(forces)
        for shift in itertools.product(*[range(-n, n + 1) for n in reach]):
            offset = np.array(shift) @ matrix
            for i, j in itertools.product(range(len(structure)), repeat=2):
                if i == j and not any(shift):
                    continue
                term = R.BKS_SILICA.pairs.get(R.pair_key(symbols[i], symbols[j]))
                if term is None:
                    continue
                vector = structure.positions[j] + offset - structure.positions[i]
                d = float(np.linalg.norm(vector))
                if d >= cutoff:
                    continue
                expected_e += 0.5 * (term.A_eV * math.exp(-term.b_per_A * d)
                                     - term.C_eV_A6 / d ** 6
                                     - (term.A_eV * math.exp(-term.b_per_A * cutoff)
                                        - term.C_eV_A6 / cutoff ** 6))
                slope = (-term.A_eV * term.b_per_A * math.exp(-term.b_per_A * d)
                         + 6 * term.C_eV_A6 / d ** 7)
                expected_f[i] += slope * vector / d
        assert energy == pytest.approx(expected_e, rel=1e-12)
        assert np.allclose(forces, expected_f, rtol=0, atol=1e-10)
        assert set(closest) == {"O-Si", "O-O"}

    def test_coulomb_part_is_the_electrostatics_module(self):
        structure = shaken(quartz())
        potential = R.RigidIon()
        energy, forces = potential.energy_and_forces(structure)
        e_short, f_short, _ = potential.short_range(structure)
        model = es.ChargeModel.per_element({"Si": 2.4, "O": -1.2}, source="check")
        coulomb = es.compute(structure, model, es.EwaldSettings(check_convergence=False))
        assert energy == pytest.approx(e_short + coulomb.energy_eV, rel=1e-14)
        assert np.allclose(forces, f_short + coulomb.forces_eV_A, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("build", ["bulk", "slab", "cluster"])
    def test_forces_are_the_energy_gradient(self, build):
        structure = shaken(quartz())
        if build == "slab":
            structure = Structure(structure.numbers, structure.positions,
                                  Cell(structure.cell.matrix * [[1], [1], [3]],
                                       (True, True, False)))
        elif build == "cluster":
            structure = Structure(structure.numbers, structure.positions + 5.0,
                                  Cell(np.eye(3) * 20.0, (False, False, False)))
        potential = R.RigidIon()
        _, forces = potential.energy_and_forces(structure)
        assert np.abs(forces - finite_difference(potential, structure)).max() < 1e-6
        assert np.abs(forces.sum(axis=0)).max() < 1e-9

    def test_energy_is_continuous_at_the_cutoff(self):
        potential = R.RigidIon(R.BKS_SILICA.with_cutoff(6.0))
        inside = potential.energy(dimer(6.0 - 1e-9, 8, 8))
        outside = potential.energy(dimer(6.0 + 1e-9, 8, 8))
        assert inside == pytest.approx(outside, abs=1e-8)


class TestBarrier:
    def test_barrier_is_the_inner_maximum_of_the_pair_interaction(self):
        for name, found in R.barriers(R.BKS_SILICA).items():
            a, b = name.split("-")
            r_b = found["r_A"]
            here = R.pair_interaction(R.BKS_SILICA, a, b, r_b)
            assert here == pytest.approx(found["height_eV"], rel=1e-12)
            assert R.pair_interaction(R.BKS_SILICA, a, b, r_b - 1e-3) < here
            assert R.pair_interaction(R.BKS_SILICA, a, b, r_b + 1e-3) < here
        assert set(R.barriers(R.BKS_SILICA)) == {"O-Si", "O-O"}
        assert 1.1 < R.barriers(R.BKS_SILICA)["O-Si"]["r_A"] < 1.3

    def test_no_dispersion_term_means_no_barrier(self):
        p = R.RigidIonParameters("x", "x", {"Na": 1.0, "Cl": -1.0},
                                 {("Na", "Cl"): R.BuckinghamPair(1000.0, 3.0, 0.0)}, "test")
        assert R.barriers(p) == {}

    def test_a_pair_inside_the_barrier_is_refused(self):
        potential = R.RigidIon()
        inside = dimer(1.15)
        with pytest.raises(R.RigidIonCollapse, match="Buckingham catastrophe"):
            potential.energy_and_forces(inside)
        ok, why = potential.supports(inside)
        assert not ok and "#1 (Si)" in why and "#2 (O)" in why
        assert potential.supports(dimer(1.25))[0]

    def test_just_outside_the_barrier_the_pair_is_pushed_apart(self):
        from scipy.optimize import minimize_scalar

        out = ClassicalSolver(R.RigidIon()).relax(dimer(1.2), fmax_eV_A=1e-5, max_steps=4000)
        assert out.convergence.converged
        d = np.linalg.norm(out.structure.positions[1] - out.structure.positions[0])
        best = minimize_scalar(lambda r: R.pair_interaction(R.BKS_SILICA, "Si", "O", r),
                               bounds=(1.25, 3.0), method="bounded", options={"xatol": 1e-10})
        assert d == pytest.approx(best.x, abs=1e-5)

    def test_dynamics_that_crosses_the_barrier_is_stopped(self):
        start = dimer(1.6)
        start.velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.5]])
        with pytest.raises(R.RigidIonCollapse):
            ClassicalSolver(R.RigidIon()).dynamics(start, steps=50, dt_fs=0.1,
                                                   initialise_velocities=False)


class TestRefusals:
    def test_unknown_element(self):
        s = quartz()
        s.numbers[0] = 13
        ok, why = R.RigidIon().supports(s)
        assert not ok and "no parameters for Al" in why

    def test_net_charge_in_a_crystal(self):
        s = quartz()
        keep = np.arange(len(s)) != int(np.flatnonzero(s.numbers == 8)[0])
        charged = Structure(s.numbers[keep], s.positions[keep], s.cell)
        ok, why = R.RigidIon().supports(charged)
        assert not ok and "net charge of +1.2 e" in why
        with pytest.raises(UnsupportedSystem):
            R.RigidIon().energy_and_forces(charged)

    def test_wire_geometry(self):
        s = quartz()
        wire = Structure(s.numbers, s.positions, Cell(s.cell.matrix, (False, False, True)))
        ok, why = R.RigidIon().supports(wire)
        assert not ok and "one-dimensional" in why.lower()

    def test_a_charged_cluster_is_allowed(self):
        s = Structure([14, 8], np.array([[5.0, 5, 5], [5, 5, 6.6]]),
                      Cell(np.eye(3) * 12.0, (False, False, False)))
        assert R.RigidIon().supports(s)[0]


class TestSolver:
    def test_registered_and_recommended_for_silica(self):
        assert "rigid-ion/bks-silica" in registry.available()
        solver = registry.create("rigid-ion/bks-silica")
        assert solver.name == "classical/rigid-ion/bks-silica"
        material = default_library().get("silicon_dioxide")
        assert registry.resolve_recommended(material, "relax") == "rigid-ion/bks-silica"
        assert registry.resolve_recommended(material, "dynamics") == "rigid-ion/bks-silica"

    def test_single_point_records_checks_and_provenance(self):
        out = registry.create("rigid-ion/bks-silica").single_point(quartz())
        energy = out.results["energy"]
        checks = energy.extra["potential_checks"]
        parts = checks["components_eV"]
        assert parts["short_range"] + parts["coulomb"] == pytest.approx(energy.value, rel=1e-9)
        assert checks["ewald_converged"] is True
        assert checks["ewald_check"]["energy_difference_eV"] < 1e-5
        assert checks["barrier_margin_A"]["O-Si"] > 0.3
        prov = energy.provenance
        assert prov.parameters["parameter_digest"] == R.BKS_SILICA.digest()
        assert R.BKS_REFERENCE in prov.references
        assert any("Buckingham barrier" in a for a in prov.approximations)

    def test_relax_keeps_the_crystal_symmetry(self):
        out = registry.create("rigid-ion/bks-silica").relax(shaken(quartz(), amplitude=0.0),
                                                            fmax_eV_A=1e-4, max_steps=2000)
        assert out.convergence.converged
        assert out.results["energy"].extra["energy_change_eV"] < 0
        assert out.results["energy"].extra["potential_checks"]["ewald_converged"]

    def test_electrostatics_solver_points_at_the_rigid_ion_model(self):
        from materia.solvers.electrostatics import ElectrostaticsSolver
        refused = ElectrostaticsSolver().relax(quartz()).results["relaxed_structure"]
        assert "rigid-ion/bks-silica" in refused.unsupported_reason
