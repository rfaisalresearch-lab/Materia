"""Physical validation of the GPAW driver.

These are **not** software tests.  Each one runs a real self-consistent
calculation and compares its output either against an exact relationship that
must hold, or against a measured value from the primary literature, and states
its tolerance and the reason for it.

They are skipped when GPAW is not installed, because there is then nothing to
validate.  A passing case means the driver reproduces what GPAW computes and
that GPAW's answer sits where the cited reference says it should; it does not
mean a density functional is a correct description of nature.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.solvers.gpaw_driver import GPAWSolver

pytestmark = pytest.mark.validation

H2_EXPERIMENTAL_BOND_A = 0.7414

BOX_A = 7.0

TIGHT = {
    "xc": "PBE", "mode": "fd", "grid_spacing_A": 0.16,
    "occupations": "fixed", "smearing_eV": 0.0,
    "energy_tol_eV_per_electron": 1.0e-5,
    "density_tol_electrons": 1.0e-5,
    "max_iterations": 200,
}


def h2(distance_A: float) -> Structure:
    centre = BOX_A / 2
    return Structure(
        np.array([1, 1]),
        np.array([[centre, centre, centre - distance_A / 2],
                  [centre, centre, centre + distance_A / 2]]),
        Cell(np.eye(3) * BOX_A, (False, False, False)),
    )


@pytest.fixture(scope="module")
def energy_curve(gpaw_environment):
    """Total energy of H2 at five separations, from five real SCF solutions."""
    if gpaw_environment is None:
        pytest.skip("GPAW is not installed, so there is nothing to validate")
    solver = GPAWSolver()
    distances = np.array([0.72, 0.74, 0.76, 0.78, 0.80])
    energies = []
    for distance in distances:
        out = solver.single_point(h2(float(distance)), TIGHT)
        assert out.results["energy"].supported, out.results["energy"].unsupported_reason
        energies.append(float(out.results["energy"].value))
    return distances, np.array(energies)


class TestForcesAreTheGradientOfTheEnergy:
    """The Hellmann-Feynman theorem is an identity, not an approximation.

    The force GPAW reports must equal minus the derivative of the total energy
    it reports, on its own energy surface.  This is the strongest check
    available on the forces path without any reference to experiment, because
    the expected relationship is exact and the two quantities travel through
    different parts of the driver.
    """

    def test_analytic_force_matches_the_numerical_derivative(self, needs_gpaw):
        solver = GPAWSolver()
        step = 0.01
        centre = 0.74
        energies = {}
        force_z = None
        for distance in (centre - step, centre, centre + step):
            out = solver.single_point(h2(distance), TIGHT)
            energies[round(distance, 4)] = float(out.results["energy"].value)
            if abs(distance - centre) < 1e-9:
                force_z = float(out.results["forces"].value[1][2])

        derivative = ((energies[round(centre + step, 4)]
                       - energies[round(centre - step, 4)]) / (2 * step))
        assert force_z == pytest.approx(-derivative, abs=0.02)

    def test_the_forces_on_a_symmetric_dimer_are_equal_and_opposite(self, needs_gpaw):
        """Newton's third law, and a check that atom ordering is not scrambled
        somewhere in the JSON round trip."""
        out = GPAWSolver().single_point(h2(0.74), TIGHT)
        forces = np.asarray(out.results["forces"].value)
        assert forces.shape == (2, 3)
        assert np.allclose(forces[0], -forces[1], atol=1e-3)
        assert abs(forces[0][0]) < 1e-3
        assert abs(forces[0][1]) < 1e-3


class TestHydrogenMoleculeBondLength:
    """PBE against the measured H2 bond length.

    A generalised-gradient functional is not expected to reproduce a
    spectroscopic bond length exactly, and PBE is known to overestimate this
    one.  The case bounds that overestimate and reports it; it does not assert
    agreement with experiment.
    """

    def test_the_energy_curve_has_a_minimum_in_the_sampled_range(self, energy_curve):
        distances, energies = energy_curve
        lowest = int(np.argmin(energies))
        assert 0 < lowest < len(distances) - 1

    def test_the_equilibrium_bond_length_is_within_three_percent(self, energy_curve):
        distances, energies = energy_curve
        coefficients = np.polyfit(distances, energies, 2)
        equilibrium = -coefficients[1] / (2 * coefficients[0])
        deviation = equilibrium - H2_EXPERIMENTAL_BOND_A
        assert 0.70 < equilibrium < 0.80
        assert abs(deviation / H2_EXPERIMENTAL_BOND_A) < 0.03
        assert deviation > 0

    def test_the_curvature_is_positive(self, energy_curve):
        distances, energies = energy_curve
        coefficients = np.polyfit(distances, energies, 2)
        assert coefficients[0] > 0


class TestTheSameCalculationTwiceGivesTheSameNumber:
    """A first-principles result has to be reproducible on the same machine,
    or nothing built on top of it can be trusted."""

    def test_two_runs_agree_to_the_convergence_tolerance(self, needs_gpaw):
        solver = GPAWSolver()
        first = solver.single_point(h2(0.74), TIGHT).results["energy"].value
        second = solver.single_point(h2(0.74), TIGHT).results["energy"].value
        assert first == pytest.approx(second, abs=1e-6)
