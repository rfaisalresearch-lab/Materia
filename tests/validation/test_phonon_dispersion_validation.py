"""Independent physical validation of periodic real-space phonons.

The silicon Gamma frequencies are compared with Materia's separately implemented
full-cell harmonic Hessian.  The fcc Lennard-Jones dispersion is compared with a
direct Born-von Karman lattice sum of the analytic central-force Hessian.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

pytest.importorskip("ase")
pytestmark = pytest.mark.validation

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import phonon_dispersion as D
from materia.physics import phonons as H
from materia.physics.potentials import LennardJones, StillingerWeber


def silicon_primitive(a=5.431):
    cell = np.array([
        [0.0, a / 2, a / 2],
        [a / 2, 0.0, a / 2],
        [a / 2, a / 2, 0.0],
    ])
    return Structure(
        [14, 14], np.array([[0.0, 0.0, 0.0], [a / 4, a / 4, a / 4]]),
        Cell(cell, (True, True, True)))


def test_silicon_gamma_agrees_with_existing_harmonic_engine():
    structure = silicon_primitive()
    potential = StillingerWeber("Si")
    periodic = D.periodic_phonon_analysis(
        structure, potential,
        D.PeriodicPhononSettings(
            supercell=(3, 3, 3), q_path=((0.0, 0.0, 0.0),),
            q_labels=("G",), dos_mesh=(2, 2, 2), dos_points=201,
            dos_width_THz=0.05))
    gamma = H.harmonic_analysis(
        structure, potential,
        H.PhononSettings(displacement_A=0.01, stencil=2, sum_rule="translational"))
    difference = np.abs(
        np.sort(periodic.frequencies_THz[0]) - np.sort(gamma.frequencies_THz))
    assert difference.max() < 1e-4
    assert periodic.mode_kinds[0].count("acoustic") == 3
    assert gamma.kinds.count("rigid-body") == 3


EPSILON = 0.0103
SIGMA = 3.40
CUTOFF = 8.5
LATTICE = 5.30


def fcc_primitive():
    cell = np.array([
        [0.0, LATTICE / 2, LATTICE / 2],
        [LATTICE / 2, 0.0, LATTICE / 2],
        [LATTICE / 2, LATTICE / 2, 0.0],
    ])
    return Structure([18], [[0.0, 0.0, 0.0]], Cell(cell, (True, True, True)))


def analytic_fcc_frequencies(structure, q_points):
    cell = structure.cell.matrix
    vectors = []
    for integer in itertools.product(range(-8, 9), repeat=3):
        lattice_vector = np.asarray(integer, dtype=int)
        cartesian = lattice_vector @ cell
        distance = float(np.linalg.norm(cartesian))
        if 1e-9 < distance < CUTOFF:
            vectors.append((lattice_vector, cartesian, distance))
    mass = structure.masses()[0]
    frequencies = []
    for q_point in q_points:
        dynamical = np.zeros((3, 3), dtype=float)
        for lattice_vector, cartesian, distance in vectors:
            unit = cartesian / distance
            first = 4 * EPSILON * (
                -12 * SIGMA ** 12 / distance ** 13
                + 6 * SIGMA ** 6 / distance ** 7)
            second = 4 * EPSILON * (
                156 * SIGMA ** 12 / distance ** 14
                - 42 * SIGMA ** 6 / distance ** 8)
            hessian = (
                (second - first / distance) * np.outer(unit, unit)
                + (first / distance) * np.eye(3))
            phase = 1.0 - math.cos(2.0 * math.pi * float(q_point @ lattice_vector))
            dynamical += phase * hessian / mass
        eigenvalues = np.linalg.eigvalsh(dynamical)
        signed = np.sign(eigenvalues) * np.sqrt(
            np.abs(eigenvalues) * H.EIGENVALUE_TO_RAD_S2) / (2.0 * math.pi * 1e12)
        frequencies.append(signed)
    return np.asarray(frequencies)


def test_fcc_lennard_jones_matches_analytic_born_von_karman_dispersion():
    structure = fcc_primitive()
    potential = LennardJones({"Ar": (EPSILON, SIGMA)}, cutoff_A=CUTOFF)
    q_points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.2, 0.1],
    ])
    result = D.periodic_phonon_analysis(
        structure, potential,
        D.PeriodicPhononSettings(
            supercell=(6, 6, 6),
            q_path=tuple(tuple(float(v) for v in q) for q in q_points),
            dos_mesh=(2, 2, 2), displacement_A=0.005,
            dos_points=201, dos_width_THz=0.02))
    expected = analytic_fcc_frequencies(structure, q_points)
    difference = np.abs(
        np.sort(result.frequencies_THz, axis=1) - np.sort(expected, axis=1))
    assert difference.max() < 1e-4
    assert result.diagnostics["raw_sum_rule_violation_relative"] < 1e-8
    assert result.diagnostics["dos_integral_states"] == pytest.approx(3.0, abs=1e-10)
