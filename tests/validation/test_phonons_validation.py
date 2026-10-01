"""Finite-displacement harmonic modes against closed forms and an independent code.

* A Lennard-Jones dimer: ``omega = sqrt(k / mu)`` with
  ``k = 36 2^(2/3) epsilon / sigma^2`` at ``r0 = 2^(1/6) sigma``.
* An fcc Lennard-Jones crystal in a 2 x 2 x 2 conventional supercell: its 96
  Gamma-point modes equal the Born-von Karman dynamical matrices of the fcc
  lattice at the 32 wavevectors commensurate with the supercell,
  ``D(q) = (1/m) sum_R [1 - cos(q.R)] [phi'' R^R^ + (phi'/R)(1 - R^R^)]``,
  summed independently over lattice vectors (Born and Huang 1954).  This
  checks the tension terms, mass weighting and the folding of atoms onto their
  own images.
* An Einstein solid: every mode at ``sqrt(k / m)``.
* ASE's ``Vibrations`` class, an independent implementation of the same
  central differences, on a relaxed EMT copper cluster.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import phonons as P
from materia.physics.potentials import Harmonic, LennardJones

pytestmark = pytest.mark.validation

EPS, SIG, CUTOFF, A = 0.0103, 3.40, 8.5, 5.30
FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def thz(eigenvalue):
    return np.sign(eigenvalue) * np.sqrt(np.abs(eigenvalue) * P.EIGENVALUE_TO_RAD_S2) \
        / (2 * math.pi * 1e12)


def test_dimer_closed_form():
    r0 = 2 ** (1 / 6) * SIG
    s = Structure([18, 18], np.array([[5, 5, 5], [5, 5, 5 + r0]]),
                  Cell(np.eye(3) * 20, (False, False, False)))
    result = P.harmonic_analysis(s, LennardJones({"Ar": (EPS, SIG)}, cutoff_A=12.0),
                                 P.PhononSettings(stencil=4))
    k = 36 * 2 ** (2 / 3) * EPS / SIG ** 2
    assert result.frequencies_THz[-1] == pytest.approx(thz(2 * k / s.masses()[0]), rel=1e-7)


def fcc_supercell(n):
    grid = np.array(list(itertools.product(range(n), repeat=3)))
    frac = ((grid[:, None, :] + FCC[None]) / n).reshape(-1, 3)
    cell = np.eye(3) * A * n
    return Structure([18] * len(frac), frac @ cell, Cell(cell, (True, True, True)))


def born_von_karman(n, mass):
    vectors = np.array([(np.array(ijk) + b) * A
                        for ijk in itertools.product(range(-3, 4), repeat=3) for b in FCC])
    d = np.linalg.norm(vectors, axis=1)
    keep = (d > 1e-9) & (d < CUTOFF)
    vectors, d = vectors[keep], d[keep]
    unit = vectors / d[:, None]
    first = 4 * EPS * (-12 * SIG ** 12 / d ** 13 + 6 * SIG ** 6 / d ** 7)
    second = 4 * EPS * (156 * SIG ** 12 / d ** 14 - 42 * SIG ** 6 / d ** 8)
    wavevectors = []
    for ijk in itertools.product(range(2 * n), repeat=3):
        q = np.array(ijk) / n
        if not any(np.allclose(q - p, np.round(q - p))
                   and len({int(round(x)) % 2 for x in q - p}) == 1 for p in wavevectors):
            wavevectors.append(q)
    frequencies = []
    for q in wavevectors:
        phase = 1 - np.cos(vectors @ (2 * math.pi / A * q))
        matrix = (np.einsum("r,r,ra,rb->ab", phase, second - first / d, unit, unit)
                  + np.eye(3) * np.sum(phase * first / d)) / mass
        frequencies.extend(thz(np.linalg.eigvalsh(matrix)))
    return len(wavevectors), np.sort(frequencies)


def test_fcc_gamma_modes_equal_the_lattice_sum_at_commensurate_wavevectors():
    s = fcc_supercell(2)
    result = P.harmonic_analysis(s, LennardJones({"Ar": (EPS, SIG)}, cutoff_A=CUTOFF),
                                 P.PhononSettings(stencil=4))
    count, expected = born_von_karman(2, s.masses()[0])
    assert count == len(s)
    assert np.abs(np.sort(result.frequencies_THz) - expected).max() < 1e-6
    assert result.kinds.count("rigid-body") == 3


def test_einstein_solid():
    s = fcc_supercell(1)
    result = P.harmonic_analysis(s, Harmonic(s.positions.copy(), k_eV_A2=1.5),
                                 P.PhononSettings(sum_rule="none"))
    assert np.allclose(result.frequencies_THz, thz(1.5 / s.masses()[0]), rtol=1e-10)


def test_matches_ase_vibrations(tmp_path, monkeypatch):
    from ase.calculators.emt import EMT
    from ase.cluster import Octahedron
    from ase.optimize import BFGS
    from ase.vibrations import Vibrations

    atoms = Octahedron("Cu", 3)
    atoms.center(vacuum=6.0)
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-5)
    structure = Structure(atoms.numbers, atoms.positions,
                          Cell(np.array(atoms.cell), (False, False, False)))
    mine = P.harmonic_analysis(structure, EMT(), P.PhononSettings(sum_rule="none"),
                               name="ase/EMT")
    monkeypatch.chdir(tmp_path)
    vibrations = Vibrations(atoms, delta=0.01, nfree=2)
    vibrations.run()
    energies = vibrations.get_energies()
    theirs = np.array([e.real if abs(e.imag) < 1e-12 else -e.imag for e in energies])
    theirs = np.sort(theirs) * 1e3 / P.THZ_TO_MEV
    assert np.abs(np.sort(mine.frequencies_THz) - theirs).max() < 1e-6
