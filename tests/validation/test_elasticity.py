"""Elastic constants against an analytic lattice sum and against LAMMPS.

* fcc Lennard-Jones at zero stress: ``C_ijkl = (1/2 Omega) sum_R (phi'' - phi'/R)
  R_i R_j R_k R_l / R^2`` (Born and Huang 1954), with the Cauchy relation
  ``C12 = C44`` that central forces impose.
* fcc Cu with the shipped Zhou (2004) EAM file: the tensor from Materia's EAM
  equals the tensor from the energies LAMMPS computes for the same strained
  cells with the same file (skipped as BLOCKED without LAMMPS).
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import tempfile

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import eam
from materia.physics import elasticity as El
from materia.physics.potentials import LennardJones

pytestmark = pytest.mark.validation
FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def fcc(a, z):
    return Structure([z] * 4, FCC * a, Cell(np.eye(3) * a, (True, True, True)))


def test_lennard_jones_matches_the_lattice_sum():
    from scipy.optimize import minimize_scalar

    eps, sig, rc = 0.0103, 3.40, 8.0
    lj = LennardJones({"Ar": (eps, sig)}, cutoff_A=rc)
    a0 = minimize_scalar(lambda a: lj.energy(fcc(a, 18)), bounds=(5.0, 5.6), method="bounded",
                         options={"xatol": 1e-10}).x
    result = El.elastic_tensor(lj, fcc(a0, 18), strain=1e-3)
    c = result.value
    vectors = np.array([(np.array(i) + b) * a0 for i in itertools.product(range(-3, 4), repeat=3)
                        for b in FCC])
    d = np.linalg.norm(vectors, axis=1)
    keep = (d > 1e-9) & (d < rc)
    vectors, d = vectors[keep], d[keep]
    first = 4 * eps * (-12 * sig ** 12 / d ** 13 + 6 * sig ** 6 / d ** 7)
    second = 4 * eps * (156 * sig ** 12 / d ** 14 - 42 * sig ** 6 / d ** 8)
    weight = (second - first / d) / d ** 2
    omega = a0 ** 3 / 4
    c11 = (weight * vectors[:, 0] ** 4).sum() / (2 * omega) * El.EV_A3_GPA
    c12 = (weight * vectors[:, 0] ** 2 * vectors[:, 1] ** 2).sum() / (2 * omega) * El.EV_A3_GPA
    assert c[0, 0] == pytest.approx(c11, rel=1e-4)
    assert c[0, 1] == pytest.approx(c12, rel=1e-4)
    assert c[3, 3] == pytest.approx(c12, rel=1e-4)
    assert result.extra["born_stable"] and result.extra["strain_check"]["converged"]


def lammps_energy(executable, structure, setfl):
    a = structure.cell.matrix
    lines = ["LAMMPS data", "", f"{len(structure)} atoms", "1 atom types", "",
             f"0 {a[0, 0]:.12f} xlo xhi", f"0 {a[1, 1]:.12f} ylo yhi", f"0 {a[2, 2]:.12f} zlo zhi",
             f"{a[1, 0]:.12f} {a[2, 0]:.12f} {a[2, 1]:.12f} xy xz yz", "", "Masses", "",
             "1 63.546", "", "Atoms # atomic", ""]
    lines += [f"{k + 1} 1 {x:.12f} {y:.12f} {z:.12f}"
              for k, (x, y, z) in enumerate(structure.positions)]
    script = f"""units metal
atom_style atomic
boundary p p p
box tilt large
read_data data.lmp
pair_style eam/alloy
pair_coeff * * {setfl} Cu
thermo_style custom pe
thermo_modify format float %.12f
run 0
"""
    with tempfile.TemporaryDirectory() as work:
        open(os.path.join(work, "data.lmp"), "w").write("\n".join(lines) + "\n")
        open(os.path.join(work, "in.lmp"), "w").write(script)
        done = subprocess.run([executable, "-in", "in.lmp", "-log", "none"], cwd=work,
                              capture_output=True, text=True, timeout=120,
                              env={**os.environ, "OMP_NUM_THREADS": "1"})
    numbers = re.findall(r"^\s*(-?\d+\.\d{6,})\s*$", done.stdout, re.M)
    return float(numbers[-1])


class LammpsEnergy:
    name = "lammps-eam"

    def __init__(self, executable, setfl):
        self.executable, self.setfl = executable, setfl
        from materia.provenance import Fidelity
        self.fidelity = Fidelity.TIER1_CLASSICAL

    def supports(self, structure):
        return True, ""

    def model_label(self):
        return "lammps/eam-alloy"

    def energy_and_forces(self, structure):
        upper = structure.copy()
        matrix = structure.cell.matrix
        q, r = np.linalg.qr(matrix.T)
        signs = np.sign(np.diag(r))
        r = (r * signs[:, None]).T
        upper.cell = Cell(r, structure.cell.pbc)
        upper.positions = structure.positions @ (q * signs[None, :])
        return lammps_energy(self.executable, upper, self.setfl), np.zeros((len(structure), 3))


def test_copper_eam_tensor_equals_lammps():
    from materia.solvers.lammps import environment

    candidates = environment.candidates()
    executable = next((c["path"] for c in candidates if c["kind"] == "executable"), None)
    if executable is None:
        pytest.skip("BLOCKED: no LAMMPS executable is installed.")
    potential = eam.load_shipped("Cu-Zhou04")
    cell = fcc(3.6146336, 29)
    mine = El.elastic_tensor(potential, cell, strain=1e-2)
    theirs = El.elastic_tensor(LammpsEnergy(executable, potential.identity.path), cell,
                               strain=1e-2, allow_stress=True)
    assert mine.extra["strain_check"]["converged"]
    assert np.allclose(mine.value, theirs.value, atol=0.05)
    assert 165 < mine.value[0, 0] < 180 and 115 < mine.value[0, 1] < 130
    assert 70 < mine.value[3, 3] < 85
