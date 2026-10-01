"""Native Tersoff against LAMMPS pair_style tersoff with the same parameter files.

Skipped as BLOCKED when LAMMPS or the parameter files are unavailable.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

import numpy as np
import pytest

from materia.elements import periodic_table as pt
from materia.materials import default_library
from materia.physics import tersoff as T
from materia.solvers import registry
from materia.structure_builder.lattice import bulk

pytestmark = pytest.mark.validation


def lammps_executable():
    from materia.solvers.lammps import environment
    return next((c["path"] for c in environment.candidates() if c["kind"] == "executable"),
                None)


def lammps(executable, structure, filename, elements):
    a = structure.cell.matrix
    types = {e: k + 1 for k, e in enumerate(elements)}
    lines = ["data", "", f"{len(structure)} atoms", f"{len(elements)} atom types", "",
             f"0 {a[0, 0]:.12f} xlo xhi", f"0 {a[1, 1]:.12f} ylo yhi",
             f"0 {a[2, 2]:.12f} zlo zhi", "", "Masses", ""]
    lines += [f"{types[e]} {pt.mass(e)}" for e in elements] + ["", "Atoms # atomic", ""]
    lines += [f"{k + 1} {types[pt.symbol(int(z))]} {x:.12f} {y:.12f} {w:.12f}"
              for k, (z, (x, y, w)) in enumerate(zip(structure.numbers, structure.positions))]
    script = ("units metal\natom_style atomic\nboundary p p p\nread_data data.lmp\n"
              f"pair_style tersoff\npair_coeff * * {filename} {' '.join(elements)}\n"
              "thermo_style custom pe\nthermo_modify format float %.12f\n"
              "dump d all custom 1 f.dump id fx fy fz\n"
              "dump_modify d format float %.12f sort id\nrun 0\n")
    with tempfile.TemporaryDirectory() as work:
        open(os.path.join(work, "data.lmp"), "w").write("\n".join(lines) + "\n")
        open(os.path.join(work, "in.lmp"), "w").write(script)
        done = subprocess.run([executable, "-in", "in.lmp", "-log", "none"], cwd=work,
                              capture_output=True, text=True, timeout=120,
                              env={**os.environ, "OMP_NUM_THREADS": "1"})
        energy = float(re.findall(r"^\s*(-?\d+\.\d{6,})\s*$", done.stdout, re.M)[-1])
        forces = np.loadtxt(os.path.join(work, "f.dump"), skiprows=9)[:, 1:]
    return energy, forces


@pytest.mark.parametrize("material, filename, elements", [
    ("silicon", "Si.tersoff", ["Si"]),
    ("silicon_carbide_3c", "SiC.tersoff", ["Si", "C"]),
])
def test_energy_and_forces_equal_lammps(material, filename, elements):
    executable = lammps_executable()
    if executable is None:
        pytest.skip("BLOCKED: no LAMMPS executable is installed.")
    try:
        path = T.fetch_lammps_file(filename)
    except Exception as exc:
        pytest.skip(f"BLOCKED: {filename} could not be obtained: {exc}")
    structure = bulk(default_library().get(material), (2, 2, 2))
    structure.positions = structure.positions + np.random.default_rng(1).normal(
        0, 0.06, structure.positions.shape)
    potential = T.Tersoff(filename)
    energy, forces = potential.energy_and_forces(structure)
    their_energy, their_forces = lammps(executable, structure, str(path), elements)
    assert energy == pytest.approx(their_energy, abs=1e-8)
    assert np.abs(forces - their_forces).max() < 1e-8
    h = 1e-5
    plus, minus = structure.copy(), structure.copy()
    plus.positions[2, 0] += h
    minus.positions[2, 0] -= h
    assert -(potential.energy(plus) - potential.energy(minus)) / (2 * h) == pytest.approx(
        forces[2, 0], abs=1e-6)


def test_silicon_cohesive_energy_and_refusal():
    try:
        T.fetch_lammps_file("Si.tersoff")
    except Exception as exc:
        pytest.skip(f"BLOCKED: Si.tersoff could not be obtained: {exc}")
    solver = registry.create("tersoff/si")
    crystal = bulk(default_library().get("silicon"))
    energy = solver.single_point(crystal)["energy"].value / len(crystal)
    assert energy == pytest.approx(-4.63, abs=0.01)
    ok, why = solver.potential.supports(bulk(default_library().get("copper")))
    assert not ok and "no Tersoff parameters for Cu" in why
