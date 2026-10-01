"""Quantum ESPRESSO pw.x driven by Materia, checked against pw.x run directly.

Skipped with BLOCKED when pw.x or the pseudopotentials are unavailable.

* The same silicon calculation written by hand and run directly with pw.x
  gives the same total energy as the Materia adapter.
* Forces on a displaced atom equal central differences of the energy.
* The PBE lattice constant of silicon from an energy-volume fit lies near the
  all-electron PBE value of about 5.47 A (Lejaeghere et al., Science 351
  (2016) aad3000), a plausibility check of pseudopotential and cutoff.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.core_model.units import RYDBERG_EV
from materia.physics import espresso as E
from materia.physics.potentials import UnsupportedSystem

pytestmark = pytest.mark.validation


def silicon(a=5.431, shift=0.0):
    cell = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    return Structure([14, 14], np.array([[0.0, 0.0, shift], [a / 4] * 3]),
                     Cell(cell, (True, True, True)))


@pytest.fixture(scope="module")
def pw():
    found = E.find_pw()
    if found is None:
        pytest.skip("BLOCKED: Quantum ESPRESSO pw.x is not installed.")
    try:
        E.fetch_pseudopotential("Si")
    except Exception as exc:
        pytest.skip(f"BLOCKED: no silicon pseudopotential: {exc}")
    return found


def model():
    return E.EspressoPotential(ecutwfc_Ry=45, ecutrho_Ry=360, kpts=(6, 6, 6))


def test_adapter_equals_pw_run_directly(pw):
    out = model().run(silicon())
    cell = silicon().cell.matrix
    text = f"""&control
 calculation='scf', pseudo_dir='{E.pseudo_dir()}', outdir='./tmp', tprnfor=.true.
/
&system
 ibrav=0, nat=2, ntyp=1, ecutwfc=45, ecutrho=360, occupations='smearing',
 smearing='gaussian', degauss=0.01
/
&electrons
 conv_thr=1e-10, mixing_beta=0.5
/
ATOMIC_SPECIES
Si 28.085 {E.PSLIBRARY_PBE['Si']}
CELL_PARAMETERS angstrom
""" + "\n".join(" ".join(f"{v:.10f}" for v in row) for row in cell) + f"""
ATOMIC_POSITIONS angstrom
Si 0.0 0.0 0.0
Si {5.431 / 4:.10f} {5.431 / 4:.10f} {5.431 / 4:.10f}
K_POINTS automatic
6 6 6 0 0 0
"""
    with tempfile.TemporaryDirectory() as work:
        done = subprocess.run([pw], input=text, capture_output=True, text=True, cwd=work,
                              timeout=1800, env={**os.environ, "OMP_NUM_THREADS": "1"})
    energy = float(re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", done.stdout,
                              re.M)[-1]) * RYDBERG_EV
    assert out["energy_eV"] == pytest.approx(energy, abs=2e-6)
    assert out["scf_iterations"] and out["pw_version"].startswith("v.")
    assert out["pseudopotentials"]["Si"]["sha256"]
    assert np.abs(out["forces_eV_A"]).max() < 1e-4


def test_forces_are_energy_gradients(pw):
    potential = model()
    h = 0.005
    displaced = silicon(shift=0.05)
    forces = potential.run(displaced)["forces_eV_A"]
    plus = potential.energy(silicon(shift=0.05 + h))
    minus = potential.energy(silicon(shift=0.05 - h))
    assert -(plus - minus) / (2 * h) == pytest.approx(forces[0, 2], abs=2e-3)
    assert abs(forces[0, 2]) > 0.05


def test_silicon_lattice_constant(pw):
    potential = model()
    scales = np.array([0.98, 0.99, 1.0, 1.01, 1.02])
    lattice = 5.431 * scales
    energies = [potential.energy(silicon(a)) for a in lattice]
    fit = np.polyfit(lattice, energies, 2)
    a0 = -fit[1] / (2 * fit[0])
    assert 5.44 < a0 < 5.50


def test_refusals(pw):
    ok, why = model().supports(Structure([14], np.zeros((1, 3)),
                                         Cell(np.eye(3) * 5, (True, True, False))))
    assert not ok and "three-dimensional" in why
    low = E.EspressoPotential(ecutwfc_Ry=20, kpts=(2, 2, 2))
    ok, why = low.supports(silicon())
    assert not ok and "below" in why


def test_stress_variable_cell_relaxation_and_bands(pw):
    potential = model()
    compressed = potential.run(silicon(5.431))["stress_eV_A3"]
    assert np.all(np.diag(compressed) < 0) and np.allclose(compressed, compressed[0, 0] * np.eye(3),
                                                            atol=1e-6)
    relaxed = potential.relax(silicon(5.431), variable_cell=True)
    a0 = (4 * relaxed["volume_A3"]) ** (1 / 3)
    assert a0 == pytest.approx(5.4686, abs=0.01)
    assert relaxed["structure"].cell.volume == pytest.approx(relaxed["volume_A3"])
    bands = potential.band_structure(silicon(a0), path="LGXUKG", density=15)
    assert bands["direct"] is False
    assert 0.5 < bands["gap_eV"] < 0.7
    assert bands["labels"] == "LGXUKG" and len(bands["kpts"]) == len(bands["eigenvalues_eV"])


def test_dfpt_gamma_equals_finite_displacement(pw):
    from materia.physics import phonons as P

    potential = E.EspressoPotential(ecutwfc_Ry=45, ecutrho_Ry=360, kpts=(6, 6, 6),
                                    smearing="gaussian", degauss_Ry=0.001)
    crystal = silicon(5.4704)
    dfpt = E.EspressoPhonons(potential).gamma(crystal)
    optical = sorted(dfpt["asr_frequencies_THz"])[-3:]
    assert np.ptp(optical) < 1e-3 and 14.5 < optical[0] < 15.6
    assert np.allclose(sorted(dfpt["asr_frequencies_THz"])[:3], 0.0, atol=1e-3)
    finite = P.harmonic_analysis(crystal, potential, P.PhononSettings(displacement_A=0.01))
    assert np.allclose(np.sort(finite.frequencies_THz)[-3:], optical, atol=0.02)


def test_dfpt_dispersion_has_the_diamond_degeneracies(pw):
    potential = E.EspressoPotential(ecutwfc_Ry=45, ecutrho_Ry=360, kpts=(6, 6, 6),
                                    smearing="gaussian", degauss_Ry=0.001, timeout_s=3000)
    out = E.EspressoPhonons(potential).dispersion(silicon(5.4704), (2, 2, 2), path="GX",
                                                  density=3)
    frequencies = np.array(out["frequencies_THz"])
    assert frequencies.min() > -0.05
    x = frequencies[-1]
    assert x[0] == pytest.approx(x[1], abs=1e-3) and x[2] == pytest.approx(x[3], abs=1e-3)
    assert x[4] == pytest.approx(x[5], abs=1e-3)
    assert np.allclose(frequencies[0][:3], 0.0, atol=1e-3)


def test_projected_dos_invariants(pw):
    potential = E.EspressoPotential(ecutwfc_Ry=45, ecutrho_Ry=360, kpts=(6, 6, 6),
                                    smearing="gaussian", degauss_Ry=0.005)
    out = E.projected_dos(potential, silicon(5.4704), nscf_kpts=(10, 10, 10), emin_eV=-10,
                          emax_eV=12)
    e = np.array(out["energies_eV"])
    dos, pdos = np.array(out["dos"]), np.array(out["pdos_sum"])
    occupied = e <= out["fermi_eV"]
    assert np.trapezoid(dos[occupied], e[occupied]) == pytest.approx(8.0, abs=0.02)
    completeness = np.trapezoid(pdos[occupied], e[occupied]) / np.trapezoid(dos[occupied],
                                                                           e[occupied])
    assert completeness == pytest.approx(1 - out["spilling"], abs=0.01)
    s = sum(np.array(v) for k, v in out["channels"].items() if k.endswith(":s"))
    p = sum(np.array(v) for k, v in out["channels"].items() if k.endswith(":p"))
    top = occupied & (e > out["fermi_eV"] - 2)
    assert np.trapezoid(p[top], e[top]) > 10 * np.trapezoid(s[top], e[top])


def test_fat_bands_give_silicon_orbital_character(pw):
    potential = E.EspressoPotential(ecutwfc_Ry=45, ecutrho_Ry=360, kpts=(6, 6, 6))
    out = E.fat_bands(potential, silicon(5.4704), path="LGX", density=10)
    e = np.array(out["eigenvalues_eV"])
    s, p = np.array(out["character"]["Si:s"]), np.array(out["character"]["Si:p"])
    atoms = [np.array(out["atom_character"][i]) for i in (0, 1)]
    spilling = np.array(out["spilling"])
    gamma = next(i for i, k in enumerate(out["kpts"]) if np.allclose(k, 0))
    assert s[gamma, 0] > 0.95 and p[gamma, 0] < 0.01
    assert np.all(p[gamma, 1:4] > 0.9) and np.all(s[gamma, 1:4] < 0.01)
    assert e[gamma, 1:4] == pytest.approx([e[gamma, 1]] * 3, abs=1e-3)
    assert np.allclose(atoms[0][:, :4].sum(axis=1), atoms[1][:, :4].sum(axis=1), atol=1e-3)
    assert np.all(spilling[:, :4] > -1e-3) and spilling[:, :4].mean() < 0.03
    assert np.allclose(s + p, 1 - spilling, atol=1e-6)


def aluminium_arsenide(a=5.73):
    cell = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a
    return Structure([13, 33], np.array([[0, 0, 0], [.25, .25, .25]]) @ cell,
                     Cell(cell, (True, True, True)))


def test_aluminium_arsenide_dielectric_response(pw):
    potential = E.EspressoPotential(ecutwfc_Ry=40, ecutrho_Ry=320, kpts=(12, 12, 12),
                                    smearing="fixed")
    out = E.EspressoPhonons(potential).dielectric(aluminium_arsenide())
    z = np.array(out["born_charges_e"])
    assert out["charge_sum_rule_violation_e"] < 0.01
    assert z[0, 0, 0] == pytest.approx(2.18, abs=0.05)
    assert np.allclose(z[0], z[0, 0, 0] * np.eye(3), atol=1e-4)
    eps = out["epsilon_infinity"][0][0]
    assert 8.16 < eps < 10.5
    assert out["longitudinal_optical_THz"] == pytest.approx(
        out["longitudinal_optical_dynmat_THz"], rel=1e-4)
    assert out["transverse_optical_THz"] == pytest.approx(10.82, rel=0.05)
    assert out["longitudinal_optical_THz"] == pytest.approx(12.05, rel=0.06)
    assert out["epsilon_static"] > eps
    assert sorted(out["gamma_frequencies_THz"])[-1] == pytest.approx(
        out["transverse_optical_THz"])


def test_dielectric_refuses_smearing(pw):
    with pytest.raises(UnsupportedSystem, match="insulator"):
        E.EspressoPhonons(E.EspressoPotential(ecutwfc_Ry=40, ecutrho_Ry=320)).dielectric(
            aluminium_arsenide())


def test_iron_is_ferromagnetic(pw):
    from materia.core_model.cell import Cell
    from materia.core_model.structure import Structure

    try:
        details = E.describe_pseudopotential(E.fetch_pseudopotential("Fe"))
    except Exception as exc:
        pytest.skip(f"BLOCKED: no iron pseudopotential: {exc}")
    assert details["functional"] == "PBE"
    a = 2.83
    iron = Structure([26], np.zeros((1, 3)), Cell(np.array(
        [[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]]),
        (True, True, True)))
    potential = E.EspressoPotential(ecutwfc_Ry=64, ecutrho_Ry=782, kpts=(8, 8, 8),
                                    smearing="mv", degauss_Ry=0.02, spin_polarized=True,
                                    starting_magnetization={"Fe": 0.5}, timeout_s=3000)
    out = potential.run(iron)
    assert 1.9 < out["magnetization_muB"] < 2.5
