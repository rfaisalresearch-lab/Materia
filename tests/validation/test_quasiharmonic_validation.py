"""Quasi-harmonic thermal expansion: parity with phonopy and agreement with experiment."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import eam
from materia.physics import quasiharmonic as Q
from materia.physics.potentials import UnsupportedSystem

pytestmark = pytest.mark.validation
SCALES = np.linspace(0.97, 1.07, 6)
TEMPERATURES = np.arange(0, 401, 25.0)


def copper(a=3.615):
    cell = Cell(np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a, (True, True, True))
    return Structure([29], [[0, 0, 0]], cell)


def phonopy_qha(pot):
    phonopy = pytest.importorskip("phonopy")
    from phonopy.structure.atoms import PhonopyAtoms

    volumes, energies, free, entropy, cv = [], [], [], [], []
    for s in SCALES:
        c = Q.scaled(copper(), s)
        unit = PhonopyAtoms(symbols=["Cu"], cell=c.cell.matrix, scaled_positions=[[0, 0, 0]],
                            masses=c.masses())
        ph = phonopy.Phonopy(unit, supercell_matrix=np.diag([8, 8, 8]))
        ph.generate_displacements(distance=0.01)
        ph.forces = [pot.energy_and_forces(Structure(sc.numbers, sc.positions,
                                                     Cell(sc.cell, (True,) * 3)))[1]
                     for sc in ph.supercells_with_displacements]
        ph.produce_force_constants()
        ph.symmetrize_force_constants()
        ph.run_mesh([16, 16, 16], is_gamma_center=False)
        ph.run_thermal_properties(t_min=0, t_max=400, t_step=25)
        tp = ph.get_thermal_properties_dict()
        volumes.append(c.cell.volume)
        energies.append(pot.energy_and_forces(c)[0])
        free.append(tp["free_energy"])
        entropy.append(tp["entropy"])
        cv.append(tp["heat_capacity"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return phonopy.PhonopyQHA(volumes=volumes, electronic_energies=energies,
                                  temperatures=tp["temperatures"], free_energy=np.array(free).T,
                                  cv=np.array(cv).T, entropy=np.array(entropy).T,
                                  eos="birch_murnaghan", t_max=375, verbose=False)


def test_copper_eam_matches_phonopy_qha():
    pot = eam.load_shipped("Cu-Zhou04")
    rows = Q.quasiharmonic(copper(), pot, SCALES, TEMPERATURES, (8, 8, 8), (16, 16, 16)).value
    reference = phonopy_qha(pot)
    ev_per_k_to_j_per_mol_k = 96485.33212
    for i in (4, 8, 12):
        row = rows[i]
        assert row["temperature_K"] == TEMPERATURES[i]
        assert row["volume_A3"] == pytest.approx(reference.volume_temperature[i], rel=1e-5)
        assert row["bulk_modulus_GPa"] == pytest.approx(
            reference.bulk_modulus_temperature[i], rel=2e-3)
        assert row["volumetric_expansion_per_K"] == pytest.approx(
            reference.thermal_expansion[i], rel=2e-3)
        assert row["heat_capacity_p_eV_K"] * ev_per_k_to_j_per_mol_k == pytest.approx(
            reference.heat_capacity_P_polyfit[i], rel=3e-3)


def test_mishin_copper_expansion_near_experiment():
    try:
        pot = eam.load_lammps("Cu_mishin1.eam.alloy")
    except UnsupportedSystem as exc:
        pytest.skip(f"BLOCKED: {exc}")
    rows = Q.quasiharmonic(copper(), pot, np.linspace(0.98, 1.08, 7), [0, 100, 200, 300, 400],
                           (6, 6, 6), (16, 16, 16)).value
    by_t = {r["temperature_K"]: r for r in rows}
    assert by_t[0]["linear_expansion_per_K"] == 0
    assert by_t[300]["linear_expansion_per_K"] == pytest.approx(16.5e-6, rel=0.2)
    assert 1.4 < by_t[300]["gruneisen"] < 2.2
    assert by_t[300]["volume_A3"] > by_t[0]["volume_A3"] > 11.80
