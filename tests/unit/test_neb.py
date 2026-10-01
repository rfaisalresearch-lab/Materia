"""Nudged elastic band settings, paths, barriers and project persistence."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import neb
from materia.physics.potentials import Potential
from materia.provenance import Fidelity


class DoubleWell(Potential):
    name = "analytic-double-well"
    fidelity = Fidelity.NON_PHYSICAL

    def __init__(self, barrier_eV=2.5, minimum_A=1.0, transverse_eV_A2=3.0):
        self.barrier_eV = barrier_eV
        self.minimum_A = minimum_A
        self.transverse_eV_A2 = transverse_eV_A2

    def energy_and_forces(self, structure):
        x, y, z = structure.positions[0]
        scale = self.barrier_eV / self.minimum_A ** 4
        energy = scale * (x * x - self.minimum_A ** 2) ** 2
        energy += 0.5 * self.transverse_eV_A2 * (y * y + z * z)
        gradient = np.array([
            4.0 * scale * x * (x * x - self.minimum_A ** 2),
            self.transverse_eV_A2 * y,
            self.transverse_eV_A2 * z,
        ])
        return float(energy), -gradient.reshape(1, 3)

    def describe(self):
        return {
            "model": self.name, "fidelity": self.fidelity.value,
            "parameters": {"barrier_eV": self.barrier_eV,
                           "minimum_A": self.minimum_A,
                           "transverse_eV_A2": self.transverse_eV_A2},
            "approximations": ["Analytic validation potential."],
            "references": [],
        }


def endpoints():
    initial = Structure([1], [[-1.0, 0.0, 0.0]], Cell.none())
    final = initial.copy()
    final.positions = [[1.0, 0.0, 0.0]]
    return initial, final


@pytest.mark.parametrize("change, fragment", [
    ({"images": 2}, "images"),
    ({"spring_eV_A2": 0.0}, "spring"),
    ({"fmax_eV_A": float("nan")}, "fmax"),
    ({"max_steps": 0}, "max_steps"),
    ({"interpolation": "spline"}, "interpolation"),
    ({"fire_maxstep_A": 2.0}, "fire_maxstep"),
])
def test_settings_validation(change, fragment):
    with pytest.raises(neb.NEBError, match=fragment):
        neb.NEBSettings(**change).validate()


def test_analytic_double_well_barrier_and_path_are_exact():
    initial, final = endpoints()
    before_initial = initial.positions.copy()
    before_final = final.positions.copy()
    path = neb.run(
        initial, final, DoubleWell(),
        neb.NEBSettings(images=7, interpolation="linear", fmax_eV_A=1e-7,
                        endpoint_fmax_eV_A=1e-10, max_steps=100))
    assert path.convergence.converged
    assert path.barrier_eV == pytest.approx(2.5, abs=1e-10)
    assert path.forward_barrier_eV == pytest.approx(2.5, abs=1e-10)
    assert path.reverse_barrier_eV == pytest.approx(2.5, abs=1e-10)
    assert path.reaction_energy_eV == pytest.approx(0.0, abs=1e-10)
    assert path.saddle_image == 3
    assert np.all(np.diff(path.reaction_coordinate_A) > 0)
    assert np.array_equal(initial.positions, before_initial)
    assert np.array_equal(final.positions, before_final)
    assert path.provenance.origin.value == "calculated"
    assert path.provenance.inputs_digest
    assert set(path.stored_arrays()) == {
        "positions", "energies", "forces", "reaction_coordinate", "history"}


def test_endpoint_mismatch_and_nonstationarity_are_refused():
    initial, final = endpoints()
    wrong = Structure([2], final.positions, final.cell, ids=final.ids)
    with pytest.raises(neb.NEBRefused, match="elements"):
        neb.run(initial, wrong, DoubleWell())
    displaced = final.copy()
    displaced.positions = [[0.7, 0.0, 0.0]]
    with pytest.raises(neb.NEBRefused, match="Endpoint maximum forces"):
        neb.run(initial, displaced, DoubleWell())


def test_unrelaxed_endpoints_and_unconverged_path_are_estimated():
    initial, final = endpoints()
    final.positions = [[0.7, 0.0, 0.0]]
    path = neb.run(
        initial, final, DoubleWell(),
        neb.NEBSettings(images=5, interpolation="linear", max_steps=1,
                        fmax_eV_A=1e-12, allow_unrelaxed_endpoints=True))
    assert path.provenance.origin.value == "estimated"
    assert not path.convergence.converged


def test_cancellation_and_fixed_atom_rules():
    initial, final = endpoints()
    with pytest.raises(neb.NEBCancelled):
        neb.run(initial, final, DoubleWell(), cancelled=lambda: True)
    initial.fixed[:] = True
    final.fixed[:] = True
    with pytest.raises(neb.NEBRefused, match="Every atom is fixed"):
        neb.run(initial, final, DoubleWell())
