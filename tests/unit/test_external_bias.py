import numpy as np
import pytest

from materia.core_model.structure import Structure
from materia.physics.external_bias import (
    ExternalBiasError,
    clear_external_biases,
    energy_and_forces,
    external_biases,
    potential_with_external_bias,
    set_constant_force,
    set_harmonic_restraint,
)
from materia.physics.potentials import Harmonic


def structure():
    return Structure([14, 14], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))


def test_constant_force_has_consistent_energy_gradient():
    atoms = structure()
    atom_id = int(atoms.ids[1])
    set_constant_force(atoms, atom_id, [0.4, -0.2, 0.1])
    first_energy, first_forces = energy_and_forces(atoms)
    atoms.positions[1, 0] += 0.25
    second_energy, second_forces = energy_and_forces(atoms)
    assert first_energy == pytest.approx(0.0)
    assert second_energy - first_energy == pytest.approx(-0.1)
    assert first_forces[1].tolist() == pytest.approx([0.4, -0.2, 0.1])
    np.testing.assert_allclose(second_forces, first_forces, rtol=1e-6, atol=1e-12)


def test_harmonic_restraint_has_consistent_energy_and_force():
    atoms = structure()
    atom_id = int(atoms.ids[0])
    set_harmonic_restraint(atoms, atom_id, 2.0, [1.0, 0.0, 0.0])
    energy, forces = energy_and_forces(atoms)
    assert energy == pytest.approx(1.0)
    assert forces[0].tolist() == pytest.approx([2.0, 0.0, 0.0])
    assert forces[1].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_composed_potential_adds_bias_without_changing_base():
    atoms = structure()
    reference = atoms.positions.copy()
    atoms.positions[0, 0] = 0.5
    set_constant_force(atoms, int(atoms.ids[0]), [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    base = Harmonic(reference, 2.0)
    biased = potential_with_external_bias(base, atoms)
    energy, forces = biased.energy_and_forces(atoms)
    assert energy == pytest.approx(-0.25)
    assert forces[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert base.energy_and_forces(atoms)[0] == pytest.approx(0.25)


def test_biases_follow_stable_ids_after_reordering():
    atoms = structure()
    target = int(atoms.ids[0])
    set_constant_force(atoms, target, [0.0, 1.0, 0.0])
    reordered = atoms.subset([int(atoms.ids[1]), target])
    _, forces = energy_and_forces(reordered)
    assert forces[reordered.index_of(target)].tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_stale_atom_id_is_refused_instead_of_silently_ignored():
    atoms = structure()
    target = int(atoms.ids[0])
    set_constant_force(atoms, target, [1.0, 0.0, 0.0])
    atoms.remove_atoms([target])
    with pytest.raises(ExternalBiasError, match="No atom with id"):
        energy_and_forces(atoms)


def test_bias_record_round_trips_with_structure():
    atoms = structure()
    target = int(atoms.ids[0])
    set_constant_force(atoms, target, [1.0, 2.0, 3.0])
    set_harmonic_restraint(atoms, target, 4.0)
    restored = Structure.from_dict(atoms.as_dict())
    assert external_biases(restored) == external_biases(atoms)


def test_clear_can_select_atom_and_kind():
    atoms = structure()
    first, second = [int(value) for value in atoms.ids]
    set_constant_force(atoms, first, [1.0, 0.0, 0.0])
    set_harmonic_restraint(atoms, first, 2.0)
    set_constant_force(atoms, second, [0.0, 1.0, 0.0])
    assert clear_external_biases(atoms, [first], "constant_force") == 1
    state = external_biases(atoms)
    assert str(first) not in state["constant_forces"]
    assert str(first) in state["restraints"]
    assert str(second) in state["constant_forces"]


@pytest.mark.parametrize("value", ([1.0, 2.0], [1.0, np.nan, 0.0], [0.0, 0.0, 0.0]))
def test_invalid_constant_force_is_rejected_without_mutation(value):
    atoms = structure()
    with pytest.raises(ExternalBiasError):
        set_constant_force(atoms, int(atoms.ids[0]), value)
    assert "external_biases" not in atoms.info


@pytest.mark.parametrize("spring", (0.0, -1.0, np.inf, True))
def test_invalid_spring_is_rejected_without_mutation(spring):
    atoms = structure()
    with pytest.raises(ExternalBiasError):
        set_harmonic_restraint(atoms, int(atoms.ids[0]), spring)
    assert "external_biases" not in atoms.info
