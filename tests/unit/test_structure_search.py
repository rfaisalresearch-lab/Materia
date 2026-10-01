import numpy as np
import pytest

from materia.core_model.structure import Structure
from materia.physics.external_bias import set_constant_force
from materia.physics.potentials import Harmonic
from materia.physics.structure_search import (
    BasinHoppingSettings,
    StructureSearchCancelled,
    StructureSearchError,
    StructureSearchRefused,
    basin_hopping,
)


def atoms():
    return Structure(
        [14, 14],
        np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )


def settings(**changes):
    values = {
        "trials": 4,
        "displacement_A": 0.1,
        "temperature_eV": 0.05,
        "fmax_eV_A": 1e-4,
        "relaxation_steps": 400,
        "keep": 3,
        "seed": 9,
    }
    values.update(changes)
    return BasinHoppingSettings(**values)


def test_search_is_reproducible_and_does_not_change_input():
    structure = atoms()
    original = structure.positions.copy()
    potential = Harmonic(original.copy(), 2.0)
    first = basin_hopping(structure, potential, settings())
    second = basin_hopping(structure, potential, settings())
    assert structure.positions.tolist() == original.tolist()
    assert first.history["energy_eV"].tolist() == pytest.approx(
        second.history["energy_eV"].tolist())
    assert first.history["accepted"].tolist() == second.history["accepted"].tolist()
    assert first.best.energy_eV == pytest.approx(0.0, abs=1e-8)
    assert first.diagnostics["global_minimum_proven"] is False


def test_search_returns_ranked_deduplicated_minima_and_arrays():
    structure = atoms()
    result = basin_hopping(
        structure, Harmonic(structure.positions.copy(), 1.0), settings())
    energies = [minimum.energy_eV for minimum in result.minima]
    assert energies == sorted(energies)
    assert [minimum.rank for minimum in result.minima] == list(range(1, len(energies) + 1))
    arrays = result.stored_arrays()
    assert arrays["candidate_positions"].data.shape[1:] == (2, 3)
    assert arrays["trial_energy"].data.shape == (4,)
    assert result.results()["best_structure"].value is not result.best.structure


def test_search_cancellation_returns_no_partial_result():
    structure = atoms()
    with pytest.raises(StructureSearchCancelled):
        basin_hopping(
            structure, Harmonic(structure.positions.copy()), settings(),
            cancelled=lambda: True)


def test_search_refuses_all_fixed_atoms():
    structure = atoms()
    structure.fixed[:] = True
    with pytest.raises(StructureSearchRefused, match="movable atom"):
        basin_hopping(structure, Harmonic(structure.positions.copy()), settings())


def test_search_refuses_external_biases():
    structure = atoms()
    set_constant_force(structure, int(structure.ids[0]), [1.0, 0.0, 0.0])
    with pytest.raises(StructureSearchRefused, match="external forces"):
        basin_hopping(structure, Harmonic(structure.positions.copy()), settings())


@pytest.mark.parametrize(
    "change",
    (
        {"trials": 0},
        {"trials": 2, "keep": 4},
        {"displacement_A": 0.0},
        {"temperature_eV": -1.0},
        {"fmax_eV_A": np.nan},
        {"relaxation_steps": True},
        {"seed": 1.5},
    ),
)
def test_invalid_settings_are_rejected(change):
    with pytest.raises(StructureSearchError):
        settings(**change).validate()
