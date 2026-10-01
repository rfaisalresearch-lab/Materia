import numpy as np
import pytest

from materia.core_model.structure import Structure
from materia.physics.external_bias import set_constant_force
from materia.physics.model_ensemble import (
    ModelEnsembleError,
    energy_change_ensemble,
    force_ensemble,
)
from materia.physics.potentials import Harmonic


def one_atom(position=1.0):
    return Structure([14], np.array([[position, 0.0, 0.0]]), ids=[8])


def potentials():
    reference = np.zeros((1, 3))
    return [Harmonic(reference, 1.0), Harmonic(reference, 2.0)]


def test_force_ensemble_reports_component_and_vector_disagreement():
    result = force_ensemble(one_atom(), potentials(), labels=["soft", "stiff"])
    assert result.labels == ("soft", "stiff")
    assert result.energies_eV.tolist() == pytest.approx([0.5, 1.0])
    assert result.mean_forces_eV_A[0].tolist() == pytest.approx([-1.5, 0.0, 0.0])
    assert result.component_std_eV_A[0, 0] == pytest.approx(np.sqrt(0.5))
    assert result.force_disagreement_eV_A.tolist() == pytest.approx([0.5])
    assert result.pairwise_force_rms_eV_A[0, 1] == pytest.approx(1.0 / np.sqrt(3.0))
    assert result.metrics["raw_energy_spread_is_uncertainty"] is False


def test_force_ensemble_arrays_preserve_model_axis_and_ids():
    result = force_ensemble(one_atom(), potentials(), labels=["a", "b"])
    arrays = result.stored_arrays()
    assert arrays["model_forces"].data.shape == (2, 1, 3)
    assert arrays["pairwise_force_rms"].data.shape == (2, 2)
    assert arrays["model_forces"].meta["atom_ids"] == [8]
    assert arrays["model_forces"].meta["models"] == ["a", "b"]


def test_energy_change_uses_within_model_differences():
    before = one_atom(0.0)
    after = one_atom(1.0)
    result = energy_change_ensemble(
        before, after, potentials(), labels=["soft", "stiff"])
    assert result.changes_eV.tolist() == pytest.approx([0.5, 1.0])
    assert result.mean_change_eV == pytest.approx(0.75)
    assert result.std_change_eV == pytest.approx(np.sqrt(0.125))
    assert result.metrics["reference_offsets_cancelled_within_each_model"] is True
    assert result.metrics["sign_agreement"] is True


def test_energy_change_refuses_changed_atom_identity():
    before = one_atom(0.0)
    after = Structure([14], np.array([[1.0, 0.0, 0.0]]), ids=[9])
    with pytest.raises(ModelEnsembleError, match="atom-id and element order"):
        energy_change_ensemble(before, after, potentials(), labels=["a", "b"])


def test_ensemble_refuses_external_mechanical_bias():
    structure = one_atom()
    set_constant_force(structure, 8, [1.0, 0.0, 0.0])
    with pytest.raises(ModelEnsembleError, match="mechanical biases"):
        force_ensemble(structure, potentials(), labels=["a", "b"])


def test_ensemble_requires_two_uniquely_named_models():
    with pytest.raises(ModelEnsembleError, match="at least two"):
        force_ensemble(one_atom(), potentials()[:1])
    with pytest.raises(ModelEnsembleError, match="unique"):
        force_ensemble(one_atom(), potentials(), labels=["same", "same"])
