"""STM spectroscopy specification and helpers without GPAW."""

from __future__ import annotations

import numpy as np
import pytest

from materia.experiments.dft import stm_spectroscopy as S


def test_broadened_count_integrates_to_the_states():
    eigen = np.array([[[-1.0, 0.5]]])
    energies = np.linspace(-3, 3, 601)
    dos = S.broadened_count(eigen, np.array([1.0]), 0.0, energies, 0.1)
    assert np.trapezoid(dos, energies) == pytest.approx(4.0, rel=1e-6)
    assert energies[int(np.argmax(dos))] == pytest.approx(-1.0, abs=0.01)


def test_point_spectrum_interpolates():
    stack = np.zeros((2, 4, 4, 4))
    stack[0] = 1.0
    stack[1, :, :, 2] = 2.0
    out = S.point_spectrum(stack, np.diag([4.0, 4.0, 4.0]), [0.0, 1.0], 1.0, 1.0, 1.5)
    assert out["ldos_per_eV"] == pytest.approx([1.0, 1.0])


def test_build_refusals():
    class Fake:
        energy_min_eV, energy_max_eV = -1.0, 0.0
    with pytest.raises(S.SpectroscopyError, match="tip"):
        S.build(Fake(), tip="d")
    with pytest.raises(S.SpectroscopyError, match="Nothing to compute"):
        S.build(Fake(), tip="s")
    with pytest.raises(S.SpectroscopyError, match="broadening"):
        S.build(Fake(), tip="p", energies_eV=[0.0], broadening_eV=5.0)
