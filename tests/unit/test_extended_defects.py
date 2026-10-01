"""Stacking faults and dislocations with the shipped Cu EAM potential."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.physics import eam
from materia.structure_builder import extended_defects as X

A_CU = 3.6146336


@pytest.fixture(scope="module")
def copper():
    return eam.load_shipped("Cu-Zhou04")


def test_stacking_fault_curve(copper):
    rigid = X.generalized_stacking_fault(copper, "Cu", A_CU, relax=False)
    relaxed = X.generalized_stacking_fault(copper, "Cu", A_CU, relax=True)
    for result in (rigid, relaxed):
        gamma = result.value["gamma_mJ_m2"]
        assert gamma[0] == 0.0
        assert result.extra["unstable_mJ_m2"] > result.extra["intrinsic_mJ_m2"] > 0
        assert np.argmax(gamma) not in (0, len(gamma) - 1)
    assert relaxed.extra["intrinsic_mJ_m2"] < rigid.extra["intrinsic_mJ_m2"]
    assert relaxed.extra["unstable_mJ_m2"] < rigid.extra["unstable_mJ_m2"]
    assert rigid.extra["partial_burgers_A"] == pytest.approx(A_CU / math.sqrt(6))
    with pytest.raises(X.DefectError, match="multiple of 3"):
        X.fcc_111_cell("Cu", A_CU, layers=10)


def test_centrosymmetry_is_zero_in_perfect_fcc():
    cell = X.fcc_111_cell("Cu", A_CU, layers=6)
    assert np.allclose(X.centrosymmetry(cell), 0.0, atol=1e-10)


def test_edge_dislocation_dissociates_on_its_glide_plane(copper):
    result = X.dislocation(copper, "fcc-edge", "Cu", A_CU, 170.9, 121.95, 76.49, radius_A=70.0,
                           fmax_eV_A=5e-3, max_steps=6000)
    assert result.extra["relaxation_converged"]
    assert np.linalg.norm(result.extra["burgers_vector_A"]) == pytest.approx(A_CU / math.sqrt(2))
    structure = result.value
    csp = np.array(result.extra["centrosymmetry_A2"])
    width = X.dissociation_width(structure, csp, A_CU, glide_axis=0, normal_axis=1)
    assert 20.0 < width < 50.0
    centre = structure.positions[~structure.fixed].mean(axis=0)
    relative = structure.positions[:, :2] - centre[:2]
    defected = (np.linalg.norm(relative, axis=1) < 40) & (csp > 0.5 * (A_CU / math.sqrt(6)) ** 2)
    assert np.ptp(relative[defected, 1]) < 3.0
    assert result.provenance.origin.value == "calculated"


def test_unknown_dislocation_is_refused(copper):
    with pytest.raises(X.DefectError, match="kind"):
        X.dislocation(copper, "hcp-basal", "Cu", A_CU, 170.9, 121.95, 76.49)


def test_sigma5_tilt_boundaries(copper):
    small = X.symmetric_tilt_boundary(copper, "Cu", A_CU, (3, 1, 0), grain_repeats=3,
                                      translations=4)
    large = X.symmetric_tilt_boundary(copper, "Cu", A_CU, (3, 1, 0), grain_repeats=5,
                                      translations=4)
    assert small.extra["sigma"] == 5
    assert small.extra["misorientation_deg"] == pytest.approx(36.8699, abs=1e-3)
    assert small.extra["relaxed"] and large.extra["relaxed"]
    assert 300 < large.value < 1500
    assert small.value == pytest.approx(large.value, rel=0.02)
    other = X.symmetric_tilt_boundary(copper, "Cu", A_CU, (2, 1, 0), grain_repeats=4,
                                      translations=3)
    assert other.extra["sigma"] == 5 and other.value > 0
    with pytest.raises(X.DefectError, match="h k 0"):
        X.symmetric_tilt_boundary(copper, "Cu", A_CU, (1, 1, 1))
