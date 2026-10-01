"""Melt and quench: refusal when the crystal does not melt, and an amorphous result."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materia.materials import default_library
from materia.physics.potentials import StillingerWeber, UnsupportedSystem
from materia.solvers import registry
from materia.structure_builder import melt_quench as MQ
from materia.structure_builder.lattice import bulk


@pytest.fixture(scope="module")
def silicon():
    crystal = bulk(default_library().get("silicon"), (2, 2, 2))
    a = crystal.cell.matrix[0, 0] / 2
    vectors = [2 * math.pi / a * np.array(v) for v in ((1, 1, 1), (1, 1, -1), (1, -1, 1),
                                                     (-1, 1, 1))]
    return crystal, vectors


def test_crystal_analysis(silicon):
    crystal, vectors = silicon
    analysis = MQ.analyse(crystal, 2.8)
    assert analysis["mean_coordination"] == 4.0
    assert analysis["bond_angle_mean_deg"] == pytest.approx(109.4712, abs=1e-3)
    assert analysis["first_peak_A"] == pytest.approx(2.35, abs=0.03)
    assert MQ.structure_factor_order(crystal, crystal, vectors) == pytest.approx(1.0)


def test_unmelted_crystal_is_refused(silicon):
    crystal, vectors = silicon
    potential = registry.create("stillinger-weber-si-vink2001").potential
    with pytest.raises(MQ.MeltQuenchError, match="did not melt"):
        MQ.melt_quench(potential, crystal, 2000.0, 300.0, melt_ps=1, cooling_K_per_ps=500,
                       hold_ps=0.5, reciprocal_vectors=vectors)


def test_vink_silicon_glass(silicon):
    crystal, vectors = silicon
    potential = registry.create("stillinger-weber-si-vink2001").potential
    result = MQ.melt_quench(potential, crystal, 6000.0, 300.0, melt_ps=4, cooling_K_per_ps=150,
                            hold_ps=2, seed=3, bond_cutoff_A=2.8, reciprocal_vectors=vectors)
    analysis = result.extra["analysis"]
    assert analysis["crystalline_order"] < 0.2
    assert 3.7 < analysis["mean_coordination"] < 4.3
    assert result.extra["energy_above_crystal_eV_per_atom"] > 0.05
    assert result.provenance.parameters["seed"] == 3
    assert "Vink" in " ".join(potential.describe()["references"])


def test_variant_lookup():
    assert StillingerWeber("Si", variant="vink2001").p.lam == 31.5
    with pytest.raises(UnsupportedSystem, match="variant"):
        StillingerWeber("Si", variant="nonexistent")
