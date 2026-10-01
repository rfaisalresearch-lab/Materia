"""Molecular engines inside Materia's workflows, checked against independent answers.

* Water at HF/STO-3G: Materia's finite-displacement vibrations from PySCF
  forces against PySCF's analytic Hessian on the same geometry.
* Ammonia inversion with GFN2-xTB through Materia's NEB: a symmetric path
  whose barrier is near the experimental 0.25 eV (24 kJ/mol, Swalen and
  Ibers 1962); reported with a loose bound because GFN2-xTB is a
  semi-empirical model, not a validation of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.physics import molecular as M
from materia.physics import neb
from materia.physics import phonons as P
from materia.solvers.molecular import create

pytestmark = pytest.mark.validation
pytest.importorskip("pyscf", reason="BLOCKED: pyscf is not installed")
pytest.importorskip("tblite", reason="BLOCKED: tblite is not installed")
pytest.importorskip("rdkit", reason="BLOCKED: rdkit is not installed")


def test_water_vibrations_match_the_pyscf_analytic_hessian():
    from pyscf import gto, scf
    from pyscf.hessian import thermo

    solver = create("pyscf/hf", basis="sto-3g")
    water = solver.relax(M.molecule_from_smiles("O"), fmax_eV_A=1e-4,
                         max_steps=2000).structure
    out = solver.phonons(water, P.PhononSettings(sum_rule="translational+rotational",
                                                 stencil=4, displacement_A=0.005))
    frequencies = out["frequencies"]
    assert frequencies.extra["kinds"] == ["rigid-body"] * 6 + ["real"] * 3
    mine = np.sort(frequencies.value)[-3:] * P.THZ_TO_CM1
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip("OHH", water.positions)],
                basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    analytic = np.sort(np.real(thermo.harmonic_analysis(mol, mf.Hessian().kernel())[
        "freq_wavenumber"]))
    assert np.abs(mine - analytic).max() < 0.05
    assert frequencies.provenance.model == "phonons/finite-displacement[pyscf/hf/sto-3g]"


def test_ammonia_inversion_barrier_with_xtb():
    solver = create("xtb/gfn2")
    pyramid = solver.relax(M.molecule_from_smiles("N"), fmax_eV_A=1e-3,
                           max_steps=3000).structure
    inverted = pyramid.copy()
    nitrogen = pyramid.positions[0]
    inverted.positions = 2 * nitrogen - pyramid.positions
    inverted = solver.relax(inverted, fmax_eV_A=1e-3, max_steps=3000).structure
    path = neb.run(pyramid, inverted, solver.potential,
                   neb.NEBSettings(images=7, fmax_eV_A=0.02, remove_rotation_translation=True))
    assert path.convergence.converged
    assert abs(path.reaction_energy_eV) < 1e-6
    assert path.forward_barrier_eV == pytest.approx(path.reverse_barrier_eV, abs=1e-6)
    assert 0.15 < path.forward_barrier_eV < 0.40
    assert path.provenance.model == "neb/ase-fire[xtb/gfn2]"
