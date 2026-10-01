"""GIAO NMR shieldings through PySCF against gas-phase absolute shieldings.

Experimental absolute shieldings (ppm): CH4 13C 195.1 and 1H 30.61, NH3 15N
264.5, H2O 17O 323.6 (C. J. Jameson, Annu. Rev. Phys. Chem. 47 (1996) 135;
R. E. Wasylishen and D. L. Bryce, J. Chem. Phys. 117 (2002) 10061).
Hartree-Fock without vibrational averaging is expected within a few ppm.
"""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.physics import molecular as M

pytestmark = pytest.mark.validation
pytest.importorskip("pyscf.prop.nmr", reason="BLOCKED: pyscf-properties is not installed")
pytest.importorskip("rdkit", reason="BLOCKED: rdkit is not installed")
METHANE = Structure([6, 1, 1, 1, 1], np.array([[0, 0, 0], [1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                                               [1, -1, -1]]) * 0.6276,
                    Cell(np.eye(3) * 20, (False,) * 3))


def relaxed(smiles):
    from materia.solvers.classical import ClassicalSolver
    pot = M.PySCFPotential("dft", basis="def2-tzvp", xc="b3lyp")
    return ClassicalSolver(pot).relax(M.molecule_from_smiles(smiles), fmax_eV_A=1e-3,
                                      max_steps=300).structure


def test_response_wrapper_equals_the_original_code():
    from pyscf.prop import nmr

    pot = M.PySCFPotential("hf", basis="cc-pvtz")
    original = np.asarray(nmr.RHF(pot.mean_field(METHANE)).kernel())
    wrapped = M.nmr_shielding(METHANE, "hf", "cc-pvtz")["nuclei"]
    for tensor, nucleus in zip(original, wrapped):
        assert np.allclose(tensor, nucleus["tensor_ppm"], atol=1e-6)


def test_shielding_is_independent_of_position_and_orientation():
    first = M.nmr_shielding(METHANE, "hf", "def2-svp")["nuclei"]
    moved = METHANE.copy()
    angle = 0.6
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    moved.positions = np.asarray(METHANE.positions) @ rotation.T + [3.0, -2.0, 5.0]
    second = M.nmr_shielding(moved, "hf", "def2-svp")["nuclei"]
    for a, b in zip(first, second):
        assert a["isotropic_ppm"] == pytest.approx(b["isotropic_ppm"], abs=2e-4)
        assert a["anisotropy_ppm"] == pytest.approx(b["anisotropy_ppm"], abs=2e-4)
    hydrogens = [n["isotropic_ppm"] for n in first if n["element"] == "H"]
    assert np.ptp(hydrogens) < 1e-6
    assert abs(first[0]["anisotropy_ppm"]) < 1e-4


@pytest.mark.parametrize("smiles,element,expected,tolerance", [
    ("C", "C", 195.1, 4.0), ("C", "H", 30.61, 1.2), ("N", "N", 264.5, 6.0),
    ("O", "O", 323.6, 5.0)])
def test_absolute_shieldings_near_experiment(smiles, element, expected, tolerance):
    nuclei = M.nmr_shielding(relaxed(smiles), "hf", "pcseg-2")["nuclei"]
    values = [n["isotropic_ppm"] for n in nuclei if n["element"] == element]
    assert np.mean(values) == pytest.approx(expected, abs=tolerance)


def test_shift_and_refusals():
    methane = METHANE
    water = relaxed("O")
    shifts = M.chemical_shifts(water, methane, "hf", "def2-svp")
    assert [s["element"] for s in shifts["shifts"]] == ["H", "H"]
    assert shifts["shifts"][0]["shift_ppm"] == pytest.approx(
        shifts["reference_shieldings_ppm"]["H"] - shifts["shifts"][0]["shielding_ppm"])
    radical = Structure([8, 1], [[0, 0, 0], [0, 0, 0.97]], Cell(np.eye(3) * 20, (False,) * 3))
    with pytest.raises(M.UnsupportedSystem):
        M.nmr_shielding(radical)
