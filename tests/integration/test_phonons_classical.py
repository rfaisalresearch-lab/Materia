"""Harmonic analysis through the classical solvers and an ASE calculator."""

from __future__ import annotations

import numpy as np
import pytest

from materia.core_model.cell import Cell
from materia.core_model.structure import Structure
from materia.materials import default_library
from materia.physics import phonons as P
from materia.provenance import Fidelity
from materia.solvers import Capability, registry
from materia.structure_builder.lattice import bulk


def silicon_primitive(a):
    return Structure([14, 14], np.array([[0, 0, 0], [a / 4] * 3]),
                     Cell(np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]),
                          (True, True, True)))


def test_classical_solvers_declare_and_run_phonons():
    solver = registry.create("stillinger-weber-si")
    assert solver.can(Capability.PHONONS)
    si = bulk(default_library().get("silicon"))
    out = solver.run(si, task="phonons", settings=P.PhononSettings(stencil=4))
    frequencies = out["frequencies"]
    assert frequencies.supported and frequencies.unit == "THz"
    kinds = frequencies.extra["kinds"]
    assert kinds.count("rigid-body") == 3 and "imaginary" not in kinds
    optical = np.sort(frequencies.value)[-3:]
    assert np.ptp(optical) < 1e-6
    primitive = solver.phonons(silicon_primitive(si.cell.matrix[0, 0]),
                               P.PhononSettings(stencil=4))
    assert np.allclose(np.sort(primitive["frequencies"].value)[-3:], optical, rtol=1e-8)
    assert out.convergence.converged
    assert set(out.keys()) >= {"frequencies", "force_constants", "eigenvectors",
                               "displacements", "zero_point_energy"}
    assert any("LO-TO" in a for a in frequencies.provenance.approximations)
    assert frequencies.provenance.model.endswith("[classical/stillinger-weber(Si)]")


def test_structure_is_not_changed():
    si = bulk(default_library().get("silicon"))
    before = si.positions.copy()
    registry.create("stillinger-weber-si").phonons(si)
    assert np.array_equal(si.positions, before)


def test_relaxed_quartz_has_no_imaginary_modes():
    solver = registry.create("rigid-ion/bks-silica")
    quartz = bulk(default_library().get("silicon_dioxide"))
    relaxed = solver.relax(quartz, fmax_eV_A=1e-5, max_steps=5000).structure
    out = solver.phonons(relaxed, P.PhononSettings(stencil=4))
    kinds = out["frequencies"].extra["kinds"]
    assert kinds.count("rigid-body") == 3
    assert "imaginary" not in kinds and "zero-within-resolution" not in kinds
    assert out["zero_point_energy"].supported
    assert any("own images" in line for line in out.log)


def test_unrelaxed_quartz_is_refused_as_an_unsupported_result():
    quartz = bulk(default_library().get("silicon_dioxide"))
    out = registry.create("rigid-ion/bks-silica").phonons(quartz)
    result = out["frequencies"]
    assert not result.supported and result.extra["status"] == "refused"
    assert "stationary point" in result.unsupported_reason
    assert "force_constants" not in out.keys()


def test_cancellation_returns_no_numbers():
    out = registry.create("stillinger-weber-si").phonons(
        bulk(default_library().get("silicon")), cancelled=lambda: True)
    assert out["frequencies"].extra["status"] == "cancelled"
    assert out.keys() == ["frequencies"]


def test_ase_calculator_source():
    from ase.calculators.emt import EMT
    from ase.cluster import Octahedron
    from ase.optimize import BFGS

    atoms = Octahedron("Cu", 2)
    atoms.center(vacuum=6.0)
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-6)
    structure = Structure(atoms.numbers, atoms.positions,
                          Cell(np.array(atoms.cell), (False, False, False)))
    result = P.harmonic_analysis(structure, EMT(),
                                 P.PhononSettings(sum_rule="translational+rotational"),
                                 name="ase/EMT")
    assert result.provenance.fidelity is Fidelity.NON_PHYSICAL
    assert result.kinds.count("rigid-body") == 6
    assert "imaginary" not in result.kinds
