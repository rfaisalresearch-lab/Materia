"""Molecular and quantum-chemistry engines as Materia solvers.

Each engine in :mod:`materia.physics.molecular` is a potential, so the
classical solver's relaxation, dynamics, harmonic vibrations, thermodynamics
and NEB machinery applies to it unchanged.  This wrapper only changes what the
solver says it is: its name is the engine's model name and its fidelity is the
engine's, never "classical".  The engine that produced a number is recorded in
every result's provenance parameters.
"""

from __future__ import annotations

from typing import Optional

from ..physics import molecular as mol
from ..provenance import Result
from ..core_model.structure import Structure
from .base import Capability
from .classical import ClassicalSolver

ENGINE_LICENCES = {
    "rdkit": "RDKit is BSD-3-Clause; called through its Python API.",
    "tblite": "tblite is LGPL-3.0-or-later; called through its Python API.",
    "pyscf": "PySCF is Apache-2.0; called through its Python API.",
    "openmm": "OpenMM is MIT and LGPL-3.0 licensed; called through its Python API.",
    "qe": "Quantum ESPRESSO is GPL-2.0-or-later; pw.x runs as a separate program.",
    "psi4": "Psi4 is LGPL-3.0; it runs in its own Python environment as a separate process.",
}


class MolecularSolver(ClassicalSolver):
    """A molecular engine behind the classical solver's workflows."""

    def __init__(self, potential, engine: str) -> None:
        super().__init__(potential)
        self.name = potential.name
        self.fidelity = potential.fidelity
        self.engine = engine
        self.capabilities = set(ClassicalSolver.capabilities)
        if hasattr(potential, "properties"):
            self.capabilities |= {Capability.PARTIAL_CHARGES, Capability.DIPOLE}
        self.description = (f"{potential.name} computed by {engine}. "
                            f"{ENGINE_LICENCES[engine]}")

    def _provenance(self, structure: Optional[Structure] = None, origin=None, **extra):
        from ..provenance import Origin

        prov = super()._provenance(structure, origin or Origin.CALCULATED, **extra)
        prov.model = self.name
        prov.approximations = [a for a in prov.approximations
                               if not a.startswith("Classical point-nucleus dynamics")]
        prov.approximations.append(
            "Born-Oppenheimer: nuclei move classically on the engine's energy surface; "
            "no zero-point motion or quantum statistics in relaxation and dynamics.")
        prov.parameters["engine"] = self.engine
        return prov

    def supports(self, structure: Structure):
        from .base import SupportReport

        ok, why = self._potential(structure).supports(structure)
        warnings = []
        if abs(structure.total_charge()) > 1e-9:
            warnings.append(
                "The structure's per-atom formal charges are not read by this engine; the "
                "total charge is the one set in the model options.")
        return SupportReport(ok, [] if ok else [why], warnings)

    def properties(self, structure: Structure) -> Result:
        """Charges, dipole and frontier-orbital gap where the engine provides them."""
        if not hasattr(self.potential, "properties"):
            return self.require(Capability.PARTIAL_CHARGES,
                                f"{self.engine} force fields carry no electronic structure")
        ok, why = self.potential.supports(structure)
        if not ok:
            from ..provenance import unsupported
            return unsupported("molecular_properties", self.name, why)
        value = self.potential.properties(structure)
        return Result("molecular_properties", value, "mixed", self._provenance(),
                      extra={"units": {"energy_eV": "eV", "charges_e": "e",
                                       "dipole_e_A": "e A", "homo_lumo_gap_eV": "eV"}})


def available(engine: str) -> bool:
    if engine == "qe":
        from ..physics.espresso import find_pw
        return find_pw() is not None
    if engine == "psi4":
        from .psi4_driver import installed_pythons
        return bool(installed_pythons())
    return mol.engine_versions().get(engine) is not None


MODELS = {
    "rdkit/mmff94": ("rdkit", lambda: mol.RDKitForceField("mmff94")),
    "rdkit/uff": ("rdkit", lambda: mol.RDKitForceField("uff")),
    "xtb/gfn2": ("tblite", lambda: mol.XTBPotential("gfn2")),
    "xtb/gfn1": ("tblite", lambda: mol.XTBPotential("gfn1")),
    "pyscf/hf": ("pyscf", lambda: mol.PySCFPotential("hf", basis="def2-svp")),
    "pyscf/b3lyp": ("pyscf", lambda: mol.PySCFPotential("dft", basis="def2-svp", xc="b3lyp")),
    "pyscf/pbe": ("pyscf", lambda: mol.PySCFPotential("dft", basis="def2-svp", xc="pbe")),
    "pyscf/mp2": ("pyscf", lambda: mol.PySCFPotential("mp2", basis="cc-pvdz")),
    "openmm/amber14-gbn2": ("openmm", lambda: _openmm("amber14-gbn2")),
    "openmm/amber14": ("openmm", lambda: _openmm("amber14")),
    "openmm/charmm36": ("openmm", lambda: _openmm("charmm36")),
    "openmm/amber99sbildn": ("openmm", lambda: _openmm("amber99sbildn")),
    "qe/pw.x": ("qe", lambda: _espresso()),
    "psi4/hf": ("psi4", lambda: _psi4("hf", "cc-pvdz")),
    "psi4/b3lyp": ("psi4", lambda: _psi4("b3lyp", "def2-svp")),
    "psi4/mp2": ("psi4", lambda: _psi4("mp2", "cc-pvdz")),
}


def _psi4(method: str, basis: str, **options):
    from .psi4_driver import Psi4Potential
    return Psi4Potential(method, basis, **options)


def _espresso(**options):
    from ..physics.espresso import EspressoPotential
    return EspressoPotential(**options)


def _openmm(force_field: str):
    from ..physics.biomolecular import OpenMMPotential
    return OpenMMPotential(force_field)


def create(name: str, **options) -> MolecularSolver:
    """A molecular solver by model name, with engine options such as basis or charge."""
    if name not in MODELS:
        raise KeyError(f"Unknown molecular model {name!r}. Known: {sorted(MODELS)}.")
    engine, factory = MODELS[name]
    potential = factory()
    if options:
        kind = type(potential)
        base = {"rdkit/mmff94": {"force_field": "mmff94"}, "rdkit/uff": {"force_field": "uff"},
                "xtb/gfn2": {"method": "gfn2"}, "xtb/gfn1": {"method": "gfn1"},
                "pyscf/hf": {"method": "hf", "basis": "def2-svp"},
                "pyscf/b3lyp": {"method": "dft", "basis": "def2-svp", "xc": "b3lyp"},
                "pyscf/pbe": {"method": "dft", "basis": "def2-svp", "xc": "pbe"},
                "pyscf/mp2": {"method": "mp2", "basis": "cc-pvdz"},
                "openmm/amber14-gbn2": {"force_field": "amber14-gbn2"},
                "openmm/amber14": {"force_field": "amber14"},
                "openmm/charmm36": {"force_field": "charmm36"},
                "openmm/amber99sbildn": {"force_field": "amber99sbildn"},
                "qe/pw.x": {},
                "psi4/hf": {"method": "hf", "basis": "cc-pvdz"},
                "psi4/b3lyp": {"method": "b3lyp", "basis": "def2-svp"},
                "psi4/mp2": {"method": "mp2", "basis": "cc-pvdz"}}[name]
        potential = kind(**{**base, **options})
    return MolecularSolver(potential, engine)
