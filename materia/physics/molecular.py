"""Molecular chemistry through established open-source engines.

Three engines are driven in process, each wrapped as a Materia
:class:`~materia.physics.potentials.Potential` so that relaxation, molecular
dynamics, harmonic vibrations and NEB pathways work with them unchanged:

``RDKitForceField``
    The MMFF94 or UFF force field as implemented in RDKit (BSD licence).
    Needs a bonded topology, taken from the molecule's stored MDL molblock or
    perceived from the geometry.  Isolated molecules only.
``XTBPotential``
    GFN2-xTB or GFN1-xTB tight binding through ``tblite`` (LGPL-3.0), with a
    total charge and unpaired-electron count.  Molecules and periodic cells.
``PySCFPotential``
    Hartree-Fock, Kohn-Sham DFT or MP2 through PySCF (Apache-2.0), with
    analytic nuclear gradients, a Gaussian basis set, a charge and a spin.
    Isolated molecules only.  :func:`pyscf_single_point` adds CCSD and
    CCSD(T) energies and molecular properties.

Every engine is optional.  When its package is missing the potential refuses
with the install command instead of substituting another model.  Unit
conversions use the CODATA values in :mod:`materia.core_model.units`.

References
----------
T. A. Halgren, J. Comput. Chem. 17 (1996) 490 (MMFF94).
A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. Goddard and W. M. Skiff,
J. Am. Chem. Soc. 114 (1992) 10024 (UFF).
C. Bannwarth, S. Ehlert and S. Grimme, J. Chem. Theory Comput. 15 (2019) 1652
(GFN2-xTB).
Q. Sun et al., J. Chem. Phys. 153 (2020) 024109 (PySCF).
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..core_model.units import BOHR_A, HARTREE_EV
from ..elements import periodic_table as pt
from ..provenance import Fidelity
from .potentials import Potential, PotentialError, UnsupportedSystem

KCAL_MOL_EV = 4184.0 / 6.02214076e23 / 1.602176634e-19
MOLECULE_INFO_KEY = "molecule"

INSTALL = {"rdkit": "pip install rdkit", "tblite": "pip install tblite",
           "pyscf": "pip install pyscf", "openmm": "pip install openmm",
           "qe": "CONDA_SUBDIR=osx-64 conda create -n materia-qe -c conda-forge qe",
           "psi4": "conda create -n materia-psi4 -c conda-forge psi4=1.9.1 'libint<2.10' "
                   "'libxc-c<7'"}


class EngineMissing(UnsupportedSystem):
    """The open-source engine this model needs is not installed."""


class EngineFailed(PotentialError):
    """The engine ran and did not produce a trustworthy answer."""


def require(module: str):
    try:
        return importlib.import_module(module)
    except ImportError:
        root = module.split(".")[0]
        raise EngineMissing(f"The {root} package is not installed. Install it with: "
                            f"{INSTALL.get(root, 'pip install ' + root)}") from None


def engine_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name, attr in (("rdkit", "__version__"), ("tblite", "__version__"),
                       ("pyscf", "__version__"), ("openmm", "__version__")):
        try:
            out[name] = str(getattr(importlib.import_module(name), attr, "unknown"))
        except ImportError:
            out[name] = None
    return out


def isolated(structure: Structure, model: str) -> None:
    if any(structure.cell.pbc):
        raise UnsupportedSystem(f"{model} treats isolated molecules only; this structure "
                                "has periodic directions.")


def molecule_from_smiles(smiles: str, seed: int = 0, vacuum_A: float = 8.0,
                         force_field: str = "mmff94", add_hydrogens: bool = True) -> Structure:
    """A 3D molecule from SMILES: ETKDG embedding, then force-field minimisation.

    The structure is isolated, centred in a cubic box with ``vacuum_A`` of
    space on every side, and records the canonical SMILES, the total charge,
    the bonded topology as an MDL molblock and every choice made.
    """
    Chem = require("rdkit.Chem")
    AllChem = require("rdkit.Chem.AllChem")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise UnsupportedSystem(f"RDKit could not parse the SMILES {smiles!r}.")
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise EngineFailed(f"RDKit could not embed {smiles!r} in three dimensions.")
    if force_field == "mmff94":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise UnsupportedSystem("MMFF94 has no parameters for this molecule; use 'uff'.")
        converged = AllChem.MMFFOptimizeMolecule(mol, maxIters=10000) == 0
    elif force_field == "uff":
        converged = AllChem.UFFOptimizeMolecule(mol, maxIters=10000) == 0
    elif force_field == "none":
        converged = True
    else:
        raise ValueError("force_field must be 'mmff94', 'uff' or 'none'.")
    if not converged:
        raise EngineFailed(f"The {force_field} minimisation of {smiles!r} did not converge.")
    positions = mol.GetConformer().GetPositions()
    span = positions.max(axis=0) - positions.min(axis=0)
    size = float(span.max() + 2.0 * vacuum_A)
    positions = positions - positions.mean(axis=0) + size / 2.0
    Geometry = require("rdkit.Geometry")
    conformer = mol.GetConformer()
    for k, (x, y, z) in enumerate(positions):
        conformer.SetAtomPosition(k, Geometry.Point3D(float(x), float(y), float(z)))
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    structure = Structure(numbers, positions, Cell(np.eye(3) * size, (False, False, False)))
    rdkit = importlib.import_module("rdkit")
    structure.info[MOLECULE_INFO_KEY] = {
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)), "input_smiles": str(smiles),
        "charge": int(Chem.GetFormalCharge(mol)), "molblock": Chem.MolToMolBlock(mol),
        "embedding": "ETKDGv3", "seed": int(seed), "pre_optimisation": force_field,
        "rdkit_version": rdkit.__version__,
    }
    return structure


def _rdkit_mol(structure: Structure):
    Chem = require("rdkit.Chem")
    record = structure.info.get(MOLECULE_INFO_KEY) or {}
    block = record.get("molblock")
    if block:
        mol = Chem.MolFromMolBlock(block, removeHs=False)
        if mol is None or mol.GetNumAtoms() != len(structure) or [
                a.GetAtomicNum() for a in mol.GetAtoms()] != list(map(int, structure.numbers)):
            raise UnsupportedSystem("The stored molecular topology no longer matches the "
                                    "atoms of this structure.")
        return mol
    rdDetermineBonds = require("rdkit.Chem.rdDetermineBonds")
    xyz = f"{len(structure)}\n\n" + "\n".join(
        f"{pt.symbol(int(z))} {x:.10f} {y:.10f} {w:.10f}"
        for z, (x, y, w) in zip(structure.numbers, structure.positions))
    mol = Chem.MolFromXYZBlock(xyz)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=int(record.get("charge", 0)))
    except Exception as exc:
        raise UnsupportedSystem(f"RDKit could not perceive bonds from the geometry: {exc}")
    return mol


class RDKitForceField(Potential):
    """MMFF94 or UFF from RDKit, on the molecule's bonded topology."""

    fidelity = Fidelity.TIER1_CLASSICAL

    def __init__(self, force_field: str = "mmff94") -> None:
        if force_field not in ("mmff94", "uff"):
            raise ValueError("force_field must be 'mmff94' or 'uff'.")
        self.force_field = force_field
        self.name = f"rdkit/{force_field}"
        self.cutoff_A = 0.0
        self._cache: Dict[str, Tuple[object, object]] = {}

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        try:
            require("rdkit")
            isolated(structure, self.name)
        except UnsupportedSystem as exc:
            return False, str(exc)
        return True, ""

    def _field(self, structure: Structure):
        key = (structure.info.get(MOLECULE_INFO_KEY) or {}).get("molblock") or \
            str(structure.numbers.tolist())
        if key in self._cache:
            return self._cache[key]
        AllChem = require("rdkit.Chem.AllChem")
        mol = _rdkit_mol(structure)
        conformer_free = mol.GetNumConformers() == 0
        if conformer_free:
            Chem = require("rdkit.Chem")
            conf = Chem.Conformer(mol.GetNumAtoms())
            mol.AddConformer(conf, assignId=True)
        if self.force_field == "mmff94":
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                raise UnsupportedSystem("MMFF94 has no parameters for this molecule; use UFF.")
            field = AllChem.MMFFGetMoleculeForceField(mol, props, ignoreInterfragInteractions=False)
        else:
            if not AllChem.UFFHasAllMoleculeParams(mol):
                raise UnsupportedSystem("UFF has no parameters for an atom of this molecule.")
            field = AllChem.UFFGetMoleculeForceField(mol, ignoreInterfragInteractions=False)
        if field is None:
            raise EngineFailed(f"RDKit could not set up {self.force_field} for this molecule.")
        self._cache[key] = (mol, field)
        return mol, field

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        self.check(structure)
        isolated(structure, self.name)
        _, field = self._field(structure)
        flat = [float(v) for v in np.asarray(structure.positions).ravel()]
        energy = field.CalcEnergy(flat) * KCAL_MOL_EV
        forces = -np.asarray(field.CalcGrad(flat), dtype=float).reshape(-1, 3) * KCAL_MOL_EV
        return float(energy), forces

    def model_label(self) -> str:
        return self.name

    def describe(self) -> dict:
        label = {"mmff94": "MMFF94 (Halgren 1996)", "uff": "UFF (Rappe et al. 1992)"}
        reference = {"mmff94": "T. A. Halgren, J. Comput. Chem. 17 (1996) 490",
                     "uff": "A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. Goddard and "
                            "W. M. Skiff, J. Am. Chem. Soc. 114 (1992) 10024"}
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": 0.0,
                "parameters": {"force_field": self.force_field,
                               "engine": "RDKit", "engine_version": engine_versions()["rdkit"]},
                "approximations": [
                    f"{label[self.force_field]} classical force field as implemented in "
                    "RDKit: fixed bonded topology, so bonds cannot break or form.",
                    "No cutoff on non-bonded terms; isolated molecule.",
                ],
                "references": [reference[self.force_field],
                               "RDKit: open-source cheminformatics, https://www.rdkit.org"]}


class XTBPotential(Potential):
    """GFN-xTB tight binding through tblite."""

    fidelity = Fidelity.TIER2_SEMI_EMPIRICAL
    METHODS = {"gfn2": "GFN2-xTB", "gfn1": "GFN1-xTB"}

    SOLVATION = ("alpb", "gbsa")

    def __init__(self, method: str = "gfn2", charge: int = 0, unpaired: int = 0,
                 accuracy: float = 1.0, electronic_temperature_K: float = 300.0,
                 max_iterations: int = 250, solvent: Optional[str] = None,
                 solvation_model: str = "alpb") -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {sorted(self.METHODS)}.")
        if solvation_model not in self.SOLVATION:
            raise ValueError(f"solvation_model must be one of {self.SOLVATION}.")
        self.solvent = solvent
        self.solvation_model = solvation_model
        self.method = method
        self.charge = int(charge)
        self.unpaired = int(unpaired)
        self.accuracy = float(accuracy)
        self.electronic_temperature_K = float(electronic_temperature_K)
        self.max_iterations = int(max_iterations)
        self.name = f"xtb/{method}" + (f"+{solvation_model}({solvent})" if solvent else "")
        self.cutoff_A = 0.0
        self.last: Dict[str, object] = {}

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        try:
            require("tblite.interface")
        except UnsupportedSystem as exc:
            return False, str(exc)
        if any(structure.cell.pbc) and not all(structure.cell.pbc):
            return False, ("tblite treats isolated molecules and three-dimensional crystals; "
                           "slabs and wires need a vacuum-padded periodic cell.")
        electrons = int(structure.numbers.sum()) - self.charge
        if (electrons - self.unpaired) % 2:
            return False, (f"{electrons} electrons with {self.unpaired} unpaired is not a "
                           "consistent spin state; set unpaired.")
        return True, ""

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        interface = require("tblite.interface")
        periodic = all(structure.cell.pbc)
        kwargs = {}
        if periodic:
            kwargs = {"lattice": np.asarray(structure.cell.matrix) / BOHR_A,
                      "periodic": np.array([True, True, True])}
        calc = interface.Calculator(self.METHODS[self.method],
                                    np.asarray(structure.numbers, dtype=int),
                                    np.asarray(structure.positions) / BOHR_A,
                                    charge=float(self.charge), uhf=self.unpaired, **kwargs)
        calc.set("verbosity", 0)
        calc.set("accuracy", self.accuracy)
        calc.set("max-iter", self.max_iterations)
        calc.set("temperature", self.electronic_temperature_K * 3.166811563e-6)
        if self.solvent:
            if periodic:
                raise UnsupportedSystem("Implicit solvation applies to molecules, not to "
                                        "periodic cells.")
            try:
                calc.add(f"{self.solvation_model}-solvation", self.solvent)
            except Exception as exc:
                raise UnsupportedSystem(f"tblite has no {self.solvation_model} parameters "
                                        f"for solvent {self.solvent!r}: {exc}") from None
        try:
            result = calc.singlepoint()
        except RuntimeError as exc:
            raise EngineFailed(f"{self.METHODS[self.method]} did not converge: {exc}") from None
        energy = float(result.get("energy")) * HARTREE_EV
        forces = -np.asarray(result.get("gradient"), dtype=float) * HARTREE_EV / BOHR_A
        self.last = {"charges_e": np.asarray(result.get("charges")).tolist(),
                     "dipole_e_A": (np.asarray(result.get("dipole")) * BOHR_A).tolist(),
                     "orbital_energies_eV": (np.asarray(result.get("orbital-energies"))
                                             * HARTREE_EV).tolist(),
                     "occupations": np.asarray(result.get("orbital-occupations")).tolist()}
        return energy, forces

    def properties(self, structure: Structure) -> dict:
        energy, forces = self.energy_and_forces(structure)
        occupations = np.asarray(self.last["occupations"])
        levels = np.asarray(self.last["orbital_energies_eV"])
        if occupations.ndim > 1:
            occupations = occupations.sum(axis=0)
            levels = levels[0]
        filled = levels[occupations > 1e-3]
        empty = levels[occupations <= 1e-3]
        gap = float(empty.min() - filled.max()) if filled.size and empty.size else None
        return {"energy_eV": energy, "max_force_eV_A": float(np.abs(forces).max()),
                "charges_e": self.last["charges_e"], "dipole_e_A": self.last["dipole_e_A"],
                "homo_lumo_gap_eV": gap}

    def model_label(self) -> str:
        return self.name

    def describe(self) -> dict:
        reference = {"gfn2": "C. Bannwarth, S. Ehlert and S. Grimme, J. Chem. Theory Comput. "
                             "15 (2019) 1652",
                     "gfn1": "S. Grimme, C. Bannwarth and P. Shushkov, J. Chem. Theory Comput. "
                             "13 (2017) 1989"}
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": 0.0,
                "parameters": {"method": self.METHODS[self.method], "charge_e": self.charge,
                               "unpaired_electrons": self.unpaired,
                               "accuracy": self.accuracy,
                               "electronic_temperature_K": self.electronic_temperature_K,
                               "solvent": self.solvent,
                               "solvation_model": self.solvation_model if self.solvent else None,
                               "engine": "tblite",
                               "engine_version": engine_versions()["tblite"]},
                "approximations": [
                    f"{self.METHODS[self.method]}: semi-empirical extended tight binding "
                    "with a minimal valence basis, fitted to reference data; bonds can "
                    "form and break.",
                    "Self-consistent charges with Fermi smearing at the stated electronic "
                    "temperature.",
                    (f"Implicit {self.solvation_model.upper()} solvation in {self.solvent} "
                     "(Ehlert et al. 2021): a continuum, no explicit solvent molecules."
                     if self.solvent else "Gas phase."),
                ],
                "references": [reference[self.method],
                               "tblite: light-weight tight-binding framework, "
                               "https://github.com/tblite/tblite"]}


class PySCFPotential(Potential):
    """Hartree-Fock, DFT or MP2 energies and analytic gradients from PySCF."""

    fidelity = Fidelity.TIER3_EXTERNAL
    METHODS = ("hf", "dft", "mp2")

    SOLVATION = ("pcm", "ddcosmo")

    def __init__(self, method: str = "dft", basis: str = "def2-svp", xc: str = "b3lyp",
                 charge: int = 0, spin: int = 0, conv_tol: float = 1e-10,
                 max_cycle: int = 200, dielectric: Optional[float] = None,
                 solvation_model: str = "pcm") -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}.")
        if solvation_model not in self.SOLVATION:
            raise ValueError(f"solvation_model must be one of {self.SOLVATION}.")
        if dielectric is not None and not dielectric >= 1.0:
            raise ValueError("dielectric must be at least 1.")
        if dielectric is not None and method == "mp2":
            raise ValueError("Solvated MP2 gradients are not available; use HF or DFT.")
        self.dielectric = dielectric
        self.solvation_model = solvation_model
        self.method = method
        self.basis = str(basis)
        self.xc = str(xc) if method == "dft" else ""
        self.charge = int(charge)
        self.spin = int(spin)
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        label = self.xc if method == "dft" else method
        self.name = f"pyscf/{label}/{self.basis}" + (
            f"+{solvation_model}(eps={dielectric:g})" if dielectric else "")
        self.cutoff_A = 0.0
        self.last: Dict[str, object] = {}

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        try:
            require("pyscf")
            isolated(structure, "PySCF molecular calculations")
        except UnsupportedSystem as exc:
            return False, str(exc)
        electrons = int(structure.numbers.sum()) - self.charge
        if (electrons - self.spin) % 2:
            return False, (f"{electrons} electrons with 2S = {self.spin} is not a consistent "
                           "spin state; set spin.")
        return True, ""

    def molecule(self, structure: Structure):
        gto = require("pyscf.gto")
        atoms = [(pt.symbol(int(z)), tuple(float(v) for v in p))
                 for z, p in zip(structure.numbers, structure.positions)]
        try:
            return gto.M(atom=atoms, unit="Angstrom", basis=self.basis, charge=self.charge,
                         spin=self.spin, verbose=0)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise UnsupportedSystem(f"PySCF could not build the molecule in basis "
                                    f"{self.basis}: {exc}") from None

    def mean_field(self, structure: Structure):
        mol = self.molecule(structure)
        if self.method == "dft":
            dft = require("pyscf.dft")
            mf = dft.RKS(mol) if self.spin == 0 else dft.UKS(mol)
            mf.xc = self.xc
        else:
            scf = require("pyscf.scf")
            mf = scf.RHF(mol) if self.spin == 0 else scf.UHF(mol)
        if self.dielectric:
            solvent = require("pyscf.solvent")
            mf = (solvent.PCM(mf) if self.solvation_model == "pcm" else solvent.ddCOSMO(mf))
            mf.with_solvent.eps = self.dielectric
        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.kernel()
        if not mf.converged:
            raise EngineFailed(f"The {self.name} self-consistent field did not converge in "
                               f"{self.max_cycle} cycles.")
        return mf

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        mf = self.mean_field(structure)
        if self.method == "mp2":
            mp = require("pyscf.mp")
            post = mp.MP2(mf).run()
            energy = float(post.e_tot)
            gradient = post.nuc_grad_method().kernel()
        else:
            energy = float(mf.e_tot)
            gradient = mf.nuc_grad_method().kernel()
        self.last = {"scf_energy_Eh": float(mf.e_tot), "total_energy_Eh": energy}
        return energy * HARTREE_EV, -np.asarray(gradient) * HARTREE_EV / BOHR_A

    def model_label(self) -> str:
        return self.name

    def describe(self) -> dict:
        method = {"hf": "Hartree-Fock", "dft": f"Kohn-Sham DFT with {self.xc}",
                  "mp2": "MP2 on a Hartree-Fock reference"}[self.method]
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": 0.0,
                "parameters": {"method": self.method, "basis": self.basis, "xc": self.xc,
                               "charge_e": self.charge, "spin_2S": self.spin,
                               "dielectric": self.dielectric,
                               "solvation_model": self.solvation_model if self.dielectric
                               else None,
                               "conv_tol_Eh": self.conv_tol, "engine": "PySCF",
                               "engine_version": engine_versions()["pyscf"]},
                "approximations": [
                    f"{method} in the {self.basis} Gaussian basis set: basis-set "
                    "incompleteness is not extrapolated away.",
                    "Isolated molecule, all-electron unless the basis carries effective "
                    "core potentials; nonrelativistic.",
                    "Restricted reference for closed shells, unrestricted otherwise.",
                    (f"Implicit {self.solvation_model.upper()} continuum solvent with "
                     f"dielectric constant {self.dielectric:g}: electrostatics only, no "
                     "cavitation or dispersion terms." if self.dielectric else "Gas phase."),
                ],
                "references": ["Q. Sun et al., J. Chem. Phys. 153 (2020) 024109"]}


def pyscf_single_point(structure: Structure, method: str = "ccsd(t)",
                       basis: str = "cc-pvdz", charge: int = 0, spin: int = 0,
                       xc: str = "b3lyp") -> dict:
    """Energy and molecular properties from PySCF, including CCSD and CCSD(T).

    Returns the total energy in eV and hartree, the reference SCF energy, the
    dipole moment in debye, Mulliken charges, orbital energies and the
    HOMO-LUMO gap of the reference determinant.
    """
    reference = "dft" if method == "dft" else "hf"
    potential = PySCFPotential(reference, basis=basis, xc=xc, charge=charge, spin=spin)
    ok, why = potential.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    mf = potential.mean_field(structure)
    energy = float(mf.e_tot)
    correlation: Dict[str, float] = {}
    if method in ("mp2",):
        energy = float(require("pyscf.mp").MP2(mf).run().e_tot)
    elif method in ("ccsd", "ccsd(t)"):
        cc = require("pyscf.cc").CCSD(mf).run()
        if not cc.converged:
            raise EngineFailed("CCSD amplitudes did not converge.")
        energy = float(cc.e_tot)
        correlation["ccsd_correlation_Eh"] = float(cc.e_corr)
        if method == "ccsd(t)":
            triples = float(cc.ccsd_t())
            correlation["triples_Eh"] = triples
            energy += triples
    elif method not in ("hf", "dft"):
        raise ValueError("method must be hf, dft, mp2, ccsd or ccsd(t).")
    occupations = np.asarray(mf.mo_occ)
    levels = np.asarray(mf.mo_energy)
    if occupations.ndim > 1:
        occupations, levels = occupations[0], levels[0]
    homo = float(levels[occupations > 0].max())
    lumo = float(levels[occupations == 0].min()) if (occupations == 0).any() else None
    charges = mf.mulliken_pop(verbose=0)[1]
    return {"method": method, "basis": basis, "charge_e": charge, "spin_2S": spin,
            "energy_Eh": energy, "energy_eV": energy * HARTREE_EV,
            "reference_energy_Eh": float(mf.e_tot), **correlation,
            "dipole_debye": np.asarray(mf.dip_moment(verbose=0)).tolist(),
            "mulliken_charges_e": np.asarray(charges).tolist(),
            "homo_eV": homo * HARTREE_EV,
            "lumo_eV": None if lumo is None else lumo * HARTREE_EV,
            "homo_lumo_gap_eV": None if lumo is None else (lumo - homo) * HARTREE_EV,
            "engine_version": engine_versions()["pyscf"]}


def excited_states(structure: Structure, nstates: int = 5, method: str = "tddft",
                   basis: str = "def2-svp", xc: str = "b3lyp", charge: int = 0) -> dict:
    """Vertical singlet excitations from PySCF: TDDFT, TDA, or CIS (TDA on Hartree-Fock).

    Returns excitation energies in eV, wavelengths in nm and oscillator
    strengths, for a closed-shell ground state.
    """
    if method not in ("tddft", "tda", "cis"):
        raise ValueError("method must be tddft, tda or cis.")
    if not 1 <= int(nstates) <= 50:
        raise ValueError("nstates must lie between 1 and 50.")
    reference = PySCFPotential("hf" if method == "cis" else "dft", basis=basis, xc=xc,
                               charge=charge)
    ok, why = reference.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    if (int(structure.numbers.sum()) - charge) % 2:
        raise UnsupportedSystem("Excited states here need a closed-shell ground state.")
    mf = reference.mean_field(structure)
    tddft = require("pyscf.tddft")
    solver = tddft.TDDFT(mf) if method == "tddft" else tddft.TDA(mf)
    solver.nstates = int(nstates)
    solver.kernel()
    if not all(np.atleast_1d(solver.converged)):
        raise EngineFailed("The excited-state eigenvalue solver did not converge.")
    energies = np.asarray(solver.e) * HARTREE_EV
    return {"method": method, "basis": basis, "xc": xc if method != "cis" else None,
            "excitation_energies_eV": energies.tolist(),
            "wavelengths_nm": (1239.841984 / energies).tolist(),
            "oscillator_strengths": np.asarray(solver.oscillator_strength()).tolist(),
            "ground_state_energy_eV": float(mf.e_tot) * HARTREE_EV,
            "engine_version": engine_versions()["pyscf"],
            "approximations": ["Vertical excitations at the ground-state geometry; singlets "
                               "only; adiabatic linear response with the stated functional."
                               if method != "cis" else "Configuration interaction singles."]}


EPSILON_PER_OSCILLATOR = 2.3148e8


def absorption_spectrum(excitations: dict, energies_eV: Sequence[float],
                        fwhm_eV: float = 0.3) -> dict:
    """Molar absorption coefficient from vertical excitations, Gaussian-broadened.

    Each transition contributes ``A f g(nu)`` with ``g`` a normalised Gaussian
    in wavenumber and ``A = N_A e^2 / (4 eps0 m_e c^2 ln 10)``
    = 2.3148e8 L mol^-1 cm^-2, so that the integrated band
    ``int eps d nu`` equals ``A f`` (R. S. Mulliken, J. Chem. Phys. 7 (1939)
    14).  The width is a stated model parameter, not a computed line shape.
    """
    if not 0.01 <= fwhm_eV <= 2.0:
        raise ValueError("fwhm_eV must lie between 0.01 and 2 eV.")
    grid = np.asarray(energies_eV, dtype=float)
    to_cm = 8065.543937
    sigma = fwhm_eV * to_cm / (2 * np.sqrt(2 * np.log(2)))
    nu = grid * to_cm
    epsilon = np.zeros_like(nu)
    for energy, strength in zip(excitations["excitation_energies_eV"],
                                excitations["oscillator_strengths"]):
        centre = energy * to_cm
        epsilon += EPSILON_PER_OSCILLATOR * strength * np.exp(
            -0.5 * ((nu - centre) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return {"energies_eV": grid.tolist(), "wavelengths_nm": (1239.841984 / grid).tolist(),
            "molar_absorption_L_mol_cm": epsilon.tolist(), "fwhm_eV": fwhm_eV,
            "line_shape": "Gaussian in wavenumber"}


def conformers(smiles: str, count: int = 10, seed: int = 0, force_field: str = "mmff94",
               rms_threshold_A: float = 0.5, vacuum_A: float = 8.0) -> List[dict]:
    """Distinct low-energy conformers by ETKDGv3 embedding and force-field minimisation.

    Conformers are pruned by heavy-atom RMSD and returned sorted by energy,
    each as a Materia structure with its force-field energy in eV.
    """
    Chem = require("rdkit.Chem")
    AllChem = require("rdkit.Chem.AllChem")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise UnsupportedSystem(f"RDKit could not parse the SMILES {smiles!r}.")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = float(rms_threshold_A)
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(count), params=params))
    if not ids:
        raise EngineFailed(f"RDKit could not embed any conformer of {smiles!r}.")
    if force_field == "mmff94":
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)
    elif force_field == "uff":
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=10000)
    else:
        raise ValueError("force_field must be 'mmff94' or 'uff'.")
    out = []
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    for conf_id, (not_converged, energy) in zip(ids, results):
        if not_converged:
            continue
        positions = mol.GetConformer(conf_id).GetPositions()
        size = float((positions.max(axis=0) - positions.min(axis=0)).max() + 2 * vacuum_A)
        positions = positions - positions.mean(axis=0) + size / 2
        single = Chem.Mol(mol, confId=conf_id)
        single.RemoveAllConformers()
        conformer = Chem.Conformer(mol.GetConformer(conf_id))
        Geometry = require("rdkit.Geometry")
        for k, (x, y, z) in enumerate(positions):
            conformer.SetAtomPosition(k, Geometry.Point3D(float(x), float(y), float(z)))
        single.AddConformer(conformer, assignId=True)
        structure = Structure(numbers, positions, Cell(np.eye(3) * size, (False, False, False)))
        structure.info[MOLECULE_INFO_KEY] = {
            "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)), "input_smiles": str(smiles),
            "charge": int(Chem.GetFormalCharge(mol)), "molblock": Chem.MolToMolBlock(single),
            "embedding": "ETKDGv3", "seed": int(seed), "pre_optimisation": force_field,
            "conformer_id": int(conf_id)}
        out.append({"structure": structure, "energy_eV": float(energy) * KCAL_MOL_EV,
                    "conformer_id": int(conf_id)})
    out.sort(key=lambda c: c["energy_eV"])
    lowest = out[0]["energy_eV"] if out else 0.0
    for c in out:
        c["relative_energy_eV"] = c["energy_eV"] - lowest
    return out


def delta_scf(structure: Structure, hole: int = 0, particle: int = 0, basis: str = "def2-svp",
              xc: str = "b3lyp", charge: int = 0) -> dict:
    """Excited state by changing orbital occupations, with the maximum overlap method.

    One alpha electron is moved from orbital HOMO-``hole`` to LUMO+``particle``
    and the occupation is held by MOM (A. T. B. Gilbert, N. A. Besley and
    P. M. W. Gill, J. Phys. Chem. A 112 (2008) 13164) through the SCF.  The
    mixed-spin determinant is combined with the lowest triplet (high-spin
    determinant) by the spin-purification formula
    ``E_S = 2 E_mixed - E_T`` (T. Ziegler, A. Rauk and E. J. Baerends,
    Theor. Chim. Acta 43 (1977) 261).  Excitation energies are in eV relative
    to the closed-shell ground state.
    """
    gto = require("pyscf.gto")
    dft = require("pyscf.dft")
    scf = require("pyscf.scf")
    reference = PySCFPotential("dft", basis=basis, xc=xc, charge=charge)
    ok, why = reference.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    if (int(structure.numbers.sum()) - charge) % 2:
        raise UnsupportedSystem("Delta-SCF here starts from a closed-shell ground state.")
    atoms = [(pt.symbol(int(z)), tuple(float(v) for v in p))
             for z, p in zip(structure.numbers, structure.positions)]
    mol = gto.M(atom=atoms, unit="Angstrom", basis=basis, charge=charge, verbose=0)
    nocc = mol.nelectron // 2
    if hole < 0 or particle < 0 or nocc - 1 - hole < 0 or nocc + particle >= mol.nao:
        raise UnsupportedSystem("The requested orbitals are outside the basis.")
    ground = dft.RKS(mol)
    ground.xc = xc
    ground.conv_tol = 1e-10
    ground.kernel()
    unrestricted = dft.UKS(mol)
    unrestricted.xc = xc
    unrestricted.conv_tol = 1e-10
    unrestricted.kernel()
    if not (ground.converged and unrestricted.converged):
        raise EngineFailed("The ground-state SCF did not converge.")
    occupation = np.array(unrestricted.mo_occ).copy()
    occupation[0][nocc - 1 - hole] = 0.0
    occupation[0][nocc + particle] = 1.0
    excited = dft.UKS(mol)
    excited.xc = xc
    excited.conv_tol = 1e-10
    excited = scf.addons.mom_occ(excited, unrestricted.mo_coeff, occupation)
    excited.kernel(dm0=excited.make_rdm1(unrestricted.mo_coeff, occupation))
    if not excited.converged:
        raise EngineFailed("The excited-state SCF with MOM did not converge.")
    high_spin = dft.UKS(gto.M(atom=atoms, unit="Angstrom", basis=basis, charge=charge, spin=2,
                              verbose=0))
    high_spin.xc = xc
    high_spin.conv_tol = 1e-10
    high_spin.kernel()
    if not high_spin.converged:
        raise EngineFailed("The triplet SCF did not converge.")
    e0 = float(ground.e_tot)
    mixed = (float(excited.e_tot) - e0) * HARTREE_EV
    triplet = (float(high_spin.e_tot) - e0) * HARTREE_EV
    return {"hole": f"HOMO-{hole}" if hole else "HOMO",
            "particle": f"LUMO+{particle}" if particle else "LUMO",
            "mixed_eV": mixed, "triplet_eV": triplet, "singlet_eV": 2 * mixed - triplet,
            "triplet_s2": float(high_spin.spin_square()[0]),
            "mixed_s2": float(excited.spin_square()[0]), "basis": basis, "xc": xc,
            "approximations": [
                "Delta-SCF with the maximum overlap method; the singlet by spin "
                "purification assumes the high-spin triplet is the same orbital excitation.",
                "The lowest triplet from a high-spin SCF; if it is a different excitation the "
                "purified singlet is not meaningful."],
            "engine_version": engine_versions()["pyscf"]}


def field_response(structure: Structure, method: str = "hf", basis: str = "aug-cc-pvdz",
                   xc: str = "b3lyp", field_au: float = 1e-3, charge: int = 0) -> dict:
    """Dipole moment and static polarizability by finite uniform electric fields.

    A field ``F`` adds ``F . r`` to the one-electron Hamiltonian (electrons
    carry charge -1, origin at the nuclear charge centre).  From SCF energies
    at ``0, +-F, +-2F`` along each axis: ``mu = -dE/dF`` (four-point stencil)
    and ``alpha_ii = -d2E/dF_i^2``; off-diagonal components from the dipole of
    each field.  Atomic units in the result (e a0, a0^3) with converted values.
    The analytic zero-field dipole is returned for comparison.
    """
    if not 1e-5 <= field_au <= 5e-3:
        raise ValueError("field_au must lie between 1e-5 and 5e-3 atomic units.")
    reference = PySCFPotential("dft" if method == "dft" else "hf", basis=basis, xc=xc,
                               charge=charge)
    ok, why = reference.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    mol = reference.molecule(structure)
    charges = mol.atom_charges()
    centre = np.einsum("i,ix->x", charges, mol.atom_coords()) / charges.sum()
    with mol.with_common_orig(centre):
        dipole_ints = mol.intor_symmetric("int1e_r", comp=3)
    scf = require("pyscf.scf")
    dft = require("pyscf.dft")

    def solve(field):
        mf = dft.RKS(mol) if method == "dft" else scf.RHF(mol)
        if method == "dft":
            mf.xc = xc
        mf.conv_tol = 1e-12
        h1 = mf.get_hcore() + np.einsum("x,xij->ij", field, dipole_ints)
        mf.get_hcore = lambda *args: h1
        mf.kernel()
        if not mf.converged:
            raise EngineFailed("A finite-field SCF did not converge.")
        dm = mf.make_rdm1()
        electronic = -np.einsum("xij,ji->x", dipole_ints, dm)
        nuclear = np.einsum("i,ix->x", charges, mol.atom_coords() - centre)
        return float(mf.e_tot), electronic + nuclear

    e0, mu0 = solve(np.zeros(3))
    h = float(field_au)
    alpha_energy = np.zeros(3)
    mu_energy = np.zeros(3)
    alpha_dipole = np.zeros((3, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        ep, dp = solve(step)
        em, dm_ = solve(-step)
        e2p, _ = solve(2 * step)
        e2m, _ = solve(-2 * step)
        mu_energy[axis] = -(-e2p + 8 * ep - 8 * em + e2m) / (12 * h)
        alpha_energy[axis] = -(-e2p + 16 * ep - 30 * e0 + 16 * em - e2m) / (12 * h * h)
        alpha_dipole[:, axis] = (dp - dm_) / (2 * h)
    debye = 2.541746473
    bohr3_to_A3 = BOHR_A ** 3
    return {"method": method, "basis": basis, "field_au": h,
            "dipole_au": mu0.tolist(), "dipole_debye": (mu0 * debye).tolist(),
            "dipole_from_energy_au": mu_energy.tolist(),
            "polarizability_au": alpha_dipole.tolist(),
            "polarizability_diagonal_from_energy_au": alpha_energy.tolist(),
            "isotropic_polarizability_au": float(np.trace(alpha_dipole) / 3),
            "isotropic_polarizability_A3": float(np.trace(alpha_dipole) / 3 * bohr3_to_A3),
            "engine_version": engine_versions()["pyscf"]}


AU_IR_KM_MOL = 974.8801
HARTREE_J = 4.3597447222071e-18
BOHR_M = BOHR_A * 1e-10
AMU_KG = 1.66053906660e-27
LIGHT_CM_S = 2.99792458e10


def _rigid_complement(coords_bohr: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, int]:
    """Orthonormal basis of mass-weighted motions orthogonal to translations and rotations.

    Rotation about a principal axis whose moment is below 1e-3 of the largest
    is not a motion of the molecule (a linear molecule has two rotations).
    """
    n = len(masses)
    root = np.sqrt(masses)
    centre = np.einsum("i,ix->x", masses, coords_bohr) / masses.sum()
    r = coords_bohr - centre
    tensor = np.einsum("i,ij,ik->jk", masses, r, r)
    moments, axes = np.linalg.eigh(np.trace(tensor) * np.eye(3) - tensor)
    vectors = []
    for axis in range(3):
        v = np.zeros((n, 3))
        v[:, axis] = root
        vectors.append(v.ravel())
    for moment, axis in zip(moments, axes.T):
        if moment > 1e-3 * moments.max():
            vectors.append((np.cross(axis, r) * root[:, None]).ravel())
    q, _ = np.linalg.qr(np.array(vectors).T, mode="complete")
    rank = len(vectors)
    return q[:, rank:], rank


def _mean_field_dipole(reference: "PySCFPotential", mol, field=None, dipole_ints=None):
    scf = require("pyscf.scf")
    dft = require("pyscf.dft")
    if reference.method == "dft":
        mf = dft.RKS(mol) if reference.spin == 0 else dft.UKS(mol)
        mf.xc = reference.xc
    else:
        mf = scf.RHF(mol) if reference.spin == 0 else scf.UHF(mol)
    mf.conv_tol = 1e-12
    if field is not None:
        h1 = mf.get_hcore() + np.einsum("x,xij->ij", field, dipole_ints)
        mf.get_hcore = lambda *args: h1
    mf.kernel()
    if not mf.converged:
        raise EngineFailed("A displaced or field-perturbed SCF did not converge.")
    dm = mf.make_rdm1()
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    charges = mol.atom_charges()
    with mol.with_common_orig((0, 0, 0)):
        ints = mol.intor_symmetric("int1e_r", comp=3)
    electronic = -np.einsum("xij,ji->x", ints, dm)
    nuclear = np.einsum("i,ix->x", charges, mol.atom_coords())
    return mf, electronic + nuclear


def _polarizability_at(reference: "PySCFPotential", mol, field_au: float) -> np.ndarray:
    with mol.with_common_orig((0, 0, 0)):
        ints = mol.intor_symmetric("int1e_r", comp=3)
    alpha = np.zeros((3, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = field_au
        _, plus = _mean_field_dipole(reference, mol, step, ints)
        _, minus = _mean_field_dipole(reference, mol, -step, ints)
        alpha[:, axis] = (plus - minus) / (2 * field_au)
    return 0.5 * (alpha + alpha.T)


def vibrational_spectra(structure: Structure, method: str = "hf", basis: str = "def2-svp",
                        xc: str = "b3lyp", charge: int = 0, spin: int = 0, raman: bool = False,
                        step_A: float = 0.005, field_au: float = 1e-3,
                        max_gradient_au: float = 1e-3) -> dict:
    """Harmonic IR intensities and, optionally, Raman activities of a molecule.

    Normal modes from PySCF's analytic Hessian, mass-weighted with the
    structure's masses and projected onto the space orthogonal to rigid
    translations and rotations.  Dipole derivatives (atomic polar tensors) are
    central differences of the SCF dipole over Cartesian displacements of
    ``step_A``; IR intensities ``(N_A pi / 3 c^2) |d mu / d Q|^2`` in km/mol.
    With ``raman``, polarizability derivatives come from finite fields of
    ``field_au`` at each displaced geometry, and the Raman activity is
    ``45 a'^2 + 7 g'^2`` in A^4/amu with the depolarization ratio for
    linearly polarized light.  The geometry must be a stationary point.
    """
    if method not in ("hf", "dft"):
        raise ValueError("method must be 'hf' or 'dft'.")
    if not 1e-4 <= step_A <= 0.02:
        raise ValueError("step_A must lie between 1e-4 and 0.02 A.")
    if not 1e-4 <= field_au <= 5e-3:
        raise ValueError("field_au must lie between 1e-4 and 5e-3 atomic units.")
    reference = PySCFPotential(method, basis=basis, xc=xc, charge=charge, spin=spin)
    ok, why = reference.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    n = len(structure)
    if n < 2:
        raise UnsupportedSystem("A single atom has no vibrations.")
    mol = reference.molecule(structure)
    mf, _ = _mean_field_dipole(reference, mol)
    gradient = np.asarray(mf.nuc_grad_method().kernel())
    largest = float(np.abs(gradient).max())
    if largest > max_gradient_au:
        raise UnsupportedSystem(f"The largest gradient is {largest:.2e} Eh/bohr; relax the "
                                "molecule first. Harmonic spectra are defined at a stationary "
                                "point.")
    hessian = np.asarray(mf.Hessian().kernel()).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    masses = np.asarray(structure.masses(), dtype=float)
    inverse_root = np.repeat(1 / np.sqrt(masses), 3)
    weighted = hessian * inverse_root[:, None] * inverse_root[None, :]
    complement, rigid = _rigid_complement(mol.atom_coords(), masses)
    eigenvalues, vectors = np.linalg.eigh(complement.T @ weighted @ complement)
    modes = complement @ vectors
    to_cm = np.sqrt(HARTREE_J / (BOHR_M ** 2 * AMU_KG)) / (2 * np.pi * LIGHT_CM_S)
    wavenumbers = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * to_cm
    cartesian = modes * inverse_root[:, None]

    step = step_A / BOHR_A
    polar_tensor = np.zeros((3, 3 * n))
    alpha_derivative = np.zeros((3, 3, 3 * n)) if raman else None
    coords = mol.atom_coords()
    for k in range(3 * n):
        displaced = {}
        for sign in (1, -1):
            moved = coords.copy()
            moved[k // 3, k % 3] += sign * step
            pmol = mol.set_geom_(moved, unit="Bohr", inplace=False)
            _, dipole = _mean_field_dipole(reference, pmol)
            displaced[sign] = (dipole, _polarizability_at(reference, pmol, field_au)
                               if raman else None)
        polar_tensor[:, k] = (displaced[1][0] - displaced[-1][0]) / (2 * step)
        if raman:
            alpha_derivative[:, :, k] = (displaced[1][1] - displaced[-1][1]) / (2 * step)
    sum_rule = polar_tensor.reshape(3, n, 3).sum(axis=1) - charge * np.eye(3)
    dmu_dq = polar_tensor @ cartesian
    intensities = AU_IR_KM_MOL * np.sum(dmu_dq ** 2, axis=0)
    out = {"method": method, "basis": basis, "xc": reference.xc,
           "wavenumbers_cm1": wavenumbers.tolist(), "ir_intensity_km_mol": intensities.tolist(),
           "rigid_modes_removed": rigid, "largest_gradient_au": largest,
           "atomic_polar_tensor_au": polar_tensor.tolist(),
           "apt_sum_rule_error_au": float(np.abs(sum_rule).max()),
           "step_A": step_A, "masses_amu": masses.tolist(),
           "engine_version": engine_versions()["pyscf"]}
    if raman:
        dalpha = np.einsum("abk,km->abm", alpha_derivative, cartesian)
        mean = np.einsum("aam->m", dalpha) / 3
        anisotropy = 0.5 * ((dalpha[0, 0] - dalpha[1, 1]) ** 2 + (dalpha[1, 1] - dalpha[2, 2]) ** 2
                            + (dalpha[2, 2] - dalpha[0, 0]) ** 2
                            + 6 * (dalpha[0, 1] ** 2 + dalpha[1, 2] ** 2 + dalpha[0, 2] ** 2))
        activity = (45 * mean ** 2 + 7 * anisotropy) * BOHR_A ** 4
        denominator = 45 * mean ** 2 + 4 * anisotropy
        depolarization = np.where(denominator > 1e-12 * max(1.0, denominator.max()),
                                  3 * anisotropy / np.where(denominator > 0, denominator, 1),
                                  np.nan)
        out.update({"raman_activity_A4_amu": activity.tolist(),
                    "depolarization_ratio": depolarization.tolist(), "field_au": field_au})
    return out


def core_binding_energy(structure: Structure, atom: int, basis: str = "cc-pcvtz",
                        xc: str = "tpss", charge: int = 0) -> dict:
    """1s core-electron binding energy of one atom by Delta-SCF with a core hole.

    The closed-shell ground state is computed first.  Its core orbitals
    (eigenvalue below -5 Eh) are rotated among themselves so that one of them
    has the largest possible Mulliken population on ``atom``, so that a hole can
    sit on one atom even when equivalent atoms share delocalised canonical core
    orbitals.  One alpha electron is removed from the localised
    1s orbital of ``atom`` and the cation is solved with the maximum overlap
    method (A. T. B. Gilbert, N. A. Besley and P. M. W. Gill, J. Phys. Chem. A
    112 (2008) 13164), letting all orbitals relax around the hole.  The
    binding energy is ``E(cation, core hole) - E(neutral)``: the XPS line
    position relative to the vacuum level, nonrelativistic.  A core-valence
    basis such as cc-pCVTZ is needed for the relaxation; hydrogen, which has no
    core-valence set, receives the matching cc-pVXZ basis.
    """
    gto = require("pyscf.gto")
    dft = require("pyscf.dft")
    scf = require("pyscf.scf")
    try:
        isolated(structure, "Core-level calculations")
    except UnsupportedSystem:
        raise
    if not 0 <= atom < len(structure):
        raise UnsupportedSystem(f"There is no atom {atom}.")
    z = int(structure.numbers[atom])
    if z < 3:
        raise UnsupportedSystem("Hydrogen and helium have no core electrons to ionise.")
    if (int(structure.numbers.sum()) - charge) % 2:
        raise UnsupportedSystem("The core-hole calculation starts from a closed-shell ground "
                                "state.")
    atoms = [(pt.symbol(int(n)), tuple(float(v) for v in p))
             for n, p in zip(structure.numbers, structure.positions)]
    basis_map = {}
    for symbol, _ in atoms:
        name = basis
        if symbol in ("H", "He") and basis.lower().startswith("cc-pcv"):
            name = "cc-pv" + basis.lower()[len("cc-pcv"):]
        basis_map[symbol] = name
    try:
        mol = gto.M(atom=atoms, unit="Angstrom", basis=basis_map, charge=charge, verbose=0)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise UnsupportedSystem(f"PySCF could not build the molecule in basis {basis}: "
                                f"{exc}") from None

    def solver(m):
        mf = dft.UKS(m)
        mf.xc = xc
        mf.conv_tol = 1e-9
        mf.max_cycle = 300
        return mf

    ground = dft.RKS(mol)
    ground.xc = xc
    ground.conv_tol = 1e-10
    ground.kernel()
    if not ground.converged:
        raise EngineFailed("The ground-state SCF did not converge.")
    occupied = ground.mo_occ > 0
    core = np.flatnonzero(occupied & (ground.mo_energy < -5.0))
    if not len(core):
        raise UnsupportedSystem("The molecule has no core orbitals below -5 Eh.")
    coefficients = ground.mo_coeff.copy()
    overlap = mol.intor_symmetric("int1e_ovlp")
    first, last = mol.aoslice_by_atom()[atom][2:]
    block = ground.mo_coeff[:, core]
    product = overlap @ block
    on_atom = block[first:last].T @ product[first:last]
    weights, rotation = np.linalg.eigh(0.5 * (on_atom + on_atom.T))
    coefficients[:, core] = block @ rotation[:, ::-1]
    population = []
    for k in core:
        c = coefficients[:, k]
        population.append(float(c[first:last] @ (overlap @ c)[first:last]))
    chosen = int(core[int(np.argmax(population))])
    if max(population) < 0.9:
        raise EngineFailed(f"No core orbital is localised on atom {atom} (largest population "
                           f"{max(population):.2f}).")
    cation = gto.M(atom=atoms, unit="Angstrom", basis=basis_map, charge=charge + 1, spin=1,
                   verbose=0)
    occupation = np.array([ground.mo_occ / 2, ground.mo_occ / 2])
    occupation[0][chosen] = 0.0
    hole = scf.addons.mom_occ(solver(cation), (coefficients, coefficients), occupation)
    hole.kernel(dm0=hole.make_rdm1((coefficients, coefficients), occupation))
    if not hole.converged:
        raise EngineFailed("The core-hole SCF with MOM did not converge.")
    remaining = float(np.sum(hole.mo_occ[0][hole.mo_energy[0] < -5.0]))
    expected = len(core) - 1
    if abs(remaining - expected) > 1e-6:
        raise EngineFailed("The core hole collapsed during the SCF: the alpha core shell is "
                           "full again.")
    binding = (float(hole.e_tot) - float(ground.e_tot)) * HARTREE_EV
    localized = coefficients[:, chosen]
    koopmans = -float(localized @ ground.get_fock() @ localized) * HARTREE_EV
    return {"atom": atom, "element": pt.symbol(z), "binding_energy_eV": binding,
            "core_population": max(population), "basis": basis, "xc": xc,
            "frozen_orbital_estimate_eV": koopmans,
            "cation_s2": float(hole.spin_square()[0]),
            "approximations": [
                "Delta-SCF: separate SCF solutions for the neutral ground state and the "
                "core-ionised cation, held by the maximum overlap method.",
                "Nonrelativistic: scalar-relativistic corrections, about 0.1 eV for carbon "
                "and 0.3 to 0.4 eV for oxygen, are not added.",
                "Vertical ionisation at the given geometry; no vibrational structure."],
            "references": ["P. S. Bagus, Phys. Rev. 139 (1965) A619",
                           "A. T. B. Gilbert, N. A. Besley and P. M. W. Gill, J. Phys. Chem. A "
                           "112 (2008) 13164",
                           "N. Pueyo Bellafont, P. S. Bagus and F. Illas, J. Chem. Phys. 142 "
                           "(2015) 214102"],
            "engine_version": engine_versions()["pyscf"]}


def _nmr_solver(mf, nmr, dft: bool):
    """A GIAO solver whose CPHF response accepts any number of trial vectors.

    pyscf-properties 0.1.0 reshapes the response input to exactly three
    vectors, which newer PySCF CPHF solvers do not always pass; the response
    is the same induced potential, computed for however many arrive.  Pure
    functionals have no coupled response for an imaginary perturbation.
    """
    from pyscf import lib
    from pyscf.prop.nmr import rhf as rhf_nmr

    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    occupied = mo_coeff[:, mo_occ > 0]
    nmo, nocc = mo_coeff.shape[1], occupied.shape[1]
    response = mf.gen_response(singlet=True, hermi=2)

    def induced(mo1):
        dm1 = np.asarray([mo_coeff @ (x * 2) @ occupied.T.conj()
                          for x in mo1.reshape(-1, nmo, nocc)])
        dm1 = dm1 - dm1.conj().transpose(0, 2, 1)
        return lib.einsum("xpq,pi,qj->xij", response(dm1), mo_coeff.conj(), occupied).ravel()

    coupled = (not dft) or mf._numint.libxc.is_hybrid_xc(mf.xc)
    base = nmr.RKS if dft else nmr.RHF

    class Solver(base):
        def solve_mo1(self, mo_energy=None, mo_coeff=None, mo_occ=None, h1=None, s1=None,
                      with_cphf=None):
            return rhf_nmr.solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1, s1,
                                     induced if coupled else False)

    return Solver(mf)


def nmr_shielding(structure: Structure, method: str = "hf", basis: str = "pcseg-2",
                  xc: str = "b3lyp", charge: int = 0) -> dict:
    """Absolute NMR shielding tensors (ppm) with gauge-including atomic orbitals.

    Closed-shell Hartree-Fock or Kohn-Sham shieldings from the PySCF
    properties extension (``pip install pyscf-properties``, Apache-2.0),
    with GIAOs, so results do not depend on where the molecule sits.
    Returns the full tensor, isotropic value and anisotropy
    ``s33 - (s11 + s22) / 2`` (Haeberlen order) for every nucleus.
    Nonrelativistic, at the given geometry, without vibrational averaging.
    """
    if method not in ("hf", "dft"):
        raise ValueError("method must be 'hf' or 'dft'.")
    reference = PySCFPotential(method, basis=basis, xc=xc, charge=charge)
    ok, why = reference.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    if (int(structure.numbers.sum()) - charge) % 2:
        raise UnsupportedSystem("NMR shieldings here need a closed-shell molecule.")
    try:
        from pyscf.prop import nmr
    except ImportError:
        raise EngineMissing("NMR shieldings need the PySCF properties extension: "
                            "pip install pyscf-properties.") from None
    mf = reference.mean_field(structure)
    solver = _nmr_solver(mf, nmr, method == "dft")
    solver.verbose = 0
    tensors = np.asarray(solver.kernel())
    nuclei = []
    for k, tensor in enumerate(tensors):
        symmetric = 0.5 * (tensor + tensor.T)
        principal = np.linalg.eigvalsh(symmetric)
        iso = float(np.trace(tensor) / 3)
        order = sorted(principal, key=lambda v: abs(v - iso))
        anisotropy = float(order[2] - (order[0] + order[1]) / 2)
        nuclei.append({"atom": k, "element": pt.symbol(int(structure.numbers[k])),
                       "isotropic_ppm": iso, "anisotropy_ppm": anisotropy,
                       "tensor_ppm": tensor.tolist()})
    return {"nuclei": nuclei, "method": method, "basis": basis, "xc": reference.xc,
            "gauge": "GIAO", "engine_version": engine_versions()["pyscf"],
            "approximations": [
                "Closed-shell SCF shieldings with gauge-including atomic orbitals; no "
                "relativistic, vibrational or solvent corrections.",
                f"{basis} basis; shieldings converge slowly with basis size."],
            "references": ["K. Wolinski, J. F. Hinton and P. Pulay, J. Am. Chem. Soc. 112 "
                           "(1990) 8251", "Q. Sun et al., J. Chem. Phys. 153 (2020) 024109"]}


def chemical_shifts(structure: Structure, reference: Structure, method: str = "hf",
                    basis: str = "pcseg-2", xc: str = "b3lyp") -> dict:
    """Shifts ``delta = sigma_ref - sigma`` against a reference molecule at the same level.

    The reference shielding of each element is the mean over its nuclei in
    ``reference`` (for example tetramethylsilane for 1H and 13C).  Elements
    absent from the reference are refused.
    """
    sample = nmr_shielding(structure, method, basis, xc)
    ref = nmr_shielding(reference, method, basis, xc)
    by_element: Dict[str, List[float]] = {}
    for n in ref["nuclei"]:
        by_element.setdefault(n["element"], []).append(n["isotropic_ppm"])
    shifts = []
    for n in sample["nuclei"]:
        if n["element"] not in by_element:
            continue
        sigma_ref = float(np.mean(by_element[n["element"]]))
        shifts.append({"atom": n["atom"], "element": n["element"],
                       "shift_ppm": sigma_ref - n["isotropic_ppm"],
                       "shielding_ppm": n["isotropic_ppm"], "reference_ppm": sigma_ref})
    if not shifts:
        raise UnsupportedSystem("The reference molecule contains none of this molecule's "
                                "elements.")
    return {"shifts": shifts, "method": method, "basis": basis,
            "reference_shieldings_ppm": {k: float(np.mean(v)) for k, v in by_element.items()}}
