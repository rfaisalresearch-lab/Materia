"""QM/MM with electrostatic embedding through PySCF.

``E = E_QM[rho in the field of the MM charges] + E_LJ(QM-MM) + E_MM(MM-MM)``

The QM region (whole molecules; no covalent bond may cross the boundary, so
no link atoms are needed) is treated by PySCF Hartree-Fock or DFT with the
MM point charges in its one-electron Hamiltonian (``pyscf.qmmm``).  Forces on
QM atoms and on MM charges are analytic.  QM-MM short-range repulsion and
dispersion are Lennard-Jones terms with Lorentz-Berthelot mixing; MM-MM terms
are Coulomb plus Lennard-Jones between different molecules.  The MM model
shipped is rigid TIP3P water (W. L. Jorgensen et al., J. Chem. Phys. 79 (1983)
926); intramolecular MM terms are not included, so MM atoms should be marked
fixed (``structure.fixed``) in relaxation and dynamics; their forces are
still computed and reported.

References
----------
A. Warshel and M. Levitt, J. Mol. Biol. 103 (1976) 227.
Q. Sun et al., J. Chem. Phys. 153 (2020) 024109 (PySCF).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..core_model.units import BOHR_A, COULOMB_K_EV_A, HARTREE_EV
from ..elements import periodic_table as pt
from ..provenance import Fidelity
from .molecular import KCAL_MOL_EV, PySCFPotential, require
from .potentials import Potential, UnsupportedSystem

TIP3P = {"charges": {"O": -0.834, "H": 0.417},
         "lj": {"O": (3.15061, 0.1521 * KCAL_MOL_EV), "H": (0.0, 0.0)}}
QM_LJ_DEFAULT = {"O": (3.15061, 0.1521 * KCAL_MOL_EV), "H": (0.0, 0.0)}


class QMMMPotential(Potential):
    """Electrostatically embedded QM region in rigid fixed-charge MM molecules."""

    fidelity = Fidelity.TIER3_EXTERNAL

    def __init__(self, qm_atoms: Sequence[int], mm_molecules: Sequence[Sequence[int]],
                 qm: Optional[PySCFPotential] = None,
                 qm_lj: Optional[Dict[str, Tuple[float, float]]] = None,
                 mm_model: Optional[dict] = None) -> None:
        self.qm_atoms = [int(i) for i in qm_atoms]
        self.mm_molecules = [[int(i) for i in m] for m in mm_molecules]
        self.mm_atoms = [i for m in self.mm_molecules for i in m]
        if set(self.qm_atoms) & set(self.mm_atoms):
            raise ValueError("An atom cannot be both QM and MM.")
        self.qm = qm or PySCFPotential("dft", basis="def2-svp", xc="b3lyp")
        if self.qm.dielectric:
            raise ValueError("Use either an implicit solvent or explicit MM, not both.")
        self.qm_lj = dict(QM_LJ_DEFAULT if qm_lj is None else qm_lj)
        self.mm_model = mm_model or TIP3P
        self.name = f"qmmm[{self.qm.name}|tip3p]"
        self.cutoff_A = 0.0

    def model_label(self) -> str:
        return self.name

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        if any(structure.cell.pbc):
            return False, "QM/MM here is for isolated clusters; no periodic MM is implemented."
        n = len(structure)
        if sorted(self.qm_atoms + self.mm_atoms) != list(range(n)):
            return False, "Every atom must be in exactly one of the QM or MM regions."
        symbols = [pt.symbol(int(z)) for z in structure.numbers]
        for i in self.mm_atoms:
            if symbols[i] not in self.mm_model["charges"]:
                return False, f"The MM model has no charge for {symbols[i]} (atom {i})."
        for i in self.qm_atoms:
            if symbols[i] not in self.qm_lj:
                return False, (f"No QM-MM Lennard-Jones parameters for {symbols[i]}; supply "
                               "qm_lj for it.")
        qm_structure = self._qm_structure(structure)
        return self.qm.composition(qm_structure)

    def _qm_structure(self, structure: Structure) -> Structure:
        return Structure(structure.numbers[self.qm_atoms],
                         np.asarray(structure.positions)[self.qm_atoms], structure.cell)

    def _lj(self, sym_a: str, sym_b: str, table_a, table_b):
        sa, ea = table_a[sym_a]
        sb, eb = table_b[sym_b]
        if ea == 0 or eb == 0:
            return 0.0, 0.0
        return 0.5 * (sa + sb), float(np.sqrt(ea * eb))

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        qmmm = require("pyscf.qmmm")
        positions = np.asarray(structure.positions, dtype=float)
        symbols = [pt.symbol(int(z)) for z in structure.numbers]
        mm_xyz = positions[self.mm_atoms]
        mm_q = np.array([self.mm_model["charges"][symbols[i]] for i in self.mm_atoms])
        mol = self.qm.molecule(self._qm_structure(structure))
        if self.qm.method == "dft":
            dft = require("pyscf.dft")
            base = dft.RKS(mol) if self.qm.spin == 0 else dft.UKS(mol)
            base.xc = self.qm.xc
        else:
            scf = require("pyscf.scf")
            base = scf.RHF(mol) if self.qm.spin == 0 else scf.UHF(mol)
        embedded = qmmm.mm_charge(base, mm_xyz, mm_q, unit="Angstrom")
        embedded.conv_tol = self.qm.conv_tol
        embedded.max_cycle = self.qm.max_cycle
        embedded.kernel()
        if not embedded.converged:
            from .molecular import EngineFailed
            raise EngineFailed("The embedded QM self-consistent field did not converge.")
        gradient = embedded.nuc_grad_method()
        qm_grad = gradient.kernel()
        dm = embedded.make_rdm1()
        mm_grad = gradient.grad_hcore_mm(dm) + gradient.grad_nuc_mm()
        energy = float(embedded.e_tot) * HARTREE_EV
        forces = np.zeros_like(positions)
        forces[self.qm_atoms] = -np.asarray(qm_grad) * HARTREE_EV / BOHR_A
        forces[self.mm_atoms] = -np.asarray(mm_grad) * HARTREE_EV / BOHR_A
        e_lj, f_lj = self._lennard_jones_and_mm(positions, symbols, mm_q)
        return energy + e_lj, forces + f_lj

    def _lennard_jones_and_mm(self, positions, symbols, mm_q):
        energy = 0.0
        forces = np.zeros_like(positions)
        charge = dict(zip(self.mm_atoms, mm_q))
        molecule_of = {i: k for k, m in enumerate(self.mm_molecules) for i in m}
        pairs = [(i, j, "qm") for i in self.qm_atoms for j in self.mm_atoms]
        pairs += [(i, j, "mm") for a, i in enumerate(self.mm_atoms) for j in self.mm_atoms[a + 1:]
                  if molecule_of[i] != molecule_of[j]]
        for i, j, kind in pairs:
            d = positions[j] - positions[i]
            r = float(np.linalg.norm(d))
            table_i = self.qm_lj if kind == "qm" else self.mm_model["lj"]
            sigma, eps = self._lj(symbols[i], symbols[j], table_i, self.mm_model["lj"])
            de_dr = 0.0
            if eps:
                sr6 = (sigma / r) ** 6
                energy += 4 * eps * (sr6 * sr6 - sr6)
                de_dr += 4 * eps * (-12 * sr6 * sr6 + 6 * sr6) / r
            if kind == "mm":
                qq = COULOMB_K_EV_A * charge[i] * charge[j]
                energy += qq / r
                de_dr += -qq / r ** 2
            forces[i] += de_dr * d / r
            forces[j] -= de_dr * d / r
        return energy, forces

    def describe(self) -> dict:
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": 0.0,
                "parameters": {"qm": self.qm.describe()["parameters"],
                               "qm_atoms": self.qm_atoms, "mm_molecules": self.mm_molecules,
                               "mm_model": "TIP3P", "qm_lj": self.qm_lj},
                "approximations": [
                    "Electrostatic embedding: MM point charges polarise the QM density; the "
                    "MM charges are not polarised by the QM region.",
                    "QM-MM van der Waals by Lennard-Jones with Lorentz-Berthelot mixing; no "
                    "covalent bonds across the boundary, no link atoms.",
                    "Rigid TIP3P water with no intramolecular MM terms: MM molecules must be "
                    "held rigid or fixed in relaxation and dynamics.",
                ] + self.qm.describe()["approximations"][:1],
                "references": ["A. Warshel and M. Levitt, J. Mol. Biol. 103 (1976) 227",
                               "W. L. Jorgensen et al., J. Chem. Phys. 79 (1983) 926",
                               "Q. Sun et al., J. Chem. Phys. 153 (2020) 024109"]}
