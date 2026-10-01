"""Psi4 as a Materia potential, run in its own Python environment.

Psi4 (LGPL-3.0; D. G. A. Smith et al., J. Chem. Phys. 152 (2020) 184108) is
found as a Python interpreter that can ``import psi4``: the one named by
``MATERIA_PSI4_PYTHON``, this interpreter, or a conda environment (a
``materia-psi4`` environment first).  Each energy or gradient is one worker
process with one thread, a timeout and a private scratch directory.

Exact (``pk``) integrals are the default so that results can be compared
with other engines to the SCF convergence; density fitting is available with
``scf_type="df"`` and is reported as an approximation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core_model.structure import Structure
from ...core_model.units import BOHR_A, HARTREE_EV
from ...elements import periodic_table as pt
from ...provenance import Fidelity
from ...physics.potentials import Potential, PotentialError, UnsupportedSystem

PYTHON_ENV = "MATERIA_PSI4_PYTHON"
WORKER = Path(__file__).with_name("worker.py")
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1"}
REFERENCE = "D. G. A. Smith et al., J. Chem. Phys. 152 (2020) 184108"


class Psi4Missing(UnsupportedSystem):
    pass


class Psi4Failed(PotentialError):
    pass


def _can_import(python: str) -> bool:
    try:
        done = subprocess.run([python, "-c", "import psi4"], capture_output=True, timeout=120,
                              env={**os.environ, **THREAD_ENV})
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def installed_pythons() -> List[str]:
    """Interpreters that have a psi4 package on disk, without starting them."""
    named = os.environ.get(PYTHON_ENV)
    if named:
        return [named]
    found = []
    if importlib.util.find_spec("psi4") is not None:
        found.append(sys.executable)
    from ...system import conda_environments, environment_python

    for env in conda_environments("materia-psi4"):
        python = environment_python(env)
        if python.is_file() and (any(env.glob("lib/python3*/site-packages/psi4/__init__.py"))
                                 or (env / "Lib" / "site-packages" / "psi4").is_dir()):
            found.append(str(python))
    return found


@lru_cache(maxsize=1)
def find_python() -> Optional[str]:
    """The first interpreter that can actually import Psi4."""
    for python in installed_pythons():
        if _can_import(python):
            return python
    return None


def run_job(job: dict, timeout_s: float) -> dict:
    python = find_python()
    if python is None:
        raise Psi4Missing("Psi4 was not found: install it in a conda environment "
                          "(conda create -n materia-psi4 -c conda-forge psi4) or set "
                          f"{PYTHON_ENV}.")
    try:
        done = subprocess.run([python, str(WORKER)], input=json.dumps(job), capture_output=True,
                              text=True, timeout=timeout_s, env={**os.environ, **THREAD_ENV})
    except subprocess.TimeoutExpired:
        raise Psi4Failed(f"Psi4 exceeded {timeout_s:g} s.") from None
    try:
        reply = json.loads(done.stdout)
    except json.JSONDecodeError:
        raise Psi4Failed(f"Psi4 returned no result: {done.stderr[-600:]}") from None
    if not reply.get("ok"):
        raise Psi4Failed(f"Psi4 failed: {reply.get('error')}")
    return reply


class Psi4Potential(Potential):
    """Energies and analytic gradients from Psi4 for an isolated molecule."""

    fidelity = Fidelity.TIER3_EXTERNAL

    def __init__(self, method: str = "hf", basis: str = "cc-pvdz", charge: int = 0,
                 multiplicity: int = 1, scf_type: str = "pk", timeout_s: float = 3600.0,
                 memory_mb: int = 1000, grid: Optional[Tuple[int, int]] = None) -> None:
        if scf_type not in ("pk", "df", "direct"):
            raise ValueError("scf_type must be 'pk', 'df' or 'direct'.")
        if multiplicity < 1:
            raise ValueError("multiplicity must be at least 1.")
        self.method = method.lower()
        self.basis = basis
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.scf_type = scf_type
        self.timeout_s = float(timeout_s)
        self.memory_mb = int(memory_mb)
        self.grid = grid
        self.name = f"psi4/{self.method}/{basis}"
        self.cutoff_A = 0.0
        self.last: Dict[str, object] = {}

    def model_label(self) -> str:
        return self.name

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        if any(structure.cell.pbc):
            return False, "Psi4 here treats isolated molecules; the structure is periodic."
        electrons = int(structure.numbers.sum()) - self.charge
        if (electrons + self.multiplicity - 1) % 2:
            return False, (f"{electrons} electrons cannot have multiplicity "
                           f"{self.multiplicity}.")
        return True, ""

    def _job(self, structure: Structure, gradient: bool, method: Optional[str] = None) -> dict:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        job = {"symbols": [pt.symbol(int(z)) for z in structure.numbers],
               "positions_A": np.asarray(structure.positions, dtype=float).tolist(),
               "charge": self.charge, "multiplicity": self.multiplicity,
               "basis": self.basis, "method": method or self.method, "gradient": gradient,
               "scf_type": self.scf_type, "memory_mb": self.memory_mb,
               "reference": "rhf" if self.multiplicity == 1 else "uhf"}
        if self.grid:
            job["dft_radial_points"], job["dft_spherical_points"] = self.grid
        return job

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        reply = run_job(self._job(structure, True), self.timeout_s)
        self.last = reply
        forces = -np.asarray(reply["gradient_Eh_bohr"]) * HARTREE_EV / BOHR_A
        return float(reply["energy_Eh"]) * HARTREE_EV, forces

    def single_point(self, structure: Structure, method: Optional[str] = None) -> dict:
        """Energy only, for methods without gradients here (for example ccsd(t))."""
        reply = run_job(self._job(structure, False, method), self.timeout_s)
        return {"energy_Eh": reply["energy_Eh"], "energy_eV": reply["energy_Eh"] * HARTREE_EV,
                "variables": reply["variables"], "psi4_version": reply["psi4_version"]}

    def excited_states(self, structure: Structure, nstates: int = 5, tda: bool = False) -> dict:
        """Singlet TDDFT (or TDA) excitation energies and oscillator strengths."""
        if not 1 <= int(nstates) <= 50:
            raise ValueError("nstates must lie between 1 and 50.")
        if self.multiplicity != 1:
            raise UnsupportedSystem("Excited states here need a closed-shell ground state.")
        job = self._job(structure, False)
        job["tdscf_states"] = int(nstates)
        job["tda"] = bool(tda)
        reply = run_job(job, self.timeout_s)
        energies = np.asarray(reply["excitation_energies_Eh"]) * HARTREE_EV
        return {"method": "tda" if tda else "tddft", "basis": self.basis, "xc": self.method,
                "excitation_energies_eV": energies.tolist(),
                "wavelengths_nm": (1239.841984 / energies).tolist(),
                "oscillator_strengths": reply["oscillator_strengths"],
                "psi4_version": reply["psi4_version"]}

    def describe(self) -> dict:
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": 0.0,
                "parameters": {"engine": "Psi4", "method": self.method, "basis": self.basis,
                               "charge_e": self.charge, "multiplicity": self.multiplicity,
                               "scf_type": self.scf_type,
                               "engine_version": self.last.get("psi4_version")},
                "approximations": [
                    f"{self.method} in the {self.basis} basis; isolated molecule, all "
                    "electrons correlated, nonrelativistic.",
                    "Exact four-index integrals." if self.scf_type == "pk" else
                    "Density-fitted integrals."],
                "references": [REFERENCE]}
