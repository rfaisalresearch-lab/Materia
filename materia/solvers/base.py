"""Solver interface and capability model.

A solver declares exactly which physical quantities it can produce.  When the
application is asked for something outside that set it returns an
:class:`~materia.provenance.Result` with ``origin=UNSUPPORTED``, the reason,
and the list of solvers that *could* answer it -- it never substitutes a
different model silently and never fabricates a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from ..core_model.structure import Structure
from ..provenance import Convergence, Fidelity, Origin, Provenance, Result, unsupported


class Capability(str, Enum):
    """Quantities a solver may be able to produce."""

    ENERGY = "energy"
    FORCES = "forces"
    STRESS = "stress"
    RELAX = "relax"
    DYNAMICS = "dynamics"
    BAND_STRUCTURE = "band_structure"
    TOTAL_DOS = "total_dos"
    LOCAL_DOS = "local_dos"
    CHARGE_DENSITY = "charge_density"
    SPIN_DENSITY = "spin_density"
    ELECTROSTATIC_POTENTIAL = "electrostatic_potential"
    SITE_POTENTIAL = "site_potential"
    ORBITALS = "orbitals"
    STM_LDOS = "stm_ldos"
    BOND_ORDER = "bond_order"
    PARTIAL_CHARGES = "partial_charges"
    DIPOLE = "dipole"
    PHONONS = "phonons"
    OPTICAL_SPECTRUM = "optical_spectrum"
    EXCITED_STATES = "excited_states"
    FORMATION_ENERGY = "formation_energy"
    CHARGED_SYSTEM = "charged_system"
    SPIN_POLARIZED = "spin_polarized"


ALL_CAPABILITIES = tuple(c for c in Capability)

CAPABILITY_DOC = {
    Capability.ENERGY: "Total potential energy of the configuration.",
    Capability.FORCES: "Analytic forces on every atom.",
    Capability.STRESS: "Cell stress tensor.",
    Capability.RELAX: "Geometry optimisation to a local minimum.",
    Capability.DYNAMICS: "Time integration of the equations of motion.",
    Capability.BAND_STRUCTURE: "Single-particle eigenvalues along a k-path.",
    Capability.TOTAL_DOS: "Total electronic density of states.",
    Capability.LOCAL_DOS: "Site- and energy-resolved density of states.",
    Capability.CHARGE_DENSITY: "Real-space electron density.",
    Capability.SPIN_DENSITY: "Real-space spin density (requires spin polarisation).",
    Capability.ELECTROSTATIC_POTENTIAL: "Hartree/electrostatic potential on a grid.",
    Capability.SITE_POTENTIAL: "Electrostatic potential and field at each atomic site.",
    Capability.ORBITALS: "Individual single-particle orbital amplitudes.",
    Capability.STM_LDOS: "Energy-integrated LDOS in the vacuum, for Tersoff-Hamann STM.",
    Capability.BOND_ORDER: "Bond orders or overlap populations from the density matrix.",
    Capability.PARTIAL_CHARGES: "Atom-projected charges.",
    Capability.DIPOLE: "Electric dipole moment.",
    Capability.PHONONS: "Vibrational frequencies and modes.",
    Capability.OPTICAL_SPECTRUM: "Optical absorption or dielectric function.",
    Capability.EXCITED_STATES: "Electronically excited states.",
    Capability.FORMATION_ENERGY: "Defect or surface formation energies.",
    Capability.CHARGED_SYSTEM: "Systems with a non-zero net charge.",
    Capability.SPIN_POLARIZED: "Independent spin channels.",
}


@dataclass
class SolverResult:
    """Everything a solver produced in one run."""

    solver: str
    structure: Structure
    results: Dict[str, Result] = field(default_factory=dict)
    convergence: Optional[Convergence] = None
    log: List[str] = field(default_factory=list)
    wall_time_s: float = 0.0

    def __getitem__(self, key: str) -> Result:
        return self.results[key]

    def get(self, key: str) -> Optional[Result]:
        return self.results.get(key)

    def keys(self) -> List[str]:
        return sorted(self.results)

    def as_dict(self, include_values: bool = True) -> dict:
        return {
            "solver": self.solver,
            "results": {k: v.as_dict(include_values) for k, v in self.results.items()},
            "convergence": self.convergence.as_dict() if self.convergence else None,
            "log": list(self.log),
            "wall_time_s": self.wall_time_s,
        }

    def summary(self) -> str:
        lines = [f"{self.solver}: {len(self.results)} result(s), {self.wall_time_s*1000:.1f} ms"]
        for k in self.keys():
            lines.append("  " + str(self.results[k]))
        return "\n".join(lines)


class Solver:
    """Base class for all solvers."""

    name = "abstract"
    fidelity = Fidelity.NON_PHYSICAL
    capabilities: Set[Capability] = set()
    description = ""

    def supports(self, structure: Structure) -> "SupportReport":
        return SupportReport(True, [], [])

    def can(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def require(self, capability: Capability, context: str = "") -> Optional[Result]:
        """Return an unsupported Result if the capability is missing, else None."""
        if self.can(capability):
            return None
        from .registry import solvers_providing
        providers = [s for s in solvers_providing(capability) if s != self.name]
        return unsupported(
            capability.value,
            self.name,
            (f"{self.name} does not compute {capability.value}"
             f"{': ' + context if context else '.'} "
             f"{CAPABILITY_DOC.get(capability, '')}").strip(),
            suggested_models=providers,
            fidelity=self.fidelity,
        )

    def run(self, structure: Structure, **kwargs) -> SolverResult:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "name": self.name,
            "fidelity": self.fidelity.value,
            "description": self.description,
            "capabilities": sorted(c.value for c in self.capabilities),
            "missing_capabilities": sorted(
                c.value for c in ALL_CAPABILITIES if c not in self.capabilities
            ),
        }


@dataclass
class SupportReport:
    """Whether a solver can run on a given structure, and why not."""

    ok: bool
    blocking: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "blocking": list(self.blocking), "warnings": list(self.warnings)}


def check_electron_bookkeeping(structure: Structure, allow_charged: bool) -> SupportReport:
    """Validate charge and spin bookkeeping before an electronic-structure run."""
    blocking: List[str] = []
    warnings: List[str] = []
    q = structure.total_charge()
    if abs(q) > 1e-9 and not allow_charged:
        blocking.append(
            f"Net charge {q:+g} e: this solver treats only charge-neutral systems. "
            "A charged periodic cell also needs a compensating background and a "
            "finite-size correction."
        )
    n_e = structure.total_electrons()
    if abs(n_e - round(n_e)) > 1e-6:
        blocking.append(
            f"Total electron count {n_e:g} is not an integer; check per-atom formal charges."
        )
    mag = structure.total_magnetic_moment()
    if abs(mag) > 1e-9:
        n_unpaired = abs(mag)
        if abs(n_unpaired - round(n_unpaired)) > 1e-6:
            warnings.append(
                f"Total moment {mag:g} mu_B does not correspond to an integer number of "
                "unpaired electrons in the spin-only approximation."
            )
        if (round(n_e) % 2 == 0) != (round(n_unpaired) % 2 == 0):
            warnings.append(
                f"Spin multiplicity implied by {mag:g} mu_B is inconsistent with "
                f"{round(n_e)} electrons (parity mismatch)."
            )
    return SupportReport(not blocking, blocking, warnings)
