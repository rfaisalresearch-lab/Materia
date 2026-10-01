"""Adapters for external open-source scientific solvers.

Materia does not bundle any of these packages.  Each adapter detects whether
its package is importable and, if it is, exposes it as a registered solver.
When the package is missing the adapter still registers, but it reports itself
as unavailable and tells the user exactly how to install it -- it never falls
back to a different model silently.

Supported targets
-----------------
=================== ================================================
adapter             what it provides when installed
=================== ================================================
``external:ase``    structure interchange, and any ASE calculator
``external:gpaw``   DFT: energies, forces, densities, LDOS, STM
``external:quantum-espresso``  plane-wave DFT through ASE
``external:lammps`` classical MD with EAM potentials, driven out of process
``external:gromacs`` biomolecular molecular dynamics
``external:orca`` molecular quantum chemistry and spectroscopy
``external:ams`` licensed multiscale chemistry and materials engines
``external:cp2k``   DFT and DFTB for large cells
``external:pyscf``  molecular quantum chemistry (HF, DFT, post-HF)
``external:psi4``   molecular quantum chemistry
``external:wannier90`` maximally localised Wannier functions
``external:openmx`` localised-basis DFT
=================== ================================================

Licensing
---------
These packages carry their own licences (GPL for Quantum ESPRESSO and
Wannier90, LGPL for ASE and GPAW, and so on).  Materia communicates with
them through their public Python interfaces or input/output files and does not
redistribute, link against or embed any of their code, so their licences apply
to your installation of them, not to Materia.  The adapters are written so
that removing a package removes exactly its capabilities and nothing else.
"""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from ..core_model.structure import Structure
from ..provenance import Fidelity, Origin, Provenance, Result, unsupported
from .base import Capability, Solver, SolverResult, SupportReport
from .registry import register


@dataclass
class AdapterSpec:
    """Declares an external solver and how to detect it."""

    name: str
    package: str
    executable: Optional[str]
    fidelity: Fidelity
    capabilities: Set[Capability]
    install_hint: str
    description: str
    license_note: str
    reference: str = ""

    def available(self) -> bool:
        if self.name == "external:lammps":
            from .lammps import discover

            return discover().available
        if self.package:
            try:
                importlib.import_module(self.package)
            except Exception:
                return False
        if self.executable and not shutil.which(self.executable):
            return False
        return True


ADAPTERS: List[AdapterSpec] = [
    AdapterSpec(
        name="external:ase", package="ase", executable=None,
        fidelity=Fidelity.NON_PHYSICAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.RELAX,
                      Capability.DYNAMICS},
        install_hint="pip install ase",
        description="Atomic Simulation Environment: structure interchange plus any "
                    "attached ASE calculator. ASE is an interface and computes "
                    "nothing itself, so the fidelity of a result comes from the "
                    "calculator that was attached, not from ASE.",
        license_note="ASE is LGPL-2.1-or-later. Materia imports its public API only.",
        reference="A. Hjorth Larsen et al., J. Phys.: Condens. Matter 29 (2017) 273002",
    ),
    AdapterSpec(
        name="external:quantum-espresso", package="ase", executable="pw.x",
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.STRESS,
                      Capability.RELAX, Capability.BAND_STRUCTURE,
                      Capability.TOTAL_DOS, Capability.LOCAL_DOS,
                      Capability.CHARGE_DENSITY, Capability.SPIN_DENSITY,
                      Capability.PHONONS, Capability.SPIN_POLARIZED,
                      Capability.CHARGED_SYSTEM},
        install_hint="Install Quantum ESPRESSO so that pw.x is on PATH "
                     "(conda install -c conda-forge qe), plus ASE for the interface.",
        description="Plane-wave pseudopotential DFT, driven through ASE's Espresso "
                    "calculator.",
        license_note="Quantum ESPRESSO is GPL-2.0-or-later. Materia writes its input "
                     "files and reads its output; no QE code is linked or redistributed.",
        reference="P. Giannozzi et al., J. Phys.: Condens. Matter 21 (2009) 395502",
    ),
    AdapterSpec(
        name="external:lammps", package="", executable="lmp",
        fidelity=Fidelity.TIER1_CLASSICAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.STRESS,
                      Capability.RELAX, Capability.DYNAMICS},
        install_hint="conda install -c conda-forge lammps, or build LAMMPS with the "
                     "MANYBODY package; then put lmp on PATH or set "
                     "MATERIA_LAMMPS_EXECUTABLE or MATERIA_LAMMPS_PYTHON.",
        description="Classical molecular dynamics with many-body potentials, run out of "
                    "process from a frozen specification: energy, forces, stress, "
                    "fixed-cell relaxation and NVE, Langevin or Nose-Hoover dynamics with "
                    "EAM potentials.",
        license_note="LAMMPS is GPL-2.0-only. Materia writes its input files and reads "
                     "its output; no LAMMPS code is linked or redistributed.",
        reference="A. P. Thompson et al., Comp. Phys. Comm. 271 (2022) 108171",
    ),
    AdapterSpec(
        name="external:gromacs", package="", executable="gmx",
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.RELAX,
                      Capability.DYNAMICS},
        install_hint="Install GROMACS so that gmx is on PATH, then provide a complete "
                     "topology and force-field assignment.",
        description="High-throughput molecular dynamics for biomolecules, solvents, "
                    "membranes and free-energy workflows.",
        license_note="GROMACS is LGPL-2.1-or-later. Materia would drive its command-line "
                     "tools without redistributing them.",
        reference="M. J. Abraham et al., SoftwareX 1-2 (2015) 19-25",
    ),
    AdapterSpec(
        name="external:orca", package="", executable="orca",
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.RELAX,
                      Capability.DYNAMICS, Capability.ORBITALS,
                      Capability.CHARGE_DENSITY, Capability.SPIN_DENSITY,
                      Capability.PARTIAL_CHARGES, Capability.DIPOLE,
                      Capability.PHONONS, Capability.OPTICAL_SPECTRUM,
                      Capability.EXCITED_STATES, Capability.SPIN_POLARIZED,
                      Capability.CHARGED_SYSTEM, Capability.BOND_ORDER},
        install_hint="Install ORCA under its licence and put its orca executable on PATH.",
        description="Molecular quantum chemistry from HF and DFT through correlated "
                    "wavefunction methods, excited states and spectroscopy.",
        license_note="ORCA uses its own licence. Materia does not redistribute ORCA.",
        reference="F. Neese, WIREs Comput. Mol. Sci. 12 (2022) e1606",
    ),
    AdapterSpec(
        name="external:ams", package="", executable="ams",
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.STRESS,
                      Capability.RELAX, Capability.DYNAMICS,
                      Capability.BAND_STRUCTURE, Capability.TOTAL_DOS,
                      Capability.CHARGE_DENSITY, Capability.PHONONS,
                      Capability.SPIN_POLARIZED, Capability.CHARGED_SYSTEM},
        install_hint="Install Amsterdam Modeling Suite under a valid licence and make "
                     "the ams executable available in the environment.",
        description="Licensed multiscale engines including molecular and periodic DFT, "
                    "DFTB, force fields, reactive dynamics and workflow tools.",
        license_note="Amsterdam Modeling Suite is commercial software. Materia does not "
                     "bundle it or grant a licence.",
        reference="SCM, Amsterdam Modeling Suite documentation",
    ),
    AdapterSpec(
        name="external:cp2k", package="ase", executable="cp2k.psmp",
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.RELAX,
                      Capability.DYNAMICS, Capability.CHARGE_DENSITY,
                      Capability.TOTAL_DOS, Capability.SPIN_POLARIZED,
                      Capability.CHARGED_SYSTEM},
        install_hint="Install CP2K so cp2k.psmp (or cp2k.popt) is on PATH.",
        description="Gaussian and plane-wave DFT and self-consistent-charge DFTB for "
                    "large condensed-phase cells.",
        license_note="CP2K is GPL-2.0-or-later; driven through files via ASE.",
        reference="T. D. Kuehne et al., J. Chem. Phys. 152 (2020) 194103",
    ),
    AdapterSpec(
        name="external:pyscf", package="pyscf", executable=None,
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.ORBITALS,
                      Capability.CHARGE_DENSITY, Capability.SPIN_DENSITY,
                      Capability.PARTIAL_CHARGES, Capability.DIPOLE,
                      Capability.EXCITED_STATES, Capability.SPIN_POLARIZED,
                      Capability.CHARGED_SYSTEM, Capability.BOND_ORDER},
        install_hint="pip install pyscf",
        description="Molecular quantum chemistry: Hartree-Fock, DFT, MP2, CCSD, TDDFT. "
                    "Use it for finite clusters, not for periodic slabs.",
        license_note="PySCF is Apache-2.0.",
        reference="Q. Sun et al., J. Chem. Phys. 153 (2020) 024109",
    ),
    AdapterSpec(
        name="external:psi4", package="psi4", executable=None,
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.ORBITALS,
                      Capability.EXCITED_STATES, Capability.DIPOLE,
                      Capability.CHARGED_SYSTEM, Capability.SPIN_POLARIZED},
        install_hint="conda install -c conda-forge psi4",
        description="Molecular quantum chemistry with high-accuracy correlated methods.",
        license_note="Psi4 is LGPL-3.0.",
        reference="D. G. A. Smith et al., J. Chem. Phys. 152 (2020) 184108",
    ),
    AdapterSpec(
        name="external:wannier90", package="", executable="wannier90.x",
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ORBITALS, Capability.BAND_STRUCTURE},
        install_hint="Install Wannier90 so wannier90.x is on PATH; run it after a "
                     "Quantum ESPRESSO or GPAW calculation.",
        description="Maximally localised Wannier functions and Wannier-interpolated "
                    "band structures.",
        license_note="Wannier90 is GPL-2.0-or-later; driven through its input files.",
        reference="G. Pizzi et al., J. Phys.: Condens. Matter 32 (2020) 165902",
    ),
    AdapterSpec(
        name="external:openmx", package="", executable="openmx",
        fidelity=Fidelity.TIER3_EXTERNAL,
        capabilities={Capability.ENERGY, Capability.FORCES, Capability.BAND_STRUCTURE,
                      Capability.TOTAL_DOS, Capability.LOCAL_DOS,
                      Capability.CHARGE_DENSITY, Capability.SPIN_POLARIZED},
        install_hint="Install OpenMX so the openmx binary is on PATH.",
        description="Localised-basis DFT, efficient for large slabs and interfaces.",
        license_note="OpenMX is GPL-3.0; driven through its input files.",
        reference="T. Ozaki, Phys. Rev. B 67 (2003) 155108",
    ),
]


class ExternalSolver(Solver):
    """A solver backed by an external package."""

    def __init__(self, spec: AdapterSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.fidelity = spec.fidelity
        self.capabilities = set(spec.capabilities)
        self.description = spec.description

    @property
    def installed(self) -> bool:
        return self.spec.available()

    def supports(self, structure: Structure) -> SupportReport:
        if not self.installed:
            return SupportReport(False, [
                f"{self.spec.name} is not installed. {self.spec.install_hint}"
            ])
        return SupportReport(True, [], [
            "External solver results carry the external package's own "
            "approximations; Materia records the settings it passed but cannot "
            "vouch for the choice of functional, basis or pseudopotential."
        ])

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "available": self.installed,
            "package": self.spec.package,
            "executable": self.spec.executable,
            "install_hint": self.spec.install_hint,
            "license_note": self.spec.license_note,
            "reference": self.spec.reference,
        })
        return d

    def run(self, structure: Structure, task: str = "energy", **kwargs) -> SolverResult:
        out = SolverResult(solver=self.name, structure=structure)
        if not self.installed:
            out.results[task] = unsupported(
                task, self.name,
                f"{self.spec.name} is not installed on this machine, so the "
                f"requested calculation cannot be run. {self.spec.install_hint}",
                suggested_models=[a.name for a in ADAPTERS
                                  if a.available() and a.name != self.name],
                fidelity=self.fidelity,
            )
            out.results[task].extra["preserved_request"] = {
                "task": task, "kwargs": {k: str(v) for k, v in kwargs.items()},
                "note": "The request is stored with the project and can be re-run "
                        "unchanged once the solver is installed.",
            }
            out.log.append(out.results[task].unsupported_reason)
            return out
        if self.spec.name == "external:ase":
            return self._run_ase(structure, task, **kwargs)
        if self.spec.name == "external:lammps":
            out.results[task] = unsupported(
                task, self.name,
                "LAMMPS is driven through lammps.run in the Python API or the LAMMPS "
                "backend of the classical panel, which freeze, audit and validate the run. "
                "This generic entry point does not run it and does not substitute another "
                "model.", fidelity=self.fidelity)
            return out
        out.results[task] = unsupported(
            task, self.name,
            f"{self.spec.name} is installed, but Materia 0.1 only drives it through "
            "an explicitly supplied calculator object. Pass calculator=<object> to "
            "run(), or use the ASE adapter.",
            suggested_models=["external:ase"], fidelity=self.fidelity)
        return out

    def _run_ase(self, structure: Structure, task: str, calculator=None,
                 **kwargs) -> SolverResult:
        out = SolverResult(solver=self.name, structure=structure)
        if calculator is None:
            out.results[task] = unsupported(
                task, self.name,
                "The ASE adapter needs a calculator: pass calculator=<an ASE "
                "calculator instance>, for example GPAW(...) or EMT().",
                suggested_models=["external:gpaw", "external:quantum-espresso"],
                fidelity=self.fidelity)
            return out
        atoms = to_ase(structure)
        atoms.calc = calculator
        tier, tier_note = calculator_fidelity(calculator)
        prov = Provenance(
            model=f"{self.name}:{type(calculator).__name__}",
            fidelity=tier,
            origin=Origin.CALCULATED,
            approximations=[
                "Computed by an external package. Its approximations (functional, "
                "basis set, pseudopotentials, k-point sampling, smearing) are those "
                "of the calculator object you supplied.",
                tier_note,
            ],
            boundary_conditions=str(structure.cell.pbc),
            parameters={"calculator": type(calculator).__name__,
                        "calculator_parameters": _safe_params(calculator)},
            references=[self.spec.reference] if self.spec.reference else [],
        )
        if task == "energy":
            energy = float(atoms.get_potential_energy())
            out.results["energy"] = Result("energy", energy, "eV", prov)
            try:
                forces = atoms.get_forces()
                out.results["forces"] = Result("forces", forces, "eV/A", prov)
            except Exception as exc:
                out.log.append(f"Forces unavailable from this calculator: {exc}")
        elif task == "relax":
            from ase.optimize import FIRE
            fmax = kwargs.get("fmax", 0.02)
            steps = kwargs.get("steps", 200)
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
            structure.positions = atoms.get_positions()
            out.results["energy"] = Result("energy", float(atoms.get_potential_energy()),
                                           "eV", prov)
            out.results["relaxed_structure"] = Result("relaxed_structure", structure,
                                                      "", prov)
        else:
            out.results[task] = unsupported(
                task, self.name,
                f"The ASE adapter implements 'energy' and 'relax'; {task!r} is not "
                "mapped. Drive the calculator directly from a script for other tasks.",
                fidelity=self.fidelity)
        return out


CALCULATOR_FIDELITY: Dict[str, Fidelity] = {
    "GPAW": Fidelity.TIER3_EXTERNAL,
    "Espresso": Fidelity.TIER3_EXTERNAL,
    "Abinit": Fidelity.TIER3_EXTERNAL,
    "Octopus": Fidelity.TIER3_EXTERNAL,
    "CP2K": Fidelity.TIER3_EXTERNAL,
    "NWChem": Fidelity.TIER3_EXTERNAL,
    "Psi4": Fidelity.TIER3_EXTERNAL,
    "Aims": Fidelity.TIER3_EXTERNAL,
    "Vasp": Fidelity.TIER3_EXTERNAL,
    "Siesta": Fidelity.TIER3_EXTERNAL,
    "Gaussian": Fidelity.TIER3_EXTERNAL,
    "DFTB": Fidelity.TIER2_SEMI_EMPIRICAL,
    "Hotbit": Fidelity.TIER2_SEMI_EMPIRICAL,
    "EMT": Fidelity.TIER1_CLASSICAL,
    "LennardJones": Fidelity.TIER1_CLASSICAL,
    "MorsePotential": Fidelity.TIER1_CLASSICAL,
    "EAM": Fidelity.TIER1_CLASSICAL,
    "LAMMPS": Fidelity.TIER1_CLASSICAL,
    "LAMMPSlib": Fidelity.TIER1_CLASSICAL,
    "TIP3P": Fidelity.TIER1_CLASSICAL,
    "TIP4P": Fidelity.TIER1_CLASSICAL,
}


def calculator_fidelity(calculator) -> Tuple[Fidelity, str]:
    """The tier a result deserves, and why, given the attached calculator."""
    name = type(calculator).__name__
    if name in CALCULATOR_FIDELITY:
        tier = CALCULATOR_FIDELITY[name]
        return tier, (f"Fidelity taken from the attached {name} calculator, not from "
                      "ASE: ASE is an interface and computes nothing itself.")
    return Fidelity.NON_PHYSICAL, (
        f"Materia does not recognise the ASE calculator {name!r} and will not guess "
        "what kind of model it is, so this result carries no fidelity tier. The "
        "number is whatever that calculator computed; classify it yourself before "
        "relying on it.")


def _safe_params(calculator) -> dict:
    try:
        params = getattr(calculator, "parameters", {}) or {}
        return {str(k): str(v)[:200] for k, v in dict(params).items()}
    except Exception:
        return {}


def to_ase(structure: Structure):
    """Convert to an ``ase.Atoms``.  Requires ASE."""
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "Converting to ASE needs the optional 'ase' package: pip install ase"
        ) from exc
    atoms = Atoms(
        numbers=[int(z) for z in structure.numbers],
        positions=structure.positions,
        cell=structure.cell.matrix if structure.cell.is_periodic else None,
        pbc=structure.cell.pbc,
    )
    atoms.set_initial_charges(structure.formal_charges)
    atoms.set_initial_magnetic_moments(structure.magnetic_moments)
    atoms.info["materia_ids"] = [int(i) for i in structure.ids]
    return atoms


def from_ase(atoms) -> Structure:
    """Convert an ``ase.Atoms`` to an Materia structure."""
    import numpy as np

    from ..core_model.cell import Cell
    cell = (Cell(np.array(atoms.cell.array), tuple(bool(p) for p in atoms.pbc))
            if atoms.cell.rank > 0 else Cell.none())
    s = Structure([int(z) for z in atoms.numbers], np.array(atoms.positions), cell)
    try:
        s.formal_charges[:] = atoms.get_initial_charges()
        s.magnetic_moments[:] = atoms.get_initial_magnetic_moments()
    except Exception:
        pass
    s.info["provenance"] = Provenance(
        model="external:ase/import", fidelity=Fidelity.TIER0_STRUCTURAL,
        origin=Origin.IMPORTED,
        notes="Converted from an ase.Atoms object.").as_dict()
    return s


def register_adapters() -> None:
    for spec in ADAPTERS:
        register(spec.name, (lambda s=spec: ExternalSolver(s)), overwrite=True,
                 meta={"tier": spec.fidelity.value, "external": True,
                       "install_hint": spec.install_hint,
                       "license_note": spec.license_note})


def availability_report() -> List[dict]:
    """What is installed right now, for the interface's solver panel.

    The GPAW driver reports itself, because "installed" means something more
    specific for it than an importable package: it needs an interpreter, the
    code and the PAW datasets, and it says which of those is missing.
    """
    out = [{
        "name": s.name, "available": s.available(), "package": s.package,
        "executable": s.executable, "install_hint": s.install_hint,
        "description": s.description, "license_note": s.license_note,
        "capabilities": sorted(c.value for c in s.capabilities),
        "reference": s.reference, "driven": s.name == "external:ase",
    } for s in ADAPTERS if s.name != "external:lammps"]

    from .lammps import discover

    lammps_spec = next(s for s in ADAPTERS if s.name == "external:lammps")
    lammps_environment = discover()
    out.insert(0, {
        "name": lammps_spec.name, "available": lammps_environment.available,
        "package": "lammps" if lammps_environment.kind == "python-module" else "",
        "executable": lammps_environment.path or lammps_spec.executable,
        "install_hint": lammps_environment.install_hint(),
        "description": lammps_spec.description, "license_note": lammps_spec.license_note,
        "capabilities": sorted(c.value for c in lammps_spec.capabilities),
        "reference": lammps_spec.reference, "driven": True,
        "version": lammps_environment.version,
        "environment": lammps_environment.as_dict(),
    })

    from .gpaw_driver import GPAWSolver

    gpaw = GPAWSolver().describe()
    environment = gpaw["environment"]
    out.insert(0, {
        "name": gpaw["name"], "available": gpaw["available"], "package": "gpaw",
        "executable": environment.get("interpreter"),
        "install_hint": environment.get("install_hint", ""),
        "description": gpaw["description"], "license_note": gpaw["license_note"],
        "capabilities": gpaw["capabilities"], "reference": gpaw["reference"],
        "driven": True, "environment": environment,
    })
    return out
