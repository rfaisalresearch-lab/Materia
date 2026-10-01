"""Biomolecular force fields through OpenMM.

OpenMM (MIT and LGPL licences) is driven through its Python API.  A protein,
nucleic acid or other biomolecule is read from a PDB file, optionally repaired
with PDBFixer (missing heavy atoms and hydrogens at a stated pH), and
parameterised with an OpenMM force-field file set such as Amber14 or
CHARMM36, in vacuum, in a generalised-Born implicit solvent, or periodic
with particle-mesh Ewald.

Two ways of using it:

``OpenMMPotential``
    A Materia potential: energy and forces for any positions of the same
    atoms, so Materia's relaxation, dynamics, vibrations and NEB apply.  No
    constraints are used, so forces are exact derivatives of the energy.
``simulate``
    OpenMM's own minimiser and Langevin dynamics with hydrogen-bond
    constraints, for production-length biomolecular runs that Materia's
    Python integrators would be too slow for.  The result says that OpenMM
    integrated it.

Units: OpenMM works in nm, ps and kJ/mol; values are converted to A, fs and
eV with exact factors.

References
----------
P. Eastman et al., PLoS Comput. Biol. 13 (2017) e1005659 (OpenMM 7).
J. A. Maier et al., J. Chem. Theory Comput. 11 (2015) 3696 (ff14SB).
"""

from __future__ import annotations

import importlib
import io
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..provenance import Fidelity, Origin, Provenance, Result
from .potentials import Potential, PotentialError, UnsupportedSystem

KJ_MOL_EV = 1000.0 / 6.02214076e23 / 1.602176634e-19
NM_A = 10.0
INFO_KEY = "biomolecule"
FORCE_FIELDS = {
    "amber14": ("amber14-all.xml", "amber14/tip3pfb.xml"),
    "amber14-gbn2": ("amber14-all.xml", "implicit/gbn2.xml"),
    "charmm36": ("charmm36.xml", "charmm36/water.xml"),
    "amber99sbildn": ("amber99sbildn.xml", "tip3p.xml"),
}
REFERENCES = ["P. Eastman et al., PLoS Comput. Biol. 13 (2017) e1005659"]
FF_REFERENCES = {
    "amber14": "J. A. Maier et al., J. Chem. Theory Comput. 11 (2015) 3696",
    "amber14-gbn2": "H. Nguyen, D. R. Roe and C. Simmerling, J. Chem. Theory Comput. 9 "
                    "(2013) 2020 (GBn2)",
    "charmm36": "J. Huang and A. D. MacKerell, J. Comput. Chem. 34 (2013) 2135",
    "amber99sbildn": "K. Lindorff-Larsen et al., Proteins 78 (2010) 1950",
}


class OpenMMMissing(UnsupportedSystem):
    pass


def _openmm():
    try:
        return (importlib.import_module("openmm"), importlib.import_module("openmm.app"),
                importlib.import_module("openmm.unit"))
    except ImportError:
        raise OpenMMMissing("OpenMM is not installed. Install it with: pip install openmm") \
            from None


def load_pdb(path: str, fix: bool = False, ph: float = 7.0) -> Structure:
    """A biomolecule from a PDB file, with its topology kept for force-field assignment.

    With ``fix`` PDBFixer adds missing heavy atoms and hydrogens at ``ph``; the
    record says so.  Without it the file must already be complete.
    """
    _, app, unit = _openmm()
    if fix:
        try:
            pdbfixer = importlib.import_module("pdbfixer")
        except ImportError:
            raise OpenMMMissing("PDBFixer is not installed: pip install pdbfixer") from None
        fixer = pdbfixer.PDBFixer(filename=str(path))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)
        topology, positions = fixer.topology, fixer.positions
    else:
        pdb = app.PDBFile(str(path))
        topology, positions = pdb.topology, pdb.positions
    buffer = io.StringIO()
    app.PDBFile.writeFile(topology, positions, buffer, keepIds=True)
    xyz = np.asarray(positions.value_in_unit(unit.angstrom), dtype=float)
    numbers = [atom.element.atomic_number for atom in topology.atoms()]
    box = topology.getPeriodicBoxVectors()
    span = xyz.max(axis=0) - xyz.min(axis=0)
    box_note = ""
    if box is not None:
        matrix = np.asarray(box.value_in_unit(unit.angstrom))
        if np.any(np.linalg.norm(matrix, axis=1) <= span):
            box_note = ("The file's CRYST1 box is smaller than the molecule (a placeholder, "
                        "as in NMR entries) and was ignored; the system is isolated.")
            box = None
            topology.setPeriodicBoxVectors(None)
    if box is not None:
        cell = Cell(matrix, (True, True, True))
    else:
        span = xyz.max(axis=0) - xyz.min(axis=0)
        size = float(span.max() + 20.0)
        xyz = xyz - xyz.mean(axis=0) + size / 2
        cell = Cell(np.eye(3) * size, (False, False, False))
        buffer = io.StringIO()
        app.PDBFile.writeFile(topology, xyz * 0.1 * unit.nanometer, buffer, keepIds=True)
    structure = Structure(numbers, xyz, cell)
    residues = list(topology.residues())
    structure.info[INFO_KEY] = {
        "source": str(path), "pdb": buffer.getvalue(), "repaired": bool(fix),
        "ph": float(ph) if fix else None, "residues": len(residues),
        "chains": [c.id for c in topology.chains()],
        "sequence": " ".join(r.name for r in residues),
        "box_note": box_note,
    }
    return structure


def _topology(structure: Structure):
    _, app, _ = _openmm()
    record = structure.info.get(INFO_KEY) or {}
    if not record.get("pdb"):
        raise UnsupportedSystem("This structure carries no biomolecular topology; load it "
                                "with materia.physics.biomolecular.load_pdb.")
    pdb = app.PDBFile(io.StringIO(record["pdb"]))
    numbers = [a.element.atomic_number for a in pdb.topology.atoms()]
    if numbers != [int(z) for z in structure.numbers]:
        raise UnsupportedSystem("The stored biomolecular topology no longer matches the "
                                "atoms of this structure.")
    return pdb.topology


class OpenMMPotential(Potential):
    """Energy and forces of an OpenMM force field, without constraints."""

    fidelity = Fidelity.TIER1_CLASSICAL

    def __init__(self, force_field: str = "amber14-gbn2", cutoff_nm: float = 1.0,
                 platform: str = "Reference") -> None:
        if force_field not in FORCE_FIELDS:
            raise ValueError(f"force_field must be one of {sorted(FORCE_FIELDS)}.")
        self.force_field = force_field
        self.cutoff_nm = float(cutoff_nm)
        self.platform = platform
        self.name = f"openmm/{force_field}"
        self.cutoff_A = 0.0
        self._contexts: Dict[str, object] = {}

    def model_label(self) -> str:
        return self.name

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        try:
            _openmm()
            _topology(structure)
        except UnsupportedSystem as exc:
            return False, str(exc)
        if any(structure.cell.pbc) and not all(structure.cell.pbc):
            return False, "OpenMM periodic systems must be periodic in all three directions."
        if self.force_field == "amber14-gbn2" and any(structure.cell.pbc):
            return False, ("The GBn2 implicit solvent is for non-periodic systems; use "
                           "amber14 with explicit water for a periodic box.")
        return True, ""

    def system(self, structure: Structure, constraints=None):
        mm, app, unit = _openmm()
        topology = _topology(structure)
        field = app.ForceField(*FORCE_FIELDS[self.force_field])
        periodic = all(structure.cell.pbc)
        if periodic:
            topology.setPeriodicBoxVectors(np.asarray(structure.cell.matrix) * 0.1)
            method = app.PME
        else:
            method = app.NoCutoff
        kwargs = {"nonbondedMethod": method, "constraints": constraints,
                  "rigidWater": constraints is not None}
        if periodic:
            kwargs["nonbondedCutoff"] = self.cutoff_nm * unit.nanometer
        try:
            return topology, field.createSystem(topology, **kwargs)
        except ValueError as exc:
            raise UnsupportedSystem(f"{self.force_field} cannot parameterise this structure: "
                                    f"{exc}") from None

    def _context(self, structure: Structure):
        key = (structure.info.get(INFO_KEY) or {}).get("pdb", "") + str(structure.cell.pbc) \
            + str(np.round(structure.cell.matrix, 8).tolist())
        if key not in self._contexts:
            mm, _, unit = _openmm()
            _, system = self.system(structure)
            integrator = mm.VerletIntegrator(0.001)
            platform = mm.Platform.getPlatformByName(self.platform)
            self._contexts = {key: mm.Context(system, integrator, platform)}
        return self._contexts[key]

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        mm, _, unit = _openmm()
        context = self._context(structure)
        context.setPositions(np.asarray(structure.positions) * 0.1)
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * KJ_MOL_EV
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)) * KJ_MOL_EV / NM_A
        if not np.isfinite(energy):
            raise PotentialError("OpenMM returned a non-finite energy.")
        return float(energy), forces

    def describe(self) -> dict:
        mm, _, _ = _openmm()
        files = FORCE_FIELDS[self.force_field]
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": 0.0,
                "parameters": {"force_field_files": list(files), "engine": "OpenMM",
                               "engine_version": mm.__version__, "platform": self.platform,
                               "periodic_cutoff_nm": self.cutoff_nm},
                "approximations": [
                    f"Fixed-charge biomolecular force field {self.force_field} "
                    f"({', '.join(files)}) as implemented in OpenMM.",
                    "No constraints in energy and force evaluation; non-periodic systems use "
                    "no cutoff, periodic ones particle-mesh Ewald.",
                    "Fixed bonded topology: no bond breaking or proton transfer.",
                ],
                "references": REFERENCES + [FF_REFERENCES[self.force_field]]}


def simulate(structure: Structure, force_field: str = "amber14-gbn2", steps: int = 1000,
             temperature_K: float = 300.0, timestep_fs: float = 2.0,
             friction_per_ps: float = 1.0, minimise: bool = True, seed: int = 0,
             report_every: int = 100, platform: str = "CPU",
             pressure_bar: Optional[float] = None) -> Dict[str, Result]:
    """Minimise and run Langevin dynamics natively in OpenMM, with H-bond constraints.

    Returns energy and trajectory results whose provenance names OpenMM as
    the integrator.  The input structure is not changed.
    """
    mm, app, unit = _openmm()
    potential = OpenMMPotential(force_field, platform=platform)
    ok, why = potential.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    if steps < 0 or timestep_fs <= 0 or timestep_fs > 4 or temperature_K <= 0:
        raise ValueError("Need steps >= 0, 0 < timestep_fs <= 4 and temperature_K > 0.")
    _, system = potential.system(structure, constraints=app.HBonds)
    if pressure_bar is not None:
        if not all(structure.cell.pbc):
            raise UnsupportedSystem("A barostat needs a periodic box; solvate the system first.")
        system.addForce(mm.MonteCarloBarostat(pressure_bar * unit.bar,
                                              temperature_K * unit.kelvin, 25))
    integrator = mm.LangevinMiddleIntegrator(temperature_K * unit.kelvin,
                                             friction_per_ps / unit.picosecond,
                                             timestep_fs * unit.femtosecond)
    integrator.setRandomNumberSeed(int(seed))
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform))
    context.setPositions(np.asarray(structure.positions) * 0.1)
    initial = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole) * KJ_MOL_EV
    if minimise:
        mm.LocalEnergyMinimizer.minimize(context, 10.0, 0)
    minimised = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole) * KJ_MOL_EV
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin, int(seed))
    frames: List[np.ndarray] = []
    record = {"step": [], "time_fs": [], "potential_eV": [], "kinetic_eV": [],
              "temperature_K": [], "density_g_cm3": []}
    total_mass = float(sum(system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(system.getNumParticles())))
    dof = system.getNumParticles() * 3 - system.getNumConstraints() - 3
    done = 0
    while done < steps:
        block = min(report_every, steps - done)
        integrator.step(block)
        done += block
        state = context.getState(getEnergy=True, getPositions=True)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) * KJ_MOL_EV
        record["step"].append(done)
        record["time_fs"].append(done * timestep_fs)
        record["potential_eV"].append(state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole) * KJ_MOL_EV)
        record["kinetic_eV"].append(kinetic)
        record["temperature_K"].append(2 * kinetic / (dof * 8.617333262e-5))
        if all(structure.cell.pbc):
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
            record["density_g_cm3"].append(
                total_mass / abs(float(np.linalg.det(np.asarray(box)))) * 1.66053906660)
        frames.append(np.asarray(state.getPositions(asNumpy=True).value_in_unit(
            unit.angstrom)))
    record["positions_A"] = frames
    description = potential.describe()
    provenance = Provenance(
        model=f"openmm/langevin-middle[{potential.name}]",
        fidelity=Fidelity.TIER1_CLASSICAL, origin=Origin.CALCULATED,
        approximations=description["approximations"][:1] + [
            "Integrated by OpenMM's LangevinMiddleIntegrator with bonds to hydrogen "
            "constrained; classical nuclei. Constrained bonds carry no stretch energy, so "
            "these energies are not comparable with the unconstrained OpenMMPotential.",
            "Local energy minimisation by OpenMM's L-BFGS minimiser before dynamics."
            if minimise else "No minimisation before dynamics."],
        parameters={**description["parameters"], "steps": steps,
                    "temperature_K": temperature_K, "timestep_fs": timestep_fs,
                    "friction_per_ps": friction_per_ps, "constraints": "HBonds",
                    "pressure_bar": pressure_bar,
                    "degrees_of_freedom": dof},
        references=description["references"] + [
            "B. Leimkuhler and C. Matthews, Appl. Math. Res. Express 2013 (2013) 34 "
            "(the BAOAB splitting the LangevinMiddle integrator follows)"],
        seed=int(seed), boundary_conditions="periodic" if all(structure.cell.pbc)
        else "isolated, no cutoff")
    return {
        "initial_energy": Result("potential_energy", initial, "eV", provenance),
        "minimised_energy": Result("potential_energy", minimised, "eV", provenance),
        "trajectory": Result("trajectory", record, "mixed", provenance,
                             extra={"units": {"time_fs": "fs", "potential_eV": "eV",
                                              "kinetic_eV": "eV", "temperature_K": "K",
                                              "positions_A": "A"}}),
    }


def solvate(structure: Structure, force_field: str = "amber14", padding_A: float = 10.0,
            ionic_strength_M: float = 0.15, neutralize: bool = True,
            water_model: str = "tip3p", box_shape: str = "cube") -> Structure:
    """Surround a biomolecule with explicit water and ions in a periodic box (OpenMM Modeller).

    Adds water at least ``padding_A`` from the solute, neutralising counter-ions
    and salt to ``ionic_strength_M``.  The result is periodic and is evaluated
    with particle-mesh Ewald.  Only force fields with an explicit water model are
    accepted.
    """
    _, app, unit = _openmm()
    if force_field not in ("amber14", "charmm36", "amber99sbildn"):
        raise UnsupportedSystem(f"{force_field} has no explicit water model; choose amber14, "
                                "charmm36 or amber99sbildn.")
    if any(structure.cell.pbc):
        raise UnsupportedSystem("The structure is already periodic; solvate an isolated "
                                "solute.")
    if padding_A <= 0 or ionic_strength_M < 0:
        raise ValueError("padding_A must be positive and ionic_strength_M not negative.")
    topology = _topology(structure)
    modeller = app.Modeller(topology, np.asarray(structure.positions) * 0.1 * unit.nanometer)
    field = app.ForceField(*FORCE_FIELDS[force_field])
    modeller.addSolvent(field, model=water_model, padding=padding_A * 0.1 * unit.nanometer,
                        ionicStrength=ionic_strength_M * unit.molar, neutralize=neutralize,
                        boxShape=box_shape)
    box = np.asarray(modeller.topology.getPeriodicBoxVectors().value_in_unit(unit.angstrom))
    xyz = np.asarray(modeller.positions.value_in_unit(unit.angstrom))
    buffer = io.StringIO()
    app.PDBFile.writeFile(modeller.topology, modeller.positions, buffer, keepIds=True)
    numbers = [a.element.atomic_number for a in modeller.topology.atoms()]
    solvated = Structure(numbers, xyz, Cell(box, (True, True, True)))
    residues = [r.name for r in modeller.topology.residues()]
    record = dict(structure.info.get(INFO_KEY) or {})
    record.update({"pdb": buffer.getvalue(), "residues": len(residues),
                   "solvation": {"engine": "OpenMM Modeller", "water_model": water_model,
                                 "padding_A": padding_A, "ionic_strength_M": ionic_strength_M,
                                 "neutralize": neutralize, "box_shape": box_shape,
                                 "waters": residues.count("HOH"),
                                 "sodium": residues.count("NA"),
                                 "chloride": residues.count("CL")},
                   "box_note": ""})
    solvated.info[INFO_KEY] = record
    return solvated
