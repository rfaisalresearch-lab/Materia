"""Plane-wave DFT through Quantum ESPRESSO's pw.x.

Quantum ESPRESSO (GPL-2.0-or-later) runs as a separate program; Materia writes
its input through ASE's Espresso calculator, runs ``pw.x`` in a fresh working
directory, checks that the self-consistent field converged and reads energy,
forces, stress and the Fermi level.  Nothing from Quantum ESPRESSO is linked
or copied into Materia.

Finding pw.x
------------
``MATERIA_QE_PW`` (an explicit path, used exclusively), then ``pw.x`` on
``PATH``, then conda environments, ``materia-qe`` first (``CONDA_SUBDIR=osx-64
conda create -n materia-qe -c conda-forge qe`` on Apple silicon, where
conda-forge has only x86_64 builds).

Pseudopotentials
----------------
Read from ``MATERIA_PSEUDO_DIR`` or ``~/.cache/materia/pseudopotentials``.
:func:`fetch_pseudopotential` downloads a PSlibrary 1.0.0 PBE file
(A. Dal Corso, Comput. Mater. Sci. 95 (2014) 337) from the Quantum ESPRESSO
site into that cache; Materia does not ship pseudopotentials.  Every result
records each file's name and SHA-256, and the cutoffs the file suggests.  A
cutoff below a file's suggestion is refused unless explicitly allowed.

The exchange-correlation functional is the one the pseudopotentials were
generated with; mixing functionals is refused.

Smearing defaults to Gaussian.  Marzari-Vanderbilt and Methfessel-Paxton
occupations are not monotonic, so in a gapped system the Fermi level can have
several solutions; for silicon with 0.01 Ry of Marzari-Vanderbilt smearing
pw.x energies were measured to jump by 4 meV between geometries 0.005 A
apart while the forces stayed smooth.  Use them for metals only.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.structure import Structure
from ..core_model.units import BOHR_A, RYDBERG_EV
from ..elements import periodic_table as pt
from ..provenance import Fidelity
from ..system import conda_environments, environment_program, is_executable
from .potentials import Potential, PotentialError, UnsupportedSystem

PW_ENV = "MATERIA_QE_PW"
PSEUDO_ENV = "MATERIA_PSEUDO_DIR"
PSEUDO_URL = "https://pseudopotentials.quantum-espresso.org/upf_files/"
PSLIBRARY_PBE = {
    "H": "H.pbe-rrkjus_psl.1.0.0.UPF", "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF", "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF", "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-rrkjus_psl.1.0.0.UPF", "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
    "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF", "As": "As.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ge": "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF", "Au": "Au.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ag": "Ag.pbe-n-kjpaw_psl.1.0.0.UPF", "Pt": "Pt.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ni": "Ni.pbe-n-kjpaw_psl.1.0.0.UPF", "W": "W.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF", "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF", "B": "B.pbe-n-kjpaw_psl.1.0.0.UPF",
    "P": "P.pbe-n-kjpaw_psl.1.0.0.UPF", "Se": "Se.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "In": "In.pbe-dn-kjpaw_psl.1.0.0.UPF",
}
REFERENCES = [
    "P. Giannozzi et al., J. Phys.: Condens. Matter 21 (2009) 395502",
    "P. Giannozzi et al., J. Phys.: Condens. Matter 29 (2017) 465901",
    "A. Dal Corso, Comput. Mater. Sci. 95 (2014) 337 (PSlibrary)",
]


class EspressoMissing(UnsupportedSystem):
    pass


class EspressoFailed(PotentialError):
    pass


def find_pw() -> Optional[str]:
    explicit = os.environ.get(PW_ENV, "").strip()
    if explicit:
        return explicit if is_executable(explicit) else None
    found = shutil.which("pw.x")
    if found:
        return found
    for env in conda_environments("materia-qe"):
        candidate = environment_program(env, "pw.x")
        if candidate is not None:
            return str(candidate)
    return None


THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1"}


@lru_cache(maxsize=8)
def pw_version(pw: str) -> str:
    """The pw.x version, asked in a scratch directory so nothing is left behind."""
    with tempfile.TemporaryDirectory(prefix="materia-qe-version-") as work:
        try:
            done = subprocess.run([pw], input="", capture_output=True, text=True, timeout=60,
                                  cwd=work, env={**os.environ, **THREAD_ENV})
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
    match = re.search(r"Program PWSCF (v\.[\w.]+)", done.stdout)
    return match.group(1) if match else "unknown"


def pseudo_dir() -> Path:
    return Path(os.environ.get(PSEUDO_ENV, "") or Path.home() / ".cache" / "materia" /
                "pseudopotentials")


def fetch_pseudopotential(element: str) -> Path:
    """The PSlibrary PBE file for an element, downloaded into the cache if absent."""
    symbol = pt.symbol(element)
    name = PSLIBRARY_PBE.get(symbol)
    if name is None:
        raise UnsupportedSystem(f"No PSlibrary PBE pseudopotential is listed for {symbol}; "
                                f"place a UPF file in {pseudo_dir()} and name it explicitly.")
    target = pseudo_dir() / name
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(PSEUDO_URL + name, timeout=60) as response:
                data = response.read()
        except OSError as exc:
            raise EspressoMissing(f"Could not download {name}: {exc}") from None
        if b"<UPF" not in data[:200] and b"PP_INFO" not in data[:2000]:
            raise EspressoFailed(f"The download of {name} is not a UPF file.")
        temporary = target.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(target)
    return target


def describe_pseudopotential(path: Path) -> dict:
    data = path.read_bytes()
    head = data[:20000].decode("latin-1")
    functional = re.search(r"Functional:\s*([^\n]+)", head) or re.search(
        r'functional="\s*([^"]+?)\s*"', head)
    wfc = re.search(r"Suggested minimum cutoff for wavefunctions:\s*([\d.]+)", head)
    rho = re.search(r"Suggested minimum cutoff for charge density:\s*([\d.]+)", head)
    return {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
            "functional": _functional_name(functional.group(1)) if functional else "unknown",
            "suggested_ecutwfc_Ry": float(wfc.group(1)) if wfc else None,
            "suggested_ecutrho_Ry": float(rho.group(1)) if rho else None}


def _functional_name(text: str) -> str:
    """A canonical functional name from a UPF header, whose spellings vary between files."""
    words = set(re.split(r"[\s(),]+", text.upper()))
    if "PBE" in words or {"PBX", "PBC"} <= words:
        return "PBE"
    if "PBESOL" in words or {"PSX", "PSC"} <= words:
        return "PBESOL"
    if "PZ" in words or "LDA" in words or ({"SLA", "PZ"} <= words):
        return "LDA"
    return " ".join(sorted(w for w in words if w))


def parse_output(text: str) -> dict:
    """Energy, forces, stress, Fermi level and convergence from pw.x output, independently of ASE.

    Rydberg and bohr are converted with CODATA 2018 values, as pw.x itself uses.
    ASE's reader uses CODATA 2006 and differs by about 1e-7 relative.
    """
    energy = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", text, re.M)
    fermi = re.findall(r"the Fermi energy is\s+(-?[\d.]+)\s+ev", text)
    forces = re.findall(r"atom\s+\d+\s+type\s+\d+\s+force =\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
                        r"(-?[\d.]+)", text)
    stress = re.search(r"total\s+stress.*\n(.*)\n(.*)\n(.*)", text)
    iterations = re.findall(r"convergence has been achieved in\s+(\d+) iterations", text)
    magnetisation = re.findall(r"total magnetization\s+=\s+(-?[\d.]+)\s+Bohr mag/cell", text)
    absolute = re.findall(r"absolute magnetization\s+=\s+(-?[\d.]+)\s+Bohr mag/cell", text)
    out = {"converged": bool(iterations),
           "iterations": int(iterations[-1]) if iterations else None,
           "energy_eV": float(energy[-1]) * RYDBERG_EV if energy else None,
           "fermi_eV": float(fermi[-1]) if fermi else None,
           "magnetization_muB": float(magnetisation[-1]) if magnetisation else None,
           "absolute_magnetization_muB": float(absolute[-1]) if absolute else None}
    if forces:
        out["forces_eV_A"] = np.array(forces, dtype=float) * RYDBERG_EV / BOHR_A
    if stress:
        out["stress_kbar"] = np.array([list(map(float, line.split()[3:6]))
                                       for line in stress.groups()])
        sigma = np.array([list(map(float, line.split()[:3])) for line in stress.groups()])
        out["stress_eV_A3"] = -sigma * RYDBERG_EV / BOHR_A ** 3
    return out


class EspressoPotential(Potential):
    """Kohn-Sham DFT energies, forces and stress from Quantum ESPRESSO pw.x."""

    fidelity = Fidelity.TIER3_EXTERNAL

    def __init__(self, ecutwfc_Ry: float = 45.0, ecutrho_Ry: Optional[float] = None,
                 kpts: Sequence[int] = (4, 4, 4), smearing: str = "gaussian",
                 degauss_Ry: float = 0.01, conv_thr_Ry: float = 1e-10,
                 pseudopotentials: Optional[Dict[str, str]] = None, spin_polarized: bool = False,
                 starting_magnetization: Optional[Dict[str, float]] = None,
                 allow_low_cutoff: bool = False, timeout_s: float = 3600.0,
                 symmetry: bool = True) -> None:
        self.symmetry = bool(symmetry)
        self.ecutwfc_Ry = float(ecutwfc_Ry)
        self.ecutrho_Ry = float(ecutrho_Ry) if ecutrho_Ry else 8.0 * self.ecutwfc_Ry
        self.kpts = tuple(int(k) for k in kpts)
        self.smearing = smearing
        self.degauss_Ry = float(degauss_Ry)
        self.conv_thr_Ry = float(conv_thr_Ry)
        self.pseudopotentials = dict(pseudopotentials or {})
        self.spin_polarized = bool(spin_polarized)
        self.starting_magnetization = dict(starting_magnetization or {})
        self.allow_low_cutoff = bool(allow_low_cutoff)
        self.timeout_s = float(timeout_s)
        self.name = "qe/pw.x"
        self.cutoff_A = 0.0
        self.last: Dict[str, object] = {}

    def model_label(self) -> str:
        return self.name

    def _pseudos(self, structure: Structure) -> Dict[str, dict]:
        out = {}
        for symbol in sorted({pt.symbol(int(z)) for z in structure.numbers}):
            name = self.pseudopotentials.get(symbol)
            path = pseudo_dir() / name if name else fetch_pseudopotential(symbol)
            if not path.is_file():
                raise EspressoMissing(f"Pseudopotential {path} is missing.")
            out[symbol] = describe_pseudopotential(path)
        functionals = {d["functional"] for d in out.values()}
        if len(functionals) > 1:
            raise UnsupportedSystem(f"The pseudopotentials mix functionals {sorted(functionals)}.")
        if not self.allow_low_cutoff:
            for symbol, d in out.items():
                if d["suggested_ecutwfc_Ry"] and self.ecutwfc_Ry < d["suggested_ecutwfc_Ry"]:
                    raise UnsupportedSystem(
                        f"ecutwfc {self.ecutwfc_Ry:g} Ry is below the {d['suggested_ecutwfc_Ry']:g} "
                        f"Ry {d['file']} suggests; raise it or set allow_low_cutoff.")
                if d["suggested_ecutrho_Ry"] and self.ecutrho_Ry < d["suggested_ecutrho_Ry"]:
                    raise UnsupportedSystem(
                        f"ecutrho {self.ecutrho_Ry:g} Ry is below the {d['suggested_ecutrho_Ry']:g} "
                        f"Ry {d['file']} suggests; raise it or set allow_low_cutoff.")
        return out

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        if find_pw() is None:
            return False, ("Quantum ESPRESSO pw.x was not found. Install it with "
                           "'conda create -n materia-qe -c conda-forge qe' (on Apple silicon "
                           "prefix CONDA_SUBDIR=osx-64) or set MATERIA_QE_PW.")
        if not all(structure.cell.pbc):
            return False, ("pw.x needs a three-dimensional periodic cell; pad slabs and "
                           "molecules with vacuum and mark all directions periodic.")
        try:
            self._pseudos(structure)
        except UnsupportedSystem as exc:
            return False, str(exc)
        return True, ""

    def input_data(self) -> dict:
        system = {"ecutwfc": self.ecutwfc_Ry, "ecutrho": self.ecutrho_Ry}
        if self.smearing == "fixed":
            system["occupations"] = "fixed"
        else:
            system.update({"occupations": "smearing", "smearing": self.smearing,
                           "degauss": self.degauss_Ry})
        if self.spin_polarized:
            system["nspin"] = 2
        if not self.symmetry:
            system["nosym"] = True
        return {"control": {"calculation": "scf", "tprnfor": True, "tstress": True,
                            "disk_io": "none"},
                "system": system,
                "electrons": {"conv_thr": self.conv_thr_Ry, "mixing_beta": 0.5}}

    def _atoms(self, structure: Structure):
        from ase import Atoms

        atoms = Atoms(numbers=structure.numbers, positions=structure.positions,
                      cell=structure.cell.matrix, pbc=True)
        if self.spin_polarized:
            atoms.set_initial_magnetic_moments(
                [self.starting_magnetization.get(pt.symbol(int(z)), 0.0)
                 for z in structure.numbers])
        return atoms

    def _execute(self, atoms, pseudos: Dict[str, dict], input_data: dict, kpts,
                 workdir: str, label: str) -> str:
        """Write one pw.x input, run it with a timeout and one thread, return the output."""
        from ase.io import write

        data = deepcopy(input_data)
        data.setdefault("control", {}).update(
            {"outdir": "./out", "prefix": "materia", "pseudo_dir": str(pseudo_dir())})
        path = Path(workdir) / f"{label}.pwi"
        write(path, atoms, format="espresso-in", input_data=data,
              pseudopotentials={s: d["file"] for s, d in pseudos.items()}, kpts=kpts)
        try:
            done = subprocess.run([find_pw(), "-in", path.name], cwd=workdir,
                                  capture_output=True, text=True, timeout=self.timeout_s,
                                  env={**os.environ, **THREAD_ENV})
        except subprocess.TimeoutExpired:
            raise EspressoFailed(f"pw.x exceeded {self.timeout_s:g} s.") from None
        (Path(workdir) / f"{label}.pwo").write_text(done.stdout)
        if done.returncode != 0 or "JOB DONE" not in done.stdout:
            raise EspressoFailed(f"pw.x stopped without finishing ({label}): "
                                 f"{(done.stdout + done.stderr)[-600:]}")
        return done.stdout

    def run(self, structure: Structure) -> dict:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        pseudos = self._pseudos(structure)
        workdir = tempfile.mkdtemp(prefix="materia-qe-")
        try:
            text = self._execute(self._atoms(structure), pseudos, self.input_data(), self.kpts,
                                 workdir, "scf")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        parsed = parse_output(text)
        if not parsed["converged"]:
            raise EspressoFailed("pw.x did not report a converged self-consistent field.")
        if parsed["energy_eV"] is None or "forces_eV_A" not in parsed:
            raise EspressoFailed("pw.x output carries no final energy or forces.")
        result = {"energy_eV": parsed["energy_eV"], "forces_eV_A": parsed["forces_eV_A"],
                  "magnetization_muB": parsed["magnetization_muB"],
                  "absolute_magnetization_muB": parsed["absolute_magnetization_muB"],
                  "stress_eV_A3": parsed.get("stress_eV_A3"), "fermi_eV": parsed["fermi_eV"],
                  "scf_iterations": parsed["iterations"], "parsed_energy_eV": parsed["energy_eV"],
                  "pseudopotentials": pseudos, "pw_version": pw_version(find_pw()),
                  "output_tail": text[-2000:]}
        self.last = result
        return result

    def relax(self, structure: Structure, variable_cell: bool = False,
              force_tolerance_eV_A: float = 1e-3, pressure_kbar: float = 0.0,
              max_steps: int = 100) -> dict:
        """Geometry (and cell) optimisation by pw.x's own BFGS ('relax' or 'vc-relax')."""
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        from ase.io import read

        pseudos = self._pseudos(structure)
        data = self.input_data()
        data["control"].update({"calculation": "vc-relax" if variable_cell else "relax",
                                "forc_conv_thr": force_tolerance_eV_A / (RYDBERG_EV / BOHR_A),
                                "etot_conv_thr": 1e-6, "nstep": int(max_steps)})
        data["ions"] = {"ion_dynamics": "bfgs"}
        if variable_cell:
            data["cell"] = {"cell_dynamics": "bfgs", "press": float(pressure_kbar),
                            "press_conv_thr": 0.1}
        workdir = tempfile.mkdtemp(prefix="materia-qe-relax-")
        try:
            text = self._execute(self._atoms(structure), pseudos, data, self.kpts, workdir,
                                 "relax")
            frames = read(Path(workdir) / "relax.pwo", format="espresso-out", index=":")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if "bfgs converged" not in text and "End of BFGS Geometry Optimization" not in text:
            raise EspressoFailed("pw.x BFGS did not converge within the allowed steps.")
        final = frames[-1]
        relaxed = structure.copy()
        relaxed.positions = final.get_positions()
        if variable_cell:
            from ..core_model.cell import Cell
            relaxed.cell = Cell(np.asarray(final.get_cell()), (True, True, True))
        parsed = parse_output(text)
        return {"structure": relaxed, "energy_eV": parsed["energy_eV"],
                "steps": len(frames), "variable_cell": variable_cell,
                "cell_A": np.asarray(final.get_cell()).tolist(),
                "volume_A3": float(abs(np.linalg.det(final.get_cell()))),
                "pw_version": pw_version(find_pw()), "pseudopotentials": pseudos}

    def band_structure(self, structure: Structure, path: Optional[str] = None,
                       density: float = 20.0, n_bands: Optional[int] = None) -> dict:
        """SCF on the k grid, then a 'bands' run along a high-symmetry path from ASE.

        Returns eigenvalues in eV along the path, the path labels, the
        valence-band maximum, conduction-band minimum and gap when one exists.
        """
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        from ase.io import read

        atoms = self._atoms(structure)
        bandpath = atoms.cell.bandpath(path, density=density)
        pseudos = self._pseudos(structure)
        workdir = tempfile.mkdtemp(prefix="materia-qe-bands-")
        try:
            first = self.input_data()
            first["control"]["disk_io"] = "low"
            scf = self._execute(atoms, pseudos, first, self.kpts, workdir, "scf")
            data = self.input_data()
            data["control"].update({"calculation": "bands", "tprnfor": False,
                                    "tstress": False, "disk_io": "low"})
            electrons = sum(_valence(pseudo_dir() / pseudos[pt.symbol(int(z))]["file"])
                            for z in structure.numbers)
            data["system"]["nbnd"] = int(n_bands or max(8, int(electrons / 2) + 8))
            self._execute(atoms, pseudos, data, bandpath, workdir, "bands")
            calc = read(Path(workdir) / "bands.pwo", format="espresso-out").calc
            eigenvalues = np.array([calc.get_eigenvalues(kpt=k, spin=0)
                                    for k in range(len(calc.get_ibz_k_points()))])
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        fermi = parse_output(scf)["fermi_eV"]
        occupied = int(round(electrons / 2))
        out = {"kpts": bandpath.kpts.tolist(), "labels": bandpath.path,
               "special_points": {k: v.tolist() for k, v in bandpath.special_points.items()},
               "eigenvalues_eV": eigenvalues.tolist(), "fermi_eV": fermi,
               "valence_electrons": electrons, "pw_version": pw_version(find_pw())}
        if electrons % 2 == 0 and not self.spin_polarized and occupied < eigenvalues.shape[1]:
            vbm = float(eigenvalues[:, occupied - 1].max())
            cbm = float(eigenvalues[:, occupied].min())
            out.update({"vbm_eV": vbm, "cbm_eV": cbm, "gap_eV": max(0.0, cbm - vbm),
                        "direct": int(np.argmax(eigenvalues[:, occupied - 1])) ==
                        int(np.argmin(eigenvalues[:, occupied]))})
        return out

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        out = self.run(structure)
        return out["energy_eV"], out["forces_eV_A"]

    def describe(self) -> dict:
        pseudos = self.last.get("pseudopotentials", {}) if self.last else {}
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": 0.0,
                "parameters": {"engine": "Quantum ESPRESSO pw.x",
                               "engine_version": self.last.get("pw_version") if self.last
                               else None,
                               "ecutwfc_Ry": self.ecutwfc_Ry, "ecutrho_Ry": self.ecutrho_Ry,
                               "kpts": list(self.kpts), "smearing": self.smearing,
                               "degauss_Ry": self.degauss_Ry, "conv_thr_Ry": self.conv_thr_Ry,
                               "spin_polarized": self.spin_polarized,
                               "symmetry": self.symmetry,
                               "pseudopotentials": pseudos},
                "approximations": [
                    "Kohn-Sham DFT in a plane-wave basis with the functional of the "
                    "pseudopotentials; pseudopotential and finite-cutoff errors are not "
                    "removed.",
                    (f"Monkhorst-Pack grid {self.kpts}, fixed occupations (insulator)."
                     if self.smearing == "fixed" else
                     f"Monkhorst-Pack grid {self.kpts}, {self.smearing} smearing of "
                     f"{self.degauss_Ry:g} Ry; the energy is the smeared total energy."),
                    ("k points reduced by the symmetry pw.x finds at each geometry; energies of "
                     "geometries with different symmetry can differ by the k-sampling error. "
                     "Use symmetry=False for finite differences." if self.symmetry else
                     "Symmetry disabled (nosym): the full k grid at every geometry, so "
                     "energies of nearby geometries are consistent."),
                ],
                "references": list(REFERENCES)}


def _valence(path: Path) -> float:
    head = path.read_bytes()[:200000].decode("latin-1")
    match = re.search(r'z_valence="\s*([\d.Ee+-]+)"', head) or re.search(
        r"([\d.]+)\s+Z valence", head)
    if not match:
        raise EspressoFailed(f"Could not read the valence charge of {path.name}.")
    return float(match.group(1))


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def sibling(program: str) -> str:
    """Another Quantum ESPRESSO executable installed beside pw.x."""
    pw = find_pw()
    if pw is None:
        raise EspressoMissing("Quantum ESPRESSO pw.x was not found.")
    path = Path(pw).with_name(program + (".exe" if pw.lower().endswith(".exe") else ""))
    if not path.is_file():
        raise EspressoMissing(f"{program} is not installed beside {pw}.")
    return str(path)


def _run_program(program: str, text: str, workdir: str, timeout_s: float, label: str) -> str:
    path = Path(workdir) / f"{label}.in"
    path.write_text(text)
    try:
        done = subprocess.run([sibling(program), "-in", path.name], cwd=workdir,
                              capture_output=True, text=True, timeout=timeout_s,
                              env={**os.environ, **THREAD_ENV})
    except subprocess.TimeoutExpired:
        raise EspressoFailed(f"{program} exceeded {timeout_s:g} s.") from None
    (Path(workdir) / f"{label}.out").write_text(done.stdout)
    if done.returncode != 0 or "JOB DONE" not in done.stdout:
        raise EspressoFailed(f"{program} did not finish: {(done.stdout + done.stderr)[-600:]}")
    return done.stdout


def _frequencies_THz(text: str) -> List[float]:
    return [float(v) for v in re.findall(r"freq \(\s*\d+\)\s*=\s*(-?[\d.]+)\s*\[THz\]", text)]


class EspressoPhonons:
    """Density-functional perturbation theory phonons with ph.x, q2r.x and matdyn.x.

    Uses the electronic settings of an :class:`EspressoPotential`.  Gamma-point
    frequencies come from ph.x and, after the crystal acoustic sum rule, from
    dynmat.x; dispersions from ph.x on a q grid, q2r.x and matdyn.x along a path.
    """

    REFERENCES = ["S. Baroni, S. de Gironcoli, A. Dal Corso and P. Giannozzi, "
                  "Rev. Mod. Phys. 73 (2001) 515"]

    def __init__(self, potential: "EspressoPotential", tr2_ph: float = 1e-16,
                 asr: str = "crystal") -> None:
        if asr not in ("crystal", "simple", "no"):
            raise ValueError("asr must be crystal, simple or no.")
        if potential.smearing != "fixed" and potential.degauss_Ry > 0.02:
            raise UnsupportedSystem("DFPT here expects small smearing or an insulator.")
        self.potential = potential
        self.tr2_ph = float(tr2_ph)
        self.asr = asr

    def _scf(self, structure: Structure, workdir: str) -> Dict[str, dict]:
        ok, why = self.potential.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        pseudos = self.potential._pseudos(structure)
        data = self.potential.input_data()
        data["control"].update({"disk_io": "low", "tprnfor": False, "tstress": False})
        self.potential._execute(self.potential._atoms(structure), pseudos, data,
                                self.potential.kpts, workdir, "scf")
        return pseudos

    def gamma(self, structure: Structure) -> dict:
        workdir = tempfile.mkdtemp(prefix="materia-ph-")
        try:
            self._scf(structure, workdir)
            text = _run_program("ph.x", f"""phonons at Gamma
&inputph
  tr2_ph = {self.tr2_ph:.1e}, prefix = 'materia', outdir = './out', fildyn = 'gamma.dyn'
/
0.0 0.0 0.0
""", workdir, self.potential.timeout_s, "ph")
            raw = _frequencies_THz(text)
            dyn = _run_program("dynmat.x", f"""&input
  fildyn = 'gamma.dyn', asr = '{self.asr}'
/
""", workdir, 600, "dynmat")
            table = dyn.split("# mode")[-1]
            corrected = [float(v) for v in re.findall(
                r"^\s*\d+\s+-?[\d.]+\s+(-?[\d.]+)\s+-?[\d.]+\s*$", table, re.M)]
            if not corrected:
                raise EspressoFailed("dynmat.x printed no frequency table.")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return {"frequencies_THz": raw, "asr_frequencies_THz": corrected,
                "asr": self.asr, "pw_version": pw_version(find_pw()),
                "references": list(self.REFERENCES)}

    def dielectric(self, structure: Structure,
                   q_direction: Sequence[float] = (1.0, 0.0, 0.0)) -> dict:
        """High-frequency dielectric tensor, Born effective charges and LO-TO splitting.

        ph.x at Gamma with the electric-field perturbation (``epsil``) gives
        epsilon_infinity and the Born charges ``Z*_k,ij = dF_k,j / dE_i`` (in e);
        dynmat.x gives Gamma frequencies without and with the non-analytic
        term for a long-wavelength phonon along ``q_direction``.  The charge
        sum rule violation ``max |sum_k Z*_k|`` measures k-point convergence and
        is returned with the charges.  For a cubic crystal with two atoms the
        LO frequency is also computed by Materia from
        ``w_LO^2 = w_TO^2 + Z*^2 e^2 / (eps0 eps_inf Omega mu)`` as a check on
        the non-analytic term, and the static constant from Lyddane-Sachs-Teller.
        Insulators only (fixed occupations).
        """
        if self.potential.smearing != "fixed":
            raise UnsupportedSystem("The electric-field response needs an insulator: use "
                                    "smearing='fixed'.")
        direction = np.asarray(q_direction, dtype=float)
        if direction.shape != (3,) or not np.linalg.norm(direction) > 0:
            raise ValueError("q_direction must be a nonzero 3-vector.")
        workdir = tempfile.mkdtemp(prefix="materia-epsil-")
        try:
            self._scf(structure, workdir)
            text = _run_program("ph.x", f"""dielectric response at Gamma
&inputph
  tr2_ph = {self.tr2_ph:.1e}, prefix = 'materia', outdir = './out', fildyn = 'gamma.dyn',
  epsil = .true.
/
0.0 0.0 0.0
""", workdir, self.potential.timeout_s, "ph")
            plain = _run_program("dynmat.x", f"""&input
  fildyn = 'gamma.dyn', asr = '{self.asr}'
/
""", workdir, 600, "dynmat0")
            q = ", ".join(f"q({i + 1}) = {v:.10f}" for i, v in enumerate(direction))
            polar = _run_program("dynmat.x", f"""&input
  fildyn = 'gamma.dyn', asr = '{self.asr}', {q}
/
""", workdir, 600, "dynmatq")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        number = r"(-?[\d.]+(?:[eE][-+]?\d+)?)"
        block = re.search(r"Dielectric constant in cartesian axis\s*\n\s*\n" +
                          r"\s*\n".join([r"\s*\(\s*" + r"\s+".join([number] * 3) + r"\s*\)"] * 3),
                          text)
        if block is None:
            raise EspressoFailed("ph.x printed no dielectric tensor.")
        epsilon = np.array(block.groups(), dtype=float).reshape(3, 3)
        section = text.split("Effective charges (d Force / dE) in cartesian axis without "
                             "acoustic sum rule applied (asr)")
        if len(section) < 2:
            raise EspressoFailed("ph.x printed no Born effective charges.")
        rows = re.findall(r"E[xyz]\s+\(\s*" + r"\s+".join([number] * 3) + r"\s*\)",
                          section[1])
        n = len(structure)
        if len(rows) < 3 * n:
            raise EspressoFailed("Could not read a Born charge tensor for every atom.")
        charges = np.array(rows[:3 * n], dtype=float).reshape(n, 3, 3)
        violation = float(np.abs(charges.sum(axis=0)).max())

        def table(output: str) -> List[float]:
            body = output.split("# mode")[-1]
            return [float(v) for v in re.findall(
                r"^\s*\d+\s+-?[\d.]+\s+(-?[\d.]+)\s+-?[\d.]+\s*$", body, re.M)]

        out = {"epsilon_infinity": epsilon.tolist(),
               "born_charges_e": charges.tolist(),
               "charge_sum_rule_violation_e": violation,
               "gamma_frequencies_THz": table(plain),
               "gamma_frequencies_with_nac_THz": table(polar),
               "q_direction": direction.tolist(), "asr": self.asr,
               "pw_version": pw_version(find_pw()),
               "references": list(self.REFERENCES) + [
                   "X. Gonze and C. Lee, Phys. Rev. B 55 (1997) 10355"]}
        cubic = (np.allclose(epsilon, epsilon[0, 0] * np.eye(3), atol=1e-4 * epsilon[0, 0])
                 and all(np.allclose(z, z[0, 0] * np.eye(3), atol=1e-3) for z in charges))
        if n == 2 and cubic:
            masses = np.asarray(structure.masses(), dtype=float) * 1.66053906660e-27
            mu = masses.prod() / masses.sum()
            z = 0.5 * (charges[0, 0, 0] - charges[1, 0, 0])
            volume = structure.cell.volume * 1e-30
            to = max(out["gamma_frequencies_THz"])
            omega_to = 2 * np.pi * to * 1e12
            gap = z ** 2 * 1.602176634e-19 ** 2 / (8.8541878128e-12 * epsilon[0, 0] * volume * mu)
            lo = np.sqrt(omega_to ** 2 + gap) / (2 * np.pi * 1e12)
            out.update({"transverse_optical_THz": to, "longitudinal_optical_THz": float(lo),
                        "longitudinal_optical_dynmat_THz": max(out["gamma_frequencies_with_nac_THz"]),
                        "epsilon_static": float(epsilon[0, 0] * (lo / to) ** 2)})
        return out

    def dispersion(self, structure: Structure, q_grid: Sequence[int] = (4, 4, 4),
                   path: Optional[str] = None, density: float = 20.0) -> dict:
        from ase import Atoms

        atoms = Atoms(numbers=structure.numbers, positions=structure.positions,
                      cell=structure.cell.matrix, pbc=True)
        bandpath = atoms.cell.bandpath(path, density=density)
        alat_bohr = None
        workdir = tempfile.mkdtemp(prefix="materia-phdisp-")
        try:
            self._scf(structure, workdir)
            nq = [int(v) for v in q_grid]
            _run_program("ph.x", f"""phonons on a grid
&inputph
  tr2_ph = {self.tr2_ph:.1e}, prefix = 'materia', outdir = './out', fildyn = 'grid.dyn',
  ldisp = .true., nq1 = {nq[0]}, nq2 = {nq[1]}, nq3 = {nq[2]}
/
""", workdir, self.potential.timeout_s * 4, "ph")
            _run_program("q2r.x", f"""&input
  fildyn = 'grid.dyn', zasr = '{self.asr}', flfrc = 'grid.fc'
/
""", workdir, 600, "q2r")
            scf_text = (Path(workdir) / "scf.pwo").read_text()
            alat_bohr = float(re.search(r"lattice parameter \(alat\)\s*=\s*([\d.]+)",
                                        scf_text).group(1))
            alat_A = alat_bohr * BOHR_A
            cartesian = bandpath.cartesian_kpts() * alat_A
            lines = "\n".join(f"{k[0]:.10f} {k[1]:.10f} {k[2]:.10f}" for k in cartesian)
            _run_program("matdyn.x", f"""&input
  asr = '{self.asr}', flfrc = 'grid.fc', flfrq = 'path.freq', q_in_band_form = .false.
/
{len(cartesian)}
{lines}
""", workdir, 600, "matdyn")
            frequencies = _read_matdyn(Path(workdir) / "path.freq")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return {"q_grid": nq, "labels": bandpath.path,
                "q_points": bandpath.kpts.tolist(),
                "special_points": {k: v.tolist() for k, v in bandpath.special_points.items()},
                "frequencies_THz": (np.asarray(frequencies) * 0.0299792458).tolist(),
                "asr": self.asr, "lo_to_splitting": self.potential.smearing == "fixed",
                "pw_version": pw_version(find_pw()),
                "references": list(self.REFERENCES)}


def _read_matdyn(path: Path) -> np.ndarray:
    text = path.read_text()
    header = re.search(r"nbnd=\s*(\d+),\s*nks=\s*(\d+)", text)
    nbnd, nks = int(header.group(1)), int(header.group(2))
    numbers = [float(v) for v in re.findall(r"-?\d+\.\d+(?:[Ee][-+]?\d+)?",
                                             text[header.end():])]
    if len(numbers) % nks:
        raise EspressoFailed("matdyn.x frequency file has an unexpected layout.")
    block = len(numbers) // nks
    leading = block - nbnd
    if leading not in (3, 4):
        raise EspressoFailed("matdyn.x frequency file has an unexpected layout.")
    rows = [numbers[k * block + leading:(k + 1) * block] for k in range(nks)]
    return np.array(rows)



def projected_dos(potential: "EspressoPotential", structure: Structure,
                  nscf_kpts: Sequence[int] = (12, 12, 12), emin_eV: float = -15.0,
                  emax_eV: float = 10.0, delta_eV: float = 0.02,
                  broadening_Ry: float = 0.01) -> dict:
    """Total and projected DOS by an nscf run on a denser grid and projwfc.x.

    Projections are onto the atomic pseudo-wavefunctions of the
    pseudopotentials (Loewdin), so they are not a complete basis: the
    spilling parameter printed by projwfc.x measures what is missed and is
    returned.  Energies are absolute, in eV, with the Fermi level from the
    nscf run.
    """
    ok, why = potential.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    pseudos = potential._pseudos(structure)
    atoms = potential._atoms(structure)
    workdir = tempfile.mkdtemp(prefix="materia-pdos-")
    try:
        first = potential.input_data()
        first["control"].update({"disk_io": "low", "tprnfor": False, "tstress": False})
        potential._execute(atoms, pseudos, first, potential.kpts, workdir, "scf")
        electrons = sum(_valence(pseudo_dir() / pseudos[pt.symbol(int(z))]["file"])
                        for z in structure.numbers)
        nscf = potential.input_data()
        nscf["control"].update({"calculation": "nscf", "disk_io": "low", "tprnfor": False,
                                "tstress": False})
        nscf["system"].update({"nbnd": int(max(8, electrons / 2 + 8)), "nosym": False})
        text = potential._execute(atoms, pseudos, nscf, tuple(nscf_kpts), workdir, "nscf")
        fermi = parse_output(text)["fermi_eV"]
        out = _run_program("projwfc.x", f"""&projwfc
  prefix = 'materia', outdir = './out', filpdos = 'materia',
  Emin = {emin_eV}, Emax = {emax_eV}, DeltaE = {delta_eV}, degauss = {broadening_Ry}, ngauss = 0
/
""", workdir, potential.timeout_s, "projwfc")
        spilling = re.findall(r"Spilling Parameter:\s+(-?[\d.]+)", out)
        total = np.loadtxt(Path(workdir) / "materia.pdos_tot")
        channels = {}
        for path in sorted(Path(workdir).glob("materia.pdos_atm*")):
            match = re.search(r"atm#(\d+)\((\w+)\)_wfc#(\d+)\((\w+)\)", path.name)
            if not match:
                continue
            data = np.loadtxt(path)
            channels[f"{match.group(2)}{match.group(1)}:{match.group(4)}"] = data[:, 1].tolist()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"energies_eV": total[:, 0].tolist(), "dos": total[:, 1].tolist(),
            "pdos_sum": total[:, 2].tolist(), "channels": channels,
            "fermi_eV": fermi, "spilling": float(spilling[-1]) if spilling else None,
            "valence_electrons": electrons, "nscf_kpts": [int(k) for k in nscf_kpts],
            "pw_version": pw_version(find_pw()),
            "references": ["P.-O. Loewdin, J. Chem. Phys. 18 (1950) 365",
                           "D. Sanchez-Portal, E. Artacho and J. M. Soler, Solid State Commun. "
                           "95 (1995) 685 (spilling parameter)"]}


ANGULAR = "spdf"


def read_atomic_projections(xml_path: Path, projwfc_text: str) -> dict:
    """Eigenvalues (eV) and squared projections [k, band, state] from projwfc.x output."""
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    header = root.find("HEADER").attrib
    bands = int(header["NUMBER_OF_BANDS"])
    states = int(header["NUMBER_OF_ATOMIC_WFC"])
    if int(header["NUMBER_OF_SPIN_COMPONENTS"]) != 1:
        raise UnsupportedSystem("Orbital-resolved bands are implemented for spin-unpolarized "
                                "calculations only.")
    eigenvalues, weights = [], []
    for block in root.find("EIGENSTATES"):
        if block.tag == "E":
            eigenvalues.append(np.array(block.text.split(), dtype=float) * RYDBERG_EV)
        elif block.tag == "PROJS":
            matrix = np.zeros((bands, states))
            for wfc in block.findall("ATOMIC_WFC"):
                values = np.array(wfc.text.split(), dtype=float).reshape(bands, 2)
                matrix[:, int(wfc.attrib["index"]) - 1] = values[:, 0] ** 2 + values[:, 1] ** 2
            weights.append(matrix)
    labels = []
    for match in re.finditer(r"state #\s*(\d+): atom\s+(\d+) \((\w+)\s*\), wfc\s+(\d+) "
                             r"\(l=(\d) m=\s*(\d+)\)", projwfc_text):
        labels.append({"state": int(match.group(1)), "atom": int(match.group(2)) - 1,
                       "element": match.group(3), "wfc": int(match.group(4)),
                       "l": int(match.group(5)), "m": int(match.group(6))})
    if len(labels) != states:
        raise EspressoFailed(f"projwfc.x listed {len(labels)} atomic states; the projection "
                             f"file has {states}.")
    return {"eigenvalues_eV": np.array(eigenvalues), "weights": np.array(weights),
            "states": labels, "fermi_eV": float(header["FERMI_ENERGY"]) * RYDBERG_EV}


def fat_bands(potential: "EspressoPotential", structure: Structure, path: Optional[str] = None,
              density: float = 20.0, n_bands: Optional[int] = None) -> dict:
    """Band structure along a path with the orbital character of every state.

    SCF, a 'bands' run along the path, then projwfc.x without symmetrisation
    (``lsym = .false.``) so that each k point keeps its own projections.  The
    character of a state is the squared Loewdin projection onto each atomic
    pseudo-wavefunction, summed into element and angular-momentum channels
    (``Si:s``, ``Si:p``) and into atoms.  The projections do not form a
    complete basis; ``1 - sum`` for each state is returned as its spilling.
    """
    if potential.spin_polarized:
        raise UnsupportedSystem("Orbital-resolved bands are implemented for spin-unpolarized "
                                "calculations only.")
    ok, why = potential.composition(structure)
    if not ok:
        raise UnsupportedSystem(why)
    from ase.io import read

    atoms = potential._atoms(structure)
    bandpath = atoms.cell.bandpath(path, density=density)
    pseudos = potential._pseudos(structure)
    electrons = sum(_valence(pseudo_dir() / pseudos[pt.symbol(int(z))]["file"])
                    for z in structure.numbers)
    workdir = tempfile.mkdtemp(prefix="materia-fatbands-")
    try:
        first = potential.input_data()
        first["control"].update({"disk_io": "low", "tprnfor": False, "tstress": False})
        scf = potential._execute(atoms, pseudos, first, potential.kpts, workdir, "scf")
        data = potential.input_data()
        data["control"].update({"calculation": "bands", "tprnfor": False, "tstress": False,
                                "disk_io": "low"})
        data["system"]["nbnd"] = int(n_bands or max(8, int(electrons / 2) + 8))
        potential._execute(atoms, pseudos, data, bandpath, workdir, "bands")
        calc = read(Path(workdir) / "bands.pwo", format="espresso-out").calc
        pw_bands = np.array([calc.get_eigenvalues(kpt=k, spin=0)
                             for k in range(len(calc.get_ibz_k_points()))])
        text = _run_program("projwfc.x", "&projwfc\n  prefix = 'materia', outdir = './out', "
                            "lsym = .false., filproj = 'materia.proj'\n/\n", workdir,
                            potential.timeout_s, "projwfc")
        projections = read_atomic_projections(
            Path(workdir) / "out" / "materia.save" / "atomic_proj.xml", text)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    eigenvalues = projections["eigenvalues_eV"]
    if eigenvalues.shape != pw_bands.shape or np.abs(eigenvalues - pw_bands).max() > 1e-3:
        raise EspressoFailed("The projected eigenvalues do not match the pw.x band energies.")
    weights = projections["weights"]
    channels: Dict[str, np.ndarray] = {}
    by_atom: Dict[int, np.ndarray] = {}
    for k, state in enumerate(projections["states"]):
        key = f"{state['element']}:{ANGULAR[state['l']]}"
        channels[key] = channels.get(key, 0) + weights[:, :, k]
        by_atom[state["atom"]] = by_atom.get(state["atom"], 0) + weights[:, :, k]
    total = weights.sum(axis=2)
    if total.max() > 1.0 + 1e-3:
        raise EspressoFailed(f"A state has a projected weight of {total.max():.4f}, above one; "
                             "the projections are not orthonormalised.")
    fermi = parse_output(scf)["fermi_eV"]
    return {"kpts": bandpath.kpts.tolist(), "labels": bandpath.path,
            "special_points": {k: v.tolist() for k, v in bandpath.special_points.items()},
            "eigenvalues_eV": eigenvalues.tolist(), "fermi_eV": fermi,
            "states": projections["states"],
            "character": {k: v.tolist() for k, v in channels.items()},
            "atom_character": {int(k): v.tolist() for k, v in by_atom.items()},
            "spilling": (1 - total).tolist(), "valence_electrons": electrons,
            "pw_version": pw_version(find_pw()),
            "references": ["P.-O. Loewdin, J. Chem. Phys. 18 (1950) 365",
                           "P. Giannozzi et al., J. Phys.: Condens. Matter 29 (2017) 465901"]}
