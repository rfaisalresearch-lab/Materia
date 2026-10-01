"""Native LAMMPS input: the data file, the input script and the potential copy.

Everything here is a pure function of a frozen specification, so the same
specification always produces byte-identical files and their SHA-256 can be
recomputed later to prove what was run.

Geometry
--------
LAMMPS stores a restricted triclinic box: ``a`` along x, ``b`` in the xy
plane, ``c`` anywhere with a positive z component.  Materia cells already in
that form are written as they are, with no rotation, so LAMMPS coordinates
and Materia coordinates are the same numbers.  A cell in any other
orientation is refused rather than rotated behind the user's back.

Periodic directions start at the origin and span the cell vector.  Atoms
outside the cell are wrapped into it and their image flags written, so the
unwrapped coordinates LAMMPS reports (``xu yu zu``) are the original
positions.  Non-periodic directions use shrink-wrapped (``s``) boundaries,
which follow the atoms and cannot lose them; such a direction must not take
part in any tilt, as LAMMPS itself requires.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .units import FS_PER_PS, fs_to_ps, velocity_to_lammps

DATA_FILE = "structure.data"
INPUT_FILE = "in.lammps"
LOG_FILE = "log.lammps"
FINAL_DUMP = "final.dump"
TRAJECTORY_DUMP = "trajectory.dump"

SECTION_MAIN = "@@MATERIA section main"
SECTION_FINAL = "@@MATERIA section final"
COMPLETE = "@@MATERIA complete"

THERMO_KEYWORDS: Tuple[str, ...] = (
    "step", "time", "pe", "ke", "etotal", "temp", "press",
    "pxx", "pyy", "pzz", "pxy", "pxz", "pyz", "atoms")
FINAL_COLUMNS: Tuple[str, ...] = (
    "id", "type", "mass", "xu", "yu", "zu", "vx", "vy", "vz", "fx", "fy", "fz")
TRAJECTORY_COLUMNS: Tuple[str, ...] = ("id", "type", "xu", "yu", "zu", "vx", "vy", "vz")

SHRINK_PAD_A = 1.0
TILT_LIMIT = 0.5
FIXED_GROUP = "materia_fixed"
MOBILE_GROUP = "materia_mobile"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class BoxError(ValueError):
    """A cell that cannot be written as a LAMMPS box without changing it."""


@dataclass(frozen=True)
class Box:
    lo: np.ndarray
    hi: np.ndarray
    tilt: np.ndarray
    images: np.ndarray
    wrapped: np.ndarray

    @property
    def triclinic(self) -> bool:
        return bool(np.any(self.tilt != 0.0))


def box_geometry(matrix: np.ndarray, pbc: Sequence[bool], positions: np.ndarray) -> Box:
    """The LAMMPS box, wrapped positions and image flags for a Materia cell."""
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    pbc = tuple(bool(p) for p in pbc)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(pos)
    scale = max(1.0, float(np.abs(m).max()) if m.size else 1.0)
    tol = 1e-9 * scale
    images = np.zeros((n, 3), dtype=np.int64)
    if not any(pbc):
        lo = pos.min(axis=0) - SHRINK_PAD_A if n else np.zeros(3)
        hi = pos.max(axis=0) + SHRINK_PAD_A if n else np.ones(3)
        return Box(lo, hi, np.zeros(3), images, pos.copy())
    for (row, col) in ((0, 1), (0, 2), (1, 2)):
        if abs(m[row, col]) > tol:
            raise BoxError(
                "The cell is not in the LAMMPS restricted triclinic orientation (a along "
                "x, b in the xy plane). Materia does not rotate structures silently; "
                "reorient the structure so that its cell matrix is lower triangular.")
    names = "xyz"
    for axis, periodic in enumerate(pbc):
        if periodic and m[axis, axis] <= tol:
            raise BoxError(f"The periodic cell vector along {names[axis]} must have a "
                           "positive diagonal component.")
    tilts = {"xy": (1, 0), "xz": (2, 0), "yz": (2, 1)}
    for name, (row, col) in tilts.items():
        value = m[row, col]
        if abs(value) <= tol:
            continue
        axes = {name[0], name[1]}
        for axis in axes:
            if not pbc["xyz".index(axis)]:
                raise BoxError(
                    f"The cell has a {name} tilt of {value:.6g} A but {axis} is not "
                    "periodic. LAMMPS allows a tilt only between periodic directions, "
                    "and a shrink-wrapped direction must not be tilted.")
        length = m[col, col]
        if abs(value) > TILT_LIMIT * length + tol:
            raise BoxError(
                f"The {name} tilt {value:.6g} A exceeds half the {name[0]} box length "
                f"{length:.6g} A. Choose the equivalent reduced cell before running LAMMPS.")
    for axis, periodic in enumerate(pbc):
        if not periodic:
            others = [m[axis, k] for k in range(3) if k != axis]
            if any(abs(v) > tol for v in others):
                raise BoxError(f"The non-periodic cell vector along {names[axis]} must lie "
                               "along its own axis.")
    basis = m.copy()
    for axis, periodic in enumerate(pbc):
        if not periodic:
            basis[axis] = 0.0
            basis[axis, axis] = 1.0
    fractional = pos @ np.linalg.inv(basis)
    for axis, periodic in enumerate(pbc):
        if periodic:
            shift = np.floor(fractional[:, axis])
            fractional[:, axis] -= shift
            images[:, axis] = shift.astype(np.int64)
    wrapped = fractional @ basis
    lo = np.zeros(3)
    hi = np.zeros(3)
    for axis, periodic in enumerate(pbc):
        if periodic:
            hi[axis] = m[axis, axis]
        else:
            lo[axis] = float(pos[:, axis].min()) - SHRINK_PAD_A if n else 0.0
            hi[axis] = float(pos[:, axis].max()) + SHRINK_PAD_A if n else 1.0
    tilt = np.array([m[1, 0], m[2, 0], m[2, 1]])
    tilt[np.abs(tilt) <= tol] = 0.0
    return Box(lo, hi, tilt, images, wrapped)


def unwrap(wrapped: np.ndarray, images: np.ndarray, matrix: np.ndarray,
           pbc: Sequence[bool]) -> np.ndarray:
    """Inverse of the wrap in :func:`box_geometry`."""
    basis = np.asarray(matrix, dtype=float).reshape(3, 3).copy()
    for axis, periodic in enumerate(pbc):
        if not periodic:
            basis[axis] = 0.0
    return np.asarray(wrapped, dtype=float) + np.asarray(images, dtype=float) @ basis


def _f(value: float) -> str:
    return repr(float(value))


def potential_file_name(spec) -> str:
    name = str(spec.potential.get("file_name") or "")
    return name if _SAFE_NAME.match(name) else "potential.eam.alloy"


def id_ranges(ids: Sequence[int]) -> List[str]:
    """Sorted identifiers compressed to LAMMPS ``a:b`` ranges."""
    values = sorted(int(i) for i in ids)
    out: List[str] = []
    start = previous = None
    for value in values:
        if start is None:
            start = previous = value
        elif value == previous + 1:
            previous = value
        else:
            out.append(str(start) if start == previous else f"{start}:{previous}")
            start = previous = value
    if start is not None:
        out.append(str(start) if start == previous else f"{start}:{previous}")
    return out


def render_data(spec) -> str:
    """The LAMMPS data file for ``spec``, in metal units."""
    positions = spec.positions()
    box = box_geometry(np.array(spec.cell_A), spec.pbc, positions)
    velocities = velocity_to_lammps(spec.velocities())
    lines = [f"LAMMPS data file written by Materia {spec.created_with}, "
             f"specification {spec.digest}", "",
             f"{spec.n_atoms} atoms", f"{len(spec.type_table)} atom types", "",
             f"{_f(box.lo[0])} {_f(box.hi[0])} xlo xhi",
             f"{_f(box.lo[1])} {_f(box.hi[1])} ylo yhi",
             f"{_f(box.lo[2])} {_f(box.hi[2])} zlo zhi"]
    if box.triclinic:
        lines.append(f"{_f(box.tilt[0])} {_f(box.tilt[1])} {_f(box.tilt[2])} xy xz yz")
    lines += ["", "Masses", ""]
    lines += [f"{row[0]} {_f(row[3])}" for row in spec.type_table]
    lines += ["", "Atoms", ""]
    for k, atom_id in enumerate(spec.atom_ids):
        x, y, z = box.wrapped[k]
        ix, iy, iz = box.images[k]
        lines.append(f"{atom_id} {spec.atom_types[k]} {_f(x)} {_f(y)} {_f(z)} "
                     f"{ix} {iy} {iz}")
    lines += ["", "Velocities", ""]
    for k, atom_id in enumerate(spec.atom_ids):
        vx, vy, vz = velocities[k]
        lines.append(f"{atom_id} {_f(vx)} {_f(vy)} {_f(vz)}")
    return "\n".join(lines) + "\n"


def render_input(spec) -> str:
    """The LAMMPS input script for ``spec``."""
    potential = potential_file_name(spec)
    elements = " ".join(row[1] for row in spec.type_table)
    thermo = " ".join(THERMO_KEYWORDS)
    lines = [
        f"units {spec.units}",
        f"atom_style {spec.atom_style}",
        f"boundary {' '.join(spec.boundary)}",
        f"read_data {DATA_FILE}",
        f"pair_style {spec.potential['style']}",
        f"pair_coeff * * {potential} {elements}",
    ]
    lines += [f"mass {row[0]} {_f(row[3])}" for row in spec.type_table]
    lines += [
        f"neighbor {_f(spec.neighbor_skin_A)} bin",
        "neigh_modify every 1 delay 0 check yes",
        f"thermo_style custom {thermo}",
        "thermo_modify format float %.17g norm no lost error flush yes",
    ]
    mobile = "all"
    fixes: List[str] = []
    if spec.fixed_ids and spec.task in ("relax", "md"):
        lines.append(f"group {FIXED_GROUP} id {' '.join(id_ranges(spec.fixed_ids))}")
        lines.append(f"group {MOBILE_GROUP} subtract all {FIXED_GROUP}")
        mobile = MOBILE_GROUP
    if spec.task == "relax":
        lines += [f"min_style {spec.min_style}", "min_modify norm max"]
        if spec.fixed_ids:
            lines.append(f"fix materia_freeze {FIXED_GROUP} setforce 0.0 0.0 0.0")
            fixes.append("materia_freeze")
        lines += [f"thermo {spec.thermo_every}", f'print "{SECTION_MAIN}"',
                  f"minimize {_f(spec.etol)} {_f(spec.ftol_eV_A)} {spec.max_iterations} "
                  f"{spec.max_evaluations}"]
    elif spec.task == "md":
        dt_ps = fs_to_ps(spec.timestep_fs)
        damping_ps = fs_to_ps(spec.damping_fs) if spec.damping_fs else None
        lines.append(f"timestep {_f(dt_ps)}")
        if spec.initial_velocities == "create":
            lines.append(f"velocity {mobile} create {_f(spec.temperature_K)} {spec.seed} "
                         "dist gaussian mom yes rot no loop geom")
        if spec.fixed_ids:
            lines.append(f"fix materia_freeze {FIXED_GROUP} setforce 0.0 0.0 0.0")
            fixes.append("materia_freeze")
        if spec.ensemble == "nvt":
            lines.append(f"fix materia_integrate {mobile} nvt temp {_f(spec.temperature_K)} "
                         f"{_f(spec.temperature_K)} {_f(damping_ps)}")
        else:
            lines.append(f"fix materia_integrate {mobile} nve")
        fixes.append("materia_integrate")
        if spec.ensemble == "langevin":
            lines.append(f"fix materia_thermostat {mobile} langevin "
                         f"{_f(spec.temperature_K)} {_f(spec.temperature_K)} "
                         f"{_f(damping_ps)} {spec.seed} zero yes")
            fixes.append("materia_thermostat")
        lines += [f"thermo {spec.thermo_every}"]
        if "trajectory" in spec.outputs:
            lines += [f"dump materia_trajectory all custom {spec.sample_every} "
                      f"{TRAJECTORY_DUMP} {' '.join(TRAJECTORY_COLUMNS)}",
                      "dump_modify materia_trajectory format float %.17g sort id"]
        lines += [f'print "{SECTION_MAIN}"', f"run {spec.steps}"]
        if "trajectory" in spec.outputs:
            lines.append("undump materia_trajectory")
    for fix in reversed(fixes):
        lines.append(f"unfix {fix}")
    lines += [
        "thermo 1",
        f'print "{SECTION_FINAL}"',
        "run 0",
        f"write_dump all custom {FINAL_DUMP} {' '.join(FINAL_COLUMNS)} "
        "modify format float %.17g sort id",
        f'print "{COMPLETE}"',
    ]
    return "\n".join(lines) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_inputs(spec, workdir: str) -> Dict[str, Dict[str, str]]:
    """Write the data file, input script and potential copy; return their identities."""
    from .spec import file_sha256

    root = Path(workdir)
    data = render_data(spec)
    script = render_input(spec)
    (root / DATA_FILE).write_text(data)
    (root / INPUT_FILE).write_text(script)
    target = root / potential_file_name(spec)
    shutil.copyfile(spec.potential["path"], target)
    copied = file_sha256(str(target))
    if copied != spec.potential["sha256"]:
        raise OSError(f"The potential copied into the run directory has SHA-256 "
                      f"{copied[:16]}..., not the frozen {spec.potential['sha256'][:16]}...")
    return {
        DATA_FILE: {"sha256": sha256_text(data), "bytes": str(len(data.encode()))},
        INPUT_FILE: {"sha256": sha256_text(script), "bytes": str(len(script.encode()))},
        target.name: {"sha256": copied, "bytes": str(target.stat().st_size)},
    }


def time_fs_from_ps(value_ps: float) -> float:
    return float(value_ps) * FS_PER_PS
