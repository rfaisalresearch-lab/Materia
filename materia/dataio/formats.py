"""Structure and data import/export.

Every writer emits the units, the cell and a provenance comment where the
format allows it.  Every reader records ``origin="imported"`` together with
the source path, so imported data is never mistaken for a calculated result.

Implemented
-----------
=========== ======= ======= ========================================
format      read    write   notes
=========== ======= ======= ========================================
xyz         yes     yes     comment line carries the cell and units
extxyz      yes     yes     ASE-compatible ``Lattice=`` key-value line
cif         yes     yes     P1 only: symmetry is written out explicitly
poscar      yes     yes     VASP 5 format with species line
pdb         yes     yes     CRYST1 + ATOM/HETATM records
lammps-data yes     yes     atom_style atomic/charge
cube        no      yes     volumetric data (density, potential, LDOS)
csv         no      yes     tabular measurement export
npz         yes     yes     raw arrays for scans
hdf5        yes     yes     optional (needs h5py)
=========== ======= ======= ========================================
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core_model.cell import Cell
from ..core_model.structure import Structure
from ..elements import periodic_table as pt
from ..provenance import Fidelity, Origin, Provenance
from ..version import __version__

READABLE = ("xyz", "extxyz", "cif", "poscar", "pdb", "lammps-data", "npz", "hdf5", "json")
WRITABLE = READABLE + ("cube", "csv", "png", "tiff")


class FormatError(ValueError):
    pass


def _banner(structure: Structure, extra: str = "") -> str:
    return (f"Materia {__version__}; units A/eV; "
            f"formula={structure.formula()}; pbc={structure.cell.pbc}; {extra}").strip("; ")


def detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    name = os.path.basename(path).upper()
    if name.startswith(("POSCAR", "CONTCAR")):
        return "poscar"
    return {
        "xyz": "xyz", "extxyz": "extxyz", "cif": "cif", "vasp": "poscar",
        "pdb": "pdb", "data": "lammps-data", "lmp": "lammps-data",
        "cube": "cube", "csv": "csv", "npz": "npz", "h5": "hdf5",
        "hdf5": "hdf5", "json": "json", "png": "png", "tif": "tiff", "tiff": "tiff",
    }.get(ext, ext)


def _import_provenance(path: str, fmt: str) -> dict:
    return Provenance(
        model=f"dataio/{fmt}",
        fidelity=Fidelity.TIER0_STRUCTURAL,
        origin=Origin.IMPORTED,
        approximations=["Geometry as supplied by the source file; not validated "
                        "against any physical model."],
        parameters={"source": os.path.abspath(path), "format": fmt},
        notes="Imported structure. Element assignment and cell come from the file.",
    ).as_dict()


def write_xyz(path: str, structure: Structure, comment: str = "") -> str:
    lines = [str(len(structure)), comment or _banner(structure)]
    for k in range(len(structure)):
        x, y, z = structure.positions[k]
        lines.append(f"{pt.symbol(int(structure.numbers[k])):<3s} "
                     f"{x:>14.8f} {y:>14.8f} {z:>14.8f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_extxyz(path: str, structure: Structure, comment: str = "") -> str:
    m = structure.cell.matrix.ravel()
    lattice = " ".join(f"{v:.8f}" for v in m)
    pbc = " ".join("T" if p else "F" for p in structure.cell.pbc)
    props = "species:S:1:pos:R:3:id:I:1:charge:R:1:role:S:1"
    header = (f'Lattice="{lattice}" Properties={props} pbc="{pbc}" '
              f'generator="Materia {__version__}" units="Angstrom"')
    if comment:
        header += f' comment="{comment}"'
    lines = [str(len(structure)), header]
    for k in range(len(structure)):
        x, y, z = structure.positions[k]
        lines.append(f"{pt.symbol(int(structure.numbers[k])):<3s} "
                     f"{x:>14.8f} {y:>14.8f} {z:>14.8f} "
                     f"{int(structure.ids[k]):>6d} "
                     f"{structure.formal_charges[k]:>8.4f} "
                     f"{str(structure.roles[k]) or 'unknown'}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_xyz(path: str) -> Structure:
    with open(path) as fh:
        raw = fh.read().strip().splitlines()
    if not raw:
        raise FormatError(f"{path} is empty")
    try:
        n = int(raw[0].split()[0])
    except Exception as exc:
        raise FormatError(f"{path}: first line must be the atom count") from exc
    comment = raw[1] if len(raw) > 1 else ""
    cell = Cell.none()
    pbc = (False, False, False)
    lat = re.search(r'Lattice="([^"]+)"', comment)
    if lat:
        vals = [float(v) for v in lat.group(1).split()]
        if len(vals) == 9:
            cell_matrix = np.array(vals).reshape(3, 3)
            pbc_m = re.search(r'pbc="([^"]+)"', comment)
            pbc = (tuple(v.upper() == "T" for v in pbc_m.group(1).split())
                   if pbc_m else (True, True, True))
            cell = Cell(cell_matrix, pbc)
    numbers, positions, ids, charges, roles = [], [], [], [], []
    for line in raw[2:2 + n]:
        parts = line.split()
        if len(parts) < 4:
            raise FormatError(f"{path}: malformed atom line {line!r}")
        numbers.append(pt.atomic_number(parts[0]))
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        ids.append(int(parts[4]) if len(parts) > 4 and parts[4].lstrip("-").isdigit() else None)
        charges.append(float(parts[5]) if len(parts) > 5 else 0.0)
        roles.append(parts[6] if len(parts) > 6 else "unknown")
    use_ids = None if any(i is None for i in ids) else ids
    s = Structure(numbers, np.array(positions), cell, ids=use_ids, roles=roles)
    s.formal_charges[:] = charges
    s.info["provenance"] = _import_provenance(path, "xyz")
    s.info["source_comment"] = comment
    return s


def write_cif(path: str, structure: Structure, name: str = "materia") -> str:
    c = structure.cell
    if not c.is_periodic:
        c = Cell.cubic(float(np.ptp(structure.positions) + 20.0))
    a, b, cc = c.lengths
    al, be, ga = c.angles_deg
    frac = c.to_fractional(structure.positions)
    lines = [
        f"# Generated by Materia {__version__} on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "# Symmetry is written as P1: every atom is listed explicitly.",
        f"data_{name}",
        f"_cell_length_a    {a:.6f}",
        f"_cell_length_b    {b:.6f}",
        f"_cell_length_c    {cc:.6f}",
        f"_cell_angle_alpha {al:.4f}",
        f"_cell_angle_beta  {be:.4f}",
        f"_cell_angle_gamma {ga:.4f}",
        f"_cell_volume      {c.volume:.6f}",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "  'x, y, z'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    for k in range(len(structure)):
        sym = pt.symbol(int(structure.numbers[k]))
        lines.append(f"  {sym}{int(structure.ids[k])} {sym} "
                     f"{frac[k][0]:.8f} {frac[k][1]:.8f} {frac[k][2]:.8f} 1.0")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_cif(path: str) -> Structure:
    text = open(path).read()
    def _num(key, default=None):
        m = re.search(rf"{key}\s+([-\d.eE()]+)", text)
        if not m:
            if default is None:
                raise FormatError(f"{path}: missing {key}")
            return default
        return float(re.sub(r"\(.*\)", "", m.group(1)))
    a, b, c = _num("_cell_length_a"), _num("_cell_length_b"), _num("_cell_length_c")
    al, be, ga = (_num("_cell_angle_alpha", 90.0), _num("_cell_angle_beta", 90.0),
                  _num("_cell_angle_gamma", 90.0))
    cell = Cell.from_parameters(a, b, c, al, be, ga)

    lines = text.splitlines()
    headers: List[str] = []
    rows: List[List[str]] = []
    in_loop = False
    for line in lines:
        st = line.strip()
        if st.startswith("loop_"):
            in_loop, headers, collecting = True, [], False
            continue
        if in_loop and st.startswith("_"):
            headers.append(st.split()[0])
            continue
        if in_loop and headers and st and not st.startswith("#"):
            if "_atom_site_fract_x" in headers:
                rows.append(st.split())
            else:
                continue
        elif in_loop and not st:
            if "_atom_site_fract_x" in headers and rows:
                break
            in_loop = False
    if "_atom_site_fract_x" not in headers:
        raise FormatError(f"{path}: no _atom_site_fract_* loop found")
    ix = headers.index("_atom_site_fract_x")
    iy = headers.index("_atom_site_fract_y")
    iz = headers.index("_atom_site_fract_z")
    isym = (headers.index("_atom_site_type_symbol")
            if "_atom_site_type_symbol" in headers else headers.index("_atom_site_label"))
    numbers, frac = [], []
    for r in rows:
        if len(r) <= max(ix, iy, iz, isym):
            continue
        sym = re.sub(r"[^A-Za-z]", "", r[isym])[:2]
        try:
            z = pt.atomic_number(sym)
        except Exception:
            z = pt.atomic_number(sym[0])
        numbers.append(z)
        frac.append([float(re.sub(r"\(.*\)", "", r[i])) for i in (ix, iy, iz)])
    s = Structure(numbers, cell.to_cartesian(np.array(frac)), cell)
    s.info["provenance"] = _import_provenance(path, "cif")
    return s


def write_poscar(path: str, structure: Structure, comment: str = "",
                 direct: bool = True) -> str:
    order = np.argsort(structure.numbers, kind="stable")
    syms = [pt.symbol(int(structure.numbers[k])) for k in order]
    uniq: List[str] = []
    counts: List[int] = []
    for s in syms:
        if not uniq or uniq[-1] != s:
            uniq.append(s)
            counts.append(1)
        else:
            counts[-1] += 1
    m = structure.cell.matrix
    lines = [comment or _banner(structure), "1.0"]
    for row in m:
        lines.append(f"  {row[0]:>18.12f} {row[1]:>18.12f} {row[2]:>18.12f}")
    lines.append("  " + "  ".join(f"{u:>4s}" for u in uniq))
    lines.append("  " + "  ".join(f"{c:>4d}" for c in counts))
    lines.append("Direct" if direct else "Cartesian")
    coords = (structure.cell.to_fractional(structure.positions) if direct
              else structure.positions)
    for k in order:
        p = coords[k]
        lines.append(f"  {p[0]:>18.12f} {p[1]:>18.12f} {p[2]:>18.12f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_poscar(path: str) -> Structure:
    lines = [l.rstrip() for l in open(path).read().splitlines()]
    scale = float(lines[1].split()[0])
    m = np.array([[float(v) for v in lines[2 + i].split()[:3]] for i in range(3)]) * scale
    species = lines[5].split()
    if species and species[0].replace("-", "").isdigit():
        raise FormatError(f"{path}: VASP 4 POSCAR without a species line is ambiguous; "
                          "add the element symbols on line 6.")
    counts = [int(v) for v in lines[6].split()]
    idx = 7
    if lines[idx].strip().lower().startswith("s"):
        idx += 1
    mode = lines[idx].strip().lower()
    idx += 1
    numbers, coords = [], []
    for sym, n in zip(species, counts):
        for _ in range(n):
            parts = lines[idx].split()
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            numbers.append(pt.atomic_number(sym))
            idx += 1
    cell = Cell(m, (True, True, True))
    pos = (cell.to_cartesian(np.array(coords)) if mode.startswith("d")
           else np.array(coords) * scale)
    s = Structure(numbers, pos, cell)
    s.info["provenance"] = _import_provenance(path, "poscar")
    return s


def write_pdb(path: str, structure: Structure) -> str:
    lines = [f"REMARK  Generated by Materia {__version__}",
             "REMARK  Coordinates in angstrom."]
    if structure.cell.is_periodic:
        a, b, c = structure.cell.lengths
        al, be, ga = structure.cell.angles_deg
        lines.append(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
                     f"{al:7.2f}{be:7.2f}{ga:7.2f} P 1           1")
    for k in range(len(structure)):
        sym = pt.symbol(int(structure.numbers[k]))
        x, y, z = structure.positions[k]
        serial = (k + 1) % 100000
        rec = [" "] * 80
        def put(start, text):
            rec[start - 1:start - 1 + len(text)] = list(text)
        put(1, "HETATM")
        put(7, f"{serial:>5d}")
        put(13, f"{sym:<4s}")
        put(18, "MOL")
        put(22, "A")
        put(23, f"{1:>4d}")
        put(31, f"{x:>8.3f}")
        put(39, f"{y:>8.3f}")
        put(47, f"{z:>8.3f}")
        put(55, f"{1.0:>6.2f}")
        put(61, f"{0.0:>6.2f}")
        put(77, f"{sym:>2s}")
        lines.append("".join(rec).rstrip())
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_pdb(path: str) -> Structure:
    numbers, pos = [], []
    cell = Cell.none()
    for line in open(path):
        if line.startswith("CRYST1"):
            a, b, c = float(line[6:15]), float(line[15:24]), float(line[24:33])
            al, be, ga = float(line[33:40]), float(line[40:47]), float(line[47:54])
            cell = Cell.from_parameters(a, b, c, al, be, ga)
        elif line.startswith(("ATOM", "HETATM")):
            sym = line[76:78].strip() or re.sub(r"[^A-Za-z]", "", line[12:16].strip())[:2]
            numbers.append(pt.atomic_number(sym.capitalize()))
            pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    s = Structure(numbers, np.array(pos), cell)
    s.info["provenance"] = _import_provenance(path, "pdb")
    return s


def write_lammps_data(path: str, structure: Structure, atom_style: str = "atomic") -> str:
    if atom_style not in ("atomic", "charge"):
        raise FormatError(f"atom_style {atom_style!r} not supported; use 'atomic' or 'charge'")
    m = structure.cell.matrix
    if not structure.cell.is_periodic:
        lo = structure.positions.min(axis=0) - 10.0
        hi = structure.positions.max(axis=0) + 10.0
        m = np.diag(hi - lo)
        origin = lo
    else:
        origin = np.zeros(3)
    xlo, ylo, zlo = origin
    xhi, yhi, zhi = origin + np.array([m[0][0], m[1][1], m[2][2]])
    xy, xz, yz = m[1][0], m[2][0], m[2][1]
    types = sorted({int(z) for z in structure.numbers})
    tmap = {z: i + 1 for i, z in enumerate(types)}
    lines = [
        f"LAMMPS data file written by Materia {__version__}; units metal (A, eV)",
        "",
        f"{len(structure)} atoms",
        f"{len(types)} atom types",
        "",
        f"{xlo:.10f} {xhi:.10f} xlo xhi",
        f"{ylo:.10f} {yhi:.10f} ylo yhi",
        f"{zlo:.10f} {zhi:.10f} zlo zhi",
    ]
    if abs(xy) > 1e-10 or abs(xz) > 1e-10 or abs(yz) > 1e-10:
        lines.append(f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz")
    lines += ["", "Masses", ""]
    for z in types:
        lines.append(f"{tmap[z]} {pt.mass(z):.6f}  # {pt.symbol(z)}")
    lines += ["", "Atoms # " + atom_style, ""]
    for k in range(len(structure)):
        z = int(structure.numbers[k])
        x, y, zz = structure.positions[k]
        if atom_style == "charge":
            lines.append(f"{k + 1} {tmap[z]} {structure.formal_charges[k]:.6f} "
                         f"{x:.8f} {y:.8f} {zz:.8f}")
        else:
            lines.append(f"{k + 1} {tmap[z]} {x:.8f} {y:.8f} {zz:.8f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_lammps_data(path: str) -> Structure:
    lines = open(path).read().splitlines()
    bounds: Dict[str, Tuple[float, float]] = {}
    tilt = (0.0, 0.0, 0.0)
    masses: Dict[int, float] = {}
    section = None
    atoms: List[Tuple[int, int, float, float, float]] = []
    style = "atomic"
    for line in lines:
        st = line.split("#")[0].strip()
        if not st:
            continue
        low = st.lower()
        if low.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            parts = st.split()
            bounds[parts[2][0]] = (float(parts[0]), float(parts[1]))
            continue
        if low.endswith("xy xz yz"):
            parts = st.split()
            tilt = (float(parts[0]), float(parts[1]), float(parts[2]))
            continue
        if low in ("masses", "atoms", "velocities") or low.startswith("atoms"):
            section = low.split()[0]
            if "#" in line:
                style = line.split("#")[1].strip() or "atomic"
            continue
        if section == "masses":
            parts = st.split()
            masses[int(parts[0])] = float(parts[1])
        elif section == "atoms":
            parts = st.split()
            t = int(parts[1])
            if style.startswith("charge") and len(parts) >= 6:
                atoms.append((int(parts[0]), t, float(parts[3]), float(parts[4]), float(parts[5])))
            else:
                atoms.append((int(parts[0]), t, float(parts[2]), float(parts[3]), float(parts[4])))
    if not atoms:
        raise FormatError(f"{path}: no Atoms section found")
    lx = bounds["x"][1] - bounds["x"][0]
    ly = bounds["y"][1] - bounds["y"][0]
    lz = bounds["z"][1] - bounds["z"][0]
    m = np.array([[lx, 0, 0], [tilt[0], ly, 0], [tilt[1], tilt[2], lz]])
    type_to_z = {}
    for t, mass in masses.items():
        best = min(pt.ELEMENTS, key=lambda e: abs((e.standard_atomic_weight or 1e9) - mass))
        type_to_z[t] = best.number
    atoms.sort(key=lambda a: a[0])
    numbers = [type_to_z.get(a[1], 1) for a in atoms]
    pos = np.array([[a[2], a[3], a[4]] for a in atoms])
    s = Structure(numbers, pos, Cell(m, (True, True, True)))
    s.info["provenance"] = _import_provenance(path, "lammps-data")
    s.info["note"] = ("Element identity recovered from the Masses section by "
                      "nearest standard atomic weight.")
    return s


BOHR_PER_A = 1.0 / 0.529177210903


def write_cube(path: str, structure: Structure, data: np.ndarray,
               origin_A: Sequence[float] = (0.0, 0.0, 0.0),
               spacing_A: Optional[np.ndarray] = None,
               comment: str = "volumetric data") -> str:
    """Gaussian CUBE file.  Coordinates and the grid are converted to bohr."""
    d = np.asarray(data, dtype=float)
    if d.ndim != 3:
        raise FormatError("CUBE data must be a 3-D array")
    nx, ny, nz = d.shape
    if spacing_A is None:
        m = structure.cell.matrix
        spacing = np.array([m[0] / nx, m[1] / ny, m[2] / nz])
    else:
        spacing = np.asarray(spacing_A, dtype=float)
        if spacing.ndim == 1:
            spacing = np.diag(spacing)
    o = np.asarray(origin_A, dtype=float) * BOHR_PER_A
    lines = [f"Materia {__version__} CUBE: {comment}",
             "Values in file units; lengths converted to bohr."]
    lines.append(f"{len(structure):5d} {o[0]:12.6f} {o[1]:12.6f} {o[2]:12.6f}")
    for i, n in enumerate((nx, ny, nz)):
        v = spacing[i] * BOHR_PER_A
        lines.append(f"{n:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}")
    for k in range(len(structure)):
        z = int(structure.numbers[k])
        p = structure.positions[k] * BOHR_PER_A
        lines.append(f"{z:5d} {float(z):12.6f} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}")
    flat = d.ravel(order="C")
    for i in range(0, len(flat), 6):
        lines.append("".join(f"{v:13.5e}" for v in flat[i:i + 6]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_csv(path: str, columns: Dict[str, Sequence[Any]], header_comment: str = "") -> str:
    keys = list(columns)
    n = len(columns[keys[0]]) if keys else 0
    with open(path, "w") as fh:
        if header_comment:
            for line in header_comment.splitlines():
                fh.write(f"# {line}\n")
        fh.write(f"# Materia {__version__}\n")
        fh.write(",".join(keys) + "\n")
        for i in range(n):
            fh.write(",".join(str(columns[k][i]) for k in keys) + "\n")
    return path


def structure_table(structure: Structure) -> Dict[str, List[Any]]:
    cols: Dict[str, List[Any]] = {
        "id": [], "element": [], "atomic_number": [], "mass_number": [],
        "x_A": [], "y_A": [], "z_A": [], "formal_charge_e": [],
        "partial_charge_e": [], "magnetic_moment_muB": [], "role": [],
        "label": [], "coordination": [], "fixed": [],
    }
    for k in range(len(structure)):
        aid = int(structure.ids[k])
        cols["id"].append(aid)
        cols["element"].append(pt.symbol(int(structure.numbers[k])))
        cols["atomic_number"].append(int(structure.numbers[k]))
        cols["mass_number"].append(int(structure.mass_numbers[k]) or "natural")
        cols["x_A"].append(f"{structure.positions[k][0]:.6f}")
        cols["y_A"].append(f"{structure.positions[k][1]:.6f}")
        cols["z_A"].append(f"{structure.positions[k][2]:.6f}")
        cols["formal_charge_e"].append(f"{structure.formal_charges[k]:.6f}")
        cols["partial_charge_e"].append(f"{structure.partial_charges[k]:.6f}")
        cols["magnetic_moment_muB"].append(f"{structure.magnetic_moments[k]:.6f}")
        cols["role"].append(str(structure.roles[k]))
        cols["label"].append(str(structure.labels[k]))
        cols["coordination"].append(len(structure.neighbors_of(aid)))
        cols["fixed"].append(bool(structure.fixed[k]))
    return cols


def write_hdf5(path: str, arrays: Dict[str, np.ndarray],
               attributes: Optional[dict] = None) -> str:
    try:
        import h5py
    except ImportError as exc:
        raise FormatError(
            "HDF5 export needs the optional 'h5py' package. "
            "Install it with: pip install h5py"
        ) from exc
    with h5py.File(path, "w") as f:
        for key, arr in arrays.items():
            f.create_dataset(key, data=np.asarray(arr), compression="gzip")
        f.attrs["generator"] = f"Materia {__version__}"
        for k, v in (attributes or {}).items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    return path


def read_hdf5(path: str) -> Dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:
        raise FormatError("HDF5 import needs the optional 'h5py' package.") from exc
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: out.__setitem__(name, np.array(obj))
                     if hasattr(obj, "shape") else None)
    return out


_WRITERS = {
    "xyz": write_xyz, "extxyz": write_extxyz, "cif": write_cif,
    "poscar": write_poscar, "pdb": write_pdb, "lammps-data": write_lammps_data,
}
_READERS = {
    "xyz": read_xyz, "extxyz": read_xyz, "cif": read_cif,
    "poscar": read_poscar, "pdb": read_pdb, "lammps-data": read_lammps_data,
}


def write_structure(path: str, structure: Structure, fmt: Optional[str] = None, **kwargs) -> str:
    fmt = fmt or detect_format(path)
    if fmt not in _WRITERS:
        raise FormatError(
            f"Cannot write structures as {fmt!r}. Supported: {sorted(_WRITERS)}. "
            f"For volumetric data use write_cube; for tables use write_csv."
        )
    return _WRITERS[fmt](path, structure, **kwargs)


def read_structure(path: str, fmt: Optional[str] = None) -> Structure:
    fmt = fmt or detect_format(path)
    if fmt not in _READERS:
        raise FormatError(
            f"Cannot read {fmt!r} structures. Supported: {sorted(_READERS)}."
        )
    return _READERS[fmt](path)
