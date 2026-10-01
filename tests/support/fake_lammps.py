"""A deterministic stand-in for LAMMPS used by the adapter tests.

It understands the subset of the LAMMPS input language that Materia writes,
evaluates energies and forces with Materia's own EAM implementation in LAMMPS
``metal`` units, and writes a log and custom dumps in the formats LAMMPS
uses.  It is not LAMMPS and proves nothing about LAMMPS's physics; it exists
so that the adapter's input writing, process handling, parsing, unit
conversion, validation and project bookkeeping can be tested exactly.

``FAKE_LAMMPS_MODE`` holds comma-separated failure modes; see ``MODES``.
``FAKE_LAMMPS_VERSION`` and ``FAKE_LAMMPS_PACKAGES`` change what it reports.
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

MODES = {
    "exit1", "error_line", "truncate", "truncate_dump", "truncate_trajectory", "nan",
    "nan_force", "lose_atom", "lose_atom_dump_only", "change_id", "change_type",
    "bad_mass", "version_log", "sleep", "maxiter", "bad_ke", "no_log",
    "malformed_thermo", "hang", "ignore_mass_command",
}

MVV2E = 1.0364269e-4
FTM2V = 1.0 / MVV2E
NKTV2P = 1.6021765e6
BOLTZ = 8.617343e-5
KEYWORD_HEADERS = {"step": "Step", "time": "Time", "pe": "PotEng", "ke": "KinEng",
                   "etotal": "TotEng", "temp": "Temp", "press": "Press", "pxx": "Pxx",
                   "pyy": "Pyy", "pzz": "Pzz", "pxy": "Pxy", "pxz": "Pxz", "pyz": "Pyz",
                   "atoms": "Atoms"}


def version() -> str:
    return os.environ.get("FAKE_LAMMPS_VERSION", "29 Aug 2024 - Update 1")


def modes() -> set:
    return {m.strip() for m in os.environ.get("FAKE_LAMMPS_MODE", "").split(",") if m.strip()}


def packages() -> list:
    return os.environ.get("FAKE_LAMMPS_PACKAGES", "MANYBODY KSPACE").split()


def styles() -> dict:
    pair = ["lj/cut", "morse"]
    if "MANYBODY" in packages():
        pair += ["eam", "eam/alloy", "eam/fs"]
    return {"atom": ["atomic", "charge", "full"], "integrate": ["verlet"],
            "minimize": ["cg", "fire", "sd"], "pair": pair,
            "fix": ["langevin", "nve", "nvt", "setforce"], "compute": ["pe", "temp"],
            "command": ["run", "minimize", "write_dump"]}


def help_text() -> str:
    lines = ["", "Large-scale Atomic/Molecular Massively Parallel Simulator - " + version(),
             "Git info (stable / fake)", "",
             "Usage example: lmp -var t 300 -echo screen -in in.alloy", "",
             "Installed packages:", "", " ".join(packages()), "",
             "List of individual style options included in this LAMMPS executable", ""]
    titles = {"atom": "Atom styles:", "integrate": "Integrate styles:",
              "minimize": "Minimize styles:", "pair": "Pair styles:", "fix": "Fix styles",
              "compute": "Compute styles:", "command": "Command styles"}
    for category, names in styles().items():
        lines += [f"* {titles[category]}", "", "  ".join(names), ""]
    return "\n".join(lines) + "\n"


class Fail(Exception):
    pass


class Stop(Exception):
    pass


class Engine:
    def __init__(self, cmdargs):
        args = list(cmdargs or [])
        self.modes = modes()
        self.truncated = False
        self.log_path = None
        self.screen = True
        self.echo_log = False
        index = 0
        while index < len(args):
            arg = args[index]
            if arg in ("-log", "-screen", "-echo") and index + 1 < len(args):
                value = args[index + 1]
                if arg == "-log":
                    self.log_path = None if value == "none" else value
                elif arg == "-screen":
                    self.screen = value != "none"
                else:
                    self.echo_log = value in ("log", "both")
                index += 2
                continue
            index += 1
        if self.log_path is None and "-log" not in args:
            self.log_path = "log.lammps"
        if "no_log" in self.modes:
            self.log_path = None
        self.log = open(self.log_path, "w") if self.log_path else None
        header_version = "1 Jan 2000" if "version_log" in self.modes else version()
        self.out(f"LAMMPS ({header_version})")
        self.boundary = ["p", "p", "p"]
        self.types = {}
        self.masses_by_type = {}
        self.thermo_every = 0
        self.fixes = {}
        self.groups = {}
        self.dumps = {}
        self.dt_ps = 0.001
        self.step = 0
        self.truncated = False
        self.potential = None
        self.elements_by_type = {}
        self.min_style = "cg"

    def out(self, text, screen=True):
        if self.log is not None and not self.truncated:
            self.log.write(text + "\n")
            self.log.flush()
        if screen and self.screen and not self.truncated:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def close(self):
        if self.log is not None:
            if not self.truncated:
                self.log.write("Total wall time: 0:00:00\n")
            self.log.close()
            self.log = None

    def file(self, path):
        for raw in Path(path).read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            if self.echo_log and self.log is not None and not self.truncated:
                self.log.write(line + "\n")
            self.command(line)

    def command(self, line):
        words = line.split()
        name, args = words[0], words[1:]
        handler = getattr(self, "cmd_" + name, None)
        if handler is None:
            raise Fail(f"Unknown command: {line}")
        handler(args, line)

    def cmd_units(self, args, line):
        if args[0] != "metal":
            raise Fail("fake supports metal units only")

    def cmd_atom_style(self, args, line):
        if args[0] not in styles()["atom"]:
            raise Fail("unknown atom style")

    def cmd_boundary(self, args, line):
        self.boundary = list(args)

    def cmd_read_data(self, args, line):
        lines = Path(args[0]).read_text().splitlines()
        counts = {}
        lo = np.zeros(3)
        hi = np.zeros(3)
        tilt = np.zeros(3)
        index = 1
        section = None
        atoms, velocities = [], {}
        while index < len(lines):
            text = lines[index].strip()
            index += 1
            if not text:
                continue
            if text in ("Masses", "Atoms", "Velocities"):
                section = text
                continue
            parts = text.split()
            if section is None:
                if parts[-1] == "atoms":
                    counts["atoms"] = int(parts[0])
                elif parts[-2:] == ["atom", "types"]:
                    counts["types"] = int(parts[0])
                elif parts[-2:] == ["xlo", "xhi"]:
                    lo[0], hi[0] = float(parts[0]), float(parts[1])
                elif parts[-2:] == ["ylo", "yhi"]:
                    lo[1], hi[1] = float(parts[0]), float(parts[1])
                elif parts[-2:] == ["zlo", "zhi"]:
                    lo[2], hi[2] = float(parts[0]), float(parts[1])
                elif parts[-3:] == ["xy", "xz", "yz"]:
                    tilt = np.array([float(p) for p in parts[:3]])
            elif section == "Masses":
                self.masses_by_type[int(parts[0])] = float(parts[1])
            elif section == "Atoms":
                atoms.append(parts)
            elif section == "Velocities":
                velocities[int(parts[0])] = [float(p) for p in parts[1:4]]
        self.lo, self.hi, self.tilt = lo, hi, tilt
        lx, ly, lz = hi - lo
        self.matrix = np.array([[lx, 0, 0], [tilt[0], ly, 0], [tilt[1], tilt[2], lz]])
        self.ids = np.array([int(a[0]) for a in atoms], dtype=np.int64)
        self.atom_types = np.array([int(a[1]) for a in atoms], dtype=np.int64)
        wrapped = np.array([[float(v) for v in a[2:5]] for a in atoms])
        images = np.array([[int(v) for v in a[5:8]] for a in atoms], dtype=float)
        basis = self.matrix.copy()
        for axis in range(3):
            if self.boundary[axis] != "p":
                basis[axis] = 0.0
        self.x = wrapped + images @ basis
        self.v = np.array([velocities.get(int(i), [0.0, 0.0, 0.0]) for i in self.ids])
        self.f = np.zeros_like(self.x)
        self.out(f"Reading data file ...\n  {len(self.ids)} atoms")

    def cmd_pair_style(self, args, line):
        if args[0] not in styles()["pair"]:
            raise Fail(f"Unrecognized pair style '{args[0]}'")
        self.pair_style = args[0]

    def cmd_pair_coeff(self, args, line):
        from materia.physics.eam import load_file

        self.potential = load_file(args[2])
        for type_id, element in enumerate(args[3:], start=1):
            self.elements_by_type[type_id] = element
            k = self.potential.elements.index(element)
            self.masses_by_type[type_id] = self.potential.table.masses[k]

    def cmd_mass(self, args, line):
        if "ignore_mass_command" in self.modes:
            return
        self.masses_by_type[int(args[0])] = float(args[1])

    def cmd_neighbor(self, args, line):
        pass

    def cmd_neigh_modify(self, args, line):
        pass

    def cmd_thermo_style(self, args, line):
        self.keywords = args[1:]

    def cmd_thermo_modify(self, args, line):
        pass

    def cmd_thermo(self, args, line):
        self.thermo_every = int(args[0])

    def cmd_group(self, args, line):
        name = args[0]
        if args[1] == "id":
            ids = set()
            for token in args[2:]:
                if ":" in token:
                    a, b = token.split(":")
                    ids.update(range(int(a), int(b) + 1))
                else:
                    ids.add(int(token))
            self.groups[name] = np.isin(self.ids, list(ids))
        elif args[1] == "subtract":
            mask = self.group(args[2]).copy()
            for other in args[3:]:
                mask &= ~self.group(other)
            self.groups[name] = mask

    def group(self, name):
        if name == "all":
            return np.ones(len(self.ids), dtype=bool)
        return self.groups[name]

    def cmd_min_style(self, args, line):
        self.min_style = args[0]

    def cmd_min_modify(self, args, line):
        pass

    def cmd_fix(self, args, line):
        self.fixes[args[0]] = {"group": args[1], "style": args[2], "args": args[3:]}

    def cmd_unfix(self, args, line):
        del self.fixes[args[0]]

    def cmd_timestep(self, args, line):
        self.dt_ps = float(args[0])

    def cmd_velocity(self, args, line):
        group = self.group(args[0])
        temperature, seed = float(args[2]), int(args[3])
        rng = np.random.default_rng(seed)
        masses = self.masses()
        v = rng.normal(size=(len(self.ids), 3)) * np.sqrt(
            BOLTZ * temperature / (masses * MVV2E))[:, None]
        v[~group] = 0.0
        momentum = (masses[group, None] * v[group]).sum(axis=0) / masses[group].sum()
        v[group] -= momentum
        dof = max(1, 3 * int(group.sum()) - 3)
        current = 2 * 0.5 * MVV2E * np.sum(masses[:, None] * v * v) / (dof * BOLTZ)
        if current > 0:
            v *= math.sqrt(temperature / current)
        self.v = v

    def cmd_dump(self, args, line):
        self.dumps[args[0]] = {"every": int(args[3]), "file": args[4], "columns": args[5:],
                               "frames": []}

    def cmd_dump_modify(self, args, line):
        pass

    def cmd_undump(self, args, line):
        dump = self.dumps.pop(args[0])
        frames = dump["frames"][:-1] if "truncate_trajectory" in self.modes else dump["frames"]
        Path(dump["file"]).write_text("".join(frames))

    def cmd_print(self, args, line):
        text = line.split(None, 1)[1].strip().strip('"')
        if text == "@@MATERIA section final":
            self.final_section = True
            if "truncate" in self.modes:
                self.out(text)
                self.truncated = True
                raise Stop()
        self.out(text)

    def masses(self):
        return np.array([self.masses_by_type[int(t)] for t in self.atom_types])

    def structure(self):
        from materia.core_model.cell import Cell
        from materia.core_model.structure import Structure
        from materia.elements import periodic_table as pt

        numbers = [pt.element(self.elements_by_type[int(t)]).number for t in self.atom_types]
        pbc = tuple(b == "p" for b in self.boundary)
        return Structure(numbers, self.x, Cell(self.matrix, pbc), ids=list(self.ids))

    def compute(self):
        result = self.potential.evaluate(self.structure())
        self.pe = float(result["energy_eV"])
        self.f = np.array(result["forces_eV_A"])
        self.sigma = result["stress_eV_A3"]
        for fix in self.fixes.values():
            if fix["style"] == "setforce":
                self.f[self.group(fix["group"])] = 0.0

    def ke(self):
        return 0.5 * MVV2E * float(np.sum(self.masses()[:, None] * self.v * self.v))

    def row(self, atoms=None):
        n = len(self.ids)
        ke = self.ke()
        dof = 3 * n - 3
        temp = 2.0 * ke / (dof * BOLTZ) if dof > 0 else 0.0
        if self.sigma is not None:
            volume = abs(np.linalg.det(self.matrix))
            kinetic = MVV2E * np.einsum("i,ia,ib->ab", self.masses(), self.v, self.v) / volume
            pressure = (kinetic - np.asarray(self.sigma)) * NKTV2P
        else:
            pressure = np.zeros((3, 3))
        pe = float("nan") if "nan" in self.modes and self.final_section else self.pe
        values = {"step": self.step, "time": self.step * self.dt_ps, "pe": pe, "ke": ke,
                  "etotal": pe + ke, "temp": temp, "press": float(np.trace(pressure) / 3),
                  "pxx": pressure[0, 0], "pyy": pressure[1, 1], "pzz": pressure[2, 2],
                  "pxy": pressure[0, 1], "pxz": pressure[0, 2], "pyz": pressure[1, 2],
                  "atoms": n if atoms is None else atoms}
        parts = []
        for keyword in self.keywords:
            value = values[keyword]
            parts.append(str(int(value)) if keyword in ("step", "atoms") else
                         format(float(value), ".17g"))
        if "malformed_thermo" in self.modes and self.final_section:
            parts.append("extra")
        self.out("   " + " ".join(parts))

    def header(self):
        self.out("   " + " ".join(KEYWORD_HEADERS[k] for k in self.keywords))

    final_section = False

    def check_failures(self):
        if "exit1" in self.modes:
            self.out("ERROR: Fake failure requested (src/fake.cpp:1)")
            raise SystemExit(1)
        if "error_line" in self.modes:
            self.out("ERROR: Fake error line with a clean exit (src/fake.cpp:2)")
        if "hang" in self.modes:
            time.sleep(3600)

    def cmd_run(self, args, line):
        steps = int(args[0])
        self.check_failures()
        self.compute()
        self.header()
        atoms = len(self.ids) - 1 if ("lose_atom" in self.modes and self.final_section) else None
        if steps == 0:
            self.row(atoms)
            self.write_dumps(force=True)
            self.out("Loop time of 0.001 on 1 procs for 0 steps with "
                     f"{len(self.ids)} atoms")
            return
        every = self.thermo_every or steps
        self.row()
        self.write_dumps(force=True)
        integrate = next((f for f in self.fixes.values() if f["style"] in ("nve", "nvt")), None)
        mobile = self.group(integrate["group"]) if integrate else np.zeros(len(self.ids), bool)
        thermostat = next((f for f in self.fixes.values() if f["style"] == "langevin"), None)
        rng = np.random.default_rng(int(thermostat["args"][3]) if thermostat else 0)
        masses = self.masses()[:, None]
        dt = self.dt_ps
        for _ in range(steps):
            if "sleep" in self.modes:
                time.sleep(0.05)
            self.v[mobile] += 0.5 * dt * self.f[mobile] / masses[mobile] * FTM2V
            self.x[mobile] += dt * self.v[mobile]
            self.compute()
            self.v[mobile] += 0.5 * dt * self.f[mobile] / masses[mobile] * FTM2V
            if thermostat is not None:
                target = float(thermostat["args"][0])
                damp = float(thermostat["args"][2])
                c1 = math.exp(-dt / damp)
                sigma = np.sqrt(BOLTZ * target / (masses[mobile] * MVV2E) * (1 - c1 * c1))
                self.v[mobile] = c1 * self.v[mobile] + sigma * rng.normal(
                    size=self.v[mobile].shape)
            if integrate is not None and integrate["style"] == "nvt":
                target = float(integrate["args"][1])
                damp = float(integrate["args"][3])
                ke = self.ke()
                dof = 3 * len(self.ids) - 3
                current = 2 * ke / (dof * BOLTZ) if dof > 0 else 0.0
                if current > 0:
                    self.v *= math.sqrt(max(0.0, 1 + dt / damp * (target / current - 1)))
            self.step += 1
            if self.step % every == 0 or _ == steps - 1:
                self.row()
            self.write_dumps()
        self.out(f"Loop time of 0.01 on 1 procs for {steps} steps with {len(self.ids)} atoms")

    def write_dumps(self, force=False):
        for dump in self.dumps.values():
            if self.step % dump["every"] == 0:
                buffer = io.StringIO()
                self.write_frame(buffer, dump["columns"])
                dump["frames"].append(buffer.getvalue())

    def write_frame(self, handle, columns, final=False):
        order = np.argsort(self.ids)
        rows = []
        masses = self.masses()
        v = self.v * (1000.0 if ("bad_ke" in self.modes and final) else 1.0)
        for k in order:
            values = {"id": int(self.ids[k]), "type": int(self.atom_types[k]),
                      "mass": masses[k] * (1.01 if "bad_mass" in self.modes and final else 1),
                      "xu": self.x[k, 0], "yu": self.x[k, 1], "zu": self.x[k, 2],
                      "vx": v[k, 0], "vy": v[k, 1], "vz": v[k, 2],
                      "fx": self.f[k, 0], "fy": self.f[k, 1], "fz": self.f[k, 2]}
            rows.append(values)
        if final and "change_id" in self.modes:
            rows[0]["id"] += 1000
        if final and "change_type" in self.modes:
            rows[0]["type"] = rows[0]["type"] % max(self.masses_by_type) + 1 \
                if len(self.masses_by_type) > 1 else rows[0]["type"] + 1
        if final and "nan_force" in self.modes:
            rows[0]["fx"] = float("nan")
        if final and ("lose_atom" in self.modes or "lose_atom_dump_only" in self.modes):
            rows = rows[1:]
        pbc = "".join(" pp" if b == "p" else " ss" for b in self.boundary)
        triclinic = bool(np.any(self.tilt != 0))
        handle.write("ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\n" % (self.step, len(rows)))
        handle.write(("ITEM: BOX BOUNDS xy xz yz" if triclinic else "ITEM: BOX BOUNDS")
                     + pbc + "\n")
        for axis in range(3):
            bounds = f"{float(self.lo[axis])!r} {float(self.hi[axis])!r}"
            if triclinic:
                bounds += f" {float(self.tilt[axis])!r}"
            handle.write(bounds + "\n")
        handle.write("ITEM: ATOMS " + " ".join(columns) + "\n")
        lines = []
        for values in rows:
            lines.append(" ".join(str(values[c]) if c in ("id", "type") else
                                  format(float(values[c]), ".17g") for c in columns))
        if final and "truncate_dump" in self.modes:
            lines = lines[:-1]
        handle.write("\n".join(lines) + ("\n" if lines else ""))
        handle.flush()

    def cmd_minimize(self, args, line):
        ftol, maxiter = float(args[1]), int(args[2])
        if "maxiter" in self.modes:
            maxiter = 2
        self.check_failures()
        self.compute()
        self.header()
        energy_initial = self.pe
        every = self.thermo_every or 1
        self.row()
        mobile = np.ones(len(self.ids), dtype=bool)
        freeze = next((f for f in self.fixes.values() if f["style"] == "setforce"), None)
        if freeze is not None:
            mobile = ~self.group(freeze["group"])
        velocity = np.zeros_like(self.x)
        dt, alpha, positive = 0.05, 0.1, 0
        criterion = "max iterations"
        iterations = 0
        previous = self.pe
        for iteration in range(1, maxiter + 1):
            power = float(np.sum(self.f * velocity))
            if power > 0:
                vnorm = np.linalg.norm(velocity)
                fnorm = np.linalg.norm(self.f)
                velocity = (1 - alpha) * velocity + alpha * vnorm * self.f / max(fnorm, 1e-300)
                positive += 1
                if positive > 5:
                    dt = min(dt * 1.1, 0.2)
                    alpha *= 0.99
            else:
                velocity[:] = 0.0
                dt *= 0.5
                alpha = 0.1
                positive = 0
            velocity += dt * self.f
            velocity[~mobile] = 0.0
            self.x += dt * velocity
            previous = self.pe
            self.compute()
            self.step += 1
            iterations = iteration
            fmax = float(np.linalg.norm(self.f[mobile], axis=1).max())
            done = fmax <= ftol
            if self.step % every == 0 or done or iteration == maxiter:
                self.row()
            if done:
                criterion = "force tolerance"
                break
        self.out(f"Loop time of 0.01 on 1 procs for {iterations} steps with "
                 f"{len(self.ids)} atoms")
        self.out("")
        self.out("Minimization stats:")
        self.out(f"  Stopping criterion = {criterion}")
        self.out("  Energy initial, next-to-last, final = ")
        self.out(f"    {float(energy_initial)!r}   {float(previous)!r}   {float(self.pe)!r}")
        self.out(f"  Force two-norm initial, final = 1.0 {float(np.linalg.norm(self.f))!r}")
        self.out("  Force max component initial, final = 1.0 "
                 f"{float(np.abs(self.f).max())!r}")
        self.out("  Final line search alpha, max atom move = 0.0 0.0")
        self.out(f"  Iterations, force evaluations = {iterations} {iterations + 1}")
        self.out("")
        self.v[:] = 0.0

    def cmd_write_dump(self, args, line):
        columns = []
        for token in args[3:]:
            if token == "modify":
                break
            columns.append(token)
        with open(args[2], "w") as handle:
            self.write_frame(handle, columns, final=True)


def run_input(cmdargs):
    arguments = list(cmdargs)
    position = arguments.index("-in")
    input_file = arguments[position + 1]
    del arguments[position:position + 2]
    engine = Engine(arguments)
    try:
        engine.file(input_file)
    except Stop:
        engine.close()
        return 0
    except Fail as exc:
        engine.out(f"ERROR: {exc}")
        engine.close()
        return 1
    engine.close()
    return 0


class ModuleFacade:
    """What ``from lammps import lammps`` gives, for the Python-module route."""

    def __init__(self, cmdargs=None, **kwargs):
        self.engine = Engine(list(cmdargs or []))
        self.installed_packages = packages()

    def version(self):
        from materia.solvers.lammps.environment import version_id

        return version_id(version())

    def available_styles(self, category):
        return styles().get(category, [])

    def has_style(self, category, name):
        return name in styles().get(category, [])

    def file(self, path):
        try:
            self.engine.file(path)
        except Stop:
            return
        except Fail as exc:
            self.engine.out(f"ERROR: {exc}")
            raise RuntimeError(str(exc)) from None

    def close(self):
        self.engine.close()


def main(argv) -> int:
    if argv[:1] == ["-h"]:
        sys.stdout.write(help_text())
        return 0
    if "-in" not in argv:
        sys.stderr.write("fake lammps needs -in\n")
        return 2
    return run_input(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
