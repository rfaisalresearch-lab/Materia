"""Reading what LAMMPS wrote: the log, thermodynamic tables and custom dumps.

The reader is strict.  A table row with the wrong number of columns, a dump
frame with fewer atom lines than it declares, a missing ``ITEM`` header or a
token that is not a number is an error, not something to skip.  Values are
returned exactly as written; unit conversion happens in
:mod:`materia.solvers.lammps.units`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .environment import version_from_log_header
from .native import COMPLETE, SECTION_FINAL, SECTION_MAIN, THERMO_KEYWORDS


class OutputError(ValueError):
    """Output that is missing, truncated, malformed or inconsistent."""


@dataclass
class ThermoTable:
    columns: Tuple[str, ...]
    rows: np.ndarray

    def column(self, keyword: str) -> np.ndarray:
        return self.rows[:, THERMO_KEYWORDS.index(keyword)]


@dataclass
class ParsedLog:
    version: str = ""
    sections: Dict[str, ThermoTable] = field(default_factory=dict)
    minimization: Optional[Dict[str, object]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    complete: bool = False
    total_wall_time: str = ""


def _floats(tokens: Sequence[str], what: str) -> List[float]:
    try:
        return [float(t) for t in tokens]
    except ValueError:
        raise OutputError(f"{what} contains a token that is not a number: "
                          f"{' '.join(tokens)[:200]!r}.") from None


def parse_log(text: str) -> ParsedLog:
    out = ParsedLog()
    lines = text.splitlines()
    for line in lines:
        version = version_from_log_header(line)
        if version:
            out.version = version
            break
    section: Optional[str] = None
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        index += 1
        if line.startswith("ERROR"):
            out.errors.append(line[:500])
            continue
        if line.startswith("WARNING"):
            out.warnings.append(line[:500])
            continue
        if line == SECTION_MAIN:
            section = "main"
            continue
        if line == SECTION_FINAL:
            section = "final"
            continue
        if line == COMPLETE:
            out.complete = True
            continue
        if line.startswith("Total wall time:"):
            out.total_wall_time = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Minimization stats:"):
            out.minimization, index = _minimization(lines, index)
            continue
        tokens = line.split()
        if tokens and tokens[0] == "Step" and section is not None:
            if section in out.sections:
                raise OutputError(f"The log has a second thermodynamic table in the {section} "
                                  "section.")
            if len(tokens) != len(THERMO_KEYWORDS):
                raise OutputError(f"The thermodynamic header has {len(tokens)} columns; "
                                  f"{len(THERMO_KEYWORDS)} were requested.")
            rows: List[List[float]] = []
            ended = False
            while index < len(lines):
                row = lines[index].strip()
                index += 1
                if row.startswith("Loop time of"):
                    ended = True
                    break
                if not row:
                    continue
                if row.startswith("WARNING"):
                    out.warnings.append(row[:500])
                    continue
                if row.startswith("ERROR"):
                    out.errors.append(row[:500])
                    break
                parts = row.split()
                if len(parts) != len(THERMO_KEYWORDS):
                    raise OutputError(f"A thermodynamic row in the {section} section has "
                                      f"{len(parts)} values, not {len(THERMO_KEYWORDS)}: "
                                      f"{row[:200]!r}.")
                rows.append(_floats(parts, f"A thermodynamic row in the {section} section"))
            if not ended:
                raise OutputError(f"The thermodynamic table of the {section} section is not "
                                  "terminated by a Loop time line; the log is truncated.")
            if not rows:
                raise OutputError(f"The {section} section printed no thermodynamic rows.")
            out.sections[section] = ThermoTable(tuple(tokens), np.array(rows, dtype=float))
            section = None
    return out


def _minimization(lines: List[str], index: int) -> Tuple[Dict[str, object], int]:
    info: Dict[str, object] = {}
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            if info:
                break
            continue
        if line.startswith("Stopping criterion"):
            info["stopping_criterion"] = line.split("=", 1)[1].strip()
        elif line.startswith("Energy initial, next-to-last, final"):
            values = lines[index].split("=", 1)[1].split()
            if not values and index + 1 < len(lines):
                index += 1
                values = lines[index].split()
            numbers = _floats(values, "The minimisation energy line")
            if len(numbers) != 3:
                raise OutputError("The minimisation energy line does not have three values.")
            info["energy_initial_eV"], info["energy_next_to_last_eV"], \
                info["energy_final_eV"] = numbers
        elif line.startswith("Iterations, force evaluations"):
            numbers = _floats(line.split("=", 1)[1].split(), "The iteration count line")
            if len(numbers) != 2:
                raise OutputError("The iteration count line does not have two values.")
            info["iterations"], info["force_evaluations"] = int(numbers[0]), int(numbers[1])
        elif line.startswith("Force two-norm initial, final"):
            info["force_two_norm"] = _floats(line.split("=", 1)[1].split(),
                                             "The force two-norm line")
        elif line.startswith("Force max component initial, final"):
            info["force_max_component"] = _floats(line.split("=", 1)[1].split(),
                                                  "The force max component line")
        elif line.startswith("Final line search alpha, max atom move"):
            info["final_line_search"] = _floats(line.split("=", 1)[1].split(),
                                                "The line search line")
        elif not line.startswith(("Energy", "Force", "Final", "Iterations", "Stopping")):
            break
        index += 1
    return info, index


@dataclass
class DumpFrame:
    timestep: int
    columns: Tuple[str, ...]
    data: np.ndarray
    box: List[List[float]]

    def column(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]


def parse_dump(text: str, expected_columns: Sequence[str]) -> List[DumpFrame]:
    lines = text.splitlines()
    frames: List[DumpFrame] = []
    index = 0
    expected = tuple(expected_columns)

    def take(what: str) -> str:
        nonlocal index
        if index >= len(lines):
            raise OutputError(f"The dump ends before {what}; it is truncated.")
        value = lines[index]
        index += 1
        return value

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if take("a frame header").strip() != "ITEM: TIMESTEP":
            raise OutputError(f"Expected ITEM: TIMESTEP at dump line {index}.")
        try:
            timestep = int(take("the timestep").strip())
        except ValueError:
            raise OutputError(f"The timestep at dump line {index} is not an integer.") from None
        if take("the atom count header").strip() != "ITEM: NUMBER OF ATOMS":
            raise OutputError(f"Expected ITEM: NUMBER OF ATOMS at dump line {index}.")
        try:
            count = int(take("the atom count").strip())
        except ValueError:
            raise OutputError(f"The atom count at dump line {index} is not an integer.") \
                from None
        if not take("the box header").strip().startswith("ITEM: BOX BOUNDS"):
            raise OutputError(f"Expected ITEM: BOX BOUNDS at dump line {index}.")
        box = [_floats(take("the box bounds").split(), "The box bounds") for _ in range(3)]
        header = take("the atoms header").split()
        if header[:2] != ["ITEM:", "ATOMS"]:
            raise OutputError(f"Expected ITEM: ATOMS at dump line {index}.")
        columns = tuple(header[2:])
        if columns != expected:
            raise OutputError(f"The dump columns are {' '.join(columns)}; "
                              f"{' '.join(expected)} were requested.")
        if index + count > len(lines):
            raise OutputError(f"The dump frame at timestep {timestep} declares {count} atoms "
                              f"but only {len(lines) - index} lines follow; it is truncated.")
        block = lines[index:index + count]
        index += count
        tokens = " ".join(block).split()
        if len(tokens) != count * len(columns):
            raise OutputError(f"The dump frame at timestep {timestep} has {len(tokens)} "
                              f"values for {count} atoms of {len(columns)} columns.")
        try:
            data = np.array(tokens, dtype=float).reshape(count, len(columns))
        except ValueError:
            raise OutputError(f"The dump frame at timestep {timestep} contains a token that "
                              "is not a number.") from None
        frames.append(DumpFrame(timestep, columns, data, box))
    return frames
