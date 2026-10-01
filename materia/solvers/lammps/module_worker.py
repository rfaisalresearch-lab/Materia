"""Run one LAMMPS input file through the ``lammps`` Python module.

Started by :mod:`materia.solvers.lammps.runner` as its own process when the
LAMMPS found is a Python module rather than an executable.  It takes the same
command-line arguments as the ``lmp`` executable, so the input, log and
output files are identical whichever route was used.
"""

from __future__ import annotations

import sys


def main(argv) -> int:
    arguments = list(argv)
    if "-in" not in arguments or arguments.index("-in") + 1 >= len(arguments):
        print("module_worker: -in <file> is required", file=sys.stderr)
        return 2
    position = arguments.index("-in")
    input_file = arguments[position + 1]
    del arguments[position:position + 2]
    from lammps import lammps

    instance = lammps(cmdargs=arguments)
    try:
        instance.file(input_file)
    finally:
        instance.close()
    return 0


if __name__ == "__main__":
    try:
        code = main(sys.argv[1:])
    except BaseException as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        code = 1
    sys.stdout.flush()
    raise SystemExit(code)
