"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .version import __version__


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="materia",
        description="Atomic-scale wafer exploration and scanning-probe simulation.")
    parser.add_argument("--version", action="version", version=f"Materia {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_ui = sub.add_parser("ui", help="open the native desktop window")
    p_ui.add_argument("--port", type=int, default=None)
    p_ui.add_argument("--project", default=None, help="open a .materia file")
    p_ui.add_argument("--debug", action="store_true",
                      help="enable the web view inspector")
    p_ui.add_argument("--fullscreen", action="store_true")

    p_serve = sub.add_parser(
        "serve",
        help="run the core with the interface on a local port (no window); "
             "use this on headless machines or over SSH")
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--open", action="store_true",
                         help="open the default browser at the local address")
    p_serve.add_argument("--verbose", action="store_true")
    p_serve.add_argument("--project", default=None)

    p_run = sub.add_parser("run", help="run a Python script against a project")
    p_run.add_argument("script")
    p_run.add_argument("--project", default=None)
    p_run.add_argument("--timeout", type=float, default=3600.0)
    p_run.add_argument("--trusted", action="store_true",
                       help="run in-process with full builtins instead of a subprocess")

    p_info = sub.add_parser("info", help="report the installed capabilities")
    p_info.add_argument("--json", action="store_true")

    p_mat = sub.add_parser("materials", help="list the material library")
    p_mat.add_argument("--search", default="")
    p_mat.add_argument("--json", action="store_true")

    p_bench = sub.add_parser("bench", help="run the benchmark scenes")
    p_bench.add_argument("--sizes", default="1000,10000,100000")
    p_bench.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "ui"
        args.port, args.project, args.debug, args.fullscreen = None, None, False, False

    if args.command == "ui":
        from .desktop_ui.desktop import launch
        return launch(project=args.project, port=args.port, debug=args.debug,
                      fullscreen=args.fullscreen)

    if args.command == "serve":
        from .desktop_ui.server import serve
        serve(port=args.port, open_browser=args.open,
              verbose=args.verbose, project=args.project)
        return 0

    if args.command == "run":
        from .python_api.execution import ScriptRunner, run_in_subprocess
        code = open(args.script).read()
        if args.trusted:
            from .project_format.project import Project
            from .python_api.api import Lab
            project = Project.load(args.project) if args.project else None
            lab = Lab(project)
            result = ScriptRunner(lab, mode="trusted").run(code)
            if args.project:
                lab.project.save(args.project)
        else:
            result = run_in_subprocess(code, project_path=args.project,
                                       timeout_s=args.timeout)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if not result.ok:
            sys.stderr.write((result.traceback or result.error) + "\n")
            return 1
        return 0

    if args.command == "info":
        from .desktop_ui.service import Service
        service = Service()
        info = {
            "version": __version__,
            "materials": len(service.library.ids()),
            "solvers": [d["name"] for d in service.solvers()["registered"]],
            "external": {d["name"]: d["available"] for d in service.solvers()["external"]},
            "plugins": [p["id"] for p in service.plugins.report() if p["ok"]],
            "python": sys.version.split()[0],
            "telemetry": "none",
        }
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"Materia {info['version']}  (Python {info['python']})")
            print(f"  materials : {info['materials']}")
            print(f"  solvers   : {len(info['solvers'])}")
            for name, ok in info["external"].items():
                print(f"    {name:<30s} {'available' if ok else 'not installed'}")
            print(f"  plugins   : {', '.join(info['plugins']) or 'none'}")
            print("  telemetry : none")
        return 0

    if args.command == "materials":
        from .materials import default_library
        lib = default_library()
        items = lib.search(args.search) if args.search else lib.all()
        if args.json:
            print(json.dumps([d.summary() for d in items], indent=2))
        else:
            for d in items:
                gap = d.properties.get("band_gap")
                print(f"{d.id:<28s} {d.formula:<8s} {d.prototype:<20s} "
                      f"gap={'' if gap is None else gap.value} eV")
        return 0

    if args.command == "bench":
        from .benchmarks import eam_benchmarks, electrostatics_benchmarks, run_benchmarks
        sizes = [int(v) for v in args.sizes.split(",")]
        results = run_benchmarks(sizes) + eam_benchmarks(sizes) + electrostatics_benchmarks(sizes)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for row in results:
                print(f"{row['scene']:<34s} {row['n_atoms']:>9d} atoms  "
                      f"{row['seconds']:>8.3f} s  {row['note']}")
        return 0

    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
