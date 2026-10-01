"""The script Materia runs inside the GPAW interpreter.

This file is executed by a different Python from the one running Materia, so it
imports nothing from Materia.  It reads one JSON job file, runs a single
self-consistent GPAW calculation, and writes one JSON result file.  Progress is
reported on standard output as ``@@MATERIA`` lines, one per SCF iteration, so
the driver can show progress and so a cancelled run leaves a record of how far
it got.

The exit status distinguishes the four outcomes the driver has to tell apart:
0 converged, 20 the SCF ran but did not converge, 30 the calculation failed.
A cancelled run does not reach any of them, because the driver terminates the
process; that case is recognised by the driver, not here.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings

EXIT_CONVERGED = 0
EXIT_NOT_CONVERGED = 20
EXIT_FAILED = 30

MARKER = "@@MATERIA "


def emit(kind, **payload):
    payload["event"] = kind
    sys.stdout.write(MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def build_atoms(spec):
    import numpy as np
    from ase import Atoms
    from ase.constraints import FixAtoms

    atoms = Atoms(
        numbers=np.asarray(spec["numbers"], dtype=int),
        positions=np.asarray(spec["positions"], dtype=float),
        cell=np.asarray(spec["cell"], dtype=float),
        pbc=[bool(p) for p in spec["pbc"]],
    )
    magmoms = spec.get("magnetic_moments")
    if magmoms is not None:
        atoms.set_initial_magnetic_moments(np.asarray(magmoms, dtype=float))
    fixed = spec.get("fixed")
    if fixed is not None:
        indices = [i for i, flag in enumerate(fixed) if flag]
        if indices:
            atoms.set_constraint(FixAtoms(indices=indices))
    return atoms


def build_calculator(settings, log_path):
    from gpaw import GPAW, PW, FermiDirac, MarzariVanderbilt

    mode = settings["mode"]
    if mode == "pw":
        discretisation = {"mode": PW(float(settings["cutoff_eV"]))}
    elif mode == "fd":
        discretisation = {"mode": "fd", "h": float(settings["grid_spacing_A"])}
    else:
        discretisation = {"mode": "lcao", "basis": str(settings["basis"])}

    scheme = settings["occupations"]
    width = float(settings["smearing_eV"])
    if scheme == "fermi-dirac":
        occupations = FermiDirac(width)
    elif scheme == "marzari-vanderbilt":
        occupations = MarzariVanderbilt(width)
    else:
        occupations = FermiDirac(0.0)

    kwargs = dict(
        xc=str(settings["xc"]),
        kpts=tuple(int(k) for k in settings["kpoints"]),
        occupations=occupations,
        charge=float(settings["charge"]),
        spinpol=bool(settings["spin_polarized"]),
        maxiter=int(settings["max_iterations"]),
        convergence={
            "energy": float(settings["energy_tol_eV_per_electron"]),
            "density": float(settings["density_tol_electrons"]),
        },
        txt=log_path,
        **discretisation,
    )
    return GPAW(**kwargs)


def dataset_identity(calc):
    out = []
    seen = set()
    try:
        setups = list(calc.setups)
    except BaseException:
        return out
    for setup in setups:
        symbol = str(getattr(setup, "symbol", ""))
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append({
            "symbol": symbol,
            "file": os.path.basename(str(getattr(setup, "filename", ""))),
            "path": str(getattr(setup, "filename", "")),
            "fingerprint": str(getattr(setup, "fingerprint", "")),
            "type": str(getattr(setup, "type", "")),
            "valence_electrons": float(getattr(setup, "Nv", 0.0) or 0.0),
            "core_electrons": float(getattr(setup, "Nc", 0.0) or 0.0),
        })
    return sorted(out, key=lambda d: d["symbol"])


def summarise_warnings(record):
    """One line per distinct warning, with how many times it was raised.

    A GPAW run can emit the same library deprecation hundreds of times; keeping
    every copy buries the one warning that matters.
    """
    counts = {}
    order = []
    for item in record:
        text = f"{item.category.__name__}: {item.message}"
        text = " ".join(text.split())
        if text not in counts:
            counts[text] = 0
            order.append(text)
        counts[text] += 1
    out = []
    for text in order[:40]:
        times = counts[text]
        out.append(text if times == 1 else f"{text} [raised {times} times]")
    return out


def scalar(getter, default=None):
    try:
        value = getter()
    except BaseException:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: worker.py <job.json> <result.json>\n")
        return EXIT_FAILED
    job_path, result_path = argv[1], argv[2]
    with open(job_path) as handle:
        job = json.load(handle)

    settings = job["settings"]
    log_path = job.get("log_path") or None
    started = time.time()
    result = {
        "schema": 1,
        "started_unix": started,
        "converged": False,
        "status": "failed",
        "warnings": [],
        "iterations": 0,
    }

    caught = []
    warnings.simplefilter("always")

    try:
        import ase
        import gpaw
        result["gpaw_version"] = getattr(gpaw, "__version__", "")
        result["ase_version"] = getattr(ase, "__version__", "")
        result["worker_python"] = sys.version.split()[0]
        try:
            from gpaw.mpi import world
            result["mpi_world_size"] = int(world.size)
        except BaseException:
            result["mpi_world_size"] = 1

        from gpaw import KohnShamConvergenceError

        atoms = build_atoms(job["structure"])
        emit("start", n_atoms=len(atoms), formula=str(atoms.symbols))

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            calc = build_calculator(settings, log_path)
            atoms.calc = calc

            def observe():
                emit("scf", iteration=int(getattr(calc.scf, "niter", 0)))

            calc.attach(observe, 1)

            try:
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces(apply_constraint=False)
                result["converged"] = bool(getattr(calc.scf, "converged", False))
                result["status"] = "converged" if result["converged"] else "not-converged"
                result["energy_eV"] = energy
                result["forces_eV_per_A"] = [[float(v) for v in row] for row in forces]
            except KohnShamConvergenceError as exc:
                result["status"] = "not-converged"
                result["converged"] = False
                result["message"] = str(exc)

            result["iterations"] = int(getattr(calc.scf, "niter", 0) or 0)
            result["fermi_level_eV"] = scalar(calc.get_fermi_level)
            result["n_electrons"] = scalar(calc.get_number_of_electrons)
            result["magnetic_moment"] = scalar(calc.get_magnetic_moment, None)
            result["datasets"] = dataset_identity(calc)
            try:
                result["n_bands"] = int(calc.get_number_of_bands())
            except BaseException:
                pass
            try:
                result["n_kpoints"] = int(len(calc.get_ibz_k_points()))
            except BaseException:
                pass
            caught = summarise_warnings(record)
    except BaseException as exc:
        result["status"] = "failed"
        result["converged"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]

    result["warnings"] = caught[:50]
    result["wall_time_s"] = time.time() - started
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, errors="replace") as handle:
                text = handle.read()
            result["log_tail"] = text[-8000:]
        except BaseException:
            pass

    with open(result_path, "w") as handle:
        json.dump(result, handle)
    emit("done", status=result["status"])

    if result["status"] == "converged":
        return EXIT_CONVERGED
    if result["status"] == "not-converged":
        return EXIT_NOT_CONVERGED
    return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
