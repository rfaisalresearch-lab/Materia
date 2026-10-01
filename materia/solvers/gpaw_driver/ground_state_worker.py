"""The ground-state and relaxation experiments, executed inside the GPAW interpreter.

This file runs under a different Python from the one running Materia, so it
imports nothing from Materia.  It reads one JSON job, builds the atoms and the
calculator from exactly the parameters it was given, runs one self-consistent
field calculation, extracts the requested observables, and writes a JSON
result plus an ``.npz`` file of arrays.

A job with a ``dos`` block converges the ground state as usual, then runs a
non-self-consistent ``fixed_density`` step on the DOS k-point grid and
evaluates ``gpaw.dos.DOSCalculator`` on the requested energy grid, total,
per spin and projected, and reports the parameters, grid, width, method and
projections it used.

A job with a ``relaxation`` block instead moves the nuclei, and for a
variable-cell relaxation the cell, with an ASE optimiser until its force and
stress criteria are met or its step limit is reached, recording every ionic
step, and then extracts the observables at the final geometry.  The optimiser,
its step length, the fixed atoms and the cell filter it actually used are
reported back so the driver can check them against what it sent.

Progress is reported on standard output as ``@@MATERIA`` lines.  Each SCF
iteration reports the total energy and the three residuals GPAW tests for
convergence, read from the same objects GPAW's own convergence check uses, so
the history Materia stores is the history GPAW acted on.

Exit status: 0 converged, 20 the SCF ran out of iterations, 30 failed.  A
cancelled run is terminated by the driver and reaches none of them.
"""

from __future__ import annotations

import hashlib
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

RESTART_SETTABLE = ("xc", "occupations", "convergence", "external")


def emit(kind, **payload):
    payload["event"] = kind
    sys.stdout.write(MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def plain(value):
    """JSON-safe copy of GPAW's parameter objects."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return plain(float(value))
    if hasattr(value, "todict"):
        try:
            return plain(value.todict())
        except BaseException:
            pass
    return str(value)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_atoms(spec):
    import numpy as np
    from ase import Atoms

    atoms = Atoms(
        numbers=np.asarray(spec["numbers"], dtype=int),
        positions=np.asarray(spec["positions"], dtype=float),
        cell=np.asarray(spec["cell"], dtype=float),
        pbc=[bool(p) for p in spec["pbc"]],
    )
    atoms.set_initial_magnetic_moments(np.asarray(spec["magnetic_moments"], dtype=float))
    atoms.set_masses(np.asarray(spec["masses"], dtype=float))
    return atoms


class History:
    """SCF iterations, read from the objects GPAW's convergence check uses."""

    def __init__(self, started):
        self.rows = []
        self.started = started
        self.step = None

    def install(self):
        import gpaw.scf as scf_module

        original = scf_module.check_convergence
        rows = self.rows

        def observed(criteria, context):
            converged, items, entries = original(criteria, context)
            try:
                rows.append(self.row(criteria, context, converged, items))
                emit("scf", **rows[-1])
            except BaseException as exc:
                emit("warning", text=f"SCF history not recorded: {exc}")
            return converged, items, entries

        scf_module.check_convergence = observed

    def row(self, criteria, context, converged, items):
        import numpy as np
        from ase.units import Ha

        nvalence = float(getattr(context.wfs, "nvalence", 0) or 0)
        energy = float(context.ham.e_total_extrapolated * Ha)
        free = float(context.ham.e_total_free * Ha)
        energy_change = None
        criterion = criteria.get("energy")
        old = getattr(criterion, "_old", None)
        if old is not None and len(old) == getattr(old, "maxlen", 0) and len(old) > 1:
            energy_change = float(np.ptp(np.asarray(list(old), dtype=float)))
        density_error = None
        error = getattr(context.dens, "error", None)
        if error is not None and nvalence > 0 and np.isfinite(error):
            density_error = float(error) / nvalence
        eigenstate_error = None
        solver = getattr(context.wfs, "eigensolver", None)
        if solver is not None and nvalence > 0:
            value = getattr(solver, "error", None)
            if value is not None and np.isfinite(value):
                eigenstate_error = float(value) * Ha * Ha / nvalence
        magmom = None
        if getattr(context.wfs, "nspins", 1) == 2:
            try:
                totmom, _ = context.dens.calculate_magnetic_moments()
                magmom = float(np.asarray(totmom).ravel()[-1])
            except BaseException:
                magmom = None
        return {
            "ionic_step": self.step,
            "iteration": int(context.niter),
            "time_s": time.time() - self.started,
            "energy_eV": energy,
            "free_energy_eV": free,
            "energy_change_eV_per_electron": energy_change,
            "density_error_electrons_per_electron": density_error,
            "eigenstate_error_eV2_per_electron": eigenstate_error,
            "magnetic_moment_muB": magmom,
            "converged": bool(converged),
            "criteria_met": {str(k): bool(v) for k, v in dict(items).items()},
        }


def datasets_used(calc):
    out = []
    seen = set()
    for setup in list(calc.setups):
        symbol = str(getattr(setup, "symbol", ""))
        if symbol in seen:
            continue
        seen.add(symbol)
        path = str(getattr(setup, "filename", "") or "")
        entry = {
            "symbol": symbol,
            "path": path,
            "file": os.path.basename(path),
            "gpaw_fingerprint": str(getattr(setup, "fingerprint", "")),
            "type": str(getattr(setup, "type", "")),
            "nuclear_charge": float(getattr(setup, "Z", 0.0) or 0.0),
            "valence_electrons": float(getattr(setup, "Nv", 0.0) or 0.0),
            "core_electrons": float(getattr(setup, "Nc", 0.0) or 0.0),
        }
        if path and os.path.isfile(path):
            entry["sha256"] = sha256_file(path)
        out.append(entry)
    return sorted(out, key=lambda d: d["symbol"])


def summarise_warnings(record):
    counts, order = {}, []
    for item in record:
        text = " ".join(f"{item.category.__name__}: {item.message}".split())
        if text not in counts:
            counts[text] = 0
            order.append(text)
        counts[text] += 1
    return [text if counts[text] == 1 else f"{text} [raised {counts[text]} times]"
            for text in order[:40]]


def extract(calc, atoms, observables, result, arrays):
    """Everything asked for, from the converged calculator."""
    import numpy as np

    wanted = set(observables)
    nspins = int(calc.get_number_of_spins())
    ibz = np.asarray(calc.get_ibz_k_points(), dtype=float)
    weights = np.asarray(calc.get_k_point_weights(), dtype=float)
    nbands = int(calc.get_number_of_bands())
    result["n_spins"] = nspins
    result["n_bands"] = nbands
    result["ibz_kpoints"] = ibz.tolist()
    result["kpoint_weights"] = weights.tolist()
    result["n_bz_kpoints"] = int(len(calc.get_bz_k_points()))
    try:
        result["symmetry_operations"] = int(len(calc.wfs.kd.symmetry.op_scc))
    except BaseException:
        result["symmetry_operations"] = None
    result["coarse_grid"] = [int(n) for n in calc.get_number_of_grid_points()]
    try:
        result["fine_grid"] = [int(n) for n in calc.density.finegd.N_c]
    except BaseException:
        result["fine_grid"] = None

    emit("stage", name="energy")
    result["energy_free_eV"] = float(atoms.get_potential_energy(force_consistent=True))
    result["energy_extrapolated_eV"] = float(atoms.get_potential_energy())
    result["reference_energy_eV"] = float(calc.get_reference_energy())
    from ase.units import Ha

    ham = calc.hamiltonian
    contributions = {}
    for label, name in (("kinetic", "e_kinetic"), ("coulomb", "e_coulomb"),
                        ("external", "e_external"), ("xc", "e_xc"),
                        ("entropy_minus_TS", "e_entropy"), ("local", "e_zero")):
        value = getattr(ham, name, None)
        if value is not None:
            contributions[label] = float(value) * Ha
    result["energy_contributions_eV"] = contributions

    if "forces" in wanted:
        emit("stage", name="forces")
        forces = atoms.get_forces(apply_constraint=False)
        result["forces_eV_per_A"] = [[float(v) for v in row] for row in forces]
    if "stress" in wanted:
        emit("stage", name="stress")
        voigt = np.asarray(atoms.get_stress(voigt=True), dtype=float)
        xx, yy, zz, yz, xz, xy = voigt
        result["stress_eV_per_A3"] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]

    eigenvalues = np.zeros((nspins, len(ibz), nbands))
    occupations = np.zeros((nspins, len(ibz), nbands))
    for s in range(nspins):
        for k in range(len(ibz)):
            eigenvalues[s, k] = calc.get_eigenvalues(kpt=k, spin=s)
            occupations[s, k] = calc.get_occupation_numbers(kpt=k, spin=s, raw=True)
    degeneracy = 2.0 / nspins
    counted = float(np.einsum("k,skn->", weights, occupations) * degeneracy)
    if "eigenvalues" in wanted:
        arrays["eigenvalues"] = eigenvalues
    if "occupations" in wanted:
        arrays["occupations"] = occupations

    try:
        result["fermi_level_eV"] = float(calc.get_fermi_level())
    except BaseException as exc:
        result["fermi_level_eV"] = None
        result["fermi_level_note"] = str(exc)
    try:
        homo, lumo = calc.get_homo_lumo()
        result["homo_eV"] = float(homo)
        result["lumo_eV"] = float(lumo)
    except BaseException:
        pass

    if nspins == 2:
        result["magnetic_moment_muB"] = float(calc.get_magnetic_moment())
        try:
            result["local_magnetic_moments_muB"] = [
                float(m) for m in calc.get_magnetic_moments()]
        except BaseException:
            result["local_magnetic_moments_muB"] = None

    volume = abs(float(np.linalg.det(np.asarray(atoms.cell))))
    accounting = {
        "gpaw_valence_electrons": float(calc.get_number_of_electrons()),
        "occupied_electrons": counted,
        "n_spins": nspins,
    }
    emit("stage", name="density")
    if nspins == 2:
        up = calc.get_all_electron_density(spin=0, gridrefinement=2)
        down = calc.get_all_electron_density(spin=1, gridrefinement=2)
        total = up + down
        dv = volume / float(np.prod(total.shape))
        accounting["spin_up_integral_e"] = float(up.sum() * dv)
        accounting["spin_down_integral_e"] = float(down.sum() * dv)
        accounting["magnetisation_integral_muB"] = float((up - down).sum() * dv)
        if "spin_density" in wanted:
            arrays["spin_density"] = up - down
    else:
        total = calc.get_all_electron_density(gridrefinement=2)
        dv = volume / float(np.prod(total.shape))
    accounting["density_integral_e"] = float(total.sum() * dv)
    accounting["density_grid"] = [int(n) for n in total.shape]
    if "density" in wanted:
        arrays["density"] = total
    if "electrostatic_potential" in wanted:
        emit("stage", name="electrostatic_potential")
        arrays["electrostatic_potential"] = np.asarray(
            calc.get_electrostatic_potential(), dtype=float)
    result["accounting"] = accounting


VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")


def stress_residual(stress, pressure, mask, hydrostatic):
    """Largest deviation of the stress from the target -p I, over the free components."""
    import numpy as np

    stress = np.asarray(stress, dtype=float).reshape(3, 3)
    if hydrostatic:
        return float(abs(np.trace(stress) / 3.0 + pressure))
    target = -pressure * np.eye(3)
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    values = [abs(stress[i, j] - target[i, j]) for (i, j), free in zip(pairs, mask) if free]
    return float(max(values)) if values else 0.0


def relax(atoms, settings, history, result, arrays, report=emit):
    """Move the nuclei, and the cell if asked, until the criteria are met.

    ``atoms`` carries its calculator.  Returns True when the force criterion,
    and for a variable-cell relaxation the stress criterion, are met.
    """
    import numpy as np
    from ase.constraints import FixAtoms
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS, FIRE

    fixed = sorted(int(i) for i in settings.get("fixed_indices") or [])
    if fixed:
        atoms.set_constraint(FixAtoms(indices=fixed))
    mobile = np.ones(len(atoms), dtype=bool)
    mobile[fixed] = False
    cell_filter = settings.get("filter")
    target = atoms
    if cell_filter:
        target = FrechetCellFilter(
            atoms, mask=[bool(v) for v in cell_filter["mask"]],
            hydrostatic_strain=bool(cell_filter["hydrostatic_strain"]),
            scalar_pressure=float(cell_filter["scalar_pressure"]))
    optimizers = {"BFGS": BFGS, "FIRE": FIRE}
    optimizer = optimizers[settings["optimizer"]](target, logfile=None,
                                                  maxstep=float(settings["maxstep"]))
    used_fixed = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            used_fixed.extend(int(i) for i in constraint.index)
    used = {"mode": settings["mode"], "optimizer": type(optimizer).__name__,
            "maxstep": float(optimizer.maxstep), "fmax": float(settings["fmax"]),
            "stress_tol": settings.get("stress_tol"), "max_steps": int(settings["max_steps"]),
            "fixed_indices": sorted(used_fixed), "filter": None}
    if cell_filter:
        stored = np.asarray(target.mask, dtype=float)
        if stored.size == 9:
            stored = stored.reshape(3, 3)
            stored = [stored[0, 0], stored[1, 1], stored[2, 2], stored[1, 2], stored[0, 2],
                      stored[0, 1]]
        used["filter"] = {"name": type(target).__name__,
                          "mask": [bool(v) for v in np.asarray(stored).ravel()],
                          "hydrostatic_strain": bool(target.hydrostatic_strain),
                          "scalar_pressure": float(target.scalar_pressure)}
    result["relaxation_used"] = used
    pressure = float(cell_filter["scalar_pressure"]) if cell_filter else 0.0
    mask = [bool(v) for v in cell_filter["mask"]] if cell_filter else []
    hydrostatic = bool(cell_filter["hydrostatic_strain"]) if cell_filter else False
    steps, positions, cells = [], [], []
    converged = False
    stop = "max_steps"
    for step in range(int(settings["max_steps"]) + 1):
        history.step = step
        before = len(history.rows)
        free = float(atoms.get_potential_energy(force_consistent=True))
        extrapolated = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        norms = np.linalg.norm(forces, axis=1)
        fmax = float(norms[mobile].max()) if mobile.any() else 0.0
        residual = None
        stress = None
        if cell_filter:
            stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
            residual = stress_residual(stress, pressure, mask, hydrostatic)
        row = {"step": step, "energy_free_eV": free, "energy_eV": extrapolated,
               "max_force_eV_A": fmax, "stress_residual_eV_A3": residual,
               "volume_A3": float(abs(np.linalg.det(np.asarray(atoms.cell)))),
               "scf_iterations": len(history.rows) - before,
               "stress_eV_A3": None if stress is None else stress.tolist()}
        steps.append(row)
        positions.append(np.asarray(atoms.get_positions(), dtype=float))
        cells.append(np.asarray(atoms.cell, dtype=float))
        report("relax", **{k: v for k, v in row.items() if k != "stress_eV_A3"})
        if fmax <= float(settings["fmax"]) and (
                residual is None or residual <= float(settings["stress_tol"])):
            converged = True
            stop = "criteria met"
            break
        if step == int(settings["max_steps"]):
            break
        optimizer.step()
    result["relaxation"] = {"steps": steps, "converged": converged, "stop_reason": stop,
                            "final_positions": positions[-1].tolist(),
                            "final_cell": cells[-1].tolist(),
                            "final_forces": np.asarray(
                                atoms.get_forces(apply_constraint=False)).tolist()}
    arrays["relax_positions"] = np.asarray(positions)
    arrays["relax_cells"] = np.asarray(cells)
    return converged


ANGULAR = ("s", "p", "d", "f")


def density_of_states(calc, settings, result, arrays, log_path=None, report=emit):
    """The DOS and projected DOS of a converged calculator, on a fixed-density grid."""
    import numpy as np
    from gpaw.dos import DOSCalculator, get_projector_numbers

    report("stage", name="nscf")
    nscf = calc.fixed_density(txt=(log_path + ".nscf") if log_path else None,
                              **settings["nscf"])
    fermi_scf = float(calc.get_fermi_level())
    fermi_nscf = float(nscf.get_fermi_level())
    reference = fermi_scf if settings["reference"] == "fermi-level" else 0.0
    grid = float(settings["energy_min"]) + float(settings["energy_step"]) * np.arange(
        int(settings["npoints"]))
    absolute = grid + reference
    width = float(settings["width"])
    report("stage", name="dos")
    calculator = DOSCalculator.from_calculator(nscf, shift_fermi_level=False)
    arrays["dos_energies"] = grid
    arrays["dos_total"] = np.asarray(calculator.raw_dos(absolute, spin=None, width=width))
    resolved = settings["spin"] == "resolved"
    if resolved:
        arrays["dos_spin"] = np.array([calculator.raw_dos(absolute, spin=s, width=width)
                                       for s in range(calculator.nspins)])
    counts = []
    used_projections = []
    if settings["projections"]:
        report("stage", name="pdos")
        totals, spins = [], []
        for projection in settings["projections"]:
            ell = ANGULAR.index(projection["angular"])
            indices = [int(a) for a in projection["indices"]]
            counts.append([len(get_projector_numbers(nscf.setups[a], ell)) for a in indices])
            totals.append(sum(np.asarray(calculator.raw_pdos(absolute, a, ell, spin=None,
                                                             width=width))
                              for a in indices))
            if resolved:
                spins.append([sum(np.asarray(calculator.raw_pdos(absolute, a, ell, spin=s,
                                                                 width=width))
                                  for a in indices) for s in range(calculator.nspins)])
            used_projections.append({"label": projection["label"], "indices": indices,
                                     "angular": ANGULAR[ell]})
        arrays["pdos_total"] = np.asarray(totals)
        if resolved:
            arrays["pdos_spin"] = np.asarray(spins)
    arrays["dos_eigenvalues"] = np.asarray(calculator.eig_skn)
    arrays["dos_kweights"] = np.asarray(calculator.weight_k)
    bound = {}
    for setup in nscf.setups:
        symbol = str(setup.symbol)
        if symbol not in bound:
            bound[symbol] = [ANGULAR[l] for l in sorted({int(l) for n, l in
                                                         zip(setup.n_j, setup.l_j)
                                                         if n >= 0}) if l < len(ANGULAR)]
    result["dos"] = {
        "used": {"reference": settings["reference"], "energy_min": float(grid[0]),
                 "energy_step": float(settings["energy_step"]), "npoints": int(len(grid)),
                 "width": width, "method": "tetrahedron" if width == 0.0 else "gaussian",
                 "spin": "resolved" if resolved else "total",
                 "projections": used_projections},
        "nscf_parameters_used": plain(dict(nscf.parameters)),
        "nscf_converged": bool(getattr(nscf.scf, "converged", False)),
        "fermi_level_scf_eV": fermi_scf, "fermi_level_nscf_eV": fermi_nscf,
        "reference_eV": reference, "n_spins": int(calculator.nspins),
        "n_ibz_kpoints": int(len(calculator.weight_k)),
        "n_bz_kpoints": int(len(nscf.get_bz_k_points())),
        "size": [int(n) for n in calculator.wfs.size],
        "bz2ibz": [int(k) for k in calculator.wfs.bz2ibz_map],
        "cell": np.asarray(nscf.atoms.cell).tolist(),
        "bound_channels": bound, "projector_counts": counts,
        "nscf_iterations": int(getattr(nscf.scf, "niter", 0) or 0),
    }


def band_structure(calc, settings, result, arrays, log_path=None, report=emit):
    """Eigenvalues of a converged calculator along an explicit list of k-points."""
    import math

    import numpy as np
    from ase.units import Bohr

    report("stage", name="bands")
    nscf_settings = dict(settings["nscf"])
    kpts = np.asarray(nscf_settings.pop("kpts"), dtype=float)
    nscf = calc.fixed_density(txt=(log_path + ".bands") if log_path else None, kpts=kpts,
                              **nscf_settings)
    fermi_scf = float(calc.get_fermi_level())
    fermi_nscf = float(nscf.get_fermi_level())
    nspins = int(nscf.get_number_of_spins())
    ibz = np.asarray(nscf.get_ibz_k_points(), dtype=float)
    bz = np.asarray(nscf.get_bz_k_points(), dtype=float)
    eigen = np.array([[np.asarray(nscf.get_eigenvalues(kpt=k, spin=s), dtype=float)
                       for k in range(len(ibz))] for s in range(nspins)])
    recip = np.asarray(nscf.wfs.gd.icell_cv, dtype=float) * 2.0 * math.pi / Bohr
    cart = ibz @ recip
    steps = np.linalg.norm(np.diff(cart, axis=0), axis=1) if len(cart) > 1 else np.zeros(0)
    breaks = [int(i) for i in settings.get("breaks") or []]
    for index in breaks:
        if 0 < index <= len(steps):
            steps[index - 1] = 0.0
    arrays["band_eigenvalues"] = eigen
    arrays["band_kpoints"] = ibz
    arrays["band_bz_kpoints"] = bz
    arrays["band_distance"] = np.concatenate([[0.0], np.cumsum(steps)])
    try:
        operations = int(len(nscf.wfs.kd.symmetry.op_scc))
    except BaseException:
        operations = None
    result["bands"] = {
        "used": {"reference": settings["reference"], "n_bands": int(settings["n_bands"]),
                 "labels": list(settings.get("labels") or []), "breaks": breaks},
        "nscf_parameters_used": plain(dict(nscf.parameters)),
        "nscf_converged": bool(getattr(nscf.scf, "converged", False)),
        "nscf_iterations": int(getattr(nscf.scf, "niter", 0) or 0),
        "nscf_symmetry_operations": operations,
        "fermi_level_scf_eV": fermi_scf, "fermi_level_nscf_eV": fermi_nscf,
        "n_spins": nspins, "n_kpoints": int(len(ibz)),
        "n_bands_computed": int(nscf.get_number_of_bands()),
        "reciprocal_cell_invA": recip.tolist(),
        "cell_A": np.asarray(nscf.atoms.cell).tolist(),
    }


def local_density_of_states(calc, settings, result, arrays, log_path=None, report=emit):
    """Pseudo-wavefunction LDOS in an energy window about the Fermi level, on the coarse grid."""
    import numpy as np

    report("stage", name="nscf")
    nscf = calc.fixed_density(txt=(log_path + ".ldos") if log_path else None,
                              **settings["nscf"])
    fermi_scf = float(calc.get_fermi_level())
    fermi_nscf = float(nscf.get_fermi_level())
    low, high = float(settings["energy_min"]), float(settings["energy_max"])
    nspins = int(nscf.get_number_of_spins())
    weights = np.asarray(nscf.get_k_point_weights(), dtype=float)
    nbands = int(settings["n_bands"])
    eigen = np.array([[np.asarray(nscf.get_eigenvalues(kpt=k, spin=s), dtype=float)
                       for k in range(len(weights))] for s in range(nspins)])
    report("stage", name="ldos")
    shape = tuple(int(n) for n in nscf.get_pseudo_wave_function(0, 0, 0).shape)
    per_spin = np.zeros((nspins,) + shape)
    degeneracy = 2.0 / nspins
    count = 0.0
    tip = settings.get("tip")
    stack_energies = settings.get("stack_energies")
    sigma = settings.get("broadening")
    p_map = np.zeros(shape) if tip == "p" else None
    stack = (np.zeros((len(stack_energies),) + shape)
             if stack_energies is not None else None)
    if p_map is not None:
        cell = np.asarray(nscf.atoms.cell, dtype=float)
        jacobian = (cell / np.array(shape, dtype=float)[:, None]).T
        to_cartesian = np.linalg.inv(jacobian.T)
        periodic = [bool(v) for v in nscf.atoms.pbc]

        def lateral_gradient_squared(psi):
            parts = []
            for axis in range(3):
                if periodic[axis]:
                    parts.append(0.5 * (np.roll(psi, -1, axis) - np.roll(psi, 1, axis)))
                else:
                    parts.append(np.gradient(psi, axis=axis))
            index_gradient = np.stack(parts)
            cartesian = np.tensordot(to_cartesian, index_gradient, axes=(1, 0))
            return (np.abs(cartesian[0]) ** 2 + np.abs(cartesian[1]) ** 2)

    for s in range(nspins):
        for k in range(len(weights)):
            for n in range(nbands):
                relative = eigen[s, k, n] - fermi_scf
                in_window = low < relative < high
                near_stack = stack is not None and np.any(
                    np.abs(np.asarray(stack_energies) - relative) < 5 * sigma)
                if not (in_window or near_stack):
                    continue
                psi = nscf.get_pseudo_wave_function(n, k, s)
                density = (psi * np.conj(psi)).real
                if in_window:
                    per_spin[s] += degeneracy * weights[k] * density
                    count += degeneracy * weights[k]
                    if p_map is not None:
                        p_map += degeneracy * weights[k] * lateral_gradient_squared(psi)
                if near_stack:
                    gauss = np.exp(-0.5 * ((np.asarray(stack_energies) - relative) / sigma) ** 2)
                    gauss /= sigma * np.sqrt(2 * np.pi)
                    stack += (degeneracy * weights[k] * gauss)[:, None, None, None] * density
    arrays["ldos_total"] = per_spin.sum(axis=0)
    if p_map is not None:
        arrays["ldos_ptip"] = p_map
    if stack is not None:
        arrays["ldos_stack"] = stack
    if settings["spin"] == "resolved":
        arrays["ldos_spin"] = per_spin
    arrays["ldos_eigenvalues"] = eigen
    arrays["ldos_kweights"] = weights
    radii = {}
    for setup in nscf.setups:
        radii[str(setup.symbol)] = float(max(setup.rcut_j)) * 0.529177210903
    try:
        operations = int(len(nscf.wfs.kd.symmetry.op_scc))
    except BaseException:
        operations = None
    result["ldos"] = {
        "used": dict({"energy_min": low, "energy_max": high, "spin": settings["spin"],
                      "n_bands": nbands},
                     **({"tip": tip} if tip is not None else {}),
                     **({"stack_energies": [float(e) for e in stack_energies],
                         "broadening": float(sigma)} if stack_energies is not None else {})),
        "nscf_parameters_used": plain(dict(nscf.parameters)),
        "nscf_converged": bool(getattr(nscf.scf, "converged", False)),
        "nscf_iterations": int(getattr(nscf.scf, "niter", 0) or 0),
        "nscf_symmetry_operations": operations,
        "fermi_level_scf_eV": fermi_scf, "fermi_level_nscf_eV": fermi_nscf,
        "states_in_window": count, "grid": list(shape),
        "cell_A": np.asarray(nscf.atoms.cell).tolist(), "augmentation_radii_A": radii,
    }


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: ground_state_worker.py <job.json> <result.json>\n")
        return EXIT_FAILED
    job_path, result_path = argv[1], argv[2]
    with open(job_path) as handle:
        job = json.load(handle)
    started = time.time()
    result = {"schema": "materia.dft.ground-state.result", "version": 1,
              "status": "failed", "converged": False, "iterations": 0,
              "warnings": [], "scf_history": []}
    arrays = {}
    history = History(started)
    log_path = job.get("log_path")
    record = []
    try:
        import numpy as np
        import ase
        import gpaw

        result["versions"] = {
            "gpaw": getattr(gpaw, "__version__", ""),
            "ase": getattr(ase, "__version__", ""),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        }
        try:
            from gpaw.mpi import world
            result["versions"]["mpi_world_size"] = int(world.size)
        except BaseException:
            result["versions"]["mpi_world_size"] = 1

        from gpaw import GPAW, KohnShamConvergenceError

        atoms = build_atoms(job["structure"])
        parameters = dict(job["parameters"])
        emit("start", n_atoms=len(atoms), formula=str(atoms.symbols))
        history.install()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            restart_in = job.get("restart_in")
            calc = None
            result["restart"] = {"requested": restart_in, "loaded": False, "written": None}
            if restart_in and os.path.isfile(restart_in):
                try:
                    calc = GPAW(restart_in, txt=log_path)
                    calc.set(**{name: parameters.get(name) for name in RESTART_SETTABLE})
                    result["restart"]["loaded"] = True
                except BaseException as exc:
                    result["restart"]["error"] = f"{type(exc).__name__}: {exc}"
                    calc = None
            if calc is None:
                calc = GPAW(txt=log_path, **parameters)
            atoms.calc = calc
            relaxation = job.get("relaxation")
            extract_final = False
            if relaxation:
                try:
                    done = relax(atoms, relaxation, history, result, arrays)
                    result["status"] = "converged" if done else "not-converged"
                    result["converged"] = bool(done)
                    extract_final = True
                except KohnShamConvergenceError as exc:
                    step = history.step
                    result["status"] = "failed"
                    result["converged"] = False
                    result["error"] = (f"The SCF cycle did not converge at ionic step {step}, "
                                       f"so the relaxation cannot continue: {exc}")
                except RuntimeError as exc:
                    if "Broken symmetry" not in str(exc):
                        raise
                    result["status"] = "failed"
                    result["converged"] = False
                    result["error"] = (
                        "GPAW stopped because an ionic step broke a symmetry it had "
                        "detected in the starting geometry. Relax with symmetry off to let "
                        "the structure lower its symmetry.")
            else:
                try:
                    atoms.get_potential_energy(force_consistent=True)
                    converged = bool(getattr(calc.scf, "converged", False))
                    result["converged"] = converged
                    result["status"] = "converged" if converged else "not-converged"
                except KohnShamConvergenceError as exc:
                    result["status"] = "not-converged"
                    result["converged"] = False
                    result["message"] = str(exc)
            result["iterations"] = int(getattr(calc.scf, "niter", 0) or 0)
            result["parameters_used"] = plain(dict(calc.parameters))
            result["datasets"] = datasets_used(calc)
            result["n_valence_electrons"] = float(calc.get_number_of_electrons())
            expected = job.get("expected_datasets") or {}
            for entry in result["datasets"]:
                want = expected.get(entry["symbol"])
                if want and (os.path.realpath(want["path"]) != os.path.realpath(entry["path"])
                             or want["sha256"] != entry.get("sha256")):
                    result["status"] = "failed"
                    result["converged"] = False
                    result["error"] = (
                        f"GPAW loaded {entry['path']} for {entry['symbol']}, not the "
                        f"pinned dataset {want['path']} (SHA-256 {want['sha256'][:12]}).")
                    extract_final = False
                    break
            if result["status"] == "converged" or extract_final:
                extract(calc, atoms, job.get("observables", []), result, arrays)
            if job.get("dos") and result["status"] == "converged":
                try:
                    density_of_states(calc, job["dos"], result, arrays, log_path)
                except KohnShamConvergenceError as exc:
                    result["status"] = "failed"
                    result["converged"] = False
                    result["error"] = ("The non-self-consistent DOS step did not converge "
                                       f"every band: {exc}")
            if job.get("bands") and result["status"] == "converged":
                try:
                    band_structure(calc, job["bands"], result, arrays, log_path)
                except KohnShamConvergenceError as exc:
                    result["status"] = "failed"
                    result["converged"] = False
                    result["error"] = ("The non-self-consistent band-structure step did not "
                                       f"converge every requested band: {exc}")
            if job.get("ldos") and result["status"] == "converged":
                try:
                    local_density_of_states(calc, job["ldos"], result, arrays, log_path)
                except KohnShamConvergenceError as exc:
                    result["status"] = "failed"
                    result["converged"] = False
                    result["error"] = ("The non-self-consistent LDOS step did not converge "
                                       f"every band: {exc}")
            if job.get("dos") and result["status"] == "converged":
                restart_out = job.get("restart_out")
                if restart_out:
                    try:
                        calc.write(restart_out, mode="all")
                        result["restart"]["written"] = restart_out
                        result["restart"]["size_bytes"] = os.path.getsize(restart_out)
                    except BaseException as exc:
                        result["restart"]["write_error"] = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        result["status"] = "failed"
        result["converged"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]

    if job.get("relaxation"):
        last = history.step
        result["relaxation_scf_iterations"] = len(history.rows)
        result["scf_history"] = [row for row in history.rows if row.get("ionic_step") == last]
    else:
        result["scf_history"] = history.rows
    result["warnings"] = summarise_warnings(record)[:50] if record else []
    result["wall_time_s"] = time.time() - started
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, errors="replace") as handle:
                result["log_tail"] = handle.read()[-8000:]
        except BaseException:
            pass
    arrays_path = job.get("arrays_path")
    if arrays and arrays_path:
        import numpy as np

        np.savez(arrays_path, **arrays)
        result["arrays"] = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                            for name, value in arrays.items()}
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
