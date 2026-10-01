"""Psi4 worker: reads one JSON job on stdin, writes one JSON reply on stdout.

Run with the Python interpreter of an environment that has Psi4.  It imports
nothing from Materia, so that environment needs only Psi4.
"""

import json
import os
import sys
import tempfile


def main() -> None:
    job = json.loads(sys.stdin.read())
    real_stdout = os.dup(1)
    os.dup2(2, 1)
    try:
        reply = run(job)
    except Exception as exc:
        reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    os.dup2(real_stdout, 1)
    sys.stdout.write(json.dumps(reply))
    sys.stdout.flush()


def run(job: dict) -> dict:
    import numpy as np
    import psi4

    scratch = tempfile.mkdtemp(prefix="materia-psi4-")
    psi4.core.IOManager.shared_object().set_default_path(scratch)
    psi4.core.set_output_file(os.path.join(scratch, "output.dat"), False)
    psi4.set_num_threads(1)
    psi4.set_memory(int(job.get("memory_mb", 1000)) * 1_000_000)
    lines = [f"{job['charge']} {job['multiplicity']}"]
    lines += [f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in
              zip(job["symbols"], job["positions_A"])]
    lines += ["units angstrom", "no_reorient", "no_com", "symmetry c1"]
    psi4.geometry("\n".join(lines))
    options = {"basis": job["basis"], "scf_type": job.get("scf_type", "pk"),
               "e_convergence": 1e-10, "d_convergence": 1e-9,
               "reference": job["reference"], "freeze_core": False,
               "puream": job.get("puream", True),
               "mp2_type": "df" if job.get("scf_type") == "df" else "conv"}
    if "dft_spherical_points" in job:
        options["dft_spherical_points"] = job["dft_spherical_points"]
        options["dft_radial_points"] = job["dft_radial_points"]
    psi4.set_options(options)
    method = job["method"]
    out = {"ok": True, "psi4_version": psi4.__version__}
    if job.get("tdscf_states"):
        from psi4.driver.procrouting.response.scf_response import tdscf_excitations

        psi4.set_options({"save_jk": True})
        energy, wfn = psi4.energy(method, return_wfn=True)
        states = tdscf_excitations(wfn, states=int(job["tdscf_states"]),
                                   tda=bool(job.get("tda")), triplets="none")
        out["energy_Eh"] = float(energy)
        out["excitation_energies_Eh"] = [float(s["EXCITATION ENERGY"]) for s in states]
        out["oscillator_strengths"] = [float(s["OSCILLATOR STRENGTH (LEN)"]) for s in states]
    elif job.get("gradient"):
        gradient, wfn = psi4.gradient(method, return_wfn=True)
        out["gradient_Eh_bohr"] = np.asarray(gradient).tolist()
        out["energy_Eh"] = float(wfn.energy())
    else:
        out["energy_Eh"] = float(psi4.energy(method))
    variables = psi4.core.variables()
    out["variables"] = {k: float(v) for k, v in variables.items()
                        if isinstance(v, float) and "ENERGY" in k}
    psi4.core.clean()
    return out


if __name__ == "__main__":
    main()
