"""Machine-learned interatomic potentials driven in a separate interpreter.

MACE-MP-0 (I. Batatia et al., arXiv:2401.00096; MIT licence) is a foundation
model trained on the Materials Project trajectory data set of PBE and PBE+U
calculations.  It runs in its own environment with PyTorch, never inside the
Materia process: Materia starts one persistent worker
(:mod:`.worker`) and exchanges JSON lines with it, with a timeout on every
call.  The interpreter is found through ``MATERIA_ML_PYTHON`` or the conda
environment ``materia-ml``.

The model is a surrogate of its training data: it inherits PBE's errors and is
least reliable for chemistry far from the Materials Project's crystals.  Every
result says so, and records the model file's SHA-256 and the MACE and PyTorch
versions.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ...core_model.structure import Structure
from ...physics.potentials import Potential, PotentialError, UnsupportedSystem
from ...provenance import Fidelity
from ...system import conda_environments, environment_python, is_executable

ENV_VAR = "MATERIA_ML_PYTHON"
WORKER = Path(__file__).with_name("worker.py")


def find_python() -> Optional[str]:
    explicit = os.environ.get(ENV_VAR, "").strip()
    if explicit:
        return explicit if is_executable(explicit) else None
    for env in conda_environments("materia-ml"):
        if env.name == "materia-ml":
            candidate = environment_python(env)
            if candidate.is_file():
                return str(candidate)
    return None


class MLPotential(Potential):
    """A MACE-MP-0 model evaluated by a persistent worker process."""

    fidelity = Fidelity.TIER2_SEMI_EMPIRICAL

    def __init__(self, size: str = "small", dtype: str = "float64",
                 timeout_s: float = 600.0, threads: int = 1) -> None:
        if size not in ("small", "medium", "large"):
            raise ValueError("size must be small, medium or large.")
        self.size = size
        self.dtype = dtype
        self.timeout_s = float(timeout_s)
        self.threads = int(threads)
        self.name = f"ml/mace-mp-0-{size}"
        self.cutoff_A = 6.0
        self._process: Optional[subprocess.Popen] = None
        self.info: dict = {}

    def model_label(self) -> str:
        return self.name

    def composition(self, structure: Structure) -> Tuple[bool, str]:
        if find_python() is None:
            return False, ("No interpreter with MACE was found. Create one with 'conda create "
                           "-n materia-ml python=3.11 pip' and 'pip install mace-torch' in it, "
                           "or set MATERIA_ML_PYTHON.")
        if any(structure.cell.pbc) and not all(structure.cell.pbc):
            return False, ("MACE here takes molecules or fully periodic cells; pad slabs with "
                           "vacuum in a fully periodic cell.")
        return True, ""

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        python = find_python()
        if python is None:
            raise UnsupportedSystem("No interpreter with MACE was found; set MATERIA_ML_PYTHON "
                                    "or create the materia-ml conda environment.")
        env = {**os.environ, "OMP_NUM_THREADS": str(self.threads),
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        self._process = subprocess.Popen([python, "-u", str(WORKER)], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, env=env)
        out = self._call({"op": "init", "size": self.size, "dtype": self.dtype,
                          "threads": self.threads})
        self.info = out["info"]

    def _call(self, request: dict) -> dict:
        process = self._process
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        ready = selector.select(timeout=self.timeout_s)
        selector.close()
        if not ready:
            self.close()
            raise PotentialError(f"The ML worker did not answer within {self.timeout_s:g} s.")
        line = process.stdout.readline()
        if not line:
            self.close()
            raise PotentialError("The ML worker exited unexpectedly.")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise PotentialError(f"The ML worker failed: {reply.get('error')}")
        return reply

    def energy_and_forces(self, structure: Structure) -> Tuple[float, np.ndarray]:
        ok, why = self.composition(structure)
        if not ok:
            raise UnsupportedSystem(why)
        self._start()
        reply = self._call({"op": "compute", "numbers": structure.numbers.tolist(),
                            "positions": np.asarray(structure.positions).tolist(),
                            "cell": np.asarray(structure.cell.matrix).tolist(),
                            "pbc": [bool(p) for p in structure.cell.pbc]})
        forces = np.asarray(reply["forces"], dtype=float)
        if not np.isfinite(reply["energy"]) or not np.all(np.isfinite(forces)):
            raise PotentialError("The ML model returned a non-finite energy or force.")
        self.last_stress = reply.get("stress")
        return float(reply["energy"]), forces

    def close(self) -> None:
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.stdin.write(json.dumps({"op": "close"}) + "\n")
                    self._process.stdin.flush()
                    self._process.wait(timeout=10)
            except Exception:
                self._process.kill()
            self._process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def describe(self) -> dict:
        return {"model": self.name, "fidelity": self.fidelity.value, "cutoff_A": self.cutoff_A,
                "parameters": {"engine": "MACE", **self.info},
                "approximations": [
                    "Machine-learned surrogate (MACE-MP-0) of PBE and PBE+U energies from the "
                    "Materials Project trajectory data set: it inherits PBE's errors and is "
                    "least reliable far from the crystals it was trained on.",
                    "No electrons, charges or magnetic moments are represented explicitly.",
                ],
                "references": ["I. Batatia et al., A foundation model for atomistic materials "
                               "chemistry, arXiv:2401.00096 (2023)",
                               "I. Batatia et al., Adv. Neural Inf. Process. Syst. 35 (2022) "
                               "11423 (MACE)"]}
