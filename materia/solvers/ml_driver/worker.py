"""Machine-learned potential worker, run under the interpreter that has the ML packages.

Reads one JSON request per line on stdin and writes one JSON reply per line on
stdout.  Requests: ``init`` (model family and size), ``compute`` (numbers,
positions in A, cell in A, periodic flags), ``info``, ``close``.  Nothing else
is printed on stdout.
"""

import hashlib
import json
import os
import sys
import traceback


CHANNEL = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr


def reply(payload):
    CHANNEL.write(json.dumps(payload) + "\n")
    CHANNEL.flush()


def main():
    calculator = None
    model_info = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "init":
                import numpy as np
                import torch
                torch.set_num_threads(int(request.get("threads", 1)))
                from mace.calculators import mace_mp
                import mace
                calculator = mace_mp(model=request.get("size", "small"),
                                     default_dtype=request.get("dtype", "float64"),
                                     device="cpu")
                known = {"small": "20231210mace128L0_energy_epoch249model",
                         "medium": "20231203mace128L1_epoch199model",
                         "large": "20231203mace128L2_epoch199model"}
                path = os.path.join(os.path.expanduser("~/.cache/mace"),
                                    known.get(request.get("size", "small"), ""))
                digest = None
                if os.path.isfile(path):
                    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
                else:
                    path = None
                model_info = {"family": "mace-mp-0", "size": request.get("size", "small"),
                              "dtype": request.get("dtype", "float64"), "model_path": path,
                              "model_sha256": digest, "mace_version": mace.__version__,
                              "torch_version": torch.__version__}
                reply({"ok": True, "info": model_info})
            elif op == "compute":
                from ase import Atoms
                atoms = Atoms(numbers=request["numbers"], positions=request["positions"],
                              cell=request["cell"], pbc=request["pbc"])
                atoms.calc = calculator
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces().tolist()
                stress = None
                if all(request["pbc"]):
                    stress = atoms.get_stress(voigt=False).tolist()
                reply({"ok": True, "energy": energy, "forces": forces, "stress": stress})
            elif op == "info":
                reply({"ok": True, "info": model_info})
            elif op == "close":
                reply({"ok": True})
                return
            else:
                reply({"ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-2000:]})


if __name__ == "__main__":
    main()
